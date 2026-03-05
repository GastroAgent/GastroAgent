import argparse
import copy
import glob
import math
import os
from timm.models.vision_transformer import VisionTransformer
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,3,4'
import gc
from transformers import ChineseCLIPConfig as CLIPConfig
from transformers import ChineseCLIPProcessor as CLIPProcessor
from transformers import ChineseCLIPModel as CLIPModel
from transformers import AutoTokenizer, AutoModel, AutoConfig, ChineseCLIPTextModel, ChineseCLIPTextConfig
import json
import torch
import random
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import transforms as T
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from diffusers import AutoencoderKL
import sys
from functools import partial
from PIL import Image
from safetensors.torch import load_model, load_file
sys.path.append('./GasAgent-main')

from utils_.train_utils import (
    find_latest_checkpoint,
    cleanup_old_checkpoints,
    ema, infiniteloop,
    warmup_lr
)
# from data_utils import create_dataset, create_dataloaders
from utils_.data_loader import MedicalJsonDataset

from conditional_flow_matcher import ConditionalFlowMatcher, OptimalTransportConditionalFlowMatcher
from my_models.unet_model import UNetModelWrapper
from my_models.unet_2d_condition import UNet2DConditionModel
from my_models.model_dispatch import dispatch_model
from model_utils.model import TripletNetwork

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Flow Matching Training Script")

    # Model configuration
    parser.add_argument("--model", type=str, default="otcfm", choices=["otcfm", "icfm"],
                        help="Flow matching model type")

    parser.add_argument("--output_dir", type=str,
                        default="./outputs/image_hint_十二指肠",
                        help="Output directory")

    # UNet configuration
    parser.add_argument("--num_channel", type=int, default=128,
                        help="Base channel of UNet")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Target learning rate")

    parser.add_argument("--grad_clip", type=float, default=1.5,
                        help="Gradient norm clipping")

    parser.add_argument("--total_steps", type=int, default=50000,
                        help="Total training steps")

    parser.add_argument("--warmup", type=int, default=500,
                        help="Learning rate warmup steps")

    parser.add_argument("--batch_size", type=int, default=3,
                        help="Batch size")

    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for DataLoader")

    parser.add_argument("--ema_decay", type=float, default=0.95,
                        help="EMA decay rate")

    parser.add_argument("--op_match_batch", type=bool, default=True,
                        help="op_match_batch")

    parser.add_argument("--reflow", type=bool, default=False,
                        help="Rectified flow")

    # Evaluation parameters
    parser.add_argument("--save_step", type=int, default=2500,
                        help="Frequency of saving checkpoints (0 to disable)")

    # Image dataset parameters
    parser.add_argument("--image_dir", type=str,
                        default="",
                        help="Directory containing training images")

    # Logging parameters
    parser.add_argument("--log_dir", type=str,
                        default="./logs_attn",
                        help="TensorBoard log directory")

    # last n checkpoints to save, delete the rest checkpoints for saving the disk space
    parser.add_argument("--keep_n_checkpoints", type=int, default=10,
                        help="Number of previous checkpoints to keep")
    return parser.parse_args()

from geomloss import SamplesLoss
loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.1**0.5)
# sinkhorn hausdorff energy gaussian laplacian
def sinkhorn_loss(bx, bx1, epsilon=0.1, n_iter=50, reduction='mean'):
    """
    Compute Sinkhorn loss (approximate Wasserstein distance) between two sets of samples.
    
    Args:
        bx (Tensor): [B, N, D] or [B, C, H, W] or [B, D]
        bx1 (Tensor): [B, N, D] or [B, C, H, W] or [B, D]
        epsilon (float): Entropy regularization strength
        n_iter (int): Number of Sinkhorn iterations
        reduction (str): 'mean' or 'sum' for batch reduction

    Returns:
        Tensor: Scalar loss
    """
    def _reshape_for_sinkhorn(x):
        if x.ndim == 2:
            B, D = x.shape
            H = W = int((D // 4) ** 0.5)
            return x.view(B, 4, H, W).permute(0, 2, 3, 1).reshape(B, H * W, -1)
        if x.ndim == 4:
            B, C, H, W = x.shape
            return x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        if x.ndim == 3:
            x = x.unsqueeze(0)
            B, C, H, W = x.shape
            return x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return x

    bx = _reshape_for_sinkhorn(bx)
    bx1 = _reshape_for_sinkhorn(bx1)
    loss = loss_fn(bx, bx1)
    return loss

def cal_wasserstein_loss(x, x1, **kwargs):
    wass_loss = sinkhorn_loss(x, x1, **kwargs)
    # wass_loss = wass_loss.sum()
    return wass_loss

# def sinkhorn_loss(bx, bx1, epsilon=0.1, n_iter=25, reduction='mean'):
#     """
#     Compute Sinkhorn loss (approximate Wasserstein distance) between two sets of samples.
    
#     Args:
#         bx (Tensor): [B, N, D] or [B, C, H, W] or [B, D]
#         bx1 (Tensor): [B, N, D] or [B, C, H, W] or [B, D]
#         epsilon (float): Entropy regularization strength
#         n_iter (int): Number of Sinkhorn iterations
#         reduction (str): 'mean' or 'sum' for batch reduction

#     Returns:
#         Tensor: Scalar loss
#     """
#     # Reshape input to [B, N, D]
#     if bx.ndim == 2:
#         B, D = bx.shape
#         H = W = int((D // 4) ** 0.5)
#         bx = bx.view(B, 4, H, W).permute(0, 2, 3, 1).reshape(B, H*W, -1)
#         bx1 = bx1.view(B, 4, H, W).permute(0, 2, 3, 1).reshape(B, H*W, -1)
#     elif bx.ndim == 4:
#         B, C, H, W = bx.shape
#         bx = bx.permute(0, 2, 3, 1).reshape(B, H*W, C)
#         bx1 = bx1.permute(0, 2, 3, 1).reshape(B, H*W, C)

#     B, N, D = bx.shape
#     _, M, _ = bx1.shape

#     # Compute cost matrix: [B, N, M]
#     x2 = torch.sum(bx**2, dim=-1, keepdim=True)                    # [B, N, 1]
#     y2 = torch.sum(bx1**2, dim=-1, keepdim=True)                   # [B, M, 1]
#     cross = torch.bmm(bx, bx1.transpose(-1, -2))                   # [B, N, M]
#     cost_matrix = x2 - 2 * cross + y2.transpose(-1, -2)            # [B, N, M]
#     cost_matrix = torch.clamp(cost_matrix, min=0.0)

#     # Kernel matrix
#     K = torch.exp(-cost_matrix / epsilon)                          # [B, N, M]

#     # Uniform marginal distributions
#     a = torch.ones(B, N, device=bx.device) / N                     # [B, N]
#     b = torch.ones(B, M, device=bx.device) / M                     # [B, M]

#     # Initialize dual variables
#     u = torch.ones_like(a)                                         # [B, N]
#     v = torch.ones_like(b)                                         # [B, M]

#     # Sinkhorn iterations
#     for _ in range(n_iter):
#         u = a / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)  # [B, N]
#         v = b / (torch.bmm(K.transpose(-1,-2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)  # [B, M]

#     # Compute transport plan
#     P = u.unsqueeze(-1) * K * v.unsqueeze(-2)  # [B, N, M]

#     # Compute loss
#     loss = torch.sum(P * cost_matrix, dim=(1,2))  # [B]

#     if reduction == 'mean':
#         return loss.mean()
#     elif reduction == 'sum':
#         return loss.sum()
#     else:
#         return loss
    
# def cal_wasserstein_loss(x, x1, **kwargs):
#     wass_loss = sinkhorn_loss(x, x1, **kwargs)
#     # wass_loss = wass_loss.sum()
#     return wass_loss

str_map = {
    '息肉': "Polyps",
    "染色息肉": "Dyed_lifted_polyps",
    "染色边缘": "Dyed_resection_margins",
    "食管炎": "Esophagitis",
    "溃疡结肠炎": "Ulcerative_colitis"
}
def check_str(st1, st2):
    return st1 == st2 or str_map.get(st1, st1) == str_map.get(st2, st2)

def neg_betas(**kwargs):  # Optimization weight for negative samples.
    '''

    return:
        int, Tensor.scaler or Tensor.Shape: [B]
    '''
    return -1

def train(args):
    """Main training function."""
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_posterior = True
    random_sample_posterior = False
    train_text_encoder = False
    train_vision_encoder = False
    use_image_feature = True
    use_text_feature = True
    caption_hidden_states_mode = 'cat' # only_text
    use_image_mask = True
    checkpoints = ''
    train_wass_model = True

    # Create output and log directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_mask = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_A = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    transform_grey = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.RandomGrayscale(p=1),  # Data augmentation: convert to grayscale.
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataloaders = []
    json_paths = glob.glob("./data/data_pairs_small/*.json")
    for json_path in tqdm(json_paths):
        dataset = MedicalJsonDataset(
            path=json_path,
            transform=transform,
            hint_transform=transform_grey,
            transform_A=transform_A,
            transform_B=transform,
            transform_mask = transform_mask
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        dataloaders.append(dataloader)
        # break

    print('DataLoad Total Steps: ', sum(len(dataloader) for dataloader in dataloaders))
    dataloopers = [infiniteloop(dataloader) for dataloader in dataloaders]

    ### Model initialization
    config = json.load(open('./flow_matcher_otcfm/unet/config.json', 'r'))
    net_model = UNet2DConditionModel(**config)
    try:
        net_model.load_state_dict(torch.load('./flow_matcher_otcfm/unet/diffusion_pytorch_model.bin'), strict=False)
    except RuntimeError as e:
        print(e)
        pass
    vae = AutoencoderKL.from_pretrained('./flow_matcher_otcfm/vae').eval()
    
    if use_image_feature:
        def process_single_image(image_path, input_size=224, dataset_mean=[0.3464, 0.2280, 0.2228],
                                 dataset_std=[0.2520, 0.2128, 0.2093]):
            transform = T.Compose([
                T.Resize((input_size, input_size)),
                T.ToTensor(),
                T.Normalize(mean=dataset_mean, std=dataset_std)
            ])

            # Open the image
            image = Image.open(image_path).convert('RGB')

            # Apply the transformations
            processed_image = transform(image)
            return processed_image
        
        vision_model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                         qkv_bias=False, norm_layer=partial(nn.LayerNorm, eps=1e-6)).eval()
        vision_model.load_state_dict(torch.load("./flow_matcher_otcfm/EndoViT/pytorch_model.bin", weights_only=False))
    else:
        vision_model = None

        def process_single_image(x):
            return x

    if use_text_feature:
        config_json = json.load(open('./flow_matcher_otcfm/text_encoder/config.json', 'r'))
        config = ChineseCLIPTextConfig(**config_json)
        text_model = ChineseCLIPTextModel(config, False).eval()
        text_tokenizer = AutoTokenizer.from_pretrained(
            './flow_matcher_otcfm/text_encoder', use_fast=True)
        try:
            state_dict = load_file("./flow_matcher_otcfm/text_encoder/model.safetensors")
            text_model.load_state_dict(state_dict)
        except:
            pass
    else:
        text_model = None
        text_tokenizer = None

    net_model = net_model.to(device=device).train()

    def _encode_latent(images):
        with torch.no_grad():
            images = images.to(vae.device)
            posterior = vae.encode(images).latent_dist
            if random_sample_posterior:
                latent = posterior.sample() if random.random() > 0.5 else posterior.mode()
            elif sample_posterior:
                latent = posterior.sample()
            else:
                latent = posterior.mode()
        return latent * 0.18215

    def init_weights(model):
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Initialize with Kaiming normal (suitable for ReLU).
                init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                # Initialize with Xavier normal.
                init.xavier_normal_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                # Initialize LayerNorm weight and bias.
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)

    init_weights(net_model)
    state_dict = {}
    # state_dict = torch.load(
    #     './outputs/disease_A2B/otcfm_weights_step_2000_A2D.pt')
    if 'ema_model' in state_dict:
        net_model.load_state_dict(state_dict['ema_model'], strict=False)

    # EMA Model
    ema_model = copy.deepcopy(net_model)
    # Optimizer and Scheduler
    optim = torch.optim.AdamW(net_model.parameters(), lr=args.lr, weight_decay=0.001)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda step: warmup_lr(step, args.warmup))
    if train_text_encoder and text_model is not None:
        text_model.train()
        text_optim = torch.optim.AdamW(text_model.parameters(), lr=args.lr * 0.5, weight_decay=0.001)
        text_sched = torch.optim.lr_scheduler.LambdaLR(text_optim, lr_lambda=lambda step: warmup_lr(step, args.warmup))
    if train_vision_encoder and vision_model is not None:
        vision_model.train()
        vision_optim = torch.optim.AdamW(vision_model.parameters(), lr=args.lr * 0.5, weight_decay=0.001)
        vision_sched = torch.optim.lr_scheduler.LambdaLR(vision_optim,
                                                         lr_lambda=lambda step: warmup_lr(step, args.warmup))
        
    ############# Attention #############
    wass_model = TripletNetwork(model='attention').to("cuda:2").eval()
    print(wass_model)
    wass_model.device = "cuda:2"
    model_size = sum(p.data.nelement() for p in wass_model.parameters())
    print(f"Wass Model params: {model_size / 1024 / 1024:.2f} M")
    state_dict = torch.load("./best_flow_weights/attention_tiny.pt")
    
    # ############# CNN #############
    # wass_model = TripletNetwork(model='resnet34').to("cuda:2").eval()
    # wass_model.device = "cuda:2"
    # model_size = sum(p.data.nelement() for p in wass_model.parameters())
    # print(f"Wass Model params: {model_size / 1024 / 1024:.2f} M")
    # state_dict = torch.load("./best_weights/simple_model_energy.pt")
    
    wass_model.load_state_dict(state_dict, strict=True)
    wass_model = wass_model.to(wass_model.device)
    if train_wass_model:
        wass_optim = torch.optim.AdamW(wass_model.parameters(), lr=args.lr, weight_decay=0.001)
    # Flow Matcher initialization
    sigma = 0.1
    if args.model == "otcfm":
        FM = OptimalTransportConditionalFlowMatcher(sigma=sigma, ot_method='exact')
    elif args.model == "icfm":
        args.op_match_batch = False
        FM = OptimalTransportConditionalFlowMatcher(sigma=sigma, ot_method='exact')
        # FM = ConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {args.model}, must be one of ['otcfm', 'icfm']"
        )

    # Directories
    savedir = os.path.join(args.output_dir, args.model)
    os.makedirs(savedir, exist_ok=True)

    # Load checkpoint if exists
    start_step = 1
    
    if checkpoints:
        latest_model = checkpoints
    else:
        latest_model = find_latest_checkpoint(savedir)
    if latest_model:
        checkpoint = torch.load(latest_model, map_location='cpu', weights_only=True)
        net_model.load_state_dict(checkpoint['net_model'], strict=False)
        ema_model.load_state_dict(checkpoint['ema_model'], strict=False)
        start_step = checkpoint['step']
        if 'text_model' in checkpoint and text_model is not None:
            text_model.load_state_dict(checkpoint['text_model'], strict=False)
        if 'vision_model' in checkpoint and vision_model is not None:
            vision_model.load_state_dict(checkpoint['vision_model'], strict=False)
        try:       
            if 'wass_model' in checkpoint and wass_model is not None:
                wass_model.load_state_dict(checkpoint['wass_model'], strict=False)
                wass_model = wass_model.to(wass_model.device)
        except:
            pass
        
        print(f"Resuming from step {start_step}")
    net_model, vae, text_model, vision_model = dispatch_model(net_model, vae, text_model, vision_model, num_device=2)

    # Temporary patch. TODO: remove global_step later.
    global_step = start_step

    # Training Loop
    with trange(start_step, args.total_steps, initial=start_step, total=args.total_steps,
                dynamic_ncols=True) as step_pbar:
        for step in step_pbar:
            global_step += 1
            args.op_match_batch = random.random() >= 0.25

            optim.zero_grad()
            if train_wass_model:
                wass_optim.zero_grad()
            if train_text_encoder and text_model is not None:
                text_optim.zero_grad()
            if train_vision_encoder and vision_model is not None:
                vision_optim.zero_grad()

            # Get batch
            datalooper = random.choice(dataloopers)
            batch = next(datalooper)
   
            x0 = batch['x0']
            x1 = batch['x1']
            x1_path = batch['x1_path']
            caption = batch['caption']
            hint = batch['hint']
            
            if 'mask_hint' in batch and random.random() < 0.75:
                mask_hint = batch['mask_hint']
            else:
                mask_hint = torch.zeros_like(hint)

            if random.random() > 0.75:
                y = (torch.ones_like(batch['class_id']) * 999).long() 
            else: 
                y = batch['class_id']

            if vae is not None:
                if args.batch_size <= 8:
                    latents = _encode_latent(torch.cat([x0, x1], dim=0))
                    x0, x1 = latents.chunk(2, dim=0)
                else:
                    x0 = _encode_latent(x0).detach()
                    x1 = _encode_latent(x1).detach()

            if text_model is not None:
                caption_input = text_tokenizer(caption, return_tensors="pt", padding=True).to(text_model.device)
                if train_text_encoder:
                    caption_outputs = text_model(**caption_input)
                else:
                    with torch.no_grad():
                        caption_outputs = text_model(**caption_input)
                text_embeds = caption_outputs['last_hidden_state'].to(text_model.device)  # [B, S, D]
            else:
                text_embeds = torch.empty((len(y)))

            if vision_model is not None:
                x1_images = torch.stack([process_single_image(image_path) for image_path in x1_path])
                if train_vision_encoder:
                    vision_embeds = vision_model.forward_features(x1_images.to(vision_model.device))
                else:
                    with torch.no_grad():
                        vision_embeds = vision_model.forward_features(x1_images.to(vision_model.device))
            else:
                vision_embeds = torch.empty((len(y)))

            if caption_hidden_states_mode == 'cat':  # Remove CLS tokens.
                caption_hidden_states = torch.cat([vision_embeds[:, 1:, ...].to(net_model.device),
                                                   text_embeds[:, 1:, ...].to(net_model.device)], dim=1)
            elif caption_hidden_states_mode == 'only_text':
                caption_hidden_states = text_embeds[:, 1:, ...].to(net_model.device)

            # Flow matching core
            cond = {
                'x0': x0.to(net_model.device),
                'x1': x1.to(net_model.device),
                'caption': caption,
                'caption_hidden_states': caption_hidden_states.to(net_model.device),
                'y': y.to(net_model.device),
                'hint': hint.to(net_model.device),
                'mask_hint': mask_hint.to(net_model.device),
                'text_embeds': text_embeds[:, 0].to(net_model.device),  # [B, D]
                'image_embeds': vision_embeds[:, 0].to(net_model.device)  # [B, D]
            }
            
            t, xt, ut = FM.get_sample_location_and_conditional_flow(x0, x1, sample_plan=args.op_match_batch, cond=cond, print_info=False, replace=False)

            x0 = cond['x0']
            x1 = cond['x1']
            y = cond['y']
            caption_hidden_states = cond['caption_hidden_states']
            xt = xt.to(net_model.device)
            t = t.to(net_model.device)
            y = y.to(net_model.device)
            ut = ut.to(net_model.device)
            caption_hidden_states = caption_hidden_states.to(net_model.device)
            
            if net_model.config.addition_embed_type in ["image_hint", "text_image", "double_image_hint",
                                                        "double_merge_image_hint"]:
                vt = net_model(xt, timestep=t.squeeze(), class_labels=y, encoder_hidden_states=caption_hidden_states,
                               added_cond_kwargs=cond, image_hint_model='cat').sample
            else:
                vt = net_model(xt, timestep=t.squeeze(), class_labels=y,
                               encoder_hidden_states=caption_hidden_states).sample

            loss = torch.mean((vt - ut) ** 2)

            Wasserstein_loss = True
            Wasserstein_loss_multi_step = True
            wasserstein_loss_beta = 1024
            wasserstein_loss_beta_neighbor = 102.4
            if Wasserstein_loss:
                mask = torch.Tensor([check_str(batch['label_A'][idx], batch['label_B'][idx]) for idx in range(len(batch["label_B"]))]).to(wass_model.device).bool()
                if mask.sum() > 0:
                    # print("Cal Wasserstein Positive Batch-Size: ", mask.sum().item())
                    if Wasserstein_loss_multi_step and random.random() > 0.5:
                        """Euler solver with Steps"""
                        # if torch.cuda.device_count() == 4:
                        #     device = "cuda:3"
                        # else:
                        #     device = wass_model.device
                        x0 = x0.to(wass_model.device)
                        x1 = x1.to(wass_model.device)
                        vt = vt.to(wass_model.device)
                        x = x0[mask].to(wass_model.device)

                        ### Fixed time steps
                        num_points = random.choice([8, 12, 16, 20, 24])

                        random_t = torch.rand(num_points, device=wass_model.device)
                        t_span = torch.sort(random_t)[0]
                        t_span = torch.clamp(t_span / (t_span.max() + 0.1), 0.0, 1.0) # Normalize.
                        t_start = 0
                        for t_idx in range(len(t_span)):
                            t = t_span[t_idx]
                            dt = t - t_start
                            t_start += t
                            if random.random() < 0.5:
                                x = x + vt[mask] * dt
                            else:
                                x = x + ut.to(wass_model.device)[mask] * dt
                            loss = loss + (wasserstein_loss_beta_neighbor * cal_wasserstein_loss(wass_model.encode(x), wass_model.encode(x1)).mean()).to(loss.device)
                    else:
                        """Euler solver with one-Step"""
                        x0 = x0.to(wass_model.device)
                        x1 = x1.to(wass_model.device)
                        vt = vt.to(wass_model.device)
                        dt = torch.clamp(t.to(wass_model.device), min=0.2, max=0.8)
                        wass_loss = wasserstein_loss_beta * cal_wasserstein_loss(wass_model.encode(x0[mask] + vt[mask] * dt[mask]), wass_model.encode(x1[mask])).mean().to(loss.device)
                        loss = loss + wass_loss  # The original loss directly optimizes MSE between x+vt and x1.
                
                # if mask.sum() < len(mask):
                #     mask = ~ mask # Invert mask.
                #     if mask.sum() > 0:
                #         print("Cal Wasserstein Negative Batch-Size: ", mask.sum().item())
                #         if Wasserstein_loss_multi_step and random.random() > 0.5:
                #             """Euler solver with Steps"""
                #             x = x0[mask].to(wass_model.device)
                #             x1 = x1.to(wass_model.device)
                #             vt = vt.to(wass_model.device)
                #             t_span = torch.linspace(0, 1, 5, device=net_model.device)
                #             for t_idx in range(len(t_span) - 1):
                #                 t = t_span[t_idx] * torch.ones(x[mask].shape[0], device=net_model.device)
                #                 x = x + vt[mask] * t
                #                 loss = loss + (torch.min(neg_betas(), 0) * wasserstein_loss_beta_neighbor * cal_wasserstein_loss(wass_model.encode(x), wass_model.encode(x1))).mean() / (
                #                             len(t_span) - 1)
                #         else:
                #             """Euler solver with one-Step"""
                #             x0 = x0.to(wass_model.device)
                #             x1 = x1.to(wass_model.device)
                #             vt = vt.to(wass_model.device)
                #             wass_loss = wasserstein_loss_beta * cal_wasserstein_loss(wass_model.encode(x0[mask] + vt[mask]),
                #                                                                      wass_model.encode(x1[mask]))
                #             loss = loss + (torch.min(neg_betas(), 0) * wass_loss).mean().to(loss.device)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(net_model.parameters(), args.grad_clip)
            if train_wass_model:
                wass_grad_norm = torch.nn.utils.clip_grad_norm_(wass_model.parameters(), args.grad_clip)
            else:
                wass_grad_norm = torch.Tensor([-1])
                
            if train_text_encoder and text_model is not None:
                text_grad_norm = torch.nn.utils.clip_grad_norm_(text_model.parameters(), args.grad_clip*0.5)
            else:
                text_grad_norm = torch.Tensor([-1])
            if train_vision_encoder and vision_model is not None:
                vision_grad_norm = torch.nn.utils.clip_grad_norm_(vision_model.parameters(), args.grad_clip*0.5)
            else:
                vision_grad_norm = torch.Tensor([-1])
     
            optim.step()
            sched.step()
            if train_wass_model:
                wass_optim.step()
    
            if train_text_encoder and text_model is not None:
                text_optim.step()
                text_sched.step()
            if train_vision_encoder and vision_model is not None:
                vision_optim.step()
                vision_sched.step()

            ema(net_model, ema_model, args.ema_decay)
            ema_model = ema_model.to(net_model.device)

            # Logging
            writer.add_scalar('Training/Loss', loss.item(), global_step)
            writer.add_scalar('Training/Grad Norm', grad_norm.item(), global_step)
            writer.add_scalar('Training/Text Grad Norm', text_grad_norm.item(), global_step)
            writer.add_scalar('Training/Vision Grad Norm', vision_grad_norm.item(), global_step)
            writer.add_scalar('Training/Wass Grad Norm', wass_grad_norm.item(), global_step)
            writer.add_scalar('Learning Rate', optim.param_groups[0]['lr'], global_step)

            # Sample and save
            if args.save_step > 0 and global_step % args.save_step == 0:
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "text_model": text_model.state_dict() if text_model is not None else None,
                        "vision_model": vision_model.state_dict() if vision_model is not None else None,
                        "wass_model": wass_model.state_dict() if wass_model is not None else None,
                        "step": global_step,
                    },
                    os.path.join(savedir, f"{args.model}_weights_step_{global_step}.pt"),
                )
                if train_wass_model:
                    torch.save(wass_model.state_dict(), os.path.join(savedir, f"{args.model}_wass_model_{global_step}.pt"))

                cleanup_old_checkpoints(savedir, args.keep_n_checkpoints)

            step_pbar.set_description( # wass_grad_norm
                f"loss: {loss.item():.4f} GradNorm: {grad_norm.item():.4f} Text GradNorm: {text_grad_norm.item():.4f} Vision GradNorm: {vision_grad_norm.item():.4f} WassModel GradNorm: {wass_grad_norm.item():.4f}")

    # Close TensorBoard writer
    writer.close()


def main():
    """Main entry point."""
    args = parse_arguments()
    train(args)

if __name__ == "__main__":
    main()