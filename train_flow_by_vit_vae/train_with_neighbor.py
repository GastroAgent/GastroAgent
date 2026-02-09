import argparse
import copy
import glob
import math
import os
from timm.models.vision_transformer import VisionTransformer
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,4,5'
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
from typing import Optional, Union, List
sys.path.append('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy')
sys.path.append('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/utils')

from train_utils import (
    find_latest_checkpoint,
    cleanup_old_checkpoints,
    ema, infiniteloop,
    warmup_lr
)
# from data_utils import create_dataset, create_dataloaders
from data_loader_test import MedicalJsonDataset

from conditional_flow_matcher import ConditionalFlowMatcher, OptimalTransportConditionalFlowMatcher
from my_models.unet_model import UNetModelWrapper
from my_models.unet_2d_condition import UNet2DConditionModel
from my_models.model_dispatch import dispatch_model
from wass_flow_train.model import TripletNetwork
from wass_flow_train.loss import *

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Flow Matching Training Script")

    # Model configuration
    parser.add_argument("--model", type=str, default="otcfm", choices=["otcfm", "icfm"],
                        help="Flow matching model type")

    parser.add_argument("--output_dir", type=str,
                        default="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/outputs/flow-match_vae",
                        help="Output directory")

    # UNet configuration
    parser.add_argument("--num_channel", type=int, default=128,
                        help="Base channel of UNet")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Target learning rate")

    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient norm clipping")

    parser.add_argument("--total_steps", type=int, default=100000,
                        help="Total training steps")

    parser.add_argument("--warmup", type=int, default=500,
                        help="Learning rate warmup steps")

    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")

    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for DataLoader")

    parser.add_argument("--ema_decay", type=float, default=0.95,
                        help="EMA decay rate")

    parser.add_argument("--op_match_batch", type=bool, default=True,
                        help="op_match_batch")

    # Evaluation parameters
    parser.add_argument("--save_step", type=int, default=5000,
                        help="Frequency of saving checkpoints (0 to disable)")

    # Image dataset parameters
    parser.add_argument("--image_dir", type=str,
                        default="",
                        help="Directory containing training images")

    # Logging parameters
    parser.add_argument("--log_dir", type=str,
                        default="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/logs_wm_neighbor",
                        help="TensorBoard log directory")

    # last n checkpoints to save, delete the rest checkpoints for saving the disk space
    parser.add_argument("--keep_n_checkpoints", type=int, default=10,
                        help="Number of previous checkpoints to keep")
    return parser.parse_args()

str_map = {
    '息肉': "Polyps",
    "染色息肉": "Dyed_lifted_polyps",
    "染色边缘": "Dyed_resection_margins",
    "食管炎": "Esophagitis",
    "溃疡结肠炎": "Ulcerative_colitis"
}
def check_str(st1, st2):
    if st1 == st2:
        return True
    if st1 in str_map:
        str1 = str_map[st1]
    else:
        str1 = st1
    if st2 in str_map:
        str2 = str_map[st2]
    else:
        str2 = st2
    if str1 == str2:
        return True
    return False

def neg_betas(**kwargs):  # 负样本的优化权重。
    '''

    return:
        int, Tensor.scaler 或 Tensor.Shape: [B]
    '''
    return -1

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

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
    use_image_mask = False
    checkpoints = ''
    train_wass_model = False
    
    ### 噪声调度器。
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(f'/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/flow_matcher_otcfm/scheduler')
    scheduler.sigma_min = 0.0
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        10,
        device,
        mu=0.75
    ) # x0 和 x1 是反的，所以我们的采样t 也是从0开始，官方的是从1开始。因为是从 x1生成 x0.
    timesteps = torch.cat([torch.zeros(1, device=timesteps.device).reshape(1), timesteps.flip(0)], dim=0) / 1000
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
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),], p=0.1), 
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 0.5))], p=0.1), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    transform_grey = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.RandomGrayscale(p=1),  # 数据增强：20% 概率灰度化
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataloaders = []
    json_paths = glob.glob(
        "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/data_tsy1/train_json/data_pairs_flow/*.json")
    # json_paths = json_paths[:5] # debug
    for json_path in tqdm(json_paths):
        try:
            dataset = MedicalJsonDataset(
                path=json_path,
                transform=transform,
                hint_transform=transform_grey,
                transform_A=transform_A,
                transform_B=transform,
                transform_mask = transform_mask
            )
            if len(dataset) < args.batch_size:
                continue
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True,
            )
            dataloaders.append(dataloader)
        # break
        except:
            continue

    print('DataLoad Total Steps: ', sum([len(dataloader) for dataloader in dataloaders]))
    dataloopers = [infiniteloop(dataloader) for dataloader in dataloaders]

    ### Model initialization
    config = json.load(open('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/flow_matcher_otcfm/unet/config.json', 'r'))
    net_model = UNet2DConditionModel(**config)
    try:
        net_model.load_state_dict(torch.load('/mnt/inaisfs/data/home/tansy_criait/flow_match/flow_matcher_otcfm/unet/diffusion_pytorch_model.bin'), strict=False)
    except RuntimeError as e:
        print(e)
        pass
    vae = AutoencoderKL.from_pretrained('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/flow_matcher_otcfm/vae').eval()
    
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
        vision_model.load_state_dict(torch.load("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/flow_matcher_otcfm/EndoViT/pytorch_model.bin", weights_only=False))
    else:
        vision_model = None

        def process_single_image(x):
            return x
     
    if use_text_feature:
        config_json = json.load(open('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/flow_matcher_otcfm/text_encoder/config.json', 'r'))
        config = ChineseCLIPTextConfig(**config_json)
        text_model = ChineseCLIPTextModel(config, False).eval()
        text_tokenizer = AutoTokenizer.from_pretrained(
            '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/flow_matcher_otcfm/text_encoder', use_fast=True)
        try:
            state_dict = load_file("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/flow_matcher_otcfm/text_encoder/model.safetensors")
            text_model.load_state_dict(state_dict)
        except:
            pass
    else:
        text_model = None
        text_tokenizer = None
        
    net_model = net_model.to(device=device).train()

    # 定义初始化函数（不再是类方法）
    def init_weights(model):
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # 使用 Kaiming 正态分布初始化（适用于 ReLU）
                init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                # 使用 Xavier 正态分布初始化
                init.xavier_normal_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm 的 weight 和 bias 初始化
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)

    init_weights(net_model)
    state_dict = {}
    # state_dict = torch.load(
    #     '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/outputs/disease_A2B/otcfm_weights_step_2000_A2D.pt')
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
    ############# Dy-Attention #############
    wass_model = TripletNetwork(pretrained=False, freeze_base=False, model='attention', dy=True).to("cuda:2").eval()
    wass_model.device = "cuda:2"
    model_size = sum(p.data.nelement() for p in wass_model.parameters())
    print(f"Wass Model params: {model_size / 1024 / 1024:.2f} M")
    state_dict = torch.load("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/best_flow_weights/attention_dy_tsy.pt")
    # "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/best_flow_weights/attention_dy_tsy_mask.pt"

    # ############# Attention #############
    # wass_model = TripletNetwork(model='attention').to("cuda:2").eval()
    # wass_model.device = "cuda:2"
    # model_size = sum(p.data.nelement() for p in wass_model.parameters())
    # print(f"Wass Model params: {model_size / 1024 / 1024:.2f} M")
    # state_dict = torch.load("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/best_flow_weights/attention_tsy.pt")
    
    # ############# Resnet34 #############
    # wass_model = TripletNetwork(model='resnet34').to("cuda:2").eval()
    # wass_model.device = "cuda:2"
    # model_size = sum(p.data.nelement() for p in wass_model.parameters())
    # print(f"Wass Model params: {model_size / 1024 / 1024:.2f} M")
    # state_dict = torch.load("/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/best_weights/simple_model_energy.pt")
    
    wass_model.load_state_dict(state_dict, strict=False)
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
        # try:       
        #     if 'wass_model' in checkpoint and wass_model is not None:
        #         wass_model.load_state_dict(checkpoint['wass_model'], strict=False)
        #         wass_model = wass_model.to(wass_model.device)
        # except:
        #     pass
        
        print(f"Resuming from step {start_step}")
    net_model, vae, text_model, vision_model = dispatch_model(net_model, vae, text_model, vision_model, num_device=3)

    # Ptach work for now. TODO: Remove the Global steps later
    global_step = start_step
    Wasserstein_loss = True
    Wasserstein_loss_multi_step = True
    wasserstein_loss_beta = 0.05
    wasserstein_loss_beta_neighbor = 0.02
    
    # Training Loop
    with trange(start_step, args.total_steps, initial=start_step, total=args.total_steps,
                dynamic_ncols=True) as step_pbar:
        for step in step_pbar:
            global_step += 1
            if random.random() < 0.25:
                args.op_match_batch = False
            else:
                args.op_match_batch = True
                
            optim.zero_grad()
            if train_wass_model:
                wass_optim.zero_grad()
            if train_text_encoder and text_model is not None:
                text_optim.zero_grad()
            if train_vision_encoder and vision_model is not None:
                vision_optim.zero_grad()

            # Get batch
            if random.random() < 0.15:
                pos = True
                datalooper = random.choice(dataloopers)
                batch = next(datalooper)
            else:
                pos = False
                dataloader_pos_id, dataloader_neg_id = random.sample(range(len(dataloaders)), k=2)
                dataloader = dataloopers[dataloader_pos_id]
                neg_dataloader = dataloopers[dataloader_neg_id]
                ans_batch = dataloader.__iter__().__next__()
                neg_batch = neg_dataloader.__iter__().__next__()
                while ans_batch['label_A'][0] == neg_batch['label_A'][0]:
                    ans_batch = dataloaders[dataloader_neg_id - 1].__iter__().__next__()
                    print("Next Batch as Ans.")
                    dataloader_neg_id = dataloader_neg_id - 1
                batch = {}
                batch['x0'] = neg_batch['x0'].clone()
                batch['x0_path'] = neg_batch['x0_path']
                batch['label_A'] = neg_batch['label_A']
                for k, v in ans_batch.items():
                    if k not in batch:
                        batch[k] = v
                # del neg_batch, ans_batch
            batch_size = args.batch_size

            x0 = batch['x0']
            x1 = batch['x1']
            x1_path = batch['x1_path']
            caption = batch['caption']
            hint = batch['hint']
            
            if 'mask_hint' in batch:
                mask_hint = batch['mask_hint']
            else:
                if use_image_mask:
                    mask_hint = torch.randn_like(x0)
                else:
                    mask_hint = None

            if random.random() > 0.75:
                y = (torch.ones_like(batch['class_id']) * 999).long() 
            else: 
                y = batch['class_id']

            if vae is None:
                pass
            else:
                if args.batch_size <= 8:
                    images = torch.cat([x0, x1], dim=0)
                    with torch.no_grad():
                        images = images.to(vae.device)
                        posterior = vae.encode(images).latent_dist
                        if random_sample_posterior:
                            if random.random() > 0.5:
                                images = posterior.sample() * 0.18215
                            else:
                                images = posterior.mode() * 0.18215
                        elif sample_posterior:
                            images = posterior.sample() * 0.18215
                        else:
                            images = posterior.mode() * 0.18215
                    x0, x1 = images.chunk(2, dim=0)
                else:
                    images = x0
                    with torch.no_grad():
                        images = images.to(vae.device)
                        posterior = vae.encode(images).latent_dist
                        if random_sample_posterior:
                            if random.random() > 0.5:
                                images = posterior.sample() * 0.18215
                            else:
                                images = posterior.mode() * 0.18215
                        elif sample_posterior:
                            images = posterior.sample() * 0.18215
                        else:
                            images = posterior.mode() * 0.18215
                    x0 = images.detach()
                    images = x1
                    with torch.no_grad():
                        images = images.to(vae.device)
                        posterior = vae.encode(images).latent_dist
                        if random_sample_posterior:
                            if random.random() > 0.5:
                                images = posterior.sample() * 0.18215
                            else:
                                images = posterior.mode() * 0.18215
                        elif sample_posterior:
                            images = posterior.sample() * 0.18215
                        else:
                            images = posterior.mode() * 0.18215
                    x1 = images.detach()

            if not train_text_encoder and text_model is not None:
                with torch.no_grad():
                    caption_input = text_tokenizer(caption, return_tensors="pt", padding=True).to(text_model.device)
                    caption_outputs = text_model(**caption_input)
                    text_embeds = caption_outputs['last_hidden_state'].to(text_model.device)  # [B, S, D]
            elif text_model is not None:
                caption_input = text_tokenizer(caption, return_tensors="pt", padding=True).to(text_model.device)
                caption_outputs = text_model(**caption_input)
                text_embeds = caption_outputs['last_hidden_state'].to(text_model.device)  # [B, S, D]
            else:
                text_embeds = torch.empty((len(y)))

            if not train_vision_encoder and vision_model is not None:
                with torch.no_grad():
                    x1_images = torch.stack([process_single_image(image_path) for image_path in x1_path])
                    vision_embeds = vision_model.forward_features(x1_images.to(vision_model.device))
            elif vision_model is not None:
                x1_images = torch.stack([process_single_image(image_path) for image_path in x1_path])
                vision_embeds = vision_model.forward_features(x1_images.to(vision_model.device))
           
            else:
                vision_embeds = torch.empty((len(y)))

            if caption_hidden_states_mode == 'cat':  # 剔除 CLS 向量。
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
                'mask_hint': mask_hint.to(net_model.device) if mask_hint is not None else None,
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
            
            if Wasserstein_loss:
                mask = torch.Tensor([check_str(batch['label_A'][idx], batch['label_B'][idx]) for idx in range(len(batch["label_B"]))]).to(wass_model.device).bool()
                optim_neg = mask * 1 + (~ mask) * -1
                wass_loss = 0
                if Wasserstein_loss_multi_step and random.random() > .5:
                    """Euler solver with Steps"""
                    x0 = x0.to(wass_model.device)
                    x1 = x1.to(wass_model.device)
                    vt = vt.to(wass_model.device)
                    x = x0.to(wass_model.device)

                    ### 固定时间步长
                    # t_span = torch.linspace(0, 1, 5, device=net_model.device)
                    ### 随机时间步长 Batch一致.
                    num_points = random.choice([4, 6, 8, 12, 16, 20, 24, 12, 8, 4])
                    # num_points = 5
                    # # # 生成 [0, 1] 范围内的随机时间点，保持递增
                    if random.random() < 0.5:
                        random_t = torch.rand(num_points, device=wass_model.device)
                        t_span = torch.sort(random_t)[0]
                        t_span = torch.clamp(t_span / (t_span.max() + torch.rand(1, device=wass_model.device) * 0.1), 0.0, 1.0) # 归一化.
                    else:
                        timesteps, num_points = retrieve_timesteps(
                            scheduler,
                            num_points,
                            device,
                            mu=0.75
                        ) # x0 和 x1 是反的，所以我们的采样t 也是从0开始，官方的是从1开始。因为是从 x1生成 x0.
                        t_span = torch.cat([torch.zeros(1, device=timesteps.device).reshape(1), timesteps.flip(0)], dim=0) / 1000
                    
                    t_start = 0
                    
                    for t_idx in range(len(t_span)):
                        t = t_span[t_idx]
                        dt = t - t_start
                        t_start += t
                        x_last = x.clone()
                        if random.random() < 0.5:
                            x = x + vt * dt.to(vt.device)
                        else:
                            x = x + ut.to(wass_model.device) * dt.to(wass_model.device)
                        wass_loss = wass_loss + (wasserstein_loss_beta_neighbor * (cal_wasserstein_loss(wass_model.encode(x_last), wass_model.encode(x)) * optim_neg).sum()).to(loss.device)
                    ### 终点对齐增强：可在多步模式末尾额外加一项，确保最终生成质量。
                    wass_loss = wass_loss + (wasserstein_loss_beta_neighbor * (cal_wasserstein_loss(wass_model.encode(x), wass_model.encode(x1)) * optim_neg).sum()).to(loss.device)
                else:
                    """Euler solver with one-Step"""
                    x = x0.to(wass_model.device)
                    x1 = x1.to(wass_model.device)
                    vt = vt.to(wass_model.device)
                    dt = torch.clamp(t.to(wass_model.device), min=0.2, max=0.8)
                    wass_loss = wasserstein_loss_beta * (cal_wasserstein_loss(wass_model.encode(x + vt * dt), wass_model.encode(x1)) * optim_neg).sum().to(loss.device) / 2
                    wass_loss = wass_loss + wasserstein_loss_beta * (cal_wasserstein_loss(wass_model.encode(x), wass_model.encode(x + vt * dt)) * optim_neg).sum().to(loss.device) / 2
            
            loss = loss + torch.clamp_min(wass_loss, - loss.detach().item() + 0.1) 
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(net_model.parameters(), args.grad_clip)
            if train_wass_model:
                wass_grad_norm = torch.nn.utils.clip_grad_norm_(wass_model.parameters(), args.grad_clip*0.5)
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
                    torch.save(wass_model.state_dict(), os.path.join(savedir, f"{args.model}_wass_model_neighbor_{global_step}.pt"))
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