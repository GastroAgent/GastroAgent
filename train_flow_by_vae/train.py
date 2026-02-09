import argparse
import copy
import glob
import math
import os
from timm.models.vision_transformer import VisionTransformer
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
import gc
from transformers import ChineseCLIPConfig as CLIPConfig
from transformers import ChineseCLIPProcessor as CLIPProcessor
from transformers import ChineseCLIPModel as CLIPModel
from transformers import AutoTokenizer, AutoModel, AutoConfig, ChineseCLIPTextModel, ChineseCLIPTextConfig
import json
import torch
import random
from torch import nn
from torch.nn import init
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import transforms as T
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from diffusers import AutoencoderKL
import sys
from functools import partial
from PIL import Image
from tqdm import tqdm
from safetensors.torch import load_model, load_file

sys.path.append('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_十二指肠')
sys.path.append('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_十二指肠/utils')

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

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Flow Matching Training Script")

    # Model configuration
    parser.add_argument("--model", type=str, default="otcfm", choices=["otcfm", "icfm"],
                        help="Flow matching model type")

    parser.add_argument("--output_dir", type=str,
                        default="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_十二指肠/outputs/image_hint_十二指肠3",
                        help="Output directory")
    
    # UNet configuration
    parser.add_argument("--num_channel", type=int, default=128,
                        help="Base channel of UNet")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Target learning rate")

    parser.add_argument("--grad_clip", type=float, default=2.0,
                        help="Gradient norm clipping")

    parser.add_argument("--total_steps", type=int, default=70000,
                        help="Total training steps")

    parser.add_argument("--warmup", type=int, default=1500,
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
                        help="矫正流")
    # Evaluation parameters
    parser.add_argument("--save_step", type=int, default=5000,
                        help="Frequency of saving checkpoints (0 to disable)")

    # Image dataset parameters
    parser.add_argument("--image_dir", type=str,
                        default="",
                        help="Directory containing training images")

    # Logging parameters
    parser.add_argument("--log_dir", type=str,
                        default="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_十二指肠/logs",
                        help="TensorBoard log directory")

    # last n checkpoints to save, delete the rest checkpoints for saving the disk space
    parser.add_argument("--keep_n_checkpoints", type=int, default=10,
                        help="Number of previous checkpoints to keep")
    return parser.parse_args()

def train(args):
    """Main training function."""
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_posterior = True
    random_sample_posterior = False
    train_text_encoder = True
    train_vision_encoder = True
    use_image_feature = True
    use_text_feature = True
    caption_hidden_states_mode = 'cat' # only_text
    use_image_mask = True
    checkpoints = ''
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
        transforms.RandomGrayscale(p=1),  # 数据增强：20% 概率灰度化
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataloaders = []
    json_paths = glob.glob(
        "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/data_tsy_12/train_json/data_pairs_flow/*.json")
    
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
        
    print('DataLoad Total Steps: ', sum([len(dataloader) for dataloader in dataloaders]))
    dataloopers = [infiniteloop(dataloader) for dataloader in dataloaders]

    ### Model initialization
    config = json.load(open('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_十二指肠/flow_matcher_otcfm/unet/config.json', 'r'))
    net_model = UNet2DConditionModel(**config)
    try:
        net_model.load_state_dict(torch.load('/mnt/inaisfs/data/home/tansy_criait/flow_match/flow_matcher_otcfm/unet/diffusion_pytorch_model.bin'), strict=False)
    except RuntimeError as e:
        print(e)
        pass
    vae = AutoencoderKL.from_pretrained('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_十二指肠/flow_matcher_otcfm/vae').eval()
    
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
        vision_model.load_state_dict(torch.load("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_十二指肠/flow_matcher_otcfm/EndoViT/pytorch_model.bin", weights_only=False))
    else:
        vision_model = None

        def process_single_image(x):
            return x

    if use_text_feature:
        config_json = json.load(open('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_十二指肠/flow_matcher_otcfm/text_encoder/config.json', 'r'))
        config = ChineseCLIPTextConfig(**config_json)
        text_model = ChineseCLIPTextModel(config, False).eval()
        text_tokenizer = AutoTokenizer.from_pretrained(
            '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_十二指肠/flow_matcher_otcfm/text_encoder', use_fast=True)
        try:
            state_dict = load_file("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_十二指肠/flow_matcher_otcfm/text_encoder/model.safetensors")
            text_model.load_state_dict(state_dict)
        except:
            pass
    else:
        text_model = None
        text_tokenizer = None

    net_model = net_model.to(device=device).train()

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
    #     '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_十二指肠/outputs/disease_A2B/otcfm_weights_step_2000_A2D.pt')
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

    # Flow Matcher initialization
    sigma = 0.1
    if args.model == "otcfm":
        FM = OptimalTransportConditionalFlowMatcher(sigma=sigma, ot_method='exact')
    elif args.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
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
        checkpoint = torch.load(latest_model, map_location=device, weights_only=True)
        net_model.load_state_dict(checkpoint['net_model'], strict=False)
        ema_model.load_state_dict(checkpoint['ema_model'], strict=False)
        start_step = checkpoint['step']
        if 'text_model' in checkpoint and text_model is not None:
            text_model.load_state_dict(checkpoint['text_model'], strict=False)
        if 'vision_model' in checkpoint and vision_model is not None:
            vision_model.load_state_dict(checkpoint['vision_model'], strict=False)
        print(f"Resuming from step {start_step}")
    net_model, vae, text_model, vision_model = dispatch_model(net_model, vae, text_model, vision_model)

    # Ptach work for now. TODO: Remove the Global steps later
    global_step = start_step

    # Training Loop
    with trange(start_step, args.total_steps, initial=start_step, total=args.total_steps,
                dynamic_ncols=True) as step_pbar:
        for step in step_pbar:
            if random.random() < 0.25:
                args.op_match_batch = False
            else:
                args.op_match_batch = True

            global_step += 1

            optim.zero_grad()
            if train_text_encoder and text_model is not None:
                text_optim.zero_grad()
            if train_vision_encoder and vision_model is not None:
                vision_optim.zero_grad()

            # Get batch
            datalooper = random.choice(dataloopers)
            batch = next(datalooper)
            batch_size = args.batch_size

            x0 = batch['x0']
            x1 = batch['x1']
            x1_path = batch['x1_path']
            caption = batch['caption']
            hint = batch['hint']
            # if 'mask_hint' in batch and random.random() < 0.75:
            #     mask_hint = batch['mask_hint']
            # else:
            #     mask_hint = torch.zeros_like(hint)

            if random.random() > 0.75:
                y = (torch.ones_like(batch['class_id']) * 999).long() 
                caption = ['将当前症状表现转变为相关条件所描述出的症状。' for x in batch['caption']]
            else: 
                y = batch['class_id']
                
            if vae is None:
                pass
            else:
                if True:
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
                # 'mask_hint': mask_hint.to(net_model.device),
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
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(net_model.parameters(), args.grad_clip)
            if train_text_encoder and text_model is not None:
                text_grad_norm = torch.nn.utils.clip_grad_norm_(text_model.parameters(), args.grad_clip*0.2)
            else:
                text_grad_norm = torch.Tensor([-1])
            if train_vision_encoder and vision_model is not None:
                vision_grad_norm = torch.nn.utils.clip_grad_norm_(vision_model.parameters(), args.grad_clip*0.2)
            else:
                vision_grad_norm = torch.Tensor([-1])
          
            optim.step()
            sched.step()
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
            writer.add_scalar('Training/Text-Grad Norm', text_grad_norm.item(), global_step)
            writer.add_scalar('Training/Vision-Grad Norm', vision_grad_norm.item(), global_step)
            writer.add_scalar('Learning Rate', optim.param_groups[0]['lr'], global_step)

            # Sample and save
            if args.save_step > 0 and global_step % args.save_step == 0:
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "text_model": text_model.state_dict() if text_model is not None else None,
                        "vision_model": vision_model.state_dict() if vision_model is not None else None,
                        "step": global_step,
                    },
                    os.path.join(savedir, f"{args.model}_weights_step_{global_step}.pt"),
                )

                cleanup_old_checkpoints(savedir, args.keep_n_checkpoints)

            step_pbar.set_description(
                f"loss: {loss.item():.4f} GradNorm: {grad_norm.item():.4f} Text GradNorm: {text_grad_norm.item():.4f} Vision GradNorm: {vision_grad_norm.item():.4f}")

    writer.close()

def main():
    """Main entry point."""
    args = parse_arguments()
    train(args)

if __name__ == "__main__":
    main()