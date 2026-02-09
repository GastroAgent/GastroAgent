import argparse
import copy
import glob
import math
import os
from timm.models.vision_transformer import VisionTransformer
import torch.nn.functional as F
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
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

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Flow Matching Training Script")

    # Model configuration
    parser.add_argument("--model", type=str, default="otcfm", choices=["otcfm", "icfm"],
                        help="Flow matching model type")

    parser.add_argument("--output_dir", type=str,
                        default="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/outputs/flow-match_vae_gan_十二指肠3",
                        help="Output directory")
    
    # UNet configuration
    parser.add_argument("--num_channel", type=int, default=128,
                        help="Base channel of UNet")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Target learning rate")

    parser.add_argument("--grad_clip", type=float, default=2.0,
                        help="Gradient norm clipping")

    parser.add_argument("--total_steps", type=int, default=100000,
                        help="Total training steps")

    parser.add_argument("--warmup", type=int, default=1500,
                        help="Learning rate warmup steps")

    parser.add_argument("--batch_size", type=int, default=4,
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
                        default="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/logs",
                        help="TensorBoard log directory")

    # last n checkpoints to save, delete the rest checkpoints for saving the disk space
    parser.add_argument("--keep_n_checkpoints", type=int, default=10,
                        help="Number of previous checkpoints to keep")
    return parser.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# ----------------------------
# 带轻量通道注意力的残差下采样块（2 conv + SE）
# ----------------------------
class AttentiveResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.norm1 = nn.InstanceNorm2d(out_ch)
        self.norm2 = nn.InstanceNorm2d(out_ch)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.downsample = nn.AvgPool2d(2) if downsample else None
        
        # Skip connection
        if in_ch != out_ch or downsample:
            self.skip = spectral_norm(nn.Conv2d(in_ch, out_ch, 1, bias=False))
        else:
            self.skip = None

        # 轻量 SE 注意力（不增加卷积层计数，视为附属模块）
        reduction = min(8, out_ch // 8)  # 避免过小通道
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            spectral_norm(nn.Conv2d(out_ch, out_ch // reduction, 1)),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(out_ch // reduction, out_ch, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        if self.skip is not None:
            identity = self.skip(identity)
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.downsample is not None:
            out = self.downsample(out)

        out = out + identity
        out = out * self.se(out)  # 通道加权
        return self.relu(out)

# ----------------------------
# 主模型：带注意力 + 多级融合
# ----------------------------
class AttentiveFusionPatchDiscriminator(nn.Module):
    """
    16 层强力判别器，集成：
      - 分阶段下采样
      - 每个主干块含通道注意力
      - 多尺度特征融合（16x16, 8x8, 4x4）
    输入: (B, 4, 64, 64)
    输出: (B, 1, 4, 4)
    卷积层数统计（仅计 Conv2d）：
      - init: 2
      - s1 (32→16): 2
      - s2 (16→8): 2
      - s3 (8→4): 2
      - refine4: 2
      - proj16/proj8/proj4: 3
      - fuse head: 2
      - 总计: 15 层 Conv2d（注意力中的 1x1 不额外计为主干层）
    """
    def __init__(self, in_channels=4, ndf=64, device="cuda:2"):
        super().__init__()

        # Stage 0: 64 → 32
        self.s0 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, ndf, 3, padding=1)),      # L1
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf, 4, stride=2, padding=1)),    # L2
            nn.InstanceNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Stage 1: 32 → 16
        self.s1 = AttentiveResBlock(ndf, ndf * 2, downsample=True)         # L3-4

        # Stage 2: 16 → 8
        self.s2 = AttentiveResBlock(ndf * 2, ndf * 4, downsample=True)     # L5-6

        # Stage 3: 8 → 4
        self.s3 = AttentiveResBlock(ndf * 4, 512, downsample=True)         # L7-8

        # Refine at 4x4 (no downsample)
        self.refine4 = AttentiveResBlock(512, 512, downsample=False)       # L9-10

        # Projection for fusion (1x1 convs)
        self.proj16 = spectral_norm(nn.Conv2d(ndf * 2, 128, 1))            # L11
        self.proj8  = spectral_norm(nn.Conv2d(ndf * 4, 128, 1))            # L12
        self.proj4  = spectral_norm(nn.Conv2d(512, 128, 1))                # L13

        # Fusion head
        self.fuse = nn.Sequential(
            spectral_norm(nn.Conv2d(384, 128, 3, padding=1)),              # L14
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 1, 3, padding=1))                 # L15
        )

        self.s0 = self.s0.to(device)
        self.s1 = self.s1.to(device)
        self.s2 = self.s2.to(device)
        self.s3 = self.s3.to(device)
        self.refine4 = self.refine4.to(device)
        self.proj16 = self.proj16.to(device)
        self.proj8 = self.proj8.to(device)
        self.proj4 = self.proj4.to(device)
        self.fuse = self.fuse.to(device)

    def forward(self, x):
        x = self.s0(x)          # 32x32
        x = self.s1(x)          # 16x16 → f16
        f16 = x
        x = self.s2(x)          # 8x8   → f8
        f8 = x
        x = self.s3(x)          # 4x4
        x = self.refine4(x)     # refined 4x4 → f4
        f4 = x

        # Upsample to 4x4
        f16_up = F.interpolate(f16, size=(4, 4), mode='bilinear', align_corners=False)
        f8_up  = F.interpolate(f8,  size=(4, 4), mode='bilinear', align_corners=False)
        
        # Project to same channel
        p16 = self.proj16(f16_up)
        p8  = self.proj8(f8_up)
        p4  = self.proj4(f4)

        # Fuse
        fused = torch.cat([p16, p8, p4], dim=1)  # (B, 384, 4, 4)
        out = self.fuse(fused)  # (B, 1, 4, 4)

        return out  # raw logits

# ==============================
# Hinge Loss Functions
# ==============================
def hinge_discriminator_loss(real_pred, fake_pred):
    d_real_loss = torch.mean(torch.relu(1.0 - real_pred))
    d_fake_loss = torch.mean(torch.relu(1.0 + fake_pred))
    return d_real_loss + d_fake_loss

def hinge_generator_loss(fake_pred):
    return 1 - torch.mean(fake_pred)

def requires_grad(model, flag):
    """
    设置模型中所有参数的 requires_grad 属性。

    Args:
        model (torch.nn.Module): 要操作的模型。
        flag (bool): True 表示启用梯度（训练），False 表示冻结参数（不计算梯度）。
    """
    for param in model.parameters():
        param.requires_grad = flag

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
    discriminator = AttentiveFusionPatchDiscriminator(device="cuda:2")
    discriminator.load_state_dict(torch.load("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/outputs/flow-match_vae_gan_十二指肠3/discriminator.pt"))
    discriminator.device = "cuda:2"
    discriminator = discriminator.to("cuda:2")
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-3, weight_decay=1e-3)
    dis_warmup_steps = 1000
    
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
        transforms.RandomGrayscale(p=1),  
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataloaders = []
    json_paths = glob.glob("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/data_tsy1/train_json/All_data_pairs_dia/*.json")
    # json_paths += glob.glob("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/data_tsy_12/train_json/data_pairs_flow/*.json")
    # json_paths += glob.glob("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/data_tsy_12/train_json/data_pairs_flow_54/*.json")
    
    for json_path in tqdm(json_paths):
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
        print(dataset.dataset[0]['label_A'], ': ', len(dataset))
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
    config = json.load(open('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/flow_matcher_otcfm/unet/config.json', 'r'))
    net_model = UNet2DConditionModel(**config)
    try:
        net_model.load_state_dict(torch.load('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/flow_matcher_otcfm/unet/diffusion_pytorch_model.bin'), strict=False)
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
    discriminator.train()
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
        start_step = checkpoint['step'] - 100
        if 'text_model' in checkpoint and text_model is not None:
            text_model.load_state_dict(checkpoint['text_model'], strict=False)
        if 'vision_model' in checkpoint and vision_model is not None:
            vision_model.load_state_dict(checkpoint['vision_model'], strict=False)
        if "discriminator" in checkpoint and discriminator is not None:
            discriminator.load_state_dict(checkpoint['discriminator'], strict=True)
            # torch.save(discriminator.state_dict(), "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/outputs/flow-match_vae_gan/discriminator.pt")
        print(f"Resuming from step {start_step}")
    net_model, vae, text_model, vision_model = dispatch_model(net_model, vae, text_model, vision_model, num_device=3)
    torch.cuda.empty_cache()
    # Ptach work for now. TODO: Remove the Global steps later
    global_step = start_step
    dis_warmup_start_step = 0 
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
            
            # if 'mask_hint' in batch:
            #     mask_hint = batch['mask_hint']
            # else:
            #     if use_image_mask:
            #         mask_hint = torch.randn_like(x0)
            #     else:
            #         mask_hint = None

            if random.random() > 0.75:
                y = (torch.ones_like(batch['class_id']) * 999).long() 
                caption = [' ' for x in batch['caption']]
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

            # -------------------------
            # Train Generator (Flow-Match)
            # -------------------------
            requires_grad(discriminator, False)
            requires_grad(net_model, True)
            if net_model.config.addition_embed_type in ["image_hint", "text_image", "double_image_hint",
                                                        "double_merge_image_hint"]:
                vt = net_model(xt, timestep=t.squeeze(), class_labels=y, encoder_hidden_states=caption_hidden_states,
                               added_cond_kwargs=cond, image_hint_model='cat').sample
            else:
                vt = net_model(xt, timestep=t.squeeze(), class_labels=y,
                               encoder_hidden_states=caption_hidden_states).sample

            dis_warmup_start_step += 1
            if dis_warmup_start_step >= dis_warmup_steps:
                loss = torch.mean((vt - ut) ** 2)
                
                fake_pred_for_g = discriminator(x0.to(discriminator.device) + vt.to(discriminator.device))
                adv_loss = hinge_generator_loss(fake_pred_for_g)
                # t = torch.rand(vt.shape[0]).to(discriminator.device, vt.dtype).reshape(-1, 1, 1, 1)

                t = torch.round(torch.rand(vt.shape[0]) * 10) / 10
                t = t.to(discriminator.device, vt.dtype).reshape(-1, 1, 1, 1)
                process_adv_loss = hinge_generator_loss(discriminator(x0.to(discriminator.device) + t * vt.to(discriminator.device)))
                loss = loss.to(adv_loss.device) + torch.clamp(0.1 * adv_loss + 0.1 * process_adv_loss, max=0.2)
                
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
                step_pbar.set_description(
                    f"loss: {loss.item():.4f} GradNorm: {grad_norm.item():.4f} Text GradNorm: {text_grad_norm.item():.4f} Vision GradNorm: {vision_grad_norm.item():.4f}")
            # -------------------------
            # Train Discriminator
            # -------------------------
            requires_grad(discriminator, True)
            requires_grad(net_model, False)

            optimizer_d.zero_grad()

            # 关键：detach 输入，避免任何梯度泄漏
            real_pred = discriminator(x1.to(discriminator.device))
            
            with torch.no_grad():
                recon = x0.to(discriminator.device) + vt.detach().to(discriminator.device)
            fake_pred = discriminator(recon)

            d_loss = hinge_discriminator_loss(real_pred, fake_pred)
            
            # t = torch.rand(vt.shape[0]).to(discriminator.device, vt.dtype).reshape(-1, 1, 1, 1)

            t = torch.round(torch.rand(vt.shape[0]) * 10) / 10
            t = t.to(discriminator.device, vt.dtype).reshape(-1, 1, 1, 1)
            with torch.no_grad():
                recon2 = x0.to(discriminator.device) + vt.detach().to(discriminator.device) * t
                
            fake_pred2 = discriminator(recon2)
            d_loss = d_loss + hinge_discriminator_loss(real_pred, fake_pred2)
            d_loss.backward()
            d_grad_norm = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
            optimizer_d.step()
\
            # Logging
            writer.add_scalar('Discriminator Training/Loss', d_loss.item(), global_step)
            writer.add_scalar('Discriminator Training/Grad Norm', d_grad_norm.item(), global_step)
            writer.add_scalar('Discriminator Learning Rate', optimizer_d.param_groups[0]['lr'], global_step)
            if step % 10 == 0:
                step_pbar.set_description(
                    f"Discriminator loss: {d_loss.item():.4f} Discriminator GradNorm: {d_grad_norm.item():.4f}")

            if args.save_step > 0 and global_step % args.save_step == 0:
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "text_model": text_model.state_dict() if text_model is not None else None,
                        "vision_model": vision_model.state_dict() if vision_model is not None else None,
                        "discriminator": discriminator.state_dict() if vision_model is not None else None,
                        "step": global_step,
                    },
                    os.path.join(savedir, f"{args.model}_weights_step_{global_step}.pt"),
                )
                torch.save(discriminator.state_dict(), "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/outputs/flow-match_vae_gan/discriminator.pt")
                cleanup_old_checkpoints(savedir, args.keep_n_checkpoints)
    writer.close()


def main():
    """Main entry point."""
    args = parse_arguments()
    train(args)


if __name__ == "__main__":
    main()