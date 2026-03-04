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

from utils.train_utils import (
    find_latest_checkpoint,
    cleanup_old_checkpoints,
    ema, infiniteloop,
    warmup_lr
)
# from data_utils import create_dataset, create_dataloaders
from utils.data_loader import MedicalJsonDataset

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
                        default="./outputs/image_hint_十二指肠3",
                        help="Output directory")

    # UNet configuration
    parser.add_argument("--num_channel", type=int, default=128,
                        help="Base channel of UNet")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Target learning rate")

    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient norm clipping")

    parser.add_argument("--total_steps", type=int, default=90000,
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
                        default="./logs_wm_neighbor",
                        help="TensorBoard log directory")

    # last n checkpoints to save, delete the rest checkpoints for saving the disk space
    parser.add_argument("--keep_n_checkpoints", type=int, default=10,
                        help="Number of previous checkpoints to keep")
    return parser.parse_args()

def sinkhorn_custom_loss(bx, bx1, epsilon=0.1, n_iter=25, reduction='sum'):
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
    # Reshape input to [B, N, D]
    if bx.ndim == 2:
        B, D = bx.shape
        H = W = int((D // 4) ** 0.5)
        bx = bx.view(B, 4, H, W).permute(0, 2, 3, 1).reshape(B, H*W, -1)
    elif bx.ndim == 4:
        B, C, H, W = bx.shape
        bx = bx.permute(0, 2, 3, 1).reshape(B, H*W, C)
    elif bx.ndim == 3:
        bx = bx.unsqueeze(0)
        B, C, H, W = bx.shape
        bx = bx.permute(0, 2, 3, 1).reshape(B, H*W, C)

    if bx1.ndim == 2:
        B, D = bx1.shape
        H = W = int((D // 4) ** 0.5)
        bx1 = bx1.view(B, 4, H, W).permute(0, 2, 3, 1).reshape(B, H*W, -1)
    elif bx1.ndim == 4:
        B, C, H, W = bx1.shape
        bx1 = bx1.permute(0, 2, 3, 1).reshape(B, H*W, C)
    elif bx1.ndim == 3:
        bx1 = bx1.unsqueeze(0)
        B, C, H, W = bx1.shape
        bx1 = bx1.permute(0, 2, 3, 1).reshape(B, H*W, C)

    B, N, D = bx.shape
    _, M, _ = bx1.shape

    # Compute cost matrix: [B, N, M]
    x2 = torch.sum(bx**2, dim=-1, keepdim=True)                    # [B, N, 1]
    y2 = torch.sum(bx1**2, dim=-1, keepdim=True)                   # [B, M, 1]
    cross = torch.bmm(bx, bx1.transpose(-1, -2))                   # [B, N, M]
    cost_matrix = x2 - 2 * cross + y2.transpose(-1, -2)            # [B, N, M]
    cost_matrix = torch.clamp(cost_matrix, min=0.0)

    # Kernel matrix
    K = torch.exp(-cost_matrix / epsilon)                          # [B, N, M]

    # Uniform marginal distributions
    a = torch.ones(B, N, device=bx.device) / N                     # [B, N]
    b = torch.ones(B, M, device=bx.device) / M                     # [B, M]

    # Initialize dual variables
    u = torch.ones_like(a)                                         # [B, N]
    v = torch.ones_like(b)                                         # [B, M]

    # Sinkhorn iterations
    for _ in range(n_iter):
        u = a / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)  # [B, N]
        v = b / (torch.bmm(K.transpose(-1,-2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)  # [B, M]

    # Compute transport plan
    P = u.unsqueeze(-1) * K * v.unsqueeze(-2)  # [B, N, M]

    # Compute loss
    loss = torch.sum(P * cost_matrix, dim=(1,2))  # [B]

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
    
from geomloss import SamplesLoss
# loss_fn = SamplesLoss("laplacian", p=2, blur=0.1**0.5)
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
    if bx.ndim == 2:
        B, D = bx.shape
        H = W = int((D // 4) ** 0.5)
        bx = bx.view(B, 4, H, W).permute(0, 2, 3, 1).reshape(B, H*W, -1)
    elif bx.ndim == 4:
        B, C, H, W = bx.shape
        bx = bx.permute(0, 2, 3, 1).reshape(B, H*W, C)
    elif bx.ndim == 3:
        bx = bx.unsqueeze(0)
        B, C, H, W = bx.shape
        bx = bx.permute(0, 2, 3, 1).reshape(B, H*W, C)

    if bx1.ndim == 2:
        B, D = bx1.shape
        H = W = int((D // 4) ** 0.5)
        bx1 = bx1.view(B, 4, H, W).permute(0, 2, 3, 1).reshape(B, H*W, -1)
    elif bx1.ndim == 4:
        B, C, H, W = bx1.shape
        bx1 = bx1.permute(0, 2, 3, 1).reshape(B, H*W, C)
    elif bx1.ndim == 3:
        bx1 = bx1.unsqueeze(0)
        B, C, H, W = bx1.shape
        bx1 = bx1.permute(0, 2, 3, 1).reshape(B, H*W, C)
    loss = loss_fn(bx, bx1) * H * W
    return loss

def cal_wasserstein_loss(anchor_emb, positive_emb, matched=False, **kwargs):
    if matched:
        with torch.no_grad():
            # 计算成对平方距离 [B, B]
            diff = anchor_emb.unsqueeze(1) - positive_emb.unsqueeze(0)  # [B, B, D]
            pairwise_sq_dist = torch.sum(diff ** 2, dim=-1)

            # 可选：防止自匹配（如果 anchor[i] 和 positive[i] 是同一实例）
            # pairwise_sq_dist.fill_diagonal_(float('inf'))

            # 找到每个 anchor 最近的 positive 索引
            min_indices = torch.argmin(pairwise_sq_dist, dim=1)  # [B]

        # 使用索引提取匹配的 positive（此操作可导，因为索引是常量）
        matched_positive = positive_emb[min_indices] 
    else:
        matched_positive = positive_emb
        
    # wass_loss = sinkhorn_custom_loss(anchor_emb, matched_positive, **kwargs)
    wass_loss = sinkhorn_loss(anchor_emb, matched_positive, **kwargs)
    return wass_loss

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
    train_text_encoder = False
    train_vision_encoder = False
    use_image_feature = True
    use_text_feature = True
    caption_hidden_states_mode = 'cat' # only_text
    use_image_mask = True
    checkpoints = './outputs/image_hint_十二指肠3/otcfm_weights_step_claude.pt'
    train_wass_model = False
    discriminator = AttentiveFusionPatchDiscriminator(device="cuda:2")
    discriminator.device = "cuda:2"
    discriminator = discriminator.to("cuda:2")
    discriminator.load_state_dict(torch.load("./outputs/flow-match_vae_gan/discriminator.pt", weights_only=True))
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
        transforms.RandomHorizontalFlip(p=0.25),
        transforms.RandomVerticalFlip(p=0.25),
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
        "./data_tsy_12/train_json/data_pairs_flow_54/*.json")
        # "./data_tsy1/train_json/All_data_pairs_dia/*.json")
    
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
    wass_model.device = "cuda:2"
    model_size = sum(p.data.nelement() for p in wass_model.parameters())
    print(f"Wass Model params: {model_size / 1024 / 1024:.2f} M")
    # state_dict = torch.load("./best_matched_flow_weights/attention_tiny_claude.pt")
    state_dict = torch.load("./best_matched_flow_weights/attention_tiny_claude.pt")
    # ############# Resnet34 #############
    # wass_model = TripletNetwork(model='resnet34').to("cuda:2").eval()
    # wass_model.device = "cuda:2"
    # model_size = sum(p.data.nelement() for p in wass_model.parameters())
    # print(f"Wass Model params: {model_size / 1024 / 1024:.2f} M")
    # state_dict = torch.load("./best_weights/simple_model_energy.pt")
    
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
    net_model, vae, text_model, vision_model = dispatch_model(net_model, vae, text_model, vision_model, num_device=2)

    # Ptach work for now. TODO: Remove the Global steps later
    global_step = start_step
    Wasserstein_loss = True
    Wasserstein_loss_multi_step = True
    wasserstein_loss_beta = .15
    wasserstein_loss_beta_neighbor = 0.1
    
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
            if random.random() < 0.05:
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
            
            if 'mask_hint' in batch and random.random() < 0.75:
                mask_hint = batch['mask_hint']
            else:
                mask_hint = torch.zeros_like(hint)

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
                    num_points = random.choice([8, 12, 16, 20, 24, 28])
                    # num_points = 5

                    # 生成 [0, 1] 范围内的随机时间点，保持递增
                    random_t = torch.rand(num_points, device=wass_model.device)
                    t_span = torch.sort(random_t)[0]
                    t_span = torch.clamp(t_span / (t_span.max() + 0.05), 0.0, 1.0) # 归一化.
                    t_start = 0
                    
                    for t_idx in range(len(t_span)):
                        t = t_span[t_idx]
                        dt = t - t_start
                        t_start += t
                        x_last = x.clone()
                        if random.random() < 0.5:
                            x = x + vt * dt
                        else:
                            x = x + ut.to(wass_model.device) * dt
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
                    # fake_pred_for_g = discriminator(x0.to(discriminator.device) + vt.to(discriminator.device) * dt.to(discriminator.device))
                    # adv_loss = hinge_generator_loss(fake_pred_for_g)
                    # wass_loss = wass_loss + 0.05 * adv_loss.to(wass_loss.device)
            loss = loss + torch.clamp_min(wass_loss, - loss.detach().item() + 0.1)  # 原loss是直接优化 x+vt和x1 的MSE。
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