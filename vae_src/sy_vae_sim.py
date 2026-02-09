import gc
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from timm.models.vision_transformer import VisionTransformer
from functools import partial
from torch import nn
from torch.nn import Identity
from safetensors.torch import load_file
from diffusers.models import AutoencoderKL
from transformers import ViTModel, ViTConfig, AutoTokenizer, AutoModel
# from transformers import CLIPModel, CLIPProcessor
from transformers import ChineseCLIPProcessor as CLIPProcessor
from transformers import ChineseCLIPModel as CLIPModel
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers.models.autoencoders.vae import VectorQuantizer
import torch.nn.functional as F
from dataclasses import dataclass
import math
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
import os
import torch
from torchvision import transforms
from transformers import AutoFeatureExtractor
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from random import choices, choice
from utils_ import _get_vector_norm

# Step 1: 自定义 AddGaussianNoise 类
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): (C, H, W) 的张量
        Returns:
            Tensor: 添加高斯噪声后的图像
        """
        noise = torch.randn(tensor.shape) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

import torch.nn.functional as F

def extract_latent(model, img_tensor, device):
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encoder(img_tensor.unsqueeze(0).to(device))
        return mu.squeeze(0).cpu()

def image_similarity(model, img_tensor1, img_tensor2, device):
    z1 = extract_latent(model, img_tensor1, device)
    z2 = extract_latent(model, img_tensor2, device)
    euclidean = torch.norm(z1 - z2, p=2).item()
    cosine = F.cosine_similarity(z1.unsqueeze(0), z2.unsqueeze(0)).item()
    return euclidean, cosine

def kl_divergence_gaussian(mu_p, sigma_p, mu_q, sigma_q):
    """
    计算两个高斯分布之间的 KL 散度 D_KL(P || Q)

    参数:
        mu_p: float or np.ndarray, 分布 P 的均值
        sigma_p: float or np.ndarray, 分布 P 的标准差
        mu_q: float or np.ndarray, 分布 Q 的均值
        sigma_q: float or np.ndarray, 分布 Q 的标准差

    返回:
        kl_div: float or np.ndarray, KL 散度值
    """
    # 添加小常数以避免除零错误
    epsilon = 1e-8
    sigma_p = np.clip(sigma_p, epsilon, None)
    sigma_q = np.clip(sigma_q, epsilon, None)

    kl_div = (np.log(sigma_q / sigma_p)
              + (sigma_p ** 2 + (mu_p - mu_q) ** 2) / (2 * sigma_q ** 2)
              - 0.5)
    return kl_div

def js_divergence_gaussian(mu_p, sigma_p, mu_q, sigma_q, epsilon=1e-8):
    """
    计算两个高斯分布之间的 JS 散度
    :param mu_p: 分布 P 的均值
    :param sigma_p: 分布 P 的标准差
    :param mu_q: 分布 Q 的均值
    :param sigma_q: 分布 Q 的标准差
    :param epsilon: 防止除零的小常数
    :return: JS 散度
    """
    # 混合分布 M 的均值和方差
    mu_m = 0.5 * mu_p + 0.5 * mu_q
    sigma_m_squared = 0.5 * (sigma_p**2 + sigma_q**2) + 0.25 * (mu_p - mu_q)**2
    sigma_m = np.sqrt(sigma_m_squared + epsilon)  # 防止除零

    # 单向 KL 散度 D_KL(P || M)
    kl_p_m = 0.5 * (np.log(sigma_m**2 / sigma_p**2) +
                    (sigma_p**2 + (mu_p - mu_m)**2) / sigma_m**2 - 1)

    # 单向 KL 散度 D_KL(Q || M)
    kl_q_m = 0.5 * (np.log(sigma_m**2 / sigma_q**2) +
                    (sigma_q**2 + (mu_q - mu_m)**2) / sigma_m**2 - 1)

    # JS 散度
    js_div = 0.5 * (kl_p_m + kl_q_m)
    return js_div

def symmetric_kl(mu_p, sigma_p, mu_q, sigma_q):
    kl_pq = kl_divergence_gaussian(mu_p, sigma_p, mu_q, sigma_q)
    kl_qp = kl_divergence_gaussian(mu_q, sigma_q, mu_p, sigma_p)
    return (kl_pq + kl_qp) / 2

def kl_divergence_vae(mu_p, logvar_p, mu_q, logvar_q):
    """
    计算两个高斯分布之间的 KL 散度（假设协方差为对角矩阵）
    :param mu_p: 编码器输出的均值 (28x28)
    :param logvar_p: 编码器输出的 log(方差) (28x28)
    :param mu_q: 目标分布的均值 (28x28)
    :param logvar_q: 目标分布的 log(方差) (28x28)
    :return: 总 KL 散度
    """
    sigma_p = np.exp(0.5 * logvar_p)
    sigma_q = np.exp(0.5 * logvar_q)

    # 逐点计算 KL 散度
    kl_pointwise = (logvar_q - logvar_p
                    - 1 + (sigma_p ** 2 + (mu_p - mu_q) ** 2) / sigma_q ** 2)
    kl_pointwise /= 2  # 因为 logvar_q - logvar_p = log(sigma_q^2) - log(sigma_p^2)

    # 总 KL 散度（求和）或均值
    total_kl = np.sum(kl_pointwise)
    mean_kl = np.mean(kl_pointwise)
    return total_kl, mean_kl

def symmetric_kl_vae(mu_p, sigma_p, mu_q, sigma_q, epsilon=1e-8):
    """
    计算双向 KL 散度（Symmetric KL Divergence）的总和或均值
    :param mu_p: 28x28 的均值矩阵（分布 P）
    :param sigma_p: 28x28 的标准差矩阵（分布 P）
    :param mu_q: 28x28 的均值矩阵（分布 Q）
    :param sigma_q: 28x28 的标准差矩阵（分布 Q）
    :param epsilon: 防止除零的小常数
    :return: 总双向 KL 散度或均值
    """
    # 防止除零
    sigma_p = np.clip(sigma_p, epsilon, None)
    sigma_q = np.clip(sigma_q, epsilon, None)

    # 单向 KL 散度 P || Q
    kl_pq = (np.log(sigma_q / sigma_p)
             + (sigma_p ** 2 + (mu_p - mu_q) ** 2) / (2 * sigma_q ** 2)
             - 0.5)

    # 单向 KL 散度 Q || P
    kl_qp = (np.log(sigma_p / sigma_q)
             + (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2 * sigma_p ** 2)
             - 0.5)

    # 双向 KL 散度
    symmetric_kl = (kl_pq + kl_qp) / 2

    # 返回总 KL 散度（求和）或均值
    total_kl = np.sum(symmetric_kl)
    mean_kl = np.mean(symmetric_kl)
    return total_kl, mean_kl

def js_kl_vae(mu_p, sigma_p, mu_q, sigma_q, epsilon=1e-8):
    """
    计算两个高斯分布之间的 JS 散度（适用于 28x28 的均值和方差矩阵）
    :param mu_p: 28x28 的均值矩阵（分布 P）
    :param sigma_p: 28x28 的标准差矩阵（分布 P）
    :param mu_q: 28x28 的均值矩阵（分布 Q）
    :param sigma_q: 28x28 的标准差矩阵（分布 Q）
    :param epsilon: 防止除零的小常数
    :return: 总 JS 散度或均值 JS 散度
    """
    # 混合分布 M 的均值和方差
    mu_m = 0.5 * mu_p + 0.5 * mu_q
    sigma_p_squared = sigma_p ** 2
    sigma_q_squared = sigma_q ** 2
    sigma_m_squared = 0.5 * (sigma_p_squared + sigma_q_squared) + 0.25 * (mu_p - mu_q) ** 2
    sigma_m = np.sqrt(sigma_m_squared + epsilon)  # 防止除零

    # 单向 KL 散度 D_KL(P || M)
    kl_p_m = 0.5 * (
        np.log(sigma_m ** 2 / sigma_p_squared) +
        (sigma_p_squared + (mu_p - mu_m) ** 2) / sigma_m ** 2 -
        1
    )

    # 单向 KL 散度 D_KL(Q || M)
    kl_q_m = 0.5 * (
        np.log(sigma_m ** 2 / sigma_q_squared) +
        (sigma_q_squared + (mu_q - mu_m) ** 2) / sigma_m ** 2 -
        1
    )

    # JS 散度
    js_div = 0.5 * (kl_p_m + kl_q_m)

    # 返回总 JS 散度或均值
    total_js = np.sum(js_div)
    mean_js = np.mean(js_div)
    return total_js, mean_js

def cal_kl_metric(vae: nn.Module, img1: torch.Tensor, img2: torch.Tensor, show=True):
    result = {}
    vae.eval()
    images = torch.stack([img1, img2], dim=0)
    hidden_states = vae.encoder(images.to(device))
    cls = hidden_states[:, 0, ...]
    latent = hidden_states[:, 1:, ...]
    latent = latent.view(latent.shape[0], vae.latent_h, vae.latent_h, latent.shape[-1])
    latent = vae.fc_mu_logvar(latent).permute(0, 3, 1, 2)
    latent = vae.quant_conv(latent)
    mu, logvar = latent.chunk(2, dim=1)
    mu = mu.cpu().detach().numpy()
    std = torch.exp(logvar / 2).cpu().detach().numpy()
    kl_total, kl_maen = kl_divergence_vae(mu[0], std[0], mu[1], std[1])
    sym_kl_total, sym_kl_mean = symmetric_kl_vae(mu[0], std[0], mu[1], std[1])
    js_kl1_total, js_kl1_mean = js_kl_vae(mu[0], std[0], mu[1], std[1])
    mse = ((mu[0] - mu[1]) ** 2).sum() ** 0.5

    print('-' * 50)
    print('mu and std shape: ',mu.shape, logvar.shape)
    print(f"KL散度值: {kl_total:.4f} {kl_maen:.4f}")
    print(f"双向KL散度值: {sym_kl_total:.4f} {sym_kl_mean:.4f}")
    print(f"JS KL1: {js_kl1_total:.4f} {js_kl1_mean:.4f}")
    print("MSE(Mean): ", mse)
    print('=' * 50)

    result['mse'] = mse
    result['kl_total'] = kl_total
    result['kl_maen'] = kl_maen
    result['sym_kl_total'] = sym_kl_total
    result['sym_kl_mean'] = sym_kl_mean
    result['js_kl_total'] = js_kl1_total
    result['js_kl_mean'] = js_kl1_mean
    return result

class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_paths = []
        for root, dirs, files in os.walk(root_dir):
            for fname in files:
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.img_paths.append(os.path.join(root, fname))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

class MedicalCLIPDataset(Dataset):
    def __init__(self, data_path, transform=None, base_transform=None, text_key = 'text', image_key = 'image'):
        self.dataset = json.load(open(data_path, 'r'))
        self.transform = transform
        self.base_transform = base_transform
        self.text_key = text_key
        self.image_key = image_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_file = self.dataset[idx][self.image_key]
        image = Image.open(img_file).convert('RGB')
        if self.transform:
            img = self.transform(image)
        if self.base_transform:
            base_img = self.base_transform(image)
        else:
            base_img = img
        try:
            text = self.dataset[idx][self.text_key]
        except:
            print(self.dataset[idx])
            raise KeyError
        batch = {
            'image': img,
            'text': text,
            'image_path': img_file,
            'base_image': base_img
        }
        return batch


class ViTEncoder(nn.Module):
    def __init__(self, local_ckpt: str = None, **kwargs):
        super().__init__()

        # 1) 加载配置 & 模型
        self.config = ViTConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072,
                                hidden_act="gelu", hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
                                initializer_range=0.02, layer_norm_eps=1e-10, image_size=(512, 496), patch_size=16,
                                num_channels=3,
                                qkv_bias=True, encoder_stride=16)
        self.H = 512 // 16
        self.W = 496 // 16
        self.vit = ViTModel(self.config, add_pooling_layer=False)
        if os.path.exists(local_ckpt):
            try:
                model_weights = torch.load(local_ckpt)['model']
                self.vit.load_state_dict(model_weights, strict=False)
            except:
                print('加载失败')

        # 2) embedding 维度
        self.embed_dim = 768

    def forward(self, x, bool_masked_pos=False):
        # x 期望是 shape=[B,3,H,W] 的 tensor，已经做过 Normalize
        if bool_masked_pos:
            bool_masked_pos = torch.zeros((x.shape[0], self.H * self.W, self.embed_dim))
        else:
            bool_masked_pos = None

        out = self.vit(x, interpolate_pos_encoding=True, bool_masked_pos=bool_masked_pos)  # BaseModelOutput
        return out.last_hidden_state


class Decoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 224*224*3),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.fc(z).view(-1, 3, 224, 224)

class QuantizerIdentity(nn.Module):
    def __init__(self):
        super(QuantizerIdentity, self).__init__()

    def forward(self, x):
        return x, 0, None # 保持 VectorQuantizer 的输出格式一致。

class UpsampleNet(nn.Module):
    def __init__(self, in_channels, out_channels=None, use_subpixel=False):
        """
        上采样网络模块

        参数:
        - in_channels: 输入通道数
        - out_channels: 输出通道数 (默认与输入相同)
        - use_subpixel: 是否使用亚像素卷积 (更高效)
        """
        super(UpsampleNet, self).__init__()
        self.out_channels = out_channels or in_channels
        self.use_subpixel = use_subpixel

        if use_subpixel:
            # 亚像素卷积方案 (更高效)
            self.upsample_block = nn.Sequential(
                nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor=2),  # 2倍上采样
                nn.ReLU(inplace=True),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
            )
        else:
            # 双线性上采样方案 (更简单)
            self.upsample_block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
            )

    def forward(self, x):
        """
        前向传播

        输入: (B, C, 256, 256)
        输出: (B, C, 512, 512)
        """
        return self.upsample_block(x)

class VAE(nn.Module):
    def __init__(self, latent_dim: int = 4, encoder_ckpt: str = None,
                 decoder_ckpt: str = None, use_VQVAE: bool = True, **kwargs):
        super().__init__()
        if encoder_ckpt is None or not os.path.exists(encoder_ckpt):
            print(f"请传入正确的 Encoder checkpoint 路径: {encoder_ckpt}")
        if decoder_ckpt is None or not os.path.isdir(decoder_ckpt):
            raise RuntimeError(f"请传入正确的 Decoder Dir 路径: {decoder_ckpt}")

        self.encoder = ViTEncoder(local_ckpt=encoder_ckpt)
        self.encoder_config = self.encoder.config
        self.fc_mu_logvar = nn.Linear(self.encoder.embed_dim, latent_dim * 2)
        self.scaler = torch.nn.Parameter(torch.Tensor([0])) # 重参数化时的缩放。
        self.logit_scaler = torch.nn.Parameter(torch.Tensor([1]))
        self.perceived_logit_scaler = torch.nn.Parameter(torch.Tensor([1])) # 图像与图像之间的。
        vae = AutoencoderKL.from_pretrained(decoder_ckpt)
        self.decoder = vae.decoder
        self.decoder_config = vae.config
        self.quant_conv = vae.quant_conv
        self.post_quant_conv = vae.post_quant_conv
        if use_VQVAE:
            self.quantize = VectorQuantizer(512, 4, beta=0.25, remap=None, sane_index_shape=False)
        else:
            self.quantize = QuantizerIdentity()
        self.upsampler = UpsampleNet(in_channels=3, use_subpixel=False)

        self.latent_h = self.latent_w = self.encoder_config.image_size // self.encoder_config.patch_size

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        hidden_states = self.encoder(x)
        cls = hidden_states[:, 0, ...]
        latent = hidden_states[:, 1:, ...]
        assert latent.shape[1] == self.latent_h * self.latent_w, '图像尺寸不正确。'
        latent = latent.view(latent.shape[0], self.latent_h, self.latent_h , latent.shape[-1])
        latent = self.fc_mu_logvar(latent).permute(0, 3, 1, 2)
        latent = self.quant_conv(latent)
        mu, logvar = latent.chunk(2, dim=1)
        recon_z = self.reparameterize(mu, logvar) * self.scaler.exp()
        recon_z, commit_loss, _ = self.quantize(recon_z)
        recon_z = self.post_quant_conv(recon_z)
        recon = self.decoder(recon_z)
        recon = self.upsampler(recon)
        return VAEOutput(recon=recon,
                         cls=cls,
                         mu=mu,
                         logvar=logvar,
                         commit_loss=commit_loss)

@dataclass
class VAEOutput:
    recon: torch.Tensor
    cls: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    commit_loss: torch.Tensor

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon = F.mse_loss(recon_x, x, reduction='mean')
    logvar = torch.clamp(logvar, -10, 10)
    kl = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    return recon + beta * kl, kl.detach()

def kl_div_1d(mu1, sigma1, mu2, sigma2):
    return (
        math.log(sigma2 / sigma1)
        + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2)
        - 0.5
    )

def train_vae(model, loader, device,
              epochs=30, lr=1e-4, beta=1.0, commit_beta = 1.0,
              save_dir='./checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-10)
    model.train()

    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        # 在 tqdm 中显示 epoch 进度
        with (tqdm(loader, desc=f"[Epoch {epoch}/{epochs}]", unit='batch') as pbar):
            for batch in pbar:
                try:
                    imgs = batch['image'].to(device)
                    base_image = batch['base_image'].to(device) # 生成的Label。
                except:
                    imgs = batch.to(device)
                    base_image = imgs
                output = model(imgs)
                loss, kl = vae_loss(output.recon, base_image, output.mu, output.logvar, beta)
                loss = loss + commit_beta * output.commit_loss

                optimizer.zero_grad()
                loss.backward()
                norm = clip_grad_norm_(model.parameters(), max_norm=1.0).item()
                optimizer.step()
                scheduler.step()

                batch_size = imgs.size(0)
                epoch_loss += loss.item() * batch_size

                # 把当前 batch 的 loss 打印到进度条上
                pbar.set_postfix({
                    'batch_loss': f"{loss.item():.4f}",
                    "kl": f"{kl:.4f}",
                    "commit_loss": f"{output.commit_loss:.4f}",
                    "norm": f"{norm:.4f}",
                })

        avg_loss = epoch_loss / len(loader) / batch_size
        print(f"Epoch {epoch}/{epochs} — Avg Loss: {avg_loss:.4f}")

        # 保存权重
        save_path = os.path.join(save_dir, f"vae_epoch{epoch}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"  ↳ saved: {save_path}")

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

def train_clip_vae(model, loader, device, clip_path = '/home/dalhxwlyjsuo/criait_tansy/weight/clip-vit-large-patch14',
              epochs=30, lr=1e-4, beta=0.1, commit_beta = 0.0, train_clip = False, perceived_beta = 0.0, freeze_clip_text_projection = False,
                   freeze_clip_vision_model = False, freeze_clip_visual_projection = False, freeze_clip_text_model = False,
                   detach_clip_text_embeded = True, detach_clip_image_embeded = True,
                   clip_beta1 = 0.1, clip_cls = 0.05, save_dir='./checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    # 冻结除VAE Encoder和fc_mu, fc_std 外的所有层。
    for name, value in model.named_parameters():
        value.requires_grad = False
        if 'encoder' in name or 'fc_mu_logvar' in name:
            value.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-10)
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(clip_path)
    processor = CLIPProcessor.from_pretrained(clip_path, use_fast=True)
    if torch.cuda.device_count() > 2:
        device_map = json.load(open('/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/device_map.json', 'r'))
        device_text = f'cuda:{device_map["text_model"]}'
        device_vision = f'cuda:{device_map["vision_model.embeddings"]}'
        clip = AutoModel.from_pretrained(clip_path, device_map=device_map,
                                         max_memory={
                                             1: "20GiB",
                                             2: "20GiB"
                                         })
        # with open('/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/device_map.json', 'w') as f:
        #     json.dump(clip.hf_device_map, f, indent=2)
    else:
        clip = AutoModel.from_pretrained(clip_path, device_map='auto',
                                     max_memory = {
                                         1: "20GiB",
                                         # 2: "10GiB"
        })
    print(clip.hf_device_map)

    # raise NotImplementedError
    if train_clip:
        if freeze_clip_vision_model:
            clip.vision_model.requires_grad_(False)
        if freeze_clip_visual_projection:
            clip.visual_projection.requires_grad_(False)
        if freeze_clip_text_model:
            clip.text_model.requires_grad_(False)
        if freeze_clip_text_projection:
            clip.text_projection.requires_grad_(False)
        optimizer_clip = torch.optim.AdamW(clip.parameters(), lr=lr * 0.5, weight_decay=1e-3)
        scheduler_clip = CosineAnnealingLR(optimizer_clip, T_max=1000, eta_min=1e-10)
    else:
        clip.requires_grad_(False)

    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        # 在 tqdm 中显示 epoch 进度
        with (tqdm(loader, desc=f"[Epoch {epoch}/{epochs}]", unit='batch') as pbar):
            for batch in pbar:
                imgs = batch['image'].to(device)
                texts = batch['text']
                base_image = batch['base_image'].to(device)
                images = [Image.open(x).convert('RGB') for x in batch['image_path']]

                # text_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(clip.device)
                # text_embeds = clip.get_text_features(**text_inputs)
                # image_inputs = processor(images=images, return_tensors="pt").to(clip.device)
                # image_embeds = clip.get_image_features(**image_inputs)

                output = model(imgs)
                cls_embed = output.cls
                loss, kl = vae_loss(output.recon, base_image, output.mu, output.logvar, beta)
                loss = loss + commit_beta * output.commit_loss
                recon_loss = loss.detach().cpu()
                # cosine similarity1

                assert len(texts) == len(images)
                if train_clip:
                    if torch.cuda.device_count() > 2:
                        text_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device_text)
                        image_inputs = processor(images=images, return_tensors="pt").to(device_vision)
                        vision_outputs = clip.vision_model(**image_inputs)
                        text_outputs = clip.text_model(**text_inputs)
                        image_embeds = vision_outputs[1].to(device_text)
                        image_embeds = clip.visual_projection(image_embeds)

                        text_embeds = text_outputs[0][:, 0, :].to(device_text)
                        text_embeds = clip.text_projection(text_embeds)

                        # torch.cuda.empty_cache()
                        # gc.collect()

                        # normalized features
                        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                        # cosine similarity as logits
                        logit_scale = clip.logit_scale.exp()
                        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
                        clip_loss1 = clip_loss(logits_per_text).to(loss.device)
                    else:
                        inputs = processor(text=texts, images=images,
                                           return_tensors="pt", padding=True).to(clip.device)
                        outputs = clip(**inputs, return_loss=True)
                        clip_loss1 = outputs['loss'].to(loss.device)
                        image_embeds = outputs['image_embeds']
                        text_embeds = outputs['text_embeds']
                    # clip_loss1.backward(retain_graph=True)
                #     logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device))
                #     logits_per_text = logits_per_text * clip.logit_scale.exp().to(text_embeds.device)
                #     clip_loss1 = clip_loss(logits_per_text).to(loss.device)
                else:
                    with torch.no_grad():
                        outputs = clip(**inputs, return_loss=False)
                        clip_loss1 = 0
                        image_embeds = outputs['image_embeds']
                        text_embeds = outputs['text_embeds']

                # 计算对比损失。
                # normalized features
                # image_embeds = image_embeds / _get_vector_norm(image_embeds)
                # text_embeds = text_embeds / _get_vector_norm(text_embeds)
                cls_embed = cls_embed / _get_vector_norm(cls_embed)

                perceived_loss = 0
                if perceived_beta > 0:
                    base_features = (image_embeds.detach() if detach_clip_image_embeded else image_embeds).to(device)
                    image_inputs = processor(images=images, return_tensors="pt").to(clip.device)
                    # print('output.recon shape: ', output.recon.shape)
                    # print('base image shape: ',image_inputs['pixel_values'].shape)
                    del image_inputs['pixel_values']
                    torch.cuda.empty_cache()
                    recon_image_inputs = image_inputs
                    recon_image_inputs['pixel_values'] = output.recon.to(clip.device)
                    recon_features = clip.get_image_features(**recon_image_inputs).to(device)
                    recon_features = recon_features / _get_vector_norm(recon_features)
                    logits_per_text = torch.matmul(base_features, recon_features.t())
                    logits_per_text = logits_per_text * model.perceived_logit_scaler.exp().to(device)
                    perceived_loss = clip_loss(logits_per_text).to(loss.device)
                    loss = loss + perceived_beta * perceived_loss

                # cosine similarity2
                logits_per_text = torch.matmul((text_embeds.detach() if detach_clip_text_embeded else text_embeds).to(cls_embed.device), cls_embed.t())
                logits_per_text = logits_per_text * model.logit_scaler.exp().to(cls_embed.device)
                clip_loss2 = clip_loss(logits_per_text)

                # if cal_img_similarity:
                #     logits_per_text = torch.matmul(image_embeds.detach().to(cls_embed), cls_embed.t())
                #     logits_per_text = logits_per_text * model.logit_scale.exp().to(cls_embed.device)
                #     clip_loss3 = clip_loss(logits_per_text)

                #     loss = loss + clip_loss2 * clip_cls + clip_loss1 * clip_beta1 + clip_cls * clip_loss3
                # else:
                #     loss = loss + clip_loss2 * clip_cls + clip_loss1 * clip_beta1
                loss = loss + clip_loss2 * clip_cls + clip_loss1 * clip_beta1

                if train_clip:
                    optimizer_clip.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                norm = clip_grad_norm_(model.parameters(), max_norm=1.0).item()
                if train_clip:
                    clip_norm = clip_grad_norm_(clip.parameters(), max_norm=1.0).item()
                    optimizer_clip.step()
                    scheduler_clip.step()
                else:
                    clip_norm = 0
                optimizer.step()
                scheduler.step()
                batch_size = imgs.size(0)
                epoch_loss += loss.item() * batch_size

                # 把当前 batch 的 loss 打印到进度条上
                pbar.set_postfix({
                    'batch_loss': f"{loss.item():.4f}",
                    "recon_loss": f"{recon_loss.item():.4f}",
                    "kl": f"{kl:.4f}",
                    "commit_loss": f"{output.commit_loss:.4f}",
                    "clip_loss": f"{clip_loss1:.4f}",
                    "clip_cls_loss": f"{clip_loss2:.4f}",
                    "perceived_loss": f"{perceived_loss:.4f}",
                    "norm": f"{norm:.4f}",
                    "clip_norm": f"{clip_norm:.4f}"
                })

        avg_loss = epoch_loss / len(loader) / batch_size
        print(f"Epoch {epoch}/{epochs} — Avg Loss: {avg_loss:.4f}")

        # 保存权重
        save_path = os.path.join(save_dir, f"vae_epoch{epoch}.pth")
        torch.save(model.state_dict(), save_path)
        if epoch % 1 == 0:
            clip.save_pretrained(os.path.join(save_dir, f'CLIPModel_{epoch}'))
            processor.save_pretrained(os.path.join(save_dir, f'CLIPModel_{epoch}'))
        print(f"  ↳ saved: {save_path}")


def train_clip(loader, device, clip_path = '/home/dalhxwlyjsuo/criait_tansy/weight/clip-vit-large-patch14',
              epochs=30, lr=1e-5, save_dir='./checkpoints'):
    os.makedirs(save_dir, exist_ok=True)

    processor = CLIPProcessor.from_pretrained(clip_path, use_fast=True)
    clip = CLIPModel.from_pretrained(clip_path, device_map = 'auto').train()
    optimizer_clip = torch.optim.AdamW(clip.parameters(), lr=lr, weight_decay=1e-3)
    scheduler_clip = CosineAnnealingLR(optimizer_clip, T_max=1000, eta_min=1e-10)

    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        # 在 tqdm 中显示 epoch 进度
        with (tqdm(loader, desc=f"[Epoch {epoch}/{epochs}]", unit='batch') as pbar):
            for batch in pbar:
                texts = batch['text']
                images = [Image.open(x).convert('RGB') for x in batch['image_path']]
                assert len(texts) == len(images)
                inputs = processor(text=texts, images=images,
                                        return_tensors="pt", padding=True).to(device)

                outputs = clip(**inputs, return_loss=True)
                loss = outputs['loss']

                optimizer_clip.zero_grad()
                loss.backward()
                norm = clip_grad_norm_(clip.parameters(), max_norm=1.0).item()
                optimizer_clip.step()
                scheduler_clip.step()
                batch_size = len(images)
                epoch_loss += loss.item() * batch_size

                # 把当前 batch 的 loss 打印到进度条上
                pbar.set_postfix({
                    'batch_loss': f"{loss.item():.4f}",
                    "norm": f"{norm:.4f}",
                })

        avg_loss = epoch_loss / len(loader) / batch_size
        print(f"Epoch {epoch}/{epochs} — Avg Loss: {avg_loss:.4f}")

        if epoch % 1 == 0:
            clip.save_pretrained(os.path.join(save_dir, f'CLIPModel_{epoch}'))
            processor.save_pretrained(os.path.join(save_dir, f'CLIPModel_{epoch}'))
            print(f"  ↳ saved: {save_dir}")

def numpy_topk_simple(arr, k, axis=-1, largest=True):
    if not largest:
        arr = -arr
    # print(arr.shape)
    indices = np.argsort(arr, axis=axis)[:, -k:]
    # print(indices.shape)

    # [:, -k:] # 全排序后取前k个
    if largest:
        indices = np.flip(indices, axis=axis)  # 降序排列
    return indices

def eval_vae_similatiry(vae, images_root, device, transforms, k = 3):
    vae.eval()
    accuracy = {}
    labels = os.listdir(images_root)
    for label_id, label in enumerate(labels):
        src_path = os.path.join(images_root, label)
        cls_correct = 0
        topk_correct = 0
        kl_correct = 0
        kl_topk_correct = 0
        js_correct = 0
        js_topk_correct = 0
        total = 0
        for _ in tqdm(range(10)):
            # print(f"{label_id}/{len(labels)}---{_}/20")
            src_imgs = choices(os.listdir(src_path), k=10)
            src_imgs = [os.path.join(src_path, x) for x in src_imgs]

            total += len(src_imgs)
            other_imgs = []
            for other in labels:
                image_dir = os.path.join(images_root, other)
                file_name = choice(os.listdir(image_dir))
                other_imgs.append(os.path.join(image_dir, file_name))

            other_imgs = [transforms(Image.open(x).convert('RGB')) for x in other_imgs]
            src_imgs = [transforms(Image.open(x).convert('RGB')) for x in src_imgs]
            images = torch.stack(other_imgs)
            # print('image shape: ', images.shape)
            with torch.no_grad():
                hidden_states = vae.encoder(images.to(device))
            cls = hidden_states[:, 0, ...]
            latent = hidden_states[:, 1:, ...]

            src_imgs = torch.stack(src_imgs)
            # print('src shape: ', src_imgs.shape)
            with torch.no_grad():
                hidden_states = vae.encoder(src_imgs.to(device))
            src_cls = hidden_states[:, 0, ...]
            src_latent = hidden_states[:, 1:, ...]
            # 计算L2范数
            cls = cls.cpu().numpy()
            src_cls = src_cls.cpu().numpy()

            norms = np.linalg.norm(cls, axis=1, keepdims=True)
            eps = 1e-8
            cls = cls / (norms + eps)
            norms = np.linalg.norm(src_cls, axis=1, keepdims=True)
            src_cls = src_cls / (norms + eps) # [10, ]
            cos_score = src_cls @ cls.T # [10, classes]

            pred = np.argmax(cos_score, axis=1) # [10]
            result = (pred == label_id)
            cls_correct += result.sum()

            indices = numpy_topk_simple(cos_score, k, 1, True) # [10, 3]
            for x in range(1, k):
                result = result | (indices[:, x] == label_id)
            topk_correct += result.sum()

            # KL Accuracy.
            latent = latent.view(latent.shape[0], vae.latent_h, vae.latent_h, latent.shape[-1])
            latent = vae.fc_mu_logvar(latent).permute(0, 3, 1, 2)
            latent = vae.quant_conv(latent)
            mu, logvar = latent.chunk(2, dim=1)
            mu = mu.cpu().detach().numpy()
            std = torch.exp(logvar / 2).cpu().detach().numpy()
            # print('mu', mu.shape)

            src_latent = src_latent.view(src_latent.shape[0], vae.latent_h, vae.latent_h, src_latent.shape[-1])
            src_latent = vae.fc_mu_logvar(src_latent).permute(0, 3, 1, 2)
            src_latent = vae.quant_conv(src_latent)
            src_mu, src_logvar = src_latent.chunk(2, dim=1)
            src_mu = src_mu.cpu().detach().numpy()
            src_std = torch.exp(src_logvar / 2).cpu().detach().numpy()
            # print('src_mu', src_mu.shape)
            # 计算双向KL [10, classes]
            kl_score = np.zeros([src_mu.shape[0], mu.shape[0]])
            for i in range(src_mu.shape[0]):
                for j in range(mu.shape[0]):
                    _, kl = symmetric_kl_vae(src_mu[i], src_std[i], mu[j], std[j])
                    kl_score[i, j] = - kl

            pred = np.argmax(kl_score, axis=1) # [10]
            result = (pred == label_id)
            kl_correct += result.sum()
            indices = numpy_topk_simple(kl_score, k, 1, True) # [10, 3]
            for x in range(1, k):
                result = result | (indices[:, x] == label_id)
            kl_topk_correct += result.sum()

            # 计算JS-KL。
            # [10, classes]
            kl_score = np.zeros([src_mu.shape[0], mu.shape[0]])
            for i in range(src_mu.shape[0]):
                for j in range(mu.shape[0]):
                    _, kl = js_kl_vae(src_mu[i], src_std[i], mu[j], std[j])
                    kl_score[i, j] = - kl

            pred = np.argmax(kl_score, axis=1) # [10]
            result = (pred == label_id)
            js_correct += result.sum()
            indices = numpy_topk_simple(kl_score, k, 1, True) # [10, 3]
            for x in range(1, k):
                result = result | (indices[:, x] == label_id)
            js_topk_correct += result.sum()

        accuracy[label + '_js'] = js_correct / total
        accuracy[label + f'_js_top{k}'] = js_topk_correct / total
        accuracy[label + '_kl'] = kl_correct / total
        accuracy[label + f'_kl_top{k}'] = kl_topk_correct / total
        accuracy[label] = cls_correct / total
        accuracy[label + f'_top{k}'] = topk_correct / total
    return accuracy

def eval_clip_similatiry(clip, images_root, device, processor, k = 3):
    clip.eval()
    accuracy = {}
    labels = os.listdir(images_root)
    for label_id, label in enumerate(labels):
        print('Label: ', label, f' {label_id + 1}/{len(labels)}')
        src_path = os.path.join(images_root, label)
        cls_correct = 0
        topk_correct = 0
        total = 0
        for _ in tqdm(range(5)):
            # print(f"{label_id}/{len(labels)}---{_}/20")
            src_imgs = choices(os.listdir(src_path), k=10)
            src_imgs = [os.path.join(src_path, x) for x in src_imgs]
            total += len(src_imgs)
            other_imgs = []
            for other in labels:
                image_dir = os.path.join(images_root, other)
                file_name = choice(os.listdir(image_dir))
                other_imgs.append(os.path.join(image_dir, file_name))

            images = [Image.open(x).convert('RGB') for x in other_imgs]
            src_imgs = [Image.open(x).convert('RGB') for x in src_imgs]
            with torch.no_grad():
                image_inputs = processor(images=images, return_tensors="pt").to(device)
                image_embeds = clip.get_image_features(**image_inputs)
                cls = image_embeds / _get_vector_norm(image_embeds)

                image_inputs = processor(images=src_imgs, return_tensors="pt").to(device)
                image_embeds = clip.get_image_features(**image_inputs)
                src_cls = image_embeds / _get_vector_norm(image_embeds)

            # 计算L2范数
            cls = cls.cpu().detach().numpy()
            src_cls = src_cls.cpu().detach().numpy()

            norms = np.linalg.norm(cls, axis=1, keepdims=True)
            eps = 1e-8
            cls = cls / (norms + eps)
            norms = np.linalg.norm(src_cls, axis=1, keepdims=True)
            src_cls = src_cls / (norms + eps) # [10, ]
            cos_score = src_cls @ cls.T # [10, classes]

            pred = np.argmax(cos_score, axis=1) # [10]
            result = (pred == label_id)
            cls_correct += result.sum()

            indices = numpy_topk_simple(cos_score, k, 1, True) # [10, 3]
            for x in range(1, k):
                result = result | (indices[:, x] == label_id)
            topk_correct += result.sum()
        accuracy[label] = cls_correct / total
        accuracy[label + f'_top{k}'] = topk_correct / total
    return accuracy

def eval_similatiry(vae, clip, images_root, device, processor, transforms, k = 3):
    clip.eval()
    vae.eval()
    accuracy = {}
    labels = os.listdir(images_root)
    for label_id, label in enumerate(labels):
        src_path = os.path.join(images_root, label)
        norm_topk_correct = 0
        topk_correct = 0
        cls_correct = 0
        total = 0
        for _ in tqdm(range(5)):
            # print(f"{label_id}/{len(labels)}---{_}/20")
            src_images = choices(os.listdir(src_path), k=10)
            src_images = [os.path.join(src_path, x) for x in src_images]
            total += len(src_images)
            other_imgs = []
            for other in labels:
                image_dir = os.path.join(images_root, other)
                file_name = choice(os.listdir(image_dir))
                other_imgs.append(os.path.join(image_dir, file_name))

            images = [Image.open(x).convert('RGB') for x in other_imgs]
            src_imgs = [Image.open(x).convert('RGB') for x in src_images]
            with torch.no_grad():
                image_inputs = processor(images=images, return_tensors="pt").to(device)
                image_embeds = clip.get_image_features(**image_inputs)
                cls = image_embeds / _get_vector_norm(image_embeds)

                image_inputs = processor(images=src_imgs, return_tensors="pt").to(device)
                image_embeds = clip.get_image_features(**image_inputs)
                src_cls = image_embeds / _get_vector_norm(image_embeds)

            # 计算L2范数
            cls = cls.cpu().detach().numpy()
            src_cls = src_cls.cpu().detach().numpy()

            norms = np.linalg.norm(cls, axis=1, keepdims=True)
            eps = 1e-8
            cls = cls / (norms + eps)
            norms = np.linalg.norm(src_cls, axis=1, keepdims=True)
            src_cls = src_cls / (norms + eps) # [10, ]
            clip_cos_score = src_cls @ cls.T # [10, classes]

            other_imgs = [transforms(Image.open(x).convert('RGB')) for x in other_imgs]
            src_imgs = [transforms(Image.open(x).convert('RGB')) for x in src_images]
            images = torch.stack(other_imgs)
            with torch.no_grad():
                hidden_states = vae.encoder(images.to(device))
            cls = hidden_states[:, 0, ...]
            src_imgs = torch.stack(src_imgs)
            with torch.no_grad():
                hidden_states = vae.encoder(src_imgs.to(device))
            src_cls = hidden_states[:, 0, ...]
            # 计算L2范数
            cls = cls.cpu().numpy()
            src_cls = src_cls.cpu().numpy()

            norms = np.linalg.norm(cls, axis=1, keepdims=True)
            eps = 1e-8
            cls = cls / (norms + eps)
            norms = np.linalg.norm(src_cls, axis=1, keepdims=True)
            src_cls = src_cls / (norms + eps)  # [10, ]
            vae_cos_score = src_cls @ cls.T  # [10, classes]
            
            normed_cos_score = vae_cos_score / (np.linalg.norm(vae_cos_score, axis=1, keepdims=True) + eps)  + clip_cos_score / (np.linalg.norm(clip_cos_score, axis=1, keepdims=True) + eps)
            cos_score = vae_cos_score + clip_cos_score * 5

            pred = np.argmax(cos_score, axis=1) # [10]
            result = (pred == label_id)
            cls_correct += result.sum()
            indices = numpy_topk_simple(cos_score, k, 1, True) # [10, 3]
            for x in range(1, k):
                result = result | (indices[:, x] == label_id)
            topk_correct += result.sum()

            pred = np.argmax(normed_cos_score, axis=1) # [10]
            result = (pred == label_id)
            indices = numpy_topk_simple(normed_cos_score, k, 1, True) # [10, 3]
            for x in range(1, k):
                result = result | (indices[:, x] == label_id)
            norm_topk_correct += result.sum()
        accuracy[label + f'_norm_top{k}'] = norm_topk_correct / total
        accuracy[label + f'_top{k}'] = topk_correct / total
        accuracy[label] = cls_correct / total
    return accuracy

if __name__ == '__main__':
    # 先根据 HF 预处理器拿到正确的 mean/std/size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feat = AutoFeatureExtractor.from_pretrained('/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/vit-base-patch16-224')
    base_transform = transforms.Compose([
        transforms.Resize((feat.size['height'], feat.size['width'])),
        transforms.ToTensor(),
        # AddGaussianNoise(mean=0., std=0.01),
        transforms.Normalize(mean=feat.image_mean, std=feat.image_std),
    ])
    transform = transforms.Compose([
        transforms.Resize((feat.size['height'], feat.size['width'])),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=feat.image_mean, std=feat.image_std),
        AddGaussianNoise(0, 0.2),
        transforms.RandomErasing(p=0.15, scale=(0.05, 0.2), ratio=(0.67, 1.33), value='random')
    ])

    # src_dir = '/home/dalhxwlyjsuo/criait_tansy/project/extend_data/cropped-2004-2010'
    # src_dir = '/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/our_data/new_cropped-2004-2010-endovit'
    # dataset = MedicalImageDataset(src_dir, transform=transform)
    # src_path = '/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/disease_data/train_clip_data2_with_diease_filtered.json'
    src_path = '/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/disease_data/train_clip_data2_with_disease_食管.json'
    # src_path = '/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/disease_data/食管/食管部位筛选/train_clip_data2_with_shiguan_diease_filtered_alexnet1.json'
    dataset = MedicalCLIPDataset(src_path, transform=transform, base_transform=base_transform, text_key='disease') # text
    loader  = DataLoader(dataset, batch_size=80, shuffle=True, num_workers=4) # 开启感知为 50，否则为 84.
    encoder_ckpt = '/home/dalhxwlyjsuo/criait_tansy/weight/EndoViT/pytorch_model.bin'
    decoder_ckpt = '/home/dalhxwlyjsuo/criait_tansy/weight/sd-vae-ft-mse'
    vae = VAE(latent_dim=4, encoder_ckpt=encoder_ckpt, decoder_ckpt=decoder_ckpt,
              use_VQVAE=False
    ).to(device)

    # state_dict = torch.load('/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/vqvae_weight/vae_epoch30.pth')
    state_dict = torch.load('/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/dia_clip_trained_vae_weight_625/vae_epoch3.pth')
    vae.load_state_dict(state_dict, strict=False)
    print(vae)

    # train_clip(loader, device, clip_path='/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/dia_clip_trained_weight/CLIPModel_2',
    #            epochs=20, lr=1e-5, save_dir='/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/dia_clip_trained_weight')

    train_clip_vae(vae, loader, device, clip_path='/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/dia_clip_trained_vae_weight_625/CLIPModel_3',
                   epochs=10, lr=1e-6, beta=.05, commit_beta=.0, train_clip=True, clip_beta1=0.1, clip_cls=.5, perceived_beta= 0.0,
                   freeze_clip_text_model=False, freeze_clip_text_projection=False, # 最好与detach_clip_image_embeded相反。
                   freeze_clip_visual_projection=False, freeze_clip_vision_model=False, # Stage2 是 False, Stage3 是 True。
                   detach_clip_text_embeded=True, detach_clip_image_embeded=True, # 不太能为False，因为重建图像的质量不稳定，且VAE的CLS也是不稳定的。
                   save_dir='/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/dia_clip_trained_vae_weight_625')

    # train_vae(vae, loader, device, epochs=30, lr=1e-4, beta=0.2, commit_beta = 0.5,
    #           save_dir='/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/vqvae_weight')


    # images_root = '/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/our_eval_data/Fake_DiaEvalImages'
    # # images_root = '/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/our_eval_data/食管/Fake_DiaEvalImages'
    # print('VAE Result: ')
    # vae = VAE(latent_dim=4, encoder_ckpt=encoder_ckpt, decoder_ckpt=decoder_ckpt,
    #           use_VQVAE=False
    # ).to(device)
    # clip_path = '/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/dia_clip_trained_vae_weight_623/CLIPModel_1'
    # state_dict = torch.load('/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/dia_clip_trained_vae_weight_stage2/vae_epoch10.pth')
    # vae.load_state_dict(state_dict, strict=False)
    #
    # print('VAE Result: ')
    # result = eval_vae_similatiry(vae, images_root, device, base_transform)
    # for k, v in result.items():
    #     print(k, v)
    # with open('/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/dia_eval_vae_result_stage2-full.json', 'w') as f:
    #     json.dump(result, f, indent=4, ensure_ascii=False)

    # print('CLIP Result: ')
    # processor = CLIPProcessor.from_pretrained(clip_path, use_fast=True)
    # clip = CLIPModel.from_pretrained(clip_path).to(device)
    # result = eval_clip_similatiry(clip, images_root, device, processor)
    # for k, v in result.items():
    #     print(k, v)
    # with open('/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/dia_eval_clip_result_622-食管.json', 'w') as f:
    #     json.dump(result, f, indent=4, ensure_ascii=False)
    # print('CLIP + VAE Result: ')
    # result = eval_similatiry(vae, clip, images_root, device, processor, base_transform)
    # for k, v in result.items():
    #     print(k, v)
    # with open('/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/dia_eval_clip_vae_result_622-食管.json', 'w') as f:
    #     json.dump(result, f, indent=4, ensure_ascii=False)



















    # # # 测试两张图片相似度。 1和2应该是更相近的。
    # img1 = base_transform(Image.open('/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/our_eval_data/new_cropped-2004-2010-endovit/01.2004101400008_1.jpg').convert('RGB'))
    # img2 = base_transform(Image.open('/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/our_eval_data/new_cropped-2004-2010-endovit/01.2004101400008_2.jpg').convert('RGB'))
    # img3 = base_transform(Image.open('/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/our_eval_data/new_cropped-2004-2010-endovit/01.2004101400022_2.jpg').convert('RGB'))
    # img4 = base_transform(Image.open('/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/our_eval_data/new_cropped-2004-2010-endovit/01.2004101400038_2.jpg').convert('RGB'))
    # vae.eval()
    # images = torch.stack([img1, img2, img3], dim=0)
    # hidden_states = vae.encoder(images.to(device))
    # cls = hidden_states[:, 0, ...]
    # latent = hidden_states[:, 1:, ...]
    # print(cls.shape)
    # print(latent.shape)
    # print('CLS Similarity.')
    #
    # cls = cls.cpu().detach().numpy()
    # cls_score = cls @ cls.T
    # print('cls dot score:\n',cls_score)
    # # 计算L2范数
    # norms = np.linalg.norm(cls, axis=1, keepdims=True)
    # # 防止除以零
    # eps = 1e-8
    # cls_normalized = cls / (norms + eps)
    # cos_score = cls_normalized @ cls_normalized.T
    # print('cls cos score:\n', cos_score)
    #
    # latent = latent.view(latent.shape[0], vae.latent_h, vae.latent_h, latent.shape[-1])
    # latent = vae.fc_mu_logvar(latent).permute(0, 3, 1, 2)
    # latent = vae.quant_conv(latent)
    # mu, logvar = latent.chunk(2, dim=1)
    # print(mu.shape, logvar.shape)
    # mu = mu.cpu().detach().numpy()
    # std = torch.exp(logvar/2).cpu().detach().numpy()
    #
    # # # 示例：计算两组图像的 KL 散度
    # # mu_p, sigma_p = 0.0, 1.0  # 图像 A 的潜在分布参数
    # # mu_q, sigma_q = 0.5, 1.5  # 图像 B 的潜在分布参数
    # # kl_value = kl_divergence_gaussian(mu_p, sigma_p, mu_q, sigma_q)
    # # print(f"Test KL散度值: {kl_value:.4f}")
    # # kl_value2 = kl_divergence_gaussian(mu_q, sigma_q, mu_p, sigma_p)
    # # print(f"Test 双向KL散度值: {kl_value/2 + kl_value2/2:.4f}")
    # # js_kl1 = js_divergence_gaussian(mu_p, sigma_p, mu_q, sigma_q)
    # # js_kl2 = js_divergence_gaussian(mu_q, sigma_q, mu_p, sigma_p)
    # # print(f"Test JS KL: {js_kl1:.4f} {js_kl2:.4f}")
    #
    # print('=' * 50)
    # print('Image1 和 Image2: ')
    # kl_total, kl_maen = kl_divergence_vae(mu[0], std[0], mu[1], std[1])
    # print(f"KL散度值: {kl_total:.4f} {kl_maen:.4f}")
    # sym_kl_total, sym_kl_mean = symmetric_kl_vae(mu[0], std[0], mu[1], std[1])
    # print(f"双向KL散度值: {sym_kl_total:.4f} {sym_kl_mean:.4f}")
    # js_kl1_total, js_kl1_mean = js_kl_vae(mu[0], std[0], mu[1], std[1])
    # print(f"JS KL1: {js_kl1_total:.4f} {js_kl1_mean:.4f}")
    # print('=' * 50)
    # print('=' * 50)
    # print('Image1 和 Image3: ')
    # kl_total, kl_maen = kl_divergence_vae(mu[0], std[0], mu[2], std[2])
    # print(f"KL散度值: {kl_total:.4f} {kl_maen:.4f}")
    # sym_kl_total, sym_kl_mean = symmetric_kl_vae(mu[0], std[0], mu[2], std[2])
    # print(f"双向KL散度值: {sym_kl_total:.4f} {sym_kl_mean:.4f}")
    # js_kl1_total, js_kl1_mean = js_kl_vae(mu[0], std[0], mu[2], std[2])
    # print(f"JS KL1: {js_kl1_total:.4f} {js_kl1_mean:.4f}")
    # print('=' * 50)
    # print('=' * 50)
    # print('Image2 和 Image3: ')
    # kl_total, kl_maen = kl_divergence_vae(mu[1], std[1], mu[2], std[2])
    # print(f"KL散度值: {kl_total:.4f} {kl_maen:.4f}")
    # sym_kl_total, sym_kl_mean = symmetric_kl_vae(mu[1], std[1], mu[2], std[2])
    # print(f"双向KL散度值: {sym_kl_total:.4f} {sym_kl_mean:.4f}")
    # js_kl1_total, js_kl1_mean = js_kl_vae(mu[1], std[1], mu[2], std[2])
    # print(f"JS KL1: {js_kl1_total:.4f} {js_kl1_mean:.4f}")
    # print('=' * 50)
    # print('=' * 50)
    # mse = ((mu[0] - mu[1]) ** 2).sum() ** 0.5
    # print("MSE(Mean) Image1和Image2: ", mse)
    # mse = ((mu[0] - mu[2]) ** 2).sum() ** 0.5
    # print("MSE(Mean) Image1和Image3: ", mse)
    # mse = ((mu[1] - mu[2]) ** 2).sum() ** 0.5
    # print("MSE(Mean) Image2和Image3: ", mse)
    # print('Image1 and Image4')
    # cal_kl_metric(vae, img1, img4)
    # print('Image3 and Image4')
    # cal_kl_metric(vae, img3, img4)
    #
