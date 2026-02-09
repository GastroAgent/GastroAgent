import gc
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
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
from tensordict import TensorDict
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from transformers import AutoFeatureExtractor
from torch.nn.utils import clip_grad_norm_
from random import choices, choice
import random
from typing import Dict, Optional, Tuple, Union
from diffusers.utils.torch_utils import randn_tensor

def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
    model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
    """
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor

class ViTEncoder(nn.Module):
    def __init__(self, local_ckpt: str = None, **kwargs):
        super().__init__()

        # 1) 加载配置 & 模型
        self.config = ViTConfig(hidden_size=768, num_hidden_layers=16, num_attention_heads=12, intermediate_size=3072,
                                hidden_act="gelu", hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
                                num_patches=1024, # Max Position Embedding.
                                initializer_range=0.02, layer_norm_eps=1e-10, image_size=(512, 512), patch_size=16,
                                num_channels=3,
                                qkv_bias=False, encoder_stride=16)
        image_size = (512, 512)
        patch_size = 16
        self.H = image_size[0] // patch_size
        self.W = image_size[1] // patch_size
        self.vit = ViTModel(self.config, add_pooling_layer=False)
        if os.path.exists(local_ckpt):
            try:
                model_weights = torch.load(local_ckpt)
                if 'model' in model_weights:
                    model_weights = model_weights['model']
                self.vit.load_state_dict(model_weights, strict=False)
            except:
                print('加载失败')

        # 2) embedding 维度
        self.embed_dim = 768

    def forward(self, x, bool_masked_pos=False, interpolate_pos_encoding=True):
        # x 期望是 shape=[B,3,H,W] 的 tensor，已经做过 Normalize
        if bool_masked_pos:
            bool_masked_pos = torch.zeros((x.shape[0], self.H * self.W, self.embed_dim))
        else:
            bool_masked_pos = None

        out = self.vit(x, interpolate_pos_encoding=interpolate_pos_encoding, bool_masked_pos=bool_masked_pos)  # BaseModelOutput
        return out.last_hidden_state

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean
from diffusers.utils.outputs import BaseOutput
from dataclasses import dataclass

@dataclass
class AutoencoderKLOutput(BaseOutput):
    """
    Output of AutoencoderKL encoding method.

    Args:
        latent_dist (`DiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
            `DiagonalGaussianDistribution` allows for sampling latents from the distribution.
    """

    latent_dist: "DiagonalGaussianDistribution"  # noqa: F821

@dataclass
class VAEOutput:
    recon: torch.Tensor
    cls: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    commit_loss: torch.Tensor

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
        try:
            vae = AutoencoderKL.from_pretrained(decoder_ckpt)
        except:
            save_config = json.load(open(os.path.join(decoder_ckpt, 'config.json'), 'r'))
            save_config.pop('_class_name')
            save_config.pop('_diffusers_version')
            vae = AutoencoderKL(**save_config)
        self.decoder = vae.decoder
        self.decoder_config = vae.config
        self.quant_conv = vae.quant_conv
        self.post_quant_conv = vae.post_quant_conv
        if use_VQVAE:
            self.quantize = VectorQuantizer(8192, latent_dim, beta=0.25, remap=None, sane_index_shape=False)
        else:
            self.quantize = QuantizerIdentity()

        if isinstance(self.encoder_config.image_size, tuple):
            self.latent_h = self.encoder_config.image_size[0] // self.encoder_config.patch_size
            self.latent_w = self.encoder_config.image_size[1] // self.encoder_config.patch_size
        elif isinstance(self.encoder_config.image_size, int):
            self.latent_h = self.latent_w = self.encoder_config.image_size // self.encoder_config.patch_size

        if os.path.exists(os.path.join(decoder_ckpt, 'pytorch_model.bin')):
            self.load_state_dict(torch.load(os.path.join(decoder_ckpt, 'pytorch_model.bin')))
            print('Load Weight: ', os.path.join(decoder_ckpt, 'pytorch_model.bin'))
        if os.path.exists(os.path.join(decoder_ckpt, 'pytorch_model.pth')):
            self.load_state_dict(torch.load(os.path.join(decoder_ckpt, 'pytorch_model.pth')))
            print('Load Weight: ', os.path.join(decoder_ckpt, 'pytorch_model.pth'))

    def reparameterize(self, mu, logvar, resample=True):
        if not resample:
            return mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

    def forward(self, x, resample=True, return_dict=True):
        hidden_states = self.encoder(x)
        cls = hidden_states[:, 0, ...]
        latent = hidden_states[:, 1:, ...]
        assert latent.shape[1] == self.latent_h * self.latent_w, '图像尺寸不正确。'
        latent = latent.view(latent.shape[0], self.latent_h, self.latent_w, latent.shape[-1])
        latent = self.fc_mu_logvar(latent).permute(0, 3, 1, 2)
        latent = self.quant_conv(latent)
        mu, logvar = latent.chunk(2, dim=1)
        recon_z = self.reparameterize(mu, logvar, resample=resample) * self.scaler.exp()
        recon_z, commit_loss, _ = self.quantize(recon_z)
        recon_z = self.post_quant_conv(recon_z)
        recon = self.decoder(recon_z)
        if not return_dict:
            return (recon, TensorDict({'mu': mu, 'logvar': logvar, 'commit_loss': commit_loss, 'cls': cls}))
            
        return VAEOutput(recon=recon,
                         cls=cls,
                         mu=mu,
                         logvar=logvar,
                         commit_loss=commit_loss)

    def get_mu_logvar(self, x, resample=True):
        hidden_states = self.encoder(x)
        cls = hidden_states[:, 0, ...]
        latent = hidden_states[:, 1:, ...]
        assert latent.shape[1] == self.latent_h * self.latent_w, '图像尺寸不正确。'
        latent = latent.view(latent.shape[0], self.latent_h, self.latent_w, latent.shape[-1])
        latent = self.fc_mu_logvar(latent).permute(0, 3, 1, 2)
        latent = self.quant_conv(latent)
        mu, logvar = latent.chunk(2, dim=1)
        return mu, logvar
    
    def encode(self, x, return_dict=True):
        hidden_states = self.encoder(x)
        cls = hidden_states[:, 0, ...]
        latent = hidden_states[:, 1:, ...]
        assert latent.shape[1] == self.latent_h * self.latent_w, '图像尺寸不正确。'
        latent = latent.view(latent.shape[0], self.latent_h, self.latent_w, latent.shape[-1])
        latent = self.fc_mu_logvar(latent).permute(0, 3, 1, 2)
        latent = self.quant_conv(latent)
        posterior = DiagonalGaussianDistribution(latent)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, x):
        if self.post_quant_conv is not None:
            x = self.post_quant_conv(x)
        recon = self.decoder(x)
        return recon

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
