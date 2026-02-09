from timm.models.vision_transformer import VisionTransformer
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from functools import partial
from torch import nn
from safetensors.torch import load_file
from diffusers.models import AutoencoderKL
from transformers import ViTModel, ViTConfig
from diffusers.models.autoencoders.vae import VectorQuantizer
import torch.nn.functional as F

class QuantizerIdentity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

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

# 使用亚像素卷积 (推荐)
upsampler = UpsampleNet(in_channels=3, use_subpixel=False)

local_ckpt = '/home/dalhxwlyjsuo/criait_tansy/weight/sd-vae-ft-mse'

vae = AutoencoderKL.from_pretrained(local_ckpt)
print(vae)
x = torch.normal(0, 1, (1, 3, 448, 448))
z = vae.encoder(x)
z = vae.quant_conv(z)
print(z.shape)  # [mu, logvar]
quant_conv = vae.quant_conv
post_quant_conv = vae.post_quant_conv

quantize = VectorQuantizer(512, 4, beta=0.25, remap=None, sane_index_shape=False)

local_ckpt = '/home/dalhxwlyjsuo/criait_tansy/weight/EndoViT/pytorch_model.bin'

config = ViTConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072,
                   hidden_act="gelu", hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
                   initializer_range=0.02, layer_norm_eps=1e-12, image_size=448, patch_size=16, num_channels=3,
                   qkv_bias=True, encoder_stride=16)
model = ViTModel(config, add_pooling_layer=False)

model_weights = torch.load(local_ckpt)
model.load_state_dict(model_weights, strict=False)
# print(model)
x = torch.randn((1, 3, 448, 448))
y = model(x).last_hidden_state
print(y.shape)

recon_y = y[:, 1:, ...]
clip_y = y[:, 0, ...]

recon_y = recon_y.view(y.shape[0], 28, 28, recon_y.shape[-1])
print(recon_y.shape)
embed_dim = 768
latent_dim = 4
fc_mu = nn.Linear(embed_dim, latent_dim)
fc_logvar = nn.Linear(embed_dim, latent_dim)
logvar = fc_logvar(recon_y).permute(0, 3, 1, 2)
mu = fc_mu(recon_y).permute(0, 3, 1, 2)
mu_logvar = quant_conv(torch.cat((mu, logvar), dim=1))
mu, logvar = mu_logvar.chunk(2, dim=1)

std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
recon_z = mu + eps * std
print(recon_z.shape)
recon_z, loss, _ = quantize(recon_z)
recon_z = post_quant_conv(recon_z)
out = vae.decoder(recon_z)
print(out.shape)
out = upsampler(out)
print(out.shape)
