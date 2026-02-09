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
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from transformers import AutoFeatureExtractor
from torch.nn.utils import clip_grad_norm_
from random import choices, choice
from utils_ import _get_vector_norm
import random

# Step 1: 自定义 AddGaussianNoise 类
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): (C, H, W) 的张量
        Returns:
            Tensor: 添加高斯噪声后的图像
        """
        if random.random() < self.p:
            noise = torch.randn(tensor.shape) * self.std + self.mean
            return tensor + noise
        else:
            return tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


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
        text = self.dataset[idx][self.text_key]
        batch = {
            'image': img,
            'text': text,
            'image_path': img_file,
            'base_image': base_img
        }
        return batch

class MedicalVAEDataset(Dataset):
    def __init__(self, data_path, transform=None, base_transform=None, text_key='text', image_key='image',
                 positive_dict:dict[str, list] = None):
        self.dataset = json.load(open(data_path, 'r'))
        self.transform = transform
        self.base_transform = base_transform
        self.text_key = text_key
        self.image_key = image_key
        self.positive_dict = positive_dict

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
        text = self.dataset[idx][self.text_key]
        if 'disease' in self.dataset[idx]:
            disease = self.dataset[idx]['disease']
            try: # 换图。
                if random.random() < 0.25 and disease:
                    base_data = random.choice(self.positive_dict[disease])
                    base_image = Image.open(base_data[self.image_key]).convert('RGB')
                    img = base_img
                    base_img = self.base_transform(base_image)
            except:
                pass

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
        self.config = ViTConfig(hidden_size=768, num_hidden_layers=16, num_attention_heads=12, intermediate_size=3072,
                                hidden_act="gelu", hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
                                num_patches=1024, # Max Position Embedding.
                                initializer_range=0.02, layer_norm_eps=1e-10, image_size=(496, 528), patch_size=16,
                                num_channels=3,
                                qkv_bias=True, encoder_stride=16)
        self.H = 496 // 16
        self.W = 528 // 16
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

    def forward(self, x, bool_masked_pos=False, interpolate_pos_encoding=False):
        # x 期望是 shape=[B,3,H,W] 的 tensor，已经做过 Normalize
        if bool_masked_pos:
            bool_masked_pos = torch.zeros((x.shape[0], self.H * self.W, self.embed_dim))
        else:
            bool_masked_pos = None

        out = self.vit(x, interpolate_pos_encoding=interpolate_pos_encoding, bool_masked_pos=bool_masked_pos)  # BaseModelOutput
        return out.last_hidden_state

class QuantizerIdentity(nn.Module):
    def __init__(self):
        super(QuantizerIdentity, self).__init__()

    def forward(self, x):
        return x, 0, None # 保持 VectorQuantizer 的输出格式一致。

class VQAE(nn.Module):
    def __init__(self, encoder_ckpt: str = None,
                 decoder_ckpt: str = None, use_VQVAE: bool = True, **kwargs):
        super().__init__()
        if encoder_ckpt is None or not os.path.exists(encoder_ckpt):
            print(f"请传入正确的 Encoder checkpoint 路径: {encoder_ckpt}")
        if decoder_ckpt is None or not os.path.isdir(decoder_ckpt):
            raise RuntimeError(f"请传入正确的 Decoder Dir 路径: {decoder_ckpt}")

        self.encoder = ViTEncoder(local_ckpt=encoder_ckpt)
        self.encoder_config = self.encoder.config

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
        self.quant_mm = nn.Linear(self.encoder.embed_dim, vae.config['latent_channels'], bias=False)
        self.quant_conv = nn.Conv2d(vae.config['latent_channels'], vae.config['latent_channels'], kernel_size=1, stride=1)
        self.post_quant_conv = vae.post_quant_conv
        if use_VQVAE:
            self.quantize = VectorQuantizer(8192, vae.config['latent_channels'], beta=0.25, remap=None, sane_index_shape=False)
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


    def forward(self, x):
        hidden_states = self.encoder(x)
        cls = hidden_states[:, 0, ...]
        latent = hidden_states[:, 1:, ...]
        assert latent.shape[1] == self.latent_h * self.latent_w, '图像尺寸不正确。'
        recon_z = self.quant_mm(latent)
        recon_z = recon_z.view(recon_z.shape[0], self.latent_h, self.latent_w, -1).permute(0, 3, 1, 2)
        recon_z = self.quant_conv(recon_z)
        recon_z, commit_loss, _ = self.quantize(recon_z)
        recon_z = self.post_quant_conv(recon_z)
        recon = self.decoder(recon_z)
        return VAEOutput(recon=recon,
                         cls=cls,
                         commit_loss=commit_loss)

@dataclass
class VAEOutput:
    recon: torch.Tensor
    cls: torch.Tensor
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
                loss = F.mse_loss(output.recon, base_image, reduction='mean')
                loss = loss + commit_beta * output.commit_loss

                optimizer.zero_grad()
                loss.backward()
                norm = clip_grad_norm_(model.parameters(), max_norm=0.5).item()
                optimizer.step()
                scheduler.step()

                batch_size = imgs.size(0)
                epoch_loss += loss.item() * batch_size

                # 把当前 batch 的 loss 打印到进度条上
                pbar.set_postfix({
                    'batch_loss': f"{loss.item():.4f}",
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
        accuracy[label + f'_j{k}'] = js_topk_correct / total
        accuracy[label + '_kl'] = kl_correct / total
        accuracy[label + f'_k{k}'] = kl_topk_correct / total
        accuracy[label] = cls_correct / total
        accuracy[label + f'_top{k}'] = topk_correct / total
    return accuracy

def eval_clip_similatiry(clip, images_root, device, processor, k = 3):
    clip.eval()
    accuracy = {}
    labels = os.listdir(images_root)
    for label_id, label in enumerate(labels):
        src_path = os.path.join(images_root, label)
        cls_correct = 0
        topk_correct = 0
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
        for _ in tqdm(range(10)):
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
            cos_score = vae_cos_score + clip_cos_score * 1.2

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
