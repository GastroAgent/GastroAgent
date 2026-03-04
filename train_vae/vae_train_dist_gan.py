import gc
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9505))
#     # debugpy.listen(("172.16.0.108", 9504)) 
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

# ### 禁用 高效 attention
# import torch
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)  # 强制使用数学实现（支持高阶导）

from einops import rearrange, repeat, reduce
# 或者只导入你需要的：
from einops.layers.torch import Rearrange
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from collections import OrderedDict
from tqdm import tqdm
from copy import deepcopy
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
import json
import numpy as np
torch.autograd.set_detect_anomaly(True)
# --- 假设这些模块存在 ---
from vae_sim import VAE, AddGaussianNoise, vae_loss
from modelscope import AutoModel

# ==============================
# Discriminator: PatchGAN
# ==============================
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class DinoV3Discriminator(nn.Module):
    def __init__(
        self,
        pretrained_model_name="./weights/dinov3-vitb16",
        device="cuda",  # ← 这个 device 应该是当前进程的 cuda:local_rank
        mode="patch",
        freeze_backbone=True,
        img_size=512,
        patch_size=16,
    ):
        super().__init__()
        self.mode = mode
        self.img_size = img_size
        self.patch_size = patch_size
        self.feat_h = img_size // patch_size
        self.feat_w = img_size // patch_size
        self.N_orig = self.feat_h * self.feat_w

        # ❌ 移除 device_map="auto"
        self.backbone = AutoModel.from_pretrained(pretrained_model_name)
        
        dim = self.backbone.norm.weight.shape[0]

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

        self.patch_head = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b (h w) d -> b d h w', h=self.feat_h, w=self.feat_w),
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, kernel_size=3, stride=2, padding=1),
            Rearrange('b 1 h w -> b (h w) 1')
        )

        # ✅ 手动将整个模型移到指定设备（由外部传入）
        self.to(device)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, x, return_features=False, mode='cls'):
        """
        return:
          - logits: (B, 1) 真假判别分数（未sigmoid）
          - patch_logits (可选): (B, N, 1) patch 分数
          - features (可选): (B, D) 或 (B, N, D)
        """
        outputs = self.backbone(x) # [cls_token, register_tokens, patch_embeddings]

        # 1) 取 token 表示（更稳：优先用 last_hidden_state）
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            tokens = outputs.last_hidden_state          # (B, N+1, D) 通常含 CLS
        else:
            tokens = None

        # 2) global 表示：优先 CLS token；其次 pooler_output
        if tokens is not None and mode == 'cls':
            global_feat = tokens[:, 0, :]              # CLS (B, D)
        elif hasattr(outputs, "pooler_output"):
            global_feat = outputs.pooler_output        # (B, D)
        else:
            raise ValueError("Model outputs has neither last_hidden_state nor pooler_output.")

        logits = self.head(global_feat)                # (B, 1)

        patch_logits = None
        patch_feat = None
        if self.mode == "patch":
            if tokens is None:
                raise ValueError("Patch mode requires last_hidden_state.")
            patch_feat = tokens[:, -self.N_orig:, :]              # (B, N, D)
            patch_logits = self.patch_head(patch_feat) # (B, N, 1)

        if return_features:
            return logits, patch_logits, global_feat, patch_feat
        return logits, patch_logits

# ==============================
# Hinge Loss Functions
# ==============================
def hinge_discriminator_loss(real_pred, fake_pred):
    d_real_loss = torch.mean(torch.relu(1.0 - real_pred))
    d_fake_loss = torch.mean(torch.relu(1.0 + fake_pred))
    return d_real_loss + d_fake_loss

def hinge_generator_loss(fake_pred):
    return - torch.mean(fake_pred)

def d_logistic_loss(d_real, d_fake):
    return torch.mean(F.softplus(-d_real)) + torch.mean(F.softplus(d_fake))

def g_logistic_loss(d_fake):
    return torch.mean(F.softplus(-d_fake))

def r1_reg(d_real, x_real, gamma=10.0):
    # d_real: (B,1) logits; x_real requires_grad=True
    grad = torch.autograd.grad(
        outputs=d_real.sum(), inputs=x_real,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    reg = grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()
    return 0.5 * gamma * reg

def d_patch_loss(patch_logits, real_or_fake_labels):
    B, N = patch_logits.shape[:2]
    # 假设 patch_logits 是 (B, N, 1)
    patch_loss = F.binary_cross_entropy_with_logits(
        patch_logits.view(-1, 1), 
        real_or_fake_labels.reshape(-1, 1).float()
    )
    return patch_loss
# ==============================
# EMA & Utils
# ==============================
@torch.no_grad()
def update_ema(ema_model, model, decay=0.99):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

# ==============================
# Dataset
# ==============================
class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, base_transform=None):
        self.img_paths = []
        for root, dirs, files in os.walk(root_dir):
            for fname in files:
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.img_paths.append(os.path.join(root, fname))
        self.transform = transform
        self.base_transform = base_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        image = self.transform(img)
        base_image = self.base_transform(img)
        return {
            'image': image,
            'image_path': self.img_paths[idx],
            'base_image': base_image
        }

# ==============================
# GAN-enhanced VAE Training
# ==============================
def train_vae_with_gan(
    model,
    discriminator,
    loader,
    device,
    epochs=30,
    lr=1e-4,
    disc_lr=1e-4,
    beta=1.0,
    commit_beta=0.0,
    save_dir='./checkpoints',
    use_perceptual=False,
    beta_perceptual=0.1,
    ema_steps=5,
    use_ema=False,
    ema_decay=0.95,
    gan_weight=0.2
):
    os.makedirs(save_dir, exist_ok=True)
    model.train()
    discriminator.train()

    # Generator optimizer (only decoder-related params)
    params_to_optimize = [
        param for name, param in model.named_parameters()
        if "encoder" not in name and param.requires_grad
    ]
    optimizer_g = optim.AdamW(params_to_optimize, lr=lr, weight_decay=1e-3)
    # optimizer_g = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=disc_lr, weight_decay=1e-3)

    # print('Trainable generator params:', len(params_to_optimize))
    print('Discriminator params:', sum(p.numel() for p in discriminator.parameters()))

    scheduler_g = CosineAnnealingLR(optimizer_g, T_max=1000, eta_min=1e-12)
    scheduler_d = CosineAnnealingLR(optimizer_d, T_max=1000, eta_min=1e-12)

    # Perceptual loss
    if use_perceptual:
        import lpips
        perceptual_loss_fn = lpips.LPIPS(net='alex', spatial=True).to(device)
    else:
        perceptual_loss_fn = None

    # EMA
    if use_ema:
        ema = deepcopy(model.module).to(device)
        requires_grad(ema, False)
    else:
        ema = None

    for epoch in range(1, epochs + 1):
        loader.sampler.set_epoch(epoch)
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        with tqdm(loader, desc=f"[Epoch {epoch}/{epochs}]", unit='batch') as pbar:
            for step, batch in enumerate(pbar):
                try:
                    imgs = batch['image'].to(device)
                    base_image = batch['base_image'].to(device)
                except:
                    imgs = batch.to(device)
                    base_image = imgs

                batch_size = imgs.size(0)

                # -------------------------
                # Train Discriminator
                # -------------------------
                requires_grad(discriminator, True)
                requires_grad(model, False)

                optimizer_d.zero_grad()

                # 关键：detach 输入，避免任何梯度泄漏
                g_base_image = base_image.clone()
                base_image.requires_grad=True
                real_pred, real_patch_pred = discriminator(base_image)
                with torch.no_grad():
                    output = model(imgs)
                    recon = output.recon
                fake_pred, fake_patch_pred = discriminator(recon.detach())

                d_loss = d_logistic_loss(real_pred, fake_pred) + \
                    0.01 * d_patch_loss(real_patch_pred, torch.ones_like(real_patch_pred)) + \
                    0.01 * d_patch_loss(fake_patch_pred, torch.zeros_like(fake_patch_pred))
                
                # d_loss = d_loss + 0.1 * r1_reg(real_pred, base_image)
                d_loss.backward()
                discriminator_norm = clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                optimizer_d.step()
                scheduler_d.step()

                # -------------------------
                # Train Generator (VAE)
                # -------------------------
                requires_grad(discriminator, False)
                requires_grad(model, True)

                optimizer_g.zero_grad()

                output = model(imgs)
                recon = output.recon
                g_loss, kl = vae_loss(output.recon, g_base_image, output.mu, output.logvar, beta, p=1)
                if output.commit_loss is not None:
                    g_loss = g_loss + commit_beta * output.commit_loss

                if perceptual_loss_fn is not None:
                    perceptual_loss = perceptual_loss_fn(recon, g_base_image).mean()
                    g_loss = g_loss + perceptual_loss * beta_perceptual
                else:
                    perceptual_loss = torch.tensor(0.0, device=device)

                fake_pred_for_g, _ = discriminator(recon)
                adv_loss = g_logistic_loss(fake_pred_for_g)
                # g_loss = g_loss + gan_weight * adv_loss

                g_loss.backward()
                norm = clip_grad_norm_(model.parameters(), max_norm=1.0).item()
                optimizer_g.step()
                scheduler_g.step()

                if step % ema_steps == 0 and ema is not None:
                    update_ema(ema, model.module, ema_decay)

                epoch_g_loss += g_loss.item() * batch_size
                epoch_d_loss += d_loss.item() * batch_size

                pbar.set_postfix({
                    'g_loss': f"{g_loss.item():.4f}",
                    'd_loss': f"{d_loss.item():.4f}",
                    'kl': f"{kl:.4f}",
                    'adv': f"{adv_loss.item():.4f}",
                    'percep': f"{perceptual_loss.item():.4f}",
                    'norm': f"{norm:.4f}"
                })

        avg_g_loss = epoch_g_loss / len(loader.dataset)
        avg_d_loss = epoch_d_loss / len(loader.dataset)
        print(f"Epoch {epoch}/{epochs} — Avg G Loss: {avg_g_loss:.4f}, Avg D Loss: {avg_d_loss:.4f}")

        if dist.get_rank() == 0 and epoch % 2 == 0:
            torch.save(model.module.state_dict(), os.path.join(save_dir, f"vit_vae_{epoch}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(save_dir, f"discriminator_{epoch}.pth"))
            if ema is not None:
                torch.save(ema.state_dict(), os.path.join(save_dir, f"vit_vae_ema.pth"))
            print(f"  ↳ saved checkpoints at epoch {epoch}")

# ==============================
# Main Entry
# ==============================
def run():
    dist.init_process_group(backend="nccl")
    # local_rank = dist.get_rank()
    # torch.cuda.set_device(local_rank)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    base_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    image_root = './data2/tsy/EndoViT/our_data/new_cropped-2004-2010-endovit'
    dataset = MedicalImageDataset(image_root, transform, base_transform)
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=3, sampler=sampler, num_workers=4, pin_memory=True)
    ######### EndoVit-VAE
    # Initialize VAE 
    encoder_ckpt = './whole_wass_flow_match/flow_matcher_otcfm/EndoViT/pytorch_model.bin'
    decoder_ckpt = './data2/tsy/EndoViT/vae_weight/VAEModel'
    vae = VAE(latent_dim=4, encoder_ckpt=encoder_ckpt, decoder_ckpt=decoder_ckpt, use_VQVAE=False).to(device).train()
    # Optional: load pre-trained VAE
    try:
        state_dict = torch.load('./whole_wass_flow_match/flow_matcher_otcfm/vit_vae/vit_vae_ema.pth', map_location=device)
        vae.load_state_dict(state_dict, strict=False)
        print("Loaded pre-trained VAE weights.")
    except Exception as e:
        print("No pre-trained VAE found:", e)
    
    # Initialize Discriminator
    discriminator = DinoV3Discriminator(device=device)
    print(discriminator)
    try:
        discriminator.load_state_dict(torch.load("./whole_wass_flow_match/flow_matcher_otcfm/vit_vae/discriminator_16.pth"))
    except:
        pass
    # Wrap with DDP
    d_vae = DDP(vae, device_ids=[local_rank], find_unused_parameters=True)
    d_disc = DDP(discriminator, device_ids=[local_rank], find_unused_parameters=True)

    # Start training
    train_vae_with_gan(
        model=d_vae,
        discriminator=d_disc,
        loader=loader,
        device=device,
        epochs=100,
        lr=1e-4,
        disc_lr=1e-4,
        beta=5e-3,
        use_perceptual=False,
        beta_perceptual=0.125,
        ema_steps=100,
        use_ema=True,
        ema_decay=0.95,
        gan_weight=0.1,
        save_dir='./wass_flow_match_tsy/train/train_gan_v2/vit-vae'
    )

    dist.destroy_process_group()

if __name__ == "__main__":
    run()