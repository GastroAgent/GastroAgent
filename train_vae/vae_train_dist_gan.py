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

# ==============================
# Discriminator: PatchGAN
# ==============================
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64, n_layers=4):  # 4 层下采样
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False)
        ]
        nf_mult = 1
        for n in range(1, n_layers + 1):  # n=1,2,3,4 → 4次下采样
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=False)
            ]
        # Final layer: kernel=3, padding=1, stride=1 → preserves spatial size (32 -> 32)
        layers += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=3, stride=1, padding=1)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)  # Output: (B, 1, 32, 32) for input (B, 3, 512, 512)
    
# ==============================
# Hinge Loss Functions
# ==============================
def hinge_discriminator_loss(real_pred, fake_pred):
    d_real_loss = torch.mean(torch.relu(1.0 - real_pred))
    d_fake_loss = torch.mean(torch.relu(1.0 + fake_pred))
    return d_real_loss + d_fake_loss

def hinge_generator_loss(fake_pred):
    return - torch.mean(fake_pred)

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
    # optimizer_g = optim.AdamW(params_to_optimize, lr=lr, weight_decay=1e-3)
    optimizer_g = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    
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
                real_pred = discriminator(base_image)
                with torch.no_grad():
                    output = model(imgs)
                    recon = output.recon
                fake_pred = discriminator(recon.detach())

                d_loss = hinge_discriminator_loss(real_pred, fake_pred)
                d_loss.backward()
                clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
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
                g_loss, kl = vae_loss(output.recon, base_image, output.mu, output.logvar, beta, p=1)
                if output.commit_loss is not None:
                    g_loss = g_loss + commit_beta * output.commit_loss

                if perceptual_loss_fn is not None:
                    perceptual_loss = perceptual_loss_fn(recon, base_image).mean()
                    g_loss = g_loss + perceptual_loss * beta_perceptual
                else:
                    perceptual_loss = torch.tensor(0.0, device=device)

                fake_pred_for_g = discriminator(recon)
                adv_loss = hinge_generator_loss(fake_pred_for_g)
                g_loss = g_loss + gan_weight * adv_loss

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
    local_rank = dist.get_rank()
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

    image_root = '/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/our_data/new_cropped-2004-2010-endovit'
    dataset = MedicalImageDataset(image_root, transform, base_transform)
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=3, sampler=sampler, num_workers=4, pin_memory=True)
    ######### EndoVit-VAE
    # Initialize VAE
    encoder_ckpt = '/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/flow_matcher_otcfm/EndoViT/pytorch_model.bin'
    decoder_ckpt = '/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/vae_weight/VAEModel'
    vae = VAE(latent_dim=4, encoder_ckpt=encoder_ckpt, decoder_ckpt=decoder_ckpt, use_VQVAE=False).to(device).train()
    # Optional: load pre-trained VAE
    try:
        state_dict = torch.load('/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/flow_matcher_otcfm/vit_vae/vit_vae_ema.pth', map_location=device)
        vae.load_state_dict(state_dict, strict=False)
        print("Loaded pre-trained VAE weights.")
    except Exception as e:
        print("No pre-trained VAE found:", e)
    
    ######### Med-VAE
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        '/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/flow_matcher_otcfm/vae_our').to(device).train()
    
    # Initialize Discriminator
    discriminator = PatchDiscriminator(in_channels=3, n_layers=3).to(device)
    print(discriminator)
    try:
        discriminator.load_state_dict(torch.load("/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/flow_matcher_otcfm/vit_vae/discriminator_16.pth"))
    except:
        pass
    # Wrap with DDP
    d_vae = DDP(vae, device_ids=[local_rank], find_unused_parameters=True)
    d_disc = DDP(discriminator, device_ids=[local_rank], find_unused_parameters=False)

    # Start training
    train_vae_with_gan(
        model=d_vae,
        discriminator=d_disc,
        loader=loader,
        device=device,
        epochs=1,
        lr=1e-5,
        disc_lr=1e-5,
        beta=1e-3,
        use_perceptual=False,
        beta_perceptual=0.25,
        ema_steps=100,
        use_ema=True,
        ema_decay=0.9,
        gan_weight=0.2,
        save_dir='/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/flow_matcher_otcfm/vae_our'
    )

    dist.destroy_process_group()

if __name__ == "__main__":
    run()