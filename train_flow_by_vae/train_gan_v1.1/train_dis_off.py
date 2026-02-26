############### 专注于 强力的判别器 ###########
import argparse
import copy
import glob
import os
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import gc
from einops import rearrange, repeat, reduce
# 或者只导入你需要的：
from einops.layers.torch import Rearrange
import json
import torch
import random
from torch import nn
from torch.nn import init
from torchvision import transforms
from tqdm import trange
import sys
from functools import partial
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from transformers import AutoModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
# ==============================
# Discriminator: PatchGAN
# ==============================
class DinoV3Discriminator(nn.Module):
    def __init__(
        self,
        pretrained_model_name = "/mnt/inaisfs/data/home/tansy_criait/weights/dinov3-vitb16",
        # pretrained_model_name = "/mnt/inaisfs/data/home/tansy_criait/weights/dinov3-convnext-large",
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
        # dim = self.backbone.layer_norm.weight.shape[0]
        
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

class DinoV3ConvDiscriminator(nn.Module):
    def __init__(
        self,
        # pretrained_model_name = "/mnt/inaisfs/data/home/tansy_criait/weights/dinov3-vitb16",
        pretrained_model_name = "/mnt/inaisfs/data/home/tansy_criait/weights/dinov3-convnext-large",
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
        
        # dim = self.backbone.norm.weight.shape[0]
        dim = self.backbone.layer_norm.weight.shape[0]
        
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

        self.patch_head = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b (h w) d -> b d h w', h=16, w=16),
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
            patch_feat = tokens[:, 1:, :]              # (B, N, D)
            patch_logits = self.patch_head(patch_feat) # (B, N // 4, 1)

        if return_features:
            return logits, patch_logits, global_feat, patch_feat
        return logits, patch_logits

def normalize_samples(x):
    x = (x / 2 + 0.5)
    x_min = x.amin(dim=(1, 2, 3), keepdim=True)  # shape: (B, 1, 1, 1)
    x_max = x.amax(dim=(1, 2, 3), keepdim=True)  # shape: (B, 1, 1, 1)

    # 防止分母为0
    scale = (x_max - x_min).clamp(min=1e-4)

    # 执行缩放
    x_scaled = (x - x_min) / scale
    return x_scaled

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

def r1_reg(d_real, x_real, gamma=.01):
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

def requires_grad(model, flag):
    """
    设置模型中所有参数的 requires_grad 属性。

    Args:
        model (torch.nn.Module): 要操作的模型。
        flag (bool): True 表示启用梯度（训练），False 表示冻结参数（不计算梯度）。
    """
    for param in model.parameters():
        param.requires_grad = flag


class MedicalFakeRealDataset(Dataset):
    def __init__(self,
                 real_dir='/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/new_cropped-2004-2010',
                 fake_dir='/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/fake',
                 transform_real=None,
                 transform_fake=None):
        self.real_imgs = glob.glob(f"{real_dir}/*.jpg") + glob.glob(f"{real_dir}/*/*.jpg") + glob.glob(f"{real_dir}/*/*/*.jpg")
        self.fake_imgs = glob.glob(f"{fake_dir}/*.jpg") + glob.glob(f"{fake_dir}/*/*.jpg") + glob.glob(f"{fake_dir}/*/*/*.jpg") + glob.glob(f"{fake_dir}/*/*/*.png")  
        self.transform_real = transform_real
        self.transform_fake = transform_fake

        
    def __len__(self):
        return min([len(self.real_imgs), len(self.fake_imgs)])

    def __getitem__(self, idx):
        real_img = Image.open(self.real_imgs[idx])
        fake_img = Image.open(self.fake_imgs[idx])
        if random.random() < 0.01:
            random.shuffle(self.real_imgs)
            random.shuffle(self.fake_imgs)
        real_tensor = self.transform_real(real_img)
        fake_tensor = self.transform_real(fake_img)
        batch = dict(
            real_tensor=real_tensor, fake_tensor=fake_tensor,
        )
        
        return batch

@torch.no_grad()
def validate_discriminator(
    discriminator,
    val_dataloader,
    device,
    threshold=0.5  # 用于二值化预测（仅用于 Precision/Recall/F1）
):
    """
    验证判别器在真假样本上的二分类性能。
    
    Args:
        discriminator: 判别器模型，输出 (logits, _) 或 logits
        val_dataloader: 包含 'real_tensor' 和 'fake_tensor' 的 DataLoader
        device: 如 'cuda'
        threshold: 将 logits 转为二分类的阈值（默认 0.5）
    
    Returns:
        dict: 包含 accuracy, auc, f1, precision, recall, avg_loss
    """
    discriminator.eval()
    
    all_labels = []
    all_preds = []      # 用于 AUC（原始 logits 或 probs）
    all_binary_preds = []  # 用于 F1/Precision/Recall
    total_loss = 0.0
    n_batches = 0

    print("Evaluating discriminator on real/fake classification...")
    for batch in tqdm(val_dataloader, desc="Validation"):
        # 真实样本（标签=1）
        real = batch['real_tensor'].to(device)
        fake = batch['fake_tensor'].to(device)
        real = normalize_samples(real)
        fake = normalize_samples(fake)

        # 前向传播
        real_logits, _ = discriminator(real)  # 假设输出 (logits, feature)
        fake_logits, _ = discriminator(fake)
        
        # 合并
        logits = torch.cat([real_logits, fake_logits], dim=0).squeeze(-1)  # [2B]
        labels = torch.cat([torch.ones(real.size(0)), torch.zeros(fake.size(0))], dim=0).to(device)  # [2B]

        # 计算 logistic loss（可选）
        loss = torch.nn.functional.softplus(-labels * logits).mean()  # 等价于 d_logistic_loss
        total_loss += loss.item()
        n_batches += 1

        # 收集预测和标签（CPU + numpy）
        probs = torch.sigmoid(logits).cpu().numpy()
        binary_pred = (probs >= threshold).astype(int)
        labels_np = labels.cpu().numpy()

        all_preds.append(probs)
        all_binary_preds.append(binary_pred)
        all_labels.append(labels_np)

    # 合并所有结果
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_preds)
    y_pred = np.concatenate(all_binary_preds)

    # 计算指标
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')  # 如果只有一类

    avg_loss = total_loss / n_batches

    results = {
        'avg_loss': avg_loss,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc
    }

    return results

def train():
    """Main training function."""
    # discriminator_checkpoints = '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/vit-vae/discriminator_26.pth'
    # discriminator_checkpoints = "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/base-flow-match_vae_gan/general_discriminator.pt"
    # discriminator_checkpoints = '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/base-flow-match_vae_gan/general_hard_discriminator.pt'
    discriminator_checkpoints = '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/discriminator/general_discriminator.pt'
    discriminator = DinoV3Discriminator(device="cuda")
    discriminator.device = "cuda"
    discriminator = discriminator.to("cuda")
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-3, weight_decay=1e-3)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),], p=0.1), 
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 0.5))], p=0.1), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # 归一化，-1~1
    ])

    train_dataset = MedicalFakeRealDataset(
        real_dir='/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/new_cropped-2004-2010',
        fake_dir='/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/hard_fake',
        transform_real=transform,
        transform_fake=transform
    )
    val_dataset = MedicalFakeRealDataset(
        real_dir='/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/val_new_cropped-2004-2010',
        fake_dir='/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/hard_fake_eval',
        transform_real=transform,
        transform_fake=transform
    )
    
    # # 定义训练集和验证集的比例
    # total_size = len(dataset)
    # train_ratio = 0.8
    # train_size = max(int(train_ratio * total_size), total_size - 6000)
    # val_size = total_size - train_size
                                                                                                                                                                                         
    # 随机划分
    # _, val_dataset = random_split(val_dataset, [len(val_dataset) - 5000, 5000])
    batch_size = 9
    
    # train_dataset = MedicalFakeRealDataset(
    #     real_dir='/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/new_cropped-2004-2010',
    #     fake_dir='/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/hard_fake',
    #     transform_real=transform,
    #     transform_fake=transform
    # )
    
    # 创建 DataLoader
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证时通常不需要打乱
        num_workers=0,
        drop_last=False,  # 验证时一般保留最后的小 batch
    )
    
    # from vae_sim import VAE
    # encoder_ckpt = '/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/flow_matcher_otcfm/EndoViT/pytorch_model.bin'
    # decoder_ckpt = '/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/vae_weight/VAEModel'
    # vae = VAE(latent_dim=4, encoder_ckpt=encoder_ckpt, decoder_ckpt=decoder_ckpt, use_VQVAE=False).to(device).eval()
    # # Optional: load pre-trained VAE
    # vae.device = device
    # state_dict = torch.load('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/flow_matcher_otcfm/vit_vae/vit_vae_ema.pth', map_location=device)
    # vae.load_state_dict(state_dict, strict=False)
    # print("Loaded pre-trained VAE weights.")
    
    if discriminator_checkpoints:
        try:
            discriminator.load_state_dict(torch.load(discriminator_checkpoints, map_location='cpu'), strict=True)
        except:
            state_dict = torch.load(discriminator_checkpoints, map_location='cpu')
            new_state_dict = {}
            for name in state_dict:
                new_state_dict[name.replace("module.", "")] = state_dict[name]
            discriminator.load_state_dict(new_state_dict, strict=True)
        discriminator.to("cuda")
    torch.cuda.empty_cache()
    # Ptach work for now. TODO: Remove the Global steps later
    start_step = global_step = 1
    save_step = len(dataloader) // 2
    epochs = 1
    total_steps = len(dataloader) * epochs

    # val_loss = 0
    # with torch.no_grad():
    #     print("Eval...")
    #     for batch in tqdm(val_dataloader):
    #         fake_tensor = batch['fake_tensor']
    #         real_tensor = batch['real_tensor']
    #         real_pred, _ = discriminator(real_tensor.to(discriminator.device))
    #         fake_pred, _ = discriminator(fake_tensor.to(discriminator.device))
    #         val_loss += d_logistic_loss(real_pred, fake_pred).item() / len(val_dataloader)
    # best_loss = val_loss
    # print("val_loss: ", val_loss)

    val_res = validate_discriminator(
        discriminator,
        val_dataloader,
        "cuda",
        threshold=0.5  # 用于二值化预测（仅用于 Precision/Recall/F1）
    )
    print(val_res)
    # Training Loop
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        for batch in tqdm(dataloader):  
            global_step += 1

            # Get batch
            fake_tensor = batch['fake_tensor']
            real_tensor = batch['real_tensor']
            fake_tensor = normalize_samples(fake_tensor)
            real_tensor = normalize_samples(real_tensor)
            # images = torch.cat([fake_tensor, real_tensor], dim=0)
            # with torch.no_grad():
            #     images = images.to(vae.device)
            #     posterior = vae.encode(images).latent_dist
            #     images = posterior.sample() * 0.18215
            #     fake_z, real_z = images.chunk(2, dim=0)

            # -------------------------
            # Train Discriminator
            # -------------------------
            requires_grad(discriminator, True)
            optimizer_d.zero_grad()
            
            # 关键：detach 输入，避免任何梯度泄漏
            real_tensor.requires_grad=True
            real_pred, real_patch_pred = discriminator(real_tensor.to(discriminator.device))
            fake_pred, fake_patch_pred = discriminator(fake_tensor.to(discriminator.device))

            d_loss = d_logistic_loss(real_pred, fake_pred) + \
                0.1 * d_patch_loss(real_patch_pred, torch.ones_like(real_patch_pred)) + \
                0.1 * d_patch_loss(fake_patch_pred, torch.zeros_like(fake_patch_pred))
            # d_loss = d_loss + r1_reg(real_pred, real_tensor, gamma = 1e-4)
            d_loss.backward()
            d_grad_norm = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
            optimizer_d.step()

            if global_step % 10 == 0:
                print(f"Discriminator loss: {d_loss.item():.4f} Discriminator GradNorm: {d_grad_norm.item():.4f}")

            if save_step > 0 and global_step % save_step == 0:
                val_loss = 0
                print("Eval...")
                with torch.no_grad():
                    for batch in tqdm(val_dataloader):
                        fake_tensor = batch['fake_tensor']
                        real_tensor = batch['real_tensor']
                        real_tensor = normalize_samples(real_tensor)
                        fake_tensor = normalize_samples(fake_tensor)
                        real_pred, _ = discriminator(real_tensor.to(discriminator.device))
                        fake_pred, _ = discriminator(fake_tensor.to(discriminator.device))
                    val_loss += d_logistic_loss(real_pred, fake_pred).item() / len(val_dataloader)
                print("val_loss: ", val_loss)
                if val_loss < best_loss:
                    torch.save(discriminator.state_dict(), "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/discriminator/general_hard_discriminator.pt")
                    best_loss = val_loss
    
def main():
    """Main entry point."""
    train()

if __name__ == "__main__":
    main()