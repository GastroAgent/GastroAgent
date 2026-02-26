############### 专注于 强力的判别器（DDP 版本） ###########
import argparse
import copy
import glob
import os
import torch.nn.functional as F
# os.environ['CUDA_VISIBLE_DEVICES'] = ''  # ← 由 torchrun 自动管理，此处清空避免冲突
import gc
from einops import rearrange, repeat, reduce
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
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from transformers import AutoModel

# ==============================
# Discriminator: PatchGAN with DINOv3
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
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, kernel_size=3, stride=1, padding=1),
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

# ==============================
# Loss Functions
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
    grad = torch.autograd.grad(
        outputs=d_real.sum(), inputs=x_real,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    reg = grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()
    return 0.5 * gamma * reg

def d_patch_loss(patch_logits, real_or_fake_labels):
    B, N = patch_logits.shape[:2]
    loss = F.binary_cross_entropy_with_logits(
        patch_logits.view(-1, 1), 
        real_or_fake_labels.reshape(-1, 1).float()
    )
    return loss

def requires_grad(model, flag):
    for param in model.parameters():
        param.requires_grad = flag


# ==============================
# Dataset
# ==============================
class MedicalFakeRealDataset(Dataset):
    def __init__(self,
                 real_dir='/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/new_cropped-2004-2010',
                 fake_dir='/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/fake',
                 transform_real=None,
                 transform_fake=None,
                 real_dir2 = '',
                 fake_dir2 = ''
                 ):
        self.real_imgs = glob.glob(f"{real_dir}/*.jpg") + \
                         glob.glob(f"{real_dir}/*/*.jpg") + \
                         glob.glob(f"{real_dir}/*/*/*.jpg")
        self.fake_imgs = glob.glob(f"{fake_dir}/*.jpg") + \
                         glob.glob(f"{fake_dir}/*/*.jpg") + \
                         glob.glob(f"{fake_dir}/*/*/*.jpg") + \
                         glob.glob(f"{fake_dir}/*/*/*.png")
        if real_dir2:
            if isinstance(real_dir2, list):
                for real_dir in real_dir2:
                    self.real_imgs += glob.glob(f"{real_dir}/*.jpg") + \
                        glob.glob(f"{real_dir}/*/*.jpg") + \
                        glob.glob(f"{real_dir}/*/*/*.jpg")
            else:
                self.real_imgs += glob.glob(f"{real_dir2}/*.jpg") + \
                    glob.glob(f"{real_dir2}/*/*.jpg") + \
                    glob.glob(f"{real_dir2}/*/*/*.jpg")

        if fake_dir2:
            if isinstance(fake_dir2, list):
                for fake_dir in fake_dir2:
                    self.real_imgs += glob.glob(f"{fake_dir}/*.jpg") + \
                        glob.glob(f"{fake_dir}/*/*.jpg") + \
                        glob.glob(f"{fake_dir}/*/*/*.jpg")
            else:
                self.fake_imgs += glob.glob(f"{fake_dir2}/*.jpg") + \
                    glob.glob(f"{fake_dir2}/*/*.jpg") + \
                    glob.glob(f"{fake_dir2}/*/*/*.jpg")
        self.transform_real = transform_real
        self.transform_fake = transform_fake

    def __len__(self):
        return min(len(self.real_imgs), len(self.fake_imgs))

    def __getitem__(self, idx):
        real_img = Image.open(self.real_imgs[idx]).convert('RGB')
        fake_img = Image.open(self.fake_imgs[idx]).convert('RGB')
        if random.random() < 0.01:
            random.shuffle(self.real_imgs)
            random.shuffle(self.fake_imgs)
        real_tensor = self.transform_real(real_img)
        fake_tensor = self.transform_real(fake_img)
        return dict(real_tensor=real_tensor, fake_tensor=fake_tensor)



# ==============================
# DDP Setup & Training
# ==============================
def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()


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

def normalize_samples(x):
    x = (x / 2 + 0.5)
    x_min = x.amin(dim=(1, 2, 3), keepdim=True)  # shape: (B, 1, 1, 1)
    x_max = x.amax(dim=(1, 2, 3), keepdim=True)  # shape: (B, 1, 1, 1)

    # 防止分母为0
    scale = (x_max - x_min).clamp(min=1e-4)

    # 执行缩放
    x_scaled = (x - x_min) / scale
    return x_scaled

def train():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    # Model
    discriminator_checkpoints = ''
    # discriminator = DinoV3ConvDiscriminator(device=device, mode="patch", freeze_backbone=True)
    discriminator = DinoV3Discriminator(device=device, mode="patch", freeze_backbone=True)
    discriminator = DDP(discriminator, device_ids=[local_rank], find_unused_parameters=False)

    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-3, weight_decay=1e-3)

    # Transform
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 0.5))], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Dataset
    train_dataset = MedicalFakeRealDataset(
        real_dir='/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/new_cropped-2004-2010',
        fake_dir='/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/fake',
        transform_real=transform,
        transform_fake=transform,
        real_dir2 = [
            # '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/hard_fake/hard_fake_55000',
            # '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/hard_fake/hard_fake_75000',
            # '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/hard_fake/hard_fake_100000',
            '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/step_score_generated/similarity_images_draw/positive'
        ],
        fake_dir2 = [
            '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/hard_fake/hard_fake_55000_vit-vae',
            '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/hard_fake/hard_fake_60000_vit-vae',
            
        ]
    )
    val_dataset = MedicalFakeRealDataset(
        real_dir='/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/val_new_cropped-2004-2010',
        fake_dir='/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/fake_eval',
        transform_real=transform,
        transform_fake=transform,
        fake_dir2 = '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/hard_fake_eval/hard_fake_60000_vit-vae'
    )
    # Samplers
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    # batch_size = 24
    batch_size = 20
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # Load checkpoint (only rank 0 loads, then broadcast via DDP)
    if os.path.exists(discriminator_checkpoints):
        map_location = {'cuda:%d' % 0: f'cuda:{local_rank}'}
        state_dict = torch.load(discriminator_checkpoints, map_location=map_location)
        discriminator.module.load_state_dict(state_dict, strict=True)
        if local_rank == 0:
            print("Loaded pre-trained discriminator weights.")

    torch.cuda.empty_cache()

    epochs = 50
    save_step = len(dataloader) // 2
    best_loss = float('inf')
    global_step = 0

    # Initial validation on rank 0
    if local_rank == 0:
        discriminator.eval()
        with torch.no_grad():            
            val_res = validate_discriminator(
                discriminator,
                val_dataloader,
                "cuda",
                threshold=0.5  # 用于二值化预测（仅用于 Precision/Recall/F1）
            )
            print(val_res)
            best_loss = val_res['avg_loss']
            print(f"Initial val_loss: {best_loss:.6f}")
            best_f1 = val_res['f1']
            print(f"Initial F1: {best_f1:.6f}")
        discriminator.train()

    # Training loop
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        for batch in tqdm(dataloader, disable=(local_rank != 0)):
            global_step += 1
            real_tensor = batch['real_tensor'].to(device, non_blocking=True)
            fake_tensor = batch['fake_tensor'].to(device, non_blocking=True)
            real_tensor = normalize_samples(real_tensor)
            fake_tensor = normalize_samples(fake_tensor)
            # Train Discriminator
            requires_grad(discriminator, True)
            optimizer_d.zero_grad()

            real_tensor.requires_grad_(True)
            real_pred, real_patch_pred = discriminator(real_tensor)
            fake_pred, fake_patch_pred = discriminator(fake_tensor.detach())

            d_loss = d_logistic_loss(real_pred, fake_pred) + \
                     0.1 * d_patch_loss(real_patch_pred, torch.ones_like(real_patch_pred)) + \
                     0.1 * d_patch_loss(fake_patch_pred, torch.zeros_like(fake_patch_pred))

            d_loss.backward()
            d_grad_norm = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
            optimizer_d.step()

            if local_rank == 0 and global_step % 10 == 0:
                print(f"Step {global_step}, D loss: {d_loss.item():.4f}, GradNorm: {d_grad_norm:.4f}")

            # Validation & Save
            if local_rank == 0 and save_step > 0 and global_step % save_step == 0:
                discriminator.eval()
                with torch.no_grad():            
                    val_res = validate_discriminator(
                        discriminator,
                        val_dataloader,
                        "cuda",
                        threshold=0.5  # 用于二值化预测（仅用于 Precision/Recall/F1）
                    )
                    print(val_res)
                    val_loss = val_res['avg_loss']
                    print(f"Eval val_loss: {val_loss:.6f}")
                    val_f1 = val_res['f1']
                    print(f"Eval F1: {val_f1:.6f}")

                if val_loss < best_loss and best_f1 < val_f1:
                    best_loss = val_loss
                    best_f1 = val_f1
                    torch.save(
                        discriminator.module.state_dict(),
                        "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v1/discriminator/general_discriminator2.pt"
                    )
                discriminator.train()

    cleanup_ddp()


def main():
    train()


if __name__ == "__main__":
    # 启动方式示例（4卡）：
    # torchrun --nproc_per_node=4 --master_port=29501 train_ddp.py
    main()