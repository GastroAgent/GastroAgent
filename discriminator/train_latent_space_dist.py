import glob
import json
import math
from tqdm import tqdm
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.utils import shuffle
from diffusers import AutoencoderKL
from math import sqrt

# ------------------ 环境设置 ------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # 根据实际 GPU 数量调整
sys.path.append('./GasAgent-main/discriminator')
# ------------------ 自定义模块导入 ------------------
from utils.data_loader import MedicalJsonDataset
from utils.data_loader import MedicalTripletJsonDataset
from utils.train_utils import infiniteloop
from utils.data_utils import create_dataloaders_by_pairs
from torch.utils.tensorboard import SummaryWriter

# ------------------ 模型定义 ------------------

class EmbeddingNetwork(nn.Module):
    def __init__(self, pretrained=True, freeze_base=False, model='resnet34'):
        super(EmbeddingNetwork, self).__init__()
        if model == 'resnet34':
            self.base_model = models.resnet34(pretrained=pretrained)
        else:
            self.base_model = models.resnet101(pretrained=pretrained)
        self.base_model.conv1 = nn.Conv2d(4, self.base_model.conv1.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 1024)
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        self.l2norm = lambda x: F.normalize(x, p=2, dim=1)

    def forward(self, x):
        features = self.base_model(x)
        return self.l2norm(features)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_chans=4, embed_dim=64):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, embed_dim=64, depth=2, num_heads=4, mlp_ratio=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class UpsampleDecoder(nn.Module):
    def __init__(self, embed_dim=64, patch_size=8, out_chans=4):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_chans)

    def forward(self, x):
        B, N, _ = x.shape
        h = w = int(N ** 0.5)
        x = self.proj(x).reshape(B, h, w, self.patch_size, self.patch_size, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, -1, h * self.patch_size, w * self.patch_size)
        return x


class AttentionDownEncoderXL(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=64, patch_size=4, in_chans=4, embed_dim=512)
        self.transformer = SimpleTransformer(embed_dim=512, depth=16, num_heads=8, mlp_ratio=4)
        self.decoder = UpsampleDecoder(embed_dim=512, patch_size=2, out_chans=4)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = self.decoder(x)
        return x.reshape(x.shape[0], -1)


class TripletNetwork(nn.Module):
    def __init__(self, pretrained=True, freeze_base=False, model='resnet34'):
        super(TripletNetwork, self).__init__()
        self.processor = None
        if model == 'attention':
            self.embedding = AttentionDownEncoderXL()
        elif model == 'convnext':
            from transformers import AutoImageProcessor, DINOv3ConvNextModel
            pretrained_model_name = "./weights/dinov3-convnext-base"
            self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
            self.embedding = DINOv3ConvNextModel.from_pretrained(pretrained_model_name)
            in_channel = 4
            output_channel = self.embedding.stages[0].downsample_layers[0].out_channels
            stride = (2, 2)
            kernel_size = (2, 2)
            padding = self.embedding.stages[0].downsample_layers[0].padding
            self.embedding.stages[0].downsample_layers[0] = nn.Conv2d(in_channel, output_channel, stride=stride, kernel_size=kernel_size, padding=padding)
        else:
            self.embedding = EmbeddingNetwork(pretrained=pretrained, freeze_base=freeze_base, model=model)

    def encode(self, x, return_image=False):
        if self.processor is not None:
            outputs = self.embedding(x)
            embeds = outputs.pooler_output
            embeds = F.normalize(embeds, p=2, dim=1)
        else:
            embeds = self.embedding(x)

        if embeds.ndim == 2:
            B, D = embeds.shape
            if return_image:
                H = W = int(sqrt(D // 4))
                return embeds.view(B, 4, H, W)
            else:
                return embeds
        elif embeds.ndim == 4:
            B, C, H, W = embeds.shape
            if return_image:
                return embeds
            else:
                return embeds.view(B, -1)

    def forward(self, anchor, positive, negative):
        anchor_emb = self.encode(anchor, False)
        positive_emb = self.encode(positive, False)
        negative_emb = self.encode(negative, False)
        return anchor_emb, positive_emb, negative_emb


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_sim = F.cosine_similarity(anchor, positive, dim=1)
        neg_sim = F.cosine_similarity(anchor, negative, dim=1)
        d_pos = 1 - pos_sim
        d_neg = 1 - neg_sim
        loss = torch.mean(torch.clamp(d_pos - d_neg + self.margin, min=0.0))
        return loss


def clip_style_triplet_loss(anchor_emb, positive_emb, negative_emb, temperature=0.07):
    B, D = anchor_emb.shape
    anchor_emb = F.normalize(anchor_emb, dim=1)
    positive_emb = F.normalize(positive_emb, dim=1)
    negative_emb = F.normalize(negative_emb, dim=1)

    sim_matrix = anchor_emb @ negative_emb.t()
    pos_sim = torch.sum(anchor_emb * positive_emb, dim=1)
    sim_matrix = sim_matrix.clone()
    sim_matrix[range(B), range(B)] = pos_sim

    logits = sim_matrix / temperature
    labels = torch.arange(B, device=anchor_emb.device)
    loss = F.cross_entropy(logits, labels)
    return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, other, target):
        cosine_similarity = F.cosine_similarity(anchor, other, dim=1)
        distance = 1 - cosine_similarity
        loss = torch.mean(target * distance + (1 - target) * torch.clamp(self.margin - distance, min=0.0))
        return loss


def sinkhorn_loss(bx, bx1, epsilon=0.1, n_iter=25, reduction='mean'):
    def reshape(x):
        if x.ndim == 2:
            B, D = x.shape
            H = W = int((D // 4) ** 0.5)
            x = x.view(B, 4, H, W).permute(0, 2, 3, 1).reshape(B, H*W, -1)
        elif x.ndim == 4:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        elif x.ndim == 3:
            x = x.unsqueeze(0)
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        return x

    bx = reshape(bx)
    bx1 = reshape(bx1)
    B, N, D = bx.shape
    _, M, _ = bx1.shape

    x2 = torch.sum(bx**2, dim=-1, keepdim=True)
    y2 = torch.sum(bx1**2, dim=-1, keepdim=True)
    cross = torch.bmm(bx, bx1.transpose(-1, -2))
    cost_matrix = x2 - 2 * cross + y2.transpose(-1, -2)
    cost_matrix = torch.clamp(cost_matrix, min=0.0)

    K = torch.exp(-cost_matrix / epsilon)
    a = torch.ones(B, N, device=bx.device) / N
    b = torch.ones(B, M, device=bx.device) / M
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(n_iter):
        u = a / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)
        v = b / (torch.bmm(K.transpose(-1,-2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)

    P = u.unsqueeze(-1) * K * v.unsqueeze(-2)
    loss = torch.sum(P * cost_matrix, dim=(1,2))
    return loss.mean() if reduction == 'mean' else loss


def cal_wasserstein_loss(x, x1, **kwargs):
    return sinkhorn_loss(x, x1, **kwargs)


class Generator:
    def __init__(self):
        self.device = "cuda"
        self.vae = AutoencoderKL.from_pretrained(
            './whole_wass_flow_match/flow_matcher_otcfm/vae'
        ).to(self.device).eval()


# ------------------ 分布式训练函数 ------------------

def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank


def cleanup():
    dist.destroy_process_group()


def evaluate_triplet(model, dataset, device, vae, rank=0):
    model.eval()
    correct = 0
    strict_correct = 0
    total = len(dataset)
    with torch.no_grad():
        for i in range(len(dataset)):
            batch = dataset[i]
            anchor, positive, negative = batch["anchor"], batch["positive"], batch["negative"]
            anchor, positive, negative = anchor.to(vae.device), positive.to(vae.device), negative.to(vae.device)
            anchor = vae.encode(anchor.unsqueeze(0)).latent_dist.sample() * 0.18215
            positive = vae.encode(positive.unsqueeze(0)).latent_dist.sample() * 0.18215
            negative = vae.encode(negative.unsqueeze(0)).latent_dist.sample() * 0.18215

            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
            pos_score = (anchor_emb @ positive_emb.T).item()
            neg_score = (anchor_emb @ negative_emb.T).item()

            if pos_score > neg_score and neg_score < 0.2 and pos_score > 0.8:
                strict_correct += 1
            if pos_score > neg_score and (pos_score - neg_score) > 0.1:
                correct += 1

    # Gather results across all ranks
    correct_tensor = torch.tensor([correct], dtype=torch.float32, device=device)
    strict_correct_tensor = torch.tensor([strict_correct], dtype=torch.float32, device=device)
    total_tensor = torch.tensor([total], dtype=torch.float32, device=device)

    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(strict_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    accuracy = correct_tensor.item() / total_tensor.item()
    strict_accuracy = strict_correct_tensor.item() / total_tensor.item()

    if rank == 0:
        print(f"Evaluation Accuracy: {accuracy:.4f}")
        print(f"Strict Evaluation Accuracy: {strict_accuracy:.4f}")

    return strict_accuracy


def train_triplet(model, dataloader, eval_dataset, criterion, optimizer, device, rank,
                  criterion_contrastive=None, generator=None, epochs=20):
    if generator is None:
        raise NotImplementedError
    vae = generator.vae
    total_step = epochs * len(dataloader)
    best_acc = 0.0

    writer = SummaryWriter(log_dir="./whole_wass_flow_match/discriminator/logs") if rank == 0 else None

    dataloader_inf = infiniteloop(dataloader)
    model.train()

    for step in tqdm(range(1, total_step + 1), disable=(rank != 0)):
        optimizer.zero_grad()
        batch = next(dataloader_inf)

        anchor, positive, negative = batch["anchor"].to(vae.device), batch["positive"].to(vae.device), batch["negative"].to(vae.device)
        with torch.no_grad():
            anchor = vae.encode(anchor).latent_dist.sample() * 0.18215
            positive = vae.encode(positive).latent_dist.sample() * 0.18215
            negative = vae.encode(negative).latent_dist.sample() * 0.18215

        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)

        loss_triplet = criterion(anchor_emb, positive_emb, negative_emb)
        loss_clip = clip_style_triplet_loss(anchor_emb, positive_emb, negative_emb)
        loss = 0.5 * loss_triplet + 0.5 * loss_clip

        emb_norm_loss = ((anchor_emb.norm(dim=1) - 1.0) ** 2 +
                         (positive_emb.norm(dim=1) - 1.0) ** 2 +
                         (negative_emb.norm(dim=1) - 1.0) ** 2).mean()
        loss += 0.25 * emb_norm_loss

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.5)
        optimizer.step()

        if rank == 0 and step % 10 == 0:
            print(f"Global Step {step}/{total_step} | Loss: {loss.item():.4f} | Grad Norm: {grad_norm.item():.4f}")
            if writer:
                writer.add_scalar('Training/Loss', loss.item(), step)

        if rank == 0 and (step % 1000 == 0 or step == total_step):
            print("Eval...")
            accuracy = evaluate_triplet(model, eval_dataset, device, vae, rank=rank)
            if accuracy > best_acc:
                best_acc = accuracy
                save_path = "./discriminator/latent_model_weight/convnext5_ddp.pt"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.module.state_dict(), save_path)
            if writer:
                writer.add_scalar('Evaling/Acc', accuracy, step)

    if rank == 0 and writer:
        writer.close()


# ------------------ 主函数 ------------------

def main():
    rank = setup_ddp()
    device = torch.device(f"cuda:{rank}")

    # Transform
    transform_B = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.25),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 0.25))], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Dataset
    train_dataset = MedicalTripletJsonDataset(
        path="./train/train_gan_v2/step_score_generated/train_triplet_all_dataset.json",
        transform=transform_B,
    )
    eval_dataset = MedicalTripletJsonDataset(
        path="./train/train_gan_v2/step_score_generated/eval_triplet_all_dataset.json",
        transform=transform_B,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    dataloader = DataLoader(
        train_dataset,
        batch_size=24,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Model
    model = TripletNetwork(pretrained=True, freeze_base=False, model='convnext').to(device)
    try:
        model.embedding.post_init()
    except:
        pass

    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # Load checkpoint (only on rank 0, then broadcast)
    # if rank == 0:
    #     try:
    #         state_dict = torch.load("./discriminator/latent_model_weight/convnext2.pt", map_location=device)
    #         model.module.load_state_dict(state_dict, strict=False)
    #         print("Checkpoint loaded.")
    #     except Exception as e:
    #         print("Failed to load checkpoint:", e)
    dist.barrier()

    # Optimizer & Loss
    criterion = TripletLoss(margin=1.0)
    criterion_contrastive = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    generator = Generator()

    # Train
    train_triplet(
        model=model,
        dataloader=dataloader,
        eval_dataset=eval_dataset,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        rank=rank,
        criterion_contrastive=criterion_contrastive,
        generator=generator,
        epochs=10
    )

    # Final eval
    if rank == 0:
        final_acc = evaluate_triplet(model, eval_dataset, device, generator.vae, rank=rank)
        print(f"Final Strict Accuracy: {final_acc:.4f}")

    cleanup()


if __name__ == '__main__':
    main()