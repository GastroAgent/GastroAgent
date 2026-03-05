import glob
import json
import math
from tqdm import tqdm
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('./GasAgent-main/discriminator')

import random
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from sklearn.utils import shuffle
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from utils_.data_loader import MedicalTripletJsonDataset
from utils_.train_utils import infiniteloop
from torch.utils.tensorboard import SummaryWriter
from diffusers import AutoencoderKL

from PIL import Image
from math import sqrt
import numpy as np
from PIL import Image

class EmbeddingNetwork(nn.Module):
    def __init__(self, pretrained=True, freeze_base=False, model='resnet34'):
        super(EmbeddingNetwork, self).__init__()
        if model == 'resnet34':
            self.base_model = models.resnet34(pretrained=pretrained)
        else:
            self.base_model = models.resnet101(pretrained=pretrained)
        
        self.base_model.conv1 = nn.Conv2d(4, self.base_model.conv1.out_channels, kernel_size=3, stride=1, padding=1,
                                          bias=False)
        # 修改输出为 256 维特征向量
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 1024)
        
        # 可选：冻结预训练模型参数
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # L2 归一化
        self.l2norm = nn.functional.normalize

    def forward(self, x):
        features = self.base_model(x)
        features = self.l2norm(features, p=2, dim=1)  # L2 归一化
        return features

class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_chans=4, embed_dim=64):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, 4, 64, 64]
        x = self.proj(x)  # -> [B, embed_dim, 8, 8]
        x = x.flatten(2)  # [B, embed_dim, 64]
        x = x.transpose(1, 2)  # [B, 64, embed_dim]
        return x  # [B, num_patches, embed_dim]


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
        # 将嵌入还原为特征图
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_chans)

    def forward(self, x):
        # x: [B, num_patches, embed_dim]
        x = self.proj(x)  # [B, num_patches, 8*8*4]
        # 重塑为 [B, C, H, W]
        B, N, _ = x.shape
        h = w = int(N ** 0.5)  # 假设是正方形
        x = x.reshape(B, h, w, self.patch_size, self.patch_size, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, -1, h * self.patch_size, w * self.patch_size)
        return x  # [B, 4, 64, 64]

class AttentionDownEncoderXL(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=64, patch_size=4, in_chans=4, embed_dim=512)
        self.transformer = SimpleTransformer(embed_dim=512, depth=16, num_heads=8, mlp_ratio=4)
        self.decoder = UpsampleDecoder(embed_dim=512, patch_size=2, out_chans=4)

    def forward(self, x):
        # x: [B, 4, 64, 64]
        x = self.patch_embed(x)        # -> [B, 64, 64]
        x = self.transformer(x)        # -> [B, 64, 64]
        x = self.decoder(x)            # -> [B, 4, 64, 64]
        B = x.shape[0]
        return x.reshape(B, -1)

class TripletNetwork(nn.Module):
    def __init__(self, pretrained=True, freeze_base=False, model='resnet34'):
        super(TripletNetwork, self).__init__()

        self.processor = None
        if model == 'attention':
            self.embedding = AttentionDownEncoderXL()
        elif model == 'convnext':
            from transformers import AutoImageProcessor, DINOv3ConvNextModel
            pretrained_model_name = "/mnt/inaisfs/data/home/tansy_criait/weights/dinov3-convnext-tiny"
            self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
            self.embedding = DINOv3ConvNextModel.from_pretrained(
                pretrained_model_name, 
            )
            # inputs.data['pixel_values']
            in_channel = 4
            output_channel =  self.embedding.stages[0].downsample_layers[0].out_channels 
            stride = (2, 2)
            kernel_size = (2, 2)
            padding = self.embedding.stages[0].downsample_layers[0].padding 
            self.embedding.stages[0].downsample_layers[0] = nn.Conv2d(in_channel, output_channel, stride=stride, kernel_size=kernel_size, padding=padding)
            self.embedding = self.embedding.to("cuda")
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
            
        elif embeds.ndim == 4 :
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

# class TripletLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super(TripletLoss, self).__init__()
#         self.margin = margin

#     def forward(self, anchor, positive, negative):
#         s_pos = torch.einsum('nd,nd->n', anchor, positive)  # torch.einsum('nd,nd->n', anchor, positive) 
#         s_neg = torch.einsum('nd,nd->n', anchor, negative) 
    
#         d_pos = 1 - s_pos
#         d_neg = 1 - s_neg

#         # Triplet Loss 公式
#         loss = torch.mean(torch.clamp(d_pos - d_neg + self.margin, min=0.0))
#         return loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # # 计算距离
        # d_pos = torch.norm(anchor - positive, p=2, dim=1)  # Anchor-Positive 距离
        # d_neg = torch.norm(anchor - negative, p=2, dim=1)  # Anchor-Negative 距离

        # 计算余弦相似度
        pos_cosine_similarity = F.cosine_similarity(anchor, positive, dim=1)
        d_pos = 1 - pos_cosine_similarity
        neg_cosine_similarity = F.cosine_similarity(anchor, negative, dim=1)
        d_neg = 1 - neg_cosine_similarity

        # Triplet Loss 公式
        loss = torch.mean(torch.clamp(d_pos - d_neg + self.margin, min=0.0))
        return loss

# class ContrastiveLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin

#     def forward(self, anchor, other, target):
#         score = torch.einsum('nd,nd->n', anchor, other) 
#         distance = 1 - score
#         loss = torch.mean(
#             target * distance + (1 - target) * torch.clamp(self.margin - distance, min=0.0)
#         )
#         return loss

def clip_style_triplet_loss(anchor_emb, positive_emb, negative_emb, temperature=0.07):
    """
    CLIP-style loss using triplet data.
    
    Args:
        anchor_emb:   [B, D]
        positive_emb: [B, D]
        negative_emb: [B, D]
        temperature:  float, softmax temperature (default 0.07 as in CLIP)
    
    Returns:
        loss: scalar tensor
    """
    B, D = anchor_emb.shape

    # L2 normalize embeddings (as in CLIP)
    anchor_emb = F.normalize(anchor_emb, dim=1)
    positive_emb = F.normalize(positive_emb, dim=1)
    negative_emb = F.normalize(negative_emb, dim=1)

    # Compute similarity matrix: [B, B]
    # S[i, j] = anchor[i] · negative[j]  for all i, j
    sim_matrix = anchor_emb @ negative_emb.t()  # [B, B]

    # Replace diagonal with anchor[i] · positive[i]
    pos_sim = torch.sum(anchor_emb * positive_emb, dim=1)  # [B]
    sim_matrix = sim_matrix.clone()
    sim_matrix[range(B), range(B)] = pos_sim

    # Scale by temperature
    logits = sim_matrix / temperature  # [B, B]

    # Labels: each row i should pick column i as positive → label = i
    labels = torch.arange(B, device=anchor_emb.device)

    # Cross-entropy loss (InfoNCE)
    loss = F.cross_entropy(logits, labels)

    return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, other, target):
        """
        - anchor: anchor 嵌入向量 (B, D)
        - other: 正样本/负样本 嵌入向量 (B, D)
        - target: 0 或 1，1 表示正样本对，0 表示负样本对
        """
        
        # distance = torch.norm(anchor - other, p=2, dim=1)  # 欧氏距离

        # 计算余弦相似度
        cosine_similarity = F.cosine_similarity(anchor, other, dim=1)
        # 转换为余弦距离：1 - 相似度
        distance = 1 - cosine_similarity
        
        loss = torch.mean(
            target * distance + (1 - target) * torch.clamp(self.margin - distance, min=0.0)
        )
        return loss


def sinkhorn_loss(bx, bx1, epsilon=0.1, n_iter=25, reduction='mean'):
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
    
def cal_wasserstein_loss(x, x1, **kwargs):
    wass_loss = sinkhorn_loss(x, x1, **kwargs)
    # wass_loss = wass_loss.sum()
    return wass_loss

class CutOut:
    def __init__(self, length=16, p=0.5):
        """
        Args:
            length (int): 遮挡区域的边长
            p (float): 应用 CutOut 的概率 (0 <= p <= 1)
        """
        self.length = length
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): 输入图像，形状为 (H, W, C)
        Returns:
            PIL Image: 应用 CutOut 后的图像
        """
        if np.random.rand() > self.p:  # 按概率决定是否应用
            return img

        img_array = np.array(img)
        h, w = img_array.shape[:2]

        # 随机选择遮挡中心
        y = np.random.randint(h)
        x = np.random.randint(w)

        # 计算遮挡区域的边界
        y1 = max(0, y - self.length // 2)
        y2 = min(h, y + self.length // 2)
        x1 = max(0, x - self.length // 2)
        x2 = min(w, x + self.length // 2)

        # 遮挡操作
        img_array[y1:y2, x1:x2, :] = 0  # 三通道置零
        return Image.fromarray(img_array)

class Generator():
    def __init__(self):
        self.device = "cuda"
        self.vae = AutoencoderKL.from_pretrained(
                './whole_wass_flow_match/flow_matcher_otcfm/vae_our').to(
                device=self.device).eval()

if __name__ == '__main__':
    # 加载数据集
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    transform_B = transforms.Compose([
        transforms.Resize((512, 512)), 
        transforms.RandomHorizontalFlip(p=0.25), 
        transforms.RandomVerticalFlip(p=0.25),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),], p=0.1), 
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 0.25))], p=0.1), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = MedicalTripletJsonDataset(
            path="./train/train_gan_v2/step_score_generated/ssim_train_triplet_all_dataset.json",
            # transform=transform,
            transform=transform_B,
    )
    
    dataloader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            drop_last=True,
    )
    
    eval_dataset = MedicalTripletJsonDataset(
        path="./train/train_gan_v2/step_score_generated/ssim_eval_triplet_all_dataset.json",
        # transform=transform,
        transform=transform_B,
    )
    
    # 初始化模型和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TripletNetwork(pretrained=True, freeze_base=False, model='convnext').to(device)
    try:
        model.embedding.post_init()
        pass
    except:
        pass

    criterion = TripletLoss(margin=1.0)
    criterion_contrastive = ContrastiveLoss(margin=1.0)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    generator = Generator()
    # generator = None
    writer = SummaryWriter(log_dir="./whole_wass_flow_match/discriminator/logs")
    def train_triplet(model, dataloader, criterion, optimizer, device="cuda",
                      criterion_contrastive=None, generator=None, cal_wasserstein_loss=None, epochs=20):
        if generator is None:
            raise NotImplementedError
        else:
            vae = generator.vae
            
        total_step = epochs * len(dataloader)
        best_acc = accuracy = evaluate_triplet(model, eval_dataset, device, vae)
        # best_acc = accuracy = 0
        writer.add_scalar('Evaling/Acc', accuracy, 0)
        dataloader = infiniteloop(dataloader)
        model.train()
        for step in tqdm(range(1, total_step)):
            model.train()
            optimizer.zero_grad()
            batch = dataloader.__iter__().__next__()

            anchor, positive, negative = batch["anchor"], batch["positive"], batch["negative"]
            anchor, positive, negative = anchor.to(vae.device), positive.to(vae.device), negative.to(vae.device)
            # 压缩至 潜在空间
            with torch.no_grad():
                anchor = vae.encode(anchor).latent_dist
                anchor = anchor.sample() * 0.18215
                positive = vae.encode(positive).latent_dist
                positive = positive.sample() * 0.18215
                negative = vae.encode(negative).latent_dist
                negative = negative.sample() * 0.18215

            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
            loss = criterion(anchor_emb, positive_emb, negative_emb)

            # if criterion_contrastive is not None:
            #     # 2. Contrastive Loss（直接从三元组中构造样本对）
            #     # 正样本对 (anchor, positive)
            #     loss_pos = criterion_contrastive(anchor_emb, positive_emb, torch.ones_like(anchor_emb[:, 0]))  # y=1

            #     # 负样本对 (anchor, negative)
            #     loss_neg = criterion_contrastive(anchor_emb, negative_emb, torch.zeros_like(anchor_emb[:, 0]))  # y=0

            #     loss = loss + (loss_pos + loss_neg) / 2

            # if cal_wasserstein_loss is not None:
            #     beta = 100
            #     loss = loss + torch.clamp(beta * cal_wasserstein_loss(anchor_emb, positive_emb, reduction='none') - 2 * beta * cal_wasserstein_loss(anchor_emb, negative_emb, reduction='none') + 2.0,  min=0.0).mean()
            loss = 0.5 * loss + 0.5 * clip_style_triplet_loss(anchor_emb, positive_emb, negative_emb)
            
            emb_norm_loss = ((anchor_emb.norm(dim=1) - 1.0) ** 2 + (positive_emb.norm(dim=1) - 1.0) ** 2 + (negative_emb.norm(dim=1) - 1.0) ** 2).mean() # 惩罚偏离单位长度
            loss = loss + 0.25 * emb_norm_loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.5)
            optimizer.step()

            if step % 10 == 0:
                print(f"Global Step {step}/{total_step} | Loss: {loss.item():.4f} | Grad Norm: {grad_norm.item():.4f}")
            writer.add_scalar('Training/Loss', loss.item(), step)

            if (step + 1) % 5000 == 0:
                print("Eval...")
                os.makedirs("./whole_wass_flow_match/discriminator/latent_model_weight", exist_ok=True)
                accuracy = evaluate_triplet(model, eval_dataset, device, vae)
                if best_acc < accuracy:
                    torch.save(model.state_dict(), f"./discriminator/latent_model_weight/convnext5.pt")
                    best_acc = accuracy
                else:
                    pass
                
                writer.add_scalar('Evaling/Acc', accuracy, step)

    # 加载 checkpoints
    try:
        state_dict = torch.load("./discriminator/latent_model_weight/convnext5.pt", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    except:
        pass
    
    model = model.to(device)
    def evaluate_triplet(model, dataset, device, vae):
        model.eval()
        correct = 0
        total = 0
        strict_correct = 0
        with torch.no_grad():
            for batch in tqdm(dataset):
                total += 1
                anchor, positive, negative = batch["anchor"], batch["positive"], batch["negative"]
                anchor, positive, negative = anchor.to(vae.device), positive.to(vae.device), negative.to(vae.device)
                # 压缩至 潜在空间
                with torch.no_grad():
                    anchor = vae.encode(anchor.unsqueeze(0)).latent_dist
                    anchor = anchor.sample() * 0.18215
                    positive = vae.encode(positive.unsqueeze(0)).latent_dist
                    positive = positive.sample() * 0.18215
                    negative = vae.encode(negative.unsqueeze(0)).latent_dist
                    negative = negative.sample() * 0.18215

                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
                pos_score =  (anchor_emb @ positive_emb.T).item()
                neg_score =  (anchor_emb @ negative_emb.T).item()
                if pos_score > neg_score and neg_score < 0.2 and pos_score > 0.8:
                # if pos_score > neg_score:
                    strict_correct += 1

                if pos_score > neg_score and (pos_score - neg_score) > 0.1:
                # if pos_score > neg_score:
                    correct += 1
        accuracy = correct / total
        strict_accuracy = strict_correct / total
        print(f"Evaluation Accuracy: {accuracy:.4f}")
        print(f"Strict Evaluation Accuracy: {strict_accuracy:.4f}")
        return strict_accuracy
    
    train_triplet(model, dataloader, criterion=criterion, optimizer=optimizer, device=device,
                  generator=generator, criterion_contrastive=criterion_contrastive,
                  cal_wasserstein_loss=None, epochs=15) 

    evaluate_triplet(model, eval_dataset, device, generator.vae)
