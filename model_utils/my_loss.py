import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from geomloss import SamplesLoss

loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.1**0.5, scaling=.65)

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, matched = False, norm = False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.matched = matched
        self.norm = norm

    def forward(self, anchor, positive, negative, pos_beta=0.2, neg_beta=1.0, use_wass=False):
        # 计算 Wass 距离
        if use_wass:
            d_pos = cal_wasserstein_loss(anchor, positive, matched=self.matched, norm=self.norm)  # Anchor-Positive 距离
            d_neg = cal_wasserstein_loss(anchor, negative, matched=self.matched, norm=self.norm)  # Anchor-Negative 距离
        else:
            d_pos = cal_l2_loss(anchor, positive, matched=self.matched, norm=self.norm)  # Anchor-Positive 距离 
            d_neg = cal_l2_loss(anchor, negative, matched=self.matched, norm=self.norm)  # Anchor-Negative 距离
        
        # Triplet Loss 公式
        loss = torch.mean(torch.clamp(pos_beta * d_pos - neg_beta * d_neg + self.margin, min=0.0))
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, matched = False, norm = False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.matched = matched
        self.norm = norm
        
    def forward(self, anchor, other, target, beta=1, use_wass=True):
        """
        - anchor: anchor 嵌入向量 (B, D)
        - other: 正样本/负样本 嵌入向量 (B, D)
        - target: 0 或 1，1 表示正样本对，0 表示负样本对
        """
        if use_wass:
            distance = cal_wasserstein_loss(anchor, other, matched=self.matched, norm=self.norm) * beta  # Wass 距离
        else:
            distance = cal_l2_loss(anchor, other, matched=self.matched, norm=self.norm) 
        loss = torch.mean(
            target * distance + (1 - target) * torch.clamp(self.margin - distance, min=0.0)
        )
        return loss

def sinkhorn_custom_loss(bx, bx1, epsilon=0.1, n_iter=25, reduction='sum'):
    """
    Compute Sinkhorn loss (approximate Wasserstein distance) between two sets of samples.
    
    Args:
        bx (Tensor): [B, N, D] or [B, C, H, W] or [B, D]
        bx1 (Tensor): [B, N, D] or [B, C, H, W] or [B, D]
        epsilon (float): Entropy regularization strength
        n_iter (int): Number of Sinkhorn iterations
        reduction (str): 'mean' or 'sum' for batch reduction

    Returns:
        Tensor: Scalar loss
    """
    # Reshape input to [B, N, D]
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
    
def sinkhorn_loss(bx, bx1, epsilon=0.1, n_iter=50, reduction='mean', **kwargs):
    """
    Compute Sinkhorn loss (approximate Wasserstein distance) between two sets of samples.
    
    Args:
        bx (Tensor): [B, N, D] or [B, C, H, W] or [B, D]
        bx1 (Tensor): [B, N, D] or [B, C, H, W] or [B, D]
        epsilon (float): Entropy regularization strength
        n_iter (int): Number of Sinkhorn iterations
        reduction (str): 'mean' or 'sum' for batch reduction

    Returns:
        Tensor: Scalar loss
    """
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
    loss = loss_fn(bx, bx1) * H * W
    return loss

def cal_l2_loss(anchor_emb, positive_emb, matched=False, norm=False, **kwargs):
    if norm:
        anchor_emb = anchor_emb / anchor_emb.norm(dim=1).unsqueeze(1)
        positive_emb = positive_emb / positive_emb.norm(dim=1).unsqueeze(1)
    if matched:
        with torch.no_grad():
            # 计算成对平方距离 [B, B]
            diff = anchor_emb.unsqueeze(1) - positive_emb.unsqueeze(0)  # [B, B, D]
            pairwise_sq_dist = torch.sum(diff ** 2, dim=-1)

            # 可选：防止自匹配（如果 anchor[i] 和 positive[i] 是同一实例）
            # pairwise_sq_dist.fill_diagonal_(float('inf'))

            # 找到每个 anchor 最近的 positive 索引
            min_indices = torch.argmin(pairwise_sq_dist, dim=1)  # [B]

        # 使用索引提取匹配的 positive（此操作可导，因为索引是常量）
        matched_positive = positive_emb[min_indices] 
    else:
        matched_positive = positive_emb
    
    distance = torch.norm(anchor_emb - matched_positive, p=2, dim=1)
    return distance

def cal_wasserstein_loss(anchor_emb, positive_emb, matched=False, norm=False, **kwargs):
    if norm:
        anchor_emb = anchor_emb / anchor_emb.norm(dim=1).unsqueeze(1)
        positive_emb = positive_emb / positive_emb.norm(dim=1).unsqueeze(1)
    if matched:
        with torch.no_grad():
            # 计算成对平方距离 [B, B]
            diff = anchor_emb.unsqueeze(1) - positive_emb.unsqueeze(0)  # [B, B, D]
            pairwise_sq_dist = torch.sum(diff ** 2, dim=-1)

            # 可选：防止自匹配（如果 anchor[i] 和 positive[i] 是同一实例）
            # pairwise_sq_dist.fill_diagonal_(float('inf'))

            # 找到每个 anchor 最近的 positive 索引
            min_indices = torch.argmin(pairwise_sq_dist, dim=1)  # [B]

        # 使用索引提取匹配的 positive（此操作可导，因为索引是常量）
        matched_positive = positive_emb[min_indices] 
    else:
        matched_positive = positive_emb
        
    # wass_loss = sinkhorn_custom_loss(anchor_emb, matched_positive, **kwargs)
    wass_loss = sinkhorn_loss(anchor_emb, matched_positive, **kwargs)
    return wass_loss

def min_match_mse_loss(anchor_emb, positive_emb):
    """
    每个 anchor 在 positive_emb 中找到最近邻（无梯度），
    然后仅对匹配对计算可导的 MSE 损失。
    """
    anchor_emb = anchor_emb.reshape(anchor_emb.shape[0], -1)
    positive_emb = positive_emb.reshape(positive_emb.shape[0], -1)
    B, D = anchor_emb.shape

    with torch.no_grad():
        # 计算成对平方距离 [B, B]
        diff = anchor_emb.unsqueeze(1) - positive_emb.unsqueeze(0)  # [B, B, D]
        pairwise_sq_dist = torch.sum(diff ** 2, dim=-1)

        # 可选：防止自匹配（如果 anchor[i] 和 positive[i] 是同一实例）
        # pairwise_sq_dist.fill_diagonal_(float('inf'))

        # 找到每个 anchor 最近的 positive 索引
        min_indices = torch.argmin(pairwise_sq_dist, dim=1)  # [B]

    # 使用索引提取匹配的 positive（此操作可导，因为索引是常量）
    matched_positive = positive_emb[min_indices]  # [B, D]

    # 计算 MSE 损失（只对 anchor 和 matched_positive 优化）
    loss = F.mse_loss(anchor_emb, matched_positive, reduction='sum')
    return loss

def clip_loss(anchor, other):
    temperature = torch.Tensor([0.07])
    logits_scale = torch.log(1 / temperature)
    N = anchor.shape[0]
    logits_per_image = anchor @ other.T  # [N, N]
    logits_per_text = other @ anchor.T
    # 创建标签：对角线上的位置是正样本对 (i, i)
    labels = torch.arange(N, device=logits_per_image.device)

    # 图像作为查询，文本作为键：第 i 个图像应匹配第 i 个文本
    loss_i2t = F.cross_entropy(logits_per_image, labels)

    # 文本作为查询，图像作为键：第 i 个文本应匹配第 i 个图像
    loss_t2i = F.cross_entropy(logits_per_text, labels)

    # 对称损失：两个方向的平均
    loss = (loss_i2t + loss_t2i) / 2
    return loss

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

# ==============================
# Hinge Loss Functions
# ==============================
def hinge_discriminator_loss(real_pred, fake_pred):
    d_real_loss = torch.mean(torch.relu(1.0 - real_pred))
    d_fake_loss = torch.mean(torch.relu(1.0 + fake_pred))
    return d_real_loss + d_fake_loss

def hinge_generator_loss(fake_pred):
    return - torch.mean(fake_pred)