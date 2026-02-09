import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.l2norm = nn.functional.normalize

    def forward(self, anchor, positive, negative):
        # 计算距离
        # anchor = self.l2norm(anchor, p=2, dim=1)
        # positive = self.l2norm(positive, p=2, dim=1)
        # negative = self.l2norm(negative, p=2, dim=1)
        d_pos = torch.norm(anchor - positive, p=2, dim=1)  # Anchor-Positive 距离
        d_neg = torch.norm(anchor - negative, p=2, dim=1)  # Anchor-Negative 距离

        # Triplet Loss 公式
        loss = torch.mean(torch.clamp(d_pos - d_neg + self.margin, min=0.0))
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
        # anchor = self.l2norm(anchor, p=2, dim=1)
        # other = self.l2norm(other, p=2, dim=1)
        distance = torch.norm(anchor - other, p=2, dim=1)  # 欧氏距离
        loss = torch.mean(
            target * distance + (1 - target) * torch.clamp(self.margin - distance, min=0.0)
        )
        return loss

from geomloss import SamplesLoss
# loss_fn = SamplesLoss("laplacian", p=2, blur=0.1**0.5)
loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.1**0.5)
# sinkhorn hausdorff energy gaussian laplacian
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
    loss = loss_fn(bx, bx1)
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