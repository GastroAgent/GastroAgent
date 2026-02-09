import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from math import sqrt

class EmbeddingNetwork(nn.Module):
    def __init__(self, pretrained=True, freeze_base=False, model='resnet34', **kwargs):
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
    def __init__(self, pretrained=False, freeze_base=False, model='resnet34', **kwargs):
        super(TripletNetwork, self).__init__()
        if model == 'attention':
            self.embedding = AttentionDownEncoderXL()
        else:
            self.embedding = EmbeddingNetwork(pretrained=pretrained, freeze_base=freeze_base, model=model)

    def encode(self, x, return_image=False, **kwargs):
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
        anchor_emb = self.embedding(anchor)
        positive_emb = self.embedding(positive)
        negative_emb = self.embedding(negative)
        return anchor_emb, positive_emb, negative_emb