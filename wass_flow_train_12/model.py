import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from einops import rearrange
from math import sqrt

from model_utils.cat import CATEncoderLayer

# ------------------------------------------------------------
# 1. 带正交正则化的线性层（核心防坍缩组件）
# ------------------------------------------------------------
class LinearWithOrthoReg(nn.Module):
    """
    标准线性层 + 正交正则化损失（鼓励权重矩阵接近正交，防止映射退化）
    """
    def __init__(self, in_features: int, out_features: int, ortho_lambda: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ortho_lambda = ortho_lambda
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [batch_size, seq_len, in_features]
        Returns:
            output: [batch_size, seq_len, out_features]
            ortho_loss: scalar (regularization loss)
        """
        # 线性变换: y = x @ W^T + b
        output = F.linear(x, self.weight, self.bias)

        # 正交正则化：根据 in/out 维度选择 Gram 矩阵
        if self.in_features <= self.out_features:
            # W: [out, in] → W @ W^T ≈ I_out
            gram = self.weight @ self.weight.t()  # [out, out]
            target = torch.eye(self.out_features, device=self.weight.device)
        else:
            # W: [out, in] → W^T @ W ≈ I_in
            gram = self.weight.t() @ self.weight  # [in, in]
            target = torch.eye(self.in_features, device=self.weight.device)

        # Frobenius 范数惩罚
        ortho_loss = self.ortho_lambda * F.mse_loss(gram, target, reduction='mean')
        return output, ortho_loss

# ------------------------------------------------------------
# 2. 带正交正则化的 FFN 模块
# ------------------------------------------------------------
class FFNWithOrtho(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1, ortho_lambda: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = LinearWithOrthoReg(d_model, dim_feedforward, ortho_lambda)
        self.linear2 = LinearWithOrthoReg(dim_feedforward, d_model, ortho_lambda)

    def forward(self, x: torch.Tensor):
        # First linear + activation
        z1, loss1 = self.linear1(x)
        z1 = F.gelu(z1)  # GELU is fine; no need for invertible activation here
        z1 = self.dropout(z1)

        # Second linear
        z2, loss2 = self.linear2(z1)
        z2 = self.dropout(z2)

        total_ortho_loss = loss1 + loss2
        return z2, total_ortho_loss

# ------------------------------------------------------------
# 3. 带正交约束的 Transformer Encoder Layer
# ------------------------------------------------------------
class TransformerEncoderLayerWithOrtho(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        batch_first: bool = True,
        norm_first: bool = True,
        ortho_lambda: float = 0.1,
    ):
        # 初始化原生 Transformer 层（保留 Attention 和 Norm）
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        # 替换原生 FFN 为带正交正则化的版本
        self.ffn_ortho = FFNWithOrtho(d_model, dim_feedforward, dropout, ortho_lambda)
        self.ortho_loss_weight = 1.0  # 可调整，但通常 ortho_lambda 已控制强度

    def forward(
        self,
        src: torch.Tensor,
        src_mask=None,
        src_key_padding_mask=None,
    ):
        # 复用原生 Attention 逻辑
        if self.norm_first:
            # Self-attention block
            src2 = self.norm1(src)
            src2 = self.self_attn(
                src2, src2, src2,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=False
            )[0]
            src = src + self.dropout1(src2)

            # FFN block with ortho regularization
            src2 = self.norm2(src)
            ffn_out, ortho_loss = self.ffn_ortho(src2)
            src = src + self.dropout2(ffn_out)
        else:
            # Self-attention
            src2 = self.self_attn(
                src, src, src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=False
            )[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)

            # FFN
            ffn_out, ortho_loss = self.ffn_ortho(src)
            src = src + self.dropout2(ffn_out)
            src = self.norm2(src)

        return src, ortho_loss


# ------------------------------------------------------------
# 4. 完整模型：SimpleTransformer + 正交约束
# ------------------------------------------------------------
class SimpleTransformerWithOrtho(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        ortho_lambda: float = 0.1,
        dy=False,
    ):
        super().__init__()
        dim_feedforward = int(embed_dim * mlp_ratio)
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithOrtho(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
                ortho_lambda=ortho_lambda,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        if dy:
            self.layers = nn.ModuleList([convert_ln_to_dyt(x) for x in self.layers])
            self.norm = convert_ln_to_dyt(self.norm)
            
    def forward(self, x: torch.Tensor):
        """
        Returns:
            output: [B, L, D]
            total_ortho_loss: scalar (averaged over layers)
        """
        total_ortho_loss = 0.0
        for layer in self.layers:
            x, ortho_loss = layer(x)
            total_ortho_loss += ortho_loss
        x = self.norm(x)
        return x, total_ortho_loss / len(self.layers)

class EmbeddingNetwork(nn.Module):
    def __init__(self, pretrained=False, freeze_base=False, model='resnet34'):
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

    def forward(self, x, return_hidden=False, **kwargs):
        features = self.base_model(x)
        if return_hidden:
            return self.l2norm(features, p=2, dim=1), features
        else:
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

class TanhTransformer(nn.Module):
    def __init__(self, embed_dim=64, depth=2, num_heads=4, mlp_ratio=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            convert_ln_to_dyt(nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.norm = convert_ln_to_dyt(self.norm)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

import torch
import torch.nn as nn
from timm.layers import LayerNorm2d

class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-6
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"

def convert_ln_to_dyt(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicTanh(module.normalized_shape, not isinstance(module, LayerNorm2d))
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child))
    del module
    return module_output


class UpsampleDecoder(nn.Module):
    def __init__(self, embed_dim=64, patch_size=8, out_chans=4, keep_input_size=False):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # 将嵌入还原为特征图
        self.proj = GatedMLPClassifier(embed_dim, embed_dim, patch_size * patch_size * out_chans, patch_size * patch_size * out_chans)
        self.linear = nn.Linear(patch_size * patch_size * out_chans, 4)
        self.keep_input_size = keep_input_size

    def forward(self, x):
        # x: [B, num_patches, embed_dim]
        x = self.proj(x)  # [B, num_patches, 4*4*4]
        # 重塑为 [B, C, H, W]
        B, N, _ = x.shape
        h = w = int(N ** 0.5)  # 假设是正方形
        
        if self.keep_input_size:
            x = rearrange(
                x,
                'b (h w) (p p c) -> b (h p) (w p) c',
                h=h, w=w, p=self.patch_size
            )
        else:
            x = x.reshape(B, h, w, -1) 
            x = self.linear(x) # return [B, 16, 16, 4]
        return x.permute(0, 3, 1, 2)  # return [B, 4, 16, 16]

class AttentionDownEncoderXL(nn.Module):
    def __init__(self, dy=False, ortho=False, cat=False):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=64, patch_size=4, in_chans=4, embed_dim=512)
        if dy:
            if ortho:
                self.transformer = SimpleTransformerWithOrtho(embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, ortho_lambda=0.05, dy=True)
            else:
                self.transformer = TanhTransformer(embed_dim=512, depth=12, num_heads=8, mlp_ratio=4)
        else:
            if ortho:
                self.transformer = SimpleTransformerWithOrtho(embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, ortho_lambda=0.05, dy=False)
            else:
                self.transformer = SimpleTransformer(embed_dim=512, depth=12, num_heads=8, mlp_ratio=4)
        self.ortho = ortho
        self.dy = dy
        if ortho:
            self.decoder = UpsampleDecoderWithOrtho(embed_dim=512, patch_size=4, out_chans=4)
        else:
            self.decoder = UpsampleDecoder(embed_dim=512, patch_size=4, out_chans=4)
        
        if cat:
            self.transformer.layers = nn.ModuleList([
                CATEncoderLayer(d_model=512, nhead=8, dim_feedforward=4*512) for _ in range(len(self.transformer.layers))
            ])

    def forward(self, x, return_hidden=False):
        # x: [B, 4, 64, 64]
        x = self.patch_embed(x)        # -> [B, 256, 512]
        if self.ortho:
            hidden_states, ortho_loss1 = self.transformer(x) 
        else:
            hidden_states = self.transformer(x)        # -> [B, 256, 512]
        if self.ortho:
            output, ortho_loss2 = self.decoder(hidden_states)
        else:
            output = self.decoder(hidden_states)            # -> [B, 4, 64, 64]
        B = output.shape[0]
        if return_hidden:
            if self.ortho:
                return output.reshape(B, -1), hidden_states, ortho_loss1 + ortho_loss2
            return output.reshape(B, -1), hidden_states

        if self.ortho:
            return output.reshape(B, -1), ortho_loss1 + ortho_loss2
        return output.reshape(B, -1)

class GatedMLPClassifierWithOrtho(nn.Module):
    def __init__(
        self,
        input_dim=4 * 64 * 64,
        hidden_dim1=4096,
        hidden_dim2=1024,
        output_dim=256,
        ortho_lambda=0.01
    ):
        super().__init__()
        self.gate1 = LinearWithOrthoReg(input_dim, hidden_dim1 * 2, ortho_lambda)
        self.gate2 = LinearWithOrthoReg(hidden_dim1, hidden_dim2 * 2, ortho_lambda)
        self.out_proj = LinearWithOrthoReg(hidden_dim2, output_dim, ortho_lambda)

    def forward(self, x: torch.Tensor):
        total_ortho_loss = 0.0

        # Layer 1
        gate_val, loss1 = self.gate1(x)
        total_ortho_loss += loss1
        gate, val = gate_val.chunk(2, dim=-1)
        x = F.silu(gate) * val

        # Layer 2
        gate_val, loss2 = self.gate2(x)
        total_ortho_loss += loss2
        gate, val = gate_val.chunk(2, dim=-1)
        x = F.silu(gate) * val

        # Output
        x, loss3 = self.out_proj(x)
        total_ortho_loss += loss3

        return x, total_ortho_loss


# ------------------------------------------------------------
# 带正则化的 UpsampleDecoder
# ------------------------------------------------------------
class UpsampleDecoderWithOrtho(nn.Module):
    def __init__(self, embed_dim=64, patch_size=8, out_chans=4, ortho_lambda=0.01, keep_input_size=False):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_chans = out_chans
        self.keep_input_size = keep_input_size

        # 使用带正则化的 MLP
        self.proj = GatedMLPClassifierWithOrtho(
            input_dim=embed_dim,
            hidden_dim1=patch_size * patch_size * out_chans * 2,  # 可调
            hidden_dim2=patch_size * patch_size * out_chans,
            output_dim=patch_size * patch_size * out_chans,
            ortho_lambda=ortho_lambda
        )
        if keep_input_size:
            self.final_linear = LinearWithOrthoReg(
                in_features=patch_size * patch_size * out_chans,
                out_features=patch_size * patch_size * out_chans,
                ortho_lambda=ortho_lambda
            )
        else:
            self.final_linear = LinearWithOrthoReg(
                in_features=patch_size * patch_size * out_chans,
                out_features=out_chans,
                ortho_lambda=ortho_lambda
            )

    def forward(self, x: torch.Tensor):
        """
        x: [B, num_patches, embed_dim]
        Returns:
            output: [B, 4, H, W]  (H=W=sqrt(num_patches)*patch_size)
            total_ortho_loss: scalar
        """
        B, N, _ = x.shape
        h = w = int(N ** 0.5)
        assert h * w == N, "Number of patches must be a perfect square"

        # MLP projection
        x, ortho_loss1 = self.proj(x)  # [B, N, P*P*C]

        # Reshape to [B, h, w, P*P*C]
        if self.keep_input_size:
            x = rearrange(
                x,
                'b (h w) (p p c) -> b (h p) (w p) c',
                h=h, w=w, p=self.patch_size
            )
        else:
            x = x.reshape(B, h, w, -1) 
            x = self.linear(x) # return [B, 16, 16, 4]

            # Final linear per-pixel
            x_flat = x.view(B * h * w, -1)  # [B*h*w, P*P*C]
            x_flat, ortho_loss2 = self.final_linear(x_flat)  # [B*h*w, 4]
            x = x_flat.view(B, h, w, 4)

        total_ortho_loss = ortho_loss1 + ortho_loss2
        return x.permute(0, 3, 1, 2), total_ortho_loss  # [B, 4, h, w]

class GatedMLPClassifier(nn.Module):
    def __init__(self, input_dim=4 * 64 * 64, hidden_dim1=4096, hidden_dim2=1024, output_dim=256):
        super().__init__()
        # 第一层：门控 MLP（SwiGLU 风格）
        self.gate1 = nn.Linear(input_dim, hidden_dim1 * 2)  # 输出拼接 [gate | value]
        # 第二层
        self.gate2 = nn.Linear(hidden_dim1, hidden_dim2 * 2)
        # 输出层（可选是否门控；这里简化为普通线性）
        self.out_proj = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        # Layer 1
        gate_val = self.gate1(x)
        gate, val = gate_val.chunk(2, dim=-1)  # split into two halves
        x = nn.functional.silu(gate) * val     # SwiGLU: SiLU(gate) ⊙ value

        # Layer 2
        gate_val = self.gate2(x)
        gate, val = gate_val.chunk(2, dim=-1)
        x = nn.functional.silu(gate) * val

        # Output projection
        x = self.out_proj(x)
        return x

class TripletNetwork(nn.Module):
    def __init__(self, pretrained=False, freeze_base=False, model='resnet34', dy=False, ortho=False, cat=False):
        super(TripletNetwork, self).__init__()
        if model == 'attention':
            self.embedding = AttentionDownEncoderXL(dy, ortho, cat)
        else:
            self.embedding = EmbeddingNetwork(pretrained=pretrained, freeze_base=freeze_base, model=model)
        self.dy = dy
        self.ortho = ortho
        self.model = model

    def encode(self, x, return_image=False, return_hidden=False):
        if self.ortho:
            if return_hidden:
                embeds, hidden_state, ortho_loss = self.embedding(x, return_hidden)
            else:
                embeds, ortho_loss = self.embedding(x, return_hidden)
        else:
            embeds = self.embedding(x, return_hidden)

            if return_hidden:
                hidden_state = embeds[1]
                embeds = embeds[0]
            
        if embeds.ndim == 2:
            B, D = embeds.shape
            if return_image:
                H = W = int(sqrt(D // 4))
                embeds = embeds.view(B, 4, H, W)

        elif embeds.ndim == 4 :
            B, C, H, W = embeds.shape
            if return_image:
                embeds = embeds 
            else:
                embeds = embeds.view(B, -1)
        
        if return_hidden:
            if self.ortho:
                return embeds, hidden_state, ortho_loss
            return embeds, hidden_state
        else:
            if self.ortho:
                return embeds, ortho_loss
            return embeds
        
    def forward(self, anchor, positive, negative):
        anchor_emb = self.encode(anchor, True)
        positive_emb = self.encode(positive, True)
        negative_emb = self.encode(negative, True)
        return anchor_emb, positive_emb, negative_emb

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, stride=stride)
        self.norm1 = nn.GroupNorm(8, out_channels)  # Stable with varying batch sizes
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.GroupNorm(8, out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x += residual
        return self.relu(x)

class AttentionBlock(nn.Module):
    """Simplified channel-wise attention (SE-style)"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class LatentDecoder(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, hidden_dim=64, use_attention=True):
        """
        Enhanced probabilistic decoder for VAE latent reconstruction.
        Supports (16x16) or (64x64) input → always outputs (64x64).
        Now with residual blocks, attention, better upsampling, and separate μ/logvar heads.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        # Stem: deeper with residual
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, 3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.ReLU(inplace=True),
            ResidualBlock(hidden_dim // 2, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim)
        )

        # Upsampling path (for 16x16 input): use PixelShuffle + residual refinement
        self.up1 = nn.Sequential(
            nn.PixelShuffle(2),  # (B, hidden_dim, 16,16) → (B, hidden_dim//4, 32,32)
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.ReLU(inplace=True),
            ResidualBlock(hidden_dim // 2, hidden_dim // 2)
        )
        self.up2 = nn.Sequential(
            nn.PixelShuffle(2),  # → (B, hidden_dim//8, 64,64)
            nn.Conv2d(hidden_dim // 8, hidden_dim // 4, 3, padding=1),
            nn.GroupNorm(8, hidden_dim // 4),
            nn.ReLU(inplace=True),
            ResidualBlock(hidden_dim // 4, hidden_dim // 4)
        )

        # Direct path (for 64x64): just refine with residual blocks
        self.direct_path = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim // 2),
            ResidualBlock(hidden_dim // 2, hidden_dim // 4)
        )

        # Optional attention before head
        if use_attention:
            self.attn = AttentionBlock(hidden_dim // 4)

        # Separate heads for μ and logσ (more stable than chunking)
        self.mu_head = nn.Conv2d(hidden_dim // 4, out_channels, kernel_size=1)
        self.logvar_head = nn.Conv2d(hidden_dim // 4, out_channels, kernel_size=1)

        # Initialize heads to small values to avoid instability
        nn.init.normal_(self.mu_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.mu_head.bias, 0.0)
        nn.init.normal_(self.logvar_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.logvar_head.bias, 0.0)

    def forward(self, h):
        B, C, H, W = h.shape
        assert C == self.in_channels, f"Expected {self.in_channels} input channels, got {C}"
        assert (H, W) in [(16, 16), (64, 64)], f"Unsupported input size: {H}x{W}"

        x = self.stem(h)  # (B, hidden_dim, H, W)

        if H == 16:
            # 16 → 32 → 64 with learnable upsampling
            x = self.up1(x)      # (B, hidden_dim//2, 32, 32)
            x = self.up2(x)      # (B, hidden_dim//4, 64, 64)
        else:
            # 64x64: direct refinement
            x = self.direct_path(x)  # (B, hidden_dim//4, 64, 64)

        if self.use_attention:
            x = self.attn(x)

        mu = self.mu_head(x)
        logvar = self.logvar_head(x)

        # Optional: clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10, max=5)

        return mu + logvar
    
if __name__ == '__main__':
    model = TripletNetwork(pretrained=False, freeze_base=False, model='attention', dy=True, cat=True).to("cuda") 
    print(model)
    x = torch.randn(8, 4, 64, 64).to("cuda")
    y = model.encode(x)