"""
================================================================================
HALT V2 探针模型架构
================================================================================
包含多种改进的探针模型架构：
1. AttentionMLPProbe: 带注意力机制的MLP
2. MultiLayerFeatureExtractor: 多层特征提取器
3. ContrastiveLearningHead: 对比学习头
4. EnsembleProbe: 集成探针模型
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import math


# ===== 1. 多头注意力模块 =====
class MultiHeadAttention(nn.Module):
    """多头注意力机制，用于融合多层特征"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim) 或 (batch_size, embed_dim)
        Returns:
            (batch_size, seq_len, embed_dim) 或 (batch_size, embed_dim)
        """
        # 处理2D输入
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, embed_dim)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, seq_len, _ = x.shape

        # QKV投影
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 加权求和
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        out = self.proj(out)

        if squeeze_output:
            out = out.squeeze(1)

        return out


# ===== 2. 注意力MLP探针模型 =====
class AttentionMLPProbe(nn.Module):
    """
    带注意力机制的MLP探针模型
    用于融合多层特征并预测幻觉风险
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        num_attention_heads: int = 4,
        dropout: float = 0.2,
        use_layer_norm: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        # 多头注意力
        self.attention = MultiHeadAttention(
            embed_dim=hidden_dims[0],
            num_heads=num_attention_heads,
            dropout=dropout
        )

        # MLP层
        self.mlp_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.mlp_layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.LayerNorm(hidden_dims[i + 1]) if use_layer_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )

        # 残差连接的投影层
        if use_residual and hidden_dims[0] != hidden_dims[-1]:
            self.residual_proj = nn.Linear(hidden_dims[0], hidden_dims[-1])
        else:
            self.residual_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim) 或 (batch_size, num_layers, input_dim)
        Returns:
            risk_score: (batch_size,) 风险分数 [0, 1]
        """
        # 输入投影
        x = self.input_proj(x)

        # 注意力机制
        attn_out = self.attention(x)

        # 残差连接
        if self.use_residual:
            residual = x
        else:
            residual = None

        # MLP层
        out = attn_out
        for layer in self.mlp_layers:
            out = layer(out)

        # 残差连接
        if self.use_residual and residual is not None:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            out = out + residual

        # 输出层
        risk_score = self.output_layer(out).squeeze(-1)

        return risk_score


# ===== 3. 多层特征提取器 =====
class MultiLayerFeatureExtractor(nn.Module):
    """
    多层特征提取器
    从模型的多个层提取特征并融合
    """

    def __init__(
        self,
        layer_positions: List[float] = [0.25, 0.5, 0.75],
        feature_types: Dict[str, bool] = None,
        aggregation: str = "concat"  # "concat", "mean", "attention"
    ):
        super().__init__()
        self.layer_positions = layer_positions
        self.feature_types = feature_types or {
            "hidden_states": True,
            "attention_weights": False,
            "token_entropy": False,
        }
        self.aggregation = aggregation

    def extract_features(
        self,
        model_outputs: Dict,
        num_layers: int
    ) -> torch.Tensor:
        """
        从模型输出中提取多层特征

        Args:
            model_outputs: 模型输出字典，包含hidden_states等
            num_layers: 模型总层数

        Returns:
            features: 提取的特征张量
        """
        features_list = []

        # 计算目标层索引
        target_layers = [int(pos * num_layers) for pos in self.layer_positions]

        # 提取隐藏状态
        if self.feature_types.get("hidden_states", True):
            hidden_states = model_outputs.get("hidden_states", None)
            if hidden_states is not None:
                for layer_idx in target_layers:
                    # hidden_states: (batch_size, seq_len, hidden_dim)
                    layer_hidden = hidden_states[layer_idx]
                    # 取最后一个token的隐藏状态
                    layer_feature = layer_hidden[:, -1, :]
                    features_list.append(layer_feature)

        # 提取注意力权重特征
        if self.feature_types.get("attention_weights", False):
            attentions = model_outputs.get("attentions", None)
            if attentions is not None:
                for layer_idx in target_layers:
                    # attentions: (batch_size, num_heads, seq_len, seq_len)
                    layer_attn = attentions[layer_idx]
                    # 计算注意力统计特征
                    attn_mean = layer_attn.mean(dim=(1, 2, 3))  # (batch_size,)
                    attn_std = layer_attn.std(dim=(1, 2, 3))
                    attn_max = layer_attn.max(dim=-1)[0].max(dim=-1)[0].mean(dim=1)
                    attn_features = torch.stack([attn_mean, attn_std, attn_max], dim=1)
                    features_list.append(attn_features)

        # 提取token熵特征
        if self.feature_types.get("token_entropy", False):
            logits = model_outputs.get("logits", None)
            if logits is not None:
                # logits: (batch_size, seq_len, vocab_size)
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                # 统计特征
                entropy_mean = entropy.mean(dim=1, keepdim=True)
                entropy_std = entropy.std(dim=1, keepdim=True)
                entropy_max = entropy.max(dim=1, keepdim=True)[0]
                entropy_features = torch.cat([entropy_mean, entropy_std, entropy_max], dim=1)
                features_list.append(entropy_features)

        # 特征聚合
        if len(features_list) == 0:
            raise ValueError("没有提取到任何特征")

        if self.aggregation == "concat":
            features = torch.cat(features_list, dim=1)
        elif self.aggregation == "mean":
            features = torch.stack(features_list, dim=1).mean(dim=1)
        else:
            raise ValueError(f"不支持的聚合方式: {self.aggregation}")

        return features


# ===== 4. 对比学习头 =====
class ContrastiveLearningHead(nn.Module):
    """
    对比学习头
    用于学习更具区分度的特征表示
    """

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 128,
        temperature: float = 0.07
    ):
        super().__init__()
        self.temperature = temperature

        # 投影头
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算对比学习损失

        Args:
            features: (batch_size, input_dim)
            labels: (batch_size,) 0=正确, 1=错误

        Returns:
            loss: 对比学习损失
        """
        # 投影
        z = self.projection(features)
        z = F.normalize(z, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(z, z.T) / self.temperature

        # 创建标签掩码（相同标签为正样本对）
        labels = labels.unsqueeze(1)
        mask = (labels == labels.T).float()
        mask.fill_diagonal_(0)  # 排除自己

        # 计算对比损失
        exp_sim = torch.exp(similarity_matrix)
        exp_sim = exp_sim * (1 - torch.eye(len(features), device=features.device))

        # 正样本对的损失
        pos_sim = (exp_sim * mask).sum(dim=1)
        all_sim = exp_sim.sum(dim=1)

        loss = -torch.log(pos_sim / (all_sim + 1e-10) + 1e-10)
        loss = loss[mask.sum(dim=1) > 0].mean()  # 只计算有正样本对的损失

        return loss


# ===== 5. Focal Loss =====
class FocalLoss(nn.Module):
    """
    Focal Loss用于处理类别不平衡
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size,) 预测概率 [0, 1]
            targets: (batch_size,) 真实标签 0 or 1

        Returns:
            loss: Focal Loss
        """
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# ===== 6. 集成探针模型 =====
class EnsembleProbe(nn.Module):
    """
    集成多个探针模型的预测
    """

    def __init__(
        self,
        probes: List[nn.Module],
        weights: Optional[List[float]] = None,
        method: str = "weighted_average"  # "weighted_average", "voting", "stacking"
    ):
        super().__init__()
        self.probes = nn.ModuleList(probes)
        self.method = method

        if weights is None:
            self.weights = [1.0 / len(probes)] * len(probes)
        else:
            assert len(weights) == len(probes)
            self.weights = weights

        # Stacking元学习器
        if method == "stacking":
            self.meta_learner = nn.Sequential(
                nn.Linear(len(probes), 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征

        Returns:
            ensemble_score: 集成后的风险分数
        """
        # 获取所有探针的预测
        predictions = []
        for probe in self.probes:
            pred = probe(x)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=1)  # (batch_size, num_probes)

        # 集成方法
        if self.method == "weighted_average":
            weights = torch.tensor(self.weights, device=x.device)
            ensemble_score = (predictions * weights).sum(dim=1)

        elif self.method == "voting":
            # 硬投票（阈值0.5）
            votes = (predictions > 0.5).float()
            ensemble_score = votes.mean(dim=1)

        elif self.method == "stacking":
            # 使用元学习器
            ensemble_score = self.meta_learner(predictions).squeeze(-1)

        else:
            raise ValueError(f"不支持的集成方法: {self.method}")

        return ensemble_score


# ===== 7. 动态阈值校准器 =====
class DynamicThresholdCalibrator:
    """
    动态阈值校准器
    根据验证集性能自动调整阈值
    """

    def __init__(
        self,
        initial_threshold: float = 0.5,
        target_precision: float = 0.85,
        target_recall: float = 0.90,
        update_frequency: int = 50
    ):
        self.threshold = initial_threshold
        self.target_precision = target_precision
        self.target_recall = target_recall
        self.update_frequency = update_frequency

        self.sample_count = 0
        self.predictions_buffer = []
        self.labels_buffer = []

    def update(self, predictions: torch.Tensor, labels: torch.Tensor):
        """
        更新阈值

        Args:
            predictions: (batch_size,) 预测分数
            labels: (batch_size,) 真实标签
        """
        self.predictions_buffer.extend(predictions.cpu().numpy())
        self.labels_buffer.extend(labels.cpu().numpy())
        self.sample_count += len(predictions)

        # 定期更新阈值
        if self.sample_count >= self.update_frequency:
            self._calibrate_threshold()
            self.predictions_buffer = []
            self.labels_buffer = []
            self.sample_count = 0

    def _calibrate_threshold(self):
        """校准阈值以达到目标精确率和召回率"""
        import numpy as np
        from sklearn.metrics import precision_recall_curve

        if len(self.predictions_buffer) == 0:
            return

        predictions = np.array(self.predictions_buffer)
        labels = np.array(self.labels_buffer)

        # 计算PR曲线
        precisions, recalls, thresholds = precision_recall_curve(labels, predictions)

        # 找到满足目标的最佳阈值
        # 目标：precision >= target_precision 且 recall >= target_recall
        valid_indices = np.where(
            (precisions[:-1] >= self.target_precision) &
            (recalls[:-1] >= self.target_recall)
        )[0]

        if len(valid_indices) > 0:
            # 选择F1最高的阈值
            f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
            best_idx = valid_indices[np.argmax(f1_scores[valid_indices])]
            self.threshold = thresholds[best_idx]
        else:
            # 如果没有满足条件的，选择F1最高的
            f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
            best_idx = np.argmax(f1_scores)
            self.threshold = thresholds[best_idx]

    def get_threshold(self) -> float:
        """获取当前阈值"""
        return self.threshold


if __name__ == "__main__":
    # 测试代码
    print("=== 测试HALT V2模型 ===\n")

    # 测试AttentionMLPProbe
    print("1. 测试AttentionMLPProbe")
    model = AttentionMLPProbe(
        input_dim=4096,
        hidden_dims=[512, 256, 128],
        num_attention_heads=4
    )
    x = torch.randn(8, 4096)
    output = model(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {output.shape}")
    print(f"   输出范围: [{output.min():.4f}, {output.max():.4f}]\n")

    # 测试FocalLoss
    print("2. 测试FocalLoss")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    predictions = torch.tensor([0.9, 0.1, 0.8, 0.2])
    targets = torch.tensor([1.0, 0.0, 1.0, 1.0])
    loss = focal_loss(predictions, targets)
    print(f"   Focal Loss: {loss.item():.4f}\n")

    # 测试DynamicThresholdCalibrator
    print("3. 测试DynamicThresholdCalibrator")
    calibrator = DynamicThresholdCalibrator(
        initial_threshold=0.5,
        target_precision=0.85,
        target_recall=0.90
    )
    print(f"   初始阈值: {calibrator.get_threshold():.4f}")

    print("\n所有测试通过！")
