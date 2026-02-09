"""
================================================================================
HALT V2 训练脚本
================================================================================
功能: 训练改进的HALT V2探针模型
改进点:
1. 多层特征融合
2. 注意力机制
3. Focal Loss处理类别不平衡
4. 对比学习增强特征区分度
5. 动态阈值校准
6. 数据增强
================================================================================
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
from datetime import datetime

# 导入HALT V2模块
from halt_v2_models import (
    AttentionMLPProbe,
    FocalLoss,
    ContrastiveLearningHead,
    DynamicThresholdCalibrator
)
from halt_v2_config import (
    HALT_V2_CONFIG,
    PATH_CONFIG,
    PERFORMANCE_CONFIG,
    LOGGING_CONFIG
)

# ===== 配置日志 =====
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== GPU配置 =====
os.environ['CUDA_VISIBLE_DEVICES'] = '2,4'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== 路径配置 =====
TRAIN_DATA_PATH = os.path.join(PATH_CONFIG["data_dir"], "train_split_with_hidden_states.json")
VAL_DATA_PATH = os.path.join(PATH_CONFIG["data_dir"], "val_split_with_hidden_states.json")
MODEL_SAVE_PATH = os.path.join(PATH_CONFIG["model_dir"], "halt_v2_probe.pth")
CHECKPOINT_DIR = PATH_CONFIG["checkpoint_dir"]

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ===== 数据增强 =====
class DataAugmentation:
    """数据增强策略"""

    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get("enabled", False)
        self.methods = config.get("methods", [])

    def apply(self, features: torch.Tensor, training: bool = True) -> torch.Tensor:
        """应用数据增强"""
        if not self.enabled or not training:
            return features

        # Dropout增强
        if "dropout" in self.methods:
            dropout_rate = self.config.get("dropout_rate", 0.1)
            mask = torch.bernoulli(torch.ones_like(features) * (1 - dropout_rate))
            features = features * mask / (1 - dropout_rate)

        # 高斯噪声
        if "noise" in self.methods:
            noise_std = self.config.get("noise_std", 0.01)
            noise = torch.randn_like(features) * noise_std
            features = features + noise

        return features


# ===== 数据集定义 =====
class HALTV2Dataset(Dataset):
    """
    HALT V2训练数据集
    支持多层特征和数据增强
    """

    def __init__(
        self,
        data_path: str,
        augmentation: DataAugmentation = None,
        use_multi_layer: bool = True
    ):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        # 过滤无效样本
        self.data = [
            d for d in self.data
            if 'middle_layer_hidden' in d and 'is_correct' in d
        ]

        self.augmentation = augmentation
        self.use_multi_layer = use_multi_layer

        logger.info(f"加载数据集: {data_path}")
        logger.info(f"  总样本数: {len(self.data)}")
        logger.info(f"  正确样本: {sum(1 for d in self.data if d['is_correct'])}")
        logger.info(f"  错误样本: {sum(1 for d in self.data if not d['is_correct'])}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 提取特征
        if self.use_multi_layer and 'multi_layer_hidden' in item:
            # 多层特征
            features = torch.tensor(item['multi_layer_hidden'], dtype=torch.float32)
        else:
            # 单层特征
            features = torch.tensor(item['middle_layer_hidden'], dtype=torch.float32)

        # 标签：错误=1（高风险），正确=0（低风险）
        label = torch.tensor(0.0 if item['is_correct'] else 1.0, dtype=torch.float32)

        # 问题类型（可选）
        question_type = item.get('question_type', 'Unknown')

        return features, label, question_type


# ===== 训练函数 =====
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    contrastive_head: nn.Module,
    optimizer: optim.Optimizer,
    augmentation: DataAugmentation,
    device: torch.device,
    use_contrastive: bool = True,
    contrastive_weight: float = 0.1
) -> Tuple[float, float]:
    """训练一个epoch"""
    model.train()
    if contrastive_head is not None:
        contrastive_head.train()

    total_loss = 0
    total_cls_loss = 0
    total_contrastive_loss = 0
    all_preds = []
    all_labels = []

    for features, labels, _ in tqdm(dataloader, desc="Training"):
        features = features.to(device)
        labels = labels.to(device)

        # 数据增强
        features = augmentation.apply(features, training=True)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(features)

        # 分类损失
        cls_loss = criterion(outputs, labels)

        # 对比学习损失
        if use_contrastive and contrastive_head is not None:
            contrastive_loss = contrastive_head(features, labels)
            loss = cls_loss + contrastive_weight * contrastive_loss
            total_contrastive_loss += contrastive_loss.item()
        else:
            loss = cls_loss
            contrastive_loss = torch.tensor(0.0)

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            HALT_V2_CONFIG["training"]["regularization"]["gradient_clip"]
        )

        optimizer.step()

        # 统计
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        all_preds.extend((outputs > 0.5).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 计算指标
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_contrastive_loss = total_contrastive_loss / len(dataloader) if use_contrastive else 0.0
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, avg_cls_loss, avg_contrastive_loss


# ===== 验证函数 =====
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    calibrator: DynamicThresholdCalibrator = None
) -> Dict:
    """验证模型"""
    model.eval()

    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    all_question_types = []

    with torch.no_grad():
        for features, labels, question_types in tqdm(dataloader, desc="Validation"):
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_question_types.extend(question_types)

    # 转换为numpy数组
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 使用校准器的阈值
    if calibrator is not None:
        threshold = calibrator.get_threshold()
    else:
        threshold = 0.5

    all_preds = (all_probs > threshold).astype(int)

    # 计算指标
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    # 计算AUC
    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except:
        auc_roc = 0.0

    # 按问题类型分析
    question_type_metrics = {}
    for qt in set(all_question_types):
        qt_indices = [i for i, q in enumerate(all_question_types) if q == qt]
        if len(qt_indices) > 0:
            qt_labels = all_labels[qt_indices]
            qt_preds = all_preds[qt_indices]
            qt_acc = accuracy_score(qt_labels, qt_preds)
            question_type_metrics[qt] = {
                "accuracy": qt_acc,
                "count": len(qt_indices)
            }

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold": threshold,
        "question_type_metrics": question_type_metrics,
        "all_probs": all_probs,
        "all_labels": all_labels,
    }


# ===== 可视化函数 =====
def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_f1s: List[float],
    save_path: str
):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 损失曲线
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # F1曲线
    ax2.plot(val_f1s, label='Val F1', marker='o', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Validation F1 Score')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"训练曲线已保存: {save_path}")


def plot_roc_pr_curves(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: str
):
    """绘制ROC和PR曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # ROC曲线
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_roc = roc_auc_score(labels, probs)
    ax1.plot(fpr, tpr, label=f'ROC (AUC={auc_roc:.3f})', linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(True)

    # PR曲线
    precision, recall, _ = precision_recall_curve(labels, probs)
    ax2.plot(recall, precision, linewidth=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"ROC和PR曲线已保存: {save_path}")


# ===== 主训练流程 =====
def main():
    logger.info("=" * 80)
    logger.info("HALT V2 探针模型训练")
    logger.info("=" * 80)

    # 加载配置
    feature_config = HALT_V2_CONFIG["feature_extraction"]
    probe_config = HALT_V2_CONFIG["probe_model"]
    training_config = HALT_V2_CONFIG["training"]
    threshold_config = HALT_V2_CONFIG["dynamic_threshold"]

    # 数据增强
    augmentation = DataAugmentation(training_config["data_augmentation"])

    # 加载数据
    logger.info(f"\n加载训练数据: {TRAIN_DATA_PATH}")
    train_dataset = HALTV2Dataset(
        TRAIN_DATA_PATH,
        augmentation=augmentation,
        use_multi_layer=feature_config["use_multi_layer"]
    )

    logger.info(f"\n加载验证数据: {VAL_DATA_PATH}")
    val_dataset = HALTV2Dataset(
        VAL_DATA_PATH,
        augmentation=None,
        use_multi_layer=feature_config["use_multi_layer"]
    )

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=PERFORMANCE_CONFIG["batch_size"],
        shuffle=True,
        num_workers=PERFORMANCE_CONFIG["num_workers"],
        pin_memory=PERFORMANCE_CONFIG["pin_memory"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=PERFORMANCE_CONFIG["batch_size"],
        shuffle=False,
        num_workers=PERFORMANCE_CONFIG["num_workers"],
        pin_memory=PERFORMANCE_CONFIG["pin_memory"]
    )

    # 获取特征维度
    sample_features, _, _ = train_dataset[0]
    input_dim = sample_features.shape[0]
    logger.info(f"\n特征维度: {input_dim}")

    # 初始化模型
    logger.info(f"\n初始化探针模型...")
    if probe_config["architecture"] == "attention_mlp":
        model = AttentionMLPProbe(
            input_dim=input_dim,
            **probe_config["attention_mlp"]
        ).to(device)
    else:
        raise ValueError(f"不支持的模型架构: {probe_config['architecture']}")

    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数
    if training_config["class_balance"]["method"] == "focal_loss":
        criterion = FocalLoss(
            alpha=training_config["class_balance"]["focal_loss_alpha"],
            gamma=training_config["class_balance"]["focal_loss_gamma"]
        )
        logger.info("使用Focal Loss处理类别不平衡")
    else:
        criterion = nn.BCELoss()
        logger.info("使用BCE Loss")

    # 对比学习头
    use_contrastive = training_config["contrastive_learning"]["enabled"]
    if use_contrastive:
        contrastive_head = ContrastiveLearningHead(
            input_dim=input_dim,
            temperature=training_config["contrastive_learning"]["temperature"]
        ).to(device)
        logger.info("启用对比学习")
    else:
        contrastive_head = None

    # 优化器
    optimizer = optim.Adam(
        list(model.parameters()) + (list(contrastive_head.parameters()) if contrastive_head else []),
        lr=1e-4,
        weight_decay=training_config["regularization"]["l2_weight"]
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )

    # 动态阈值校准器
    if threshold_config["enabled"] and threshold_config["method"] == "adaptive":
        calibrator = DynamicThresholdCalibrator(
            **threshold_config["adaptive"]
        )
        logger.info("启用动态阈值校准")
    else:
        calibrator = None

    # 训练循环
    num_epochs = 20
    best_f1 = 0.0
    train_losses = []
    val_losses = []
    val_f1s = []

    logger.info(f"\n开始训练 (epochs={num_epochs})...\n")

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info("-" * 40)

        # 训练
        train_loss, train_acc, train_cls_loss, train_cont_loss = train_epoch(
            model, train_loader, criterion, contrastive_head,
            optimizer, augmentation, device,
            use_contrastive=use_contrastive
        )

        logger.info(f"  Train Loss: {train_loss:.4f} (Cls: {train_cls_loss:.4f}, Cont: {train_cont_loss:.4f})")
        logger.info(f"  Train Accuracy: {train_acc:.4f}")

        # 验证
        val_metrics = validate(model, val_loader, criterion, device, calibrator)

        logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        logger.info(f"  AUC-ROC: {val_metrics['auc_roc']:.4f}")
        logger.info(f"  TP: {val_metrics['tp']}, FP: {val_metrics['fp']}, TN: {val_metrics['tn']}, FN: {val_metrics['fn']}")
        logger.info(f"  Threshold: {val_metrics['threshold']:.4f}")

        # 更新校准器
        if calibrator is not None:
            calibrator.update(
                torch.tensor(val_metrics['all_probs']),
                torch.tensor(val_metrics['all_labels'])
            )

        # 学习率调度
        scheduler.step(val_metrics['f1'])

        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_metrics['loss'])
        val_f1s.append(val_metrics['f1'])

        # 保存最佳模型
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'threshold': val_metrics['threshold'],
                'config': HALT_V2_CONFIG,
            }, MODEL_SAVE_PATH)
            logger.info(f"  ✓ 保存最佳模型 (F1={best_f1:.4f})")

        logger.info("")

    # 绘制训练曲线
    plot_training_curves(
        train_losses, val_losses, val_f1s,
        os.path.join(CHECKPOINT_DIR, "training_curves.png")
    )

    # 绘制ROC和PR曲线
    plot_roc_pr_curves(
        val_metrics['all_labels'],
        val_metrics['all_probs'],
        os.path.join(CHECKPOINT_DIR, "roc_pr_curves.png")
    )

    logger.info("=" * 80)
    logger.info("训练完成！")
    logger.info(f"最佳F1分数: {best_f1:.4f}")
    logger.info(f"模型保存路径: {MODEL_SAVE_PATH}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
