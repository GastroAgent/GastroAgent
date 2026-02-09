"""
================================================================================
HALT探针模型训练脚本
================================================================================
功能: 训练轻量级探针模型，用于预测幻觉风险
输入: 带有标注的训练数据（包含正确/错误标签）
输出: 训练好的探针模型权重
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# ===== 配置参数 =====
# 数据路径
TRAIN_DATA_PATH = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/halt/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-all-RL-2500/kvasir/train_with_hidden_states.json'
VAL_DATA_PATH = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/halt/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-all-RL-2500/kvasir/val_with_hidden_states.json'

# 模型路径
MODEL_ID = '/mnt/inaisfs/data/home/tansy_criait/weights/tsy/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-all-RL-2500'
PROBE_SAVE_PATH = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/stomach/halt/models/halt_probe_latest.pth'


# 训练参数
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
HALT_MIDDLE_LAYER_RATIO = 0.5
HALT_PROBE_HIDDEN_DIM = 256

# os.environ['CUDA_VISIBLE_DEVICES'] = '2,4'

# ===== 探针模型定义（与推理脚本保持一致）=====
class HALTProbeModel(nn.Module):
    """HALT轻量级探针模型"""
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_states):
        return self.mlp(hidden_states).squeeze(-1)

# ===== 数据集定义 =====
class HALTDataset(Dataset):
    """
    HALT训练数据集
    每个样本包含：
    - hidden_states: 中间层隐藏状态
    - label: 0=正确答案（低风险），1=错误答案（高风险/幻觉）
    """
    def __init__(self, data_path, hidden_states_key='middle_layer_hidden'):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        # 过滤掉没有隐藏状态的样本
        self.data = [d for d in self.data if hidden_states_key in d and 'is_correct' in d]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 隐藏状态
        hidden_states = torch.tensor(item['middle_layer_hidden'], dtype=torch.float32)

        # 标签：错误答案=1（高风险），正确答案=0（低风险）
        label = torch.tensor(0.0 if item['is_correct'] else 1.0, dtype=torch.float32)

        return hidden_states, label

# ===== 训练函数 =====
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for hidden_states, labels in tqdm(dataloader, desc="Training"):
        hidden_states = hidden_states.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(hidden_states)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend((outputs > 0.5).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy

# ===== 验证函数 =====
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for hidden_states, labels in tqdm(dataloader, desc="Validation"):
            hidden_states = hidden_states.to(device)
            labels = labels.to(device)

            outputs = model(hidden_states)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_probs.extend(outputs.cpu().numpy())
            all_preds.extend((outputs > 0.5).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    return avg_loss, accuracy, precision, recall, f1, auc

# ===== 主训练流程 =====
def main():
    print("=" * 80)
    print("HALT探针模型训练")
    print("=" * 80)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    print(f"\n加载训练数据: {TRAIN_DATA_PATH}")
    train_dataset = HALTDataset(TRAIN_DATA_PATH)
    print(f"训练样本数: {len(train_dataset)}")

    print(f"\n加载验证数据: {VAL_DATA_PATH}")
    val_dataset = HALTDataset(VAL_DATA_PATH)
    print(f"验证样本数: {len(val_dataset)}")

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 获取隐藏层维度（从第一个样本）
    sample_hidden, _ = train_dataset[0]
    hidden_dim = sample_hidden.shape[0]
    print(f"\n隐藏层维度: {hidden_dim}")

    # 初始化模型
    print(f"初始化探针模型 (hidden_dim={HALT_PROBE_HIDDEN_DIM})...")
    model = HALTProbeModel(
        input_dim=hidden_dim,
        hidden_dim=HALT_PROBE_HIDDEN_DIM
    ).to(device)

    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练循环
    best_f1 = 0.0
    print(f"\n开始训练 (epochs={NUM_EPOCHS})...\n")

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # 验证
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = validate(
            model, val_loader, criterion, device
        )
        print(f"  Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        print(f"  Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs(os.path.dirname(PROBE_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), PROBE_SAVE_PATH)
            print(f"  ✓ 保存最佳模型 (F1={best_f1:.4f})")

        print()

    print("=" * 80)
    print("训练完成！")
    print(f"最佳F1分数: {best_f1:.4f}")
    print(f"模型保存路径: {PROBE_SAVE_PATH}")
    print("=" * 80)

if __name__ == "__main__":
    main()

