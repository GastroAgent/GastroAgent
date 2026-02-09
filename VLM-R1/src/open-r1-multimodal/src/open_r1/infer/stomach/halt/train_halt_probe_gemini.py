import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
import os

# ===== 1. 数据加载与预处理 =====
class HaltDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # 提取特征和标签
        # 假设 hidden_states 存储在 'hidden_state' 字段，标签基于 'is_correct'
        self.features = []
        self.labels = []
        
        for item in self.data:
            # 获取隐藏状态 (确保转为 float32)
            feat = np.array(item['middle_layer_hidden'], dtype=np.float32)
            self.features.append(feat)
            
            # 标签：1 代表幻觉/错误 (False)，0 代表正确 (True)
            # 注意：HALT探针通常预测的是“风险”，所以错误样本标记为1
            label = 1 if not item['is_correct'] else 0
            self.labels.append(label)
            
        self.features = torch.tensor(self.features)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ===== 2. 带权重补偿的训练器 =====
def train_probe():
    # 路径配置
    train_path = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/halt/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-all-RL-2500/merge_stomach_claude/train_with_hidden_states.json'
    val_path = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/halt/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-all-RL-2500/merge_stomach_claude/val_with_hidden_states.json'
    
    train_ds = HaltDataset(train_path)
    val_ds = HaltDataset(val_path)
    
    # --- 计算权重补偿 ---
    num_pos = torch.sum(train_ds.labels)  # 错误样本数 (Label=1)
    num_neg = len(train_ds.labels) - num_pos # 正确样本数 (Label=0)
    # pos_weight = 负样本数 / 正样本数
    pos_weight = torch.tensor([num_neg / num_pos]).to('cuda')
    print(f"检测到类别不平衡: 正确(0)={int(num_neg)}, 错误(1)={int(num_pos)}")
    print(f"应用 Loss 权重补偿 (pos_weight): {pos_weight.item():.4f}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # --- 模型定义 ---
    # 维度根据你的 Llava-Qwen2-7B 自动适配
    input_dim = train_ds.features.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.4),  # 强力 Dropout 防止过拟合
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1)
    ).to('cuda')

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # --- 训练循环 ---
    best_auc = 0
    epochs = 50
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for feats, lbls in train_loader:
            feats, lbls = feats.to('cuda'), lbls.to('cuda')
            optimizer.zero_grad()
            outputs = model(feats).squeeze()
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for feats, lbls in val_loader:
                feats = feats.to('cuda')
                outputs = torch.sigmoid(model(feats).squeeze())
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(lbls.numpy())

        auc = roc_auc_score(all_labels, all_preds)
        f1 = f1_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])

        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val AUC: {auc:.4f} | F1: {f1:.4f}")

        # 保存最优模型
        if auc > best_auc:
            best_auc = auc
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_auc': best_auc,
                'input_dim': input_dim
            }, 'halt_probe_best_merge_kvasir_stomach_claude.pth')
            print(f"已保存最优模型 (AUC: {best_auc:.4f})")

if __name__ == "__main__":
    train_probe()