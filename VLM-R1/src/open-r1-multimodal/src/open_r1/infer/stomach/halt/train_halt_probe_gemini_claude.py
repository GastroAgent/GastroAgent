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
    def __init__(self, json_path, feature_type='auto'):
        """
        Args:
            json_path: 数据文件路径
            feature_type: 特征类型选择
                - 'auto': 自动检测（推荐）
                - 'single': 使用单层特征 (middle_layer_hidden)
                - 'layer_50': 使用第50%层特征
                - 'layer_70': 使用第70%层特征
                - 'layer_85': 使用第85%层特征
                - 'concat': 使用拼接的多层特征
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.feature_type = feature_type

        # 提取特征和标签
        self.features = []
        self.labels = []

        # 检测数据格式
        if len(self.data) > 0:
            first_item = self.data[0]
            is_multi_layer = first_item.get('use_multi_layer', False)

            if is_multi_layer:
                print(f"✓ 检测到多层特征数据格式")
                available_layers = list(first_item['hidden_states'].keys())
                print(f"  可用层: {available_layers}")

                # 自动选择特征
                if feature_type == 'auto':
                    # 默认使用concat特征（多层融合）
                    if 'concat' in available_layers:
                        self.feature_type = 'concat'
                        print(f"  自动选择: concat (多层融合特征)")
                    else:
                        # 如果没有concat，使用最深的层
                        self.feature_type = available_layers[-1]
                        print(f"  自动选择: {self.feature_type}")
            else:
                print(f"✓ 检测到单层特征数据格式")
                self.feature_type = 'single'

        # 提取特征
        for item in self.data:
            # 根据数据格式提取特征
            if item.get('use_multi_layer', False):
                # 多层模式
                hidden_states = item['hidden_states']

                if self.feature_type == 'concat':
                    feat = np.array(hidden_states['concat'], dtype=np.float32)
                elif self.feature_type in hidden_states:
                    feat = np.array(hidden_states[self.feature_type], dtype=np.float32)
                else:
                    # 如果指定的层不存在，使用concat或第一个可用层
                    if 'concat' in hidden_states:
                        feat = np.array(hidden_states['concat'], dtype=np.float32)
                    else:
                        first_key = list(hidden_states.keys())[0]
                        feat = np.array(hidden_states[first_key], dtype=np.float32)
            else:
                # 单层模式（向后兼容）
                feat = np.array(item['middle_layer_hidden'], dtype=np.float32)

            self.features.append(feat)

            # 标签：1 代表幻觉/错误 (False)，0 代表正确 (True)
            # 注意：HALT探针通常预测的是"风险"，所以错误样本标记为1
            label = 1 if not item['is_correct'] else 0
            self.labels.append(label)

        self.features = torch.tensor(self.features)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

        print(f"✓ 加载完成: {len(self.labels)} 条数据")
        print(f"  特征维度: {self.features.shape[1]}")
        print(f"  使用特征: {self.feature_type}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ===== 2. 带权重补偿的训练器 =====
def train_probe(feature_type='auto'):
    """
    训练HALT探针

    Args:
        feature_type: 特征类型选择
            - 'auto': 自动检测（推荐）
            - 'single': 使用单层特征
            - 'layer_50', 'layer_70', 'layer_85': 使用指定层
            - 'concat': 使用多层融合特征
    """
    # 路径配置
    train_path = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/halt/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-all-RL-2500/merge_stomach_claude/train_with_hidden_states.json'
    val_path = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/halt/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-all-RL-2500/merge_stomach_claude/val_with_hidden_states.json'

    print("=" * 80)
    print("HALT探针训练")
    print("=" * 80)
    print(f"训练集: {train_path}")
    print(f"验证集: {val_path}")
    print(f"特征类型: {feature_type}")
    print()

    # 加载数据
    print("正在加载训练集...")
    train_ds = HaltDataset(train_path, feature_type=feature_type)
    print("\n正在加载验证集...")
    val_ds = HaltDataset(val_path, feature_type=feature_type)
    print()
    
    # --- 计算权重补偿 ---
    num_pos = torch.sum(train_ds.labels)  # 错误样本数 (Label=1)
    num_neg = len(train_ds.labels) - num_pos # 正确样本数 (Label=0)
    # pos_weight = 负样本数 / 正样本数
    pos_weight = torch.tensor([num_neg / num_pos]).to('cuda')

    print("=" * 80)
    print("数据统计")
    print("=" * 80)
    print(f"训练集:")
    print(f"  正确样本 (Label=0): {int(num_neg)} ({num_neg/len(train_ds.labels)*100:.1f}%)")
    print(f"  错误样本 (Label=1): {int(num_pos)} ({num_pos/len(train_ds.labels)*100:.1f}%)")
    print(f"  类别比例: {num_neg/num_pos:.2f}:1")
    print(f"\nLoss权重补偿 (pos_weight): {pos_weight.item():.4f}")
    print("=" * 80)
    print()

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # --- 模型定义 ---
    # 维度根据你的 Llava-Qwen2-7B 自动适配
    input_dim = train_ds.features.shape[1]

    print("=" * 80)
    print("模型配置")
    print("=" * 80)
    print(f"输入维度: {input_dim}")
    print(f"隐藏层: 512 -> 256")
    print(f"输出维度: 1 (二分类)")
    print(f"Dropout: 0.4, 0.3")
    print("=" * 80)
    print()

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

    print("开始训练...")
    print("=" * 80)
    
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
            model_save_path = f'halt_probe_best_{train_ds.feature_type}.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_auc': best_auc,
                'input_dim': input_dim,
                'feature_type': train_ds.feature_type
            }, model_save_path)
            print(f"✓ 已保存最优模型: {model_save_path} (AUC: {best_auc:.4f})")

    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    print(f"最佳验证AUC: {best_auc:.4f}")
    print(f"模型保存路径: halt_probe_best_{train_ds.feature_type}.pth")
    print("=" * 80)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='训练HALT探针')
    parser.add_argument('--feature', type=str, default='auto',
                        choices=['auto', 'single', 'layer_50', 'layer_70', 'layer_85', 'concat'],
                        help='特征类型选择')
    parser.add_argument('--compare', action='store_true',
                        help='对比训练所有可用层（忽略--feature参数）')

    args = parser.parse_args()

    if args.compare:
        # 对比模式：训练所有可用层
        print("\n" + "=" * 80)
        print("对比训练模式：将训练所有可用层的探针")
        print("=" * 80 + "\n")

        feature_types = ['layer_50', 'layer_70', 'layer_85', 'concat']
        results = {}

        for ft in feature_types:
            print(f"\n{'#' * 80}")
            print(f"# 训练特征类型: {ft}")
            print(f"{'#' * 80}\n")

            try:
                train_probe(feature_type=ft)
                # 这里可以记录结果，但需要修改train_probe返回值
                results[ft] = "完成"
            except Exception as e:
                print(f"✗ 训练 {ft} 时出错: {e}")
                results[ft] = f"失败: {e}"

        # 打印对比结果
        print("\n" + "=" * 80)
        print("对比训练结果总结")
        print("=" * 80)
        for ft, status in results.items():
            print(f"{ft:15s}: {status}")
        print("=" * 80)
        print("\n提示: 查看各个模型文件的AUC分数来选择最佳层")
    else:
        # 单次训练模式
        train_probe(feature_type=args.feature)