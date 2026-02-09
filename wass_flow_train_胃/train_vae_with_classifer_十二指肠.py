import glob
import json
import math
from tqdm import tqdm
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import random
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from flow_matcher import create_generator # 不包含 x0, x1.
from utils.data_loader_test import MedicalJsonDataset, MedicalClSJsonDataset
from utils.train_utils import infiniteloop
from PIL import Image
from math import sqrt
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from einops.layers.torch import Rearrange

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

    def forward(self, x, **kwargs):
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
        self.proj = GatedMLPClassifier(embed_dim, embed_dim, patch_size * patch_size * out_chans, patch_size * patch_size * out_chans)
        self.linear = nn.Linear(patch_size * patch_size * out_chans, 4)
    def forward(self, x):
        # x: [B, num_patches, embed_dim]
        x = self.proj(x)  # [B, num_patches, 8*8*4]
        # 重塑为 [B, C, H, W]
        B, N, _ = x.shape
        h = w = int(N ** 0.5)  # 假设是正方形
        x = x.reshape(B, h, w, -1)
        x = self.linear(x)
        return x.permute(0,3,1,2)  # [B, 16, 16, 4]

class AttentionDownEncoderXL(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=64, patch_size=4, in_chans=4, embed_dim=512)
        self.transformer = SimpleTransformer(embed_dim=512, depth=12, num_heads=8, mlp_ratio=4)
        self.decoder = UpsampleDecoder(embed_dim=512, patch_size=4, out_chans=4)

    def forward(self, x, return_hidden=False):
        # x: [B, 4, 64, 64]
        x = self.patch_embed(x)        # -> [B, 256, 512]
        hidden_states = self.transformer(x)        # -> [B, 256, 512]
        output = self.decoder(hidden_states)            # -> [B, 4, 64, 64]
        B = output.shape[0]
        if return_hidden:
            return output.reshape(B, -1), hidden_states
        return output.reshape(B, -1)

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
    def __init__(self, pretrained=False, freeze_base=False, model='resnet34'):
        super(TripletNetwork, self).__init__()
        if model == 'attention':
            self.embedding = AttentionDownEncoderXL()
        else:
            self.embedding = EmbeddingNetwork(pretrained=pretrained, freeze_base=freeze_base, model=model)

    def encode(self, x, return_image=False, return_hidden=False):
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
            return embeds, hidden_state
        else:
            return embeds
        
    def forward(self, anchor, positive, negative):
        anchor_emb = self.encode(anchor, True)
        positive_emb = self.encode(positive, True)
        negative_emb = self.encode(negative, True)
        return anchor_emb, positive_emb, negative_emb

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.l2norm = nn.functional.normalize

    def forward(self, anchor, positive, negative, pos_beta=1, neg_beta=0.2):
        # 计算 Wass 距离
        d_pos = torch.norm(anchor - positive, p=2, dim=1)  # Anchor-Positive 距离
        d_neg = torch.norm(anchor - negative, p=2, dim=1)  # Anchor-Negative 距离

        # Triplet Loss 公式
        loss = torch.mean(torch.clamp(pos_beta * d_pos - neg_beta * d_neg + self.margin, min=0.0))
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, other, target, beta=1):
        """
        - anchor: anchor 嵌入向量 (B, D)
        - other: 正样本/负样本 嵌入向量 (B, D)
        - target: 0 或 1，1 表示正样本对，0 表示负样本对
        """
        distance = torch.norm(anchor - other, p=2, dim=1) * beta  # Wass 距离
        loss = torch.mean(
            target * distance + (1 - target) * torch.clamp(self.margin - distance, min=0.0)
        )
        return loss

def sinkhorn_loss(bx, bx1, epsilon=0.1, n_iter=25, reduction='sum'):
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

if __name__ == '__main__':
    # 加载数据集
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_A = transforms.Compose([
        # transforms.RandomApply([        
        #     transforms.RandomRotation(
        #         degrees=30,
        #         expand=True,            # 是否扩展画布以容纳完整图像（True会改变图像大小）
        #         center=None,             # 旋转中心，默认中心点；可设为 (x, y)
        #         fill=(0, 0, 0)           # 填充颜色，例如填白色：(255, 255, 255)
        #     )], p=0.25),
        transforms.Resize((512, 512)), 
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomVerticalFlip(p=0.5), 
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),], p=0.15), 
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 0.5))], p=0.15), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    transform_grey = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.RandomGrayscale(p=1),  # 数据增强：50% 概率灰度化
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    batch_size = 16
    
    json_paths = glob.glob( # 需要是 同类的数据。
        "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_十二指肠/data/data_pairs/*.json")
    # json_paths = json_paths[:4]
    dataset = MedicalClSJsonDataset(
        path=json_paths,
        transform=transform,
        hint_transform=transform_grey,
        transform_A=transform_A,
        transform_B=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    test_json = [
        "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_十二指肠/data/eval_all.json",
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TripletNetwork(pretrained=False, freeze_base=False, model='attention').to(device)
    criterion = nn.CrossEntropyLoss()
    
    generator = create_generator(only_vae=False)
    use_generate = True
    os.makedirs("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_十二指肠/logs_flow", exist_ok=True)
    writer = SummaryWriter(log_dir='/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_十二指肠/logs_flow')
    classifer = nn.Sequential(
        UpsampleDecoder(embed_dim=512, patch_size=4, out_chans=4),
        # Rearrange('b c h w -> b (c h w)'), 
        Rearrange('b ... -> b (...)'),
        GatedMLPClassifier(input_dim=4 * 16 * 16, hidden_dim1=4096, hidden_dim2=1024, output_dim=256)
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": 1e-5, "weight_decay": 0.01},
            {"params": classifer.parameters(), "lr": 2e-5, "weight_decay": 0.001}
        ]
    )
    def train_triplet(model, classifer, dataloaders, criterion, optimizer, device="cuda",
                      criterion_contrastive=None, generator=None, cal_wasserstein_loss=None, epochs=20):
        if generator is None:
            raise NotImplementedError
        else:
            vae = generator.vae
        total_step = epochs * len(dataloaders)
        dataset = []
        for i in test_json:
            dataset += json.load(open(i, "r"))
        best_acc = evaluate_triplet(model, classifer, dataset, device, generator, 0)
        dataloader = infiniteloop(dataloaders)
        for step in tqdm(range(total_step)):
            model.train()
            batch = dataloader.__iter__().__next__()

            if generator is not None and use_generate:
                generator.args.op_match = False
                generator.num_steps = random.choice([16, 20, 24])
                generator.args.num_steps = generator.num_steps
                generator.use_gt_vt = True if random.random() > 0.5 else False
                if generator.use_gt_vt:  # 插值的 数据增强。
                    sample_step = random.choice([-1, -1, 1, -2])
                    result = generator.generate(batch, sample_step=sample_step)
                    anchor = result["samples"]
                else:
                    if random.random() < 0.25:
                        anchor = batch["x0"]
                        anchor = anchor.to(vae.device)
                        # 压缩至 潜在空间
                        with torch.no_grad():
                            anchor = vae.encode(anchor).latent_dist
                            anchor = anchor.sample() * 0.18215
                    else:
                        sample_step = random.choice([-1, 1, -1, -2]) # 考虑信噪比
                        result = generator.generate(batch, sample_step=sample_step)
                        anchor = result["samples"]
            else:
                anchor = batch["x0"]
                anchor = anchor.to(vae.device)
                # 压缩至 潜在空间
                with torch.no_grad():
                    anchor = vae.encode(anchor).latent_dist
                    anchor = anchor.sample() * 0.18215

            anchor = anchor.to(device)
            optimizer.zero_grad()
            _, hidden_state = model.encode(anchor, False, True)
            B = hidden_state.shape[0]
            # hidden_state = hidden_state.reshape(B, -1)
            logits = classifer(hidden_state)
            loss = criterion(logits, batch['class_id'].long().to(device))

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            writer.add_scalar('Training/Loss', loss.item(), step)
            writer.add_scalar('Training/Grad Norm', grad_norm.item(), step)
            if step % 10 == 0:
                print(f"Global Step {step}/{total_step} | Loss: {loss.item():.4f} | Grad Norm: {grad_norm.item():.4f}")
            save_step = 500
            if (step + 1) % save_step == 0:
                print("Eval...")
                dataset = []
                for i in test_json:
                    dataset += json.load(open(i, "r"))
                accuracy = evaluate_triplet(model, classifer, dataset, device, generator, 0)
                writer.add_scalar('Evaling/Acc', accuracy, step)
                if best_acc < accuracy:
                    torch.save(model.state_dict(), f"/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_十二指肠/cls_weights/tiny_attention.pt")
                    torch.save(classifer.state_dict(), f"/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_十二指肠/cls_weights/tiny_attention_classifer.pt")
                    best_acc = accuracy
                else:
                    pass
                
        accuracy = evaluate_triplet(model, classifer, dataset, device, generator, 0)      
        if best_acc < accuracy:
            torch.save(model.state_dict(), f"/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_十二指肠/cls_weights/tiny_attention.pt")
            torch.save(classifer.state_dict(), f"/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_十二指肠/cls_weights/tiny_attention_classifer.pt")
            best_acc = accuracy
        else:
            pass

    ### 加载 checkpoints
    try:
        state_dict = torch.load("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_十二指肠/cls_weights/attention.pt", weights_only=True)
        model.load_state_dict(state_dict)
        state_dict = torch.load("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_十二指肠/cls_weights/attention_classifer.pt", weights_only=True)
        classifer.load_state_dict(state_dict)
    except:
        pass
    
    model = model.to(device)
    classifer = classifer.to(device)
    print(model)
    label_map = json.load(open("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_十二指肠/utils/label_map.json", "r"))
    
    def evaluate_triplet(model, classifer, dataset, device, generator, step=0, k=5):
        if generator is None:
            raise NotImplementedError
        else:
            vae = generator.vae
        model.eval()
        classifer.eval()
        correct = 0
        errors = {}
        correct_dict = {}
        static = {}
        correct_topk = 0
        total = 0
        l2_correct = 0
        with torch.no_grad():
            for data in tqdm(dataset):
                total += 1
                answer_id = data['label_A_id']
                label_A = data['label_A']
                if label_A not in errors:
                    errors[label_A] = []
                    correct_dict[label_A] = 0
                    static[label_A] = 0
                anchor = transform(Image.open(data['x0']).convert("RGB")).to(device).unsqueeze(0)
                anchor = anchor.to(vae.device)
                static[label_A] += 1
                
                # 压缩至 潜在空间
                with torch.no_grad():
                    anchor = vae.encode(anchor).latent_dist
                    anchor = anchor.sample() * 0.18215
                anchor = anchor.to(device)
                anchor_emb, hidden_state = model.encode(anchor, False, True)
                B = hidden_state.shape[0]
                # hidden_state = hidden_state.reshape(B, -1)
                logits = classifer(hidden_state) # shape: [1, 256]
                pred_option_id = torch.argmax(logits, dim=1)
                data["pred_option_id"] = pred_option_id.item()
                if answer_id == pred_option_id:
                    correct += 1
                    correct_dict[label_A] += 1 
                else:
                    question_path = data['x0'] 
                    errors[label_A].append(question_path)
                
                topk_indices = np.argsort(logits[0].cpu().numpy())[:3]  # 因为分数越小越好
   
                # 判断正确答案是否在 top-k 预测中
                if answer_id in topk_indices.tolist():
                    correct_topk += 1
        accuracy = correct / total
        print(f"Evaluation Accuracy: {accuracy:.4f} Top 3: {(correct_topk / total):.4f}")
        return accuracy
    
    # train_triplet(model, classifer, dataloader, criterion=criterion, optimizer=optimizer, device=device,
    #               generator=generator, criterion_contrastive=None,
    #               cal_wasserstein_loss=cal_wasserstein_loss, epochs=15) 

    dataset = []
    for i in test_json:
        dataset += json.load(open(i, "r"))
    evaluate_triplet(model, classifer, dataset, device, generator, 0)