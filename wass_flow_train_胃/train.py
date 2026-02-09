import glob
import json
import math
from tqdm import tqdm
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,5"
import random
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
from math import sqrt
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from einops.layers.torch import Rearrange
import sys
sys.path.append("/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match")
from utils.data_loader_test import MedicalJsonDataset
from utils.train_utils import infiniteloop
from utils.optim import *
from flow_matcher import create_generator # 不包含 x0, x1.
from loss import *
from model import *

if __name__ == '__main__':
    # 加载数据集
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_A = transforms.Compose([
        transforms.Resize((512, 512)), 
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),], p=0.1), 
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 0.5))], p=0.1), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    transform_grey = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.RandomGrayscale(p=1),  # 数据增强：50% 概率灰度化
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    batch_size = 2
    dataloaders = []
    
    json_paths = glob.glob(
        "/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/data/胃/data_pairs_tiny/*.json")
    # json_paths = json_paths[:4]
    for json_path in tqdm(json_paths):
        dataset = MedicalJsonDataset(
            path=json_path,
            transform=transform,
            hint_transform=transform_grey,
            transform_A=transform_A,
            transform_B=transform,
            flip=True,
            pflip=0.25
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        dataloaders.append(dataloader)
        # break
    
    test_json = [
        "/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/data/胃/new_eval_all.json",
    ]
    
    # 初始化模型和优化器
    use_generate = True
    free_classifer = False
    use_muon = True
    beta = 1.024 * 0.35
    generator = create_generator(only_vae=(not use_generate), device="cuda:0")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TripletNetwork(pretrained=False, freeze_base=False, model='attention')
    classifer = nn.Sequential(
        UpsampleDecoder(embed_dim=512, patch_size=4, out_chans=4),
        # Rearrange('b c h w -> b (c h w)'), 
        Rearrange('b ... -> b (...)'),
        GatedMLPClassifier(input_dim=4 * 16 * 16, hidden_dim1=4096, hidden_dim2=1024, output_dim=256)
    )
    
    if not use_muon:
        if free_classifer:
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.001)
        else:
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.parameters(), "lr": 2e-5, "weight_decay": 0.01},
                    {"params": classifer.parameters(), "lr": 1e-5, "weight_decay": 0.001}
                ]
            )
    else:
        if free_classifer:
            hidden_weights = [p for p in model.parameters() if p.ndim >= 2]      # 如 Linear.weight
            other_params = [p for p in model.parameters() if p.ndim < 2]        # 如 bias

            # 构建参数组
            param_groups = [
                dict(params=hidden_weights, use_muon=True, lr=0.02, weight_decay=0.01),
                dict(params=other_params, use_muon=False, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
            ]

            optimizer = MuonWithAuxAdam(param_groups)
        else:
            hidden_weights = [p for p in model.parameters() if p.ndim >= 2]      # 如 Linear.weight
            other_params = [p for p in model.parameters() if p.ndim < 2]        # 如 bias

            # 构建参数组
            param_groups = [
                dict(params=hidden_weights, use_muon=True, lr=0.001, weight_decay=0.01),
                dict(params=other_params, use_muon=False, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
                {"params": classifer.parameters(), "lr": 1e-4, "weight_decay": 0.001, "use_muon":False}
            ]

            optimizer = MuonWithAuxAdam(param_groups)
        
    criterion_crossentropy = nn.CrossEntropyLoss()
    criterion_triplet = TripletLoss(margin=1.0)
    criterion_contrastive_wass = ContrastiveLoss(margin=1.0)
    criterion_contrastive = ContrastiveLoss(margin=1.0)
    
    os.makedirs("/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/logs/logs_flow", exist_ok=True)
    # TensorBoard writer
    writer = SummaryWriter(log_dir='/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/logs/logs_flow')
    
    def train_triplet(epochs=20, dataloaders=None):
        if generator is None:
            raise NotImplementedError
        else:
            vae = generator.vae
            
        total_step = epochs * sum([len(dataloader) for dataloader in dataloaders])
        dataset = []
        for i in test_json:
            dataset += json.load(open(i, "r"))
        best_acc = evaluate_triplet(model, dataset, device, generator, 0)
        # best_acc = 0
        dataloaders = [infiniteloop(dataloader) for dataloader in dataloaders]
        for step in tqdm(range(total_step)):
            model.train()
            
            dataloader_pos_id, dataloader_neg_id = random.sample(range(len(dataloaders)), k=2)
            dataloader = dataloaders[dataloader_pos_id]
            neg_dataloader = dataloaders[dataloader_neg_id]
            try:
                batch = dataloader.__iter__().__next__()
                target_ids = batch['class_id'].long().to(device)
                neg_batch = neg_dataloader.__iter__().__next__()
                while batch['label_A'][0] == neg_batch['label_A'][0]:
                    neg_batch = dataloaders[dataloader_neg_id - 1].__iter__().__next__()
                    dataloader_neg_id = dataloader_neg_id - 1
            except Exception as e:
                print(e)
                continue

            if generator is not None and use_generate:
                generator.args.op_match = True if random.random() < 0.25 else False
                generator.num_steps = random.choice([6, 8, 10, 12, 14, 16, 18])
                generator.args.num_steps = generator.num_steps
                generator.use_gt_vt = True if random.random() > 0.75 else False
                if generator.use_gt_vt:  # 插值的 数据增强。
                    sample_step = random.choice([-1, 1, -1, -2, -3, 2])
                    result = generator.generate(batch, sample_step=sample_step)
                    anchor = result["x0_vaed"] if random.random() < 0.5 else result["samples"]
                    positive = result["x0_vaed"]
                    negative = neg_batch["x0"] if random.random() > 0.5 else neg_batch["x1"]
                    with torch.no_grad():       
                        negative = negative.to(vae.device)    
                        negative = vae.encode(negative).latent_dist
                        negative = negative.sample() * 0.18215
                else:
                    if random.random() < 0.25:
                        anchor, positive = batch["x0"], batch["x1"]
                        anchor, positive = anchor.to(vae.device), positive.to(vae.device)
                        # 压缩至 潜在空间
                        with torch.no_grad():
                            anchor = vae.encode(anchor).latent_dist
                            anchor = anchor.sample() * 0.18215
                            positive = vae.encode(positive).latent_dist
                            positive = positive.sample() * 0.18215
                    else:
                        sample_step = random.choice([-1, 1, -1, -2, -3, 2]) # 考虑信噪比
                        result = generator.generate(batch, sample_step=sample_step)
                        positive = result["samples"]
                        anchor = result["x0_vaed"] if random.random() < 0.5 else result["x1_vaed"]
                    
                    if random.random() < .5:
                        negative = neg_batch["x0"] if random.random() > 0.5 else neg_batch["x1"]
                        with torch.no_grad():
                            negative = negative.to(vae.device)    
                            negative = vae.encode(negative).latent_dist
                            negative = negative.sample() * 0.18215
                    else: # 9月23号 新增。
                        new_neg_batch = deepcopy(neg_batch)
                        new_neg_batch['x0'] = anchor
                        new_neg_batch['label_A'] = batch['label_A']
                        new_neg_batch['x0_path'] = batch['x0_path']
                        if 'label_A_id' in batch:
                            new_neg_batch['label_A_id'] = batch['label_A_id']
                        sample_step = random.choice([-1, 1, -1, -2, -3, 2])
                        result = generator.generate(new_neg_batch, sample_step=sample_step)
                        negative = result["samples"]     
            else:
                anchor, positive = batch["x0"], batch["x1"]
                negative = neg_batch["x0"] if random.random() > 0.5 else neg_batch["x1"]

                anchor, positive, negative = anchor.to(vae.device), positive.to(vae.device), negative.to(vae.device)
                # 压缩至 潜在空间
                with torch.no_grad():
                    anchor = vae.encode(anchor).latent_dist
                    anchor = anchor.sample() * 0.18215
                    positive = vae.encode(positive).latent_dist
                    positive = positive.sample() * 0.18215
                    negative = vae.encode(negative).latent_dist
                    negative = negative.sample() * 0.18215

            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()
            anchor_emb, hidden_state = model.encode(anchor, False, True)
            negative_emb = model.encode(negative, False, False)
            # with torch.no_grad(): # 待优化。。。
            positive_emb = model.encode(positive, False, False)
            ### 三元组损失
            loss = criterion_triplet(anchor_emb, positive_emb, negative_emb, 1, 1.0)
            
            ### 交叉熵损失
            # logits = classifer(hidden_state.detach()) 
            logits = classifer(hidden_state)
            cross_loss = criterion_crossentropy(logits, target_ids)
            loss = loss + 0.025 * cross_loss

            ### 对比损失
            loss_pos = criterion_contrastive(anchor_emb, positive_emb, torch.ones_like(anchor_emb[:, 0]), beta=beta)  # y=1
            loss_neg = criterion_contrastive(anchor_emb, negative_emb, torch.zeros_like(anchor_emb[:, 0]), beta=beta)  # y=0
            loss = loss + loss_pos + loss_neg
            # clip_style_triplet_loss(anchor_emb, positive_emb, negative_emb)
            
            ### Wass 对比损失
            wass_loss_pos = criterion_contrastive_wass(anchor_emb, positive_emb, torch.ones_like(anchor_emb[:, 0]), beta=beta, use_wass=True)  # y=1
            wass_loss_neg = criterion_contrastive_wass(anchor_emb, negative_emb, torch.zeros_like(anchor_emb[:, 0]), beta=0.25*beta, use_wass=True)  # y=0 
            loss = loss + wass_loss_pos + wass_loss_neg
            
            ### Wass 损失 # 前部分等价。
            wass_loss = (beta * cal_wasserstein_loss(anchor_emb, positive_emb).mean() - torch.clamp(1.25 * beta * cal_wasserstein_loss(anchor_emb, negative_emb).mean(), max=1))
            loss = loss + wass_loss 
            
            ### 最小匹配MSE
            mse_loss = min_match_mse_loss(anchor_emb, positive_emb)
            neg_mse_loss = torch.clamp_max(min_match_mse_loss(negative_emb, positive_emb), 1.0)
            loss = loss + 0.05 * mse_loss - 0.12 * neg_mse_loss
            
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            writer.add_scalar('Training/Loss', loss.item(), step)
            writer.add_scalar('Training/Grad Norm', grad_norm.item(), step)
            if step % 10 == 0:
                print(f"Global Step {step}/{total_step} | Loss: {loss.item():.4f} | Grad Norm: {grad_norm.item():.4f}")
            save_step = 1000
            if (step + 1) % save_step == 0:
                print("Eval...")
                dataset = []
                for i in test_json:
                    dataset += json.load(open(i, "r"))
                accuracy = evaluate_triplet(model, dataset, device, generator, step)
                writer.add_scalar('Evaling/Acc', accuracy, step)
                if best_acc < accuracy:
                    torch.save(model.state_dict(), f"/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/best_flow_weights/attention_胃.pt")
                    torch.save(classifer.state_dict(), f"/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/best_flow_weights/attention_胃_classifer.pt")
                    best_acc = accuracy
                else:
                    pass
                
        if best_acc < accuracy:
            torch.save(model.state_dict(), f"/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/best_flow_weights/attention_胃.pt")
            torch.save(classifer.state_dict(), f"/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/best_flow_weights/attention_胃_classifer.pt")
            best_acc = accuracy
        else:
            pass

    ### 加载 checkpoints
    try:
        state_dict = torch.load("/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/best_flow_weights/attention_胃.pt", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        state_dict = torch.load("/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/best_flow_weights/attention_胃_classifer.pt", weights_only=True)
        classifer.load_state_dict(state_dict)
    except:
        pass
    
    model = model.to(device)
    classifer = classifer.to(device)
    label_map = json.load(open("/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/utils/label_map.json", "r"))
    
    def evaluate_triplet(model, dataset, device, generator, step=0, k=5):
        if generator is None:
            raise NotImplementedError
        else:
            vae = generator.vae
        model.eval()
        correct = 0
        errors = {}
        correct_dict = {}
        static = {}
        correct_topk = 0
        total = 0
        label_correct = 0
        l2_correct = 0
        with torch.no_grad():
            for data in tqdm(dataset):
                # if random.random() < 0.8:
                #     continue
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
                anchor = vae.encode(anchor).latent_dist
                anchor = anchor.sample() * 0.18215
                anchor = anchor.to(device)
                anchor_emb, hidden_state = model.encode(anchor, False, True)
                logits = classifer(hidden_state)
                if torch.argmax(logits, dim=1).item() == answer_id:
                    label_correct += 1
                scores = np.zeros((k, len(data['x1_labels'])))
                scores_l2 = np.zeros((k, len(data['x1_labels'])))
                for i in range(k):
                    for idx in range(len(data['x1_labels'])):
                        test_image_dir = data['x1_dirs'][idx]
                        test_images = os.listdir(test_image_dir)
                        test_image_path = os.path.join(test_image_dir, random.choice(test_images))
                        test_image = transform(Image.open(test_image_path).convert("RGB")).to(device).unsqueeze(0)
                        test_image = test_image.to(vae.device)
                        # 压缩至 潜在空间
                       
                        test_image = vae.encode(test_image).latent_dist
                        test_image = test_image.sample() * 0.18215
                        test_image = test_image.to(device)
                                
                        test_emb, _ = model.encode(test_image, False, True)
                        wass = cal_wasserstein_loss(anchor_emb, test_emb, norm=False).squeeze()
                        
                        scores[i, idx] = wass.item()
                        scores_l2[i, idx] = torch.norm(anchor_emb - test_emb, p=2, dim=1).squeeze().item()
                scores_m = np.min(scores, axis=0) 
                scores_l2_m = np.min(scores_l2, axis=0)
                pred_option_id = np.argmin(scores_m)
                pred_option_id2 = np.argmin(scores_l2_m)
                data["pred_option_id"] = pred_option_id
                if answer_id == data['label_B_ids'][pred_option_id]:
                    correct += 1
                    correct_dict[label_A] += 1 
                else:
                    question_path = data['x0'] 
                    errors[label_A].append((question_path, data['x1_labels'][pred_option_id]))
                if answer_id == data['label_B_ids'][pred_option_id2]:
                    l2_correct += 1
                topk_indices = np.argsort(scores_m)[:3]  # 因为分数越小越好
                # 获取这些索引对应的选项 ID
                topk_option_ids = [data['label_B_ids'][idx] for idx in topk_indices]

                # 判断正确答案是否在 top-k 预测中
                if answer_id in topk_option_ids:
                    correct_topk += 1
                    
        # with open('/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/data/errors3.json', 'w', encoding='utf-8') as f:
        #     json.dump(errors, f, ensure_ascii=False, indent=4)
            
        accuracy = correct / total
        print(f"Evaluation Accuracy: {accuracy:.4f} Top 3: {(correct_topk / total):.4f} Label Acc:{(label_correct/total):.4f} L2 Acc:{(l2_correct / total):.4f}")
        return accuracy

    # train_triplet(epochs=5, dataloaders=dataloaders) 

    dataset = []
    for i in test_json:
        dataset += json.load(open(i, "r"))
    evaluate_triplet(model, dataset, device, generator, 0)
    # checkpoinsts = [
    #     "/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/best_flow_weights/attention_tiny.pt", # 7614
    #     "/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/best_flow_weights/attention_tiny2.pt", # 
    # ]

    # # 9s9WGzvh
    # for path in checkpoinsts:
    #     try:
    #         state_dict = torch.load(path, weights_only=True)
    #         print(path)
    #         model.load_state_dict(state_dict, strict=False)
    #         evaluate_triplet(model, dataset, device, generator, 0)
    #         print("-"*50)
    #     except:
    #         continue
