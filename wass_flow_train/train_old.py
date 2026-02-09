import glob
import json
import math
from tqdm import tqdm
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import random
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from sklearn.utils import shuffle
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from utils.data_loader import MedicalJsonDataset
from utils.data_utils import create_dataloaders_by_pairs
from utils.train_utils import infiniteloop
from PIL import Image
from math import sqrt
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from old_flow_matcher import create_generator # 不包含 x0, x1.
from model_utils.model import *
from model_utils.my_loss import *

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
        # transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),], p=0.25), 
        # transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 0.5))], p=0.25), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    transform_grey = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.RandomGrayscale(p=1),  # 数据增强：50% 概率灰度化
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    batch_size = 3
    dataloaders = []
        
    json_paths = glob.glob( # 需要是 同类的数据。
        "/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/simple_data_test/data_pairs/*.json")

    for json_path in tqdm(json_paths):
        dataset = MedicalJsonDataset(
            path=json_path,
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
        dataloaders.append(dataloader)
        # break
    
    test_json = "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/data/食管/eval_all.json"
    
    # 初始化模型和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TripletNetwork(pretrained=False, freeze_base=False, model='attention').to(device)
    criterion = TripletLoss(margin=1.0)
    criterion_contrastive = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.001)
    # random.seed(1234)
    generator = create_generator()
    # generator = None
    use_generate = True
    os.makedirs("/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/logs_flow", exist_ok=True)
    # TensorBoard writer
    writer = SummaryWriter(log_dir='/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/logs_flow')
    
    def train_triplet(model, dataloaders, criterion, optimizer, device="cuda",
                      criterion_contrastive=None, generator=None, cal_wasserstein_loss=None, epochs=20):
        if generator is None:
            raise NotImplementedError
        else:
            vae = generator.vae
        total_step = epochs * sum([len(dataloader) for dataloader in dataloaders])
        dataset = json.load(open(test_json, "r"))
        best_acc = evaluate_triplet(model, dataset, device, generator, 0)
        wass_step = int(total_step * 0.05)
        dataloaders = [infiniteloop(dataloader) for dataloader in dataloaders]
        for step in tqdm(range(total_step)):
            model.train()
            dataloader_pos_id, dataloader_neg_id = random.sample(range(len(dataloaders)), k=2)
            dataloader = dataloaders[dataloader_pos_id]
            neg_dataloader = dataloaders[dataloader_neg_id]
            try:
                batch = dataloader.__iter__().__next__()
                neg_batch = neg_dataloader.__iter__().__next__()
                while batch['label_A'][0] == neg_batch['label_A'][0]:
                    neg_batch = dataloaders[dataloader_neg_id - 1].__iter__().__next__()
            except:
                continue
            if generator is not None and use_generate:
                generator.args.op_match = True if random.random() < 0.30 else False
                generator.num_steps = random.choice([8, 10, 12, 16, 20, 24])
                generator.args.num_steps = generator.num_steps
                generator.use_gt_vt = True if random.random() > 0.5 else False
                if generator.use_gt_vt:  # 插值的 数据增强。
                    sample_step = random.choice([-1, -1, 1, -2])
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
                        sample_step = random.choice([-1, 1, -1, -2]) # 考虑信噪比
                        result = generator.generate(batch, sample_step=sample_step)
                        positive = result["samples"]
                        anchor = result["x0_vaed"]
                    
                    if random.random() < 1.5:
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
                        sample_step = sample_step = random.choice([-1, 1])
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
            anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            if criterion_contrastive is not None:
                loss_pos = criterion_contrastive(anchor_emb, positive_emb, torch.ones_like(anchor_emb[:, 0]))  # y=1
                loss_neg = criterion_contrastive(anchor_emb, negative_emb, torch.zeros_like(anchor_emb[:, 0]))  # y=0
                loss = loss + (loss_pos * 4 / 10 + loss_neg * 6 / 10)
                # clip_style_triplet_loss(anchor_emb, positive_emb, negative_emb)
                    
            if cal_wasserstein_loss is not None and step >= wass_step:
                loss = loss + 0.1024 * (cal_wasserstein_loss(anchor_emb, positive_emb).mean() - 1 * cal_wasserstein_loss(anchor_emb, negative_emb).mean()) 

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            writer.add_scalar('Training/Loss', loss.item(), step)
            writer.add_scalar('Training/Grad Norm', grad_norm.item(), step)
            if step % 10 == 0:
                print(f"Global Step {step}/{total_step} | Loss: {loss.item():.4f} | Grad Norm: {grad_norm.item():.4f}")
            save_step = 250
            if (step + 1) % save_step == 0:
                print("Eval...")
                dataset = json.load(open(test_json, "r"))
                accuracy = evaluate_triplet(model, dataset, device, generator, step)
                writer.add_scalar('Evaling/Acc', accuracy, step)
                if best_acc < accuracy:
                    torch.save(model.state_dict(), f"/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/best_flow_weights/attention_食管.pt")
                    best_acc = accuracy
                else:
                    pass
                
        if best_acc < accuracy:
            torch.save(model.state_dict(), f"/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/best_flow_weights/attention_食管.pt")
            best_acc = accuracy
        else:
            pass

    ### 加载 checkpoints
    try:
        state_dict = torch.load("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/best_flow_weights/attention_食管.pt", weights_only=True) # 0.
        model.load_state_dict(state_dict)
    except:
        pass

    model = model.to(device)
    print(model)
    label_map = json.load(open("/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/utils/label_map.json", "r"))
    
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
                anchor_emb = model.encode(anchor, True)
                scores = np.zeros((k, len(data['x1_labels'])))
                for i in range(k):
                    for idx in range(len(data['x1_labels'])):
                        test_image_dir = data['x1_dirs'][idx]
                        test_images = os.listdir(test_image_dir)
                        test_image_path = os.path.join(test_image_dir, random.choice(test_images))
                        test_image = transform(Image.open(test_image_path).convert("RGB")).to(device).unsqueeze(0)
                        test_image = test_image.to(vae.device)
                        # 压缩至 潜在空间
                        with torch.no_grad():
                            test_image = vae.encode(test_image).latent_dist
                            test_image = test_image.sample() * 0.18215
                        test_image = test_image.to(device)
                            
                        test_emb = model.encode(test_image, True)
                        wass = cal_wasserstein_loss(anchor_emb, test_emb).squeeze()
                        scores[i, idx] = wass.item()
                scores_m = np.min(scores, axis=0) 

                pred_option_id = np.argmin(scores_m)
                data["pred_option_id"] = pred_option_id
                if answer_id == data['label_B_ids'][pred_option_id]:
                    correct += 1
                    correct_dict[label_A] += 1 
                else:
                    question_path = data['x0'] 
                    errors[label_A].append(question_path)
                
                topk_indices = np.argsort(scores_m)[:3]  # 因为分数越小越好
                # 获取这些索引对应的选项 ID
                topk_option_ids = [data['label_B_ids'][idx] for idx in topk_indices]

                # 判断正确答案是否在 top-k 预测中
                if answer_id in topk_option_ids:
                    correct_topk += 1
        accuracy = correct / total
        print(f"Evaluation Accuracy: {accuracy:.4f} Top 3: {(correct_topk / total):.4f}")
        return accuracy
    
    train_triplet(model, dataloaders, criterion=criterion, optimizer=optimizer, device=device,
                  generator=generator, criterion_contrastive=criterion_contrastive,
                  cal_wasserstein_loss=cal_wasserstein_loss, epochs=5) 

    dataset = json.load(open(test_json, "r"))
    evaluate_triplet(model, dataset, device, generator, 0, 5)
