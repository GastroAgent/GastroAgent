import glob
import json
from tqdm import tqdm
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
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
    batch_size = 4
    dataloaders = []
    
    json_paths = glob.glob(
        "/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/data/Disease_Extra/data_pairs/*.json")
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
        "/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/data/Disease_Extra/exam_dataset_all.json",
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

            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        else:
            hidden_weights = [p for p in model.parameters() if p.ndim >= 2]      # 如 Linear.weight
            other_params = [p for p in model.parameters() if p.ndim < 2]        # 如 bias

            # 构建参数组
            param_groups = [
                dict(params=hidden_weights, use_muon=True, lr=0.001, weight_decay=0.01),
                dict(params=other_params, use_muon=False, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
                {"params": classifer.parameters(), "lr": 1e-4, "weight_decay": 0.001, "use_muon":False}
            ]

            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        
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
                generator.use_gt_vt = True if random.random() > 0.35 else False
                if generator.use_gt_vt:  # 插值的 数据增强。
                    sample_step = random.choice([-1, 1, -1, -2, -3, 2])
                    result = generator.generate(batch, sample_step=sample_step)
                    positive = result["samples"]
                    anchor = result["x0_vaed"] if random.random() < 0.5 else result["x1_vaed"]
                    negative = neg_batch["x0"] if random.random() > 0.5 else neg_batch["x1"]
                    with torch.no_grad():       
                        negative = negative.to(vae.device)    
                        negative = vae.encode(negative).latent_dist
                        negative = negative.sample() * 0.18215
                else:
                    if random.random() < 0.5:
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
            loss = loss + 0.05 * cross_loss

            ### 对比损失
            loss_pos = criterion_contrastive(anchor_emb, positive_emb, torch.ones_like(anchor_emb[:, 0]), beta=beta)  # y=1
            loss_neg = criterion_contrastive(anchor_emb, negative_emb, torch.zeros_like(anchor_emb[:, 0]), beta=beta)  # y=0
            loss = loss + loss_pos + loss_neg
            # clip_style_triplet_loss(anchor_emb, positive_emb, negative_emb)
            
            ### Wass 对比损失
            wass_loss_pos = criterion_contrastive_wass(anchor_emb, positive_emb, torch.ones_like(anchor_emb[:, 0]), beta=beta, use_wass=True)  # y=1
            wass_loss_neg = criterion_contrastive_wass(anchor_emb, negative_emb, torch.zeros_like(anchor_emb[:, 0]), beta=1.25*beta, use_wass=True)  # y=0 
            loss = loss + wass_loss_pos + wass_loss_neg
            
            # ### Wass 损失 # 前部分等价。
            # wass_loss = (beta * cal_wasserstein_loss(anchor_emb, positive_emb).mean() - torch.clamp(1.25 * beta * cal_wasserstein_loss(anchor_emb, negative_emb).mean(), max=1))
            # loss = loss + 0.5 * wass_loss 
            
            ### 最小匹配MSE
            # mse_loss = min_match_mse_loss(anchor_emb, positive_emb)
            # neg_mse_loss = torch.clamp_max(min_match_mse_loss(negative_emb, positive_emb), 1.0)
            # loss = loss + 0.05 * mse_loss - 0.12 * neg_mse_loss
            
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
                    torch.save(model.state_dict(), f"/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/best_flow_weights/attention_disease_extra.pt")
                    torch.save(classifer.state_dict(), f"/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/best_flow_weights/attention_disease_extra_classifer.pt")
                    best_acc = accuracy
                else:
                    pass
                
        if best_acc < accuracy:
            torch.save(model.state_dict(), f"/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/best_flow_weights/attention_disease_extra.pt")
            torch.save(classifer.state_dict(), f"/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/best_flow_weights/attention_disease_extra_classifer.pt")
            best_acc = accuracy
        else:
            pass

    ### 加载 checkpoints
    try:
        state_dict = torch.load("/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/best_flow_weights/attention_disease_extra.pt", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        state_dict = torch.load("/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/best_flow_weights/attention_disease_extra_classifer.pt", weights_only=True)
        classifer.load_state_dict(state_dict)
    except:
        pass
    
    model = model.to(device)
    classifer = classifer.to(device)
    label_map = json.load(open("/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/utils/label_map.json", "r"))
    
    def evaluate_triplet(model, dataset, device, generator, step=0, k=1):
        if generator is None:
            raise NotImplementedError
        else:
            vae = generator.vae
        model.eval()
        correct = 0
        total = 0
        l2_correct = 0

        answer_map = {"option_A": 0, "option_B": 1, "option_C": 2, "option_D": 3}
        with torch.no_grad():
            for data in tqdm(dataset):
                if 'image' not in data:
                    data["image"] = data['image_paths'][0]
                if not os.path.exists(data["image"]):
                    continue
                total += 1
                answer_id = answer_map[data["answer"]]
                anchor = transform(Image.open(data["image"]).convert("RGB")).to(device).unsqueeze(0)
                anchor = anchor.to(vae.device)
                # 压缩至 潜在空间
                with torch.no_grad():
                    anchor = vae.encode(anchor).latent_dist
                    anchor = anchor.sample() * 0.18215
                anchor = anchor.to(device)
                anchor_emb = model.encode(anchor, False)
                wass_A = wass_B = wass_C = wass_D = 0
                l2_A = l2_B = l2_C = l2_D = 0
                for _ in range(k):
                    if "option_A_dir" in data:
                        test_image_dir = data["option_A_dir"]
                        test_images = os.listdir(test_image_dir)
                        test_image_path = os.path.join(test_image_dir, random.choice(test_images))
                        test_image = transform(Image.open(test_image_path).convert("RGB")).to(device).unsqueeze(0)
                        test_image = test_image.to(vae.device)
                        # 压缩至 潜在空间
                        with torch.no_grad():
                            test_image = vae.encode(test_image).latent_dist
                            test_image = test_image.sample() * 0.18215
                        test_image = test_image.to(device)
                            
                        test_emb = model.encode(test_image, False)
                        wass_A += cal_wasserstein_loss(anchor_emb, test_emb).squeeze()
                        l2_A += torch.norm(anchor_emb - test_emb, p=2, dim=1).squeeze()
                    else:
                        wass_A = torch.Tensor([float('inf')]).to(device).squeeze()
                        l2_A = torch.Tensor([float('inf')]).to(device).squeeze()

                    if "option_B_dir" in data:
                        test_image_dir = data["option_B_dir"]
                        test_images = os.listdir(test_image_dir)
                        test_image_path = os.path.join(test_image_dir, random.choice(test_images))
                        test_image = transform(Image.open(test_image_path).convert("RGB")).to(device).unsqueeze(0)
                        test_image = test_image.to(vae.device)
                        # 压缩至 潜在空间
                        with torch.no_grad():
                            test_image = vae.encode(test_image).latent_dist
                            test_image = test_image.sample() * 0.18215
                        test_image = test_image.to(device)
                        
                        test_emb = model.encode(test_image, False)
                        wass_B += cal_wasserstein_loss(anchor_emb, test_emb).squeeze()
                        l2_B += torch.norm(anchor_emb - test_emb, p=2, dim=1).squeeze()
                    else:
                        wass_B = torch.Tensor([float('inf')]).to(device).squeeze()
                        l2_B = torch.Tensor([float('inf')]).to(device).squeeze()

                    if "option_C_dir" in data:
                        test_image_dir = data["option_C_dir"]
                        test_images = os.listdir(test_image_dir)
                        test_image_path = os.path.join(test_image_dir, random.choice(test_images))
                        test_image = transform(Image.open(test_image_path).convert("RGB")).to(device).unsqueeze(0)
                        test_image = test_image.to(vae.device)
                        # 压缩至 潜在空间
                        with torch.no_grad():
                            test_image = vae.encode(test_image).latent_dist
                            test_image = test_image.sample() * 0.18215
                        test_image = test_image.to(device)
                        
                        test_emb = model.encode(test_image, False)
                        wass_C += cal_wasserstein_loss(anchor_emb, test_emb).squeeze()
                        l2_C += torch.norm(anchor_emb - test_emb, p=2, dim=1).squeeze()
                    else:
                        wass_C = torch.Tensor([float('inf')]).to(device).squeeze()
                        l2_C = torch.Tensor([float('inf')]).to(device).squeeze()

                    if "option_D_dir" in data:
                        test_image_dir = data["option_D_dir"]
                        test_images = os.listdir(test_image_dir)
                        test_image_path = os.path.join(test_image_dir, random.choice(test_images))
                        test_image = transform(Image.open(test_image_path).convert("RGB")).to(device).unsqueeze(0)
                        test_image = test_image.to(vae.device)
                        # 压缩至 潜在空间
                        with torch.no_grad():
                            test_image = vae.encode(test_image).latent_dist
                            test_image = test_image.sample() * 0.18215
                        test_image = test_image.to(device)
                        test_emb = model.encode(test_image, False)
                        wass_D += cal_wasserstein_loss(anchor_emb, test_emb).squeeze()
                        l2_D += torch.norm(anchor_emb - test_emb, p=2, dim=1).squeeze()
                    else:
                        wass_D = torch.Tensor([float('inf')]).to(device).squeeze()
                        l2_D = torch.Tensor([float('inf')]).to(device).squeeze()

                pred_option_id = torch.argmin(torch.stack([wass_A, wass_B, wass_C, wass_D], dim=0)).item()
                pred_option_id_l2 = torch.argmin(torch.stack([l2_A, l2_B, l2_C, l2_D], dim=0)).item()
                data["pred_option_id"] = pred_option_id
                data["pred_option_id_l2"] = pred_option_id_l2
                if answer_id == pred_option_id:
                    correct += 1
                if answer_id == pred_option_id_l2:
                    l2_correct += 1

        accuracy = correct / total
        print(f"Evaluation Accuracy: {accuracy:.4f}")
        l2_accuracy = l2_correct / total
        print(f"L2 Evaluation Accuracy: {l2_accuracy:.4f}")
        return accuracy

    # train_triplet(epochs=20, dataloaders=dataloaders) 

    dataset = []
    for i in test_json:
        dataset += json.load(open(i, "r"))
    # evaluate_triplet(model, dataset, device, generator, 0)
    # checkpoinsts = [
    #     "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/best_flow_weights/attention_tiny.pt", # 7614
    #     "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/best_flow_weights/attention_tiny2.pt", # 
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

    def evaluate_triplet_new(model, dataset, device, generator, step=0, k=1):
        if generator is None:
            raise NotImplementedError
        else:
            vae = generator.vae
        model.eval()
        correct = 0
        total = 0
        l2_correct = 0

        answer_map = {"option_A": 0, "option_B": 1, "option_C": 2, "option_D": 3}
        with torch.no_grad():
            for data in tqdm(dataset):
                if 'image' not in data:
                    data["image"] = data['image_paths'][0]
                if not os.path.exists(data["image"]):
                    continue
                total += 1
                answer_id = answer_map[data["answer"]]
                anchor = transform(Image.open(data["image"]).convert("RGB")).to(device).unsqueeze(0)
                anchor = anchor.to(vae.device)
                # 压缩至 潜在空间
                with torch.no_grad():
                    anchor = vae.encode(anchor).latent_dist
                    anchor = anchor.sample() * 0.18215
                anchor = anchor.to(device)
                anchor_emb = model.encode(anchor, False)
                wass_A = []
                wass_B = []
                wass_C = []
                wass_D = []
                for _ in range(k):
                    if "option_A_dir" in data:
                        test_image_dir = data["option_A_dir"]
                        test_images = os.listdir(test_image_dir)
                        test_image_path = os.path.join(test_image_dir, random.choice(test_images))
                        test_image = transform(Image.open(test_image_path).convert("RGB")).to(device).unsqueeze(0)
                        test_image = test_image.to(vae.device)
                        # 压缩至 潜在空间
                        with torch.no_grad():
                            test_image = vae.encode(test_image).latent_dist
                            test_image = test_image.sample() * 0.18215
                        test_image = test_image.to(device)
                            
                        test_emb = model.encode(test_image, False)
                        wass_A.append(cal_wasserstein_loss(anchor_emb, test_emb).squeeze().item())
                    else:
                        wass_A = [float('inf')]

                    if "option_B_dir" in data:
                        test_image_dir = data["option_B_dir"]
                        test_images = os.listdir(test_image_dir)
                        test_image_path = os.path.join(test_image_dir, random.choice(test_images))
                        test_image = transform(Image.open(test_image_path).convert("RGB")).to(device).unsqueeze(0)
                        test_image = test_image.to(vae.device)
                        # 压缩至 潜在空间
                        with torch.no_grad():
                            test_image = vae.encode(test_image).latent_dist
                            test_image = test_image.sample() * 0.18215
                        test_image = test_image.to(device)
                        
                        test_emb = model.encode(test_image, False)
                        wass_B.append(cal_wasserstein_loss(anchor_emb, test_emb).squeeze().item())
                    else:
                        wass_B = [float('inf')]

                    if "option_C_dir" in data:
                        test_image_dir = data["option_C_dir"]
                        test_images = os.listdir(test_image_dir)
                        test_image_path = os.path.join(test_image_dir, random.choice(test_images))
                        test_image = transform(Image.open(test_image_path).convert("RGB")).to(device).unsqueeze(0)
                        test_image = test_image.to(vae.device)
                        # 压缩至 潜在空间
                        with torch.no_grad():
                            test_image = vae.encode(test_image).latent_dist
                            test_image = test_image.sample() * 0.18215
                        test_image = test_image.to(device)
                        
                        test_emb = model.encode(test_image, False)
                        wass_C.append(cal_wasserstein_loss(anchor_emb, test_emb).squeeze().item())
                    else:
                        wass_C = [float('inf')]

                    if "option_D_dir" in data:
                        test_image_dir = data["option_D_dir"]
                        test_images = os.listdir(test_image_dir)
                        test_image_path = os.path.join(test_image_dir, random.choice(test_images))
                        test_image = transform(Image.open(test_image_path).convert("RGB")).to(device).unsqueeze(0)
                        test_image = test_image.to(vae.device)
                        # 压缩至 潜在空间
                        with torch.no_grad():
                            test_image = vae.encode(test_image).latent_dist
                            test_image = test_image.sample() * 0.18215
                        test_image = test_image.to(device)
                        test_emb = model.encode(test_image, False)
                        wass_D.append(cal_wasserstein_loss(anchor_emb, test_emb).squeeze().item())
                    else:
                        wass_D = [float('inf')]
                wass_A = torch.Tensor([min(wass_A)])
                wass_B = torch.Tensor([min(wass_B)])
                wass_C = torch.Tensor([min(wass_C)])
                wass_D = torch.Tensor([min(wass_D)])
                pred_option_id = torch.argmin(torch.stack([wass_A, wass_B, wass_C, wass_D], dim=0)).item()
                data["pred_option_id"] = pred_option_id
                if answer_id == pred_option_id:
                    correct += 1

        accuracy = correct / total
        print(f"Evaluation Accuracy: {accuracy:.4f}")
        return accuracy

    def evaluate_new_triplet_option(model, dataset, device, generator, step=0, k=1):
        if generator is None:
            raise NotImplementedError

        model.eval()
        correct = 0
        total = 0
        answer_map = {"option_A": 0, "option_B": 1, "option_C": 2, "option_D": 3}
        with torch.no_grad():
            for data in tqdm(dataset):
                batch = {
                    # 'caption': [data['gt_answer']],
                    'x0': transform(Image.open(data['x0'])).unsqueeze(0),
                    # 'x1': transform(data['x1']),
                    'label_A': [data['gt_answer']],
                    # 'label_B': [data['label_B']],
                    # 'class_id': torch.LongTensor([label_map[data['label_B']]]),
                    'x0_path': [data['x0']],
                    # 'x1_path': [data['x1_path']],
                    'label_A_id': [label_map[data['gt_answer']]],
                    # 'label_B_id': data['label_B_id'].unsqueeze(0),
                    # 'hint': data['hint'].unsqueeze(0),
                }
                    
                total += 1
                answer_id = answer_map[data["answer"]]
                wass_A_list = []
                for _ in range(k):
                    ### 选项A
                    batch_A = deepcopy(batch)
                    batch_A['label_B'] = [data['option_A']]
                    batch_A['caption'] = [data['option_A']]
                    batch_A['class_id'] = torch.LongTensor([label_map[data['option_A']]])
                    batch_A['x1_path'] = [os.path.join(data['option_A_dir'], random.choice(os.listdir(data['option_A_dir'])))]
                    batch_A['label_B_id'] = [label_map[data['option_A']]]
                    batch_A['hint'] = transform_grey(Image.open(batch_A['x1_path'][0])).unsqueeze(0)
                    batch_A['x1'] = transform(Image.open(batch_A['x1_path'][0])).unsqueeze(0)

                    result = generator.generate(batch_A, sample_step=-1, mode='sim_stop', return_all_steps=True)
                    trajectorys = result["all_samples"]
                    wass_A = 0
                    with torch.no_grad():
                        trajectorys = torch.cat(trajectorys, dim=0).to("cuda:0")
                        trajectorys_emb = model.encode(trajectorys, False, False)
                        
                        for i in range(len(trajectorys)-1):
                            last_embeding = trajectorys_emb[i:i+1]
                            cur_embeding = trajectorys_emb[i+1:i+2]
                            wass_A += cal_wasserstein_loss(last_embeding, cur_embeding).squeeze()
                    wass_A_list.append(wass_A)
                wass_A = min(wass_A_list)

                wass_B_list = []
                for _ in range(k):
                    ### 选项B
                    batch_B = deepcopy(batch)
                    batch_B['label_B'] = [data['option_B']]
                    batch_B['caption'] = [data['option_B']]
                    batch_B['class_id'] = torch.LongTensor([label_map[data['option_B']]])
                    batch_B['x1_path'] = [os.path.join(data['option_B_dir'], random.choice(os.listdir(data['option_B_dir'])))]
                    batch_B['label_B_id'] = [label_map[data['option_B']]]
                    batch_B['hint'] = transform_grey(Image.open(batch_B['x1_path'][0])).unsqueeze(0)
                    batch_B['x1'] = transform(Image.open(batch_B['x1_path'][0])).unsqueeze(0)

                    result = generator.generate(batch_B, sample_step=-1, mode='sim_stop', return_all_steps=True)
                    trajectorys = result["all_samples"]
                    wass_B = 0
                    with torch.no_grad():
                        trajectorys = torch.cat(trajectorys, dim=0).to("cuda:0")
                        trajectorys_emb = model.encode(trajectorys, False, False)
                        for i in range(len(trajectorys)-1):
                            last_embeding = trajectorys_emb[i:i+1]
                            cur_embeding = trajectorys_emb[i+1:i+2]
                            wass_B += cal_wasserstein_loss(last_embeding, cur_embeding).squeeze()
                    wass_B_list.append(wass_B)
                wass_B = min(wass_B_list)
            
                wass_C_list = []
                for _ in range(k):
                    batch_C = deepcopy(batch)
                    batch_C['label_B'] = [data['option_C']]
                    batch_C['caption'] = [data['option_C']]
                    batch_C['class_id'] = torch.LongTensor([label_map[data['option_C']]])
                    batch_C['x1_path'] = [os.path.join(data['option_C_dir'], random.choice(os.listdir(data['option_C_dir'])))]
                    batch_C['label_B_id'] = [label_map[data['option_C']]]
                    batch_C['hint'] = transform_grey(Image.open(batch_C['x1_path'][0])).unsqueeze(0)
                    batch_C['x1'] = transform(Image.open(batch_C['x1_path'][0])).unsqueeze(0)

                    result = generator.generate(batch_C, sample_step=-1, mode='sim_stop', return_all_steps=True)
                    trajectorys = result["all_samples"]
                    wass_C = 0
                    with torch.no_grad():
                        trajectorys = torch.cat(trajectorys, dim=0).to("cuda:0")
                        trajectorys_emb = model.encode(trajectorys, False, False)
                        for i in range(len(trajectorys)-1):
                            last_embeding = trajectorys_emb[i:i+1]
                            cur_embeding = trajectorys_emb[i+1:i+2]
                            wass_C += cal_wasserstein_loss(last_embeding, cur_embeding).squeeze()
                    wass_C_list.append(wass_C)
                wass_C = min(wass_C_list)

                wass_D_list = []
                for _ in range(k):
                    ### 选项D
                    batch_D = deepcopy(batch)
                    batch_D['label_B'] = [data['option_D']]
                    batch_D['caption'] = [data['option_D']]
                    batch_D['class_id'] = torch.LongTensor([label_map[data['option_D']]])
                    batch_D['x1_path'] = [os.path.join(data['option_D_dir'], random.choice(os.listdir(data['option_D_dir'])))]
                    batch_D['label_B_id'] = [label_map[data['option_D']]]
                    batch_D['hint'] = transform_grey(Image.open(batch_D['x1_path'][0])).unsqueeze(0)
                    batch_D['x1'] = transform(Image.open(batch_D['x1_path'][0])).unsqueeze(0)

                    result = generator.generate(batch_D, sample_step=-1, mode='sim_stop', return_all_steps=True)
                    trajectorys = result["all_samples"]
                    wass_D = 0
                    with torch.no_grad():
                        trajectorys = torch.cat(trajectorys, dim=0).to("cuda:0")
                        trajectorys_emb = model.encode(trajectorys, False, False)
                        
                        for i in range(len(trajectorys)-1):
                            last_embeding = trajectorys_emb[i:i+1]
                            cur_embeding = trajectorys_emb[i+1:i+2]
                            wass_D += cal_wasserstein_loss(last_embeding, cur_embeding).squeeze()
                    wass_D_list.append(wass_D)
                wass_D = min(wass_D_list)

                pred_option_id = torch.argmin(torch.stack([wass_A, wass_B, wass_C, wass_D], dim=0)).item()
                data["pred_option_id"] = pred_option_id
                if answer_id == pred_option_id:
                    correct += 1

        accuracy = correct / total
        print(f"Evaluation Accuracy: {accuracy:.4f}")
        return accuracy

    def evaluate_new_triplet(model, dataset, device, generator, step=0, k=1):
        if generator is None:
            raise NotImplementedError

        model.eval()
        correct = 0
        total = 0
      
        with torch.no_grad():
            for data in tqdm(dataset):
                batch = {
                    # 'caption': [data['gt_answer']],
                    'x0': transform(Image.open(data['x0'])).unsqueeze(0),
                    # 'x1': transform(data['x1']),
                    'label_A': [data['label_A']],
                    # 'label_B': [data['label_B']],
                    # 'class_id': torch.LongTensor([label_map[data['label_B']]]),
                    'x0_path': [data['x0']],
                    # 'x1_path': [data['x1_path']],
                    'label_A_id': [data['label_A_id']],
                    # 'label_B_id': data['label_B_id'].unsqueeze(0),
                    # 'hint': data['hint'].unsqueeze(0),
                }
                    
                total += 1
                wass_list = []

                for i in range(len(data['x1_labels'])):
                    wass_min = []
                    for _ in range(k):
                        batch_A = deepcopy(batch)
                        batch_A['label_B'] = [data['x1_labels'][i]]
                        batch_A['caption'] = [data['x1_labels'][i]]
                        batch_A['class_id'] = torch.LongTensor([data['label_B_ids'][i]])
                        batch_A['x1_path'] = [os.path.join(data['x1_dirs'][i], random.choice(os.listdir(data['x1_dirs'][i])))]
                        batch_A['label_B_id'] = [data['label_B_ids'][i]]
                        batch_A['hint'] = transform_grey(Image.open(batch_A['x1_path'][0])).unsqueeze(0)
                        batch_A['x1'] = transform(Image.open(batch_A['x1_path'][0])).unsqueeze(0)

                        result = generator.generate(batch_A, sample_step=-1, mode='sim_stop', return_all_steps=True)
                        trajectorys = result["all_samples"]
                        wass = 0
                        with torch.no_grad():
                            trajectorys = torch.cat(trajectorys, dim=0).to("cuda:0")
                            trajectorys_emb = model.encode(trajectorys, False, False)
                            
                            for i in range(len(trajectorys)-1):
                                last_embeding = trajectorys_emb[i:i+1]
                                cur_embeding = trajectorys_emb[i+1:i+2]
                                wass += cal_wasserstein_loss(last_embeding, cur_embeding).squeeze()
                        wass_min.append(wass)
                    wass_list.append(min(wass_min))
                pred_option_id = torch.argmin(torch.stack(wass_list, dim=0)).item()
                data["pred_option_id"] = pred_option_id
                if data['label_A_id'] == data['label_B_ids'][pred_option_id]:
                    correct += 1

        accuracy = correct / total
        print(f"Evaluation Accuracy: {accuracy:.4f}")
        return accuracy
        
    evaluate_new_triplet_option(model, dataset, device, generator, 0, 5)