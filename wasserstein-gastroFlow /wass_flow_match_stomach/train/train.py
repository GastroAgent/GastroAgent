import glob
import json
import math
from tqdm import tqdm
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from utils.data_loader import MedicalClSJsonDataset, MedicalJsonDataset
from utils.data_utils import create_dataloaders_by_pairs
from utils.train_utils import infiniteloop
from PIL import Image
from math import sqrt
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from einops.layers.torch import Rearrange
from model_utils.my_loss import *
from model_utils.model import *

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
    batch_size = 16
    train_data_dir = ''
    json_paths = glob.glob( # 需要是 同类的数据。
        f"{train_data_dir}/*.json")
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
        "./GasAgent-main/dataset/eval_data/stomach.json",
    ]
    
    # 初始化模型和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TripletNetwork(pretrained=False, freeze_base=False, model='attention', dy=True).to(device)
    criterion = nn.CrossEntropyLoss()
    use_generate = False
    generator = create_generator(only_vae=(not use_generate))
    os.makedirs("./logs_flow", exist_ok=True)
    writer = SummaryWriter(log_dir='./logs_flow')
    
    ### attention
    classifer = nn.Sequential(
        UpsampleDecoder(embed_dim=512, patch_size=4, out_chans=4),
        # Rearrange('b c h w -> b (c h w)'), 
        Rearrange('b ... -> b (...)'),
        GatedMLPClassifier(input_dim=4 * 16 * 16, hidden_dim1=4096, hidden_dim2=1024, output_dim=256)
    )
    # ### resnet34
    # classifer = nn.Sequential(
    #     GatedMLPClassifier(input_dim=4 * 16 * 16, hidden_dim1=4096, hidden_dim2=1024, output_dim=256)
    # )

    print(model)
    print(classifer)
    criterion = nn.CrossEntropyLoss()
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
                generator.num_steps = random.choice([8, 10, 12, 14, 16, 18, 20, 22, 24])
                generator.args.num_steps = generator.num_steps
                generator.use_gt_vt = True if random.random() > 0.0 else False
                if generator.use_gt_vt:  # 插值的 数据增强。
                    sample_step = random.choice([-1, 1, -1, -2, -3, 2]) # 考虑信噪比
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
                        sample_step = random.choice([-1, 1, -1, -2, -3, 2]) # 考虑信噪比
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
                    torch.save(model.state_dict(), f"./cls_weights/attention_tiny_tanh.pt")
                    torch.save(classifer.state_dict(), f"./cls_weights/attention_tiny_classifer_tanh.pt")
                    best_acc = accuracy
                else:
                    pass
                
        accuracy = evaluate_triplet(model, classifer, dataset, device, generator, 0)      
        if best_acc < accuracy:
            torch.save(model.state_dict(), f"./cls_weights/attention_tiny_tanh.pt")
            torch.save(classifer.state_dict(), f"./cls_weights/attention_tiny_classifer_tanh.pt")
            best_acc = accuracy
        else:
            pass

    ### 加载 checkpoints
    try:
        state_dict = torch.load("./cls_weights/attention_tiny_tanh.pt", weights_only=True)
        model.load_state_dict(state_dict)
        state_dict = torch.load("./cls_weights/attention_tiny_classifer_tanh.pt", weights_only=True)
        classifer.load_state_dict(state_dict)
    except:
        pass
    
    model = model.to(device)
    classifer = classifer.to(device)
    print(model)
    label_map = json.load(open("./utils/label_map.json", "r"))
    
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