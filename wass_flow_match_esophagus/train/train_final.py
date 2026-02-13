import glob
import json
import math
from tqdm import tqdm
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,4,5"
import random
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.amp
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
from math import sqrt
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from einops.layers.torch import Rearrange
import sys
sys.path.append("/mnt/inaisfs/data/home/tansy_criait/GasAgent-main")
from utils.data_loader import MedicalJsonDataset
from utils.train_utils import infiniteloop
from utils.optim import *
from flow_matcher import create_generator # 不包含 x0, x1.
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
    batch_size = 4
    dataloaders = []
    
    json_paths = glob.glob(
        "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/data_tsy1/train_json/data_pairs_flow/*.json")
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
    
    test_json = [
        "/mnt/inaisfs/data/home/tansy_criait/GasAgent-main/dataset/eval_data/new_eval_tsy.json",
    ]
    
    # 初始化模型和优化器
    free_classifer = False
    use_muon = False
    beta = 1.024 * 1.35
    generator = create_generator(only_vae=False, device="cuda:0", num_device=4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TripletNetwork(pretrained=False, freeze_base=False, model='attention', dy=True).to(device) 

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
    epoch = 20
    scaler = torch.amp.GradScaler()
    criterion_crossentropy = nn.CrossEntropyLoss()
    criterion_triplet = TripletLoss(margin=1.0)
    criterion_contrastive_wass = ContrastiveLoss(margin=1.0)
    criterion_contrastive = ContrastiveLoss(margin=1.0)
    def get_single_batch(batch: dict, index: int):
        data = {}
        for key in batch:
            data[key] = batch[key][index:index+1]
        return data

    os.makedirs("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/logs/logs_flow", exist_ok=True)
    # TensorBoard writer
    writer = SummaryWriter(log_dir='/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/logs/logs_flow')
    
    def train_triplet(epochs=20, dataloaders=None):
        if generator is None:
            raise NotImplementedError
        else:
            vae = generator.vae
            
        total_step = epochs * sum([len(dataloader) for dataloader in dataloaders])
        dataset = []
        for i in test_json:
            dataset += json.load(open(i, "r"))
        best_acc = evaluate_new_triplet(model, dataset, device, generator, 0)
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

            if generator is not None:
                generator.args.op_match = True if random.random() < 0.25 else False
                generator.num_steps = random.choice([2, 4, 8, 10, 8, 10, 8, 4, 2])
                generator.args.num_steps = generator.num_steps
                generator.use_gt_vt = True if random.random() > 0.75 else False
                if generator.use_gt_vt:  # 插值的 数据增强。
                    generator.args.use_gt_vt = generator.need_ut = True
                    generator.args.solver = 'euler'
                else:
                    generator.args.use_gt_vt = generator.need_ut = False
                    generator.args.solver = 'heun'

                batchs_pos_trajectorys = []
                batchs_neg_trajectorys = []
                
                for i in range(batch_size):
                    result = generator.generate(get_single_batch(batch, i), sample_step=-1, mode='sim_stop', return_all_steps=True)
                    trajectorys = result["all_samples"]
                    batchs_pos_trajectorys.append(trajectorys)
                
                neg_batch['x0'] = batch['x0']
                neg_batch['label_A'] = batch['label_A']
                neg_batch['x0_path'] = batch['x0_path'] 
                neg_batch['label_A_id'] = batch['label_A_id'] 
                for i in range(batch_size):
                    result = generator.generate(get_single_batch(batch, i), sample_step=-1, mode='sim_stop', return_all_steps=True)
                    trajectorys = result["all_samples"]
                    batchs_neg_trajectorys.append(trajectorys)

            optimizer.zero_grad()
            pos_costs = torch.zeros(len(batchs_pos_trajectorys)).to("cuda:0")
            neg_costs = torch.zeros(len(batchs_neg_trajectorys)).to("cuda:0")
            for bx, (pos_trajectorys, neg_trajectorys) in enumerate(zip(batchs_pos_trajectorys, batchs_neg_trajectorys)):
                pos_trajectorys = torch.cat(pos_trajectorys, dim=0).to("cuda:0")
                pos_trajectorys_emb = model.encode(pos_trajectorys, False, False)
                
                for i in range(len(pos_trajectorys)-1):
                    last_embeding = pos_trajectorys_emb[i:i+1]
                    cur_embeding = pos_trajectorys_emb[-2:-1]
                    ignore = random.sample(list(range(len(pos_trajectorys)-1)), k=min([int(0.5 * (len(pos_trajectorys)-1)), 1]))
                    if i in ignore:
                        pos_costs[bx] = pos_costs[bx] + cal_wasserstein_loss(last_embeding, cur_embeding).detach()
                    else:
                        pos_costs[bx] = pos_costs[bx] + cal_wasserstein_loss(last_embeding, cur_embeding)

                neg_trajectorys = torch.cat(neg_trajectorys, dim=0).to("cuda:0")
                neg_trajectorys_emb = model.encode(neg_trajectorys, False, False)
                
                for i in range(len(neg_trajectorys)-1):
                    last_embeding = neg_trajectorys_emb[i:i+1]
                    cur_embeding = neg_trajectorys_emb[-2:-1]
                    ignore = random.sample(list(range(len(neg_trajectorys)-1)), k=min([int(0.5 * (len(neg_trajectorys)-1)), 1]))
                    if i in ignore:
                        neg_costs[bx] = neg_costs[bx] + cal_wasserstein_loss(last_embeding, cur_embeding).detach()
                    else:
                        neg_costs[bx] = neg_costs[bx] + cal_wasserstein_loss(last_embeding, cur_embeding)
            
            margin = 0.5  # 可调超参
            loss = torch.relu(pos_costs - neg_costs + margin).mean()
            ### 询问GPT，要求 优化代码。
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            writer.add_scalar('Training/Loss', loss.item(), step)
            writer.add_scalar('Training/Grad Norm', grad_norm.item(), step)
            if step % 10 == 0:
                print(f"Global Step {step}/{total_step} | Loss: {loss.item():.4f} | Grad Norm: {grad_norm.item():.4f}")
            save_step = 2000
            if (step + 1) % save_step == 0:
                print("Eval...")
                dataset = []
                for i in test_json:
                    dataset += json.load(open(i, "r"))
                accuracy = evaluate_new_triplet(model, dataset, device, generator, step)
                writer.add_scalar('Evaling/Acc', accuracy, step)
                if best_acc < accuracy:
                    torch.save(model.state_dict(), f"/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/best_flow_weights/sum3_attention_dy_tsy.pt")
                    torch.save(classifer.state_dict(), f"/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/best_flow_weights/sum3_attention_dy_tsy_classifer.pt")
                    best_acc = accuracy
                else:
                    pass
        if best_acc < accuracy:
            torch.save(model.state_dict(), f"/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/best_flow_weights/sum3_attention_dy_tsy.pt")
            torch.save(classifer.state_dict(), f"/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/best_flow_weights/sum3_attention_dy_tsy_classifer.pt")
            best_acc = accuracy
        else:
            pass

    ### 加载 checkpoints
    try:
        state_dict = torch.load("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/best_flow_weights/attention_dy_tsy.pt", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        state_dict = torch.load("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/best_flow_weights/attention_dy_tsy_classifer.pt", weights_only=True)
        classifer.load_state_dict(state_dict)
    except:
        pass
    
    model = model.to(device)
    classifer = classifer.to(device)
    label_map = json.load(open("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/data_tsy1/label_map.json", "r"))

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
                                cur_embeding = trajectorys_emb[-2:-1]
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

    train_triplet(epochs=epoch, dataloaders=dataloaders) 

    dataset = []
    for i in test_json:
        dataset += json.load(open(i, "r"))
    evaluate_new_triplet(model, dataset, device, generator, 0)