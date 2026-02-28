import glob
import json
import numpy as np
from tqdm import tqdm
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import random
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from PIL import Image
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from einops.layers.torch import Rearrange

from flow_matcher_vit import create_generator # 不包含 x0, x1.
from utils.data_loader import MedicalJsonDataset
from utils.train_utils import infiniteloop
from utils.optim import *
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
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.RandomGrayscale(p=1),  # 数据增强：50% 概率灰度化
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    batch_size = 4
    dataloaders = []
    
    json_paths = glob.glob(
        "./data_tsy1/train_json/data_pairs_flow/*.json")
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
        "./dataset/eval_data/new_eval_tsy.json",
    ]
    
    # 初始化模型和优化器
    use_generate = True
    free_classifer = False
    use_muon = False
    beta = 1.024 * 12.35
    generator = create_generator(only_vae=(not use_generate), device="cuda:0")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = TripletNetwork(pretrained=False, freeze_base=False, model='attention')
    model = TripletNetwork(pretrained=False, freeze_base=False, model='attention', dy=True).to(device) # vit-vae
    classifer = nn.Sequential(
        UpsampleDecoder(embed_dim=512, patch_size=4, out_chans=4),
        # Rearrange('b c h w -> b (c h w)'), 
        Rearrange('b ... -> b (...)'),
        GatedMLPClassifier(input_dim=16 * 16, hidden_dim1=1024, hidden_dim2=512, output_dim=256)
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
    
    os.makedirs("./logs/logs_flow", exist_ok=True)
    # TensorBoard writer
    writer = SummaryWriter(log_dir='./logs/logs_flow')
    
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
                generator.args.op_match = True if random.random() < 0.125 else False
                generator.num_steps = random.choice([8, 10, 12, 14, 16, 18])
                generator.args.num_steps = generator.num_steps
                generator.use_gt_vt = True if random.random() > 0.875 else False
                if generator.use_gt_vt:
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
                    
                    if random.random() < 0.5:
                        negative = neg_batch["x0"] if random.random() > 0.5 else neg_batch["x1"]
                        with torch.no_grad():
                            negative = negative.to(vae.device)    
                            negative = vae.encode(negative).latent_dist
                            negative = negative.sample() * 0.18215
                    else:
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
            norms = torch.norm(anchor_emb, p=2, dim=1)
            loss = 0.25 *(
                F.mse_loss(norms, torch.ones_like(norms), reduction='mean') +
                F.mse_loss(torch.norm(negative_emb, p=2, dim=1), torch.ones_like(norms), reduction='mean') +
                F.mse_loss(torch.norm(positive_emb, p=2, dim=1), torch.ones_like(norms), reduction='mean')
            )

            ### 三元组损失
            loss = loss + criterion_triplet(anchor_emb, positive_emb, negative_emb, 1, 1.0)
            
            ### 对比损失 辅助
            # loss_pos = criterion_contrastive(anchor_emb, positive_emb, torch.ones_like(anchor_emb[:, 0]), beta=beta)  # y=1
            # loss_neg = criterion_contrastive(anchor_emb, negative_emb, torch.zeros_like(anchor_emb[:, 0]), beta=beta)  # y=0
            # loss = loss + loss_pos + loss_neg
            loss = loss + clip_style_triplet_loss(anchor_emb, positive_emb, negative_emb) / 2
            loss = loss + clip_style_triplet_loss(positive_emb, anchor_emb, negative_emb) / 2
            loss = loss + 0.01 * clip_loss(anchor_emb, anchor_emb)
            loss = loss + 0.01 * clip_loss(positive_emb, positive_emb)
            loss = loss + 0.01 * clip_loss(negative_emb, negative_emb)

            ## Wass 对比损失
            wass_loss_pos = criterion_contrastive_wass(anchor_emb, positive_emb, torch.ones_like(anchor_emb[:, 0]), beta=beta, use_wass=True)  # y=1
            wass_loss_neg = criterion_contrastive_wass(anchor_emb, negative_emb, torch.zeros_like(anchor_emb[:, 0]), beta=1.25*beta, use_wass=True)  # y=0 
            loss = loss + wass_loss_pos + wass_loss_neg
            
            ### Wass 直接损失
            wass_loss = (beta * cal_wasserstein_loss(anchor_emb, positive_emb).mean() - torch.clamp(0.25 * beta * cal_wasserstein_loss(anchor_emb, negative_emb).mean(), max=1))
            loss = loss + 1.15 * wass_loss 
             
            ### 交叉熵损失 辅助
            # logits = classifer(hidden_state.detach()) 
            logits = classifer(hidden_state)
            cross_loss = criterion_crossentropy(logits, target_ids)
            loss = loss + 0.01 * cross_loss
            
            ## 最小匹配MSE 辅助
            # mse_loss = min_match_mse_loss(anchor_emb, positive_emb)
            # neg_mse_loss = torch.clamp_max(min_match_mse_loss(negative_emb, positive_emb), 1.0)
            # loss = loss + 0.5 * mse_loss - 0.12 * neg_mse_loss
            
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                    torch.save(model.state_dict(), f"./best_flow_weights/attention_tsy_dy_vit.pt")
                    torch.save(classifer.state_dict(), f"./best_flow_weights/attention_tsy_dy_vit_classifer.pt")
                    best_acc = accuracy
                else:
                    pass
                
        if best_acc < accuracy:
            torch.save(model.state_dict(), f"./best_flow_weights/attention_tsy_dy_vit.pt")
            torch.save(classifer.state_dict(), f"./best_flow_weights/attention_tsy_dy_vit_classifer.pt")
            best_acc = accuracy
        else:
            pass

    ### 加载 checkpoints
    try:
        state_dict = torch.load("./best_flow_weights/attention_tsy_dy_vit.pt", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        state_dict = torch.load("./best_flow_weights/attention_tsy_dy_vit_classifer.pt", weights_only=True)
        classifer.load_state_dict(state_dict)
    except:
        pass
    
    model = model.to(device)
    classifer = classifer.to(device)
    label_map = json.load(open("./utils/label_map.json", "r"))

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
                    wass_list.append(min(wass_min)) # 取min或取 sum(wass_A) / len(wass_A), python 无内置 mean。
                pred_option_id = torch.argmin(torch.stack(wass_list, dim=0)).item()
                data["pred_option_id"] = pred_option_id
                if data['label_A_id'] == data['label_B_ids'][pred_option_id]:
                    correct += 1

        accuracy = correct / total
        print(f"Evaluation Accuracy: {accuracy:.4f}")
        return accuracy
    
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
    # train_triplet(epochs=20, dataloaders=dataloaders) 

    dataset = []
    for i in test_json:
        dataset += json.load(open(i, "r"))
    evaluate_triplet(model, dataset, device, generator, 0)
    
    # checkpoinsts = [
    #     "./best_flow_weights/attention_tsy.pt",
    #     "./best_flow_weights/attention_norm_tsy.pt"
    # ]
    # for path in checkpoinsts:
    #     print(path)
    #     try:
    #         state_dict = torch.load(path, weights_only=True)
    #         model.load_state_dict(state_dict, strict=False)
    #         evaluate_triplet(model, dataset, device, generator, 0)
    #     except Exception as e:
    #         print(e)
    #         continue
    #     print("-"*50)
    # model = TripletNetwork(pretrained=False, freeze_base=False, model='attention', dy=True).to(device)
    # state_dict = torch.load("./best_flow_weights/attention_dy_tsy.pt", weights_only=True)
    # model.load_state_dict(state_dict, strict=False)
    # evaluate_triplet(model, dataset, device, generator, 0)