import glob
import json
from tqdm import tqdm
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import random
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from PIL import Image
from copy import deepcopy

from old_flow_matcher import create_generator # 不包含 x0, x1.
from utils.data_loader_test import MedicalJsonDataset
from utils.data_utils_test import create_dataloaders_by_pairs
from utils.train_utils import infiniteloop
from old_loss import *
from old_model import *

if __name__ == '__main__':
    # 加载数据集
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_B = transforms.Compose([
        # transforms.RandomApply([        
        #     transforms.RandomRotation(
        #         degrees=30,
        #         expand=True,            # 是否扩展画布以容纳完整图像（True会改变图像大小）
        #         center=None,             # 旋转中心，默认中心点；可设为 (x, y)
        #         fill=(0, 0, 0)           # 填充颜色，例如填白色：(255, 255, 255)
        #     )], p=0.25),
        transforms.Resize((512, 512)), 
        transforms.RandomHorizontalFlip(p=0.25), 
        transforms.RandomVerticalFlip(p=0.25),
        # transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),], p=0.25), 
        # transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 0.5))], p=0.25), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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
        transforms.RandomHorizontalFlip(p=0.25), 
        transforms.RandomVerticalFlip(p=0.25),
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
    batch_size = 4
    dataloaders = []
    # json_paths = glob.glob( # 需要是 同类的数据。
    #     "/mnt/inaisfs/data/home/tansy_criait/flow_match/data/mask_data_pairs/*.json")
    
    # for json_path in json_paths:
    #     dataset = MedicalJsonDataset(
    #         path=json_path,
    #         transform=transform,
    #         hint_transform=transform_grey,
    #         transform_A=transform_A,
    #         transform_B=transform_B,
    #     )
    #     if len(dataset) < 8:
    #         continue
    #     dataloader = DataLoader(
    #         dataset,
    #         batch_size=8,
    #         shuffle=True,
    #         num_workers=0,
    #         drop_last=True,
    #     )
    #     dataloaders.append(dataloader)
    
    # for json_path in json_paths:
    #     dataset = MedicalJsonDataset(
    #         path=json_path,
    #         transform=transform,
    #         hint_transform=transform_grey,
    #         transform_A=transform,
    #         transform_B=transform,
    #     )
    #     if len(dataset) < 8:
    #         continue
    #     dataloader = DataLoader(
    #         dataset,
    #         batch_size=8,
    #         shuffle=True,
    #         num_workers=0,
    #         drop_last=True,
    #     )
    #     dataloaders.append(dataloader)
        
    json_paths = glob.glob( # 需要是 同类的数据。
        "/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/data/Disease/data_pairs/*.json")
    for json_path in tqdm(json_paths):
        dataset = MedicalJsonDataset(
            path=json_path,
            transform=transform,
            hint_transform=transform_grey,
            transform_A=transform,
            transform_B=transform,
        )
        if len(dataset) < 8:
            continue
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        dataloaders.append(dataloader)
        break
    for json_path in tqdm(json_paths):
        dataset = MedicalJsonDataset(
            path=json_path,
            transform=transform,
            hint_transform=transform_grey,
            transform_A=transform_A,
            transform_B=transform_B,
        )
        if len(dataset) < 8:
            continue
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        dataloaders.append(dataloader)
        break
    pairs = [
        ("食管炎A级", "食管炎A级"), ("食管炎B级", "食管炎B级"), ("食管炎C级", "食管炎C级"), ("食管炎D级", "食管炎D级"),
    ]
    caption_map = {
        '食管炎A级': '食管炎A级',
        '食管炎C级': '食管炎C级',
        '食管炎D级': '食管炎D级',
        '食管炎B级': '食管炎B级'
    }
    dataloaders.extend(create_dataloaders_by_pairs("/mnt/inaisfs/data/home/tansy_criait/flow_match/data/食管_Cropped2",
                                                   pairs, batchsize=batch_size, caption_map=caption_map,
                                                    transform_A=transform_A, transform=transform, transform_B=transform_B,
    ))
    dataloaders.extend(create_dataloaders_by_pairs("/mnt/inaisfs/data/home/tansy_criait/flow_match/data/食管_Cropped2",
                                                   pairs, batchsize=batch_size, caption_map=caption_map,
                                                    transform_A=transform, transform=transform, transform_B=transform,
    ))
    
    test_json = "/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/data/Disease/exam_dataset_419.json"
    
    # 初始化模型和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TripletNetwork(pretrained=False, freeze_base=False, model='resnet34').to(device)
    # model = TripletNetwork(pretrained=False, freeze_base=False, model='attention').to(device)
    criterion = TripletLoss(margin=1.0)
    criterion_contrastive = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0001)

    generator = create_generator()
    # generator = None
    use_generate = True
    beta = 1.024
    def train_triplet(model, dataloaders, criterion, optimizer, device="cuda",
                      criterion_contrastive=None, generator=None, cal_wasserstein_loss=None, epochs=20):
        if generator is None:
            raise NotImplementedError
        else:
            vae = generator.vae
        total_step = epochs * sum([len(dataloader) for dataloader in dataloaders])
        dataset = json.load(open(test_json, "r"))
        best_acc = evaluate_triplet(model, dataset, device, generator, 0)
        wass_step = int(total_step * 0.1)
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
                generator.args.op_match = True if random.random() < 0.5 else False
                generator.num_steps = random.choice([4, 6, 8, 10, 12])
                generator.args.num_steps = generator.num_steps
                if generator.use_gt_vt:  # 插值的 数据增强。
                    sample_step = random.choice([-1, -1, 1])
                    result = generator.generate(batch, sample_step=sample_step)
                    anchor = result["x0"] if random.random() < 0.5 else result["samples"]
                    positive = result["x1"]
                    negative = neg_batch["x0"]
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
                        sample_step = random.choice([-1, 1, -1, -2]) # 考虑信噪比
                        result = generator.generate(batch, sample_step=sample_step)
                        positive = result["samples"]
                        anchor = result["x0_vaed"]
                    
                    if random.random() < 0.5:
                        negative = neg_batch["x0"]
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
                negative = neg_batch["x0"]

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

            if step < wass_step:
                loss = criterion(anchor_emb, positive_emb, negative_emb)

                if criterion_contrastive is not None:
                    # 2. Contrastive Loss（直接从三元组中构造样本对）
                    # 正样本对 (anchor, positive)
                    loss_pos = criterion_contrastive(anchor_emb, positive_emb, torch.ones_like(anchor_emb[:, 0]))  # y=1

                    # 负样本对 (anchor, negative)
                    loss_neg = criterion_contrastive(anchor_emb, negative_emb, torch.zeros_like(anchor_emb[:, 0]))  # y=0

                    loss = loss + (loss_pos + loss_neg) / 2

                    # clip_style_triplet_loss(anchor_emb, positive_emb, negative_emb)
                    
            if cal_wasserstein_loss is not None and step >= wass_step:
                ### 三元组 wass loss
                ## loss = torch.mean(torch.clamp(d_pos - d_neg + self.margin, min=0.0))
                loss = torch.clamp(cal_wasserstein_loss(anchor_emb, positive_emb, reduction='none') - 2 * cal_wasserstein_loss(anchor_emb, negative_emb, reduction='none') + 2.0,  min=0.0).mean()
                ###
                ## distance = torch.norm(anchor - other, p=2, dim=1)  # 欧氏距离
                ## loss = torch.mean(
                ##     target * distance + (1 - target) * torch.clamp(self.margin - distance, min=0.0)
                ## )
                pos_distance = cal_wasserstein_loss(anchor_emb, positive_emb, reduction='none')
                loss = loss + torch.mean(pos_distance)
                neg_distance = cal_wasserstein_loss(anchor_emb, negative_emb, reduction='none')
                loss = loss + 2 * torch.mean(torch.clamp(1 - neg_distance, min=0.0))
                
                loss = loss + torch.clamp_max(cal_wasserstein_loss(anchor_emb, positive_emb, reduction='mean'), max=1) \
                       - 2 * torch.clamp_max(cal_wasserstein_loss(anchor_emb, negative_emb, reduction='mean'), max=1)
                loss = loss * beta
                
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            if step % 10 == 0:
                print(f"Global Step {step}/{total_step} | Loss: {loss.item():.4f} | Grad Norm: {grad_norm.item():.4f}")
            save_step = 2500
            if (step + 1) % save_step == 0:
                print("Eval...")
                dataset = json.load(open(test_json, "r"))
                accuracy = evaluate_triplet(model, dataset, device, generator, step)
                if best_acc < accuracy:
                    torch.save(model.state_dict(), f"/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/best_flow_weights/old_attention_Disease_Extra.pt")
                    best_acc = accuracy
                else:
                    pass

    ### 加载 checkpoints
    try:
        state_dict = torch.load("/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/best_flow_weights/model_Disease.pt", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    except:
        pass
    
    model = model.to(device)
    print(model)
    label_map = json.load(open("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match/utils/label_map.json", "r"))
    
    def evaluate_triplet(model, dataset, device, generator, step=0, k=1):
        if generator is None:
            raise NotImplementedError
        else:
            vae = generator.vae
        model.eval()
        correct = 0
        total = 0
        l2_correct = 0
        save_image_root = '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match/data/test_Disease_exam_images'
        new_dataset = []
        old_image_root = '/mnt/inaisfs/data/home/tansy_criait/flow_match/data/exam_images_backup'
        answer_map = {"option_A": 0, "option_B": 1, "option_C": 2, "option_D": 3}
        with torch.no_grad():
            for data in tqdm(dataset):
                if 'image' not in data:
                    data["image"] = data['image_paths'][0]
                if not os.path.exists(data["image"]):
                    continue
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
                    question_id = data['question_id']
                    label_A = data['gt_answer'].replace(' ', '_')
                    caption_templete = "将当前未知的医疗病症图像 转化为 {} 病症的相关图像。"
                    label_A_id = label_map[label_A]
                    
                    try:
                        if "option_A_dir" in data:
                            test_image_dir = data["option_A_dir"]
                            os.makedirs(test_image_dir.replace(old_image_root, save_image_root),exist_ok=True)
                            test_images = os.listdir(test_image_dir)
                            test_image_path = os.path.join(test_image_dir, random.choice(test_images))
                            test_image = Image.open(test_image_path).convert("RGB")
                            test_image_A = test_image
                            test_image_A_path = test_image_path.replace(old_image_root, save_image_root)
                            test_image.save(test_image_path.replace(old_image_root, save_image_root))
                            test_image = transform(test_image).unsqueeze(0)
                            test_image = test_image.to(vae.device)
                            # 压缩至 潜在空间
                            with torch.no_grad():
                                test_image = vae.encode(test_image).latent_dist
                                test_image = test_image.sample() * 0.18215
                            test_image = test_image.to(device)
                            new_data = {
                                "question_id": question_id,
                                "x0": data["image"],
                                "label_A": label_A,
                                "label_A_id": label_A_id,
                                "label_B": data['option_A'],
                                "label_B_id": label_map[data['option_A']],
                                "caption": caption_templete.format(data['option_A']),
                                "x1": test_image_path.replace(old_image_root, save_image_root),
                                "hint_path": test_image_path.replace(old_image_root, save_image_root)
                            }
                            new_dataset.append(new_data)
                            test_emb = model.encode(test_image, False)
                            wass_A += cal_wasserstein_loss(anchor_emb, test_emb).squeeze()
                            l2_A += torch.norm(anchor_emb - test_emb, p=2, dim=1).squeeze()
                        else:
                            wass_A = torch.Tensor([float('inf')]).to(device).squeeze()
                            l2_A = torch.Tensor([float('inf')]).to(device).squeeze()
                    except:
                            wass_A = torch.Tensor([float('inf')]).to(device).squeeze()
                            l2_A = torch.Tensor([float('inf')]).to(device).squeeze()
                    
                    try:
                        if "option_B_dir" in data:
                            test_image_dir = data["option_B_dir"]
                            os.makedirs(test_image_dir.replace(old_image_root, save_image_root),exist_ok=True)
                            test_images = os.listdir(test_image_dir)
                            test_image_path = os.path.join(test_image_dir, random.choice(test_images))
                            test_image = Image.open(test_image_path).convert("RGB")
                            test_image_B = test_image
                            test_image_B_path = test_image_path.replace(old_image_root, save_image_root)
                            test_image.save(test_image_path.replace(old_image_root, save_image_root))
                            test_image = transform(test_image).unsqueeze(0)
                            test_image = test_image.to(vae.device)
                            # 压缩至 潜在空间
                            with torch.no_grad():
                                test_image = vae.encode(test_image).latent_dist
                                test_image = test_image.sample() * 0.18215
                            test_image = test_image.to(device)
                            new_data = {
                                "question_id": question_id,
                                "x0": data["image"],
                                "label_A": label_A,
                                "label_A_id": label_A_id,
                                "label_B": data['option_B'],
                                "label_B_id": label_map[data['option_B']],
                                "caption": caption_templete.format(data['option_B']),
                                "x1": test_image_path.replace(old_image_root, save_image_root),
                                "hint_path": test_image_path.replace(old_image_root, save_image_root)
                            }
                            new_dataset.append(new_data)
                            test_emb = model.encode(test_image, False)
                            wass_B += cal_wasserstein_loss(anchor_emb, test_emb).squeeze()
                            l2_B += torch.norm(anchor_emb - test_emb, p=2, dim=1).squeeze()
                        else:
                            wass_B = torch.Tensor([float('inf')]).to(device).squeeze()
                            l2_B = torch.Tensor([float('inf')]).to(device).squeeze()
                    except:
                            wass_B = torch.Tensor([float('inf')]).to(device).squeeze()
                            l2_B = torch.Tensor([float('inf')]).to(device).squeeze()
                    try:  
                        if "option_C_dir" in data:
                            test_image_dir = data["option_C_dir"]
                            os.makedirs(test_image_dir.replace(old_image_root, save_image_root),exist_ok=True)
                            test_images = os.listdir(test_image_dir)
                            test_image_path = os.path.join(test_image_dir, random.choice(test_images))
                            test_image = Image.open(test_image_path).convert("RGB")
                            test_image_C = test_image
                            test_image_C_path = test_image_path.replace(old_image_root, save_image_root)
                            test_image.save(test_image_path.replace(old_image_root, save_image_root))
                            test_image = transform(test_image).unsqueeze(0)
                            test_image = test_image.to(vae.device)
                            # 压缩至 潜在空间
                            with torch.no_grad():
                                test_image = vae.encode(test_image).latent_dist
                                test_image = test_image.sample() * 0.18215
                            test_image = test_image.to(device)
                            new_data = {
                                "question_id": question_id,
                                "x0": data["image"],
                                "label_A": label_A,
                                "label_A_id": label_A_id,
                                "label_B": data['option_C'],
                                "label_B_id": label_map[data['option_C']],
                                "caption": caption_templete.format(data['option_C']),
                                "x1": test_image_path.replace(old_image_root, save_image_root),
                                "hint_path": test_image_path.replace(old_image_root, save_image_root)
                            }
                            new_dataset.append(new_data)
                            test_emb = model.encode(test_image, False)
                            wass_C += cal_wasserstein_loss(anchor_emb, test_emb).squeeze()
                            l2_C += torch.norm(anchor_emb - test_emb, p=2, dim=1).squeeze()
                        else:
                            wass_C = torch.Tensor([float('inf')]).to(device).squeeze()
                            l2_C = torch.Tensor([float('inf')]).to(device).squeeze()
                    except:
                            wass_C = torch.Tensor([float('inf')]).to(device).squeeze()
                            l2_C = torch.Tensor([float('inf')]).to(device).squeeze()
                    try:
                        if "option_D_dir" in data:
                            test_image_dir = data["option_D_dir"]
                            os.makedirs(test_image_dir.replace(old_image_root, save_image_root),exist_ok=True)
                            test_images = os.listdir(test_image_dir)
                            test_image_path = os.path.join(test_image_dir, random.choice(test_images))
                            test_image = Image.open(test_image_path).convert("RGB")
                            test_image_D = test_image
                            test_image_D_path = test_image_path.replace(old_image_root, save_image_root)
                            test_image.save(test_image_path.replace(old_image_root, save_image_root))
                            test_image = transform(test_image).unsqueeze(0)
                            test_image = test_image.to(vae.device)
                            # 压缩至 潜在空间
                            with torch.no_grad():
                                test_image = vae.encode(test_image).latent_dist
                                test_image = test_image.sample() * 0.18215
                            test_image = test_image.to(device)
                            new_data = {
                                "question_id": question_id,
                                "x0": data["image"],
                                "label_A": label_A,
                                "label_A_id": label_A_id,
                                "label_B": data['option_D'],
                                "label_B_id": label_map[data['option_D']],
                                "caption": caption_templete.format(data['option_D']),
                                "x1": test_image_path.replace(old_image_root, save_image_root),
                                "hint_path": test_image_path.replace(old_image_root, save_image_root)
                            }
                            new_dataset.append(new_data)
                            test_emb = model.encode(test_image, False)
                            wass_D += cal_wasserstein_loss(anchor_emb, test_emb).squeeze()
                            l2_D += torch.norm(anchor_emb - test_emb, p=2, dim=1).squeeze()
                        else:
                            wass_D = torch.Tensor([float('inf')]).to(device).squeeze()
                            l2_D = torch.Tensor([float('inf')]).to(device).squeeze()
                    except:
                            wass_D = torch.Tensor([float('inf')]).to(device).squeeze()
                            l2_D = torch.Tensor([float('inf')]).to(device).squeeze()
                pred_option_id = torch.argmin(torch.stack([wass_A, wass_B, wass_C, wass_D], dim=0)).item()
                pred_option_id_l2 = torch.argmin(torch.stack([l2_A, l2_B, l2_C, l2_D], dim=0)).item()
                data["pred_option_id"] = pred_option_id
                data["pred_option_id_l2"] = pred_option_id_l2
                if answer_id == pred_option_id:
                    correct += 1
                #     test_image_A.save(test_image_A_path)
                #     test_image_B.save(test_image_B_path)
                #     test_image_C.save(test_image_C_path)
                #     test_image_D.save(test_image_D_path)
                if answer_id == pred_option_id_l2:
                    l2_correct += 1
                total += 1
        
        accuracy = correct / total
        print(f"Evaluation Accuracy: {accuracy:.4f}")
        l2_accuracy = l2_correct / total
        print(f"L2 Evaluation Accuracy: {l2_accuracy:.4f}")
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
    # train_triplet(model, dataloaders, criterion=criterion, optimizer=optimizer, device=device,
    #               generator=generator, criterion_contrastive=criterion_contrastive,
    #               cal_wasserstein_loss=cal_wasserstein_loss, epochs=10) 

    dataset = json.load(open(test_json, "r"))
    evaluate_triplet(model, dataset, device, generator, 0, 1)
