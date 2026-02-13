import gc
import os
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import sys
sys.path.append('/mnt/inaisfs/data/home/tansy_criait/GasAgent-main/train_vae')
from vqae_sim import VQAE, AddGaussianNoise, vae_loss
import torch
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from timm.models.vision_transformer import VisionTransformer
from functools import partial
from torch import nn
from torch.nn import Identity
from safetensors.torch import load_file
from diffusers.models import AutoencoderKL
from transformers import ViTModel, ViTConfig, AutoTokenizer, AutoModel
# from transformers import CLIPModel, CLIPProcessor
from transformers import ChineseCLIPProcessor as CLIPProcessor
from transformers import ChineseCLIPModel as CLIPModel
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers.models.autoencoders.vae import VectorQuantizer
import torch.nn.functional as F
from dataclasses import dataclass
import math
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
import os
import torch
from torchvision import transforms
from transformers import AutoFeatureExtractor
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from random import choices, choice
from utils.utils_ import _get_vector_norm
import torch.nn.functional as F

class MedicalVAEDataset(Dataset):
    def __init__(self, data_path, transform=None, base_transform=None, text_key='text', image_key='image',
                 positive_dict:dict[str, list] = None):
        self.dataset = json.load(open(data_path, 'r'))
        self.transform = transform
        self.base_transform = base_transform
        self.text_key = text_key
        self.image_key = image_key
        self.positive_dict = positive_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_file = self.dataset[idx][self.image_key]
        image = Image.open(img_file).convert('RGB')
        if self.transform:
            img = self.transform(image)
        if self.base_transform:
            base_img = self.base_transform(image)
        else:
            base_img = img
        text = self.dataset[idx][self.text_key]
        if 'disease' in self.dataset[idx]:
            disease = self.dataset[idx]['disease']
            try: # 换图。
                if random.random() < 0.25 and disease:
                    base_data = random.choice(self.positive_dict[disease])
                    base_image = Image.open(base_data[self.image_key]).convert('RGB')
                    img = base_img
                    base_img = self.base_transform(base_image)
            except:
                pass

        batch = {
            'image': img,
            'text': text,
            'image_path': img_file,
            'base_image': base_img
        }
        return batch


def train_vae(model, loader, device, epochs=30, lr=1e-4, beta=1.0, commit_beta=1.0, save_dir='./checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    print('Trainable params size: ', len(list(model.parameters())))
    scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-10)
    model.train()

    for epoch in range(1, epochs + 1):
        loader.sampler.set_epoch(epoch)  # 保证每轮数据不重复
        epoch_loss = 0.0

        with tqdm(loader, desc=f"[Epoch {epoch}/{epochs}]", unit='batch') as pbar:
            for batch in pbar:
                try:
                    imgs = batch['image'].to(model.device)
                    base_image = batch['base_image'].to(model.device)
                except:
                    imgs = batch.to(model.device)
                    base_image = imgs

                output = model(imgs)
                loss = F.mse_loss(output.recon, base_image, reduction='mean')
                loss = loss + commit_beta * output.commit_loss

                optimizer.zero_grad()
                loss.backward()
                norm = clip_grad_norm_(model.parameters(), max_norm=1.0).item()
                optimizer.step()
                scheduler.step()

                batch_size = imgs.size(0)
                epoch_loss += loss.item() * batch_size

                pbar.set_postfix({
                    'batch_loss': f"{loss.item():.4f}",
                    "commit_loss": f"{output.commit_loss:.4f}",
                    "norm": f"{norm:.4f}",
                })

        avg_loss = epoch_loss / len(loader.dataset)
        print(f"Epoch {epoch}/{epochs} — Avg Loss: {avg_loss:.4f}")

        if dist.get_rank() == 0:
            save_path = os.path.join(save_dir, f"vqae_epoch{epoch}.pth")
            torch.save(model.module.state_dict(), save_path)
            print(f"  ↳ saved: {save_path}")

def numpy_topk_simple(arr, k, axis=-1, largest=True):
    if not largest:
        arr = -arr
    # print(arr.shape)
    indices = np.argsort(arr, axis=axis)[:, -k:]
    # print(indices.shape)

    # [:, -k:] # 全排序后取前k个
    if largest:
        indices = np.flip(indices, axis=axis)  # 降序排列
    return indices

def eval_vae_similatiry(vae, images_root, device, transforms, k=3):
    vae.eval()
    accuracy = {}
    labels = os.listdir(images_root)
    for label_id, label in enumerate(labels):
        src_path = os.path.join(images_root, label)
        cls_correct = 0
        topk_correct = 0
        kl_correct = 0
        kl_topk_correct = 0
        js_correct = 0
        js_topk_correct = 0
        total = 0
        for _ in tqdm(range(10)):
            # print(f"{label_id}/{len(labels)}---{_}/20")
            src_imgs = choices(os.listdir(src_path), k=10)
            src_imgs = [os.path.join(src_path, x) for x in src_imgs]

            total += len(src_imgs)
            other_imgs = []
            for other in labels:
                image_dir = os.path.join(images_root, other)
                file_name = choice(os.listdir(image_dir))
                other_imgs.append(os.path.join(image_dir, file_name))

            other_imgs = [transforms(Image.open(x).convert('RGB')) for x in other_imgs]
            src_imgs = [transforms(Image.open(x).convert('RGB')) for x in src_imgs]
            images = torch.stack(other_imgs)
            # print('image shape: ', images.shape)
            with torch.no_grad():
                hidden_states = vae.encoder(images.to(device))
            cls = hidden_states[:, 0, ...]
            latent = hidden_states[:, 1:, ...]

            src_imgs = torch.stack(src_imgs)
            # print('src shape: ', src_imgs.shape)
            with torch.no_grad():
                hidden_states = vae.encoder(src_imgs.to(device))
            src_cls = hidden_states[:, 0, ...]
            src_latent = hidden_states[:, 1:, ...]
            # 计算L2范数
            cls = cls.cpu().numpy()
            src_cls = src_cls.cpu().numpy()

            norms = np.linalg.norm(cls, axis=1, keepdims=True)
            eps = 1e-8
            cls = cls / (norms + eps)
            norms = np.linalg.norm(src_cls, axis=1, keepdims=True)
            src_cls = src_cls / (norms + eps)  # [10, ]
            cos_score = src_cls @ cls.T  # [10, classes]

            pred = np.argmax(cos_score, axis=1)  # [10]
            result = (pred == label_id)
            cls_correct += result.sum()

            indices = numpy_topk_simple(cos_score, k, 1, True)  # [10, 3]
            for x in range(1, k):
                result = result | (indices[:, x] == label_id)
            topk_correct += result.sum()

            # KL Accuracy.
            latent = latent.view(latent.shape[0], vae.latent_h, vae.latent_h, latent.shape[-1])
            latent = vae.fc_mu_logvar(latent).permute(0, 3, 1, 2)
            latent = vae.quant_conv(latent)
            mu, logvar = latent.chunk(2, dim=1)
            mu = mu.cpu().detach().numpy()
            std = torch.exp(logvar / 2).cpu().detach().numpy()
            # print('mu', mu.shape)

            src_latent = src_latent.view(src_latent.shape[0], vae.latent_h, vae.latent_h, src_latent.shape[-1])
            src_latent = vae.fc_mu_logvar(src_latent).permute(0, 3, 1, 2)
            src_latent = vae.quant_conv(src_latent)
            src_mu, src_logvar = src_latent.chunk(2, dim=1)
            src_mu = src_mu.cpu().detach().numpy()
            src_std = torch.exp(src_logvar / 2).cpu().detach().numpy()
            # print('src_mu', src_mu.shape)

            # 计算双向KL [10, classes]
            kl_score = np.zeros([src_mu.shape[0], mu.shape[0]])
            for i in range(src_mu.shape[0]):
                for j in range(mu.shape[0]):
                    _, kl = symmetric_kl_vae(src_mu[i], src_std[i], mu[j], std[j])
                    kl_score[i, j] = - kl

            pred = np.argmax(kl_score, axis=1)  # [10]
            result = (pred == label_id)
            kl_correct += result.sum()
            indices = numpy_topk_simple(kl_score, k, 1, True)  # [10, 3]
            for x in range(1, k):
                result = result | (indices[:, x] == label_id)
            kl_topk_correct += result.sum()

            # 计算JS-KL。
            # [10, classes]
            kl_score = np.zeros([src_mu.shape[0], mu.shape[0]])
            for i in range(src_mu.shape[0]):
                for j in range(mu.shape[0]):
                    _, kl = js_kl_vae(src_mu[i], src_std[i], mu[j], std[j])
                    kl_score[i, j] = - kl

            pred = np.argmax(kl_score, axis=1)  # [10]
            result = (pred == label_id)
            js_correct += result.sum()
            indices = numpy_topk_simple(kl_score, k, 1, True)  # [10, 3]
            for x in range(1, k):
                result = result | (indices[:, x] == label_id)
            js_topk_correct += result.sum()

        accuracy[label + '_js'] = js_correct / total
        accuracy[label + f'_j{k}'] = js_topk_correct / total
        accuracy[label + '_kl'] = kl_correct / total
        accuracy[label + f'_k{k}'] = kl_topk_correct / total
        accuracy[label] = cls_correct / total
        accuracy[label + f'_top{k}'] = topk_correct / total
    return accuracy

# torchrun --nnodes=1 --nproc_per_node=2 --master_port=25001 /home/dalhxwlyjsuo/criait_tansy/project/EndoViT/vae_train_dist.py
def run():
    # 分布式初始化
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # 加载特征提取器
    feat = AutoFeatureExtractor.from_pretrained('/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/clip_trained_weight_disease/CLIPModel_base')
    base_transform = transforms.Compose([
        transforms.Resize((feat.size['width'], feat.size['height'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=feat.image_mean, std=feat.image_std),
    ])
    transform = transforms.Compose([
        transforms.Resize((feat.size['width'], feat.size['height'])),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=feat.image_mean, std=feat.image_std),
        AddGaussianNoise(0, 0.1, 0.1),
        # transforms.RandomErasing(p=0.25, scale=(0.05, 0.2), ratio=(0.67, 1.33), value='random')
    ])

    src_path = '/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/disease_data/train_clip_data2_with_disease.json'
    dataset = MedicalVAEDataset(src_path, transform=transform, base_transform=base_transform, text_key='disease',
                                positive_dict=None,
                                # positive_dict=json.load(open('/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/disease_data/positive_dict_total.json', 'r'))
    )
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=4)

    encoder_ckpt = ''
    decoder_ckpt = '/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/vqae_weight/VQAEModel'
    vae = VQAE(latent_dim=4, encoder_ckpt=encoder_ckpt, decoder_ckpt=decoder_ckpt).to(device).train()
    if True:
        state_dict = torch.load('/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/vqae_weight/vae_epoch2.pth')
        vae.load_state_dict(state_dict, strict=False)
    d_vae = DDP(vae, device_ids=[local_rank], find_unused_parameters=True)
    train_vae(d_vae, loader, device, epochs=10, lr=1e-5, beta=0.01, commit_beta=0.1,
              save_dir='/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/vqae_weight')

    dist.destroy_process_group()

if __name__ == "__main__":
    run()