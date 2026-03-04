import os
import random

from torch.nn.utils import clip_grad_norm_
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from copy import deepcopy
from tqdm import tqdm
from diffusers.models import AutoencoderKL
from collections import OrderedDict
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import numpy as np
import os
import torch
from torchvision import transforms
import torch.nn.functional as F

class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, base_transform=None):
        self.img_paths = []
        for root, dirs, files in os.walk(root_dir):
            for fname in files:
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.img_paths.append(os.path.join(root, fname))
        self.transform = transform
        self.base_transform = base_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')

        image = self.transform(img)
        base_image = self.base_transform(img)
        batch = {
            'image': image,
            'image_path': self.img_paths[idx],
            'base_image': base_image
        }
        return batch

def vae_loss(recon_x, x, mu, logvar, beta=1.0, p=1):
    if p == 1:
        recon = F.l1_loss(recon_x, x, reduction='mean')
    else:
        recon = F.mse_loss(recon_x, x, reduction='mean')
    logvar = torch.clamp(logvar, -10, 10)
    kl = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    return recon + beta * kl, kl.detach()

# Step 1: 自定义 AddGaussianNoise 类
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): (C, H, W) 的张量
        Returns:
            Tensor: 添加高斯噪声后的图像
        """
        if random.random() < self.p:
            noise = torch.randn(tensor.shape) * self.std + self.mean
            return tensor + noise
        else:
            return tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

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


def train_vae(model, loader, device, epochs=30, lr=1e-4, beta=1.0, save_dir='./checkpoints', use_perceptual = False, beta_perceptual = 0.1, ema_steps=5, use_ema=False, ema_decay = 0.95):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    print('Trainable params size: ', len(list(model.parameters())))
    scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-10)
    model.train()

    if use_perceptual:
        import lpips
        import torch

        perceptual_loss_fn = lpips.LPIPS(net='alex').to(model.device)
    else:
        perceptual_loss_fn = None

    if use_ema:
        ema = deepcopy(model.module).to(model.device)
        requires_grad(ema, False)
    else:
        ema = None

    for epoch in range(1, epochs + 1):
        loader.sampler.set_epoch(epoch)  # 保证每轮数据不重复
        epoch_loss = 0.0

        with tqdm(loader, desc=f"[Epoch {epoch}/{epochs}]", unit='batch') as pbar:
            for step, batch in enumerate(pbar):
                try:
                    imgs = batch['image'].to(model.device)
                    base_image = batch['base_image'].to(model.device)
                except:
                    imgs = batch.to(model.device)
                    base_image = imgs

                recon_x, latent_dist = model(imgs, sample_posterior=True, return_dict=False) # recon ~ p(x|y), p(x|y)
                loss, kl = vae_loss(recon_x.sample, base_image, latent_dist.mean, latent_dist.logvar, beta, p=1)
                
                if perceptual_loss_fn is not None:
                    # print(recon_x.sample.device, base_image.device)
                    perceptual_loss = perceptual_loss_fn.forward(recon_x.sample, base_image).mean()
                    loss = loss + perceptual_loss * beta_perceptual
                else:
                    perceptual_loss = torch.Tensor(0)

                optimizer.zero_grad()
                loss.backward()
                norm = clip_grad_norm_(model.parameters(), max_norm=1.0).item()
                optimizer.step()
                scheduler.step()
                if step % ema_steps == 0 and ema is not None:
                    update_ema(ema, model.module, ema_decay)

                batch_size = imgs.size(0)
                epoch_loss += loss.item() * batch_size

                pbar.set_postfix({
                    'batch_loss': f"{loss.item():.4f}",
                    "kl": f"{kl:.4f}",
                    "norm": f"{norm:.4f}",
                    "perceptual_loss": f"{perceptual_loss.item():.4f}"
                })

        avg_loss = epoch_loss / len(loader.dataset)
        print(f"Epoch {epoch}/{epochs} — Avg Loss: {avg_loss:.4f}")

        if dist.get_rank() == 0 and epoch % 2 == 0:
            save_model = ema
            ema.save_pretrained(save_dir)
            print(f"  ↳ saved: {save_dir}")

@torch.no_grad()
def update_ema(ema_model, model, decay=0.99):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

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

# torchrun --nnodes=1 --nproc_per_node=2 --master_port=25001 ./EndoViT/vae_train_dist.py
def run():
    # 分布式初始化
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # 加载特征提取器
    base_transform = transforms.Compose([
        # transforms.Resize((feat.size['width'], feat.size['height'])),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform = transforms.Compose([
        # transforms.Resize((feat.size['width'], feat.size['height'])),
        transforms.Resize((512, 512)),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # AddGaussianNoise(0, 0.1, p=0.05),
        # transforms.RandomErasing(p=0.05, scale=(0.05, 0.1), ratio=(0.67, 1.33), value='random')
    ])

    # src_path = './EndoViT/disease_data/train_clip_data2_with_disease.json'
    # dataset = MedicalVAEDataset(src_path, transform=transform, base_transform=base_transform, text_key='disease',
    #                             positive_dict=None,
    #                             # positive_dict=json.load(open('./EndoViT/disease_data/positive_dict_total.json', 'r'))
    # )

    image_root = './EndoViT/our_data/new_cropped-2004-2010-endovit'
    # image_root = './CropImages/All'

    dataset = MedicalImageDataset(image_root, transform, base_transform)
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=3, sampler=sampler, num_workers=0)

    vae = AutoencoderKL.from_pretrained(
        './flow_matcher_otcfm/vae_our').to(device).train()
    # state_dict = torch.load('./EndoViT/sd-ema-vae_weight/sd-vae_epoch_ema_711.pth')
    # vae.load_state_dict(state_dict, strict=False)

    d_vae = DDP(vae, device_ids=[local_rank], find_unused_parameters=True)
    train_vae(d_vae, loader, device, epochs=1, lr=1e-5, beta=1e-3, use_perceptual = True, beta_perceptual = 0.25,
              ema_steps=5, use_ema=True, ema_decay = 0.9,
              save_dir='./flow_matcher_otcfm/vae_our')

    dist.destroy_process_group()

if __name__ == "__main__":
    run()