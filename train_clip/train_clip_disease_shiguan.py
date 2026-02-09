import gc
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from timm.models.vision_transformer import VisionTransformer
from functools import partial
from torch import nn
from torch.nn import Identity
from safetensors.torch import load_file
from diffusers.models import AutoencoderKL
from transformers import ViTModel, ViTConfig, AutoTokenizer, AutoModel
from transformers import ChineseCLIPConfig as CLIPConfig
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
from transformers import AutoFeatureExtractor
from torch.nn.utils import clip_grad_norm_
from random import choices, choice
from utils_ import _get_vector_norm
from torchvision.transforms.functional import solarize

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

class MedicalCLIPDataset(Dataset):
    def __init__(self, data_path, transform=None, base_transform=None, text_key = 'text', image_key = 'image'):
        self.dataset = json.load(open(data_path, 'r'))
        self.transform = transform
        self.base_transform = base_transform
        self.text_key = text_key
        self.image_key = image_key

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
        batch = {
            'image': img,
            'text': text,
            'disease': self.dataset[idx]['disease'],
            'image_path': img_file,
            'base_image': base_img
        }
        return batch

def smooth_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.ndim == logits.ndim:
        # 计算目标损失
        log_probs = F.log_softmax(logits, dim=1)
        loss = - torch.sum(target * log_probs) / 128
        return loss
    else:
        raise NotImplementedError("平滑损失, Target ndim should be 2!")

def smooth_contrastive_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return smooth_cross_entropy(logits, target)

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    gt_sim = torch.arange(len(logits), device=logits.device)
    return nn.functional.cross_entropy(logits, gt_sim)

def check_match(text1, text2):
    if text1 == text2 and text1 != '' and text2 != '':
        return True
    else:
        return False

# diseases = ['123', '456', '123', '456', '123456']
def smooth_clip_loss(similarity: torch.Tensor, diseases, smooth=0.2, keep_neg=False) -> torch.Tensor:
    with torch.no_grad():
        target = torch.eye(len(similarity), device=similarity.device) - smooth
        if keep_neg:
            neg_mask = target < 0  # 提取负数的掩码
            denorm_neg_mask = neg_mask.sum(dim=-1, keepdim=True).float()  # 每个样本中负数的个数
            # 防止除以 0，当某样本中无负数时，denorm_neg_mask 为 0，但我们不会使用它
            denorm_neg_mask = torch.clamp(denorm_neg_mask, min=1e-8)
            # 仅对负数部分进行归一化
            normalized_neg_part = torch.clamp(target / denorm_neg_mask, min = -0.1, max = -0.005)
            # 使用 torch.where 选择性地更新负数部分，正数部分保持不变
            target = torch.where(neg_mask, normalized_neg_part, target)
        else:
            target = target * torch.eye(len(similarity), device=similarity.device)
        target_smooth = torch.zeros_like(target)
        for idx_r, disease_r in enumerate(diseases):
            for idx_l, disease_l in enumerate(diseases):
                target_smooth[idx_r, idx_l] = smooth if check_match(disease_l, disease_r) else 0

        target_smooth_mask = target_smooth > 0
        if target_smooth_mask.any():
            target_smooth = target_smooth / torch.clamp_min(target_smooth_mask.sum(dim=-1, keepdim=True), 1)
            target = target + target_smooth
        else:
            target = torch.eye(len(similarity), device=similarity.device)
    if random.random() > 0.9:
        print(similarity)
        print(target)
    caption_loss = smooth_contrastive_loss(similarity, target)
    image_loss = smooth_contrastive_loss(similarity.t(), target.t())
    return (caption_loss + image_loss) / 2.0

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

def train_clip(loader, device, clip_path = '/home/dalhxwlyjsuo/criait_tansy/weight/clip-vit-large-patch14',
              epochs=30, lr=1e-5, save_dir='./checkpoints', clip_text_path=''):
    os.makedirs(save_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(clip_path, use_fast=True)
    processor = CLIPProcessor.from_pretrained(clip_path, use_fast=True)

    gpu_num = torch.cuda.device_count()
    if gpu_num > 1:
        print("Avail GPUs: ", gpu_num)
        try:
            device_map = json.load(open(f'/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/device_map_{gpu_num}.json', 'r'))
        except (FileNotFoundError, FileExistsError):
            print('Load Device Map')
            device_map = json.load(
                open(f'/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/device_map_{gpu_num}.json', 'r'))
        except:
            device_map = 'auto'
    else:
        device_map = 'auto'

    try:
        clip = CLIPModel.from_pretrained(
            clip_path,
            device_map = device_map,
        ).train()
    except:
        base_clip_path = os.path.join(clip_path, 'config.json')
        new_clip_config = CLIPConfig.from_pretrained(
            base_clip_path,
            device_map=device_map,
        )
        clip = CLIPModel(new_clip_config)
        clip.save_pretrained(clip_path)
        clip = CLIPModel.from_pretrained(
            clip_path,
            device_map = device_map,
        ).train()
    if os.path.exists(clip_text_path) and clip_text_path:
        text_device = clip.text_model.device
        clip.text_model.from_pretrained(clip_text_path).to(text_device)
    # with open(f'/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/base_device_map_{gpu_num}.json', 'w') as f:
    #     json.dump(clip.hf_device_map, f, indent=2)

    if os.path.exists(clip_text_path) and clip_text_path:
        text_device = clip.text_model.device
        print('Old CLIP Text Model __class__', clip.text_model.__class__)
        try:
            print(clip_text_path)
            text_model = AutoModel.from_pretrained(clip_text_path).to(text_device).train()
            print('New Text Model __class__', text_model.__class__)
            clip.text_model = text_model
            tokenizer = AutoTokenizer.from_pretrained(clip_text_path, use_fast=True)
        except Exception as e:
            print(e)

        print('New CLIP Text Model __class__', clip.text_model.__class__)
        print('加载 New CLIP Text Module.')
        clip.save_pretrained(os.path.join(save_dir, f'CLIPModel_Text'))
        tokenizer.save_pretrained(os.path.join(save_dir, f'CLIPModel_Text'))
        processor.save_pretrained(os.path.join(save_dir, f'CLIPModel_Text'))
    # with open(f'/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/base_device_map_{gpu_num}.json', 'w') as f:
    #     json.dump(clip.hf_device_map, f, indent=2)

    optimizer_clip = torch.optim.AdamW(clip.parameters(), lr=lr, weight_decay=1e-4)
    scheduler_clip = CosineAnnealingLR(optimizer_clip, T_max=1000, eta_min=1e-12)
    print(clip.hf_device_map)
    print(clip)
    if clip_text_path:
        os.makedirs(clip_text_path, exist_ok=True)

    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        # 在 tqdm 中显示 epoch 进度
        with (tqdm(loader, desc=f"[Epoch {epoch}/{epochs}]", unit='batch') as pbar):
            for batch in pbar:
                texts = batch['text']
                images = [Image.open(x).convert('RGB') for x in batch['image_path']]
                assert len(texts) == len(images)
                diseases = batch['disease']

                if gpu_num >= 1:
                    text_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(clip.device)
                    image_inputs = processor(images=images, return_tensors="pt")
                    if random.random() > 0.25:
                        del image_inputs['pixel_values']
                        image_inputs['pixel_values'] = batch['image']
                    image_inputs = image_inputs.to(clip.device)
                    vision_outputs = clip.vision_model(**image_inputs, interpolate_pos_encoding=True)
                    text_outputs = clip.text_model(**text_inputs)
                    image_embeds = vision_outputs[1].to(clip.device)
                    image_embeds = clip.visual_projection(image_embeds)

                    text_embeds = text_outputs[0][:, 0, :].to(clip.device)
                    text_embeds = clip.text_projection(text_embeds)

                    # normalized features
                    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                    # cosine similarity as logits
                    logit_scale = clip.logit_scale.exp().to(text_embeds.device)
                    if random.random() > 0.9:
                        print('logit scale: ', logit_scale.detach().item())
                    logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds)) * logit_scale
                    assert int(logits_per_text.shape[0]) == len(diseases), "某数据缺少 disease 值。"
                    loss = smooth_clip_loss(logits_per_text, diseases, smooth=0.6, keep_neg=True)
                    # ### keep_neg 设置为 True，是否和 能量函数 的优化很像！！！

                    # loss = clip_loss(logits_per_text)
                else:
                    inputs = processor(text=texts, images=images,
                                       return_tensors="pt", padding=True).to(clip.device)
                    outputs = clip(**inputs, return_loss=True)
                    loss = outputs['loss']

                optimizer_clip.zero_grad()
                loss.backward()
                norm = clip_grad_norm_(clip.parameters(), max_norm=1.0)
                optimizer_clip.step()
                scheduler_clip.step()
                batch_size = len(images)
                epoch_loss += loss.item() * batch_size
                norm = norm.item()

                # 把当前 batch 的 loss 打印到进度条上
                pbar.set_postfix({
                    'batch_loss': f"{loss.item():.4f}",
                    "norm": f"{norm:.4f}",
                })

        avg_loss = epoch_loss / len(loader) / batch_size
        print(f"Epoch {epoch}/{epochs} — Avg Loss: {avg_loss:.4f}")

        if epoch % 2 == 0:
            clip.save_pretrained(os.path.join(save_dir, f'CLIPModel_{epoch}'))
            processor.save_pretrained(os.path.join(save_dir, f'CLIPModel_{epoch}'))
            if os.path.exists(clip_text_path) and clip_text_path:
                save_clip_text_path = clip_text_path + f'_disease_{epoch}'
                clip.text_model.save_pretrained(save_clip_text_path)
                processor.save_pretrained(save_clip_text_path)
                print(f"TextModel ↳ saved: {save_clip_text_path}")
            print(f"  ↳ saved: {save_dir}")

def train_biomed_clip(loader, device, clip_path = '/home/dalhxwlyjsuo/criait_tansy/weight/clip-vit-large-patch14',
              epochs=30, lr=1e-5, save_dir='./checkpoints', model_weight_path=''):
    os.makedirs(save_dir, exist_ok=True)
    from open_clip import create_model_and_transforms, get_tokenizer
    from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
    model_name = "biomedclip_local"

    with open(os.path.join(clip_path, "open_clip_config.json"), "r") as f:
        config = json.load(f)
        model_cfg = config["model_cfg"]
        preprocess_cfg = config["preprocess_cfg"]

    if (not model_name.startswith(HF_HUB_PREFIX)
            and model_name not in _MODEL_CONFIGS
            and config is not None):
        _MODEL_CONFIGS[model_name] = model_cfg

    tokenizer = get_tokenizer(model_name)
    model_path = model_weight_path if os.path.exists(model_weight_path) else os.path.join(clip_path, "open_clip_pytorch_model.bin")
    print('Model path:', model_path)
    clip, _, processor = create_model_and_transforms(
        model_name=model_name,
        pretrained=model_path,
        **{f"image_{k}": v for k, v in preprocess_cfg.items()},
    )
    clip.to(device).train()
    clip.requires_grad_(True)
    print(clip)
    optimizer_clip = torch.optim.AdamW(clip.parameters(), lr=lr, weight_decay=1e-4)
    scheduler_clip = CosineAnnealingLR(optimizer_clip, T_max=1000, eta_min=1e-12)

    trainable_keys = [name for name, param in clip.named_parameters() if param.requires_grad]
    print('trainable params: ',trainable_keys)

    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        # 在 tqdm 中显示 epoch 进度
        with (tqdm(loader, desc=f"[Epoch {epoch}/{epochs}]", unit='batch') as pbar):
            for batch in pbar:
                texts = batch['text']
                images = [Image.open(x).convert('RGB') for x in batch['image_path']]
                assert len(texts) == len(images)
                diseases = batch['disease']

                text_inputs = tokenizer([f'这是一张关于 "{l}" 疾病的照片。' for l in diseases], context_length=256).to(device)
                text_embeds = clip.encode_text(text_inputs, normalize=True)
                try:
                    image_embeds = clip.encode_image(batch['image'].to(device), normalize=True)
                except Exception as e:
                    print(e)

                logit_scale = clip.logit_scale.exp().to(text_embeds.device)
                if random.random() > 0.9:
                    print('logit scale: ', logit_scale.detach().item())
                logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds)) * logit_scale
                assert int(logits_per_text.shape[0]) == len(diseases), "某数据缺少 disease 值。"
                loss = smooth_clip_loss(logits_per_text, diseases, smooth=0.5, keep_neg=True)
                # loss = clip_loss(logits_per_text)

                optimizer_clip.zero_grad()
                loss.backward()
                norm = clip_grad_norm_(clip.parameters(), max_norm=5.0)
                optimizer_clip.step()
                scheduler_clip.step()
                batch_size = len(images)
                epoch_loss += loss.item() * batch_size
                norm = norm.item()

                # 把当前 batch 的 loss 打印到进度条上
                pbar.set_postfix({
                    'batch_loss': f"{loss.item():.4f}",
                    "norm": f"{norm:.4f}",
                })

        avg_loss = epoch_loss / len(loader) / batch_size
        print(f"Epoch {epoch}/{epochs} — Avg Loss: {avg_loss:.4f}")

        if epoch % 1 == 0:
            torch.save(clip.state_dict(), os.path.join(save_dir,f"open_clip_pytorch_model_{epoch}.bin"))
            print(f"  ↳ saved: {save_dir}")


def train_clip_with_mean_pool(loader, device, clip_path = '/home/dalhxwlyjsuo/criait_tansy/weight/clip-vit-large-patch14',
              epochs=30, lr=1e-5, save_dir='./checkpoints', clip_text_path='', freeze_text = False):
    os.makedirs(save_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(clip_path, use_fast=True)
    processor = CLIPProcessor.from_pretrained(clip_path, use_fast=True)

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        # print('token_embeddings: ', token_embeddings.shape)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    gpu_num = torch.cuda.device_count()
    if gpu_num > 1:
        print("Avail GPUs: ", gpu_num)
        try:
            device_map = json.load(open(f'/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/freeze_text_device_map_{gpu_num}.json', 'r'))
        except (FileNotFoundError, FileExistsError):
            device_map = 'auto'
    else:
        device_map = 'auto'
    try:
        clip = CLIPModel.from_pretrained(
            clip_path,
            device_map = device_map,
        ).train()
    except:
        base_clip_path = os.path.join(clip_path, 'config.json')
        new_clip_config = CLIPConfig.from_pretrained(
            base_clip_path,
            device_map=device_map,
        )
        clip = CLIPModel(new_clip_config)
        clip.save_pretrained(clip_path)
        clip = CLIPModel.from_pretrained(
            clip_path,
            device_map = device_map,
        )

    if os.path.exists(clip_text_path) and clip_text_path:
        text_device = clip.text_model.device
        print('Old CLIP Text Model __class__', clip.text_model.__class__)
        try:
            print(clip_text_path)
            text_model = AutoModel.from_pretrained(clip_text_path).to(text_device).train()
            print('New Text Model __class__', text_model.__class__)
            clip.text_model = text_model
            tokenizer = AutoTokenizer.from_pretrained(clip_text_path, use_fast=True)
        except Exception as e:
            print(e)

        print('New CLIP Text Model __class__', clip.text_model.__class__)
        print('加载 New CLIP Text Module.')
        clip.save_pretrained(os.path.join(save_dir, f'CLIPModel_Text'))
        tokenizer.save_pretrained(os.path.join(save_dir, f'CLIPModel_Text'))
        processor.save_pretrained(os.path.join(save_dir, f'CLIPModel_Text'))
    # with open(f'/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/base_device_map_{gpu_num}.json', 'w') as f:
    #     json.dump(clip.hf_device_map, f, indent=2)

    if freeze_text:
        clip.text_model.requires_grad_(False)

    optimizer_clip = torch.optim.AdamW(clip.parameters(), lr=lr, weight_decay=1e-3)
    scheduler_clip = CosineAnnealingLR(optimizer_clip, T_max=1000, eta_min=1e-10)
    print(clip.hf_device_map)
    print(clip)
    clip.train()
    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        # 在 tqdm 中显示 epoch 进度
        with (tqdm(loader, desc=f"[Epoch {epoch}/{epochs}]", unit='batch') as pbar):
            for batch in pbar:
                texts = batch['text']
                images = [Image.open(x).convert('RGB') for x in batch['image_path']]
                assert len(texts) == len(images)
                diseases = batch['disease']

                if gpu_num >= 1:
                    image_inputs = processor(images=images, return_tensors="pt")
                    if random.random() > 0.25:
                        del image_inputs['pixel_values']
                        image_inputs['pixel_values'] = batch['image']
                    image_inputs = image_inputs.to(clip.device)
                    vision_outputs = clip.vision_model(**image_inputs, interpolate_pos_encoding=True)
                    image_embeds = vision_outputs[1].to(clip.device)
                    image_embeds = clip.visual_projection(image_embeds)

                    if freeze_text:
                        with torch.no_grad():
                            text_inputs = tokenizer(diseases, padding=True, truncation=True, return_tensors='pt',
                                                    max_length=256).to(clip.device)
                            text_outputs = clip.text_model(**text_inputs)
                            text_embeds = mean_pooling(text_outputs, text_inputs['attention_mask'])
                            text_embeds = clip.text_projection(text_embeds)
                    else:
                        text_inputs = tokenizer(diseases, padding=True, truncation=True, return_tensors='pt', max_length=256).to(clip.device)
                        text_outputs = clip.text_model(**text_inputs)
                        text_embeds = mean_pooling(text_outputs, text_inputs['attention_mask'])
                        text_embeds = clip.text_projection(text_embeds)

                    # normalized features
                    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

                    # cosine similarity as logits
                    logit_scale = clip.logit_scale.exp().to(text_embeds)
                    logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds)) * logit_scale
                    assert int(logits_per_text.shape[0]) == len(diseases), "某数据缺少 disease 值。"
                    # loss = smooth_clip_loss(logits_per_text, diseases, smooth=0.6, keep_neg=False)
                    loss = clip_loss(logits_per_text)
                else:
                    inputs = processor(text=texts, images=images,
                                       return_tensors="pt", padding=True).to(clip.device)
                    outputs = clip(**inputs, return_loss=True)
                    loss = outputs['loss']

                optimizer_clip.zero_grad()
                loss.backward()
                norm = clip_grad_norm_(clip.parameters(), max_norm=1.0)
                optimizer_clip.step()
                scheduler_clip.step()
                batch_size = len(images)
                epoch_loss += loss.item() * batch_size
                norm = norm.item()

                # 把当前 batch 的 loss 打印到进度条上
                pbar.set_postfix({
                    'batch_loss': f"{loss.item():.4f}",
                    "norm": f"{norm:.4f}",
                })

        avg_loss = epoch_loss / len(loader) / batch_size
        print(f"Epoch {epoch}/{epochs} — Avg Loss: {avg_loss:.4f}")

        if epoch % 1 == 0:
            clip.save_pretrained(os.path.join(save_dir, f'CLIPModel-Mean-Pool_{epoch}'))
            processor.save_pretrained(os.path.join(save_dir, f'CLIPModel-Mean-Pool_{epoch}'))
            if os.path.exists(clip_text_path) and clip_text_path:
                save_clip_text_path = clip_text_path + f'_disease-Mean-Pool_{epoch}'
                clip.text_model.save_pretrained(save_clip_text_path)
                processor.save_pretrained(save_clip_text_path)
                print(f"TextModel ↳ saved: {save_clip_text_path}")
            print(f"  ↳ saved: {save_dir}")


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

def eval_clip_similatiry(clip, images_root, device, processor, k = 3):
    clip.eval()
    accuracy = {}
    labels = os.listdir(images_root)
    for label_id, label in enumerate(labels):
        src_path = os.path.join(images_root, label)
        cls_correct = 0
        topk_correct = 0
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

            images = [Image.open(x).convert('RGB') for x in other_imgs]
            src_imgs = [Image.open(x).convert('RGB') for x in src_imgs]
            with torch.no_grad():
                image_inputs = processor(images=images, return_tensors="pt").to(device)
                image_embeds = clip.get_image_features(**image_inputs)
                cls = image_embeds / _get_vector_norm(image_embeds)

                image_inputs = processor(images=src_imgs, return_tensors="pt").to(device)
                image_embeds = clip.get_image_features(**image_inputs)
                src_cls = image_embeds / _get_vector_norm(image_embeds)

            # 计算L2范数
            cls = cls.cpu().detach().numpy()
            src_cls = src_cls.cpu().detach().numpy()

            norms = np.linalg.norm(cls, axis=1, keepdims=True)
            eps = 1e-8
            cls = cls / (norms + eps)
            norms = np.linalg.norm(src_cls, axis=1, keepdims=True)
            src_cls = src_cls / (norms + eps) # [10, ]
            cos_score = src_cls @ cls.T # [10, classes]

            pred = np.argmax(cos_score, axis=1) # [10]
            result = (pred == label_id)
            cls_correct += result.sum()

            indices = numpy_topk_simple(cos_score, k, 1, True) # [10, 3]
            for x in range(1, k):
                result = result | (indices[:, x] == label_id)
            topk_correct += result.sum()
        accuracy[label] = cls_correct / total
        accuracy[label + f'_top{k}'] = topk_correct / total
    return accuracy

# 自定义Solarization
class Solarize:
    def __init__(self, threshold=0.5):
        self.threshold = threshold  # 0~1之间的阈值

    def __call__(self, img):
        return solarize(img, self.threshold)

if __name__ == '__main__':
    # 先根据 HF 预处理器拿到正确的 mean/std/size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ### 疾病分类任务。
    feat = AutoFeatureExtractor.from_pretrained('/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/clip_trained_weight_disease/CLIPModel_base')
    base_transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.Resize((feat.size['width'], feat.size['height'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),  # 归一化
    ])

    transform = transforms.Compose([
        # 几何变换
        # transforms.RandomHorizontalFlip(p=0.1),  # 随机水平翻转
        # transforms.RandomVerticalFlip(p=0.1),  # 随机垂直翻转
        # transforms.RandomApply([transforms.RandomRotation(15)], p=0.4),  # 随机旋转（±15°）
        # transforms.RandomPerspective(distortion_scale=0.15, p=0.2),  # 透视变换
        # transforms.Resize((224, 224)),
        transforms.Resize((feat.size['width'], feat.size['height'])),
        # 颜色变换
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1)], p=0.25),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),  # 扩展颜色抖动
        # Solarize(threshold=0.8),  # 自定义Solarize
        transforms.RandomGrayscale(p=0.1),  # 随机灰度化
        # 空间变换
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),  # 归一化
        AddGaussianNoise(0, 0.1, 0.25),  # 自定义高斯噪声
        # transforms.RandomErasing(p=0.25, scale=(0.05, 0.1), ratio=(0.67, 1.33), value='random'),  # 随机擦除
    ])

    #src_path = '/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/disease_data/train_clip_data2_with_disease.json'
    src_path = '/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/disease_data/食管/train_clip_data2_with_shiguan_diease_filtered_rest.json'

    dataset = MedicalCLIPDataset(src_path, transform=transform, base_transform=base_transform, text_key='disease')
    loader  = DataLoader(dataset, batch_size=96, shuffle=True, num_workers=4) # 开启感知为 50，否则为 84.

    train_clip(loader, device, clip_path='/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/big_clip_trained_weight_disease/CLIPModel_10_712',
               epochs=10, lr=1e-6, save_dir='/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/big_clip_trained_weight_disease_shiguan',
               clip_text_path='')

    # train_biomed_clip(loader, device, clip_path='/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/BiomedCLIP/Finetune_BiomedCLIP',
    #            epochs=10, lr=1e-6, save_dir='/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/BiomedCLIP')

    # train_clip_with_mean_pool(loader, device, clip_path='/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/big_clip_trained_weight_disease/CLIPModel_3',
    #            epochs=3, lr=1e-5, freeze_text = True,
    #            save_dir='/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/big_clip_trained_text_med_v4',
    #            clip_text_path='/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/big_clip_trained_text_med_v4/medical_embedded_v4_disease_3')
