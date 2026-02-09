import gc
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import json
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
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
from transformers import AutoModel, AutoProcessor, AutoImageProcessor
import torch.nn.functional as F
from dataclasses import dataclass
import random
import math
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from transformers import AutoFeatureExtractor
from torch.nn.utils import clip_grad_norm_
from random import choices, choice
from utils_ import _get_vector_norm
import ot

def compute_wasserstein_ot(z1: torch.Tensor, z2: torch.Tensor, p: int = 2) -> float:
    """
    使用 Optimal Transport 计算 Wasserstein 距离
    z1, z2: [N, D] 张量表示两个分布的采样点
    p: 距离范数（p=1 对应 Wasserstein-1）
    """
    if z1.ndim > 2:
        D = z1.shape[1]
        z1 = z1.reshape(-1, D)
    if z2.ndim > 2:
        D = z2.shape[1]
        z1 = z2.reshape(-1, D)
    # 转成 numpy
    x = z1.detach().cpu().numpy()
    y = z2.detach().cpu().numpy()

    # 样本权重（均匀分布）
    a = np.ones((x.shape[0],)) / x.shape[0]
    b = np.ones((y.shape[0],)) / y.shape[0]

    # 成本矩阵：每两个点的 Lp 距离（可改为其他距离度量）
    M = ot.dist(x, y, metric='euclidean') ** p  # 如果是 p=1 就不需要 **p

    # 求解最优传输距离
    wasserstein_distance = ot.emd2(a, b, M)  # 返回的是 Wasserstein-p^p 距离
    return wasserstein_distance ** (1 / p)

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

def eval_clip_similatiry(vision_model, images_root, device, processor, k=5):
    vision_model.eval()
    accuracy = {}
    labels = os.listdir(images_root)
    for label_id, label in enumerate(labels):
        print(label)
        src_path = os.path.join(images_root, label)
        cls_correct = 0
        topk_correct = 0
        total = 0
        for _ in tqdm(range(10)):
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
            image_input = processor(images, return_tensors="pt").to(device)
            src_image_input = processor(src_imgs, return_tensors="pt").to(device)

            with torch.no_grad():
                cls = vision_model(**image_input).last_hidden_state
                cls = F.normalize(cls[:, 0], p=2, dim=1)

                src_cls = vision_model(**src_image_input).last_hidden_state
                src_cls = F.normalize(src_cls[:, 0], p=2, dim=1)

            cls = cls.cpu().detach().numpy()
            src_cls = src_cls.cpu().detach().numpy()
            cos_score = src_cls @ cls.T  # [10, classes]
            print('Cos Score: ', cos_score.shape, cos_score)
            pred = np.argmax(cos_score, axis=1)  # [10]
            result = (pred == label_id)
            cls_correct += result.sum()

            indices = numpy_topk_simple(cos_score, k, 1, True)  # [10, 3]
            for x in range(1, k):
                result = result | (indices[:, x] == label_id)
            topk_correct += result.sum()
            break
        accuracy[label] = cls_correct / total
        accuracy[label + f'_top{k}'] = topk_correct / total
        print(cls_correct / total, topk_correct / total)
    return accuracy

def eval_clip_text_similatiry(text_model, vision_model, images_root, device, processor, k=5, dataset = None):
    text_model.eval()
    vision_model.eval()
    accuracy = {}
    total = 10
    labels = os.listdir(images_root)
    if dataset is None:
        dataset = {}
        # 创建一个数据集
        GTs = []
        Images = []
        for label_id, label in enumerate(labels):
            print(label)
            src_path = os.path.join(images_root, label)
            for _ in tqdm(range(total)):
                GTs.append(label_id)
                Images.append(os.path.join(src_path, random.choice(os.listdir(src_path))))
        dataset['images'] = Images
        dataset['gts'] = GTs
        with open('/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/src/eval_disease_classify.json', 'w') as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)
    else:
        Images = dataset['images']
        GTs = dataset['gts']

    text_embeds = text_model.encode(labels, normalize_embeddings=True)

    for image, gt in zip(Images, GTs):
        image = [Image.open(image).convert('RGB')]
        with torch.no_grad():
            inputs = processor(image, return_tensors="pt").to(device)
            img_emb = vision_model(**inputs).last_hidden_state
            image_embeds = F.normalize(img_emb[:, 0], p=2, dim=1)
            image_embeds = image_embeds.cpu().detach().numpy() # [1, 768]
            cos_score = image_embeds @ text_embeds.T  # [1, labels]

            pred = np.argmax(cos_score, axis=1)
            result = (pred == gt)
            cls_correct = result.sum()

            indices = numpy_topk_simple(cos_score, k, 1, True)  # [5, 3]
            for x in range(1, k):
                result = result | (indices[:, x] == gt)
            topk_correct = result.sum()

        try:
            accuracy[labels[gt]] += cls_correct / total
            accuracy[labels[gt] + f'_top{k}'] += topk_correct / total
        except:
            accuracy[labels[gt]] = cls_correct / total
            accuracy[labels[gt] + f'_top{k}'] = topk_correct / total

    print('Label Size: ', len(labels))
    return accuracy


if __name__ == '__main__':
    # 先根据 HF 预处理器拿到正确的 mean/std/size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feat = AutoFeatureExtractor.from_pretrained('/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/big_clip_trained_weight_disease/CLIPModel_large')
    base_transform = transforms.Compose([
        transforms.Resize((feat.size['height'], feat.size['width'])),
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feat.image_mean, std=feat.image_std),
    ])
    clip_path = '/home/dalhxwlyjsuo/criait_tansy/weight/BiomedCLIP'
    images_root = '/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/our_eval_data/Fake_DiaEvalImages'
    from sentence_transformers import SentenceTransformer

    print('CLIP Result: ')
    text_model = SentenceTransformer("/home/dalhxwlyjsuo/criait_tansy/weight/nomic-embed-text-v1.5", trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained("/home/dalhxwlyjsuo/criait_tansy/weight/nomic-embed-vision-v1.5")
    vision_model = AutoModel.from_pretrained("/home/dalhxwlyjsuo/criait_tansy/weight/nomic-embed-vision-v1.5", trust_remote_code=True)
    print(text_model)
    print(vision_model)
    text_model = text_model.to(device)
    vision_model = vision_model.to(device)

    result = eval_clip_similatiry(vision_model, images_root, device, processor)
    for k, v in result.items():
        print(k, v)
    with open('/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/eval_nomic_707.json', 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    result = eval_clip_text_similatiry(text_model, vision_model, images_root, device, processor)
    for k, v in result.items():
        print(k, v)
    with open('/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/eval_nomic_text_707.json', 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

