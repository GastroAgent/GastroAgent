import gc
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
from torch.nn import Identity
from safetensors.torch import load_file
from diffusers.models import AutoencoderKL
from transformers import ViTModel, ViTConfig, AutoTokenizer, AutoModel, AutoConfig
from transformers import ChineseCLIPTextConfig as CLIPConfig
from transformers import ChineseCLIPProcessor as CLIPProcessor
from transformers import ChineseCLIPModel as CLIPModel
from transformers import ChineseCLIPModel
from transformers import ChineseCLIPTextConfig

from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers.models.autoencoders.vae import VectorQuantizer
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
from vae_sim import kl_divergence_vae, symmetric_kl_vae, js_kl_vae, VAE, vae_loss, AddGaussianNoise
from vqae_sim import VQAE
import random


# import ot

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


def eval_vae_cls_similatiry(vae, images_root, device, transforms, k=5):
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
            print(latent.shape, vae.latent_h, vae.latent_w)
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

            indices = numpy_topk_simple(cos_score, k, 1, True)  # [1, 3]
            for x in range(1, k):
                result = result | (indices[:, x] == label_id)
            topk_correct += result.sum()

            # KL Accuracy.
            latent = latent.view(latent.shape[0], vae.latent_h, vae.latent_w, latent.shape[-1])
            latent = vae.fc_mu_logvar(latent).permute(0, 3, 1, 2)
            latent = vae.quant_conv(latent)
            mu, logvar = latent.chunk(2, dim=1)
            mu = mu.cpu().detach().numpy()
            std = torch.exp(logvar / 2).cpu().detach().numpy()
            # print('mu', mu.shape)

            src_latent = src_latent.view(src_latent.shape[0], vae.latent_h, vae.latent_w, src_latent.shape[-1])
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


def mse(matrix_a, matrix_b):
    # 确保输入是 numpy 数组
    matrix_a = np.array(matrix_a)
    matrix_b = np.array(matrix_b)

    # 确保形状一致
    if matrix_a.shape != matrix_b.shape:
        raise ValueError("矩阵 A 和 B 必须具有相同的形状")

    # 计算 MSE
    return np.mean((matrix_a - matrix_b) ** 2)


def eval_diff_vae_similatiry(model, images_root, device, transforms, k=5, dataset=None):
    model.eval()
    accuracy = {}
    kl_correct = 0
    kl_topk_correct = 0
    js_correct = 0
    js_topk_correct = 0
    mm_correct = 0
    mm_topk_correct = 0

    labels = os.listdir(images_root)
    if dataset is None:
        dataset = []
        # 创建一个数据集
        for label_id, label in enumerate(labels):
            pos_images = []
            neg_images = []
            dir = os.path.join(images_root, label)
            image_names = os.listdir(dir)
            for image_name in image_names:
                if 'pos' in image_name:
                    pos_images.append(os.path.join(dir, image_name))
                elif 'neg' in image_name:
                    neg_images.append(os.path.join(dir, image_name))
            if len(pos_images) > 1:
                dataset.append({
                    'pos_images': pos_images,
                    'neg_images': neg_images,
                    'label': label
                })

        with open('/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/src/eval_diff_vae_dataset.json', 'w') as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)
    else:
        pass

    total = 0
    for data in tqdm(dataset):
        for pos_image in data['pos_images']:
            total += 1
            images_gts = [(x, 1) if 'pos' in x else (x, 0) for x in data['neg_images'] + data['pos_images'] if
                          x != pos_image]
            images, gts = zip(*images_gts)
            theld = len(data['neg_images'])

            neg_imgs = [transforms(Image.open(x).convert('RGB')) for x in images]
            pos_imgs = [transforms(Image.open(pos_image).convert('RGB'))]
            neg_imgs = torch.stack(neg_imgs)  # [nn + np - 1, ...]
            pos_imgs = torch.stack(pos_imgs)  # [np, ...]

            with torch.no_grad():
                _, pos_latent_dist = model(pos_imgs.to(model.device), sample_posterior=True)
                neg_latent_dist = []
                for x in neg_imgs:
                    x = x.unsqueeze(0)
                    _, neg_latent_dist_tmp = model(x.to(model.device), sample_posterior=True)
                    neg_latent_dist.append(neg_latent_dist_tmp)

            pos_mu = pos_latent_dist.mean.cpu().detach().numpy()  # # [1, ...]
            pos_std = torch.exp(pos_latent_dist.logvar / 2).cpu().detach().numpy()  # [1, ...]

            neg_mu = torch.cat([x.mean for x in neg_latent_dist], dim=0).cpu().detach().numpy()  # # [nn + np - 1, ...]
            neg_std = torch.exp(
                torch.cat([x.logvar for x in neg_latent_dist], dim=0) / 2).cpu().detach().numpy()  # [nn + np - 1, ...]

            kl_score = np.zeros([pos_mu.shape[0], neg_mu.shape[0]])  # [1, nn + np - 1]
            for i in range(pos_mu.shape[0]):
                for j in range(neg_mu.shape[0]):
                    _, kl = symmetric_kl_vae(pos_mu[i], pos_std[i], neg_mu[j], neg_std[j])
                    kl_score[i, j] = - kl

            pred = np.argmax(kl_score, axis=1)  # [1]
            if pred[0] >= theld:
                kl_correct += 1

            # 计算前top个是否有符合要求的。
            topk = numpy_topk_simple(kl_score, k, 1, True)[0]  # [1, k]
            for i in topk:
                if i >= theld:
                    kl_topk_correct += 1
                    break

            # 计算JS-KL。
            kl_score = np.zeros([pos_mu.shape[0], neg_mu.shape[0]])  # [1, nn + np - 1]
            for i in range(pos_mu.shape[0]):
                for j in range(neg_mu.shape[0]):
                    _, kl = js_kl_vae(pos_mu[i], pos_std[i], neg_mu[j], neg_std[j])
                    kl_score[i, j] = - kl

            pred = np.argmax(kl_score, axis=1)  # [1]
            if pred[0] >= theld:
                js_correct += 1

            # 计算前top个是否有符合要求的。
            topk = numpy_topk_simple(kl_score, k, 1, True)[0]  # [1, k]
            for i in topk:
                if i >= theld:
                    js_topk_correct += 1
                    break

            mse_score = np.zeros([pos_mu.shape[0], neg_mu.shape[0]])  # [1, nn + np - 1]
            for i in range(pos_mu.shape[0]):
                for j in range(neg_mu.shape[0]):
                    mse_dis = mse(pos_mu[i], neg_mu[j])
                    mse_score[i, j] = - mse_dis

            pred = np.argmax(mse_score, axis=1)  # [1]
            if pred[0] >= theld:
                mm_correct += 1

            # 计算前top个是否有符合要求的。
            topk = numpy_topk_simple(mse_score, k, 1, True)[0]  # [1, k]
            for i in topk:
                if i >= theld:
                    mm_topk_correct += 1
                    break

    label = 'mean'
    accuracy[label + '_js'] = js_correct / total
    accuracy[label + f'_j{k}'] = js_topk_correct / total
    accuracy[label + '_kl'] = kl_correct / total
    accuracy[label + f'_k{k}'] = kl_topk_correct / total
    accuracy[label + '_mm'] = mm_correct / total
    accuracy[label + f'_m{k}'] = mm_topk_correct / total
    return accuracy


def eval_clip_similatiry(clip, images_root, device, processor, k=5):
    clip.eval()
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
            with torch.no_grad():
                image_inputs = processor(images=images, return_tensors="pt").to(device)
                image_embeds = clip.get_image_features(**image_inputs, interpolate_pos_encoding=True)
                cls = image_embeds / _get_vector_norm(image_embeds)

                image_inputs = processor(images=src_imgs, return_tensors="pt").to(device)
                image_embeds = clip.get_image_features(**image_inputs, interpolate_pos_encoding=True)
                src_cls = image_embeds / _get_vector_norm(image_embeds)

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

def eval_clip_caption_similatiry(clip, images_root, device, processor, tokenizer, k=5, dataset=None,
                                 text_key: str = None):
    clip.eval()
    accuracy = {}
    if dataset is None:
        dataset = []
        eval_dataset = json.load(open(images_root, 'r'))

        for idx, data in enumerate(eval_dataset):
            eval_data = {}
            if 'caption' in data:
                text = data['caption']
            elif 'text' in data and 'disease' in data and data['disease'] != '':
                text = data['text'] + '最可能被诊断为: ' + data['disease']
            else:
                text = data['text']
            if text_key is not None and isinstance(text, str):
                text = data[text_key]
            image_file = data['image']
            id_ = data['id']
            eval_data = {
                'id': id_,
                'image': image_file,
                'text': text
            }
            dataset.append(eval_data)

        with open('/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/src/eval_caption_classify.json', 'w') as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)
    else:
        pass

    random_neg_text = [x['text'] for x in dataset]
    random_neg_image = [x['image'] for x in dataset]

    total = 0
    num_neg = 20
    text_correct = 0  # 1个Text——多个Image
    image_correct = 0  # 1个Image——多个Text
    text_topk_correct = 0  # 1个Text——多个Image
    image_topk_correct = 0  # 1个Image——多个Text

    for data in dataset:
        text = data['text']
        img_file = data['image']
        neg_img_files = choices(random_neg_image, k=num_neg)
        neg_img_files = [x for x in neg_img_files if x != img_file]
        neg_texts = choices(random_neg_text, k=num_neg)
        neg_texts = [x for x in neg_texts if x != text]

        with torch.no_grad():
            text_inputs = tokenizer([text], return_tensors="pt", padding=True).to(device)
            text_embeds = clip.get_text_features(**text_inputs)
            text_embeds = text_embeds / _get_vector_norm(text_embeds)
            src_text_embeds = text_embeds.clone()

            text_inputs = tokenizer(neg_texts, return_tensors="pt", padding=True).to(device)
            text_embeds = clip.get_text_features(**text_inputs)
            text_embeds = text_embeds / _get_vector_norm(text_embeds)
            neg_text_embeds = text_embeds.clone()  # [neg, 768]

            image_inputs = processor(images=[Image.open(img_file).convert('RGB')], return_tensors="pt").to(device)
            image_embeds = clip.get_image_features(**image_inputs, interpolate_pos_encoding=True)
            image_embeds = image_embeds / _get_vector_norm(image_embeds)
            src_image_embeds = image_embeds.clone()  # [1, 768]

            image_inputs = processor(images=[Image.open(x) for x in neg_img_files], return_tensors="pt").to(device)
            image_embeds = clip.get_image_features(**image_inputs, interpolate_pos_encoding=True)
            image_embeds = image_embeds / _get_vector_norm(image_embeds)
            neg_image_embeds = image_embeds.clone()  # [neg, 768]

            text_embeds = torch.cat([src_text_embeds, neg_text_embeds], dim=0)
            text_embeds = text_embeds.cpu().detach().numpy()  # [neg + 1, 768]
            image_embeds = torch.cat([src_image_embeds, neg_image_embeds], dim=0)
            image_embeds = image_embeds.cpu().detach().numpy()  # [neg + 1, 768]

            src_text_embeds = src_text_embeds.cpu().detach().numpy()
            src_image_embeds = src_image_embeds.cpu().detach().numpy()

        text_score = src_text_embeds @ image_embeds.T  # [1, neg + 1]
        image_score = src_image_embeds @ text_embeds.T  # [1, neg + 1]

        # 计算 text_correct
        pred = np.argmax(text_score, axis=1)  # [1]
        if pred[0] == 0:
            text_correct += 1
        indices = numpy_topk_simple(text_score, k, 1, True)  # [1, k]
        if any(indices[0] <= k):
            text_topk_correct += 1

        # 计算 image_correct
        pred = np.argmax(image_score, axis=1)  # [1]
        if pred[0] == 0:
            image_correct += 1
        indices = numpy_topk_simple(image_score, k, 1, True)  # [1, k]
        if any(indices[0] <= k):
            image_topk_correct += 1

        total += 1
    accuracy['Image Acc'] = image_correct / total
    accuracy['Text Acc'] = text_correct / total
    accuracy[f'Image Top{k} Acc'] = image_topk_correct / total
    accuracy[f'Text Top{k} Acc'] = text_topk_correct / total
    return accuracy


def eval_clip_text_similatiry_mean_pool(clip, images_root, device, processor, tokenizer, k=5, dataset=None):
    clip.eval()
    accuracy = {}
    total = 10
    labels = os.listdir(images_root)

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        # print('token_embeddings: ', token_embeddings.shape)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                      min=1e-9)

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
        with open('/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/src/eval_disease_classify.json', 'w') as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)
    else:
        Images = dataset['images']
        GTs = dataset['gts']

    with torch.no_grad():
        text_inputs = tokenizer(labels, padding=True, truncation=True, return_tensors='pt',
                                max_length=256).to(device)
        text_outputs = clip.text_model(**text_inputs)
        text_embeds = mean_pooling(text_outputs, text_inputs['attention_mask'])
        text_embeds = clip.text_projection(text_embeds)

    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds.cpu().detach().numpy()  # [labels, 768]
    print(text_embeds.shape)

    for image, gt in zip(Images, GTs):
        images = [Image.open(image).convert('RGB')]
        with torch.no_grad():
            image_inputs = processor(images=images, return_tensors="pt").to(device)
            image_embeds = clip.get_image_features(**image_inputs, interpolate_pos_encoding=True)
            image_embeds = image_embeds / _get_vector_norm(image_embeds)

            image_embeds = image_embeds.cpu().detach().numpy()  # [1, 768]

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


def eval_similatiry(vae, clip, images_root, device, processor, transforms, k=3):
    clip.eval()
    vae.eval()
    accuracy = {}
    labels = os.listdir(images_root)
    for label_id, label in enumerate(labels):
        src_path = os.path.join(images_root, label)
        norm_topk_correct = 0
        topk_correct = 0
        cls_correct = 0
        total = 0
        for _ in tqdm(range(10)):
            # print(f"{label_id}/{len(labels)}---{_}/20")
            src_images = choices(os.listdir(src_path), k=10)
            src_images = [os.path.join(src_path, x) for x in src_images]
            total += len(src_images)
            other_imgs = []
            for other in labels:
                image_dir = os.path.join(images_root, other)
                file_name = choice(os.listdir(image_dir))
                other_imgs.append(os.path.join(image_dir, file_name))

            images = [Image.open(x).convert('RGB') for x in other_imgs]
            src_imgs = [Image.open(x).convert('RGB') for x in src_images]
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

            clip_cos_score = src_cls @ cls.T  # [10, classes]

            other_imgs = [transforms(Image.open(x).convert('RGB')) for x in other_imgs]
            src_imgs = [transforms(Image.open(x).convert('RGB')) for x in src_images]
            images = torch.stack(other_imgs)
            with torch.no_grad():
                hidden_states = vae.encoder(images.to(device))
            cls = hidden_states[:, 0, ...]
            src_imgs = torch.stack(src_imgs)
            with torch.no_grad():
                hidden_states = vae.encoder(src_imgs.to(device))
            src_cls = hidden_states[:, 0, ...]
            # 计算L2范数
            cls = cls.cpu().numpy()
            src_cls = src_cls.cpu().numpy()

            norms = np.linalg.norm(cls, axis=1, keepdims=True)
            eps = 1e-8
            cls = cls / (norms + eps)
            norms = np.linalg.norm(src_cls, axis=1, keepdims=True)
            src_cls = src_cls / (norms + eps)  # [10, ]
            vae_cos_score = src_cls @ cls.T  # [10, classes]

            normed_cos_score = vae_cos_score / (
                    np.linalg.norm(vae_cos_score, axis=1, keepdims=True) + eps) + clip_cos_score / (
                                       np.linalg.norm(clip_cos_score, axis=1, keepdims=True) + eps)
            cos_score = vae_cos_score + clip_cos_score * 1.2

            pred = np.argmax(cos_score, axis=1)  # [10]
            result = (pred == label_id)
            cls_correct += result.sum()
            indices = numpy_topk_simple(cos_score, k, 1, True)  # [10, 3]
            for x in range(1, k):
                result = result | (indices[:, x] == label_id)
            topk_correct += result.sum()

            pred = np.argmax(normed_cos_score, axis=1)  # [10]
            result = (pred == label_id)
            indices = numpy_topk_simple(normed_cos_score, k, 1, True)  # [10, 3]
            for x in range(1, k):
                result = result | (indices[:, x] == label_id)
            norm_topk_correct += result.sum()
        accuracy[label + f'_norm_top{k}'] = norm_topk_correct / total
        accuracy[label + f'_top{k}'] = topk_correct / total
        accuracy[label] = cls_correct / total
    return accuracy


def load_model_mean_pool(clip_path, clip_text_path=''):
    device_map = 'auto'
    processor = CLIPProcessor.from_pretrained(clip_path, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(clip_path)
    try:
        clip = CLIPModel.from_pretrained(
            clip_path,
            device_map=device_map,
        ).train()
    except:
        base_clip_path = os.path.join(clip_path, 'config.json')
        new_clip_config = AutoConfig.from_pretrained(
            base_clip_path,
            device_map=device_map,
        )
        clip = CLIPModel(new_clip_config)
        clip.save_pretrained(clip_path)
        clip = CLIPModel.from_pretrained(
            clip_path,
            device_map=device_map,
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
    return clip, tokenizer, processor

if __name__ == '__main__':
    # 先根据 HF 预处理器拿到正确的 mean/std/size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_text_path = ''
    clip_path = '/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/big_clip_trained_weight_disease/CLIPModel_10_712'
    feat = AutoFeatureExtractor.from_pretrained(clip_path)
    base_transform = transforms.Compose([
        transforms.Resize((feat.size['width'], feat.size['height'])),
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feat.image_mean, std=feat.image_std),
    ])

    print('CLIP Result: ')
    processor = CLIPProcessor.from_pretrained(clip_path, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(clip_path)
    gpu_num = torch.cuda.device_count()
    if gpu_num > 1:
        print("Avail GPUs: ", gpu_num)
        try:
            device_map = json.load(
                open(f'/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/device_map_{gpu_num}.json', 'r'))
        except (FileNotFoundError, FileExistsError):
            print('Load Device Map')
            device_map = json.load(
                open(f'/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/device_map_{gpu_num}.json', 'r'))
        except:
            device_map = 'auto'
    else:
        device_map = 'auto'

    try:
        clip = CLIPModel.from_pretrained(
            clip_path,
            device_map=device_map,
        ).eval()
    except Exception as e:
        print(e)
        base_clip_path = os.path.join(clip_path, 'config.json')
        new_clip_config = CLIPConfig.from_pretrained(
            base_clip_path,
            device_map=device_map,
        )
        clip = CLIPModel(new_clip_config)
        clip.save_pretrained(clip_path)
        clip = CLIPModel.from_pretrained(
            clip_path,
            device_map=device_map,
        ).eval()

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

    images_root = '/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/disease_data/eval_clip_data2_with_diease_filtered.json'
    result = eval_clip_caption_similatiry(clip, images_root, device, processor, tokenizer)
    for k, v in result.items():
        print(k, v)
    with open('/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/eval_clip_caption_text_721.json', 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


