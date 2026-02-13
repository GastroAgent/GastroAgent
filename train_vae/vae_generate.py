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
from torchvision import datasets
from transformers import AutoFeatureExtractor
from torch.nn.utils import clip_grad_norm_
from random import choices, choice
from utils.utils_ import _get_vector_norm
import torch
from PIL import Image
import numpy as np
from torchvision.utils import save_image
from vae_sim import VAE
from vqae_sim import VQAE

# Step 1: 自定义 AddGaussianNoise 类
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): (C, H, W) 的张量
        Returns:
            Tensor: 添加高斯噪声后的图像
        """
        noise = torch.randn(tensor.shape) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_paths = []
        for root, dirs, files in os.walk(root_dir):
            for fname in files:
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.img_paths.append(os.path.join(root, fname))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

# 定义逆向的 Normalize 操作
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """将张量还原为原始图像范围 [0, 1]"""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # 反向操作: (t * std) + mean
    return tensor

def single_generated_save(vae, image_path, device, transforms, base_transforms, feat, save_dir, sample_posterior = True):
    vae.eval()
    os.makedirs(save_dir, exist_ok=True)
    image = Image.open(image_path).convert('RGB')
    image.save(os.path.join(save_dir, "raw_image.jpg"))

    base_img = base_transforms(image)
    img = transforms(image)
    denorm_image = denormalize(img.clone(), mean=feat.image_mean, std=feat.image_std)
    # 保存图像
    save_image(denorm_image, os.path.join(save_dir, "aug_image.jpg"))

    with torch.no_grad():
        try:
            output = vae(img.unsqueeze(0).to(device))
            recon_x = output.recon  # Raw VAE 返回的是 return DecoderOutput(sample=dec), posterior
        except Exception as e:
            print(e)
            print('-'*100)
            output = vae(img.unsqueeze(0).to(device), sample_posterior=sample_posterior)
            recon_x = output.sample

    print(recon_x.shape)
    denorm_image = denormalize(recon_x.clone(), mean=feat.image_mean, std=feat.image_std)
    save_image(denorm_image, os.path.join(save_dir, "gen_aug_image.jpg"))

    with torch.no_grad():
        try:
            output = vae(base_img.unsqueeze(0).to(device))
            recon_x = output.recon  # Raw VAE 返回的是 return DecoderOutput(sample=dec), posterior
        except:
            output = vae(base_img.unsqueeze(0).to(device), sample_posterior=sample_posterior)
            recon_x = output.sample

    denorm_image = denormalize(recon_x.clone(), mean=feat.image_mean, std=feat.image_std)
    save_image(denorm_image, os.path.join(save_dir, "gen_image.jpg"))

def single_generated(vae, image_path, device, transforms):
    vae.eval()
    image = Image.open(image_path).convert('RGB')
    img = transforms(image)

    with torch.no_grad():
        output = vae(img.unsqueeze(0).to(device))
        try:
            recon_x = output.recon  # Raw VAE 返回的是 return DecoderOutput(sample=dec), posterior
        except:
            recon_x = output[0].sample
    return recon_x

#
def consistency_generate():
    from diffusers import StableDiffusionPipeline
    from consistencydecoder import ConsistencyDecoder, save_image, load_image

    # encode with stable diffusion vae
    pipe = StableDiffusionPipeline.from_pretrained(
        "/home/dalhxwlyjsuo/criait_tansy/weight/stable-diffusion-v1-5", torch_dtype=torch.bfloat16, device="cuda"
    )
    pipe.vae.cuda()
    decoder_consistency = ConsistencyDecoder(device="cuda", download_root='/home/dalhxwlyjsuo/criait_tansy/weight')  # Model size: 2.49 GB

    image = load_image("/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/our_eval_data/new_cropped-2004-2010-endovit/01.2018030100098_7-VAE/raw_image.jpg", size=(512, 480), center_crop=False)
    latent = pipe.vae.encode(image.to(torch.bfloat16).cuda()).latent_dist.mean

    # decode with gan
    sample_gan = pipe.vae.decode(latent).sample.detach().float()
    save_image(sample_gan, "/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/our_eval_data/new_cropped-2004-2010-endovit/01.2018030100098_7-GAN.jpg")
    print(pipe.vae)
    print('-'*100)
    # decode with vae
    sample_consistency = decoder_consistency(latent).float()
    print(decoder_consistency)
    save_image(sample_consistency, "/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/our_eval_data/new_cropped-2004-2010-endovit/01.2018030100098_7-Consistency.jpg")

if __name__ == '__main__':
    # consistency_generate()

    # 先根据 HF 预处理器拿到正确的 mean/std/size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feat = AutoFeatureExtractor.from_pretrained('/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/clip_trained_weight_disease/CLIPModel_base')
    base_transform = transforms.Compose([
        transforms.Resize((feat.size['width'], feat.size['height'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[
            0.5,
            0.5,
            0.5
          ], std=[
            0.5,
            0.5,
            0.5
          ]),
    ])
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[
        0.5,
        0.5,
        0.5
      ], std=[
        0.5,
        0.5,
        0.5
      ]),
        # AddGaussianNoise(0, 0.25),
        # transforms.RandomErasing(p=0.25, scale=(0.01, 0.05), ratio=(0.67, 1.33), value='random')
    ])

    # ### Vit-VQVAE
    # encoder_ckpt = ''
    # decoder_ckpt = '/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/vqvae_weight/VQVAEModel'
    # vae = VAE(latent_dim=4, encoder_ckpt=encoder_ckpt, decoder_ckpt=decoder_ckpt, use_VQVAE=True).to(device).train()
    # if True:
    #     state_dict = torch.load('/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/vqvae_weight/vqvae_epoch_ema.pth')
    #     vae.load_state_dict(state_dict, strict=False)

    ### Vit-VAE
    encoder_ckpt = '/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/flow_matcher_otcfm/EndoViT/pytorch_model.bin'
    decoder_ckpt = '/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/vae_weight/VAEModel'
    vae = VAE(latent_dim=4, encoder_ckpt=encoder_ckpt, decoder_ckpt=decoder_ckpt, use_VQVAE=False).to(device).train()
    # Optional: load pre-trained VAE
    try:
        state_dict = torch.load('/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/flow_matcher_otcfm/vit_vae/vit_vae_ema.pth', map_location=device)
        vae.load_state_dict(state_dict, strict=False)
        print("Loaded pre-trained VAE weights.")
    except Exception as e:
        print("No pre-trained VAE found:", e)

    # ### Conv VAE
    # vae = AutoencoderKL.from_pretrained(
    #     '/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/sd-vae-high-res_weight/sd-vae-ft-mse-high-res').to(device).eval()
    # vae.load_state_dict(
    #     torch.load('/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/sd-vae-high-res_weight/sd-vae_epoch_ema_711.pth'), strict=False)

    ### Raw ema-VAE
    # vae = AutoencoderKL.from_pretrained(
    #     '/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/flow_matcher_otcfm/vae_our').to(device).eval()

    # ## Raw mse-VAE
    # vae = AutoencoderKL.from_pretrained(
    #     '/home/dalhxwlyjsuo/criait_tansy/weight/sd-vae-ft-mse').to(device).eval()

    # ### Raw ema-VAE
    # vae = AutoencoderKL.from_pretrained(
    #     '/home/dalhxwlyjsuo/criait_tansy/weight/sd-vae-ft-ema').to(device).eval()

    # ### VAE
    # encoder_ckpt = ''
    # decoder_ckpt = '/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/vae_weight/VAEModel'
    # vae = VAE(latent_dim=4, encoder_ckpt=encoder_ckpt, decoder_ckpt=decoder_ckpt, use_VQVAE=False).to(device).eval()
    # vae.load_state_dict(torch.load('/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/vae_weight/vae_epoch8.pth'), strict=False)
    #
    # print(vae)
    # images_root = '/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/our_eval_data/Fake_DiaEvalImages'

    # base_transforms = transforms.Compose([
    #     transforms.Resize((336, 336)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=feat.image_mean, std=feat.image_std),
    # ])
    # image = Image.open('/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/our_eval_data/Fake_DiaEvalImages/出血糜烂性胃炎/01.2017011900019_15.jpg').convert('RGB')
    # base_img = base_transforms(image)
    # denorm_image = denormalize(base_img.clone(), mean=feat.image_mean, std=feat.image_std)
    # # 保存图像
    # save_image(denorm_image, '/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/src/336.jpg')
    # /mnt/inaisfs/data/home/tansy_criait/weights/Qwen-Image-Edit-2511/01.2021082610132_1.jpg
    # /mnt/inaisfs/data/home/tansy_criait/weights/Qwen-Image-Edit-2511/01.2021083100100_3.jpg
    single_generated_save(vae, '/mnt/inaisfs/data/home/tansy_criait/weights/Qwen-Image-Edit-2511/01.2021082610132_1.jpg',
                          device,
                          transform,
                          base_transform,
                          feat,
                          "/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/vae_src/2021082610132_1-image3-Vit-VAE",
                          sample_posterior = True
    )

    print("Done.")



