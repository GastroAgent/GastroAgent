### 验证 Stop Strategy 的性能。
from copy import deepcopy
import os
import random
from timm.models.vision_transformer import VisionTransformer
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'
# os.environ["TOKENIZERS_PARALLELISM="] = "False"
import sys
sys.path.append('/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match')
sys.path.append('/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/utils')
import glob
from data_loader_test import MedicalDataset, MedicalJsonDataset
from data_utils_test import *
from functools import partial
import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from torchvision import transforms
from torchvision import transforms as T
from torchvision.utils import save_image, make_grid
from torchcfm.models.unet.unet import UNetModelWrapper
from torch.utils.data import DataLoader
# ===== 对照度量：L2_RMS / LPIPS / SWD =====
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from data_loader_hiaug import MedicalCLIPTinyAUGDataset
import ot
from my_models.unet_2d_condition import UNet2DConditionModel
from my_models.model_dispatch import dispatch_model
from my_models.model_wass import AttentionAutoencoder
from diffusers import AutoencoderKL
from transformers import ChineseCLIPConfig as CLIPConfig
from transformers import ChineseCLIPProcessor as CLIPProcessor
from transformers import ChineseCLIPModel as CLIPModel
from transformers import AutoTokenizer, AutoModel, AutoConfig, ChineseCLIPTextModel, ChineseCLIPTextConfig
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import save_image, make_grid

def calculate_gray_image_similarity(image1, image2):
    """
    计算两张灰度图image1和image2之间的相似度。
    
    参数:
    - image1: PIL.Image格式的灰度图像。
    - image2: PIL.Image格式的灰度图像。
    
    返回:
    - float: 相似度分数，值越接近1表示越相似。
    """
    # 确保图片尺寸相同，如果不同则调整第二张图片大小以匹配第一张
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)
    
    # 将PIL图像转换为numpy数组
    img_array1 = np.array(image1)
    img_array2 = np.array(image2)
    
    # 计算SSIM
    score = ssim(img_array1, img_array2)
    
    return score

# ---- LPIPS 初始化（若没有安装会跳过） ----
_lpips_model = None
try:
    import lpips  # pip install lpips

    # _lpips_model = lpips.LPIPS(net='alex').to('cuda:0' if torch.cuda.is_available() else 'cpu').eval()
except Exception as e:
    print("[WARN] LPIPS 未安装或初始化失败，将跳过 LPIPS 计算：", e)

def process_single_image(image_path, input_size=224, dataset_mean=[0.3464, 0.2280, 0.2228],
                         dataset_std=[0.2520, 0.2128, 0.2093]):
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=dataset_mean, std=dataset_std)
    ])

    # Open the image
    image = Image.open(image_path).convert('RGB')

    # Apply the transformations
    processed_image = transform(image)
    return processed_image

def parse_args():
    parser = argparse.ArgumentParser(description='Sampling script for CFM model')
    parser.add_argument('--checkpoint', type=str,
                        default = '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v1/base-flow-match_vae_gan/otcfm/otcfm_weights_step_100000.pt',
                        # default = '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/outputs/image_hint_胃/otcfm/otcfm_weights_step_50000.pt',
                        # default = '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_十二指肠/outputs/image_hint_十二指肠3/otcfm/otcfm_weights_step_70000.pt',
                        help='Path to the checkpoint file')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Total number of images to generate')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for generation')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v1/generated_all_data/image_hint_gan_100000',
                        help='Directory to save generated images')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512],
                        help='Image size (height, width)')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='Number of steps in the ODE solver')
    parser.add_argument('--num_channels', type=int, default=128,
                        help='Number of base channels in UNet')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Use EMA model for inference')
    parser.add_argument('--solver', type=str, default='heun',
                        choices=['euler', 'heun'],
                        help='ODE solver type')
    parser.add_argument('--save_grid', action='store_true', default=False,
                        help='Save grid of samples for each batch')
    parser.add_argument('--save_intermediates', action='store_true', default=False,
                        help='Save intermediate steps during generation')
    parser.add_argument('--intermediate_freq', type=int, default=1,
                        help='Frequency of saving intermediate steps')
    parser.add_argument('--wass_model_path', type=str, 
                        default="",
                        help='')
    # 新增：选择在哪个空间计算 W2 / 度量  wass_model_type
    parser.add_argument('--wass_model_type', type=str, 
                        choices=['resnet34', 'attention'],
                        default="attention", # resnet34 或 attention
                        help='指定模型类型')

    # 新增：选择在哪个空间计算 W2 / 度量
    parser.add_argument(
        '--wass_space',
        type=str,
        choices=['latent', 'image_pixels', 'image_feats'],
        default='latent',
        help="Where to compute distances: 'latent' (VAE latent), 'image_pixels' (decode to RGB), "
             "'image_feats' (decode then feature space)."
    )
    return parser.parse_args()

class ImageGenerator:
    def __init__(self, args):
        self.args = args
        self.use_gt_vt = False
        if self.use_gt_vt:
            self.args.solver = "euler"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net_model, vae, (text_model, text_tokenizer), (vision_model, process_single_image) = self._load_model(
            args.checkpoint, args.use_ema)
        self.net_model = net_model
        self.vae = vae
        self.text_model = text_model
        self.text_tokenizer = text_tokenizer
        self.vision_model = vision_model
        self.process_single_image = process_single_image
        self._setup_directories()
        self.intermediate_images = []  # Store intermediate images

    def _setup_directories(self):
        """Create necessary directories for outputs"""
        os.makedirs(self.args.output_dir, exist_ok=True)
        if self.args.save_intermediates:
            self.intermediates_dir = os.path.join(self.args.output_dir, 'intermediates')
            os.makedirs(self.intermediates_dir, exist_ok=True)

    def _load_model(self, checkpoint: str, use_ema=False):
        """Initialize and load the model"""
        if not self.use_gt_vt:
            config = json.load(open(
                '/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/flow_matcher_otcfm/unet/config.json',
                'r'))
            net_model = UNet2DConditionModel(**config)
            # class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
            # init.normal_(class_embedding.weight, mean=0.0, std=0.01)  # 正态分布初始化
            state_dict = torch.load(f'{checkpoint}', map_location='cpu')
            if use_ema and 'ema_model' in state_dict:
                net_model.load_state_dict(state_dict['ema_model'], strict=False)
            elif 'net_model' in state_dict:
                net_model.load_state_dict(state_dict['net_model'], strict=False)

            vae = AutoencoderKL.from_pretrained(
                '/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/flow_matcher_otcfm/vae').to(
                device='cuda:0').eval()

            # from vae_sim import VAE
            # encoder_ckpt = '/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/flow_matcher_otcfm/EndoViT/pytorch_model.bin'
            # decoder_ckpt = '/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/vae_weight/VAEModel'
            # vae = VAE(latent_dim=4, encoder_ckpt=encoder_ckpt, decoder_ckpt=decoder_ckpt, use_VQVAE=False).to(self.device).eval()
            # # Optional: load pre-trained VAE
            # vae.device = self.device
            # state_dict = torch.load('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/flow_matcher_otcfm/vit_vae/vit_vae_ema.pth', map_location=self.device)
            # vae.load_state_dict(state_dict, strict=False)
            # print("Loaded pre-trained VAE weights.")
            # device = self.device

            # vae.load_state_dict(torch.load('/dev/shm/jmf/mllm_weight/sd-ema-vae_weight/sd-vae_epoch_ema.pth'), strict=False)
            text_model_config = json.load(open('/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/flow_matcher_otcfm/text_encoder/config.json','r'))
            text_model_config = ChineseCLIPTextConfig(**text_model_config)
            text_model = ChineseCLIPTextModel(text_model_config).eval()
            text_tokenizer = AutoTokenizer.from_pretrained(
                '/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/flow_matcher_otcfm/text_encoder', use_fast=True)
            if 'text_model' in state_dict:
                text_model.load_state_dict(state_dict['text_model'], strict=False)

            # Define the model (ensure this matches your model's architecture)
            vision_model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                             qkv_bias=True,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6))
            if 'vision_model' in state_dict:
                vision_model.load_state_dict(state_dict['vision_model'], strict=False)

            net_model = net_model.to(device='cuda:0')
            vae = vae.to('cuda:0')
            if text_model is not None:
                text_model = text_model.to('cuda:0')
            if vision_model is not None:
                vision_model = vision_model.to('cuda:0')
                vision_model.device = 'cuda:0'

            from my_models.model_dispatch import dispatch_model
            net_model, vae, text_model, vision_model = dispatch_model(net_model, vae, text_model, vision_model, num_device = 1)
            def process_single_image(image_path, input_size=224, dataset_mean=[0.3464, 0.2280, 0.2228],
                                    dataset_std=[0.2520, 0.2128, 0.2093]):
                transform = T.Compose([
                    T.Resize((input_size, input_size)),
                    T.ToTensor(),
                    T.Normalize(mean=dataset_mean, std=dataset_std)
                ])
                # Open the image
                image = Image.open(image_path).convert('RGB')

                # Apply the transformations
                processed_image = transform(image)
                return processed_image
            
            if  self.args.wass_model_path:
                from train_vae_with_wass import TripletNetwork
                if 'attention.pt' in self.args.wass_model_path:
                    model = TripletNetwork(pretrained=True, freeze_base=False, model='attention')
                else:
                    model = TripletNetwork(pretrained=True, freeze_base=False, model='resnet34')
                state_dict = torch.load(self.args.wass_model_path, weights_only=True)
                model.load_state_dict(state_dict, strict=False)
                model = model.to("cuda")
                self.wass_model = model
                self.wass_model.eval()
            elif 'wass_model' in state_dict:
                from train_vae_with_wass import TripletNetwork
                if self.args.wass_model_type == '':
                    self.args.wass_model_type = 'resnet34'
                model = TripletNetwork(pretrained=True, freeze_base=False, model=self.args.wass_model_type)
                model.load_state_dict(state_dict['wass_model'], strict=False)
                model = model.to("cuda")
                self.wass_model = model
                self.wass_model.eval()
            else:   
                self.wass_model = None
            return net_model, vae, (text_model, text_tokenizer), (vision_model, process_single_image)

        else:
            net_model = None
            text_model = None
            text_tokenizer = None
            vision_model = None
            process_single_image = None
            vae = AutoencoderKL.from_pretrained(
                '/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/flow_matcher_otcfm/vae').to(
                device='cuda:0').eval()

            # from vae_sim import VAE
            # encoder_ckpt = '/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/flow_matcher_otcfm/EndoViT/pytorch_model.bin'
            # decoder_ckpt = '/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/vae_weight/VAEModel'
            # vae = VAE(latent_dim=4, encoder_ckpt=encoder_ckpt, decoder_ckpt=decoder_ckpt, use_VQVAE=False).to(device).eval()
            # # Optional: load pre-trained VAE
            # vae.device = device
            # state_dict = torch.load('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/flow_matcher_otcfm/vit_vae/vit_vae_ema.pth', map_location=device)
            # vae.load_state_dict(state_dict, strict=False)
            # print("Loaded pre-trained VAE weights.")

            return net_model, vae, (text_model, text_tokenizer), (vision_model, process_single_image)

    @staticmethod
    def normalize_samples(x):
        """
          对每个样本独立进行最大最小化归一化 (Min-Max Scaling)

          参数:
              x: Tensor, shape (B, C, H, W)
              new_min: float, 缩放后的最小值
              new_max: float, 缩放后的最大值
              eps: float, 防止除以0的小值

          返回:
              x_scaled: Tensor, shape (B, C, H, W), 缩放后的数据
          """
        x = (x / 2 + 0.5)
        # 计算每个样本的全局最小值和最大值 (在 C, H, W 维度上)
        # keepdim=True 保证维度不变，便于广播
        x_min = x.amin(dim=(1, 2, 3), keepdim=True)  # shape: (B, 1, 1, 1)
        x_max = x.amax(dim=(1, 2, 3), keepdim=True)  # shape: (B, 1, 1, 1)

        # 防止分母为0
        scale = (x_max - x_min).clamp(min=1e-4)

        # 执行缩放
        x_scaled = (x - x_min) / scale
        return x_scaled
        # """Normalize samples to [0, 1] range"""
        # # x = x.clip(-1, 1)
        # x = (x / 2 + 0.5)
        # return x.clip(0, 1)

    def store_intermediate(self, x, step_idx):
        """Store intermediate generation steps"""
        x = self.vae.decode(x / 0.18215).sample
        samples = self.normalize_samples(x)
        grid = make_grid(samples, nrow=samples.shape[0])
        grid_np = grid.cpu().numpy().transpose(1, 2, 0)
        self.intermediate_images.append((step_idx, grid_np))

    def save_intermediate_grid(self, batch_idx):
        """Save all intermediate steps in a single matplotlib grid"""
        if not self.intermediate_images:
            return

        num_steps = len(self.intermediate_images)
        fig_width = 15
        fig_height = (fig_width * num_steps) / 4  # Adjust aspect ratio

        fig, axes = plt.subplots(num_steps, 1, figsize=(fig_width, fig_height))
        if num_steps == 1:
            axes = [axes]

        fig.suptitle(f'Generation Progress (Batch {batch_idx})', fontsize=16)

        for (step_idx, img), ax in zip(self.intermediate_images, axes):
            ax.imshow(img)
            ax.set_title(f'Step {step_idx}')
            ax.axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.intermediates_dir, f'progress_batch_{batch_idx:03d}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        # Clear stored intermediates
        self.intermediate_images = []

    def euler_solver(self, x0, t_span, batch_idx, batch, pbar=None):
        """Euler solver that also returns the full latent trajectory"""
        x = x0
        x1 = batch['x1']
        dt = t_span[1] - t_span[0]
        caption_hidden_states = batch['caption_hidden_states']
        y = batch['y']

        # 用来收集每个 time step 的 latent
        trajectory = [x.detach().clone()]

        # use_gt_vt = self.use_gt_vt
        use_gt_vt = True

        for t_idx in range(len(t_span) - 1):
            t = t_span[t_idx] * torch.ones(x0.shape[0], device=self.device)

            ### 使用 真实 速度场。
            if use_gt_vt:
                ut = batch['ut']
                dx = ut
            else:
                dx = self.net_model(x, timestep=t.squeeze(), class_labels=y,
                                    encoder_hidden_states=caption_hidden_states,
                                    added_cond_kwargs=batch).sample
            x = x + dx * dt

            # 收集
            if (t_idx + 1) % self.args.intermediate_freq == 0:
                trajectory.append(x.detach().clone())
        trajectory.append(x1.detach().clone())
        return x, trajectory

    def heun_solver(self, x0, t_span, batch_idx, batch, pbar=None):
        """Heun's solver that saves and returns the full latent trajectory."""
        x = x0
        x1 = batch['x1']
        dt = t_span[1] - t_span[0]
        caption_hidden_states = batch['caption_hidden_states']
        y = batch['y']

        # 用来收集轨迹
        trajectory = [x.detach().clone()]
        for t_idx in range(len(t_span) - 1):
            t = t_span[t_idx] * torch.ones(x0.shape[0], device=self.device)
            t_next = t_span[t_idx + 1] * torch.ones(x0.shape[0], device=self.device)

            # First step: Euler
            dx1 = self.net_model(x, timestep=t.squeeze(), class_labels=y,
                                 encoder_hidden_states=caption_hidden_states, added_cond_kwargs=batch).sample
            x_euler = x + dx1 * dt

            # Second step: Correction
            dx2 = self.net_model(x_euler, timestep=t_next.squeeze(), class_labels=y,
                                 encoder_hidden_states=caption_hidden_states, added_cond_kwargs=batch).sample
            x = x + (dx1 + dx2) * dt / 2

            # 收集当前 latent
            if (t_idx + 1) % self.args.intermediate_freq == 0:
                trajectory.append(x.detach().clone())
        trajectory.append(x1.detach().clone())
        # 返回最终的 x 以及完整的轨迹
        return x, trajectory

    def save_batch(self, samples, batch_idx, start_idx):
        """Save a batch of generated samples"""
        # Save individual samples
        output_dir = os.path.join(
            self.args.output_dir,
            f'samples'
        )
        os.makedirs(output_dir, exist_ok=True)
        for i, sample in enumerate(samples):
            sample_idx = start_idx + i
            individual_path = os.path.join(
                output_dir,
                f'sample_{sample_idx:05d}.png'
            )
            save_image(sample, individual_path)

        # Save grid
        if self.args.save_grid:
            output_dir = os.path.join(
                self.args.output_dir,
                f'grid'
            )
            os.makedirs(output_dir, exist_ok=True)
            grid_path = os.path.join(
                output_dir,
                f'grid_batch_{batch_idx:03d}.png'
            )
            save_image(
                samples,
                grid_path,
                nrow=min(8, int(samples.shape[0] ** 0.5))
            )

        # Save intermediate progress grid
        if self.args.save_intermediates:
            self.save_intermediate_grid(batch_idx)

    def _flatten_features(self, z, add_coords=False, coord_lambda=0.0):
        """
        z: [C, H, W] (单张) 或 [B, C, H, W]（批）
        返回： [N, D] 或 [B, N, D]，其中 D = C (+ 2 如果拼空间坐标)
        """
        if z.dim() == 4:
            B, C, H, W = z.shape
            feats = z.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B,N,C]
            if add_coords:
                yy, xx = torch.meshgrid(
                    torch.linspace(0, 1, H, device=z.device),
                    torch.linspace(0, 1, W, device=z.device),
                    indexing='ij'
                )
                coords = torch.stack([yy, xx], dim=-1).reshape(1, H * W, 2).repeat(B, 1, 1)  # [B,N,2]
                feats = torch.cat([coord_lambda * coords, feats], dim=-1)  # [B,N,C+2]
            return feats  # [B,N,D]
        else:
            C, H, W = z.shape
            feats = z.permute(1, 2, 0).reshape(H * W, C)  # [N,C]
            if add_coords:
                yy, xx = torch.meshgrid(
                    torch.linspace(0, 1, H, device=z.device),
                    torch.linspace(0, 1, W, device=z.device),
                    indexing='ij'
                )
                coords = torch.stack([yy, xx], dim=-1).reshape(H * W, 2)  # [N,2]
                feats = torch.cat([coord_lambda * coords, feats], dim=-1)  # [N,C+2]
            return feats  # [N,D]

    def _random_subsample(self, X, max_points):
        """
        X: [N, D] 或 [B, N, D]
        返回同形状但 N' = min(N, max_points)
        """
        if X.dim() == 3:
            B, N, D = X.shape
            n_take = min(N, max_points)
            idx = torch.randperm(N, device=X.device)[:n_take]
            return X[:, idx]  # [B, n_take, D]
        else:
            N, D = X.shape
            n_take = min(N, max_points)
            idx = torch.randperm(N, device=X.device)[:n_take]
            return X[idx]  # [n_take, D]

    def sinkhorn_w2_between(self, *, z1, z2, max_points=4096, epsilon=0.05,
                            add_coords=False, coord_lambda=0.0):
        """
        计算两帧特征的 Sinkhorn W2（对 batch 取平均）。
        z1, z2: [B,C,H,W] 或 [C,H,W]
        返回：标量 float（W2）
        """
        F1 = self._flatten_features(z1, add_coords=add_coords, coord_lambda=coord_lambda) # [1, D]
        F2 = self._flatten_features(z2, add_coords=add_coords, coord_lambda=coord_lambda) # [1, D]

        if F1.dim() == 3:  # batched
            B = F1.shape[0]
            w2_list = []
            for b in range(B):
                X = self._random_subsample(F1[b], max_points)  # [N', D]
                Y = self._random_subsample(F2[b], max_points)  # [N', D]
                Xn = X.detach().cpu().numpy()
                Yn = Y.detach().cpu().numpy()
                Cmat = ot.dist(Xn, Yn, metric="euclidean") ** 2  # [N',N']
                a = np.ones(Xn.shape[0]) / Xn.shape[0]
                b = np.ones(Yn.shape[0]) / Yn.shape[0]
                W2_sq = ot.sinkhorn2(a, b, Cmat, reg=epsilon, numItermax=500)  # 近似 W2^2
                w2_list.append(float(np.sqrt(W2_sq)))
            return float(np.mean(w2_list))
        else:  # single
            X = self._random_subsample(F1, max_points)
            Y = self._random_subsample(F2, max_points)
            Xn = X.detach().cpu().numpy()
            Yn = Y.detach().cpu().numpy()
            Cmat = ot.dist(Xn, Yn, metric="euclidean") ** 2
            a = np.ones(Xn.shape[0]) / Xn.shape[0]
            b = np.ones(Yn.shape[0]) / Yn.shape[0]
            W2_sq = ot.sinkhorn2(a, b, Cmat, reg=epsilon, numItermax=500)
            return float(np.sqrt(W2_sq))

    @torch.no_grad()
    def _to_minus1_1(self, img):
        """
        img: [B,3,H,W]，数值可能在 [-1,1] 或 [0,1]
        统一转为 LPIPS 需要的 [-1,1]
        """
        mn, mx = img.min().item(), img.max().item()
        if mx <= 1.0 and mn >= 0.0:
            img = img * 2.0 - 1.0
        return img.clamp(-1, 1)

    @torch.no_grad()
    def compute_lpips_from_latents(self, z1, z2, vae):
        """
        z1, z2: [B,C,H,W] 的 VAE latent
        vae.decode 期望输入是 latent / 0.18215 （SD 约定）
        返回：标量（对 batch 求平均）
        """
        if _lpips_model is None:
            return None
        dev_img = next(_lpips_model.parameters()).device
        # 解码到图像空间，范围约在 [-1,1]
        x1 = self.vae.decode(z1.to(vae.device) / 0.18215).sample
        x2 = self.vae.decode(z2.to(vae.device) / 0.18215).sample
        x1 = self._to_minus1_1(x1).to(dev_img)
        x2 = self._to_minus1_1(x2).to(dev_img)
        val = _lpips_model(x1, x2)  # [B,1,1,1] 或 [B]
        return float(val.mean().item())

    @torch.no_grad()
    def compute_l2_rms(self, z1, z2):
        """
        z1, z2: [B,C,H,W]
        返回：标量（对 batch 的每个样本 RMS 后再平均）
        """
        diff = (z2 - z1).reshape(z1.shape[0], -1)  # [B, CHW]
        rms = diff.pow(2).mean(dim=1).sqrt()  # [B]
        return float(rms.mean().item())

    @torch.no_grad()
    def compute_swd(self, z1, z2, num_projections=128, eps=1e-8):
        """
        Sliced-Wasserstein-2（对 batch 取平均）：
        - 把每个空间位置的 C 维向量当作一个点（忽略坐标）
        - 随机投影到 1D 后，排序并计算两分布在 1D 上的 W2（等价于排序差的 L2）
        z1, z2: [B,C,H,W]
        返回：标量
        """
        B, C, H, W = z1.shape
        X = z1.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, N, C]
        Y = z2.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, N, C]
        # 为了稳定，中心化（可选）
        Xc = X - X.mean(dim=1, keepdim=True)
        Yc = Y - Y.mean(dim=1, keepdim=True)

        swd_list = []
        for _ in range(num_projections):
            # 生成单位随机方向 u: [B, C]（也可用共享方向，差别不大）
            u = torch.randn(B, C, device=X.device)
            u = u / (u.norm(dim=1, keepdim=True) + eps)  # 归一化
            # 投影到 1D: [B, N]
            x1d = (Xc * u.unsqueeze(1)).sum(dim=2)
            y1d = (Yc * u.unsqueeze(1)).sum(dim=2)
            # 排序后做 L2
            x_sorted, _ = torch.sort(x1d, dim=1)
            y_sorted, _ = torch.sort(y1d, dim=1)
            w2_1d = (x_sorted - y_sorted).pow(2).mean(dim=1).sqrt()  # [B]
            swd_list.append(w2_1d)  # 收集 [B]
        swd_all = torch.stack(swd_list, dim=0).mean(dim=0)  # [B]，对投影平均
        return float(swd_all.mean().item())  # 对 batch 平均

    @torch.no_grad()
    def cal_Wass(self, batch_idx, batch=None, pbar=None, result_path=None):
        """Generate a batch of samples"""
        # Generate random initial noise
        x0 = batch['x0']
        # Create time steps
        t_span = torch.linspace(0, 1, self.args.num_steps, device=self.device)

        # Select solver and generate samples
        solver = self.euler_solver if self.args.solver == 'euler' else self.heun_solver
        samples, trajectory = solver(x0, t_span, batch_idx, batch, pbar)
        results = {}
        results['samples'] = samples
        results['trajectory'] = trajectory
        return results

    def generate(self):
        """Main generation loop"""
        print(self.args.output_dir)
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_A = transforms.Compose([
            transforms.Resize((512, 512)), 
            transforms.RandomHorizontalFlip(p=0.25), 
            transforms.RandomVerticalFlip(p=0.25),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        transform_grey = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.RandomGrayscale(p=1),  # 数据增强：50% 概率灰度化
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        datasets = []
        json_paths = glob.glob(
            "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v1/2004-2010_data_pairs_dia/*.json")
        for json_path in tqdm(json_paths):
            dataset = MedicalJsonDataset(
                path=json_path,
                transform=transform,
                hint_transform=transform_grey,
                transform_A=transform,
                transform_B=transform,
            )
            datasets.append(dataset)
            # break
        for json_path in tqdm(json_paths):
            dataset = MedicalJsonDataset(
                path=json_path,
                transform=transform,
                hint_transform=transform_grey,
                transform_A=transform_A,
                transform_B=transform,
            )
            datasets.append(dataset)
            # break
            
        op_match_batch = False
        sample_posterior = True
        new_dataset = []
        caption_hidden_states_mode = 'cat'
        sigma = 0.0
        from conditional_flow_matcher import ConditionalFlowMatcher, OptimalTransportConditionalFlowMatcher
        FM = OptimalTransportConditionalFlowMatcher(sigma=sigma, ot_method='exact')
        text_embeds_map = {}
        batch_size = 1
        total = sum([len(dataset) for dataset in datasets])
        current = 0
        for dataset in datasets:
            for idx in tqdm(range(0, len(dataset), batch_size)):
                current += 1
                if idx > 200:
                    break
                
                batch = deepcopy(dataset[idx])
                if idx % 5 == 0:
                    try:
                        print(f"{batch['label_A']}:{batch['label_A_id']} --> {batch['label_B']}:{batch['label_B_id']}")
                        print(f'---------------------------------{current}/{total}----------------------------------')
                    except:
                        pass
                              
                x0 = batch['x0'].to(self.vae.device).unsqueeze(0)
                x1 = batch['x1'].to(self.vae.device).unsqueeze(0)
                caption = [batch['caption']]
                y = batch['class_id'].to(self.vae.device).unsqueeze(0)
                x1_path = [batch['x1_path']]
                hint = batch['hint'].to(self.vae.device).unsqueeze(0)

                images = torch.cat([x0, x1], dim=0)
                with torch.no_grad():
                    posterior = self.vae.encode(images).latent_dist
                    images = posterior.sample() * 0.18215
                    x0, x1 = images.chunk(2, dim=0)
                        
                if self.args.solver == 'euler':
                    self.text_model = None
                    self.vision_model = None

                if self.text_model is not None and caption[0] not in text_embeds_map:
                    with torch.no_grad():
                        caption_input = self.text_tokenizer(caption, return_tensors="pt", padding=True).to(
                            self.text_model.device)
                        caption_outputs = self.text_model(**caption_input)
                        text_embeds = caption_outputs['last_hidden_state'].to(self.text_model.device)
                    if len(text_embeds_map) < 200:
                        text_embeds_map[caption[0]] = text_embeds
                elif caption[0] in text_embeds_map:
                    text_embeds = text_embeds_map[caption[0]]
                else:
                    text_embeds = torch.zeros((len(y), 2, 1))
                if self.vision_model is not None:
                    with torch.no_grad():
                        x1_images = torch.stack([process_single_image(image_path) for image_path in x1_path])
                        vision_embeds = self.vision_model.forward_features(x1_images.to(self.vision_model.device))
                else:
                    vision_embeds = torch.zeros((len(y), 2, 1))

                if caption_hidden_states_mode == 'cat':
                    caption_hidden_states = torch.cat([vision_embeds[:, 1:, ...].to(self.net_model.device),
                                                    text_embeds[:, 1:, ...].to(self.net_model.device)], dim=1)
                elif caption_hidden_states_mode == 'only_text':
                    caption_hidden_states = text_embeds[:, 1:, ...].to(self.net_model.device)

                conds = {
                    'x0': x0,
                    'x1': x1,
                    'caption': caption,
                    'caption_hidden_states': caption_hidden_states.to(self.device),
                    'y': y.to(self.device),
                    'hint': hint.to(self.device),
                    'text_embeds': text_embeds[:, 0].to(self.device),  # [B, D]
                    'image_embeds': vision_embeds[:, 0].to(self.device)  # [B, D]
                }

                t, xt, ut = FM.get_sample_location_and_conditional_flow(x0, x1,
                                                                        sample_plan=op_match_batch, # 只有一个
                                                                        cond=conds,
                                                                        print_info=False)
                self.args.num_steps = random.choice([8, 10, 12, 14, 16, 18, 20, 22, 24])
                # self.args.num_steps = random.choice([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48])
                conds['ut'] = ut.to(self.device)
                results = self.cal_Wass(idx, conds)
                trajectory = results['trajectory']
                step = 1
                score = -1
                label_A = batch['label_A']
                label_B = batch['label_B']
                output_dir = self.args.output_dir
                image_dir = os.path.join(output_dir, f"{label_A}2{label_B}")
                os.makedirs(image_dir, exist_ok=True)
                x0_path = batch['x0_path']
                x1_path = batch['x1_path']
                x0_name = x0_path.split('/')[-1]
                x1_name = x1_path.split('/')[-1]
                new_x0_path = os.path.join(image_dir, x0_name)
                new_x1_path = os.path.join(image_dir, x1_name)
                new_x0 = Image.open(x0_path).convert('RGB')
                new_x1 = Image.open(x1_path).convert('RGB')
                new_x0.save(new_x0_path)
                new_x1.save(new_x1_path)
                x0_name2 = x0_name[:-4]
                for x in trajectory[1:-1]:
                    x = self.vae.decode(x / 0.18215).sample
                    sample = self.normalize_samples(x)
                    save_image(sample, "/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/discriminator/tmp12.jpg") # 临时保存
                    img = Image.open("/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/discriminator/tmp12.jpg").convert("RGB")
                    score = float(similarity(img, new_x1).item())
                    if True:
                        score2 = float(similarity(img, new_x0).item())
                        img_path = os.path.join(image_dir, f'step{step}_{score:.4f}-{score2:.4f}_{x0_name2}#to#{x1_name}')
                    else:
                        img_path = os.path.join(image_dir, f'step{step}_{score:.4f}_{x0_name2}#to#{x1_name}')
                    img.save(img_path)
                    step += 1

def main():
    args = parse_args()
    generator = ImageGenerator(args)
    for _ in range(25):
        generator.generate()

def tensor_to_pil(tensor):
    tensor = tensor.cpu()  # 移到 CPU
    array = tensor.detach().numpy()  # 转为 numpy

    # 如果是 (C, H, W)，转为 (H, W, C)
    if array.ndim == 3 and array.shape[0] in [1, 3]:
        array = np.transpose(array, (1, 2, 0))

    # 反归一化（如果值在 [-1, 1] 范围）
    if array.max() <= 1.0 and array.min() >= -1.0:
        array = (array + 1.0) * 127.5  # [-1,1] -> [0,255]
    else:
        array = array * 255.0  # [0,1] -> [0,255]

    array = array.astype(np.uint8)  # 确保数据类型正确
    return Image.fromarray(array)

# ### llm env
# from gme_inference import GmeQwen2VL
# gme = GmeQwen2VL("/mnt/inaisfs/data/home/tansy_criait/weights/gme-Qwen2-VL-7B-Instruct", device="cuda:1")

# def similarity(image1: str, image2: str):
#     e_image = gme.get_image_embeddings(images=[image1, image2])
#     e_image = F.normalize(e_image, p=2, dim=1)
#     return e_image[0] @ e_image[1].T

### flow env
from transformers import AutoImageProcessor, DINOv3ConvNextModel, DINOv3ViTModel
from transformers.image_utils import load_image
pretrained_model_name = "/mnt/inaisfs/data/home/tansy_criait/weights/dinov3-vit7b16"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = DINOv3ViTModel.from_pretrained(
    pretrained_model_name, 
    device_map="auto", 
    max_memory={1:"40GiB"} 
)
def similarity(image1: str, image2: str):
    if isinstance(image1, str):
        image1 = load_image(image1)
    if isinstance(image2, str):
        image2 = load_image(image2)
    inputs = processor(images=[image1, image2], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
        embeds = outputs.pooler_output
    embeds = F.normalize(embeds, p=2, dim=1)  # dim=1 表示对每个向量做归一化
    return embeds[0] @ embeds[1].T

# ### vllm_serve
# from qwen3_vl_embedding import Qwen3VLEmbedder
# # 加载模型（仅视觉部分，不加载语言头以节省显存）
# model_name = "/mnt/inaisfs/data/home/tansy_criait/weights/Qwen3-VL-Embedding-8B"

# # model = Qwen3VLEmbedder(model_name_or_path=model_name, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
# model = Qwen3VLEmbedder(model_name_or_path=model_name, torch_dtype=torch.float)

# def get_qwen2vl_image_embedding(image_path: str):
#     # Qwen2-VL 需要构造一个 dummy prompt 来触发视觉编码
#     messages = [{"image": image_path}]
    
#     with torch.inference_mode():
#         embeddings = model.process(messages)

#     return embeddings

# def similarity(image1: str, image2: str):
#     emb1 = get_qwen2vl_image_embedding(image1)
#     emb2 = get_qwen2vl_image_embedding(image2)
#     return emb1 @ emb2.T

if __name__ == "__main__":
    main()

