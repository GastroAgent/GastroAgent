from copy import deepcopy
import os
from timm.models.vision_transformer import VisionTransformer
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append('/mnt/inaisfs/data/home/tansy_criait/GasAgent-main')
from collections import OrderedDict
from utils.data_loader import MedicalDataset, MedicalJsonDataset
from utils.data_utils import *
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
import ot
from my_models.unet_2d_condition import UNet2DConditionModel
from my_models.model_dispatch import dispatch_model
from diffusers import AutoencoderKL
from transformers import ChineseCLIPConfig as CLIPConfig
from transformers import ChineseCLIPProcessor as CLIPProcessor
from transformers import ChineseCLIPModel as CLIPModel
from transformers import AutoTokenizer, AutoModel, AutoConfig, ChineseCLIPTextModel, ChineseCLIPTextConfig
from conditional_flow_matcher import ConditionalFlowMatcher, OptimalTransportConditionalFlowMatcher

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

from geomloss import SamplesLoss
# loss_fn = SamplesLoss("laplacian", p=2, blur=0.1**0.5)
loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.1**0.5)
# sinkhorn hausdorff energy gaussian laplacian
def sinkhorn_loss(bx, bx1, epsilon=0.1, n_iter=50, reduction='mean',  **kwargs):
    """
    Compute Sinkhorn loss (approximate Wasserstein distance) between two sets of samples.
    
    Args:
        bx (Tensor): [B, N, D] or [B, C, H, W] or [B, D]
        bx1 (Tensor): [B, N, D] or [B, C, H, W] or [B, D]
        epsilon (float): Entropy regularization strength
        n_iter (int): Number of Sinkhorn iterations
        reduction (str): 'mean' or 'sum' for batch reduction

    Returns:
        Tensor: Scalar loss
    """
    if bx.ndim == 2:
        B, D = bx.shape
        H = W = int((D // 4) ** 0.5)
        bx = bx.view(B, 4, H, W).permute(0, 2, 3, 1).reshape(B, H*W, -1)
    elif bx.ndim == 4:
        B, C, H, W = bx.shape
        bx = bx.permute(0, 2, 3, 1).reshape(B, H*W, C)
    elif bx.ndim == 3:
        bx = bx.unsqueeze(0)
        B, C, H, W = bx.shape
        bx = bx.permute(0, 2, 3, 1).reshape(B, H*W, C)

    if bx1.ndim == 2:
        B, D = bx1.shape
        H = W = int((D // 4) ** 0.5)
        bx1 = bx1.view(B, 4, H, W).permute(0, 2, 3, 1).reshape(B, H*W, -1)
    elif bx1.ndim == 4:
        B, C, H, W = bx1.shape
        bx1 = bx1.permute(0, 2, 3, 1).reshape(B, H*W, C)
    elif bx1.ndim == 3:
        bx1 = bx1.unsqueeze(0)
        B, C, H, W = bx1.shape
        bx1 = bx1.permute(0, 2, 3, 1).reshape(B, H*W, C)
    loss = loss_fn(bx, bx1)
    return loss

@torch.no_grad()
def cal_wasserstein_loss(x, x1, **kwargs):
    wass_loss = sinkhorn_loss(x, x1, **kwargs)
    # wass_loss = wass_loss.sum()
    return wass_loss

# @torch.no_grad()
# def sinkhorn_loss(bx, bx1, epsilon=0.1, n_iter=25, reduction='mean', **kwargs):
#     """
#     Compute Sinkhorn loss (approximate Wasserstein distance) between two sets of samples.
    
#     Args:
#         bx (Tensor): [B, N, D] or [B, C, H, W] or [B, D]
#         bx1 (Tensor): [B, N, D] or [B, C, H, W] or [B, D]
#         epsilon (float): Entropy regularization strength
#         n_iter (int): Number of Sinkhorn iterations
#         reduction (str): 'mean' or 'sum' for batch reduction

#     Returns:
#         Tensor: Scalar loss
#     """
#     # Reshape input to [B, N, D]
#     # Reshape input to [B, N, D]
#     if bx.ndim == 2:
#         B, D = bx.shape
#         H = W = int((D // 4) ** 0.5)
#         bx = bx.view(B, 4, H, W).permute(0, 2, 3, 1).reshape(B, H*W, -1)
#     elif bx.ndim == 4:
#         B, C, H, W = bx.shape
#         bx = bx.permute(0, 2, 3, 1).reshape(B, H*W, C)
#     elif bx.ndim == 3:
#         bx = bx.unsqueeze(0)
#         B, C, H, W = bx.shape
#         bx = bx.permute(0, 2, 3, 1).reshape(B, H*W, C)

#     if bx1.ndim == 2:
#         B, D = bx1.shape
#         H = W = int((D // 4) ** 0.5)
#         bx1 = bx1.view(B, 4, H, W).permute(0, 2, 3, 1).reshape(B, H*W, -1)
#     elif bx1.ndim == 4:
#         B, C, H, W = bx1.shape
#         bx1 = bx1.permute(0, 2, 3, 1).reshape(B, H*W, C)
#     elif bx1.ndim == 3:
#         bx1 = bx1.unsqueeze(0)
#         B, C, H, W = bx1.shape
#         bx1 = bx1.permute(0, 2, 3, 1).reshape(B, H*W, C)

#     B, N, D = bx.shape
#     _, M, _ = bx1.shape

#     # Compute cost matrix: [B, N, M]
#     x2 = torch.sum(bx**2, dim=-1, keepdim=True)                    # [B, N, 1]
#     y2 = torch.sum(bx1**2, dim=-1, keepdim=True)                   # [B, M, 1]
#     cross = torch.bmm(bx, bx1.transpose(-1, -2))                   # [B, N, M]
#     cost_matrix = x2 - 2 * cross + y2.transpose(-1, -2)            # [B, N, M]
#     cost_matrix = torch.clamp(cost_matrix, min=0.0)

#     # Kernel matrix
#     K = torch.exp(-cost_matrix / epsilon)                          # [B, N, M]

#     # Uniform marginal distributions
#     a = torch.ones(B, N, device=bx.device) / N                     # [B, N]
#     b = torch.ones(B, M, device=bx.device) / M                     # [B, M]

#     # Initialize dual variables
#     u = torch.ones_like(a)                                         # [B, N]
#     v = torch.ones_like(b)                                         # [B, M]

#     # Sinkhorn iterations
#     for _ in range(n_iter):
#         u = a / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)  # [B, N]
#         v = b / (torch.bmm(K.transpose(-1,-2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)  # [B, M]

#     # Compute transport plan
#     P = u.unsqueeze(-1) * K * v.unsqueeze(-2)  # [B, N, M]

#     # Compute loss
#     loss = torch.sum(P * cost_matrix, dim=(1,2))  # [B]

#     if reduction == 'mean':
#         return loss.mean()
#     elif reduction == 'sum':
#         return loss.sum()
#     else:
#         return loss

# @torch.no_grad()
# def cal_wasserstein_loss(x, x1, **kwargs):
#     wass_loss = sinkhorn_loss(x, x1, **kwargs)
#     # wass_loss = wass_loss.sum()
#     return float(wass_loss.cpu().item())

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
        self.text_embeds_map = {}
        self.FM = OptimalTransportConditionalFlowMatcher(sigma=0, ot_method='exact')
        
    def _setup_directories(self):
        """Create necessary directories for outputs"""
        os.makedirs(self.args.output_dir, exist_ok=True)
        if self.args.save_intermediates:
            self.intermediates_dir = os.path.join(self.args.output_dir, 'intermediates')
            os.makedirs(self.intermediates_dir, exist_ok=True)

    def _load_model(self, checkpoint: str, use_ema=False):
        """Initialize and load the model"""
        use_gt_vt = True
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
                device='cuda').eval()
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

            net_model = net_model.to(device='cuda')
            vae = vae.to('cuda')
            if text_model is not None:
                text_model = text_model.to('cuda')
            if vision_model is not None:
                vision_model = vision_model.to('cuda')
                vision_model.device = 'cuda'

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
                self.wass_model = model.eval()
                self.wass_model.eval()
                
            elif 'wass_model' in state_dict:
                from train_vae_with_wass import TripletNetwork
                if self.args.wass_model_type == '':
                    self.args.wass_model_type = 'resnet34'
                model = TripletNetwork(pretrained=True, freeze_base=False, model=self.args.wass_model_type)
                model.load_state_dict(state_dict['wass_model'], strict=False)
                model = model.to("cuda")
                self.wass_model = model.eval()
                self.wass_model.eval()
            else:   
                self.wass_model = None
            
            if self.args.sim_model_type == 'gme':
                from gme_inference import GmeQwen2VL
                model = GmeQwen2VL(self.args.sim_model_path, device="cuda:1")

                def similarity(image1: str, image2: str, step=0):
                    if step == 0:
                        return 0
                    e_image = model.get_image_embeddings(images=[image1, image2])
                    e_image = F.normalize(e_image, p=2, dim=1)
                    return e_image[0] @ e_image[1].T
                self.similarity = similarity
            elif self.args.sim_model_type == 'dinov3':
                # from modelscope import AutoImageProcessor, AutoModel
                # from transformers.image_utils import load_image
                # pretrained_model_name = self.args.sim_model_path
                # processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
                # model = AutoModel.from_pretrained(
                #     pretrained_model_name, 
                #     device_map="auto", 
                #     max_memory={1:"40GiB"} 
                # )
                # def similarity(image1: str, image2: str, step=0):
                #     if step == 0:
                #         return 0
                #     if isinstance(image1, str):
                #         image1 = load_image(image1)
                #     if isinstance(image2, str):
                #         image2 = load_image(image2)
                #     inputs = processor(images=[image1, image2], return_tensors="pt").to(model.device)
                #     with torch.inference_mode():
                #         outputs = model(**inputs)
                #         embeds = outputs.pooler_output
                #     embeds = F.normalize(embeds, p=2, dim=1)  # dim=1 表示对每个向量做归一化
                #     return embeds[0] @ embeds[1].T
                self.similarity = None
            elif self.args.sim_model_type == 'convnext':
                import sys
                sys.path.append("/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/discriminator")
                sys.path.append('/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match')
                sys.path.append('/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/utils')
                from discriminator.train_latent_space import TripletNetwork
                model = TripletNetwork(pretrained=True, freeze_base=False, model='convnext')
                try:
                    state_dict = torch.load("/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/discriminator/latent_model_weight/convnext.pt", weights_only=True)
                    model.load_state_dict(state_dict, strict=False)
                except:
                    pass
                model = model.to("cuda")
                def similarity(image1, image2, step=0):
                    if step == 0:
                        return 0
                    with torch.inference_mode():
                        outputs = model.encode(torch.cat([image1, image2], dim=0).to(model.device), False)
                        embeds = outputs.pooler_output
                    embeds = F.normalize(embeds, p=2, dim=1)  # dim=1 表示对每个向量做归一化
                    return embeds[0] @ embeds[1].T
                self.similarity = similarity
            return net_model, vae, (text_model, text_tokenizer), (vision_model, process_single_image)

        else:
            net_model = None
            text_model = None
            text_tokenizer = None
            vision_model = None
            process_single_image = None
            vae = AutoencoderKL.from_pretrained(
                '/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/flow_matcher_otcfm/vae').to(
                device='cuda').eval()
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
        x_min = x.amin(dim=(1, 2, 3), keepdim=True)  # shape: (B, 1, 1, 1)
        x_max = x.amax(dim=(1, 2, 3), keepdim=True)  # shape: (B, 1, 1, 1)

        # 防止分母为0
        scale = (x_max - x_min).clamp(min=1e-4)

        # 执行缩放
        x_scaled = (x - x_min) / scale
        return x_scaled

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
        x1 = batch['x1']
        x0 = batch['x0']
        caption_hidden_states = batch['caption_hidden_states']
        y = batch['y']

        # 用来收集每个 time step 的 latent
        trajectory = [x0.detach().clone()]
        ycache = OrderedDict()
        vcache = OrderedDict()
        imgcache = OrderedDict()
        use_gt_vt = self.args.use_gt_vt
        max_t = self.args.min_num_steps if self.args.fix_num_steps < 0 else self.args.fix_num_steps
        temp = batch['x0_path'][0]
        step = len(trajectory) - 1
        
        def continue_strategy(image1: str, image2: str, step: int, method='direct', *args, **kwargs):
            if step == 0:
                return True
            
            if method == 'direct':
                return self.similarity(temp, batch['x1_path'][0], step) < self.args.stop_beta 
            elif method == 'diff':
                current_sim = self.similarity(temp, batch['x1_path'][0], step)
                diff_sim = current_sim - self.last_sim
                self.last_sim = current_sim
                return abs(diff_sim) > self.args.stop_beta 
            elif method == 'second_diff':
                current_sim = self.similarity(temp, batch['x1_path'][0], step)
                current_diff_sim = current_sim - self.last_sim
                self.last_sim = current_sim
                second_diff_sim = current_diff_sim - self.last_diff_sim
                self.last_diff_sim = current_diff_sim
                return abs(second_diff_sim) > self.args.stop_beta 
            else:
                return self.similarity(temp, batch['x1_path'][0], step) < self.args.stop_beta 
            
        while continue_strategy(temp, batch['x1_path'][0], step, method = self.args.stop_method) and max_t <= self.args.num_steps and (self.args.fix_num_steps < 0 or step == 0): 
            x = x0
            step += 1
            dt = t_span[1] - t_span[0]
            for t_idx in range(len(t_span) - 1):
                t = t_span[t_idx] * torch.ones(x0.shape[0], device=self.device)
                key_t = str(float(t[0].item()))[:4]
                next_t = t_span[t_idx + 1] * torch.ones(x0.shape[0], device=self.device)
                next_key_t = str(float(next_t[0].item()))[:4]
                # if next_key_t in ycache:
                #     x = ycache[next_key_t]
                # else:
                if True:
                    if key_t in vcache:
                        dx = vcache[key_t]
                        ycache[next_key_t] = x.clone()
                    else:
                        if use_gt_vt:
                            ut = batch['ut']
                            dx = ut
                        else:
                            dx = self.net_model(x, timestep=t.squeeze(), class_labels=y,
                                                encoder_hidden_states=caption_hidden_states,
                                                added_cond_kwargs=batch).sample
                        if self.args.use_cache or key_t == '0.0':
                            vcache[key_t] = dx.clone()
                    
                    x = x + dx * dt
                    ycache[next_key_t] = x.clone()

                    # ### 查看临时的 Temp 图像。
                    # if (str(step) + '_' +next_key_t) in imgcache:
                    #     temp = imgcache[(str(step) + '_' +next_key_t)]
                    # else:
                    #     img = self.vae.decode(x.to(self.vae.device) / 0.18215).sample
                    #     img = self.normalize_samples(img)
                    #     os.makedirs(f"{self.args.temp}/{batch_idx}", exist_ok=True)
                    #     imgcache[(str(step) + '_' +next_key_t)] = temp = f"{self.args.temp}/{batch_idx}/{(str(step) + '_' +next_key_t)}.jpg"
                    #     save_image(img, temp)  

            max_t *= 2 
            t_span = torch.linspace(0, 1, max_t + 1, device=self.device)
            
            if step in imgcache:
                temp = imgcache[step]
            else:
                img = self.vae.decode(x.to(self.vae.device) / 0.18215).sample
                img = self.normalize_samples(img)
                os.makedirs(f"{self.args.temp}/{batch_idx}", exist_ok=True)
                imgcache[step] = temp = f"{self.args.temp}/{batch_idx}/{step}.jpg"
                save_image(img, temp)

        for t_idx, (_, xt) in enumerate(sorted(ycache.items())): 
            if (t_idx + 1) % self.args.intermediate_freq == 0:
                trajectory.append(xt.detach().clone())
        trajectory.append(x1.detach().clone())
        return x, trajectory

    def heun_solver(self, x0, t_span, batch_idx, batch, pbar=None):
        """Heun's solver that saves and returns the full latent trajectory."""
        x1 = batch['x1']
        x0 = batch['x0']
        caption_hidden_states = batch['caption_hidden_states']
        y = batch['y']

        # 用来收集每个 time step 的 latent
        trajectory = [x0.detach().clone()]
        ycache = OrderedDict()
        vcache = OrderedDict()
        imgcache = OrderedDict()
        max_t = self.args.min_num_steps if self.args.fix_num_steps < 0 else self.args.fix_num_steps
        temp = batch['x0_path'][0]
        self.last_sim = 0
        self.last_diff_sim = 0
        step = len(trajectory) - 1

        def continue_strategy(image1: str, image2: str, step: int, method='direct', *args, **kwargs):
            if step == 0:
                return True
            
            if method == 'direct':
                return self.similarity(temp, batch['x1_path'][0], step) < self.args.stop_beta 
            elif method == 'diff':
                current_sim = self.similarity(temp, batch['x1_path'][0], step)
                diff_sim = current_sim - self.last_sim
                self.last_sim = current_sim
                return abs(diff_sim) > self.args.stop_beta 
            elif method == 'second_diff':
                current_sim = self.similarity(temp, batch['x1_path'][0], step)
                current_diff_sim = current_sim - self.last_sim
                self.last_sim = current_sim
                second_diff_sim = current_diff_sim - self.last_diff_sim
                self.last_diff_sim = current_diff_sim
                return abs(second_diff_sim) > self.args.stop_beta 
            else:
                return self.similarity(temp, batch['x1_path'][0], step) < self.args.stop_beta 
            
        while continue_strategy(temp, batch['x1_path'][0], step, method = self.args.stop_method) and max_t <= self.args.num_steps and (self.args.fix_num_steps < 0 or step == 0):
            x = x0
            step += 1
            print(f"Current Need Iteration is {max_t}")
            dt = t_span[1] - t_span[0]
            for t_idx in range(len(t_span) - 1):
                t = t_span[t_idx] * torch.ones(x0.shape[0], device=self.device)
                key_t = str(float(t[0].item()))[:4]
                next_t = t_next = t_span[t_idx + 1] * torch.ones(x0.shape[0], device=self.device)

                next_key_t = str(float(next_t[0].item()))[:4]
                # if next_key_t in ycache:
                #     x = ycache[next_key_t]
                # else:
                if True:
                    if key_t in vcache:
                        dx = vcache[key_t]
                        x = x + dx * dt
                        ycache[next_key_t] = x.clone()
                    else:
                        dx1 = self.net_model(x, timestep=t.squeeze(), class_labels=y,
                                            encoder_hidden_states=caption_hidden_states, added_cond_kwargs=batch).sample
                        x_euler = x + dx1 * dt
                        dx2 = self.net_model(x_euler, timestep=t_next.squeeze(), class_labels=y,
                                 encoder_hidden_states=caption_hidden_states, added_cond_kwargs=batch).sample
                        if self.args.use_cache or key_t == '0.0':
                            vcache[key_t] = (dx1 + dx2).clone() / 2
                    
                        x = x + (dx1 + dx2) * dt / 2
                        ycache[next_key_t] = x.clone()

                    # ### 查看临时的 Temp 图像。
                    # if (str(step) + '_' +next_key_t) in imgcache:
                    #     temp = imgcache[(str(step) + '_' +next_key_t)]
                    # else:
                    #     img = self.vae.decode(x.to(self.vae.device) / 0.18215).sample
                    #     img = self.normalize_samples(img)
                    #     os.makedirs(f"{self.args.temp}/{batch_idx}", exist_ok=True)
                    #     imgcache[(str(step) + '_' +next_key_t)] = temp = f"{self.args.temp}/{batch_idx}/{(str(step) + '_' +next_key_t)}.jpg"
                    #     save_image(img, temp)  
            max_t *= 2
            t_span = torch.linspace(0, 1, max_t + 1, device=self.device)
            
            if step in imgcache:
                temp = imgcache[step]
            else:
                img = self.vae.decode(x.to(self.vae.device) / 0.18215).sample
                img = self.normalize_samples(img)
                os.makedirs(f"{self.args.temp}/{batch_idx}", exist_ok=True)
                imgcache[step] = temp = f"{self.args.temp}/{batch_idx}/{step}.jpg"
                save_image(img, temp)
        
        for t_idx, (_, xt) in enumerate(sorted(ycache.items())): 
            if (t_idx + 1) % self.args.intermediate_freq == 0:
                trajectory.append(xt.detach().clone())
        trajectory.append(x1.detach().clone())
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
        F1 = self._flatten_features(z1, add_coords=add_coords, coord_lambda=coord_lambda) # [H * W, C]
        F2 = self._flatten_features(z2, add_coords=add_coords, coord_lambda=coord_lambda) # [H * W, C]

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
                W2_sq = ot.sinkhorn2(a, b, Cmat, reg=epsilon, numItermax=200)  # 近似 W2^2
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
            W2_sq = ot.sinkhorn2(a, b, Cmat, reg=epsilon, numItermax=200)
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
    def cal_Wass_with_T(self, batch_idx, batch=None, pbar=None, result_path=None):
        """Generate a batch of samples"""
        # Generate random initial noise
        x0 = batch['x0']
        # Create time steps
        max_t = self.args.min_num_steps if self.args.fix_num_steps < 0 else self.args.fix_num_steps
        t_span = torch.linspace(0, 1, max_t + 1, device=self.device)

        # Select solver and generate samples
        solver = self.euler_solver if self.args.solver == 'euler' else self.heun_solver
        samples, trajectory = solver(x0, t_span, batch_idx, batch, pbar)
        print('--------------------- 计算 Wass 距离 --------------------')
        # ====== 在你的生成代码中启用（替换原“计算 W₂”的注释块） ======
        distances_emd_w2 = []
        distances_sinkhorn_w2_latent = []
        distances_sinkhorn_w2_latent_mapped = []
        distances_sinkhorn_w2_latent_mapped_bias1 = []
        distances_sinkhorn_w2_latent_mapped_bias2 = []
        distances_sinkhorn_w2_image = []
        Neighbor_distances_sinkhorn_w2_latent = []
        Neighbor_distances_sinkhorn_w2_image = []
        Neighbor_distances_sinkhorn_w2_latent_mapped = []
        Neighbor_distances_sinkhorn_w2_latent_mapped_bias1 = []
        Neighbor_distances_sinkhorn_w2_latent_mapped_bias2 = []
        
        dist_l2 = []
        dist_lpips = []
        dist_swd = []
        full = self.args.full
        max_points = self.args.max_points
        freq = 1
        indices = list(range(0, len(trajectory), freq))
        if indices[-1] != len(trajectory) - 1:
            indices.append(len(trajectory) - 1)

        if not full:
            z_last = trajectory[-1]
            wass_cache = {len(indices) - 1: self.wass_model.encode(z_last, True).squeeze(0)}
            for idx in range(len(indices) - 1):
                i = indices[idx]
                j = indices[idx + 1]
                z1 = trajectory[i]  # [1,C,H,W]
                z2 = trajectory[j] 
                
                if self.wass_model is not None:
                    if i in wass_cache:
                        tz1 = wass_cache[i]
                    else:
                        tz1 = self.wass_model.encode(z1, True).squeeze(0)
                        wass_cache[i] = tz1
                        
                    W2_sinkhorn_latent_mapped = cal_wasserstein_loss(
                        tz1,
                        wass_cache[len(indices) - 1],
                        max_points=max_points,  # 控制规模（可调：1024~8192）
                        epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                        add_coords=False,  # 是否拼接空间坐标
                        coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                    ).item()
                    distances_sinkhorn_w2_latent_mapped.append(W2_sinkhorn_latent_mapped)
                    if self.args.bias:
                        W2_sinkhorn_latent_mapped_b1 = cal_wasserstein_loss(
                            tz1,
                            tz1,
                            max_points=max_points,  # 控制规模（可调：1024~8192）
                            epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                            add_coords=False,  # 是否拼接空间坐标
                            coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                        ).item()
                        distances_sinkhorn_w2_latent_mapped_bias1.append(W2_sinkhorn_latent_mapped_b1)
                        if len(distances_sinkhorn_w2_latent_mapped_bias2) == 0:
                            W2_sinkhorn_latent_mapped_b2 = cal_wasserstein_loss(
                                wass_cache[len(indices) - 1],
                                wass_cache[len(indices) - 1],
                                max_points=max_points,  # 控制规模（可调：1024~8192）
                                epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                                add_coords=False,  # 是否拼接空间坐标
                                coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                            ).item()
                            distances_sinkhorn_w2_latent_mapped_bias2.append(W2_sinkhorn_latent_mapped_b2)

                ####################################### Neighbor #################################
                if self.wass_model is not None:
                    if i in wass_cache:
                        tz1 = wass_cache[i]
                    else:
                        tz1 = self.wass_model.encode(z1, True).squeeze(0)
                        wass_cache[i] = tz1
                    if j in wass_cache:
                        tz2 = wass_cache[j]
                    else:
                        tz2 = self.wass_model.encode(z2, True).squeeze(0)
                        wass_cache[j] = tz2
                        
                    W2_sinkhorn_latent_mapped = cal_wasserstein_loss(
                        tz1,
                        tz2,
                        max_points=max_points,  # 控制规模（可调：1024~8192）
                        epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                        add_coords=False,  # 是否拼接空间坐标
                        coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                    ).item()
                    Neighbor_distances_sinkhorn_w2_latent_mapped.append(W2_sinkhorn_latent_mapped)
                    if self.args.bias:
                        W2_sinkhorn_latent_mapped_b1 = cal_wasserstein_loss(
                            tz1,
                            tz1,
                            max_points=max_points,  # 控制规模（可调：1024~8192）
                            epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                            add_coords=False,  # 是否拼接空间坐标
                            coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                        ).item()
                        Neighbor_distances_sinkhorn_w2_latent_mapped_bias1.append(W2_sinkhorn_latent_mapped_b1)
                        W2_sinkhorn_latent_mapped_b2 = cal_wasserstein_loss(
                            tz2,
                            tz2,
                            max_points=max_points,  # 控制规模（可调：1024~8192）
                            epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                            add_coords=False,  # 是否拼接空间坐标
                            coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                        ).item()
                        Neighbor_distances_sinkhorn_w2_latent_mapped_bias2.append(W2_sinkhorn_latent_mapped_b2)

            results = {}
        else:
            z_last = trajectory[-1]
            x_last = self.vae.decode(z_last.to(self.vae.device) / 0.18215).sample
            x_last = self.normalize_samples(x_last)
            x_last = F.interpolate(
                x_last,
                size=(64, 64),
                mode='bilinear',
                align_corners=False
            )
            if self.wass_model is not None:
                wass_cache = {len(indices) - 1: self.wass_model.encode(z_last, True).squeeze(0)}
            x_cache = {len(indices) - 1: x_last}
            for idx in range(len(indices) - 1):
                i = indices[idx]
                j = indices[idx + 1]
                z1 = trajectory[i]  # [1,C,H,W]
                z2 = trajectory[j] 
                
                W2_sinkhorn_latent = cal_wasserstein_loss(
                    z1.squeeze(0),
                    z_last.squeeze(0),
                    max_points=max_points,  # 控制规模（可调：1024~8192）
                    epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                    add_coords=False,  # 是否拼接空间坐标
                    coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                ).item()
                distances_sinkhorn_w2_latent.append(W2_sinkhorn_latent)
                # print(f"step {i} → {j}: W₂_sinkhorn_latent = {W2_sinkhorn_latent:.6f}")
                
                if self.wass_model is not None:
                    if i in wass_cache:
                        tz1 = wass_cache[i]
                    else:
                        tz1 = self.wass_model.encode(z1, True).squeeze(0)
                        wass_cache[i] = tz1
                        
                    W2_sinkhorn_latent_mapped = cal_wasserstein_loss(
                        tz1,
                        wass_cache[len(indices) - 1],
                        max_points=max_points,  # 控制规模（可调：1024~8192）
                        epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                        add_coords=False,  # 是否拼接空间坐标
                        coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                    ).item()
                    distances_sinkhorn_w2_latent_mapped.append(W2_sinkhorn_latent_mapped)
                    if self.args.bias:
                        W2_sinkhorn_latent_mapped_b1 = cal_wasserstein_loss(
                            tz1,
                            tz1,
                            max_points=max_points,  # 控制规模（可调：1024~8192）
                            epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                            add_coords=False,  # 是否拼接空间坐标
                            coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                        ).item()
                        distances_sinkhorn_w2_latent_mapped_bias1.append(W2_sinkhorn_latent_mapped_b1)
                        if len(distances_sinkhorn_w2_latent_mapped_bias2) == 0:
                            W2_sinkhorn_latent_mapped_b2 = cal_wasserstein_loss(
                                wass_cache[len(indices) - 1],
                                wass_cache[len(indices) - 1],
                                max_points=max_points,  # 控制规模（可调：1024~8192）
                                epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                                add_coords=False,  # 是否拼接空间坐标
                                coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                            ).item()
                            distances_sinkhorn_w2_latent_mapped_bias2.append(W2_sinkhorn_latent_mapped_b2)
                
                # # ================sinkhorn_image_wass==================
                if i in x_cache:
                    x1 = x_cache[i]
                else:
                    x1 = self.vae.decode(z1.to(self.vae.device) / 0.18215).sample  # [B,3,H,W]
                    x1 = self.normalize_samples(x1)
                    x1 = F.interpolate(
                        x1,
                        size=(64, 64),
                        mode='bilinear',
                        align_corners=False
                    )
                    x_cache[i] = x1

                W2_sinkhorn_image = cal_wasserstein_loss(
                    x1,
                    x_cache[len(indices) - 1],
                    max_points=max_points,  # 控制规模（可调：1024~8192）
                    epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                    add_coords=False,  # 是否拼接空间坐标
                    coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                ).item()
                distances_sinkhorn_w2_image.append(W2_sinkhorn_image)
                # print(f"step {i} → {j}: W₂_sinkhorn_image = {W2_sinkhorn_image:.6f}")

                ####################################### Neighbor #################################
                W2_sinkhorn_latent = cal_wasserstein_loss(
                    z1.squeeze(0),
                    z2.squeeze(0),
                    max_points=max_points,  # 控制规模（可调：1024~8192）
                    epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                    add_coords=False,  # 是否拼接空间坐标
                    coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                ).item()
                Neighbor_distances_sinkhorn_w2_latent.append(W2_sinkhorn_latent)
                # print(f"step {i} → {j}: W₂_sinkhorn_latent = {W2_sinkhorn_latent:.6f}")
                
                if self.wass_model is not None:
                    if i in wass_cache:
                        tz1 = wass_cache[i]
                    else:
                        tz1 = self.wass_model.encode(z1, True).squeeze(0)
                        wass_cache[i] = tz1
                    if j in wass_cache:
                        tz2 = wass_cache[j]
                    else:
                        tz2 = self.wass_model.encode(z2, True).squeeze(0)
                        wass_cache[j] = tz2
                        
                    W2_sinkhorn_latent_mapped = cal_wasserstein_loss(
                        tz1,
                        tz2,
                        max_points=max_points,  # 控制规模（可调：1024~8192）
                        epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                        add_coords=False,  # 是否拼接空间坐标
                        coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                    ).item()
                    Neighbor_distances_sinkhorn_w2_latent_mapped.append(W2_sinkhorn_latent_mapped)
                    if self.args.bias:
                        W2_sinkhorn_latent_mapped_b1 = cal_wasserstein_loss(
                            tz1,
                            tz1,
                            max_points=max_points,  # 控制规模（可调：1024~8192）
                            epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                            add_coords=False,  # 是否拼接空间坐标
                            coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                        ).item()
                        Neighbor_distances_sinkhorn_w2_latent_mapped_bias1.append(W2_sinkhorn_latent_mapped_b1)
                        W2_sinkhorn_latent_mapped_b2 = cal_wasserstein_loss(
                            tz2,
                            tz2,
                            max_points=max_points,  # 控制规模（可调：1024~8192）
                            epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                            add_coords=False,  # 是否拼接空间坐标
                            coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                        ).item()
                        Neighbor_distances_sinkhorn_w2_latent_mapped_bias2.append(W2_sinkhorn_latent_mapped_b2)
                
                # # ================sinkhorn_image_wass==================
                if i in x_cache:
                    x1 = x_cache[i]
                else:
                    x1 = self.vae.decode(z1.to(self.vae.device) / 0.18215).sample  # [B,3,H,W]
                    x1 = self.normalize_samples(x1)
                    x1 = F.interpolate(
                        x1,
                        size=(64, 64),
                        mode='bilinear',
                        align_corners=False
                    )
                    x_cache[i] = x1
                    
                if j in x_cache:
                    x2 = x_cache[j]
                else:
                    x2 = self.vae.decode(z2.to(self.vae.device) / 0.18215).sample  # [B,3,H,W]
                    x2 = self.normalize_samples(x2)
                    x2 = F.interpolate(
                        x2,
                        size=(64, 64),
                        mode='bilinear',
                        align_corners=False
                    )
                    x_cache[j] = x2
                    
                W2_sinkhorn_image = cal_wasserstein_loss(
                    x1,
                    x2,
                    max_points=max_points,  # 控制规模（可调：1024~8192）
                    epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                    add_coords=False,  # 是否拼接空间坐标
                    coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                ).item()
                Neighbor_distances_sinkhorn_w2_image.append(W2_sinkhorn_image)
                # print(f"step {i} → {j}: W₂_sinkhorn_image = {W2_sinkhorn_image:.6f}")

            results = {
                'distances_sinkhorn_w2_latent': np.array(distances_sinkhorn_w2_latent).mean(),
                'distances_sinkhorn_w2_latent_all': distances_sinkhorn_w2_latent,
                'distances_sinkhorn_w2_image': np.array(distances_sinkhorn_w2_image).mean(),
                'distances_sinkhorn_w2_image_all': distances_sinkhorn_w2_image,
                
                'Neighbor_distances_sinkhorn_w2_latent_': np.array(Neighbor_distances_sinkhorn_w2_latent).mean(),
                'Neighbor_distances_sinkhorn_w2_latent_all': Neighbor_distances_sinkhorn_w2_latent,
                'Neighbor_distances_sinkhorn_w2_image': np.array(Neighbor_distances_sinkhorn_w2_image).mean(),
                'Neighbor_distances_sinkhorn_w2_image_all': Neighbor_distances_sinkhorn_w2_image,
            }
            
        if self.wass_model is not None:
            results['Neighbor_distances_sinkhorn_w2_latent_mapped'] = np.array(Neighbor_distances_sinkhorn_w2_latent_mapped).mean()
            results['Neighbor_distances_sinkhorn_w2_latent_mapped_all'] = Neighbor_distances_sinkhorn_w2_latent_mapped
            results['Neighbor_distances_sinkhorn_w2_latent_mapped_bias1'] = Neighbor_distances_sinkhorn_w2_latent_mapped_bias1
            results['Neighbor_distances_sinkhorn_w2_latent_mapped_bias2'] = Neighbor_distances_sinkhorn_w2_latent_mapped_bias2
            
            results['distances_sinkhorn_w2_latent_mapped'] = np.array(distances_sinkhorn_w2_latent_mapped).mean()
            results['distances_sinkhorn_w2_latent_mapped_all'] = distances_sinkhorn_w2_latent_mapped
            results['distances_sinkhorn_w2_latent_mapped_bias1'] = distances_sinkhorn_w2_latent_mapped_bias1
            results['distances_sinkhorn_w2_latent_mapped_bias2'] = distances_sinkhorn_w2_latent_mapped_bias2

        return results

    @torch.no_grad()
    def cal_SingleWass_with_T(self, batch=None, mode='neighbor', max_points = 1024, 
                              freq = 1, min_num_steps = -1, max_num_steps=-1):
        """Generate a batch of samples"""
        # Generate random initial noise
        x0 = batch['x0']
        # Create time steps
        max_t = self.args.min_num_steps if min_num_steps < 0 else min_num_steps
        if max_num_steps > max_t and max_num_steps > 0:
            self.args.num_steps = max_num_steps
        t_span = torch.linspace(0, 1, max_t + 1, device=self.device)

        # Select solver and generate samples
        solver = self.euler_solver if self.args.solver == 'euler' else self.heun_solver
        samples, trajectory = solver(x0, t_span, 0, batch, None)
        print('--------------------- 计算 Wass 距离 --------------------')
        if mode == 'neighbor':
            Neighbor_distances_sinkhorn_w2_latent_mapped = []
            Neighbor_distances_sinkhorn_w2_latent_mapped_bias1 = []
            Neighbor_distances_sinkhorn_w2_latent_mapped_bias2 = []
        else:
            distances_sinkhorn_w2_latent_mapped = []
            distances_sinkhorn_w2_latent_mapped_bias1 = []
            distances_sinkhorn_w2_latent_mapped_bias2 = []

        indices = list(range(0, len(trajectory), freq))
        if indices[-1] != len(trajectory) - 1:
            indices.append(len(trajectory) - 1)

        results = {}
        z_last = trajectory[-1]
        wass_cache = {len(indices) - 1: self.wass_model.encode(z_last, True).squeeze(0)}
        for idx in range(len(indices) - 1):
            i = indices[idx]
            j = indices[idx + 1]
            z1 = trajectory[i]  # [1,C,H,W]
            z2 = trajectory[j] 
            
            if mode == 'neighbor':
                ####################################### Neighbor #################################
                if self.wass_model is not None:
                    if i in wass_cache:
                        tz1 = wass_cache[i]
                    else:
                        tz1 = self.wass_model.encode(z1, True).squeeze(0)
                        wass_cache[i] = tz1
                    if j in wass_cache:
                        tz2 = wass_cache[j]
                    else:
                        tz2 = self.wass_model.encode(z2, True).squeeze(0)
                        wass_cache[j] = tz2
                        
                    W2_sinkhorn_latent_mapped = cal_wasserstein_loss(
                        tz1,
                        tz2,
                        max_points=max_points,  # 控制规模（可调：1024~8192）
                        epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                        add_coords=False,  # 是否拼接空间坐标
                        coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                    ).item()
                    Neighbor_distances_sinkhorn_w2_latent_mapped.append(W2_sinkhorn_latent_mapped)
                    if self.args.bias:
                        W2_sinkhorn_latent_mapped_b1 = cal_wasserstein_loss(
                            tz1,
                            tz1,
                            max_points=max_points,  # 控制规模（可调：1024~8192）
                            epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                            add_coords=False,  # 是否拼接空间坐标
                            coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                        ).item()
                        Neighbor_distances_sinkhorn_w2_latent_mapped_bias1.append(W2_sinkhorn_latent_mapped_b1)
                        W2_sinkhorn_latent_mapped_b2 = cal_wasserstein_loss(
                            tz2,
                            tz2,
                            max_points=max_points,  # 控制规模（可调：1024~8192）
                            epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                            add_coords=False,  # 是否拼接空间坐标
                            coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                        ).item()
                        Neighbor_distances_sinkhorn_w2_latent_mapped_bias2.append(W2_sinkhorn_latent_mapped_b2)
                else:
                    if i in wass_cache:
                        tz1 = wass_cache[i]
                    else:
                        tz1 = z1.squeeze(0)
                        wass_cache[i] = tz1
                    if j in wass_cache:
                        tz2 = wass_cache[j]
                    else:
                        tz2 = z2.squeeze(0)
                        wass_cache[j] = tz2
                        
                    W2_sinkhorn_latent_mapped = cal_wasserstein_loss(
                        tz1,
                        tz2,
                        max_points=max_points,  # 控制规模（可调：1024~8192）
                        epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                        add_coords=False,  # 是否拼接空间坐标
                        coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                    ).item()

                    Neighbor_distances_sinkhorn_w2_latent_mapped.append(W2_sinkhorn_latent_mapped)
                    if self.args.bias:
                        W2_sinkhorn_latent_mapped_b1 = cal_wasserstein_loss(
                            tz1,
                            tz1,
                            max_points=max_points,  # 控制规模（可调：1024~8192）
                            epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                            add_coords=False,  # 是否拼接空间坐标
                            coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                        ).item()
                        Neighbor_distances_sinkhorn_w2_latent_mapped_bias1.append(W2_sinkhorn_latent_mapped_b1)
                        W2_sinkhorn_latent_mapped_b2 = cal_wasserstein_loss(
                            tz2,
                            tz2,
                            max_points=max_points,  # 控制规模（可调：1024~8192）
                            epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                            add_coords=False,  # 是否拼接空间坐标
                            coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                        ).item()
                        Neighbor_distances_sinkhorn_w2_latent_mapped_bias2.append(W2_sinkhorn_latent_mapped_b2)
            else:
                if self.wass_model is not None:
                    if i in wass_cache:
                        tz1 = wass_cache[i]
                    else:
                        tz1 = self.wass_model.encode(z1, True).squeeze(0)
                        wass_cache[i] = tz1
                        
                    W2_sinkhorn_latent_mapped = cal_wasserstein_loss(
                        tz1,
                        wass_cache[len(indices) - 1],
                        max_points=max_points,  # 控制规模（可调：1024~8192）
                        epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                        add_coords=False,  # 是否拼接空间坐标
                        coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                    ).item()
                    distances_sinkhorn_w2_latent_mapped.append(W2_sinkhorn_latent_mapped)
                    if self.args.bias:
                        W2_sinkhorn_latent_mapped_b1 = cal_wasserstein_loss(
                            tz1,
                            tz1,
                            max_points=max_points,  # 控制规模（可调：1024~8192）
                            epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                            add_coords=False,  # 是否拼接空间坐标
                            coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                        ).item()
                        distances_sinkhorn_w2_latent_mapped_bias1.append(W2_sinkhorn_latent_mapped_b1)
                        if len(distances_sinkhorn_w2_latent_mapped_bias2) == 0:
                            W2_sinkhorn_latent_mapped_b2 = cal_wasserstein_loss(
                                wass_cache[len(indices) - 1],
                                wass_cache[len(indices) - 1],
                                max_points=max_points,  # 控制规模（可调：1024~8192）
                                epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                                add_coords=False,  # 是否拼接空间坐标
                                coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                            ).item()
                            distances_sinkhorn_w2_latent_mapped_bias2.append(W2_sinkhorn_latent_mapped_b2)
                else:
                    if i in wass_cache:
                        tz1 = wass_cache[i]
                    else:
                        tz1 = self.wass_model.encode(z1, True).squeeze(0)
                        wass_cache[i] = tz1
                        
                    W2_sinkhorn_latent_mapped = cal_wasserstein_loss(
                        tz1,
                        wass_cache[len(indices) - 1],
                        max_points=max_points,  # 控制规模（可调：1024~8192）
                        epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                        add_coords=False,  # 是否拼接空间坐标
                        coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                    ).item()
                    distances_sinkhorn_w2_latent_mapped.append(W2_sinkhorn_latent_mapped)
                    if self.args.bias:
                        W2_sinkhorn_latent_mapped_b1 = cal_wasserstein_loss(
                            tz1,
                            tz1,
                            max_points=max_points,  # 控制规模（可调：1024~8192）
                            epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                            add_coords=False,  # 是否拼接空间坐标
                            coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                        ).item()
                        distances_sinkhorn_w2_latent_mapped_bias1.append(W2_sinkhorn_latent_mapped_b1)
                        if len(distances_sinkhorn_w2_latent_mapped_bias2) == 0:
                            W2_sinkhorn_latent_mapped_b2 = cal_wasserstein_loss(
                                wass_cache[len(indices) - 1],
                                wass_cache[len(indices) - 1],
                                max_points=max_points,  # 控制规模（可调：1024~8192）
                                epsilon=0.05,  # Sinkhorn 熵正则（越大越平滑，越小越接近真 W2）
                                add_coords=False,  # 是否拼接空间坐标
                                coord_lambda=0.5  # 位移成本系数（0 表示不关心“像素移动”）
                            ).item()
                            distances_sinkhorn_w2_latent_mapped_bias2.append(W2_sinkhorn_latent_mapped_b2)
            if mode == 'neighbor':
                results['Neighbor_distances_sinkhorn_w2_latent_mapped'] = np.array(Neighbor_distances_sinkhorn_w2_latent_mapped).mean()
                results['Neighbor_distances_sinkhorn_w2_latent_mapped_all'] = Neighbor_distances_sinkhorn_w2_latent_mapped
                results['Neighbor_distances_sinkhorn_w2_latent_mapped_bias1'] = Neighbor_distances_sinkhorn_w2_latent_mapped_bias1
                results['Neighbor_distances_sinkhorn_w2_latent_mapped_bias2'] = Neighbor_distances_sinkhorn_w2_latent_mapped_bias2
            else:
                results['distances_sinkhorn_w2_latent_mapped'] = np.array(distances_sinkhorn_w2_latent_mapped).mean()
                results['distances_sinkhorn_w2_latent_mapped_all'] = distances_sinkhorn_w2_latent_mapped
                results['distances_sinkhorn_w2_latent_mapped_bias1'] = distances_sinkhorn_w2_latent_mapped_bias1
                results['distances_sinkhorn_w2_latent_mapped_bias2'] = distances_sinkhorn_w2_latent_mapped_bias2

        return results

    def get_conds_from_batch(self, batch, transform, transform_grey, **kwargs):
        raise NotImplementedError
    
    def get_conds_from_items(self, x0, x1, label_id, caption, transform, transform_grey, **kwargs):
        conds = {}
        if isinstance(x0, str):
            x0 = Image.open(x0).convert("RGB")
        if isinstance(x1, str):
            x1 = Image.open(x1).convert("RGB")

        x0 = transform(x0).to(self.vae.device).unsqueeze(0)
        x1 = transform(x1).to(self.vae.device).unsqueeze(0)

        caption = [caption]
        y = torch.LongTensor([label_id]).to(self.vae.device)
        hint = transform_grey(x1).to(self.vae.device).unsqueeze(0)

        images = torch.cat([x0, x1], dim=0)
        with torch.no_grad():
            posterior = self.vae.encode(images).latent_dist
            images = posterior.sample() * 0.18215
            x0, x1 = images.chunk(2, dim=0)
                    
            if self.text_model is not None and caption[0] not in self.text_embeds_map:
                with torch.no_grad():
                    caption_input = self.text_tokenizer(caption, return_tensors="pt", padding=True).to(
                        self.text_model.device)
                    caption_outputs = self.text_model(**caption_input)
                    text_embeds = caption_outputs['last_hidden_state'].to(self.text_model.device)
                if len(self.text_embeds_map) < 500:
                    self.text_embeds_map[caption[0]] = text_embeds
            elif caption[0] in self.text_embeds_map:
                text_embeds = self.text_embeds_map[caption[0]]
            else:
                text_embeds = torch.zeros((len(y), 2, 1))
            if self.vision_model is not None:
                with torch.no_grad():
                    x1_images = torch.stack([process_single_image(image_path) for image_path in x1_path])
                    vision_embeds = self.vision_model.forward_features(x1_images.to(self.vision_model.device))
            else:
                vision_embeds = torch.zeros((len(y), 2, 1))

            caption_hidden_states = torch.cat([vision_embeds[:, 1:, ...].to(self.net_model.device),
                                                   text_embeds[:, 1:, ...].to(self.net_model.device)], dim=1)

            conds = {
                'x0': x0.to(self.device),
                'x1': x1.to(self.device),
                'caption': caption,
                'caption_hidden_states': caption_hidden_states.to(self.device),
                'y': y.to(self.device),
                'hint': hint.to(self.device),
                'text_embeds': text_embeds[:, 0].to(self.device),  # [B, D]
                'image_embeds': vision_embeds[:, 0].to(self.device)  # [B, D]
            }
         
            if self.args.use_gt_vt:
                t, xt, ut = self.FM.get_sample_location_and_conditional_flow(x0, x1,
                                                                    sample_plan=False,
                                                                    cond=conds,
                                                                    print_info=False)
                conds['ut'] = ut.to(self.device)
            return conds
        
    def generate(self):
        """Main generation loop"""
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform_grey = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.RandomGrayscale(p=1),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        eval_dataset = json.load(open(self.args.data_path, "r"))
        k = 2
        dataset = []
        total = 0
        for data in eval_dataset:
            if not os.path.exists(data["x0"]):
                continue
            total += 1
            for i in range(len(data['x1_dirs'])):
                x1_paths = os.listdir(data['x1_dirs'][i])
                for x1_path in x1_paths[:k]:
                    x1_path = os.path.join(data['x1_dirs'][i], x1_path)
                    if os.path.exists(x1_path):
                        new_data = {}
                        new_data["question_id"] = data["question_id"]
                        new_data["x0"] = data["x0"]
                        new_data["label_A"] = data["label_A"]
                        new_data["label_A_id"] = data["label_A_id"]
                        new_data["label_B"] = data["x1_labels"][i]
                        new_data["label_B_id"] = data["label_B_ids"][i]
                        new_data["caption"] = data["caption"][i]
                        new_data["x1"] = x1_path
                        new_data["hint_path"] = x1_path
                        dataset.append(new_data)
                        
        with open('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v1/images_dia_exam_flatten.json', 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
            
        dataset = MedicalJsonDataset(
            path="/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/simple_data_test/eval_all_flatten.json",
            transform=transform,
            transform_A=transform,
            transform_B=transform,
            hint_transform=transform_grey
        )

        idx = 0
        op_match_batch = False
        new_dataset = []
        caption_hidden_states_mode = 'cat'
        sigma = 0.0

        FM = OptimalTransportConditionalFlowMatcher(sigma=sigma, ot_method='exact')
        text_embeds_map = {}
        for idx in tqdm(range(len(dataset))):
            if idx < 0:
                continue
            batch = deepcopy(dataset[idx])
            if idx % 100 == 0:
                print(f"{batch['label_A']}:{batch['label_A_id']} --> {batch['label_B']}:{batch['label_B_id']}")
                print(f'---------------------------------{idx}/{len(dataset)}----------------------------------')
                
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
                'x0_path': [batch['x0_path']],
                'x1_path': [batch['x1_path']],
                'caption': caption,
                'caption_hidden_states': caption_hidden_states.to(self.device),
                'y': y.to(self.device),
                'hint': hint.to(self.device),
                'text_embeds': text_embeds[:, 0].to(self.device),  # [B, D]
                'image_embeds': vision_embeds[:, 0].to(self.device)  # [B, D]
            }
            self.use_gt_vt = self.args.use_gt_vt
            if self.use_gt_vt:
                t, xt, ut = FM.get_sample_location_and_conditional_flow(x0, x1,
                                                                    sample_plan=op_match_batch,
                                                                    cond=conds,
                                                                    print_info=False)
                conds['ut'] = ut.to(self.device)
            wass_dict = self.cal_Wass_with_T(idx, conds)
            wass_dict.update(dataset[idx])
            wass_dict.pop('x0', None)
            wass_dict.pop('x1', None)
            wass_dict.pop('hint', None)
            wass_dict.pop('mask_hint', None)
            wass_dict['class_id'] = int(wass_dict['class_id'].item())
            new_dataset.append(wass_dict)

            if idx % 1000 == 0:
                output_path = os.path.join(self.args.output_dir, 'result.json')
                with open(output_path, 'w') as f:
                    json.dump(new_dataset, f, indent=4, ensure_ascii=False)
                print(f'Current Saved to {output_path}')
        output_path = os.path.join(self.args.output_dir, 'result.json')
        with open(output_path, 'w') as f:
            json.dump(new_dataset, f, indent=4, ensure_ascii=False)
        print(f'Saved to {output_path}')
        return new_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Sampling script for CFM model')
    parser.add_argument('--data_path', type=str,
                        default='/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v1/images_dia_exam.json',
                        help='数据路径')
    parser.add_argument('--checkpoint', type=str,
                        default='/mnt/inaisfs/data/home/tansy_criait/new_flow_match/outputs/image_hint_十二指肠/otcfm/otcfm_weights_step_30000.pt',
                        help='Path to the checkpoint file')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/result/image_hint_attn_other',
                        help='Directory to save generated images')
    parser.add_argument('--wass_model_path', type=str, 
                        default="/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match_十二指肠/best_weights_min/old_attention_十二指肠球部.pt",
                        help='优先级 高于 权重')
    parser.add_argument('--wass_model_type', type=str, 
                        choices=['resnet34', 'attention'], 
                        default="attention", 
                        # default="resnet34", 
                        help='指定模型类型') 
    parser.add_argument('--full', type=bool, default=True,
                        help='')
    parser.add_argument('--bias', type=bool, default=True,
                        help='')
    parser.add_argument('--stop_method', type=str, default='direct',
                        choices=['diff', 'second_diff', 'direct'],
                        help='ODE solver type')
    parser.add_argument('--use_cache', type=bool, default=False,
                        help='使用 Cache 加速，但可能会影响结果。')
    parser.add_argument('--sim_model_type', type=str, 
                        choices=['gme', 'dinov3'],
                        default="dinov3",
                        help='指定相似模型类型')
    parser.add_argument('--stop_beta', type=float, default=0.91,
                        help='控制停止生成的阈值')
    parser.add_argument('--num_steps', type=int, default=8,
                        help='Max Number of steps in the ODE solver')
    parser.add_argument('--min_num_steps', type=int, default=2,
                        help='Min Number of steps in the ODE solver')
    parser.add_argument('--fix_num_steps', type=int, default=-1,
                        help='Fix Number of steps in the ODE solver')
    parser.add_argument('--temp', type=str, default='/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/cal_wass/temp1',
                        help='')
    parser.add_argument('--sim_model_path', type=str, 
                        default="/mnt/inaisfs/data/home/tansy_criait/weights/dinov3-vit7b16",
                        help='指定相似模型类型')
    parser.add_argument('--num_channels', type=int, default=128,
                        help='Number of base channels in UNet')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Use EMA model for inference')
    parser.add_argument('--solver', type=str, default='heun',
                        choices=['euler', 'heun'],
                        help='ODE solver type')
    parser.add_argument('--use_gt_vt', type=bool, default=False,
                        help='强制引导生成, 但是用 euler solver。')
    parser.add_argument('--save_grid', action='store_true', default=False,
                        help='Save grid of samples for each batch')
    parser.add_argument('--save_intermediates', action='store_true', default=True,
                        help='Save intermediate steps during generation')
    parser.add_argument('--intermediate_freq', type=int, default=1,
                        help='Frequency of saving intermediate steps')
    parser.add_argument('--max_points', type=int, default=1024,
                        help='')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Total number of images to generate')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for generation')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512],
                        help='Image size (height, width)')
    parser.add_argument(
        '--wass_space',
        type=str,
        choices=['latent', 'image_pixels', 'image_feats'],
        default='latent',
        help="Where to compute distances: 'latent' (VAE latent), 'image_pixels' (decode to RGB), "
             "'image_feats' (decode then feature space)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    generator = ImageGenerator(args)
    generator.generate()

if __name__ == "__main__":
    main()

"""
python generate_samples_grid.py --checkpoint outputs/results_otcfm_32_otcfm-large-batch_exp/otcfm/otcfm_weights_step_2000000.pt  --num_samples 4 --batch_size 4 --output_dir sample_ot-cfm_large_batch --image_size 128 128 --num_steps 8 --use_ema --solver heun --save_grid --save_intermediates --intermediate_freq 
""" 