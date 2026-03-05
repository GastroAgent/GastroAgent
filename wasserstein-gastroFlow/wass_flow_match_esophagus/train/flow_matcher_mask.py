from collections import OrderedDict
from copy import copy
import random
import sys
sys.path.append("./GasAgent-main")
from safetensors.torch import load_file
from functools import partial
import torch.nn.functional as F
import torch
from torch import nn
import argparse
import json
from torchvision.utils import save_image, make_grid
from timm.models.vision_transformer import VisionTransformer
from torchvision import transforms as T

from PIL import Image
from diffusers import AutoencoderKL, StableDiffusionPipeline
from my_models.unet_2d_condition import UNet2DConditionModel
from transformers import ChineseCLIPModel, ChineseCLIPTextModel, AutoTokenizer, ChineseCLIPTextConfig
from conditional_flow_matcher import ConditionalFlowMatcher, OptimalTransportConditionalFlowMatcher

from my_models.model_dispatch import dispatch_model
from model_utils.model import *

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

class ImageGenerator:
    def __init__(self, args, only_vae=False, device=None, use_gt=False, need_ut=False):
        self.only_vae = only_vae
        self.args = args
        self.solver = self.args.solver
        self.need_ut = need_ut
        if use_gt:
            self.solver = self.args.solver = 'euler'
            self.need_ut = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        net_model, vae, (text_model, text_tokenizer), (vision_model, process_single_image) = self._load_model(
            args.checkpoint, args.use_ema)
        if device is not None:
            self.net_model = net_model.to(self.device) if net_model is not None else net_model
            self.vae = vae.to(self.device) if vae is not None else vae
            self.text_model = text_model.to(self.device) if text_model is not None else text_model
            self.text_tokenizer = text_tokenizer 
            self.vision_model = vision_model.to(self.device) if vision_model is not None else vision_model
            if vision_model is not None:
                self.vision_model.device = self.device
        else:
            self.net_model = net_model
            self.vae = vae
            self.text_model = text_model
            self.text_tokenizer = text_tokenizer 
            self.vision_model = vision_model
            try:
                self.vision_model.device
            except:
                self.vision_model.device = self.device
                
        self.process_single_image = process_single_image
        self.FM = OptimalTransportConditionalFlowMatcher(sigma=0.0, ot_method='exact')

    def _load_model(self, checkpoint: str, use_ema=False):
        """Initialize and load the model"""
        if not self.only_vae:
            config = json.load(open(
                './flow_matcher_otcfm/unet_mask/config.json',
                'r'))
            net_model = UNet2DConditionModel(**config)
            if checkpoint:
                state_dict = torch.load(f'{checkpoint}', map_location='cpu')
            else:
                state_dict = {}
            if use_ema and 'ema_model' in state_dict:
                net_model.load_state_dict(state_dict['ema_model'], strict=False)
            elif 'net_model' in state_dict:
                net_model.load_state_dict(state_dict['net_model'], strict=False)

            vae = AutoencoderKL.from_pretrained(
                './flow_matcher_otcfm/vae').to(
                device=self.device).eval()

            text_model_config = json.load(open('./flow_matcher_otcfm/text_encoder/config.json','r'))
            text_model_config = ChineseCLIPTextConfig(**text_model_config)
            text_model = ChineseCLIPTextModel(text_model_config).eval()
            text_model.load_state_dict(load_file("./flow_matcher_otcfm/text_model/model.safetensors"), strict=False)
            text_tokenizer = AutoTokenizer.from_pretrained(
                './flow_matcher_otcfm/text_model', use_fast=True)
            
            if 'text_model' in state_dict:
                text_model.load_state_dict(state_dict['text_model'], strict=False)

            # Define the model (ensure this matches your model's architecture)
            vision_model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                             qkv_bias=False,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6)).eval()
            if 'vision_model' in state_dict:
                vision_model.load_state_dict(state_dict['vision_model'], strict=False)

            net_model = net_model.to(device=self.device)
            vae = vae.to(self.device)
            if text_model is not None:
                text_model = text_model.to(self.device)
            if vision_model is not None:
                vision_model = vision_model.to(self.device)
                vision_model.device = self.device

            if self.args.use_controlnet:
                from my_models.unet_2d_condition_controlnet import ControlUNet2DConditionModel
                try:
                    controlnet_config = json.load(
                        open("./flow_matcher_otcfm/unet/controlnet_config.json",
                            'r'))
                    controlnet_config['in_channels'] = 3
                    controlnet_config['encoder_hid_dim_type'] = 'text_proj'
                    controlnet_config['encoder_hid_dim'] = 768
                    controlnet = ControlUNet2DConditionModel(**controlnet_config)
                    controlnet = controlnet.to(self.device)
                    if "controlnet" in state_dict:
                        controlnet.load_state_dict(state_dict['controlnet'], strict=False)
                    self.controlnet = controlnet
                except:
                    self.args.use_controlnet = False
                    self.controlnet = None
                    
            else:
                self.controlnet = None

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
            
            if self.args.wass_model_path:
                model = TripletNetwork(pretrained=True, freeze_base=False, model=self.args.wass_model_type, dy=True)
                state_dict = torch.load(self.args.wass_model_path, weights_only=True)
                model.load_state_dict(state_dict, strict=False)
                model = model.to(self.device)
                self.wass_model = model
                self.wass_model.eval()
            elif 'wass_model' in state_dict:
                if self.args.wass_model_type == '':
                    self.args.wass_model_type = 'resnet34'
                model = TripletNetwork(pretrained=True, freeze_base=False, model=self.args.wass_model_type)
                model.load_state_dict(state_dict['wass_model'], strict=False)
                model = model.to(self.device)
                self.wass_model = model
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
                from modelscope import AutoImageProcessor, AutoModel
                from transformers.image_utils import load_image
                pretrained_model_name = self.args.sim_model_path
                processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
                model = AutoModel.from_pretrained(
                    pretrained_model_name, 
                    device_map="auto", 
                    max_memory={1:"40GiB"} 
                )
                def similarity(image1: str, image2: str, step=0):
                    if step == 0:
                        return 0
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
                self.similarity = similarity
            elif self.args.sim_model_type == 'convnext':
                sys.path.append('.')
                from discriminator.train_latent_space import TripletNetwork
                model = TripletNetwork(pretrained=True, freeze_base=False, model='convnext')
                try:
                    state_dict = torch.load(self.args.sim_model_path, weights_only=True)
                    model.load_state_dict(state_dict, strict=False)
                except:
                    pass
                model = model.to(self.device)
                def similarity(image1, image2, step=0):
                    if step == 0:
                        return 0
                    with torch.inference_mode():
                        embeds = model.encode(torch.cat([image1, image2], dim=0), False)
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
                './flow_matcher_otcfm/vae').to(
                device=self.device).eval()
            return net_model, vae, (text_model, text_tokenizer), (vision_model, process_single_image)

    @staticmethod
    def normalize_samples(x):
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
    
    def euler_solver_stop(self, x0, t_span, batch_idx, batch):
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
        step = len(trajectory) - 1
        temp = x0.clone()

        def continue_strategy(image1, image2, step: int, method='direct', *args, **kwargs):
            if step == 0:
                return True
            
            if method == 'direct':
                self.sim = self.similarity(image1, image2, step)
                return self.sim < self.args.stop_beta 
            elif method == 'diff':
                current_sim = self.similarity(image1, image2, step)
                diff_sim = current_sim - self.last_sim
                self.last_sim = current_sim
                self.sim = current_sim
                return abs(diff_sim) > self.args.stop_beta 
            elif method == 'second_diff':
                current_sim = self.similarity(image1, image2, step)
                current_diff_sim = current_sim - self.last_sim
                self.last_sim = current_sim
                second_diff_sim = current_diff_sim - self.last_diff_sim
                self.last_diff_sim = current_diff_sim
                self.sim = current_sim
                return abs(second_diff_sim) > self.args.stop_beta 
            else:
                self.sim = self.similarity(image1, image2, step)
                return self.sim < self.args.stop_beta 
            
        while continue_strategy(temp, x1, step, method = self.args.stop_method) and max_t <= self.args.num_steps and (self.args.fix_num_steps < 0 or step == 0): 
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

            max_t *= 2 
            t_span = torch.linspace(0, 1, max_t + 1, device=self.device)
            
            if step in imgcache:
                temp = imgcache[step]
            else:
                imgcache[step] = x.clone()

        for t_idx, (_, xt) in enumerate(sorted(ycache.items())): 
            if (t_idx + 1) % self.args.intermediate_freq == 0:
                trajectory.append(xt.detach().clone())
        trajectory.append(x1.detach().clone())
        return x, trajectory, copy(self.sim)

    def heun_solver_stop(self, x0, t_span, batch_idx, batch):
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
        temp = x0.clone()
        self.last_sim = 0
        self.last_diff_sim = 0
        step = len(trajectory) - 1

        def continue_strategy(image1, image2, step: int, method='direct', *args, **kwargs):
            if step == 0:
                return True
            if not self.args.sim_model_type == 'convnext' and not isinstance(image1, str):
                img = self.vae.decode(image1.to(self.vae.device) / 0.18215).sample
                img = self.normalize_samples(img)
                save_image(img, self.args.temp + '/x0.jpg')
                image1 = self.args.temp + '/x0.jpg'
                
            if not self.args.sim_model_type == 'convnext' and not isinstance(image2, str):
                img = self.vae.decode(image2.to(self.vae.device) / 0.18215).sample
                img = self.normalize_samples(img)
                save_image(img, self.args.temp + '/x1.jpg')
                image2 = self.args.temp + '/x1.jpg'
                
            if method == 'direct':
                self.sim = self.similarity(image1, image2, step)
                return self.sim < self.args.stop_beta 
            elif method == 'diff':
                current_sim = self.similarity(image1, image2, step)
                diff_sim = current_sim - self.last_sim
                self.last_sim = current_sim
                self.sim = current_sim
                return abs(diff_sim) > self.args.stop_beta 
            elif method == 'second_diff':
                current_sim = self.similarity(image1, image2, step)
                current_diff_sim = current_sim - self.last_sim
                self.last_sim = current_sim
                second_diff_sim = current_diff_sim - self.last_diff_sim
                self.last_diff_sim = current_diff_sim
                self.sim = current_sim
                return abs(second_diff_sim) > self.args.stop_beta 
            else:
                self.sim = self.similarity(image1, image2, step)
                return self.sim < self.args.stop_beta 
            
        while continue_strategy(temp, x1, step, method = self.args.stop_method) and max_t <= self.args.num_steps and (self.args.fix_num_steps < 0 or step == 0): 
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

            max_t *= 2
            t_span = torch.linspace(0, 1, max_t + 1, device=self.device)
            
            if step in imgcache:
                temp = imgcache[step]
            else:
                temp = x.clone()
        
        for t_idx, (_, xt) in enumerate(sorted(ycache.items())): 
            if (t_idx + 1) % self.args.intermediate_freq == 0:
                trajectory.append(xt.detach().clone())
        trajectory.append(x1.detach().clone())
        return x, trajectory, copy(self.sim)
    
    def euler_solver(self, x0, t_span, batch):
        """Euler solver with intermediate saves"""
        x = x0
        sample_steps = []
        dt = t_span[1] - t_span[0]
        caption_hidden_states = batch['caption_hidden_states']
        y = batch['y']

        use_gt_vt = True
        if not use_gt_vt:
            if "mask_hint" in batch and self.args.use_controlnet:
                condition_x = batch['mask_hint']
                condition_x = F.interpolate(
                    condition_x,
                    size=(64, 64),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                pass
        
        for t_idx in range(len(t_span) - 1):
            t = t_span[t_idx] * torch.ones(x0.shape[0], device=self.device)
            if use_gt_vt:
                ut = batch['ut']
                dx = ut
            else:
                with torch.no_grad():
                    if self.args.use_controlnet:
                        # controlnet_output = self.controlnet(condition_x, t.to(controlnet.device))
                        controlnet_output = self.controlnet(condition_x.to(self.controlnet.device),
                                                            t.to(self.controlnet.device),
                                                            encoder_hidden_states=caption_hidden_states.detach().to(
                                                                    self.controlnet.device))
                        down_block_additional_residuals = controlnet_output.down_block_additional_residuals
                        mid_block_additional_residual = controlnet_output.mid_block_additional_residual.to(
                                self.net_model.device)
                        # torch.cuda.empty_cache()
                    else:
                        down_block_additional_residuals = None
                        mid_block_additional_residual = None
                        
                    dx = self.net_model(x.to(self.device), timestep=t.squeeze().to(self.device), class_labels=y.to(self.device),
                                        encoder_hidden_states=caption_hidden_states.to(self.device), added_cond_kwargs=batch,
                                        image_hint_model='cat',
                                        down_block_additional_residuals=down_block_additional_residuals,
                                        mid_block_additional_residual=mid_block_additional_residual).sample.to(self.device)

            x = x.to(self.device) + dx.to(self.device) * dt.to(self.device)
            if t_idx % self.args.intermediate_freq == 0:
                sample_steps.append(x)

        return sample_steps

    def heun_solver(self, x0, t_span, batch):
        """Heun's solver with intermediate saves"""
        x = x0
        sample_steps = []
        dt = t_span[1] - t_span[0]
        caption_hidden_states = batch['caption_hidden_states']
        y = batch['y']

        if "mask_hint" in batch:
            condition_x = batch['mask_hint']
            condition_x = F.interpolate(
                condition_x,
                size=(64, 64),
                mode='bilinear',
                align_corners=False
            )
        else:
            pass
                
        for t_idx in range(len(t_span) - 1):
            t = t_span[t_idx] * torch.ones(x0.shape[0], device=self.device)
            t_next = t_span[t_idx + 1] * torch.ones(x0.shape[0], device=self.device)
            with torch.no_grad():
                if self.args.use_controlnet:
                    # controlnet_output = self.controlnet(condition_x, t.to(controlnet.device))
                    controlnet_output = self.controlnet(condition_x.to(self.controlnet.device),
                                                        t.to(self.controlnet.device),
                                                        encoder_hidden_states=caption_hidden_states.detach().to(
                                                            self.controlnet.device))
                    down_block_additional_residuals = controlnet_output.down_block_additional_residuals
                    mid_block_additional_residual = controlnet_output.mid_block_additional_residual.to(
                        self.net_model.device)
                    # torch.cuda.empty_cache()
                else:
                    down_block_additional_residuals = None
                    mid_block_additional_residual = None
                    
                # First step: Euler
                dx1 = self.net_model(x, timestep=t.squeeze(), class_labels=y,
                                    encoder_hidden_states=caption_hidden_states, added_cond_kwargs=batch,
                                    image_hint_model='cat',
                                    down_block_additional_residuals=down_block_additional_residuals,
                                    mid_block_additional_residual=mid_block_additional_residual).sample.to(self.device)
                x_euler = x + dx1 * dt
                if self.args.use_controlnet:
                    controlnet_output = self.controlnet(condition_x.to(self.controlnet.device),
                                                        t_next.to(self.controlnet.device),
                                                        encoder_hidden_states=caption_hidden_states.detach().to(
                                                            self.controlnet.device))
                    down_block_additional_residuals = controlnet_output.down_block_additional_residuals
                    mid_block_additional_residual = controlnet_output.mid_block_additional_residual.to(
                        self.net_model.device)
                    # torch.cuda.empty_cache()
                else:
                    down_block_additional_residuals = None
                    mid_block_additional_residual = None
                    
                # Second step: Correction
                dx2 = self.net_model(x_euler, timestep=t_next.squeeze(), class_labels=y,
                                    encoder_hidden_states=caption_hidden_states, added_cond_kwargs=batch,
                                    image_hint_model='cat',
                                    down_block_additional_residuals=down_block_additional_residuals,
                                    mid_block_additional_residual=mid_block_additional_residual).sample.to(self.device)
            x = x + (dx1 + dx2) * dt / 2
            if t_idx % self.args.intermediate_freq == 0:
                sample_steps.append(x)
        return sample_steps

    def generate_batch(self, batch=None, sample_step=-1, return_all_steps=False):
        """Generate a batch of samples"""
        x0 = batch['x0']
        if self.args.min_num_steps > 0:
            num_steps = random.choice([x for x in range(self.args.num_steps) if x >= self.args.min_num_steps])
        else:
            num_steps = self.args.num_steps

        # Create time steps
        t_span = torch.linspace(0, 1, num_steps, device=self.device)
        # Select solver and generate samples
        solver = self.euler_solver if self.args.solver == 'euler' else self.heun_solver
        sample_steps = solver(x0, t_span, batch)
        samples = sample_steps[sample_step]
        if return_all_steps:
            return samples, sample_steps
        return samples
    
    def generate_batch_sim_stop(self, batch=None):
        """Generate a batch of samples"""
        # Generate random initial noise
        x0 = batch['x0']
        # Create time steps
        max_t = self.args.min_num_steps if self.args.fix_num_steps < 0 else self.args.fix_num_steps
        t_span = torch.linspace(0, 1, max_t + 1, device=self.device)

        # Select solver and generate samples
        solver = self.euler_solver_stop if self.args.solver == 'euler' else self.heun_solver_stop
        samples, trajectory, stop_step_sim = solver(x0, t_span, 0, batch)
        return samples, trajectory, stop_step_sim
        
    def generate(self, batch=None, test=False, sample_step=-1, return_all_steps=False, mode='default'):
        ##############
        x0 = batch['x0'].to(self.vae.device)
        x1 = batch['x1'].to(self.vae.device)
        caption = batch['caption']
        y = batch['class_id'].to(self.vae.device)
        x1_path = batch['x1_path']
        x0_path = batch['x0_path']
        hint = batch['hint']
        if self.args.use_controlnet and 'mask_hint' in batch:
            mask_hint_paths = batch['mask_hint_path']
            mask_hint = batch['mask_hint']
        else:
            pass
        
        if 'mask_hint' in batch:
            mask_hint = batch['mask_hint']
        else:
            if use_image_mask:
                mask_hint = torch.randn_like(x0)
            else:
                mask_hint = None

        x0_raw = x0.clone()
        result = {}
        with torch.no_grad():
            if x0.shape[1] == 4:
                posterior = self.vae.encode(x1).latent_dist
                x1 = posterior.sample() * 0.18215
            else:
                images = torch.cat([x0, x1], dim=0)
                posterior = self.vae.encode(images).latent_dist
                images = posterior.sample() * 0.18215
                x0, x1 = images.chunk(2, dim=0)

            if self.text_model is not None:
                caption_input = self.text_tokenizer(caption, return_tensors="pt", padding=True).to(
                    self.text_model.device)
                caption_outputs = self.text_model(**caption_input)
                text_embeds = caption_outputs['last_hidden_state'].to(self.text_model.device)
            if self.vision_model is not None:
                x1_images = torch.stack([process_single_image(image_path) for image_path in x1_path])
                vision_embeds = self.vision_model.forward_features(x1_images.to(self.vision_model.device))

            if self.args.caption_hidden_states_mode == 'cat' and self.text_model is not None and self.vision_model is not None:
                caption_hidden_states = torch.cat([vision_embeds[:, 1:, ...].to(self.net_model.device),
                                                   text_embeds[:, 1:, ...].to(self.net_model.device)], dim=1)
            elif self.args.caption_hidden_states_mode == 'only_text' and self.text_model is not None and self.vision_model is not None:
                caption_hidden_states = text_embeds[:, 1:, ...].to(self.net_model.device)
                
        ### Flow matching core
        if self.args.use_controlnet:
            conds = {
                'x0': x0.to(self.device),
                # 'x0_raw': x0_raw,
                'x1': x1.to(self.device),
                # 'x1_raw': x1_raw,
                'caption': caption.to(self.device),
                'caption_hidden_states': caption_hidden_states.to(self.device) if self.text_model is not None else None,
                'y': y.to(self.device),
                "hint": hint.to(x0.device) if self.vision_model is not None else torch.zeros_like(x0_raw),
                "mask_hint": mask_hint.to(x0.device) if self.vision_model is not None else torch.zeros_like(x0_raw),
                'text_embeds': text_embeds[:, 0] if self.text_model is not None else None,  # [B, D]
                'image_embeds': vision_embeds[:, 0] if self.vision_model is not None else None,  # [B, D]
            }
        else:
            if mask_hint is not None:
                conds = {
                    'x0': x0.to(self.device),
                    'x0_path': x0_path,
                    # 'x0_raw': x0_raw,
                    'x1': x1.to(self.device),
                    'x1_path': x1_path,
                    # 'x1_raw': x1_raw,
                    'caption': caption,
                    'caption_hidden_states': caption_hidden_states.to(self.device),
                    'y': y.to(self.device),
                    "hint": hint.to(self.device),
                    'mask_hint': mask_hint.to(self.device),
                    'text_embeds': text_embeds[:, 0].to(self.device),  # [B, D]
                    'image_embeds': vision_embeds[:, 0].to(self.device),  # [B, D]
                }
            else:
                conds = {
                    'x0': x0.to(self.device),
                    'x0_path': x0_path,
                    # 'x0_raw': x0_raw,
                    'x1': x1.to(self.device),
                    'x1_path': x1_path,
                    # 'x1_raw': x1_raw,
                    'caption': caption,
                    'caption_hidden_states': caption_hidden_states.to(self.device),
                    'y': y.to(self.device),
                    "hint": hint.to(self.device),
                    'text_embeds': text_embeds[:, 0].to(self.device),  # [B, D]
                    'image_embeds': vision_embeds[:, 0].to(self.device),  # [B, D]
                }     


        for key in batch:
            if key not in conds:
                conds[key] = batch[key]
        if self.need_ut:
            t, xt, ut = self.FM.get_sample_location_and_conditional_flow(x0, x1, sample_plan=self.args.op_match, cond=conds,
                                                                    replace=False, print_info=False)
            conds['ut'] = ut.to(self.device)

        # Generate batch Sim Stop:
        if mode == 'sim_stop':
            if return_all_steps:
                samples, trajectory, stop_step_sim = self.generate_batch_sim_stop(conds)
                result['all_samples'] = trajectory
                result['stop_step_sim'] = stop_step_sim.item()
                result['stop_steps'] = len(trajectory)
            else:
                samples = self.generate_batch(conds, sample_step=sample_step)
        # Generate batch Default
        else:
            if return_all_steps:
                samples, all_samples = self.generate_batch(conds, sample_step=sample_step)
                result['all_samples'] = all_samples
            else:
                samples = self.generate_batch(conds, sample_step=sample_step)
            
        result['samples'] = samples
        result["x0_vaed"] = x0
        result["x1_vaed"] = x1
        return result

def generate_parse_args():
    parser = argparse.ArgumentParser(description='Sampling script for CFM model')
    parser.add_argument('--data_path', type=str,
                        default='./data/胃/new_eval_all_flatten.json',
                        help='数据路径') 
    parser.add_argument('--checkpoint', type=str,
                        default='./outputs/flow-match_vae_mask/otcfm/otcfm_weights_step_50000.pt',
                        help='Path to the checkpoint file') 
    parser.add_argument('--output_dir', type=str,
                        default='./result/image_hint_Anatomy',
                        help='Directory to save generated images')
    parser.add_argument('--num_steps', type=int, default=8,
                        help='Max Number of steps in the ODE solver')
    parser.add_argument('--min_num_steps', type=int, default=-1,
                        help='Min Number of steps in the ODE solver')
    parser.add_argument('--fix_num_steps', type=int, default=-1,
                        help='Fix Number of steps in the ODE solver')                                   
    parser.add_argument('--stop_beta', type=float, default=0.91,
                        help='控制停止生成的阈值')
    parser.add_argument('--use_cache', type=bool, default=False,
                        help='使用 Cache 加速，但可能会影响结果。')
    parser.add_argument('--bias', type=bool, default=False,
                        help='')  
    parser.add_argument('--temp', type=str, default='./temp/temp2',
                        help='')
    parser.add_argument('--stop_method', type=str, default='direct',
                        choices=['diff', 'second_diff', 'direct'],
                        help='判停策略')
    parser.add_argument('--wass_model_path', type=str, 
                        default="./best_flow_weights/attention_tsy.pt",
                        help='优先级 高于 权重')
    parser.add_argument('--wass_model_type', type=str, 
                        choices=['resnet34', 'attention'], 
                        default="attention", 
                        # default="resnet34", 
                        help='指定模型类型') 
    parser.add_argument('--sim_model_type', type=str, 
                        choices=['convnext', 'gme', 'dinov3'],
                        default="convnext",
                        help='指定相似模型类型')
    parser.add_argument('--full', type=bool, default=True,
                        help='')
    parser.add_argument('--sim_model_path', type=str, 
                        default = "./discriminator/latent_model_weight/convnext3.pt",
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
    parser.add_argument('--caption_hidden_states_mode', type=str, default="cat",
                        help='')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Total number of images to generate')
    parser.add_argument('--use_controlnet', type=bool, default=False,
                        help='')
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

def create_generator(**kwargs):
    args = generate_parse_args()
    generator = ImageGenerator(args, **kwargs)
    return generator


if __name__ == "__main__":
    create_generator()

"""
python generate_samples_grid.py --checkpoint outputs/results_otcfm_32_otcfm-large-batch_exp/otcfm/otcfm_weights_step_2000000.pt  --num_samples 4 --batch_size 4 --output_dir sample_ot-cfm_large_batch --image_size 128 128 --num_steps 8 --use_ema --solver heun --save_grid --save_intermediates --intermediate_freq 
"""