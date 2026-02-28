from collections import OrderedDict
from copy import copy
import random
import glob
from functools import partial
import sys
import torch.nn.functional as F
import torch
from torch import nn
import argparse
import json
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from timm.models.vision_transformer import VisionTransformer
from torchvision import transforms as T
import sys
sys.path.append("./GasAgent-main")
from PIL import Image
from diffusers import AutoencoderKL, StableDiffusionPipeline
from my_models.unet_2d_condition import UNet2DConditionModel
from transformers import ChineseCLIPModel, ChineseCLIPTextModel, AutoTokenizer, ChineseCLIPTextConfig
from conditional_flow_matcher import ConditionalFlowMatcher, OptimalTransportConditionalFlowMatcher
from safetensors.torch import load_file
from utils.data_loader import MedicalJsonDataset
from my_models.model_dispatch import dispatch_model
from my_models.unet_2d_condition_controlnet import ControlUNet2DConditionModel

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
    def __init__(self, args, only_vae=False):
        self.only_vae = only_vae
        self.args = args
        self.use_gt_vt = False
        if self.only_vae:
            self.args.solver = "euler"
            self.use_gt_vt = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net_model, vae, (text_model, text_tokenizer), (vision_model, process_single_image) = self._load_model(
            args.checkpoint, args.use_ema)
        self.net_model = net_model
        self.vae = vae
        self.text_model = text_model
        self.text_tokenizer = text_tokenizer
        self.vision_model = vision_model
        self.process_single_image = process_single_image
        self.FM = OptimalTransportConditionalFlowMatcher(sigma=0.0, ot_method='exact')
    
    def _load_model(self, checkpoint: str, use_ema=False):
        """Initialize and load the model"""
        if not self.only_vae:
            config = json.load(open(
                './flow_matcher_otcfm/unet/config.json',
                'r'))
            net_model = UNet2DConditionModel(**config)
            # class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
            # init.normal_(class_embedding.weight, mean=0.0, std=0.01)  # 正态分布初始化
            state_dict = torch.load(f'{checkpoint}', map_location='cuda')
            if use_ema and 'ema_model' in state_dict:
                net_model.load_state_dict(state_dict['ema_model'], strict=False)
            elif 'net_model' in state_dict:
                net_model.load_state_dict(state_dict['net_model'], strict=False)

            vae = AutoencoderKL.from_pretrained(
                './flow_matcher_otcfm/vae').to(
                device='cuda').eval()
            # vae.load_state_dict(torch.load('/dev/shm/jmf/mllm_weight/sd-ema-vae_weight/sd-vae_epoch_ema.pth'), strict=False)
            text_model_config = json.load(open('./flow_matcher_otcfm/text_encoder/config.json','r'))
            text_model_config = ChineseCLIPTextConfig(**text_model_config)
            text_model = ChineseCLIPTextModel(text_model_config).eval()
            text_tokenizer = AutoTokenizer.from_pretrained(
                './flow_matcher_otcfm/text_encoder', use_fast=True)
            if 'text_model' in state_dict:
                text_model.load_state_dict(state_dict['text_model'], strict=False)

            # Define the model (ensure this matches your model's architecture)
            vision_model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                             qkv_bias=False, 
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6)).eval()
            if 'vision_model' in state_dict:
                vision_model.load_state_dict(state_dict['vision_model'], strict=False)

            net_model = net_model.to(device='cuda')
            vae = vae.to('cuda')
            if text_model is not None:
                text_model = text_model.to('cuda')
            if vision_model is not None:
                vision_model = vision_model.to('cuda')
                vision_model.device = 'cuda'

            if self.args.use_controlnet:
                try:
                    controlnet_config = json.load(
                        open("./flow_matcher_otcfm/unet/controlnet_config.json",
                            'r'))
                    controlnet_config['in_channels'] = 3
                    controlnet_config['encoder_hid_dim_type'] = 'text_proj'
                    controlnet_config['encoder_hid_dim'] = 768
                    controlnet = ControlUNet2DConditionModel(**controlnet_config)
                    controlnet = controlnet.to('cuda')
                    if "controlnet" in state_dict:
                        controlnet.load_state_dict(state_dict['controlnet'], strict=False)
                    self.controlnet = controlnet
                except:
                    self.args.use_controlnet = False
                    self.controlnet = None
                    
            else:
                self.controlnet = None

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
                from train_old_kvasir_Disease import TripletNetwork
                if 'attention.pt' in self.args.wass_model_path:
                    model = TripletNetwork(pretrained=False, freeze_base=False, model='attention')
                else:
                    model = TripletNetwork(pretrained=False, freeze_base=False, model='resnet34')
                state_dict = torch.load(self.args.wass_model_path, weights_only=True)
                model.load_state_dict(state_dict, strict=False)
                model = model.to("cuda")
                self.wass_model = model
                self.wass_model.eval()
            elif 'wass_model' in state_dict:
                from train_old_kvasir_Disease import TripletNetwork
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
                './flow_matcher_otcfm/vae').to(
                device='cuda').eval()
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
        # """Normalize samples to [0, 1] range"""
        # # x = x.clip(-1, 1)
        # x = (x / 2 + 0.5)
        # return x.clip(0, 1)

    def euler_solver(self, x0, t_span, batch):
        """Euler solver with intermediate saves"""
        x = x0
        sample_steps = []
        dt = t_span[1] - t_span[0]
        caption_hidden_states = batch['caption_hidden_states']
        y = batch['y']

        use_gt_vt = self.use_gt_vt
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
                        
                    dx = self.net_model(x, timestep=t.squeeze(), class_labels=y,
                                        encoder_hidden_states=caption_hidden_states, added_cond_kwargs=batch,
                                        image_hint_model='cat',
                                        down_block_additional_residuals=down_block_additional_residuals,
                                        mid_block_additional_residual=mid_block_additional_residual).sample

            x = x + dx * dt
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
                                    mid_block_additional_residual=mid_block_additional_residual).sample
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
                                    mid_block_additional_residual=mid_block_additional_residual).sample
            x = x + (dx1 + dx2) * dt / 2
            if t_idx % self.args.intermediate_freq == 0:
                sample_steps.append(x)
        return sample_steps

    @torch.no_grad()
    def generate_batch(self, batch=None, sample_step=-1):
        """Generate a batch of samples"""
        if batch is None:
            x0 = torch.randn(
                1,
                3,
                self.args.image_size[0],
                self.args.image_size[1],
                device=self.device
            )
            batch['x0'] = x0
        else:
            x0 = batch['x0']

        if self.args.min_num_steps > 0:
            num_steps = random.choice([x for x in range(self.args.num_steps) if x >= self.args.min_num_steps])
        else:
            num_steps = self.args.num_steps

        # Create time steps
        t_span = torch.linspace(0, 1, num_steps, device=self.device)
        # Select solver and generate samples
        solver = self.euler_solver if self.args.solver == 'euler' else self.heun_solver
        if self.use_gt_vt:
            solver = self.euler_solver
        sample_steps = solver(x0, t_span, batch)
        samples = sample_steps[sample_step]
        # samples = self.vae.decode(samples / 0.18215).sample
        # Normalize samples
        # samples = self.normalize_samples(samples)
        return samples

    def generate(self, batch=None, test=False, sample_step=-1):
        if test:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            transform_grey = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.RandomGrayscale(p=1),  # 数据增强：50% 概率灰度化
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            dataloaders = []
            json_paths = glob.glob(
                "./data/mask_data_pairs/*.json")
            for json_path in json_paths:
                dataset = MedicalJsonDataset(
                    path=json_path,
                    transform=transform,
                    hint_transform=transform_grey,
                    transform_A=transform,
                    transform_B=transform,
                )
                if len(dataset) <= self.args.batch_size - 1:
                    continue
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.args.batch_size,
                    shuffle=True,
                    num_workers=0,
                    drop_last=True,
                )
                dataloaders.append(dataloader)

            dataloader = random.choice(dataloaders)
            # dataloader = dataloaders[0]
            batch = dataloader.__iter__().__next__()
        if batch is None:
            raise ValueError("Batch is None.")
        ##############
        x0 = batch['x0'].to(self.vae.device)
        x1 = batch['x1'].to(self.vae.device)
        caption = batch['caption']
        y = batch['class_id'].to(self.vae.device)
        x1_path = batch['x1_path']
        hint = batch['hint']
        if self.args.use_controlnet and 'mask_hint' in batch:
            mask_hint_paths = batch['mask_hint_path']
            mask_hint = batch['mask_hint']
        else:
            # mask_hint = torch.zeros_like(hint) 
            pass
            
        x0_raw = x0.clone()
        x1_raw = x1.clone()
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
        # torch.cuda.empty_cache()
        ### Flow matching core
        if self.args.use_controlnet:
            conds = {
                'x0': x0,
                # 'x0_raw': x0_raw,
                'x1': x1,
                # 'x1_raw': x1_raw,
                'caption': caption,
                'caption_hidden_states': caption_hidden_states if self.text_model is not None else None,
                'y': y,
                "hint": hint.to(x0.device) if self.vision_model is not None else torch.zeros_like(x0_raw),
                "mask_hint": mask_hint.to(x0.device) if self.vision_model is not None else torch.zeros_like(x0_raw),
                'text_embeds': text_embeds[:, 0] if self.text_model is not None else None,  # [B, D]
                'image_embeds': vision_embeds[:, 0] if self.vision_model is not None else None,  # [B, D]
            }
        else:
            conds = {
                'x0': x0,
                # 'x0_raw': x0_raw,
                'x1': x1,
                # 'x1_raw': x1_raw,
                'caption': caption,
                'caption_hidden_states': caption_hidden_states if self.text_model is not None else None,
                'y': y,
                "hint": hint.to(x0.device) if self.vision_model is not None else torch.zeros_like(x0_raw),
                'text_embeds': text_embeds[:, 0] if self.text_model is not None else None,  # [B, D]
                'image_embeds': vision_embeds[:, 0] if self.vision_model is not None else None,  # [B, D]
            }
        for key in batch:
            if key not in conds:
                conds[key] = batch[key]
        
        t, xt, ut = self.FM.get_sample_location_and_conditional_flow(x0, x1, sample_plan=self.args.op_match, cond=conds,
                                                                replace=False, print_info=False)
        conds['ut'] = ut.to(self.device)

        # Generate batch
        samples = self.generate_batch(conds, sample_step=sample_step)
        result['samples'] = samples
        result["x0_vaed"] = x0
        result["x1_vaed"] = x1
        # result["x0_raw"] = conds['x0_raw']
        # result["x1_raw"] = conds['x1_raw']
        # Memory cleanup
        # torch.cuda.empty_cache()
        return result

def generate_parse_args():
    parser = argparse.ArgumentParser(description='Sampling script for CFM model')
    parser.add_argument('--checkpoint', type=str,
                        # default = "./outputs/disease_self_exam_json_endovit_image_hint_wassmodelDXL_neighbor_nfree/otcfm/otcfm_weights_step_30000.pt",
                        # default = './outputs/disease_self_exam_json_endovit_image_hint_resnetmodel_neighbor_nfree/otcfm/otcfm_weights_step_30000.pt',
                        ### All
                        # default = './outputs/disease_self_exam_json_endovit_image_hint_resnetmodel_neighbor_nfree/otcfm/otcfm_weights_step_50000.pt',
                        # default = "./outputs/disease_self_exam_json_endovit_image_hint_wassmodelDXL_neighbor_nfree/otcfm/otcfm_weights_step_50000.pt",
                        default = './outputs/disease_self_exam_json_endovit_image_hint_all/otcfm/otcfm_weights_step_50000.pt',
                        help='Path to the checkpoint file')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512],
                        help='Image size (height, width)')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='Number of steps in the ODE solver')
    parser.add_argument('--min_num_steps', type=int, default=-1,
                        help='Number of steps in the ODE solver')
    parser.add_argument('--num_channels', type=int, default=128,
                        help='Number of base channels in UNet')
    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='Use EMA model for inference')
    parser.add_argument('--solver', type=str, default='heun',
                        choices=['euler', 'heun'],
                        help='ODE solver type')
    parser.add_argument('--intermediate_freq', type=int, default=2,
                        help='Frequency of saving intermediate steps')
    parser.add_argument('--use_controlnet', type=bool, default=False,
                        help='Frequency of saving intermediate steps')
    parser.add_argument('--caption_hidden_states_mode', type=str, default='cat',
                        help='')
    parser.add_argument('--op_match', type=bool, default=False,
                        help='')
    parser.add_argument('--wass_model_path', type=str, 
                        default="./best_flow_weights/model_Disease.pt",
                        help='')
    parser.add_argument('--wass_model_type', type=str, 
                        choices=['resnet34', 'attention'],
                        default="resnet34", # resnet34 或 attention
                        help='指定模型类型')
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