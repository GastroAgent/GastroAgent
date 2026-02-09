# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import logging
import os
import socket
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized
import gc
import torch
import math
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    # Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
    Qwen2Config,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad
# from trl import GRPOTrainer
from trl.extras.profiling import profiling_context, profiling_decorator
from accelerate.utils import is_peft_model, set_seed
from trl.import_utils import is_deepspeed_available, is_liger_kernel_available, is_rich_available, is_vllm_available
from PIL import Image
from vllm import LLM, SamplingParams
from trl.extras.vllm_client import VLLMClient
from llava.model.language_model.llava_qwen import LlavaQwen2ForCausalLM, LlavaConfig
from transformers import AutoConfig
import copy
from torch.utils.data import Sampler
import warnings
from trl import GRPOTrainer
from trl.trainer.grpo_trainer import deepspeed
import re
import PIL

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

from vlm_modules.vlm_module import VLMBaseModule

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
from transformers.utils import logging

logger = logging.get_logger(__name__)


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility.
    """

    def __init__(
            self,
            data_source: Sized,
            mini_repeat_count: int,
            batch_size: int = 1,
            repeat_count: int = 1,
            seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i: i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count

torch_compile_options = {
    "epilogue_fusion": True,
    "max_autotune": False,  # 寻找最优策略会花费大量时间。
    "shape_padding": True,
    "trace.enabled": False,
    "triton.cudagraphs": False,
}

### GSPO 取消了 KL 约束。
@torch.compile(dynamic=True, fullgraph=True, options=torch_compile_options)
def gspo_compute_loss(per_token_logps: torch.Tensor, old_per_token_logps: torch.Tensor, advantages: torch.Tensor,
                      completion_mask: torch.Tensor, epsilon_low=0.1, epsilon_high=0.2):
    log_coef = per_token_logps - old_per_token_logps  # [B, S]
    log_si_theta = (log_coef * completion_mask).sum(-1) / completion_mask.sum(-1)  # [B] 公式(7)
    si_theta = log_si_theta.exp()
    per_seq_loss1 = si_theta * advantages.squeeze()  # advantages: [B, 1]
    per_seq_loss2 = torch.clamp(si_theta, 1 - epsilon_low, 1 + epsilon_high) * advantages.squeeze()
    per_seq_loss = - torch.min(per_seq_loss1, per_seq_loss2)  # [B] 公式(5)

    return per_seq_loss.mean()


@torch.compile(dynamic=True, fullgraph=True, options=torch_compile_options)
def gspo_token_compute_loss(per_token_logps: torch.Tensor, old_per_token_logps: torch.Tensor, advantages: torch.Tensor,
                            completion_mask: torch.Tensor, epsilon_low=0.1, epsilon_high=0.2, dapo=False):
    log_coef = per_token_logps - old_per_token_logps  # [B, S]
    si_theta = ((log_coef * completion_mask).sum(-1) / completion_mask.sum(-1)).exp()  # [B] 公式(7)
    sit_theta = si_theta.detach().unsqueeze(1) * (per_token_logps - per_token_logps.detach()).exp()

    per_seq_loss1 = sit_theta * advantages  # advantages: [B, 1]
    per_seq_loss2 = torch.clamp(sit_theta, 1 - epsilon_low, 1 + epsilon_high) * advantages
    per_seq_loss = - torch.min(per_seq_loss1, per_seq_loss2)  # [B, S] 公式(13)
    if dapo:
        loss = (per_seq_loss * completion_mask).sum() / completion_mask.sum()  # [1]
    else:
        loss_seq = (per_seq_loss * completion_mask).sum(1) / completion_mask.sum(1)  # [B]
        loss = loss_seq.mean()
    return loss


@torch.compile(dynamic=True, fullgraph=True, options=torch_compile_options)
def grpo_compute_loss(per_token_logps: torch.Tensor, old_per_token_logps: torch.Tensor, advantages: torch.Tensor,
                      completion_mask: torch.Tensor,
                      ref_per_token_logps: torch.Tensor = None, epsilon_low=0.1, epsilon_high=0.2, bi_kl=False,
                      use_neg_clamp=True):
    coef_1 = torch.exp(per_token_logps - old_per_token_logps)
    coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high) # 只对 正优势 有用。
    per_token_loss1 = coef_1 * advantages
    per_token_loss2 = coef_2 * advantages
    if use_neg_clamp:
        per_token_loss = - per_token_loss2
    else:
        ###
        ## A < 0, coef_1 < coef_2
        per_token_loss = - torch.min(per_token_loss1, per_token_loss2)

    if ref_per_token_logps is None:
        ref_per_token_logps = old_per_token_logps

    if bi_kl:  # 双向KL
        per_token_kl = (torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
                        + torch.exp(per_token_logps - ref_per_token_logps) - (
                                per_token_logps - ref_per_token_logps) - 1) / 2
    else:  # 二阶近似的 KL, 不可能是 负的。
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        # per_token_kl = torch.clamp(torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1, min=0, max=500)

    with torch.inference_mode():
        # completion_length = completion_mask.sum(dim=1).mean()
        mean_kl_per_reward = (per_token_kl * completion_mask).sum(1) / completion_mask.sum(dim=1)
        mean_kl = ((per_token_kl * completion_mask).sum() / completion_mask.sum())

    return per_token_loss, per_token_kl, mean_kl, mean_kl_per_reward, per_token_loss1, per_token_loss2, coef_1, coef_2


import os
import json
from safetensors.torch import save_file
import torch


def save_model_with_custom_sharding(model, output_dir, shards_size_gb=2):
    """
    根据指定大小保存模型为多个 .safetensors 分片，并生成相应的 .safetensors.index.json 文件。

    :param model: 要保存的 PyTorch 模型实例
    :param output_dir: 输出目录路径
    :param shards_size_gb: 每个分片的最大大小（GB）
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    weight_map = {}
    current_shard_weights = {}
    current_shard_size = 0
    shard_index = 0

    # 将模型参数转换为字典格式
    state_dict = model.state_dict()

    for key, param in state_dict.items():
        tensor_size_mb = param.element_size() * param.nelement() / (1024 ** 2)
        if current_shard_size + tensor_size_mb > shards_size_gb * 1024:
            # 保存当前分片
            shard_file_path = os.path.join(output_dir, f"model-{shard_index:05d}-of-{len(state_dict)}.safetensors")
            save_file(current_shard_weights, shard_file_path)

            # 更新 weight_map
            for k in current_shard_weights.keys():
                weight_map[k] = os.path.basename(shard_file_path)

            # 开始新的分片
            shard_index += 1
            current_shard_weights = {}
            current_shard_size = 0

        current_shard_weights[key] = param
        current_shard_size += tensor_size_mb

    # 保存最后一个分片
    if current_shard_weights:
        shard_file_path = os.path.join(output_dir, f"model-{shard_index:05d}-of-{len(state_dict)}.safetensors")
        save_file(current_shard_weights, shard_file_path)

        # 更新 weight_map
        for k in current_shard_weights.keys():
            weight_map[k] = os.path.basename(shard_file_path)

    # 创建 .safetensors.index.json 文件
    index = {
        "metadata": {},
        "weight_map": weight_map
    }

    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)


def crop_right_square(image: Image.Image, target_weight_size: int = 1300, target_height_size=1080) -> Image.Image:
    """
    从图像右侧裁剪出一个正方形区域（target_size x target_size）。

    参数:
        image (Image.Image): 输入的 PIL 图像对象。
        target_size (int): 裁剪目标区域的边长（默认为 1080）。

    返回:
        Image.Image: 裁剪后的图像对象。

    异常:
        ValueError: 如果图像尺寸不足以裁剪出目标区域。
    """
    # 获取图像尺寸
    width, height = image.size

    # 检查图像尺寸是否满足裁剪需求
    if width < target_weight_size or height < target_height_size:
        return image

    # 计算裁剪区域（右上角的正方形）
    left = width - target_weight_size
    top = 0
    right = width
    bottom = target_height_size

    # 执行裁剪
    cropped_image = image.crop((left, top, right, bottom))

    return cropped_image


def add_eos_token(completion_ids, eos_token_id, pad_token_id=0):
    """
    为 completion_ids 中的每个序列在有效内容末尾添加 EOS token。

    Args:
        completion_ids (torch.Tensor): 形状为 (batch_size, seq_len) 的张量，包含 token IDs。
        eos_token_id (int): EOS token 的 ID。
        pad_token_id (int, optional): PAD token 的 ID。默认为 0。

    Returns:
        torch.Tensor: 新的张量，形状可能为 (batch_size, seq_len + 1)，其中已在每个序列的有效末尾添加了 EOS token。
    """
    batch_size, seq_len = completion_ids.shape
    device = completion_ids.device

    # 创建一个新张量，长度比原序列多1，用于存放结果
    new_completion_ids = torch.full((batch_size, seq_len + 1), pad_token_id, dtype=completion_ids.dtype, device=device)

    for i in range(batch_size):
        seq = completion_ids[i]

        # 查找 PAD token 的位置
        pad_mask = (seq == pad_token_id)

        if pad_mask.any():
            # 如果存在 PAD token
            # 找到最后一个 PAD token 的位置（即第一个 PAD 的索引）
            # 注意：我们通常假设 PAD 是连续的，并且从某个点开始到序列末尾
            first_pad_idx = (seq == pad_token_id).nonzero(as_tuple=True)[0][0].item()
            # 在最后一个有效 token（即第一个 PAD 之前）的位置插入 EOS
            insert_pos = first_pad_idx

            # 将原序列中 PAD 之前的内容复制到新序列
            new_completion_ids[i, :insert_pos] = seq[:insert_pos]
            # 在 insert_pos 位置插入 EOS
            new_completion_ids[i, insert_pos] = eos_token_id
            # insert_pos 之后的位置保持为 PAD (由 torch.full 初始化保证)

        else:
            # 如果没有 PAD token，则直接在末尾添加 EOS
            new_completion_ids[i, :-1] = seq  # 复制整个原序列
            new_completion_ids[i, -1] = eos_token_id  # 在最后添加 EOS

    return new_completion_ids


class VLMGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            args: GRPOConfig = None,
            vlm_module: VLMBaseModule = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
                    None, None),
            peft_config: Optional["PeftConfig"] = None,
            freeze_vision_modules: Optional[bool] = False,
            freeze_mm_mlp_adapter: Optional[bool] = False,
            freeze_backbone: Optional[bool] = False,
            backbone_layer_id: Optional[int] = -1,
            attn_implementation: str = "flash_attention_2",
            torch_dtype: str = "bfloat16",
            **kwargs,
    ):
        self.alg = kwargs.pop('alg', 'grpo')
        self.grpo_denorm_length = kwargs.pop('grpo_denorm_length', 4096)
        self.bi_kl = kwargs.pop('bi_kl', False)
        self.patch_size = kwargs.pop('patch_size', 14)
        self.prune_threshold = kwargs.pop('prune_threshold', 0)
        self.prune_ratio = kwargs.pop('prune_ratio', 0)
        # self.model_args = kwargs.pop('model_args', None)
        self.advantages_clip_up = kwargs.pop('advantages_clip_up', 0)
        self.advantages_clip_down = kwargs.pop('advantages_clip_down', 0)
        self.use_advantages_clip = kwargs.pop('use_advantages_clip', False)
        self.decouple_adv = kwargs.pop('decouple_adv', False)
        self.high_entropy = kwargs.pop('high_entropy', False)
        self.use_neg_clamp = kwargs.pop('use_neg_clamp', False)
        self.vllm_mode = kwargs.pop('vllm_mode', "colocate")
        self.vllm_tensor_parallel_size = kwargs.pop('vllm_tensor_parallel_size', 1)
        print('self.decouple_adv: ', self.decouple_adv)
 
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        self.vlm_module = vlm_module
        self.vlm_module.is_message = kwargs.pop('is_message', False)
        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        # FIXME
        # Remember to modify it in the invernvl
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype

        assert isinstance(model, str), "model must be a string in the current implementation"
        model_id = model
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # Disable caching if gradient checkpointing is enabled (not supported)
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )

        model_config = AutoConfig.from_pretrained(model_id,
                                                  #  dtype=torch.bfloat16,
                                                  **model_init_kwargs)

        model_cls = self.vlm_module.get_model_class(model_id, model_init_kwargs)
        model = model_cls.from_pretrained(model_id, **model_init_kwargs)

        self.vision_modules_keywords = self.vlm_module.get_vision_modules_keywords()
        self.mm_projector_keywords = self.vlm_module.get_mm_projector_keywords()
        if freeze_backbone:
            model.model.requires_grad_(False)

        # LoRA
        if peft_config is not None:
            if os.environ.get("RANK", -1) == '0':
                print("Applying LoRA...")

            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    # LoRA is not applied to the vision modules
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in lora_module_names:  # needed for 16-bit
                    if "embed_tokens" in m:
                        lora_module_names.remove(m)
                return list(lora_module_names)

            if peft_config.target_modules is None:
                target_modules = find_all_linear_names(model, self.vision_modules_keywords)
            else:
                if isinstance(peft_config.target_modules, str):
                    target_modules = [x.strip() for x in peft_config.target_modules.split(',')]
                else:
                    target_modules = peft_config.target_modules

            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)

        # Freeze vision modules
        if freeze_vision_modules and self.mm_projector_keywords is not None:
            if os.environ.get("RANK", -1) == '0':
                print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False

        if peft_config is not None and not freeze_vision_modules and self.mm_projector_keywords is not None:
            for key, p in model.get_model().vision_tower.named_parameters():
                if 'lora_' in key and 'vision_tower' in key:
                    p.requires_grad = True
                elif 'lora_' not in key and 'vision_tower' in key:
                    p.requires_grad = False

        if freeze_mm_mlp_adapter and self.mm_projector_keywords is not None:
            if os.environ.get("RANK", -1) == '0':
                print("Freezing mm_mlp modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.mm_projector_keywords):
                    p.requires_grad = False

        if not freeze_mm_mlp_adapter and self.mm_projector_keywords is not None:
            if os.environ.get("RANK", -1) == '0':
                print("Unfreezing mm_mlp modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.mm_projector_keywords):
                    p.requires_grad = True

        if peft_config is not None and freeze_backbone and self.mm_projector_keywords is not None:
            for key, p in model.get_model().named_parameters():
                if 'layers.' in key and 'vision_tower' not in key:
                    p.requires_grad = False

        if not freeze_backbone and backbone_layer_id > 1 and self.mm_projector_keywords is not None:
            for key, p in model.get_model().named_parameters():
                if 'layers.' in key and 'vision_tower' not in key:
                    layers_id = re.findall(r"layers\.(.*?)\.", key)[-1]
                    if layers_id % backbone_layer_id == 0:
                        if peft_config is not None:  # 只调整 Lora。
                            if 'lora_' not in key:  # 是否为 Lora 架构。
                                continue
                        p.requires_grad = True
                    else:
                        pass

        # Compute the number of trainable parameters and print the parameter that is trainable
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in trainable_params)
    
        if os.environ.get("RANK", -1) == '0':
            print(f"Total trainable parameters: {total_params}")
            print(f"Trainable parameters: {trainable_names}")

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = model_cls.from_pretrained(model_id, **model_init_kwargs)
            # self.ref_model = LlavaQwen2ForCausalLM(config)
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        # Processing class
        pad_token_id = None
        if processing_class is None:
            processing_cls = self.vlm_module.get_processing_class()
            processing_class = processing_cls.from_pretrained(model_id,
                                                              use_fast=True,
                                                              trust_remote_code=model_init_kwargs.get(
                                                                  "trust_remote_code", None),
                                                              patch_size=self.patch_size)
            for component, processing_keyword in self.vlm_module.get_custom_processing_keywords():
                if processing_keyword in kwargs:
                    # If we cannot find component in processing_class, return the processing_class itself
                    processing_component = getattr(processing_class, component, processing_class)
                    setattr(processing_component, processing_keyword, kwargs[processing_keyword])
            if getattr(processing_class, "tokenizer", None) is not None:
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            else:
                assert isinstance(processing_class,
                                  PreTrainedTokenizerBase), "processing_class must be an instance of PreTrainedTokenizerBase if it has no tokenizer attribute"
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        # self.max_prompt_length = None
        if args.max_prompt_length is not None:
            warnings.warn("Setting max_prompt_length is currently not supported, it has been set to None")

        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper

        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            top_k=30,
            pad_token_id=pad_token_id,
            repetition_penalty=1.2,
            use_cache=True
        )

        if not hasattr(self.generation_config, "image_token"):
            try:
                self.generation_config.image_token = processing_class.image_token
                self.generation_config.image_token_id = processing_class.tokenizer.convert_tokens_to_ids(
                    processing_class.image_token)
            except:
                pass
        if hasattr(self.vlm_module, "get_eos_token_id"):  # For InternVL
            self.generation_config.eos_token_id = self.vlm_module.get_eos_token_id(processing_class)

        self.beta = args.beta
        self.last_beta = self.beta
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        if (self.alg == 'gspo_token') and self.num_iterations > 1:
            print('when use gspo_token, self.num_iterations can greater than 1.')
            # self.num_iterations = 1

        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        self.collected_dataset = []

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)
        self.use_vllm = args.use_vllm
        self.args.vllm_enable_sleep_mode = False
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install trl[vllm]` to use it."
                )

            if self.vllm_mode == "server":
                if self.accelerator.is_main_process:
                    # self.use_vllm = False
                    if getattr(args, 'vllm_server_base_url', None) is not None:
                        base_url = args.vllm_server_base_url
                    else:
                        base_url = f"http://{args.vllm_server_host}:{args.vllm_server_port}"
                    self.vllm_client = VLLMClient(host=args.vllm_server_host, server_port=args.vllm_server_port, connection_timeout=args.vllm_server_timeout)
                    # self.vllm_client.init_communicator(device=torch.cuda.current_device())
                    self.vllm_client.init_communicator()
        #                     >>> client = VLLMClient()
        # >>> client.generate(["Hello, AI!", "Tell me a joke"])
        # [[2980, 498, 1492, 752, 448, 264, 13027, 8645, 30, 358, 2776, 4460, 311, 3270, 264, 2025],
        #  [911, 7988, 1251, 382, 3838, 653, 498, 1618, 4325, 879, 2581, 20027, 264, 21428, 30, 362]]

        # >>> from transformers import AutoModelForCausalLM

        # >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="cuda")
        # >>> client.init_communicator(device="cuda")
        # >>> client.update_model_params(model)

            elif self.vllm_mode == "colocate":
                # Make sure vllm_tensor_parallel_size group size evenly divides the world size - each group should have the same number of ranks.
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                    # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)

                # os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
                # os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12345")

                # Ensure distributed rendezvous variables are set without colliding across concurrent runs
                def _is_port_free(port: int, host: str = "127.0.0.1") -> bool:
                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                            s.bind((host, port))
                            return True
                    except OSError:
                        return False
                def _find_free_port() -> int:
                    candidates = (29500, 23456, 12355, 12345)
                    for p in candidates:
                        if _is_port_free(p):
                            return p
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(("", 0))
                        return s.getsockname()[1]
                    
                def ensure_master_addr_port(addr: Optional[str] = None, port: Optional[int] = None) -> None:
                    """
                    Ensure `MASTER_ADDR`/`MASTER_PORT` are set safely.

                    - Respects existing environment variables.
                    - Defaults `MASTER_ADDR` to localhost if unset.
                    - Chooses a free TCP port if `MASTER_PORT` is unset to avoid collisions.
                    - If `MASTER_PORT` is set to `"0"` or `"auto"`, it is resolved to a free port.
                    """
                    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR") or addr or "localhost"

                    env_port = os.environ.get("MASTER_PORT", "").strip().lower()
                    if port is None and env_port not in {"", "0", "auto"}:
                        try:
                            port = int(env_port)
                        except ValueError:
                            pass

                    os.environ["MASTER_PORT"] = str(_find_free_port() if port in (None, 0) else port)

                ensure_master_addr_port()

                if self.max_prompt_length is not None and self.max_completion_length is not None:
                    max_model_len = self.max_prompt_length + self.max_completion_length
                else:
                    max_model_len = None
                self.llm = LLM(
                    model=model.name_or_path,
                    tensor_parallel_size= self.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    # max_num_seqs=self.args.per_device_train_batch_size * self.vllm_tensor_parallel_size * self.args.num_iterations,
                    max_model_len=max_model_len,
                    distributed_executor_backend="external_launcher",
                    ### 告诉 vLLM：分布式进程已经由外部工具（如 torchrun, deepspeed, Slurm, Kubernetes`）启动好了，
                    ### 你只需“加入”这个分布式环境，不要自己启动子进程。

                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    # Latest vLLM v1 memory profiler is misled by the high default value (i.e., 32768) - thinking there's not enough memory
                    max_num_batched_tokens=32768,
                    # model_impl=self.args.vllm_model_impl,
                    enable_sleep_mode=self.args.vllm_enable_sleep_mode,
                    # Important so temperature scaling/logit tweaking affects the TIS log probs
                    # logprobs_mode="processed_logprobs",
                )
                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)
            else:
                raise ValueError(f"vllm_mode must be either 'server' or 'colocate', got '{self.vllm_mode}'.")
                # pass

            # vLLM specific sampling arguments
            self.guided_decoding_regex = args.vllm_guided_decoding_regex

            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()

        self.question_types_state_dict = {}
        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            # if self.is_deepspeed_enabled:
            if is_deepspeed_zero3_enabled():
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
            # model.vision_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            if getattr(model, "language_model", None) is not None:
                # For InternVL; these operations are copied from the original training script of InternVL
                model.language_model.config.use_cache = False
                model.vision_model.gradient_checkpointing = True
                model.vision_model.encoder.gradient_checkpointing = True
                model.language_model._set_gradient_checkpointing()
                # This line is necessary, otherwise the `model.gradient_checkpointing_enable()` will be executed during the training process, leading to an error since InternVL does not support this operation.
                args.gradient_checkpointing = False
            else:
                model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
                "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, completion_ids, name, logits_to_keep=None, **custom_multimodal_inputs):
        if logits_to_keep is not None:
            logits_to_keep = logits_to_keep - 1
        else:
            promt_completion_length = input_ids.shape[1]
            completion_length = completion_ids.shape[1]
            logits_to_keep = promt_completion_length - completion_length - 1
        try:
            if self.mm_projector_keywords is not None:
                logits = model(input_ids=input_ids, attention_mask=attention_mask, **custom_multimodal_inputs).logits  # (B, L, V)
                batch_num_images = custom_multimodal_inputs['batch_num_images']
                assert (sum(batch_num_images) / batch_num_images[0]) == len(batch_num_images), "batch_num_images 元素不同未实现。"
                logits = logits[:, logits_to_keep + batch_num_images[0] * 575:] # [B, C+1, D]
            else:
                if 'batch_num_images' in custom_multimodal_inputs:
                    batch_num_images = custom_multimodal_inputs.pop('batch_num_images')
                logits = model(input_ids=input_ids, attention_mask=attention_mask, **custom_multimodal_inputs).logits
                logits = logits[:, logits_to_keep + batch_num_images[0] * 575:]
        except Exception as e:
            if 'batch_num_images' in custom_multimodal_inputs:
                batch_num_images = custom_multimodal_inputs.pop('batch_num_images')
            logits = model(input_ids=input_ids, attention_mask=attention_mask, **custom_multimodal_inputs).logits
            logits = logits[:, logits_to_keep + batch_num_images[0] * 575:]
        logits = logits[:, :-1, :] # (B, C, V), exclude the last logit: it corresponds to the next token pred
        labels = completion_ids # (B, C, V)

        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, labels):
            log_probs = logits_row.log_softmax(dim=-1)
            if len(log_probs) != len(input_ids_row):
                print(name)
                print('input_ids shape: ', input_ids.shape)
                print('attention_mask shape: ', attention_mask.shape)
                print('completion_ids shape: ', completion_ids.shape)
                print('multimodal_inputs batch_num_images: ', custom_multimodal_inputs['batch_num_images'])
                print('multimodal_inputs pixel_values: ', custom_multimodal_inputs['pixel_values'].shape)
                print('input_ids shape in get_per_token_logps: ', input_ids.shape)
                print('logits shape in get_per_token_logps: ', logits.shape)
                print('logits_to_keep: ', logits_to_keep)
                min_length = min([log_probs.shape[0], input_ids_row.shape[0]])
                ### 保留前面的。
                log_probs = log_probs[:min_length]
                input_ids_row = input_ids_row[:min_length]
                ### 保留后面的。
                # log_probs = log_probs[-min_length:]
                # input_ids_row = input_ids_row[-min_length:]
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def _prepare_inputs(self, inputs):
        # Simple pass-through, just like original
        return inputs

    def _get_key_from_inputs(self, x, key):
        ele = x.get(key, None)
        assert ele is not None, f"The key {key} is not found in the input"
        if isinstance(ele, list):
            return ele
        else:
            return [ele]

    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]], model) -> dict[
        str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        # for _ in range(len(inputs)):
        #     inputs[_]['ref_responses'] = '<answer>A</answer>'
        
        # for key in inputs[0]:
        #     if isinstance(inputs[0][key], torch.Tensor):
        #         continue
        #     else:
        #         print(key)
        #         print(inputs[0][key])
        #         print('#'*50)

        if all([True if ('ref_responses' in x and x['ref_responses'] != '') else False for x in inputs]) and len(
                inputs) == self.num_generations:
            # 当 len(inputs) == self.num_generations 时，每个 ref_response 都是 一样的
            ref_responses = inputs[0]['ref_responses']
            if isinstance(ref_responses, str):
                ref_responses = [ref_responses.replace(self.processing_class.tokenizer.eos_token,
                                                       '') + self.processing_class.tokenizer.eos_token]
                # ref_responses = [ref_responses.replace(self.processing_class.tokenizer.eos_token, '')]
            else:
                try:
                    ref_responses = [
                        x.replace(self.processing_class.tokenizer.eos_token, '') + self.processing_class.tokenizer.eos_token
                        for x in ref_responses]
                except:
                    ref_responses = [
                        x.replace(self.processing_class.eos_token, '') + self.processing_class.eos_token
                        for x in ref_responses]
                    
        else:
            ref_responses = []

        assert self.num_generations % len(inputs) == 0, f"暂时仅支持 num_generations % batchSize_per_gpu == 0 ."
        num_inputs = len(inputs) - len(ref_responses)
        inputs = inputs[:num_inputs]
        prompts = [x["prompt"] for x in inputs]
        prompts_text = self.vlm_module.prepare_prompt(self.processing_class, inputs)
        # Handle both pre-loaded images and image paths
        images = []
     
        batch_num_images = [0 for _ in prompts]

        for idx, x in enumerate(inputs):
            if "image" in x and isinstance(x["image"], str):
                imgs = self._get_key_from_inputs(x, "image")
            elif "image" in x and isinstance(x["image"], list):
                try:
                    imgs = [PIL.Image.open(p).convert('RGB') for p in self._get_key_from_inputs(x, "image")]
                except:
                    print('Images Load Error!')
                    imgs = [PIL.Image.open('/home/work/ds-vl-on-policy_test/01.2009061500040_3.jpg') for p in
                            self._get_key_from_inputs(x, "image")]
            elif "image_paths" in x and x["image_paths"] is not None:
                try:
                    imgs = [PIL.Image.open(p).convert('RGB') for p in self._get_key_from_inputs(x, "image_paths")]
                except:
                    print('Images Load Error!')
                    imgs = [PIL.Image.open('/home/work/ds-vl-on-policy_test/01.2009061500040_3.jpg') for p in
                            self._get_key_from_inputs(x, "image_paths")]
            elif "image_path" in x and x["image_path"] is not None:
                imgs = [PIL.Image.open(p).convert('RGB') for p in self._get_key_from_inputs(x, "image_path")]
            else:
                imgs = []
            pass

            for img in imgs:
                # img = crop_right_square(img)

                try:
                    # Ensure minimum dimensions of 28 pixels
                    w, h = img.size
                    if w < 28 or h < 28:
                        # Calculate new dimensions maintaining aspect ratio
                        if w < h:
                            new_w = 28
                            new_h = int(h * (28 / w))
                        else:
                            new_h = 28
                            new_w = int(w * (28 / h))
                        img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
                except Exception as e:
                    print(e)
                    pass
                batch_num_images[idx] += 1
                images.append(img)

        prompt_inputs = self.vlm_module.prepare_model_inputs(
            self.processing_class,
            prompts_text,
            images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        # ### max_prompt_length is not supported yet
        # if self.max_prompt_length is not None:
        #     prompt_ids = prompt_ids[:, -self.max_prompt_length :]
        #     prompt_inputs["input_ids"] = prompt_ids
        #     prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        #     prompt_inputs["attention_mask"] = prompt_mask

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            if self.mm_projector_keywords is not None:
                raise NotImplementedError('虽然vllm支持多模态，但是自定义的 vllm 并未支持。')
            
            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm() # vllm == 0.7.2
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                has_images = len(images) > 0
                all_prompts_text = gather_object(prompts_text)
                if has_images:
                    all_images = gather_object(images)

                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]

                    if has_images:
                        ordered_set_of_images = all_images[:: self.num_generations]
                    else:
                        ordered_set_of_images = None

                    with profiling_context(self, "vLLM.generate"):
                        completion_ids = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            images=ordered_set_of_images,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                else:
                    completion_ids = [None] * len(all_prompts_text)
                # Broadcast the completions from the main process to all processes, ensuring each process receives its
                # corresponding slice.
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                if self.args.vllm_enable_sleep_mode:
                    # wake up colocated vLLM instances if needed
                    # torch.cuda.empty_cache()  # required to avoid OOM in some cases
                    self.llm.wake_up()

            
                if self.guided_decoding_regex:
                    from vllm.sampling_params import GuidedDecodingParams
                    guided_decoding = GuidedDecodingParams(regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None

                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": 1.05,
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "top_k": -1 ,
                    "min_p": 0.0,
                    "max_tokens": self.max_completion_length,
                    "guided_decoding": guided_decoding,
                    "logprobs": 0,  # only return the logprob of the generated token
                }
                
                if getattr(self.args, 'generation_kwargs', None) is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]

                    if images is not None and len(images) > 0:
                        gathered_images = [None for _ in range(self.vllm_tensor_parallel_size)]
                        torch.distributed.all_gather_object(gathered_images, images, group=self.tp_group)
                        all_images = [img for sublist in gathered_images for img in sublist]
                    else:
                        all_images = None
                else:
                    all_prompts_text = prompts_text
                    all_images = images

                if images is not None and all_images:
                    vllm_inputs = []
                    for prompt, image_list in zip(all_prompts_text, all_images):
                        vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image_list}})

                else:
                    vllm_inputs = all_prompts_text
                    
                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)

                all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
                # all_logprobs = [
                #     [next(iter(lp.values())).logprob for lp in output.logprobs]
                #     for outputs in all_outputs
                #     for output in outputs.outputs
                # ]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = all_completion_ids[tp_slice]
                    # logprobs = all_logprobs[tp_slice]
                else:
                    completion_ids = all_completion_ids
                    # logprobs = all_logprobs

                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)
        
                completion_ids = [torch.Tensor(completion_id).to(torch.long).to(prompt_ids.device) for completion_id in completion_ids]
                completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
        else:
            # Generate completions
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                if self.mm_projector_keywords is not None:
                    unwrapped_model.eval()
                    # unwrapped_model.config.use_cache = True
                    # unwrapped_model.gradient_checkpointing = False
                    num_per_infer = 4
                else:
                    # if self.max_completion_length >= 2048:
                    #     num_per_infer = 2
                    # elif self.max_completion_length < 2048 and self.max_completion_length >= 1024:
                    #     num_per_infer = 4
                    # else:
                    num_per_infer = 12
                    unwrapped_model.eval()
                    unwrapped_model.config.use_cache = True
                    unwrapped_model.gradient_checkpointing = False

                if len(prompt_ids) > num_per_infer:
                    infer_count = len(prompt_ids) // num_per_infer
                    generate_returned_result = []
                    for i in range(0, infer_count):
                        prompt_inputs_tmp = {k: v[i * num_per_infer:i * num_per_infer + num_per_infer, ...] for k, v in
                                             prompt_inputs.items() if
                                             k not in self.vlm_module.get_non_generate_params()}
                        if 'im_grid_hw' in prompt_inputs:
                            prompt_inputs_tmp['im_grid_hw'] = prompt_inputs_tmp['im_grid_hw']

                        if self.mm_projector_keywords is not None:
                            generate_returned_result_tmp = unwrapped_model.generate(
                                **{k: v for k, v in prompt_inputs_tmp.items() if
                                   k not in self.vlm_module.get_non_generate_params()},
                                generation_config=self.generation_config,
                                batch_num_images=batch_num_images[i * num_per_infer:i * num_per_infer + num_per_infer],
                                use_cache=True
                            )
                            generate_returned_result.extend([x for x in generate_returned_result_tmp])
                        else:  ### Qwen-VL
                            generate_returned_result_tmp = unwrapped_model.generate(
                                **{k: v for k, v in prompt_inputs_tmp.items() if
                                   k not in self.vlm_module.get_non_generate_params()},
                                generation_config=self.generation_config,
                                use_cache=True
                            )
                            generate_returned_result.extend([x for x in generate_returned_result_tmp])

                        # # torch.cuda.empty_cache()
                    # 补余数
                    res = len(prompt_ids) % num_per_infer
                    if res > 0:
                        prompt_inputs_tmp = {k: v[-res:, ...] for k, v in prompt_inputs.items() if
                                             k not in self.vlm_module.get_non_generate_params()}
                        if self.mm_projector_keywords is not None:
                            generate_returned_result_tmp = unwrapped_model.generate(
                                **{k: v for k, v in prompt_inputs_tmp.items() if
                                   k not in self.vlm_module.get_non_generate_params()},
                                generation_config=self.generation_config,
                                batch_num_images=batch_num_images[-res:],
                                use_cache=True
                            )
                            generate_returned_result.extend([x for x in generate_returned_result_tmp])
                        else:
                            print('Qwen 分块推理。')
                            generate_returned_result_tmp = unwrapped_model.generate(
                                **{k: v for k, v in prompt_inputs_tmp.items() if
                                   k not in self.vlm_module.get_non_generate_params()},
                                generation_config=self.generation_config,
                                use_cache=True
                            )
                            generate_returned_result.extend([x for x in generate_returned_result_tmp])
                            # # torch.cuda.empty_cache()
                    generate_returned_result = pad(generate_returned_result,
                                                   padding_value=self.processing_class.pad_token_id)
                    # if self.accelerator.is_main_process:
                    # print('Batch Size: ', len(generate_returned_result))
                else:
                    if self.mm_projector_keywords is not None:
                        generate_returned_result = unwrapped_model.generate(
                            **{k: v for k, v in prompt_inputs.items() if
                               k not in self.vlm_module.get_non_generate_params()},
                            generation_config=self.generation_config,
                            batch_num_images=batch_num_images,
                            use_cache=True
                        )
                    else:
                        unwrapped_model = unwrapped_model.to(torch.bfloat16)
                        for k in prompt_inputs:
                            if isinstance(prompt_inputs[k], torch.FloatTensor) and not isinstance(prompt_inputs[k], torch.Long):
                                prompt_inputs[k] = prompt_inputs[k].to(torch.bfloat16)
                        generate_returned_result = unwrapped_model.generate(
                            **{k: v for k, v in prompt_inputs.items() if
                               k not in self.vlm_module.get_non_generate_params()},
                            generation_config=self.generation_config,
                            use_cache=True
                        )
                        # # torch.cuda.empty_cache()
                # if self.mm_projector_keywords is not None:
                unwrapped_model.train()
                unwrapped_model.gradient_checkpointing = True
                unwrapped_model.config.use_cache = False
                unwrapped_model.gradient_checkpointing_enable()
                prompt_length = prompt_ids.size(1)

                if not self.vlm_module.is_embeds_input():
                    prompt_completion_ids = generate_returned_result
                    prompt_ids = prompt_completion_ids[:, :prompt_length]
                    completion_ids = prompt_completion_ids[:, prompt_length:]
                else:
                    # In this case, the input of the LLM backbone is the embedding of the combination of the image and text prompt
                    # So the returned result of the `generate` method only contains the completion ids
                    completion_ids = generate_returned_result

        if ref_responses:
            ref_prompt_ids = prompt_ids[0:1].repeat(len(ref_responses), 1)
            ref_prompt_mask = prompt_mask[0:1].repeat(len(ref_responses), 1)
            try:
                ref_completion_ids = \
                self.processing_class.tokenizer.batch_encode_plus(ref_responses, padding=True, truncation=True,
                                                              return_tensors='pt')['input_ids']
            except:
                ref_completion_ids = \
                self.processing_class.batch_encode_plus(ref_responses, padding=True, truncation=True,
                                                              return_tensors='pt')['input_ids']
            ref_completion_ids = ref_completion_ids[:, :self.max_completion_length]
            # prompt_ids = pad([x for x in prompt_ids] + [x for x in ref_prompt_ids], self.processing_class.pad_token_id,
            #                  'left')
            prompt_ids = torch.cat([prompt_ids, ref_prompt_ids])
            prompt_mask = torch.cat([prompt_mask, ref_prompt_mask])
            completion_ids = pad([x for x in completion_ids] + [x for x in ref_completion_ids],
                                 self.processing_class.pad_token_id, 'right')

        # if completion_ids.shape[1] == self.max_completion_length:
        #     completion_ids = add_eos_token(completion_ids[:, :-1], self.processing_class.eos_token_id, self.processing_class.pad_token_id)

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        prompt_length = prompt_ids.size(1)
        if self.accelerator.is_main_process:
            if ref_responses:
                print('ref_responses: ', ref_responses[0])
                # print('ref_completion_ids: ', ref_completion_ids[0][-1])
            print('Generate Batch Size: ', len(completion_ids))
            print('prompt_completion_ids shape: ', prompt_completion_ids.shape)
            print('prompt_ids shape: ', prompt_ids.shape)
            print('completion_ids shape: ', completion_ids.shape)
            print('prompt_mask shape: ', prompt_mask.shape)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        if self.accelerator.is_main_process:
            print('attention_mask shape: ', attention_mask.shape)
            print('completion_mask shape: ', completion_mask.shape)
        # Get the multimodal inputs
        multimodal_keywords = self.vlm_module.get_custom_multimodal_keywords()
        if ref_responses:
            if self.mm_projector_keywords is not None:
                for key in prompt_inputs:
                    for _ in range(len(ref_responses)):
                        prompt_inputs[key] = torch.cat([prompt_inputs[key], prompt_inputs[key][0:1]], dim=0)
            else:
                keys = list(prompt_inputs.keys())
                if 'image_grid_thw' in keys and 'pixel_values' in keys:
                    num_embeds_per = torch.prod(prompt_inputs['image_grid_thw'][0]).item()
                    for _ in range(len(ref_responses)):
                        prompt_inputs['pixel_values'] = torch.cat([prompt_inputs['pixel_values'], prompt_inputs['pixel_values'][0:num_embeds_per]], dim=0)

                for key in keys:
                    if 'pixel_values' == key:
                        continue
                    
                    for _ in range(len(ref_responses)):
                        prompt_inputs[key] = torch.cat([prompt_inputs[key], prompt_inputs[key][0:1]], dim=0)
                        
        batch_num_images += [batch_num_images[0]] * len(ref_responses)
        multimodal_inputs = {k: prompt_inputs[k] if k in prompt_inputs else None for k in multimodal_keywords}

        if self.mm_projector_keywords is not None:
            multimodal_inputs.update({'batch_num_images': batch_num_images})

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1 and self.alg != 'gspo_token':
                # if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    model, prompt_completion_ids, attention_mask, completion_ids, '-----old_per_token_logps-----', logits_to_keep = prompt_ids.shape[1],
                    **multimodal_inputs,
                )
                # old_per_token_logps = old_per_token_logps[:, prompt_length - 1:]
                # # torch.cuda.empty_cache()
                # gc.collect()
            else:
                old_per_token_logps = None

            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, completion_ids,
                    '-----ref_per_token_logps-----', logits_to_keep = prompt_ids.shape[1], **multimodal_inputs
                )
            else:
                if 'gspo' in self.alg: # gspo 时，KL 无效？
                    ref_per_token_logps = None
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            model, prompt_completion_ids, attention_mask, completion_ids,
                            '-----ref_per_token_logps-----', logits_to_keep = prompt_ids.shape[1],
                            **multimodal_inputs,
                        )
                    # ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
            # # torch.cuda.empty_cache()
            # gc.collect()

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if self.collected_dataset is not None:
            collected_data_templete = {
                "image": inputs[0]["image"] if 'image' in inputs[0] else None,
                "ref_responses": inputs[0]["ref_responses"] if 'ref_responses' in inputs[0] else None,
                "problem": inputs[0]["problem"] if 'problem' in inputs[0] else None,
                "question": inputs[0]["problem"] if 'question' in inputs[0] else None,
                "prompt": inputs[0]["prompt"] if 'prompt' in inputs[0] else None,
                "answer": inputs[0]["answer"] if 'answer' in inputs[0] else None,
                "gt_answer": inputs[0]["gt_answer"] if 'gt_answer' in inputs[0] else None,
                "question_id": inputs[0]["question_id"] if 'question_id' in inputs[0] else None,
            }
            qas = self.processing_class.batch_decode(prompt_completion_ids, skip_special_tokens=False)
            for qa in qas:
                collected_data = collected_data_templete.copy()
                collected_data["rl_sample_qa"] = qa
                self.collected_dataset.append(collected_data)

            if len(self.collected_dataset) % 1000 == 0:
                with open(self.args.output_dir + '/rl_sample_dataset.json', "w") as f:
                    json.dump(self.collected_dataset, f, indent=4, ensure_ascii=False)

        completions = [x.replace('<|Image_start|><image><|Image_end|>', '<image>') for x in completions]

        if self.accelerator.is_main_process:
            print(f'prompts_text: {prompts_text[0]}')
            print(f'completion: {completions[0]}')
            print(f"Answer: {inputs[0]['answer']}")
            # print(inputs[0])
        # print(f'prompts_text: {prompts_text[0]}')
        # print(f'completion: {completions[0]}')
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
        # Compute the rewards
        # No need to duplicate prompts as we're not generating multiple completions per prompt

        rewards_per_func = torch.zeros(len(completions), len(self.reward_funcs), device=device)
        num_refs = len(ref_responses)
        for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    example = inputs[0]
                    for _ in range(len(completions)):
                        # No need to duplicate prompts as we're not generating multiple completions per prompt
                        # reward_kwargs[key].extend([example[key]] * len(prompt_ids))
                        reward_kwargs[key].extend([example[key]])
                output_reward_func = reward_func(prompts=[prompts[0]] * len(completions), completions=completions,
                                                 prompts_text=[prompts_text[0]] * len(completions),
                                                 decouple_adv=[self.decouple_adv] * len(completions), **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
                if num_refs > 0:
                    rewards_per_func[-num_refs:, i] = rewards_per_func[-num_refs:, i] * (1 - 0.0)

        # Gather rewards across processes, Cat Tensor in Batch.
        # if self.accelerator.is_main_process:
        #     print("rewards_per_func shape Before accelerator.gather: ", rewards_per_func.shape)
            # print(rewards_per_func)
        rewards_per_func = self.accelerator.gather(rewards_per_func)
        # gather 是“全收集”（all-gather）的一种实现，即每个进程最终都得到全部数据
        # if self.accelerator.is_main_process:
        #     print("rewards_per_func shape After accelerator.gather: ", rewards_per_func.shape)
            # print(rewards_per_func)
            
        if not self.decouple_adv:
            # Sum the rewards from all reward functions
            rewards = rewards_per_func.sum(dim=1)  # [B, num_reward] --> [B]
            # Compute grouped-wise rewards
            # Each group consists of num_generations completions for the same prompt
            if ref_responses:
                # repeat_inter = len(inputs)
                repeat_inter = self.num_generations
                mean_grouped_rewards = rewards.view(-1, repeat_inter).mean(dim=1)
                std_grouped_rewards = rewards.view(-1, repeat_inter).std(dim=1)
            else:
                repeat_inter = self.num_generations
                mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
                std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

            # Normalize the rewards to compute the advantages
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(repeat_inter, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(repeat_inter, dim=0)

            # advantages = torch.clamp(len(self.reward_funcs) - mean_grouped_rewards, min=0.4, max=0.8) * (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        else:
            decouple_adv_beta = [1.0, 0.5, 0.5]
            advantages = torch.zeros_like(rewards_per_func)  # [B, num_reward]
            for i in range(len(self.reward_funcs)):
                rewards = rewards_per_func[:, i]
                if ref_responses:
                    repeat_inter = self.num_generations
                    mean_grouped_rewards = rewards.view(-1, repeat_inter).mean(dim=1)
                    std_grouped_rewards = rewards.view(-1, repeat_inter).std(dim=1)
                else:
                    # repeat_inter = len(inputs)
                    repeat_inter = self.num_generations
                    mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
                    std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

                # Normalize the rewards to compute the advantages
                mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(repeat_inter, dim=0)
                std_grouped_rewards = std_grouped_rewards.repeat_interleave(repeat_inter, dim=0)
                
                advantages[:, i] = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
                ### min 也很重要？！！ 因为总是关注 困难样本，会使得 较易样本 也不会。
                advantages[:, i] = torch.clamp(decouple_adv_beta[i] - mean_grouped_rewards.detach(), min = 0.35, max = 1.0).detach() * advantages[:, i]
                
                # if self.accelerator.is_main_process:
                #     print(f'Advantages {i}:', advantages[:, i].detach())
            advantages = advantages.sum(dim=1)
        # if self.accelerator.is_main_process:
        #     print('Advantages before Sliced:', advantages.detach())
            
        # Get only the local slice of advantages
        process_slice = slice(
            self.accelerator.process_index * len(completions),
            (self.accelerator.process_index + 1) * len(completions),
        )
        advantages = advantages[process_slice] # 每张卡所特有的。

        k = int(len(advantages) * self.prune_ratio)  # 剪掉的个数
        if self.accelerator.is_main_process:
            print('Advantages after Sliced:', advantages.detach())

        if self.prune_threshold > 0:
            num_pruned = int((torch.abs(advantages.detach()) <= self.prune_threshold).sum().item())
            k = 3 * k if num_pruned > 3 * k else k
            if k >= int(len(advantages)):
                k = int(len(advantages)) - 1

        if k > 0:
            with torch.no_grad():
                assert advantages.dim() == 1, "advantages It must be a one-dimensional tensen."
                abs_advantages = torch.abs(advantages + 0.25).detach()
                # abs_advantages = advantages.detach()
                _, min_indices = torch.topk(abs_advantages, k=k, largest=False)  # 直接topk可能会乱序。
                batch_valid_idx = torch.ones_like(advantages).to(bool)
                batch_valid_idx[min_indices] = False
        else:
            batch_valid_idx = torch.ones_like(advantages).to(bool)

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        # self.accelerator.empty_cache()

        return {
            "prompt": prompts[0],
            "prompt_ids": prompt_ids[batch_valid_idx],
            "prompt_mask": prompt_mask[batch_valid_idx],
            "completion_ids": completion_ids[batch_valid_idx],
            "completion_mask": completion_mask[batch_valid_idx],
            "old_per_token_logps": old_per_token_logps[
                batch_valid_idx] if old_per_token_logps is not None else old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps[
                batch_valid_idx] if ref_per_token_logps is not None else ref_per_token_logps,
            "advantages": advantages[batch_valid_idx],
            "multimodal_inputs": multimodal_inputs,
            'prompt_completion_ids': prompt_completion_ids[batch_valid_idx],
            'attention_mask': attention_mask[batch_valid_idx],
            'prompt_length': prompt_length,
            'batch_valid_idx': batch_valid_idx,
            'full_prompt_completion_ids': prompt_completion_ids,
            'full_attention_mask': attention_mask,
            'full_completion_ids': completion_ids
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, use_compile=True):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Check if we need to generate new completions or use buffered ones

        if self.state.global_step % self.num_iterations == 0:
            with torch.no_grad():
                inputs = self._generate_and_score_completions(inputs, model)
            self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            self.beta = 0.0
        else:
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        self._step += 1

        # Get the prepared inputs
        prompt_ids = inputs["prompt_ids"]
        completion_mask = inputs["completion_mask"]
        multimodal_inputs = inputs["multimodal_inputs"]
        full_prompt_completion_ids = inputs['full_prompt_completion_ids']
        full_attention_mask = inputs['full_attention_mask']
        full_completion_ids = inputs['full_completion_ids']
        multimodal_inputs = inputs['multimodal_inputs']
        batch_valid_idx = inputs['batch_valid_idx']
        # Get the current policy's log probabilities
        per_token_logps = self._get_per_token_logps(model, full_prompt_completion_ids, full_attention_mask,
                                                    full_completion_ids, '-----new_per_token_logps-----', logits_to_keep = None,
                                                    **multimodal_inputs)[batch_valid_idx]
        # per_token_logps = per_token_logps[:, prompt_length - 1:]

        # # torch.cuda.empty_cache()
        # gc.collect()

        # Get the advantages from inputs
        advantages = inputs["advantages"]
        if self.use_advantages_clip and self.advantages_clip_down > 0:
            advantages = advantages.clamp_min(- self.advantages_clip_down)
        if self.use_advantages_clip and self.advantages_clip_up > 0:
            advantages = advantages.clamp_max(self.advantages_clip_up)
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
        # and use per_token_logps.detach() instead
        if self.accelerator.is_main_process:
            print('Cliped Advantages:', advantages.detach())
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()

        # if torch.abs(advantages).sum() == 0:
        #     with torch.no_grad():
        #         advantages = advantages - 0.1
        
        if False:
            with torch.no_grad():
                chunk_size = 10
                lamb = 0.99
                advantages = advantages.unsqueeze(1)
                batch_size = per_token_logps.shape[0]
                length = per_token_logps.shape[-1]
                chunk_num = math.ceil(length / chunk_size)
                advantages_ = advantages.repeat(1, chunk_num)
                lambda_ = torch.ones_like(advantages_) * lamb
                cum_lambda = torch.cumprod(lambda_, dim=-1).flip(-1) / lamb
                cum_lambda = torch.clamp(cum_lambda, min=0.1, max=1.0)
                cum_lambda = cum_lambda.unsqueeze(-1).repeat(1, 1, chunk_size).reshape(batch_size, -1)
                advantages = advantages * cum_lambda[:, -length:]
        else:
            advantages = advantages.unsqueeze(1)

        if (old_per_token_logps is not None and
                inputs["ref_per_token_logps"] is not None and
                not (per_token_logps.shape[1] == old_per_token_logps.shape[1] == inputs["ref_per_token_logps"].shape[1])
        ):
            print('prompt: ', inputs["prompt"])
            print('per_token_logps.shape: ', per_token_logps.shape)
            print('old_per_token_logps.shape: ', old_per_token_logps.shape)
            print('completion_mask.shape: ', completion_mask.shape)
            print('ref_per_token_logps.shape: ', inputs["ref_per_token_logps"].shape)
            print('#' * 200)
            min_length = min(
                [per_token_logps.shape[1], old_per_token_logps.shape[1], inputs["ref_per_token_logps"].shape[1],
                 completion_mask.shape[1]])
            per_token_logps = per_token_logps[:, :min_length]
            old_per_token_logps = old_per_token_logps[:, :min_length]
            completion_mask = completion_mask[:, :min_length]
            inputs["ref_per_token_logps"] = inputs["ref_per_token_logps"][:, :min_length]
        
        if self.alg == 'gspo':
            loss = gspo_compute_loss(per_token_logps, old_per_token_logps,
                                     advantages, completion_mask, self.epsilon_low, self.epsilon_high)
        elif self.alg == 'gspo_token':
            loss = gspo_token_compute_loss(per_token_logps, per_token_logps.detach(),
                                           advantages, completion_mask, self.epsilon_low, self.epsilon_high, False)
        elif self.alg == 'unbias_gspo_token':
            loss = gspo_token_compute_loss(per_token_logps, per_token_logps.detach(),
                                           advantages, completion_mask, self.epsilon_low, self.epsilon_high, True)
        elif use_compile:
            per_token_loss, per_token_kl, mean_kl, mean_kl_per_reward, per_token_loss1, per_token_loss2, coef_1, coef_2 = grpo_compute_loss(
                per_token_logps, old_per_token_logps,
                advantages, completion_mask,
                inputs["ref_per_token_logps"], self.epsilon_low, self.epsilon_high, self.bi_kl, use_neg_clamp=self.use_neg_clamp)

            mean_kl = self.accelerator.gather_for_metrics(mean_kl).mean().item()
            self._metrics["mean_kl"].append(mean_kl)
            # self._metrics["completion_length"].append(self.accelerator.gather_for_metrics(completion_length).mean().item())
            self._metrics["mean_kl_per_reward"].append(
                self.accelerator.gather_for_metrics(mean_kl_per_reward).mean().item())
            # 正常范围内。
            if mean_kl > 0.1:
                self.beta = self.last_beta / 2
            if mean_kl > 1.0:
                self.beta = self.last_beta
            if mean_kl > 10.0:
                self.beta = self.last_beta * 1.5

            if mean_kl > 50:  # 为什么有的论文不要 kl 呢？？!
                self.beta = self.last_beta * 2  # 若不往回拉的话，采样过程中将出现 nan 而报错。
                # self.beta = 0.0
                if self.accelerator.is_main_process and random.random() > 0.75:
                    print('advantages:', advantages.detach())
                    print('coef_1 max', coef_1.detach().max())
                    print('coef_2 max', coef_2.detach().max())
                    print('per_token_loss1 max', per_token_loss1.detach().max())
                    print('per_token_loss2 max', per_token_loss2.detach().max())
                    print('coef_1 min', coef_1.detach().min())
                    print('coef_2 min', coef_2.detach().min())
                    print('per_token_loss1 min', per_token_loss1.detach().min())
                    print('per_token_loss2 min', per_token_loss2.detach().min())

            if self.high_entropy:  # 间接的 高熵 Tokens。 可配合 torch.min(per_token_loss1, per_token_loss2)
                entroy_beta = old_per_token_logps.detach().exp() # 增强 正信号 引导强度。
                entroy_beta = torch.clamp(entroy_beta, min=0, max=1)
                per_token_loss = (1 - entroy_beta) * per_token_loss
            
            if self.beta != 0 and True:
                per_token_loss = per_token_loss + self.beta * per_token_kl

            # Compute final loss
            if self.alg == 'grpo':  # mean(G), mean(O)
                loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            elif self.alg == 'grpo-unbias':  # mean(G, O) DAPO
                loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
                if self.accelerator.is_main_process and random.random() > 0.9:
                    print('grpo_denorm_length: ', (completion_mask.sum().item()))
            elif self.alg == 'grpo-unbias2': # Dr.GRPO
                loss = (per_token_loss * completion_mask).sum() / (self.grpo_denorm_length)

            # if self.beta != 0:
            #     kl_loss = self.beta * per_token_kl
            #     kl_loss = (kl_loss * completion_mask).sum() / completion_mask.sum()
            #     kl_loss.backward(retain_graph=True)
            #     max_grad_norm = 2.0
            #     # with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            #     grad_norm_kl = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
        else:
            # Compute the policy ratio and clipped version
            coef_1 = torch.exp(per_token_logps - old_per_token_logps)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

            # Add KL penalty if beta > 0

            ref_per_token_logps = inputs["ref_per_token_logps"]
            if self.bi_kl:  # 双向KL
                per_token_kl = (torch.exp(ref_per_token_logps - per_token_logps) - (
                        ref_per_token_logps - per_token_logps) - 1
                                + torch.exp(per_token_logps - ref_per_token_logps) - (
                                        per_token_logps - ref_per_token_logps) - 1) / 2
            else:
                per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (
                        ref_per_token_logps - per_token_logps) - 1

            # Log KL divergence
            with torch.inference_mode():
                # completion_length = completion_mask.sum(dim=1).mean()
                mean_kl_per_reward = (per_token_kl * completion_mask).sum(1) / completion_mask.sum(dim=1)
                mean_kl = ((per_token_kl * completion_mask).sum() / completion_mask.sum())
            mean_kl = self.accelerator.gather_for_metrics(mean_kl).mean().item()
            self._metrics["mean_kl"].append(mean_kl)
            # self._metrics["completion_length"].append(self.accelerator.gather_for_metrics(completion_length).mean().item())
            self._metrics["mean_kl_per_reward"].append(
                self.accelerator.gather_for_metrics(mean_kl_per_reward).mean().item())

            if mean_kl > 1.0:
                self.beta = self.last_beta / 1.5
            if mean_kl > 2.0:
                self.beta = self.last_beta
            if mean_kl > 4.0:
                self.beta = self.last_beta * 1.5
            # 异常范围内。
            if mean_kl > 100:  # 有些论文压根不要 KL.
                self.beta = 0.01 * self.last_beta

                if mean_kl > 1e3:  # 为什么有的论文不要 kl 呢？？! 很容易出现 nan 值报错的。
                    self.beta = self.last_beta

                if self.accelerator.is_main_process:
                    print('advantages:', advantages.detach())
                    print('coef_1 max', coef_1.detach().max())
                    print('coef_2 max', coef_2.detach().max())
                    print('per_token_loss1 max', per_token_loss1.detach().max())
                    print('per_token_loss2 max', per_token_loss2.detach().max())
                    print('coef_1 min', coef_1.detach().min())
                    print('coef_2 min', coef_2.detach().min())
                    print('per_token_loss1 min', per_token_loss1.detach().min())
                    print('per_token_loss2 min', per_token_loss2.detach().min())

            if False:  # 间接的 高熵 Tokens。
                entroy_beta = old_per_token_logps.detach().exp()
                entroy_beta = torch.clamp(entroy_beta, min=0, max=1)
                per_token_loss = (1 - entroy_beta) * per_token_loss
            if self.beta != 0:
                per_token_loss = per_token_loss + self.beta * per_token_kl
            # Compute final loss
            if self.alg == 'grpo':  # mean(G), mean(O)
                loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            elif self.alg == 'grpo-unbias':  # mean(G, O)
                loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
            elif self.alg == 'grpo-unbias2':
                loss = (per_token_loss * completion_mask).sum() / self.grpo_denorm_length

            # print(per_token_loss1)
            # print(per_token_loss2)
        # Log clip ratio
        try:
            is_clipped = (per_token_loss1.detach() < per_token_loss2.detach()).float()
            clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
            self._metrics["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
            self._metrics["beta"].append(self.beta)
        except:
            pass

        if self.accelerator.is_main_process:
            print('completion_mask total: ', completion_mask.detach().sum().item(), completion_mask.shape)
            # print('grad_norm_kl: ', grad_norm_kl.item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
            self,
            model_name: Optional[str] = None,
            dataset_name: Optional[str] = None,
            tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def _get_train_sampler(self, *args, **kwargs) -> Sampler:
        """Returns a sampler that ensures proper data sampling for GRPO training."""
        effective_batch_size = (
                self.args.per_device_train_batch_size
                * self.accelerator.num_processes
                * self.args.gradient_accumulation_steps
        )

        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        """Returns a sampler for evaluation."""
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _fix_param_name_to_vllm(self, name, extra_prefixes: Optional[list[str]] = None):
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    @profiling_decorator
    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                    fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                    if fsdp_version == 1:
                        self._sync_fsdp1_params_to_vllm(
                            self.model
                        )  # use memory-efficient post-order traversal for FSDP
                    elif fsdp_version == 2:
                        self._sync_fsdp2_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                if fsdp_version == 1:
                    self._sync_fsdp1_params_to_vllm(self.model)  # use memory-efficient post-order traversal for FSDP
                elif fsdp_version == 2:
                    self._sync_fsdp2_params_to_vllm(self.model)
            else:
                for name, param in self.model.named_parameters():
                    name = self._fix_param_name_to_vllm(name)
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()
