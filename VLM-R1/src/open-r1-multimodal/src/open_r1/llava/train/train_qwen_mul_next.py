# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9505))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     print(e)
#     pass

from qwen_vl_utils import process_vision_info
import random
import re
from packaging import version
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from trl.data_utils import maybe_apply_chat_template
import torch

import transformers
import tokenizers
import sys
sys.path.append('/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1')

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer_med import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.model.language_model.llava_qwen_mul import PloyLlavaLlamaForCausalLM

from llava.mm_utils import tokenizer_image_token

from PIL import Image

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=True)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_vision_module: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    vision_lora_enable: bool = False
    lora_enable: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    loss_denorm: int = None
    

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        try:
            from_str = sentence["from"]
        except:
            print(sentence)
            print('='* 100)
            print(source)

        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             str(sentence["value"]) + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments,
    num_image = 1
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    count_images = 0
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in str(sentence['value']):
                # default_image_token = ['mmtag', 'qwen_v2', 'direct']
                # if any([x in conversation_lib.default_conversation.version for x in default_image_token]):
                    # sentence['value'] = sentence['value'].replace('<|Image_start|>' + DEFAULT_IMAGE_TOKEN + '<|Image_end|>',
                    #                                               DEFAULT_IMAGE_TOKEN)
                    # sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN,
                    #                                               '<|Image_start|>' + DEFAULT_IMAGE_TOKEN + '<|Image_end|>')
                sentence['value'] = sentence['value'].replace('<|Image_start|>' + DEFAULT_IMAGE_TOKEN + '<|Image_end|>',
                                                                  DEFAULT_IMAGE_TOKEN)
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                                  '<|Image_start|>' + DEFAULT_IMAGE_TOKEN + '<|Image_end|>')
                if sentence['from'] == 'user' or sentence['from'] == 'human': # 只统计用户中的。
                    cur_num_image = sentence['value'].count('<|Image_start|>' + DEFAULT_IMAGE_TOKEN + '<|Image_end|>')
                    count_images += cur_num_image
                if sentence['from'] == 'gpt' or sentence['from'] == 'assistant': # 防止模型回复出现。
                    sentence['value'] = sentence['value'].replace('<|Image_start|>' + DEFAULT_IMAGE_TOKEN + '<|Image_end|>',
                                                                  DEFAULT_IMAGE_TOKEN)
                # sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                # for i in range(cur_num_image):
                #     sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                # default_image_token = ['mmtag', 'qwen_v2', 'direct']
                # if any([x in conversation_lib.default_conversation.version for x in default_image_token]):
                #     sentence['value'] = sentence['value'].replace('<|Image_start|>' + DEFAULT_IMAGE_TOKEN + '<|Image_end|>',
                #                                                   DEFAULT_IMAGE_TOKEN)
                #     sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN,
                #                                                   '<|Image_start|>' + DEFAULT_IMAGE_TOKEN + '<|Image_end|>')
                
                replace_token = DEFAULT_IMAGE_TOKEN
                if data_args.mm_use_im_start_end:
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = str(sentence["value"]).replace(DEFAULT_IMAGE_TOKEN, replace_token)
    assert count_images == num_image, 'num_image is not num_image_tokens.'
    return sources

def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

# fix: add qwen2
def preprocess_qwen_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    # print('-----preprocess_qwen_2-------')
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1], "system": "system", "user": conv.roles[0], "assistant": conv.roles[1]}
    system_prompt = conv.system
    
    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human(is from system).
                system_prompt = source[0]["value"]
                source = source[1:]
        except:
            print('='* 100)
            print(source)
            print('='* 100)
        prompt = [
            {
                'role': 'system',
                'content': system_prompt
            }
        ]
        conv.messages = []
        for j, sentence in enumerate(source):
            # print('role', roles)
            # print('role', sentence)
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            if role == 'user':
                prompt.append({
                    'role': 'user',
                    'content': sentence['value']
                })
            elif role == 'assistant':
                prompt.append({
                    'role': 'assistant',
                    'content': sentence['value'].replace(tokenizer.eos_token, '') + tokenizer.eos_token + conv.sep2
                })
        conversations.append({
            'prompt':prompt
        })              
    conversations = [maybe_apply_chat_template(example, tokenizer)["prompt"] for example in conversations]
    # conversations = [ x + tokenizer.eos_token for x in conversations]

    # # 替换系统提示词
    # if system_prompt is not None and system_prompt:
    #     system_prompt = f'<|im_start|>system\n{system_prompt}<|im_end|>'
    #     # conversations = [x.replace('<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>', system_prompt) for x in conversations]
    #     # re.DOTALL：使得 . 可以匹配任何字符，包括换行符。
    #     conversations = [
    #         re.sub(r"^<\|im_start\|>system\n.*?<\|im_end\|>", system_prompt, x, count=1, flags=re.DOTALL) + tokenizer.eos_token
    #         for x in conversations
    #     ]
        
    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(x, tokenizer, return_tensors='pt') for x in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.QWEN_2

    # Mask targets
    sep = conv.sep + conv.roles[1] + "\n"
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        rounds_len = len(rounds) - 1 # 1 指的是 "" 空字符串。
        cur_len = 0
        # target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_ids = tokenizer_image_token(rou, tokenizer)
                instruction_ids = tokenizer_image_token(parts[0], tokenizer)
                equal_parts = [x == y for x, y in zip(round_ids, instruction_ids)]

                instruction_len = equal_parts.index(False) if False in equal_parts else len(equal_parts)
                round_len = len(round_ids)

            else:
                round_ids = tokenizer(rou).input_ids
                instruction_ids = tokenizer(parts[0]).input_ids
                equal_parts = [x == y for x, y in zip(round_ids, instruction_ids)]

                instruction_len = equal_parts.index(False) if False in equal_parts else len(equal_parts)
                round_len = len(round_ids)

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1
            if random.random() > 0.0001:
            # if random.random() > 0.25:  # 以一定的概率 学习 上下文.
                target[cur_len: cur_len + instruction_len] = IGNORE_INDEX # Mask Context。

            cur_len += round_len
            
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len + rounds_len - 1:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
                print(sources)
                print('-'*100)
    
    if os.environ.get("RANK", -1) == '0' and random.random() < 0.1:
        print('Prompt:\n', conversations[0])
    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_qwen_vl(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    processor
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    text = tokenizer.apply_chat_template(
        sources, tokenize=False, add_generation_prompt=False
    )
    image_inputs, video_inputs = process_vision_info(sources)
    # images = [img.resize((336, 336), Image.Resampling.LANCZOS) for img in image_inputs]
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        size={"height": 336, "width": 336, 'shortest_edge': 224*224, 'longest_edge': 448*448},
        do_resize=True
    )
    input_ids = inputs['input_ids']
    # print(image_inputs)
    # print(inputs['pixel_values'].shape)
    # print(input_ids.shape)
    targets = copy.deepcopy(input_ids)
    return dict(inputs=inputs, labels=targets)

def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        #assert len(source) == 2
        #assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # print('=======conversations',conversations)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # print("conversation:",conversation_lib.default_conversation.version)
    # conversation_lib.default_conversation.version == "qwen_v2"

    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        # print('--v1--')
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        # print('--mpt--')
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # fix: add qwen2
    if conversation_lib.default_conversation.version.startswith("qwen_v2"):
        # print('--qwen_v2--')
        return preprocess_qwen_2(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations

    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        #print('============mm_image_plain?')
        #print('=======conversations',conversations)
        # print('============mm_image_preprocess?')
        # print('=======conversations',conversations)
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

from PIL import Image
import cv2
import numpy as np

# 读取图像
def crop_save(path, new_dir = '', old_dir = ''):
    image = cv2.imread(path)
    # 将图像转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义红色的HSV范围（这里假设红色是主要目标）
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 70, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # 合并两个掩码
    mask = mask1 + mask2

    # 使用形态学操作清理噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 找到轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 假设最大的轮廓就是我们要找的红色区域
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # 裁剪图像
        cropped_image = image[y:y+h, x:x+w]
        save_path = path.replace(old_dir,new_dir)

        # 保存裁剪后的图像
        cv2.imwrite(save_path, cropped_image)
        return ""
    else:
        return path
def crop_right_square(image: Image.Image,
                      target_weight_size: int = 1280,
                      target_height_size=1080,
                      identy = True) -> Image.Image:
    """
    从图像右侧裁剪出一个正方形区域（target_size x target_size）。

    参数:
        image (Image.Image): 输入的 PIL 图像对象。
        target_size (int): 裁剪目标区域的边长（默认为 1080）。

    返回:
        Image.Image: 裁剪后的图像对象。
    """
    if identy:
        return image
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


class LazySupervisedQwenVLDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedQwenVLDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.processor = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list
        
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        sys_prompt = conversation_lib.qwen2_vl.system
        # print(sources)
        image_files = sources['image']
        images_prompt = [{
            "type": "image",
            "image": image_file,
        } for image_file in image_files]
        # query = sources['formatted_text']
        query = sources['conversations'][0]['value'].replace('Image: <|Image_start|><image><|Image_end|>\n', '')
        responces = sources['conversations'][1]['value']
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": sys_prompt},
                ],
            },
            {
                "role": "user",
                "content": images_prompt + [
                    {"type": "text", "text": query},
                ],
            },
            {
                'role': 'assistant',
                "content": [
                    {"type": "text", "text": responces},
                ],
            }
        ]
        
        for conv in sources['conversations'][2:]:
            if conv['from'] == 'human':
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": conv['value']},
                    ],
                })
            elif conv['from'] == 'gpt':
                messages.append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": conv['value']},
                    ],
                })
        batch = preprocess_qwen_vl(messages, self.processor.tokenizer, self.processor)
        # text = self.processor.apply_chat_template(
        #     messages, tokenize=False, add_generation_prompt=True
        # )
        return batch

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list
        
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0] or 'image_paths' in sources[0]:
            try:
                image_file = self.list_data_dict[i]['image']
            except:
                self.list_data_dict[i]['image'] = self.list_data_dict[i]['image_paths']
                image_file = self.list_data_dict[i]['image_paths']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor

            if isinstance(image_file, list):
                image = []
                #print(image_file)
                for img_file in image_file:
                    #print('img_file',img_file)
                    try:
                        # crop_save(img_file, '', '')
                        image.append(Image.open(img_file).convert('RGB'))
                    except:
                        raise ValueError
                        try:
                            img_file = img_file.replace('/home/dalhxwlyjsuo/criait_tansy/project/',
                                                        '/home/lab/data/')
                            image.append(Image.open(img_file).convert('RGB'))
                        except:
                            raise ValueError
                            image.append(Image.open('/home/work/ds-vl-on-policy_test/01.2013051500235_5.jpg').convert('RGB'))
            else:
                # crop_save(image_file, '', '')
                image = Image.open(os.path.join(image_file)).convert('RGB')

            if isinstance(image, list):
                image = [crop_right_square(x) for x in image]
            else:
                image = crop_right_square(image)

            try:
                if isinstance(image, list):
                    image = [processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in image]
                # elif len(image) == 0:
                #     image = torch.zeros(0)
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            except:
                print('Image Error: ',image)
                print('Image Path: ',image_file)
                raise NotImplementedError

            try:
                sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.data_args,
                    num_image=len(image))
            except Exception as e:
                print(sources)
                print('Exception:', e)
                sources = copy.deepcopy([e["conversations"] for e in sources])
                print('='* 100)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
            
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(0, crop_size['height'], crop_size['width'])
            image = []
        data_dict['num_images'] = len(image)
        # if len(data_dict['image']) > 1:
        # data_dict['image']
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        num_images = None
        if 'num_images' in instances[0]:
            input_ids, labels, num_images = tuple([instance[key] if key in instance else None for instance in instances]
                                      for key in ("input_ids", "labels", 'num_images'))
        else:
            input_ids, labels = tuple([instance[key] for instance in instances]
                                      for key in ("input_ids", "labels"))   
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            num_images = num_images
        )

        if 'image' in instances[0] :
            images = [instance['image'] for instance in instances]
            if isinstance(images[0], list):
                #images = torch.stack([torch.stack(img, dim=0) for img in images], dim = 0)
                # valid_imgs = [img for img in images if len(img) > 0]
                valid_imgs = images
                if len(valid_imgs) > 0:
                    images = [torch.stack(img, dim=0) if not len(img) == 0 else torch.zeros(0) for img in valid_imgs]
                    
                batch['images'] = images
    
            else:
                if all(x is not None and x.shape == images[0].shape for x in images):
                    batch['images'] = torch.stack(images)
                else:
                    batch['images'] = images
        else:
            raise NotImplementedError('Image Key meybe error.')
        return batch

@dataclass
class DataCollatorForSupervisedQwenVLDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask, pixel_values, image_grid_thw = tuple([instance['inputs'][key] for instance in instances]
                                  for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"))
        
        labels = torch.nn.utils.rnn.pad_sequence([x['labels'].squeeze() for x in instances],
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX, 
                                                 ).long()
        input_ids = torch.nn.utils.rnn.pad_sequence([x.squeeze() for x in input_ids],
                                                 batch_first=True,
                                                 padding_value=self.tokenizer.tokenizer.pad_token_id).long()
        attention_mask = torch.nn.utils.rnn.pad_sequence([x.squeeze() for x in attention_mask],
                                                 batch_first=True,
                                                 padding_value=0).long()

        return dict(
            labels = labels,
            input_ids = input_ids,
            attention_mask = attention_mask,
            pixel_values = torch.cat(pixel_values).float(),
            image_grid_thw = torch.cat(image_grid_thw).long()
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                data_type=None) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if data_type == 'qwen-vl':
        train_dataset = LazySupervisedQwenVLDataset(tokenizer=tokenizer,
                                        data_path=data_args.data_path,
                                        data_args=data_args
        )
        data_collator = DataCollatorForSupervisedQwenVLDataset(tokenizer=tokenizer)
        return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    print(model_args.model_name_or_path)
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        
        elif 'qwen' in model_args.model_name_or_path.lower() and 'vl' in model_args.model_name_or_path.lower():
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
            from qwen_vl_utils import process_vision_info
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
            processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, use_fast=True)
            model_args.vision_tower = None
        else:
            model = PloyLlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        if 'qwen' in model_args.model_name_or_path.lower() and 'vl' in model_args.model_name_or_path.lower():
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
            from qwen_vl_utils import process_vision_info
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
            processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, use_fast=True)
        else:
            model = transformers.Qwen2ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
            
    model.config.use_cache = False
    print(model)
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "up_proj", "down_proj", "gate_proj"
            ],
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="left",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token
        else:  # use qwen
            tokenizer.legacy = False
        if model_args.version in conversation_lib.conv_templates:
            # print('version:', model_args.version)
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter


        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.freeze_vision_module:
            for p in model.get_model().vision_tower.parameters():
                p.requires_grad = False
        else:
            for p in model.get_model().vision_tower.parameters():
                p.requires_grad = True

        if model_args.tune_mm_mlp_adapter: # 只微调 MM。
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
        
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    if 'qwen' in model_args.model_name_or_path.lower() and 'vl' in model_args.model_name_or_path.lower():
        # processor.tokenizer = tokenizer
        data_module = make_supervised_data_module(tokenizer=processor,
                                              data_args=data_args,
                                              data_type='qwen-vl')

    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    
    # special_tokens = {'additional_special_tokens': ['<image>', '<|Image_start|>', '<|Image_end|>', '<answer>', '</answer>', '<think>', '</think>']}
    # tokenizer.add_special_tokens(special_tokens)
    # if model.lm_head.out_features < tokenizer.vocab_size:
    #     model.resize_token_embeddings(len(tokenizer))
    
    trainer = LLaVATrainer(model=model,
                           tokenizer=tokenizer,
                           args=training_args,
                           **data_module)
    if 'qwen' in model_args.model_name_or_path.lower() and 'vl' in model_args.model_name_or_path.lower():
        pass
    else:
        if training_args.lora_enable and model_args.freeze_backbone:
            for key, p in model.get_model().named_parameters():
                if 'layers.' in key and 'vision_tower' not in key:
                    p.requires_grad = False

        if training_args.vision_lora_enable and training_args.freeze_vision_module and not model_args.tune_mm_mlp_adapter:
            for key, p in model.get_model().named_parameters():
                if 'vision_tower.' in key and 'lora_' in key:
                    p.requires_grad = False

        elif training_args.vision_lora_enable and not training_args.freeze_vision_module and not model_args.tune_mm_mlp_adapter:
            for key, p in model.get_model().named_parameters():
                if 'vision_tower.' in key and 'lora_' in key:
                    p.requires_grad = True

        elif not training_args.vision_lora_enable and training_args.freeze_vision_module and not model_args.tune_mm_mlp_adapter:
            for key, p in model.get_model().named_parameters():
                if 'vision_tower.' in key:
                    p.requires_grad = False

        elif not training_args.vision_lora_enable and not training_args.freeze_vision_module and not model_args.tune_mm_mlp_adapter:
            for key, p in model.get_model().named_parameters():
                if 'vision_tower.' in key:
                    p.requires_grad = True

        if training_args.loss_denorm is not None and training_args.loss_denorm > 0:
            model.config.loss_denorm = training_args.loss_denorm
        else:
            model.config.loss_denorm = None

    if os.environ.get("RANK", -1) == '0':
        trainable_keys = [name for name, param in model.named_parameters() if param.requires_grad]
        print('trainable params: ',trainable_keys)
        print(tokenizer)
    model.train()
    print('Model Mode: ', model.training)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=False)
    else:
        trainer.train()
    print('训练完成')
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            
            # # 判断是否是 PEFT 模型
            # if isinstance(model, PeftModel):
            #     # 或者：融合后保存完整模型
            #     model = model.merge_and_unload()
            #     model.save_pretrained(training_args.output_dir + 'whole')
            # else:
            #     # 非 PEFT，直接保存
            #     model.save_pretrained(training_args.output_dir + 'whole')
    else:
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            safe_save_model_for_hf_trainer(trainer=trainer,
                                        output_dir=training_args.output_dir)
    print(training_args.output_dir)
    print('保存完成')
if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
    # train()