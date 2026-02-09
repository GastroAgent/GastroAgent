import gc
import os
import random
import json
import PIL
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, GenerationConfig
import sys

sys.path.append('/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/VLM-R1/src/open-r1-multimodal/src/open_r1')  # 服务器调试
from llava.model.llavaProcessor import LlavaProcessor
from llava.model.language_model.llava_qwen_mul import PloyLlavaLlamaForCausalLM
from trl.data_utils import maybe_apply_chat_template
from tqdm import tqdm

import torch.distributed as dist
import os

# 初始化分布式环境
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ['LOCAL_RANK'])  # 获取当前进程的本地GPU编号
device = torch.device(f'cuda:{local_rank}')  # 指定当前进程使用的设备

model_id = '/home/dalhxwlyjsuo/criait_tansy/weights/Med-Llava-Qwen2-7B-GSPO'

data_name = 'Kvasir-en'
batch_size = 1
model_name = 'Med-Llava-Qwen2-7B-GSPO'  # 2: is_message=True
is_message = True
use_llava_v1_prompt = False  # no_tag # 不再重要了。
llava_v1_prompt_post = ''

if use_llava_v1_prompt:
    is_message = False
    llava_v1_prompt_post = '-no-tag'

data_path = f'/home/dalhxwlyjsuo/criait_tansy/jmf/ds-vl-on-policy-test/Kvasir-en.json'

lora_model_path = None  # SFT or Whole Model.

model_save_path = None  # 不保存完整权重。

data_save_path = f'/home/dalhxwlyjsuo/criait_tansy/jmf/GRPO_result/{model_name}/{data_name}-{is_message}{llava_v1_prompt_post}_rank{rank}_VQA.json'
os.makedirs(f'/home/dalhxwlyjsuo/criait_tansy/jmf/GRPO_result/{model_name}', exist_ok=True)

model_init_kwargs = {
    'attn_implementation': 'flash_attention_2',
    'torch_dtype': 'bfloat16',  # float32, bfloat16 # flash attention 只支持bfloat16。
    'use_cache': True,
}

model = PloyLlavaLlamaForCausalLM.from_pretrained(
    model_id,
    **model_init_kwargs
).to(device)  # 显式将模型移动到当前设备

if lora_model_path is not None and lora_model_path:
    model = PeftModel.from_pretrained(model, lora_model_path)
    model = model.merge_and_unload()  # 合并 LoRA 权重

processing_class = LlavaProcessor.from_pretrained(model_id,
                                                  use_fast=True,
                                                  trust_remote_code=model_init_kwargs.get("trust_remote_code", None),
                                                  patch_size=14)

processing_class.tokenizer.padding_side = 'left'
if is_message:
    processing_class.chat_template = processing_class.tokenizer.chat_template

if model_save_path is not None and model_save_path:
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    processing_class.save_pretrained(model_save_path)

dataset = json.load(open(data_path, 'r'))


# 处理图像
def get_key_from_inputs(x: dict, key: str):
    ele = x.get(key, None)
    if isinstance(ele, list):
        return ele
    else:
        return [ele]


generation_config = GenerationConfig(
    max_new_tokens=512,
    do_sample=True,
    temperature=0.1,
    pad_token_id=processing_class.tokenizer.pad_token_id,
)
results = []
batch_size = 1
print('data save path is:', data_save_path)
for i in tqdm(range(rank, len(dataset), world_size), desc=f'Rank {rank}'):
    inputs = dataset[i:i + batch_size]

    # system_prompt = ("You are a helpful Medical AI assistant and an authoritative expert in th e area of the Upper Digestive Tract and Digestive System. "
    #        "As the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, and use natural language to assist users in various tasks. "
    #        "As the expert, you need to make sure that your response is logical and factual, and that it conveys the authority of the expert. "
    #        "The visual content will be provided with the following format: <|Image_start|>visual content<|Image_end|>.")

    # system_prompt = ("You are a helpful AI assistant and an authoritative expert in the field of the medicine. You have a solid foundation in medicine and are proficient in answering various questions related to medical topics. "
    #        "\nAs the AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, and use natural language to assist users in various tasks. "
    #        "\nAs the medical expert, you need to make sure that your response is logical and factual, and that it conveys the authority of the expert. "
    #        "\nThe visual content (most about medical imaging) will be provided with the following format: <|Image_start|>visual content<|Image_end|>.")

    system_prompt = (
        "You are a helpful Medical AI assistant and an authoritative expert in the field of the Upper Digestive Tract and Digestive System (Especially Endoscopy). "
        "As the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, and use natural language to assist users in various tasks. "
        "As the expert, you need to make sure that your response is logical and factual, and that it conveys the authority of the expert. "
        "The visual content (most about medical imaging) will be provided with the following format: <|Image_start|>visual content<|Image_end|>.")

    ############################################################################################################################
    # system_prompt = ("You are a helpful Medical AI assistant and an authoritative expert in the field of the Upper Digestive Tract and Digestive System. You have a solid foundation in medicine and are proficient in answering various visual questions related to medical topics. "
    #        "As the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, and use natural language to assist users in various tasks. "
    #        "As the expert, you need to make sure that your response is logical and factual, and that it conveys the authority of the expert. "
    #        "The visual content (most about medical imaging) will be provided with the following format: <|Image_start|>visual content<|Image_end|>.")

    # system_prompt = ("You are a helpful Medical AI assistant and an authoritative expert in the field of the Upper Digestive Tract and Digestive System (Especially Endoscopy). You have a solid foundation in medicine and are proficient in answering various visual questions related to medical topics. "
    #        "As the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, and use natural language to assist users in various tasks. "
    #        "As the expert, you need to make sure that your response is logical and factual, and that it conveys the authority of the expert. "
    #        "The visual content (most about medical imaging) will be provided with the following format: <|Image_start|>visual content<|Image_end|>.")

    # system_prompt = ("You are a helpful Medical AI assistant and authoritative expert in the medical field. You have a solid foundation in medicine and are proficient in answering various visual questions related to medical topics. "
    #        "As the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, and use natural language to assist users in various tasks. "
    #        "As the expert, you need to make sure that your response is logical and factual, and that it conveys the authority of the expert. "
    #        "The visual content (most about medical imaging) will be provided with the following format: <|Image_start|>visual content<|Image_end|>.")

    if is_message:
        for example in inputs:
            example['prompt'] = [
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': example['formatted_text'],  # Qwen2 纯文本提示词
                }
            ]
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
    elif use_llava_v1_prompt:
        prompts_text = [example['llava_v1_prompt_text'] for example in inputs]
    else:
        prompts_text = [example['prompt_text'] for example in inputs]

    images = []
    for x in inputs:
        if "image_paths" in x and x["image_paths"] is not None:
            imgs = [PIL.Image.open(p).convert('RGB') for p in get_key_from_inputs(x, "image_paths")]
        elif "image_path" in x and x["image_path"] is not None:
            imgs = [PIL.Image.open(p).convert('RGB') for p in get_key_from_inputs(x, "image_path")]
        elif "image" in x and x["image"] is not None:
            imgs = [PIL.Image.open(p).convert('RGB') for p in get_key_from_inputs(x, "image")]
        else:
            imgs = []

        for img in imgs:
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
            images.append(img)

    if len(images) > 0:
        prompt_inputs = processing_class(
            text=prompts_text,
            images=images,
            return_tensors='pt',
            padding=True,
            padding_side='left',
            add_special_tokens=False
        )
    else:
        prompt_inputs = processing_class(
            text=prompts_text,
            return_tensors='pt',
            padding=True,
            padding_side='left',
            add_special_tokens=False
        )
    prompt_inputs = prompt_inputs.to(model.device)
    generate_returned_result = model.generate(
        **prompt_inputs,
        generation_config=generation_config,
        batch_num_images=[1]  # 每个 样本对应的图像数量。
    )
    prompt_ids = prompt_inputs["input_ids"]
    prompt_length = prompt_inputs["input_ids"].size(1)

    completions = processing_class.batch_decode(generate_returned_result, skip_special_tokens=True)
    # print(completions[0])
    for generated_text, input_, prompt_text in zip(completions, inputs, prompts_text):
        input_['generated_text'] = generated_text
        results.append({
            'generated_text': generated_text,
            'prompt_text': prompt_text,
            "answer": input_['answer'],
            "gt_answer": input_['gt_answer'],
            "image_paths": input_['image_paths'],
            "question_id": input_['question_id'],
            "question_type": input_['question_type'],
        })
        if random.random() < 0.5:
            print(f'prompt_text: {prompt_text}')
            print(f'generated_text: {generated_text}\n')

json.dump(results, open(data_save_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
print('Infer Done!')
dist.barrier()
dist.destroy_process_group()
