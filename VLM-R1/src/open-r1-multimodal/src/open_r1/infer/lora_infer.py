
import gc
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import json
import PIL
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, GenerationConfig
import sys

sys.path.append('/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1') # 服务器调试
from llava.model.llavaProcessor import LlavaProcessor
from llava.model.language_model.llava_qwen_mul import PloyLlavaLlamaForCausalLM
from trl.data_utils import maybe_apply_chat_template

model_id = '/mnt/inaisfs/data/home/tansy_criait/weights/Med-Llava-Qwen2-7B-GSPO-1216'
# model_id = /mnt/inaisfs/data/home/tansy_criait/weights/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-75
# model_id = /mnt/inaisfs/data/home/tansy_criait/weights/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-buwei
# model_id = /mnt/inaisfs/data/home/tansy_criait/weights/LlavaQwen2-GRPO-Tricks-Total-CoT-6000
data_name = 'Kvasir-en'
batch_size = 1 
model_name = 'Med-Llava-Qwen2-7B-GSPO-1216' # 2: is_message=True
is_message = True
llava_v1_prompt_post = ''
use_llava_v1_prompt = False

if use_llava_v1_prompt:
    is_message = False
    llava_v1_prompt_post = '-no-tag'

data_path = f'/mnt/inaisfs/data/home/tansy_criait/wass_flow_match/data/{data_name}.json'

# lora_model_path = f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/checkpoints/{model_name}'
lora_model_path = None
model_save_path = None

data_save_path = f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/results/Kvasir/{model_name}/{data_name}.json'
os.makedirs(f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/results/Kvasir/{model_name}', exist_ok=True)

model_init_kwargs = {
    'attn_implementation': 'flash_attention_2',
    'torch_dtype': 'bfloat16', # float32, bfloat16 # flash attention 只支持bfloat16。
    'use_cache': True,
}

model = PloyLlavaLlamaForCausalLM.from_pretrained(
    model_id, 
    device_map = 'auto',
    **model_init_kwargs
)

if lora_model_path is not None and lora_model_path:
    # 加载 LoRA 模型并合并权重
    model = PeftModel.from_pretrained(model, lora_model_path)
    model = model.merge_and_unload()  # 合并 LoRA权重 并卸载 LoRA层 为普通 Model。

processing_class = LlavaProcessor.from_pretrained(model_id,
                            use_fast=True,
                            trust_remote_code=model_init_kwargs.get("trust_remote_code", None),
                            patch_size = 14)
print(processing_class.tokenizer.vocab_size)
processing_class.tokenizer.padding_side = 'left'
if is_message:
    processing_class.chat_template = processing_class.tokenizer.chat_template

if model_save_path is not None and model_save_path:
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    processing_class.save_pretrained(model_save_path)

dataset = json.load(open(data_path,'r'))

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
            temperature=0.2,
            pad_token_id= processing_class.tokenizer.pad_token_id,
)
results = []
print('data save path is:', data_save_path)
for i in range(0, len(dataset), batch_size):
    inputs = dataset[i:i+batch_size]
    print(f"Process: {i}/{len(dataset)}")

    system_prompt = (("You are a helpful Medical AI assistant and an authoritative expert in the area of the Upper Digestive Tract and Digestive System, skilled in medical imaging analysis and disease diagnosis. "
           "\nAs the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, and use natural language to assist users in various tasks. "
           "\nAs the expert, you need to make sure that your response is logical and factual, and that it conveys the authority of the expert. "
           "\nThe visual content (most about medical imaging) will be provided with the following format: <|Image_start|>visual content<|Image_end|>.")
                     if 'easy' not in data_name else "You are a helpful medical AI assistant.")

    # system_prompt = (("You are a helpful Medical AI assistant and authoritative expert in the medical field. You have a solid foundation in medicine and are proficient in answering various visual questions related to medical topics. "
    #        "\nAs the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, and use natural language to assist users in various tasks. "
    #        "\nAs the expert, you need to make sure that your response is logical and factual, and that it conveys the authority of the expert. "
    #        "\nThe visual content (most about medical imaging) will be provided with the following format: <|Image_start|>visual content<|Image_end|>.")
    #                  if 'easy' not in data_name else "You are a helpful medical AI assistant.")
    if is_message:
        for example in inputs:
            example['prompt'] = [
                {'role': 'system',
                 'content': system_prompt
                 },
                {
                'role': 'user',
                'content': example['formatted_text'].replace('<|Image_start|><image><|Image_end|>', '<image>').replace('<image>', '<|Image_start|><image><|Image_end|>') + "Let's think step by step.", # Qwen2 纯文本提示词
                }
            ]
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
    elif use_llava_v1_prompt:
        prompts_text = [example['llava_v1_prompt_text'] for example in inputs]
    else:
        prompts_text = [example['prompt_text'] for example in inputs]

    images = []
    for x in inputs:
        if "image" in x and x["image"] is not None:
            imgs = [PIL.Image.open(p).convert('RGB') for p in get_key_from_inputs(x, "image")]
        else:
            imgs = [PIL.Image.open(p).convert('RGB') for p in get_key_from_inputs(x, "image_paths")]

        for img in imgs:
            try:
                # Ensure minimum dimensions of 28 pixels
                w, h = img.size
                if w < 28 or h < 28:
                # Calculate new dimensions maintaining aspect ratio
                    if w < h:
                        new_w = 28
                        new_h = int(h * (28/w))
                    else:
                        new_h = 28
                        new_w = int(w * (28/h))
                    img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
                images.append(img)
            except Exception as e:
                print(e)
                pass

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
        # batch_num_images=[1]
    )
    prompt_ids = prompt_inputs["input_ids"]
    prompt_length =  prompt_inputs["input_ids"].size(1)

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

json.dump(results, open(data_save_path,'w', encoding='utf-8'), indent=4, ensure_ascii=False)
print('Infer Done!')