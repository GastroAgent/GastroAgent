
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
sys.path.append('/home/work/VLM-R1/src/open-r1-multimodal/src/open_r1') # 本地调试
sys.path.append('/home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1') # 服务器调试
from llava.model.llavaProcessor import LlavaProcessor
from llava.model.language_model.llava_qwen_mul import PloyLlavaLlamaForCausalLM
from trl.data_utils import maybe_apply_chat_template

# model_id = '/home/dalhxwlyjsuo/criait_tansy/jmf/code/model_match/serve_llavaqwen2' # 原始模型
model_id = '/home/dalhxwlyjsuo/criait_tansy/GRPO_weights/LlavaQwen2-GRPO-Tricks-Stage3-Epoch3-MyAlg'

# model_id = '/home/dalhxwlyjsuo/criait_tansy/GRPO_weights/LlavaQwen2-GRPO-Med-CRC100k' # 合并后模型
# model_id = '/home/work/code/model_match/local_llavaqwen2' # 本地调试
data_name = 'Kvasir-en'
batch_size = 1 
model_name = 'LlavaQwen2-GRPO-Tricks-Stage3-Epoch3-MyAlgC' # 2: is_message=True
is_message = False
use_llava_v1_prompt = False # no_tag # 不再重要了。
llava_v1_prompt_post = ''

if use_llava_v1_prompt:
    is_message = False
    llava_v1_prompt_post = '-no-tag'

# sbatch -p vip_gpu_01 --gpus=1 /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/lora_infers/lora_infer.sh

data_path = f'/home/dalhxwlyjsuo/criait_tansy/jmf/ds-vl-on-policy-test/{data_name}.json'
# data_path = '/home/work/ds-vl-on-policy-2/Kvasir-en.json' # 本地调试

# lora_model_path = None # SFT or Whole Model.
# lora_model_path = f'/home/dalhxwlyjsuo/criait_tansy/jmf/GRPO_peft_weights/{model_name}'
lora_model_path = '/home/dalhxwlyjsuo/criait_tansy/GRPO_checkpoints/LlavaQwen2-GRPO-Tricks-Stage3-Epoch3-MyAlgC'

model_save_path = None # 不保存完整权重。
# model_save_path = f'/home/dalhxwlyjsuo/criait_tansy/GRPO_weights/{model_name}'

data_save_path = f'/home/dalhxwlyjsuo/criait_tansy/jmf/GRPO_result/{model_name}/{data_name}-{is_message}{llava_v1_prompt_post}.json'
os.makedirs(f'/home/dalhxwlyjsuo/criait_tansy/jmf/GRPO_result/{model_name}', exist_ok=True)

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

# inputs = [
#     {
#         'prompt': """According to the content of the image, answer the following single-choice questions.\n\nImage: <image>\nQuestion: What modality was used to capture this image?.\n  - option_A: Biopsy\n  - option_B: CT scan\n  - option_C: Colonoscopy\n  - option_D: Fundus imaging\n\nPlease give the reasoning process as detailed as possible and put the Corresponding Option for the final answer inside the '<answer><\\answer>' tag.""",
#         'image_paths': [
#             '/home/work/Images/ACRIMA/Im351_g_ACRIMA.png'
#             ],
#         'prompt_text': """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nWhat imaging technique was employed to obtain this picture?Here are 4 candidate answers:['option_A: Biopsy', 'option_B: CT scan', 'option_C:  Colonoscopy', 'option_D: Fundus imaging']. Only return what you think is the correct answer from the candidate answers, and put the corresponding option for the final answer inside the '<answer><\\answer>' tag. Do not return any other irrelevant text! ASSISTANT:"""
#     }
# ]

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
    system_prompt = (("A chat between a curious user and a Medical AI assistant in the field of Endoscopy and Human Stomach. "
                    "As the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, and use natural language to assist users in various tasks."
                    "The visual content will be provided with the following format: <|Image_start|>visual content<|Image_end|>.")
                     if 'easy' not in data_name else "You are a helpful medical AI assistant.")
    if is_message:
        for example in inputs:
            example['prompt'] = [
                {'role': 'system',
                 'content': system_prompt
                 },
                {
                'role': 'user',
                'content': example['formatted_text'], # Qwen2 纯文本提示词
                    # [ # Qwen2-VL 系列的提示词格式。
                    # *({'type': 'image', 'text': example['image_paths'][i]} for i in range(len(example['image_paths']))),
                    # {'type': 'text', 'text': example['formatted_text']}
                # ]
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
            imgs = [PIL.Image.open(p) for p in get_key_from_inputs(x, "image_paths")]
        elif "image_path" in x and x["image_path"] is not None:
            imgs = [PIL.Image.open(p) for p in get_key_from_inputs(x, "image_path")]
        elif "image" in x and x["image"] is not None:
            imgs = [PIL.Image.open(p) for p in get_key_from_inputs(x, "image")]
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
                        new_h = int(h * (28/w))
                    else:
                        new_h = 28
                        new_w = int(w * (28/h))
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
        generation_config=generation_config
    )
    prompt_ids = prompt_inputs["input_ids"]
    prompt_length =  prompt_inputs["input_ids"].size(1)

    completions = processing_class.batch_decode(generate_returned_result, skip_special_tokens=True)

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