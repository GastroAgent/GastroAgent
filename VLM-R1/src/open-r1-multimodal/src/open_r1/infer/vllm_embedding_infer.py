
import gc
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import random
import json
import PIL
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, GenerationConfig
import sys

sys.path.append('/home/lab/work/VLM-R1/src/open-r1-multimodal/src/open_r1') # 服务器调试
from llava.model.llavaProcessor import LlavaProcessor
from llava.model.language_model.llava_qwen_mul import PloyLlavaLlamaForCausalLM
from trl.data_utils import maybe_apply_chat_template

model_id = '/dev/shm/jmf/mllm_weight/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-75'
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
    # system_prompt = (("You are a helpful Medical AI assistant and an expert in th e area of the Upper Digestive Tract and Digestive System. "
    #        "As the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, and use natural language to assist users in various tasks. "
    #        "As the expert, you need to make sure that your response is logical and factual, and that it conveys the authority of the expert. "
    #        "The visual content will be provided with the following format: <|Image_start|>visual content<|Image_end|>.")
    #                  if 'easy' not in data_name else "You are a helpful medical AI assistant.")

    system_prompt = (("You are a helpful Medical AI assistant and authoritative expert in the medical field. You have a solid foundation in medicine and are proficient in answering various visual questions related to medical topics. "
           "As the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, and use natural language to assist users in various tasks. "
           "As the expert, you need to make sure that your response is logical and factual, and that it conveys the authority of the expert. "
           "The visual content (most about medical imaging) will be provided with the following format: <|Image_start|>visual content<|Image_end|>.")
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

    ext = [example.replace('<image>', '<|Image_start><image><|Image_end|>') for example in prompts_text]

    images = []
    for x in inputs:
        if "image" in x and x["image"] is not None:
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
    
    from vllm import LLM

    # Inference with image embeddings as input
    llm = LLM(model="llava-hf/llava-1.5-7b-hf")

    # Refer to the HuggingFace repo for the correct format to use
    prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

    # Embeddings for single image
    # torch.Tensor of shape (1, image_feature_size, hidden_size of LM)
    image_embeds = torch.load(...)

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": image_embeds},
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    
    
    generate_returned_result = model.generate(
        **prompt_inputs,
        generation_config=generation_config
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