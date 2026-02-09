import json
import PIL
import os
import sys

import torch
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from transformers import AutoConfig, AutoModelForCausalLM, AutoModel, AutoProcessor

# model = AutoModelForCausalLM.from_pretrained("/home/dalhxwlyjsuo/criait_tansy/GRPO_weights/LlavaQwen2-GRPO-Tricks-Total-CoT-6000")
# vllm 不支持 自定义模型类。但支持 Qwen-VL.
# ValueError: LlavaQwen2ForCausalLM has no vLLM implementation and the Transformers implementation is not compatible with vLLM.
from vllm import LLM, SamplingParams
MODEL_PATH = '/home/dalhxwlyjsuo/criait_tansy/weight/Qwen2.5-VL-32B-Instruct'
data_name = 'Kvasir-en'
model_name = 'Qwen2.5-VL-32B-Instruct' # 2: is_message=True
is_message = True
data_path = f'/home/dalhxwlyjsuo/criait_tansy/jmf/ds-vl-on-policy-test/{data_name}.json'
data_save_path = f'/home/dalhxwlyjsuo/criait_tansy/jmf/GRPO_result/{model_name}/{data_name}-{is_message}.json'

llm = LLM(MODEL_PATH,
          gpu_memory_utilization=0.6,
          tensor_parallel_size=1,
          limit_mm_per_prompt={"image": 10, "video": 0},
          dtype=torch.bfloat16)
print('加载成功')

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=256,
    stop_token_ids=[],
)

dataset = json.load(open(data_path,'r'))
# sbatch -p vip_gpu_01 --gpus=1 

processor = AutoProcessor.from_pretrained(MODEL_PATH)


max_new_tokens=512
do_sample=True
temperature=0.2
pad_token_id= processor.tokenizer.pad_token_id
results = []
print('data save path is:', data_save_path)
for data in dataset:
    image_messages = [
        {"role": "system", "content": "A chat between a curious user and a Medical AI assistant in the field of Endoscopy and Human Stomach. "
            "As the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, and use natural language to assist users in various tasks."
            "The visual content will be provided with the following format: <|Image_start|>visual content<|Image_end|>."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": data['image_paths'],
                    "min_pixels": 224 * 224,
                    "max_pixels": 1280 * 28 * 28,
                },
                {"type": "text", "text": data['formatted_text']},
            ],
        },
    ]
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
# prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"
# 
# # Load the image using PIL.Image
# # 使用 PIL.Image 加载图像
# image = PIL.Image.open(
#     '/home/dalhxwlyjsuo/criait_tansy/project/Multi-Modality-Arena-main/Multi-Modality-Arena-main/OmniMedVQA/OmniMedVQA/Images/kvasir-dataset-v2/normal-pylorus/7e679490-9b55-4fbb-b98a-a5a612f737fb.jpg')
# 
# samplingparams = SamplingParams(
#     temperature=0.7,
#     top_p=0.95,
#     max_tokens=2048
# 
# )
# 
# # Single prompt inference
# # 单提示词推理
# outputs = llm.generate(
#     {
#         "prompt": prompt,
#         "multi_modal_data": {"image": image},
#     },
#     samplingparams,
#     lora_request=None)
# 
# for o in outputs:
#     generated_text = o.outputs[0].text
#     print(generated_text)

# llm = LLM(
#     model='/home/dalhxwlyjsuo/criait_tansy/GRPO_weights/LlavaQwen2-GRPO-Tricks-Total-CoT-6000',
#     trust_remote_code=True,  # Required to load Phi-3.5-vision
#     max_model_len=4096,  # Otherwise, it may not fit in smaller GPUs
#     limit_mm_per_prompt={"image": 2},  # The maximum number to accept
#     gpu_memory_utilization=0.6
# )
#
# # Refer to the HuggingFace repo for the correct format to use
# # 参考 HuggingFace 仓库以使用正确的格式
# prompt = "<|user|>\n<|image_1|>\n<|image_2|>\nWhat is the content of each image?<|end|>\n<|assistant|>\n"
#
#
# # Load the images using PIL.Image
# # 使用 PIL.Image 加载图像
# image1 = PIL.Image.open(...)
# image2 = PIL.Image.open(...)
#
#
# outputs = llm.generate({
#     "prompt": prompt,
#     "multi_modal_data": {
#         "image": [image1, image2]
#     },
# })
#
#
# for o in outputs:
#     generated_text = o.outputs[0].text
#     print(generated_text)