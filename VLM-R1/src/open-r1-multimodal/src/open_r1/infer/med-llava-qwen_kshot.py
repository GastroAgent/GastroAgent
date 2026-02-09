import sys
import gc
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
import random
import json
import PIL
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, GenerationConfig
from numpy import argmin
from torchvision import transforms
import pandas as pd
from cal_Single_sample_with_T import create_generator
sys.path.append('/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1')  # 服务器调试
from llava.model.llavaProcessor import LlavaProcessor
from llava.model.language_model.llava_qwen_mul import PloyLlavaLlamaForCausalLM
from trl.data_utils import maybe_apply_chat_template
from tqdm import tqdm

dataset = json.load(open('/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/Kvasir/Kvasir-en.json', 'r'))
questions = [question for question in dataset if not len(question['image_paths']) > 10]

data_name = 'Kvasir-en'
batch_size = 1
model_name = 'Med-Llava-Qwen2-7B-Final'
model_id = f'/mnt/inaisfs/data/home/tansy_criait/weights/{model_name}'

k = 5
# dir_path = '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match/data/exam_images/images_options' # All
dir_path = '/mnt/inaisfs/data/home/tansy_criait/flow_match/data/exam_images_backup/images_options' # Disease
answers_file = f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/results/Kvasir/{model_name}/Kavisr-en-{k}shot_old_attention_Disease_Whole.json'
print(answers_file)
os.makedirs(f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/results/Kvasir/{model_name}', exist_ok=True)
student_answers_file = answers_file.replace('.json', '_student.json')
try:
    results = json.load(open(answers_file))
    students = json.load(open(student_answers_file))
    have_questions = [x['question_id'] for x in students]
except:
    have_questions = []
    results = []
    students = []
    
model_init_kwargs = {
    'attn_implementation': 'flash_attention_2',
    'torch_dtype': 'bfloat16',  # float32, bfloat16 # flash attention 只支持bfloat16。
    'use_cache': True,
    'max_memory': {0:"40GiB"},
}

model = PloyLlavaLlamaForCausalLM.from_pretrained(
    model_id,
    **model_init_kwargs
).to("cuda:0")  # 显式将模型移动到当前设备

processing_class = LlavaProcessor.from_pretrained(model_id,
                                                  use_fast=True,
                                                  trust_remote_code=model_init_kwargs.get("trust_remote_code", None),
                                                  patch_size=14)

processing_class.tokenizer.padding_side = 'left'
processing_class.chat_template = processing_class.tokenizer.chat_template

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
sys_prompt = (
        "You are a helpful Medical AI assistant and an authoritative expert in the field of the medicine. "
        "As the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, and use natural language to assist users in various tasks. "
        "As the expert, you need to make sure that your response is logical and factual, and that it conveys the authority of the expert. "
        "The visual content (most about medical imaging) will be provided with the following format: <|Image_start|>visual content<|Image_end|>." ### Qwen不需要这个。
)

user_prompt = '''
Your task is to revise and supplement the provided reference answers to medical questions, based on medical expertise. The reference response are derived based on medical image similarity and may contain errors or be incomplete. 
You will need:
 - Identify and correct medical errors or inaccuracies in reference response.
 - Supplementary references to key medical evidence, diagnostic criteria, or treatment recommendations missing from the response
 - Ensure that the final answer is complete, accurate, professional, in line with current medical guidelines, and can correctly answer medical questions
 - Please directly output the complete answer after correction, without additional explanation of the reason for correction.


{question}

The reference response:
{reference}
'''
# Note: the reference response is not reliable when it comes to issues that are not of a disease-diagnosis; it is merely a reference suggestion.
print(model_name)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_grey = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.RandomGrayscale(p=1),  # 数据增强：50% 概率灰度化
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
label_map = json.load(open('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match/utils/label_map.json', 'r'))
caption_templete = "将当前未知的医疗病症图像 转化为 {disease} 等相关病症图像。"
mode='neighbor'
generator = create_generator()
id2option = {0: "option_A", 1: "option_B", 2: "option_C", 3: "option_D"}
question_ids = json.load(open("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match/data/exam_dataset_419.json"))
question_ids = [x['question_id'] for x in question_ids]
for idx, question in tqdm(enumerate(questions)):
    question_type = question['question_type']
    question_id = question['question_id'].replace(" ", "_")
    if question_id not in question_ids:
        continue
    if question_id in have_questions:
        continue

    if question_type != 'Disease Diagnosis': # 'Modality Recognition' 'Anatomy Identification' 'Disease Diagnosis'
        continue

    option_A = question['option_A'].replace(" ", "_")
    option_B = question['option_B'].replace(" ", "_")
    option_C = question['option_C'].replace(" ", "_")
    option_D = question['option_D'].replace(" ", "_")
    score_A = 0
    score_B = 0
    score_C = 0
    score_D = 0
    x0_path = question['image_paths'][0]

    ########## 计算 选项A 的Wass距离 ###########
    try:
        option_A_dir = os.path.join(dir_path, option_A)
        option_A_images = os.listdir(option_A_dir)
        option_A_images = random.sample(option_A_images, k=k) if len(option_A_images) > k else option_A_images
        caption = caption_templete.format(disease=option_A)
        try:
            label_id = label_map[option_A]
        except:
            label_id = len(label_map)
            label_map[option_A] = label_id
        for option_A_image in option_A_images:
            x1_path = os.path.join(option_A_dir, option_A_image)
            if k > 0:
                conds = generator.get_conds_from_items(x0_path, x1_path, [x0_path], [x1_path], label_id, caption, transform, transform_grey)
                result = generator.cal_SingleWass_with_T(conds, mode=mode)
                if mode == 'neighbor':
                    score_A += sum([(x - 1.*y) for x, y in zip(result['Neighbor_distances_sinkhorn_w2_latent_mapped_all'], result['Neighbor_distances_sinkhorn_w2_latent_mapped_bias1'])])
                else:
                    score_A += result['distances_sinkhorn_w2_latent']
    except:
        # continue
        score_A += 99

    ########## 计算 选项B 的Wass距离 ###########
    try:
        option_B_dir = os.path.join(dir_path, option_B)
        option_B_images = os.listdir(option_B_dir)
        option_B_images = random.sample(option_B_images, k=k) if len(option_B_images) > k else option_B_images
        caption = caption_templete.format(disease=option_B)
        try:
            label_id = label_map[option_B]
        except:
            label_id = len(label_map)
            label_map[option_B] = label_id
        for option_B_image in option_B_images:
            x1_path = os.path.join(option_B_dir, option_B_image)
            if k > 0:
                conds = generator.get_conds_from_items(x0_path, x1_path, [x0_path], [x1_path], label_id, caption, transform, transform_grey)
                result = generator.cal_SingleWass_with_T(conds, mode=mode)
                if mode == 'neighbor':
                    score_B += sum([(x - 1.*y) for x, y in zip(result['Neighbor_distances_sinkhorn_w2_latent_mapped_all'], result['Neighbor_distances_sinkhorn_w2_latent_mapped_bias1'])])
                else:
                    score_B += result['distances_sinkhorn_w2_latent']
    except:
        # continue
        score_B += 99

    ########## 计算 选项C 的Wass距离 ###########
    try:
        option_C_dir = os.path.join(dir_path, option_C)
        option_C_images = os.listdir(option_C_dir)
        option_C_images = random.sample(option_C_images, k=k) if len(option_C_images) > k else option_C_images
        caption = caption_templete.format(disease=option_C)
        try:
            label_id = label_map[option_C]
        except:
            label_id = len(label_map)
            label_map[option_C] = label_id
        for option_C_image in option_C_images:
            x1_path = os.path.join(option_C_dir, option_C_image)
            if k > 0:
                conds = generator.get_conds_from_items(x0_path, x1_path, [x0_path], [x1_path], label_id, caption, transform, transform_grey)
                result = generator.cal_SingleWass_with_T(conds, mode=mode)
                if mode == 'neighbor':
                    score_C += sum([(x - 1.*y) for x, y in zip(result['Neighbor_distances_sinkhorn_w2_latent_mapped_all'], result['Neighbor_distances_sinkhorn_w2_latent_mapped_bias1'])])
                else:
                    score_C += result['distances_sinkhorn_w2_latent']
    except:
        # continue
        score_C += 99

    ########## 计算 选项D 的Wass距离 ###########
    try:
        option_D_dir = os.path.join(dir_path, option_D)
        option_D_images = os.listdir(option_D_dir)
        option_D_images = random.sample(option_D_images, k=k) if len(option_D_images) > k else option_D_images
        caption = caption_templete.format(disease=option_D)
        try:
            label_id = label_map[option_D]
        except:
            label_id = len(label_map)
            label_map[option_D] = label_id
        for option_D_image in option_D_images:
            x1_path = os.path.join(option_D_dir, option_D_image)
            if k > 0:
                conds = generator.get_conds_from_items(x0_path, x1_path, [x0_path], [x1_path], label_id, caption, transform, transform_grey)
                result = generator.cal_SingleWass_with_T(conds, mode=mode)
                if mode == 'neighbor':
                    score_D += sum([(x - 1.*y) for x, y in zip(result['Neighbor_distances_sinkhorn_w2_latent_mapped_all'], result['Neighbor_distances_sinkhorn_w2_latent_mapped_bias1'])])
                else:
                    score_D += result['distances_sinkhorn_w2_latent']
    except:
        # continue
        score_D += 99
        
    if k > 0:
        student_id = int(argmin([score_A, score_B, score_C, score_D]))
        student_option = id2option[student_id]
        student_answer = question[student_option] 
        students.append({
            "question_id": question["question_id"],
            "student_id": student_id,
            "student_option": student_option,
            "student_answer": student_answer,
            "gt_answer": question['gt_answer'],
            "answer": question['answer']
        })
        
    if k > 0:
        ref = f"""The image presented by the current problem is the most similar to the image content of the “{student_option}: {student_answer}”."""
    else:
        ref = "Sorry, I know nothing about it."

    text = question['formatted_text'].replace("Question: ", "")
    text = user_prompt.format(question=text, reference=ref)

    example = {
        'prompt':
            [{
                'role': 'system',
                'content': sys_prompt
            },
            {
                'role': 'user',
                'content': text,  # Qwen2 纯文本提示词
            }]
    }
    prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"]]

    images = []
    for x in [question]:
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

    model_response = processing_class.tokenizer.batch_decode(
        generate_returned_result, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    if isinstance(model_response, list):
        model_response = model_response[0]
    question['generated_text'] = model_response
    print(model_response)
    results.append({
        'generated_text': model_response,
        'prompt_text': text,
        "answer": question['answer'],
        "gt_answer": question['gt_answer'],
        "image_paths": question['image_paths'],
        "question_id": question['question_id'],
        "question_type": question['question_type'],
        "option_A": question['option_A'],
        "option_B": question['option_B'],
        "option_C": question['option_C'],
        "option_D": question['option_D'],
    })
    if idx % 100 == 0:
        with open(student_answers_file, 'w') as f:
            json.dump(students, f, indent=4, ensure_ascii=False)
        with open(answers_file, 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

with open(student_answers_file, 'w') as f:
    json.dump(students, f, indent=4, ensure_ascii=False)
with open(answers_file, 'w') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
print('Infer Done!')

