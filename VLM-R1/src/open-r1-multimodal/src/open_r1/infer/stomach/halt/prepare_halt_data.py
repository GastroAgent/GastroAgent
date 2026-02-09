"""
================================================================================
数据准备脚本：提取中后层隐藏状态用于HALT探针训练 (优化版)
================================================================================
功能:
1. 运行模型推理并提取特定层隐藏状态 (建议使用深层，如0.8)
2. 对比预测答案和标准答案，生成is_correct标签
3. 自动划分为训练集和验证集并保存
"""

import os
import json
import PIL
import sys
import torch
import random
import numpy as np
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm

# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'

sys.path.append('/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1')
from llava.model.llavaProcessor import LlavaProcessor
from llava.model.language_model.llava_qwen_mul_workflow import PloyLlavaLlamaForCausalLM
from trl.data_utils import maybe_apply_chat_template

# ===== 配置参数 =====
model_id = '/mnt/inaisfs/data/home/tansy_criait/weights/tsy/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-all-RL-2500'

# 输入文件 (确保该文件已经包含了 gt_letter 字段)
input_data_path = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/halt/merge/stomach/213/kvasir_stomach_full_dataset_143.json'

# 输出目录 (脚本会自动生成 train_... 和 val_... 两个文件)
output_dir = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/halt/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-all-RL-2500/merge_stomach_143'

# 【关键修改1】调整层数比例
# 0.5 (中间层) -> 0.85 (深层)。
# Qwen2-7B 有 32 层，0.85 * 32 ≈ 27层。深层通常包含更多关于答案正确性的语义信息。
HALT_MIDDLE_LAYER_RATIO = 0.85 

batch_size = 2
val_split_ratio = 0.2  # 验证集比例
random_seed = 42       # 随机种子

# ===== 初始化 =====
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

os.makedirs(output_dir, exist_ok=True)
train_output_path = os.path.join(output_dir, 'train_with_hidden_states.json')
val_output_path = os.path.join(output_dir, 'val_with_hidden_states.json')

# ===== 加载模型 =====
print(f"正在加载模型: {model_id}")
model_init_kwargs = {
    'attn_implementation': 'flash_attention_2',
    'torch_dtype': 'bfloat16',
    'use_cache': True,
}

model = PloyLlavaLlamaForCausalLM.from_pretrained(
    model_id,
    device_map='auto',
    **model_init_kwargs
)

processing_class = LlavaProcessor.from_pretrained(
    model_id,
    use_fast=True,
    trust_remote_code=model_init_kwargs.get("trust_remote_code", None),
    patch_size=14
)
processing_class.tokenizer.padding_side = 'left'
processing_class.chat_template = processing_class.tokenizer.chat_template

print("模型加载完成!")

# ===== 选项token ID准备 =====
choice_tokens = ["A", "B", "C", "D"]
choice_ids = []
for ch in choice_tokens:
    ids = processing_class.tokenizer(ch, add_special_tokens=False)["input_ids"]
    choice_ids.append(ids[0])
choice_ids = torch.tensor(choice_ids, device=model.device)

# ===== 工具函数 =====
def get_key_from_inputs(x: dict, key: str):
    ele = x.get(key, None)
    return ele if isinstance(ele, list) else [ele]

def extract_target_layer_hidden_states(model, inputs, layer_ratio=0.85):
    """提取特定层的隐藏状态"""
    num_layers = model.config.num_hidden_layers
    # 确保索引不超过最大层数
    target_layer = min(int(num_layers * layer_ratio), num_layers - 1)
    
    # print(f"  [Debug] Extracting Layer: {target_layer}/{num_layers}")

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

    all_hidden_states = outputs.hidden_states
    # hidden_states tuple通常包含输入embedding，所以长度是num_layers+1
    # 我们直接取 output_hidden_states[target_layer + 1] 或者简单按索引
    # Transformers输出通常 index 0 是 embedding，1 是 layer 1 output...
    # 这里我们直接取 target_layer (从0开始计数，对应第 target_layer+1 层)
    target_hidden = all_hidden_states[target_layer]
    
    # 取最后一个 token (EOS/Last Valid Token) 的特征
    last_token_hidden = target_hidden[:, -1, :]

    return last_token_hidden

# ===== 主流程 =====
print(f"正在读取数据: {input_data_path}")
if not os.path.exists(input_data_path):
    print(f"错误: 找不到输入文件 {input_data_path}")
    sys.exit(1)

dataset = json.load(open(input_data_path, 'r'))
print(f"数据集大小: {len(dataset)}")

# 简单检查数据格式
if 'gt_letter' not in dataset[0]:
    print("警告: 输入数据中缺少 'gt_letter' 字段，脚本可能会报错！")

system_prompt = (
    "You are a helpful Medical AI assistant and authoritative expert in the medical field. "
    "You have a solid foundation in medicine and are proficient in answering various visual questions related to medical topics. "
    "As the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, "
    "and use natural language to assist users in various tasks. "
    "As the expert, you need to make sure that your response is logical and factual, and that it conveys the authority of the expert. "
    "The visual content (most about medical imaging) will be provided with the following format: <|Image_start|>visual content<|Image_end|>."
)

results = []
print(f"开始提取隐藏状态 (Layer Ratio: {HALT_MIDDLE_LAYER_RATIO})...\n")

for i in tqdm(range(0, len(dataset), batch_size), desc="Processing"):
    inputs = dataset[i:i+batch_size]

    # 构建prompt
    for example in inputs:
        example['prompt'] = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': example['formatted_text']}
        ]

    prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]

    # 处理图像
    images = []
    for x in inputs:
        if "image_paths" in x and x["image_paths"] is not None:
            imgs = [PIL.Image.open(p).convert('RGB') for p in get_key_from_inputs(x, "image_paths")]
        else:
            imgs = []

        for img in imgs:
            try:
                w, h = img.size
                if w < 28 or h < 28:
                    if w < h:
                        new_w, new_h = 28, int(h * (28/w))
                    else:
                        new_h, new_w = 28, int(w * (28/h))
                    img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
            except Exception as e:
                print(f"图像处理错误: {e}")
            images.append(img)

    # 编码输入
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

    # 提取隐藏状态
    with torch.no_grad():
        target_hidden = extract_target_layer_hidden_states(
            model=model,
            inputs=prompt_inputs,
            layer_ratio=HALT_MIDDLE_LAYER_RATIO
        )

        # 获取概率分布
        p_G, outputs = model.get_pG_from_inputs(
            candidate_ids=choice_ids,
            **prompt_inputs
        )

        # 预测答案
        argmax_i = p_G.argmax(dim=-1)

    # 转移到CPU
    target_hidden_cpu = target_hidden.detach().float().cpu()
    argmax_i_cpu = argmax_i.detach().cpu()

    # 处理每个样本
    for b_idx, input_ in enumerate(inputs):
        # 提取隐藏状态
        hidden_states = target_hidden_cpu[b_idx].tolist()

        # 预测答案
        pred_idx = int(argmax_i_cpu[b_idx].item())
        pred_letter = choice_tokens[pred_idx] if pred_idx < len(choice_tokens) else None

        # 标准答案
        gt_letter = input_.get('gt_letter') # 使用 .get 防止报错
        if not gt_letter: 
            # 如果没有gt_letter，尝试从gt_answer反推或者跳过
            continue

        # 判断是否正确
        is_correct = (pred_letter == gt_letter)

        # 构建结果
        result_item = {
            'question_id': input_['question_id'],
            'question_type': input_['question_type'],
            'formatted_text': input_['formatted_text'],
            'gt_letter': gt_letter,
            'pred_letter': pred_letter,
            'is_correct': is_correct,
            'middle_layer_hidden': hidden_states,  # 这里保存的是深层特征
            'layer_ratio': HALT_MIDDLE_LAYER_RATIO,
        }

        results.append(result_item)

# ===== 【关键修改2】自动切分训练集和验证集 =====
print(f"\n处理完成，共生成 {len(results)} 条数据。")
print("正在随机切分训练集和验证集...")

# 打乱数据
random.shuffle(results)

# 计算切分点
split_idx = int(len(results) * (1 - val_split_ratio))
train_data = results[:split_idx]
val_data = results[split_idx:]

# 保存训练集
print(f"正在保存训练集 ({len(train_data)}条) 到: {train_output_path}")
with open(train_output_path, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)

# 保存验证集
print(f"正在保存验证集 ({len(val_data)}条) 到: {val_output_path}")
with open(val_output_path, 'w', encoding='utf-8') as f:
    json.dump(val_data, f, indent=4, ensure_ascii=False)

# ===== 统计信息 =====
train_correct = sum(1 for r in train_data if r['is_correct'])
val_correct = sum(1 for r in val_data if r['is_correct'])

print('\n' + '=' * 80)
print('数据准备与切分完成！')
print(f'训练集: {len(train_data)} (正确: {train_correct}, 错误: {len(train_data)-train_correct})')
print(f'验证集: {len(val_data)} (正确: {val_correct}, 错误: {len(val_data)-val_correct})')
print('=' * 80)