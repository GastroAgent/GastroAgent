"""
================================================================================
数据准备脚本：提取中后层隐藏状态用于HALT探针训练 (优化版 v2.0)
================================================================================
功能:
1. 运行模型推理并提取特定层隐藏状态
2. 对比预测答案和标准答案，生成is_correct标签
3. 自动划分为训练集和验证集并保存

新增功能 (v2.0):
✨ 多层特征提取: 支持同时提取多个层的特征并拼接
✨ 数据平衡: 自动检查并平衡正负样本比例，防止类别不平衡
✨ 详细统计: 提供更详细的数据分布和配置信息

配置说明:
- USE_MULTI_LAYER: 是否启用多层特征提取
- HALT_LAYER_RATIOS: 要提取的层比例列表 (如 [0.5, 0.7, 0.85])
- ENABLE_BALANCE: 是否启用数据平衡
- BALANCE_RATIO: 正负样本最大比例 (如 1.5 表示较多类不超过较少类的1.5倍)
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
model_id = '/mnt/inaisfs/data/home/tansy_criait/weights/Llava-Qwen2-7B-tune-med-SFT-RL-New'

# 输入文件 (确保该文件已经包含了 gt_letter 字段)
input_data_path = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/halt/merge/shiguan/122/kvasir_shiguan_full_dataset_70.json'

# 输出目录 (脚本会自动生成 train_... 和 val_... 两个文件)
output_dir = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/halt/Llava-Qwen2-7B-tune-med-SFT-RL-New/merge_shiguan'

# 【关键修改1】多层特征提取配置
# 提取多个层的特征进行对比实验
# Qwen2-7B 有 32 层，可以提取中层、中后层、深层的特征
HALT_LAYER_RATIOS = [0.5, 0.7, 0.85]  # 中层(16层)、中后层(22层)、深层(27层)
USE_MULTI_LAYER = True  # 是否使用多层特征融合

# 【新增】数据平衡配置
ENABLE_BALANCE = True  # 是否启用数据平衡
BALANCE_RATIO = 1.5    # 正负样本最大比例（较多类/较少类 <= 1.5）

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

def extract_multi_layer_hidden_states(model, inputs, layer_ratios=[0.5, 0.7, 0.85]):
    """
    提取多层隐藏状态并拼接

    Args:
        model: 模型
        inputs: 输入数据
        layer_ratios: 要提取的层比例列表

    Returns:
        dict: 包含各层特征和融合特征的字典
    """
    num_layers = model.config.num_hidden_layers

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

    all_hidden_states = outputs.hidden_states

    # 提取各层特征
    layer_features = {}
    concat_features = []

    for ratio in layer_ratios:
        target_layer = min(int(num_layers * ratio), num_layers - 1)
        target_hidden = all_hidden_states[target_layer]
        last_token_hidden = target_hidden[:, -1, :]

        layer_features[f'layer_{int(ratio*100)}'] = last_token_hidden
        concat_features.append(last_token_hidden)

    # 拼接所有层的特征
    layer_features['concat'] = torch.cat(concat_features, dim=-1)

    return layer_features

def balance_dataset(results, balance_ratio=1.5, random_seed=42):
    """
    平衡正负样本

    Args:
        results: 数据列表
        balance_ratio: 正负样本最大比例（较多类/较少类）
        random_seed: 随机种子

    Returns:
        balanced_results: 平衡后的数据列表
    """
    random.seed(random_seed)

    # 分离正负样本
    correct_samples = [r for r in results if r['is_correct']]
    incorrect_samples = [r for r in results if not r['is_correct']]

    n_correct = len(correct_samples)
    n_incorrect = len(incorrect_samples)

    print(f"\n{'='*80}")
    print(f"数据平衡性检查")
    print(f"{'='*80}")
    print(f"原始数据分布:")
    print(f"  正样本 (is_correct=True):  {n_correct} ({n_correct/(n_correct+n_incorrect)*100:.1f}%)")
    print(f"  负样本 (is_correct=False): {n_incorrect} ({n_incorrect/(n_correct+n_incorrect)*100:.1f}%)")
    print(f"  正负比例: {max(n_correct, n_incorrect) / max(min(n_correct, n_incorrect), 1):.2f}:1")

    # 检查是否需要平衡
    if n_correct == 0 or n_incorrect == 0:
        print(f"⚠️  警告: 只有一种类别的样本，无法进行平衡！")
        return results

    ratio = max(n_correct, n_incorrect) / min(n_correct, n_incorrect)

    if ratio <= balance_ratio:
        print(f"✓ 数据已经相对平衡 (比例 {ratio:.2f} <= {balance_ratio})，无需调整")
        return results

    # 需要平衡：对多数类进行下采样
    print(f"\n⚠️  数据不平衡 (比例 {ratio:.2f} > {balance_ratio})，进行下采样...")

    if n_correct > n_incorrect:
        # 正样本过多，下采样正样本
        target_size = int(n_incorrect * balance_ratio)
        correct_samples = random.sample(correct_samples, target_size)
        print(f"  正样本: {n_correct} -> {len(correct_samples)}")
        print(f"  负样本: {n_incorrect} (保持不变)")
    else:
        # 负样本过多，下采样负样本
        target_size = int(n_correct * balance_ratio)
        incorrect_samples = random.sample(incorrect_samples, target_size)
        print(f"  正样本: {n_correct} (保持不变)")
        print(f"  负样本: {n_incorrect} -> {len(incorrect_samples)}")

    # 合并并打乱
    balanced_results = correct_samples + incorrect_samples
    random.shuffle(balanced_results)

    n_correct_new = len([r for r in balanced_results if r['is_correct']])
    n_incorrect_new = len([r for r in balanced_results if not r['is_correct']])

    print(f"\n平衡后数据分布:")
    print(f"  正样本: {n_correct_new} ({n_correct_new/len(balanced_results)*100:.1f}%)")
    print(f"  负样本: {n_incorrect_new} ({n_incorrect_new/len(balanced_results)*100:.1f}%)")
    print(f"  正负比例: {max(n_correct_new, n_incorrect_new) / min(n_correct_new, n_incorrect_new):.2f}:1")
    print(f"  总样本数: {len(results)} -> {len(balanced_results)}")
    print(f"{'='*80}\n")

    return balanced_results

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
if USE_MULTI_LAYER:
    print(f"开始提取多层隐藏状态 (Layers: {HALT_LAYER_RATIOS})...\n")
else:
    print(f"开始提取单层隐藏状态 (Layer Ratio: {HALT_LAYER_RATIOS[0]})...\n")

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
        if USE_MULTI_LAYER:
            # 提取多层特征
            layer_features = extract_multi_layer_hidden_states(
                model=model,
                inputs=prompt_inputs,
                layer_ratios=HALT_LAYER_RATIOS
            )
        else:
            # 提取单层特征
            target_hidden = extract_target_layer_hidden_states(
                model=model,
                inputs=prompt_inputs,
                layer_ratio=HALT_LAYER_RATIOS[0]
            )

        # 获取概率分布
        p_G, outputs = model.get_pG_from_inputs(
            candidate_ids=choice_ids,
            **prompt_inputs
        )

        # 预测答案
        argmax_i = p_G.argmax(dim=-1)

    # 转移到CPU
    if USE_MULTI_LAYER:
        layer_features_cpu = {k: v.detach().float().cpu() for k, v in layer_features.items()}
    else:
        target_hidden_cpu = target_hidden.detach().float().cpu()
    argmax_i_cpu = argmax_i.detach().cpu()

    # 处理每个样本
    for b_idx, input_ in enumerate(inputs):
        # 提取隐藏状态
        if USE_MULTI_LAYER:
            # 多层特征：保存各层特征和融合特征
            hidden_states_dict = {}
            for key, features in layer_features_cpu.items():
                hidden_states_dict[key] = features[b_idx].tolist()
        else:
            # 单层特征
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
        }

        # 添加隐藏状态特征
        if USE_MULTI_LAYER:
            # 多层模式：保存各层特征
            result_item['hidden_states'] = hidden_states_dict
            result_item['layer_ratios'] = HALT_LAYER_RATIOS
            result_item['use_multi_layer'] = True
        else:
            # 单层模式：保持原有格式兼容性
            result_item['middle_layer_hidden'] = hidden_states
            result_item['layer_ratio'] = HALT_LAYER_RATIOS[0]
            result_item['use_multi_layer'] = False

        results.append(result_item)

# ===== 【关键修改2】数据平衡与切分 =====
print(f"\n处理完成，共生成 {len(results)} 条数据。")

# 数据平衡
if ENABLE_BALANCE:
    results = balance_dataset(results, balance_ratio=BALANCE_RATIO, random_seed=random_seed)
else:
    print(f"\n数据平衡已禁用，跳过平衡步骤。")

print("\n正在随机切分训练集和验证集...")

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
print('=' * 80)
print(f'配置信息:')
print(f'  多层特征提取: {"启用" if USE_MULTI_LAYER else "禁用"}')
if USE_MULTI_LAYER:
    print(f'  提取层比例: {HALT_LAYER_RATIOS}')
else:
    print(f'  提取层比例: {HALT_LAYER_RATIOS[0]}')
print(f'  数据平衡: {"启用" if ENABLE_BALANCE else "禁用"}')
if ENABLE_BALANCE:
    print(f'  平衡比例: {BALANCE_RATIO}:1')
print()
print(f'数据集统计:')
print(f'  训练集: {len(train_data)} 条')
print(f'    - 正样本 (正确): {train_correct} ({train_correct/len(train_data)*100:.1f}%)')
print(f'    - 负样本 (错误): {len(train_data)-train_correct} ({(len(train_data)-train_correct)/len(train_data)*100:.1f}%)')
print(f'  验证集: {len(val_data)} 条')
print(f'    - 正样本 (正确): {val_correct} ({val_correct/len(val_data)*100:.1f}%)')
print(f'    - 负样本 (错误): {len(val_data)-val_correct} ({(len(val_data)-val_correct)/len(val_data)*100:.1f}%)')
print()
print(f'输出文件:')
print(f'  训练集: {train_output_path}')
print(f'  验证集: {val_output_path}')
print('=' * 80)