"""
================================================================================
步骤1: 模型推理 + Trigger策略分析（支持动态选项数量）
================================================================================
功能: 对输入数据进行模型推理，计算概率分布和trigger相关指标
输入: new_eval_tsy_llm.json (原始问答数据)
输出: new_eval_tsy_llm_with_trigger.json (推理结果 + trigger字段)
"""

import os
import random
import json
import math
import PIL
import re
import sys
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, GenerationConfig

os.environ['CUDA_VISIBLE_DEVICES'] = '5,4'

sys.path.append('/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1')
from llava.model.llavaProcessor import LlavaProcessor
from llava.model.language_model.llava_qwen_mul_workflow import PloyLlavaLlamaForCausalLM
from trl.data_utils import maybe_apply_chat_template

# ===== 配置参数 =====
model_id = '/mnt/inaisfs/data/home/tansy_criait/weights/LlavaQwen2-GRPO-Tricks-Total-CoT-6000'
data_name = 'kvasir'
batch_size = 1  # 建议设为1以支持不同选项数量
model_name = 'LlavaQwen2-GRPO-Tricks-Total-CoT-6000'

# 输入输出路径
input_data_path = f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/kvasir_doctor_exam/new_eval_tsy_llm_62.json'
output_data_path = f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/kvasir_doctor_exam/cot-419-0.4/new_eval_tsy_llm_with_trigger.json'

lora_model_path = None
model_save_path = None

os.makedirs(os.path.dirname(output_data_path), exist_ok=True)

# ===== 支持的最大选项数量 =====
ALL_CHOICE_TOKENS = ["A", "B", "C", "D", "E", "F", "G", "H"]

# ===== 问题类型定义 =====
LONG_QTYPES = {"Modality Recognition", "Anatomy Identification"}
TAIL_QTYPES = {"Disease Diagnosis"}

# ===== Trigger策略阈值 =====
# 旧版策略
THETA_U = 0.55  # 熵阈值
THETA_P = 0.55  # 置信度阈值

# 新版三层过滤策略
THRES_MAX_P = 0.63      # 最大概率阈值
THRES_GAP = 0.20        # 概率差距阈值
THRES_ENTROPY = 0.4    # 归一化熵阈值

# ===== 工具函数 =====
def get_available_options(example):
    """
    检测样本中实际存在的选项
    返回: 选项字母列表，如 ['A', 'B'] 或 ['A', 'B', 'C', 'D']
    """
    available = []
    for letter in ALL_CHOICE_TOKENS:
        option_key = f'option_{letter}'
        if option_key in example and example[option_key] is not None and example[option_key] != '':
            available.append(letter)
    return available

def get_key_from_inputs(x: dict, key: str):
    """从输入字典中获取指定key的值，确保返回列表格式"""
    ele = x.get(key, None)
    return ele if isinstance(ele, list) else [ele]

def extract_answer_letter(text: str, valid_choices=None):
    """
    从模型输出中提取答案选项字母
    valid_choices: 有效选项列表，如 ['A', 'B']，用于验证提取的答案是否有效
    """
    if valid_choices is None:
        valid_choices = ALL_CHOICE_TOKENS

    # 构建正则表达式，只匹配有效选项
    choices_pattern = '|'.join(valid_choices)

    # 尝试匹配 <answer>option_X</answer> 格式
    m = re.search(rf"<answer>\s*option[_\s]*({choices_pattern})\b[^<]*</answer>",
                  text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 尝试匹配 <answer>X</answer> 格式
    m = re.search(rf"<answer>\s*({choices_pattern})\b[^<]*</answer>",
                  text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None

# ===== 模型初始化 =====
print("正在加载模型...")
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

if lora_model_path is not None:
    model = PeftModel.from_pretrained(model, lora_model_path)
    model = model.merge_and_unload()

processing_class = LlavaProcessor.from_pretrained(
    model_id,
    use_fast=True,
    trust_remote_code=model_init_kwargs.get("trust_remote_code", None),
    patch_size=14
)
processing_class.tokenizer.padding_side = 'left'
processing_class.chat_template = processing_class.tokenizer.chat_template

if model_save_path is not None:
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    processing_class.save_pretrained(model_save_path)

print("模型加载完成!")

# ===== 预计算所有可能选项的token ID =====
all_choice_ids_map = {}
for ch in ALL_CHOICE_TOKENS:
    ids = processing_class.tokenizer(ch, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        print(f"[WARN] 选项 {ch} 分成了多 token: {ids}")
    all_choice_ids_map[ch] = ids[0]

print(f"选项token映射: {all_choice_ids_map}")

# ===== 主推理流程 =====
print(f"正在读取数据: {input_data_path}")
dataset = json.load(open(input_data_path, 'r'))
print(f"数据集大小: {len(dataset)}")

generation_config = GenerationConfig(
    max_new_tokens=512,
    do_sample=True,
    temperature=0.2,
    pad_token_id=processing_class.tokenizer.pad_token_id,
)

results = []
print(f'输出路径: {output_data_path}')
print("开始推理...\n")

# 系统提示词
system_prompt = (
    "You are a helpful Medical AI assistant and authoritative expert in the medical field. "
    "You have a solid foundation in medicine and are proficient in answering various visual questions related to medical topics. "
    "As the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, "
    "and use natural language to assist users in various tasks. "
    "As the expert, you need to make sure that your response is logical and factual, and that it conveys the authority of the expert. "
    "The visual content (most about medical imaging) will be provided with the following format: <|Image_start|>visual content<|Image_end|>."
)

for i in range(0, len(dataset), batch_size):
    inputs = dataset[i:i+batch_size]
    print(f"处理进度: {i+1}/{len(dataset)}")

    # ===== 检测每个样本的可用选项 =====
    batch_available_options = []
    for example in inputs:
        available = get_available_options(example)
        if len(available) < 2:
            print(f"[WARN] 问题 {example.get('question_id', 'unknown')} 只有 {len(available)} 个选项，跳过")
            continue
        batch_available_options.append(available)

    # 如果batch中所有样本都被跳过，继续下一个batch
    if len(batch_available_options) == 0:
        continue

    # 过滤掉选项不足的样本
    valid_inputs = [inp for inp, opts in zip(inputs, batch_available_options) if len(opts) >= 2]

    # ===== 检查batch内选项数量是否一致 =====
    option_counts = [len(opts) for opts in batch_available_options]
    if len(set(option_counts)) > 1:
        # 选项数量不一致，需要逐个处理
        print(f"[INFO] Batch内选项数量不一致: {option_counts}，将逐个处理")
        # 递归处理每个样本（batch_size=1）
        for single_input in valid_inputs:
            # 这里可以递归调用，但为了简化，我们直接在循环中处理
            pass

    # 使用第一个样本的选项配置（假设batch内一致，或batch_size=1）
    current_choice_tokens = batch_available_options[0]
    current_choice_ids = torch.tensor(
        [all_choice_ids_map[ch] for ch in current_choice_tokens],
        device=model.device
    )

    print(f"当前batch选项: {current_choice_tokens}")

    # 构建prompt
    for example in valid_inputs:
        example['prompt'] = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': example['formatted_text']}
        ]

    prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in valid_inputs]

    # 处理图像
    images = []
    for x in valid_inputs:
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

    # ===== 计算概率分布和trigger策略 =====
    with torch.no_grad():
        # 获取概率分布（使用当前batch的选项）
        p_G, outputs = model.get_pG_from_inputs(
            candidate_ids=current_choice_ids,
            **prompt_inputs
        )

        # 标记问题类型
        is_long_list = [inp["question_type"] in LONG_QTYPES for inp in valid_inputs]
        is_tail_list = [inp["question_type"] in TAIL_QTYPES for inp in valid_inputs]
        is_long_qtype = torch.tensor(is_long_list, dtype=torch.bool, device=p_G.device)
        is_tail_qtype = torch.tensor(is_tail_list, dtype=torch.bool, device=p_G.device)

        # 旧版Trigger policy
        policy = model.trigger_policy_from_pG(
            p_G=p_G,
            tail_indices=None,
            theta_u=THETA_U,
            theta_p=THETA_P,
            ood_distance=None,
            theta_ood=None,
            rule_based_flag=is_tail_qtype,
        )

        # 转移旧版trigger相关数据到CPU
        p_G_cpu = p_G.detach().float().cpu()
        trigger_mask = policy["trigger_mask"].detach().cpu().numpy()
        high_entropy = policy["high_entropy"].detach().cpu().numpy()
        low_confidence = policy["low_confidence"].detach().cpu().numpy()
        ood_flag = policy["ood_flag"].detach().cpu().numpy()
        rule_flag = policy["rule_flag"].detach().cpu().numpy()
        entropy_tensor = policy["entropy"]  # 保留在GPU上用于后续计算

        # ===== 新版三层过滤模型计算 =====
        # 1. 基础概率指标
        max_p_val, argmax_i = p_G.max(dim=-1)
        sorted_p, _ = torch.sort(p_G, descending=True, dim=-1)

        # 处理只有2个选项的情况
        if sorted_p.shape[-1] >= 2:
            prob_gap = sorted_p[:, 0] - sorted_p[:, 1]
        else:
            prob_gap = torch.ones_like(max_p_val)  # 只有1个选项时，gap设为1

        # 计算归一化熵
        K = len(current_choice_tokens)
        H_max = math.log(K) if K > 1 else 1.0
        H_norm = entropy_tensor / H_max if H_max > 0 else entropy_tensor

        # 转移到CPU
        max_p_val_cpu = max_p_val.detach().float().cpu()
        prob_gap_cpu = prob_gap.detach().float().cpu()
        H_norm_cpu = H_norm.detach().float().cpu()
        argmax_i_cpu = argmax_i.detach().cpu()
        entropy_cpu = entropy_tensor.detach().float().cpu()

    # ===== 生成答案 =====
    generate_returned_result = model.generate(
        **prompt_inputs,
        generation_config=generation_config
    )
    completions = processing_class.batch_decode(generate_returned_result, skip_special_tokens=True)

    # ===== 综合判定trigger_final =====
    trigger_final_list = []
    for b_idx in range(len(valid_inputs)):
        qt = valid_inputs[b_idx]["question_type"]
        is_tail = (qt in TAIL_QTYPES)

        trigger_val = (
            (max_p_val_cpu[b_idx].item() < THRES_MAX_P) or
            (prob_gap_cpu[b_idx].item() < THRES_GAP) or
            (H_norm_cpu[b_idx].item() > THRES_ENTROPY)
        )
        trigger_final_list.append(bool(trigger_val))

    # ===== 处理每个样本的结果 =====
    for b_idx, (generated_text, input_, prompt_text) in enumerate(zip(completions, valid_inputs, prompts_text)):
        # 获取当前样本的可用选项
        available_opts = batch_available_options[b_idx]

        # 提取概率分布（只保存实际存在的选项）
        probs = p_G_cpu[b_idx].tolist()
        prob_dict = {f"p_{ch}": float(probs[j]) for j, ch in enumerate(available_opts) if j < len(probs)}

        # 提取问题类型
        qt = input_["question_type"]
        is_long = (qt in LONG_QTYPES)
        is_tail = (qt in TAIL_QTYPES)

        # 新版指标
        max_p_val = float(max_p_val_cpu[b_idx].item())
        prob_gap_val = float(prob_gap_cpu[b_idx].item())
        H_norm_val = float(H_norm_cpu[b_idx].item())
        argmax_idx = int(argmax_i_cpu[b_idx].item())
        entropy_val = float(entropy_cpu[b_idx].item())
        trigger_final_val = trigger_final_list[b_idx]

        # 计算辅助指标
        entropy_score = 1.0 - H_norm_val

        # 临时提取答案用于调试（注意：步骤2会用LLM重新提取）
        temp_answer_letter = extract_answer_letter(generated_text, valid_choices=available_opts)

        # 构建结果字典（只包含实际存在的选项）
        result_item = {
            # 基础信息
            'question_id': input_['question_id'],
            'question_type': qt,
            'is_long_qtype': is_long,
            'is_tail_qtype': is_tail,
            'image_paths': input_['image_paths'],
            'available_options': available_opts,  # 新增：记录可用选项

            # 问题和选项（只保存实际存在的选项）
            'formatted_text': input_['formatted_text'],
            'gt_answer': input_['gt_answer'],
        }

        # 动态添加选项字段
        for letter in ALL_CHOICE_TOKENS:
            option_key = f'option_{letter}'
            if option_key in input_:
                result_item[option_key] = input_[option_key]

        # 模型输出
        result_item.update({
            'prompt_text': prompt_text,
            'generated_text': generated_text,
            'temp_answer_letter': temp_answer_letter,

            # 概率分布
            **prob_dict,
            'pred_idx': argmax_idx,
            'pred_letter': available_opts[argmax_idx] if argmax_idx < len(available_opts) else None,

            # 新版三层过滤模型指标（核心）
            'max_prob': max_p_val,
            'prob_gap': prob_gap_val,
            'h_norm': H_norm_val,
            'entropy': entropy_val,
            'entropy_score': entropy_score,
            'trigger_final': trigger_final_val,
        })

        results.append(result_item)

        # 随机打印调试信息
        if random.random() < 0.05:
            print(f'\n--- 调试信息 ---')
            print(f'问题ID: {input_["question_id"]}, 可用选项: {available_opts}')
            print(f'问题类型: {qt}, is_tail={is_tail}')
            print(f'临时答案: {temp_answer_letter}, 预测索引: {argmax_idx}')
            print(f'概率分布: {prob_dict}')
            print(f'max_prob={max_p_val:.3f}, prob_gap={prob_gap_val:.3f}, H_norm={H_norm_val:.3f}')
            print(f'Trigger (新版): {trigger_final_val}')
            print('----------------\n')

# ===== 保存结果 =====
print(f"\n正在保存结果到: {output_data_path}")
with open(output_data_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print('=' * 80)
print('步骤1完成！')
print(f'共处理 {len(results)} 条数据')
print(f'输出文件: {output_data_path}')
print('=' * 80)
