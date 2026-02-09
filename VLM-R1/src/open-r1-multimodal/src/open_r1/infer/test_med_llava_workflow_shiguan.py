"""
================================================================================
README: 医疗视觉问答任务推理脚本 (Medical VQA Inference Script)
================================================================================

【脚本概述】
本脚本用于医疗领域的视觉问答（VQA）任务推理，基于Llava-Qwen2多模态大模型
进行医疗图像问答评估。脚本实现了完整的推理流程，包括概率分布计算、置信度
评估和智能触发机制。

【主要功能模块】

1. 模型初始化与加载
   - 加载预训练的Llava-Qwen2-7B模型（医疗领域微调版本）
   - 支持LoRA权重加载和合并
   - 配置Flash Attention 2加速推理
   - 初始化图像和文本处理器

2. 数据预处理
   - 读取JSON格式的评估数据集
   - 构建多轮对话格式的prompt（system + user）
   - 处理医疗图像（支持多图像输入）
   - 图像尺寸检查和自动调整（最小28x28）

3. 概率分布计算 (p_G(c|q))
   - 计算模型对每个选项（A-H）的概率分布
   - 使用模型内部的get_pG_from_inputs方法
   - 提取选项token ID并计算logits概率

4. 三层过滤Trigger机制（核心创新）
   本脚本实现了部署安全的触发机制，不使用ground truth，通过以下三层过滤：
   s
   a) 基础概率指标
      - max_prob: 最高概率值（需 >= 0.65）
      - prob_gap: 第一名与第二名的概率差距（需 >= 0.20）
      - H_norm: 归一化熵（需 <= 0.40）
   
   b) 语义一致性检查（关键修正）
      - 比较模型生成的答案字母与概率分布argmax是否一致
      - 防止"嘴上说的和心里想的不一样"的情况
      - 如果生成答案与最大概率选项不一致，标记为不安全
   
   c) 综合判定
      - 满足任一不安全条件则触发（需要额外人工审核）
      - 不触发需同时满足：高概率 AND 大分差 AND 低熵 AND 逻辑一致

5. 答案生成与提取
   - 使用temperature=0.2的采样策略生成答案
   - 从生成文本中提取<answer>标签内的选项字母
   - 支持"option_A"和"A"两种格式的标准化

6. 置信度计算（部署安全）
   - conf_mix: 混合置信度 = 0.3 * max_prob + 0.7 * entropy_score
   - entropy_score: 熵分数 = 1.0 - H_norm
   - 不使用ground truth，完全基于模型输出

7. 结果评估与保存
   - 计算准确率（correct）：预测答案与ground truth的匹配
   - 记录所有概率分布（p_A, p_B, p_C, p_D等）
   - 保存完整的推理结果到JSON文件

【问题类型分类】
- LONG_QTYPES: {"Modality Recognition", "Anatomy Identification"} - 长问题类型
- TAIL_QTYPES: {"Disease Diagnosis"} - 尾部问题类型（高风险医疗诊断）

【关键参数配置】
- theta_u = 0.55: 熵阈值（用于基础trigger策略）
- theta_p = 0.55: 置信度阈值（用于基础trigger策略）
- THRES_MAX_P = 0.65: 最大概率阈值（三层过滤）
- THRES_GAP = 0.20: 概率差距阈值（三层过滤）
- THRES_ENTROPY = 0.40: 归一化熵阈值（三层过滤）
- temperature = 0.2: 生成温度（较低温度保证稳定性）

【输出字段说明】
- generated_text: 模型生成的完整文本
- answer_letter: 从生成文本中提取的答案字母
- gt_letter: 正确答案（从gt_answer反推）
- correct: 是否正确（0/1）
- p_chosen: 生成答案对应的概率值
- max_prob: 最大概率值
- prob_gap: 第一名与第二名的概率差距
- h_norm: 归一化熵
- is_consistent: 语义一致性（生成答案与argmax是否一致）
- trigger_final: 最终触发决策（True=需要额外处理）
- conf_mix: 混合置信度（部署安全指标）
- entropy_score: 熵分数（1.0 - h_norm）

【使用场景】
1. 医疗图像诊断辅助系统的评估
2. 模型置信度分析和质量控制
3. 高风险样本的自动识别和人工审核
4. 模型部署前的安全性评估

【注意事项】
- 脚本假设输入数据包含image_paths字段（可为None）
- 支持A-H共8个选项，但当前主要使用A-D
- 所有概率计算在GPU上进行以提高效率
- 结果保存为JSON格式，包含完整的推理信息

【文件路径配置】
- 模型路径: /mnt/inaisfs/data/home/tansy_criait/weights/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-75
- 数据路径: /mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/doctor_食管/new_eval_tsy_llm.json
- 输出路径: /mnt/inaisfs/data/home/tansy_criait/VLM-R1/results/doctor_食管_workflow_gemini/{model_name}/{data_name}.json

【技术亮点】
1. 部署安全的置信度计算（不依赖ground truth）
2. 三层过滤机制确保高风险样本被正确识别
3. 语义一致性检查防止模型内部矛盾
4. 支持多种问题类型的差异化处理
5. 完整的概率分布记录便于后续分析

================================================================================
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

sys.path.append('/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1')
from llava.model.llavaProcessor import LlavaProcessor
from llava.model.language_model.llava_qwen_mul_workflow import PloyLlavaLlamaForCausalLM
from trl.data_utils import maybe_apply_chat_template

# ===== 配置参数 =====
model_id = '/mnt/inaisfs/data/home/tansy_criait/weights/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-75'
data_name = '胃'
batch_size = 1
model_name = 'Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-75'

data_path = f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/doctor_胃/new_eval_tsy_llm.json'
lora_model_path = None
model_save_path = None

data_save_path = f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/results/doctor_胃_workflow_gemini/{model_name}/{data_name}.json'
os.makedirs(f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/results/doctor_胃_workflow_gemini/{model_name}', exist_ok=True)

# ===== 问题类型定义 =====
LONG_QTYPES = {"Modality Recognition", "Anatomy Identification"}
TAIL_QTYPES = {"Disease Diagnosis"}

# ===== 模型初始化 =====
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

# ===== 选项token ID准备 =====
choice_tokens = ["A", "B", "C", "D"]
choice_ids = []
for ch in choice_tokens:
    ids = processing_class.tokenizer(ch, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        print(f"[WARN] 选项 {ch} 分成了多 token: {ids}")
    choice_ids.append(ids[0])
choice_ids = torch.tensor(choice_ids, device=model.device)

# ===== 工具函数 =====
def extract_answer_letter(text: str):
    """从模型输出中提取答案选项字母（A~D）"""
    m = re.search(r"<answer>\s*option[_\s]*([A-D])\b[^<]*</answer>", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    
    m = re.search(r"<answer>\s*([A-D])\b[^<]*</answer>", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    
    return None

def get_gt_option(example):
    """根据gt_answer文本内容，反推出正确选项（如'option_A'）"""
    gt_text = example.get("gt_answer", None)
    if gt_text is None:
        return None
    
    for letter in "ABCDEFGH":
        key = f"option_{letter}"
        if key in example and example[key] == gt_text:
            return key
    return None

def normalize_letter(x: str | None):
    """标准化选项字母，支持'option_B'或'B'格式"""
    if x is None:
        return None
    s = str(x).strip()
    
    if s.lower().startswith("option_"):
        s = s.split("_", 1)[1]
    
    for ch in s:
        if ch.upper() in "ABCDEFGH":
            return ch.upper()
    return None

def get_key_from_inputs(x: dict, key: str):
    """从输入字典中获取指定key的值，确保返回列表格式"""
    ele = x.get(key, None)
    return ele if isinstance(ele, list) else [ele]

# ===== 主推理流程 =====
dataset = json.load(open(data_path, 'r'))
generation_config = GenerationConfig(
    max_new_tokens=512,
    do_sample=True,
    temperature=0.2,
    pad_token_id=processing_class.tokenizer.pad_token_id,
)

results = []
print(f'数据保存路径: {data_save_path}')

for i in range(0, len(dataset), batch_size):
    inputs = dataset[i:i+batch_size]
    print(f"处理进度: {i}/{len(dataset)}")

    # 构建prompt
    system_prompt = (
        "You are a helpful Medical AI assistant and authoritative expert in the medical field. "
        "You have a solid foundation in medicine and are proficient in answering various visual questions related to medical topics. "
        "As the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, "
        "and use natural language to assist users in various tasks. "
        "As the expert, you need to make sure that your response is logical and factual, and that it conveys the authority of the expert. "
        "The visual content (most about medical imaging) will be provided with the following format: <|Image_start|>visual content<|Image_end|>."
    )

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

    # ===== 计算概率分布和trigger策略 =====
    with torch.no_grad():
        p_G, outputs = model.get_pG_from_inputs(
            candidate_ids=choice_ids,
            **prompt_inputs
        )

        # 标记问题类型
        is_long_list = [inp["question_type"] in LONG_QTYPES for inp in inputs]
        is_tail_list = [inp["question_type"] in TAIL_QTYPES for inp in inputs]
        is_long_qtype = torch.tensor(is_long_list, dtype=torch.bool, device=p_G.device)
        is_tail_qtype = torch.tensor(is_tail_list, dtype=torch.bool, device=p_G.device)

        # Trigger policy配置
        theta_u = 0.55
        theta_p = 0.55

        policy = model.trigger_policy_from_pG(
            p_G=p_G,
            tail_indices=None,
            theta_u=theta_u,
            theta_p=theta_p,
            ood_distance=None,
            theta_ood=None,
            rule_based_flag=is_tail_qtype,
        )

        # 转移到CPU（保留用于后续分析）
        p_G_cpu = p_G.detach().float().cpu()
        trigger_mask = policy["trigger_mask"].detach().cpu().numpy()
        high_entropy = policy["high_entropy"].detach().cpu().numpy()
        low_confidence = policy["low_confidence"].detach().cpu().numpy()
        ood_flag = policy["ood_flag"].detach().cpu().numpy()
        rule_flag = policy["rule_flag"].detach().cpu().numpy()
        entropy_cpu = policy["entropy"].detach().float().cpu().tolist()
        max_prob_cpu = policy["max_prob"].detach().float().cpu().tolist()
        argmax_idx_cpu = policy["argmax_idx"].detach().cpu().tolist()
        
        # 保留policy中的entropy tensor用于后续计算（仍在GPU上）
        entropy_tensor = policy["entropy"]

    # ===== 生成答案 =====
    generate_returned_result = model.generate(
        **prompt_inputs,
        generation_config=generation_config
    )
    completions = processing_class.batch_decode(generate_returned_result, skip_special_tokens=True)

    # ===== 专家级 Trigger 策略修正（三层过滤模型）=====
    # 1. 基础概率提取（在GPU上计算，提高效率）
    max_p_val, argmax_i = p_G.max(dim=-1)
    sorted_p, _ = torch.sort(p_G, descending=True, dim=-1)
    prob_gap = sorted_p[:, 0] - sorted_p[:, 1]  # 第一名与第二名的差距
    
    # 计算归一化熵
    K = len(choice_tokens)
    H_max = math.log(K) if K > 1 else 1.0
    H_norm = entropy_tensor / H_max if H_max > 0 else entropy_tensor
    
    # 2. 语义一致性检查（关键修正）
    # 提取模型生成的选项对应的索引，与 Logits 的 argmax 进行比对
    matched_consistency = []
    for b_idx in range(len(inputs)):
        ans_letter = extract_answer_letter(completions[b_idx])
        if ans_letter in choice_tokens:
            gen_idx = choice_tokens.index(ans_letter)
            # 如果模型生成的答案字母与概率分布最高的字母不一致，则不安全
            is_consistent = (gen_idx == argmax_i[b_idx].item())
        else:
            # 如果无法提取答案字母，认为不一致
            is_consistent = False
        matched_consistency.append(is_consistent)
    is_consistent_tensor = torch.tensor(matched_consistency, device=p_G.device)
    
    # 3. 定义安全边界（可调参数）
    THRES_MAX_P = 0.65       # 提高门槛，确保不触发的样本极度自信
    THRES_GAP = 0.20         # 确保没有明显的干扰项
    THRES_ENTROPY = 0.40     # 归一化后的熵需低于此值
    
    # 4. 综合判定：满足以下任一条件则 Trigger（需额外处理）
    # 反之，如果不触发，则必须满足：高概率 AND 大分差 AND 低熵 AND 逻辑一致
    trigger_final = (
        (max_p_val < THRES_MAX_P) |           # 概率不够高
        (prob_gap < THRES_GAP) |              # 干扰项太强
        (H_norm > THRES_ENTROPY) |            # 分布太乱
        (~is_consistent_tensor)              # 嘴上说的和心里想的不一样
        #is_tail_qtype                         # 医疗高风险类别强制触发
    )
    
    # 转移到CPU用于后续处理
    trigger_final_cpu = trigger_final.detach().cpu().numpy()
    max_p_val_cpu = max_p_val.detach().float().cpu().tolist()
    prob_gap_cpu = prob_gap.detach().float().cpu().tolist()
    H_norm_cpu = H_norm.detach().float().cpu().tolist()
    argmax_i_cpu = argmax_i.detach().cpu().tolist()

    # ===== 处理每个样本的结果 =====
    for b_idx, (generated_text, input_, prompt_text) in enumerate(zip(completions, inputs, prompts_text)):
        input_['generated_text'] = generated_text

        # 提取概率分布
        probs = p_G_cpu[b_idx].tolist()
        prob_dict = {f"p_{ch}": float(probs[j]) for j, ch in enumerate(choice_tokens) if j < len(probs)}

        # 提取答案和ground truth
        answer_letter = extract_answer_letter(generated_text)
        gt_letter = get_gt_option(input_)
        norm_answer = normalize_letter(answer_letter)
        norm_gt = normalize_letter(gt_letter)
        correct = 1 if (norm_answer is not None and norm_gt is not None and norm_answer == norm_gt) else 0

        # 计算p_chosen（生成文本对应的概率）
        p_chosen = None
        if answer_letter is not None and answer_letter in choice_tokens:
            idx = choice_tokens.index(answer_letter)
            if idx < len(probs):
                p_chosen = float(probs[idx])

        # 使用新的三层过滤模型的触发决策
        trigger_final_val = bool(trigger_final_cpu[b_idx])
        
        # 保留旧版trigger标记用于兼容性分析
        trig = bool(trigger_mask[b_idx])
        trig_H = bool(high_entropy[b_idx])
        trig_P = bool(low_confidence[b_idx])
        trig_ood = bool(ood_flag[b_idx])
        qt = input_["question_type"]
        is_long = (qt in LONG_QTYPES)
        is_tail = (qt in TAIL_QTYPES)

        # 提取新策略计算的指标
        max_p_val = float(max_p_val_cpu[b_idx])
        prob_gap_val = float(prob_gap_cpu[b_idx])
        H_norm_val = float(H_norm_cpu[b_idx])
        argmax_i = int(argmax_i_cpu[b_idx])
        is_consistent = bool(matched_consistency[b_idx])
        
        # 部署安全的置信度计算（用于分析）
        ent_val = float(entropy_cpu[b_idx])
        entropy_score = 1.0 - H_norm_val

        alpha, beta = 0.3, 0.7
        conf_mix = alpha * max_p_val + beta * entropy_score
        theta_p_deploy = 0.58
        low_confidence1 = conf_mix < theta_p_deploy
        trig_P1 = bool(low_confidence1)

        # 评估用的置信度（包含correct，仅用于分析）
        alpha_eval, beta_eval = 0.7, 0.3
        conf_mix_with_correct = alpha_eval * max_p_val + beta_eval * correct

        # 代理指标：用于识别correct=1的样本（不直接使用correct，仅用于分析）
        if p_chosen is not None:
            proxy_score = 0.6 * p_chosen + 0.4 * entropy_score
        else:
            proxy_score = 0.6 * max_p_val + 0.4 * entropy_score
        
        # 保留旧版组合trigger逻辑用于对比分析
        prob_std = float(torch.std(torch.tensor(probs)).item()) if len(probs) > 1 else 0.0
        trig_uniform_dist = prob_std < 0.12
        sorted_probs = sorted(probs, reverse=True)
        if len(sorted_probs) >= 2:
            prob_gap_old = sorted_probs[0] - sorted_probs[1]
            trig_small_gap = prob_gap_old < 0.15
        else:
            prob_gap_old = 0.0
            trig_small_gap = False
        trig_entropy_extra = entropy_score < 0.35
        main_condition = trig_P1 or trig_H
        secondary_condition = trig_uniform_dist or trig_small_gap or trig_entropy_extra
        trig_combined = main_condition or (secondary_condition and trig)

        # 构建结果字典
        result_item = {
            'generated_text': generated_text,
            'prompt_text': prompt_text,
            "gt_answer": input_['gt_answer'],
            "answer_letter": answer_letter,
            "gt_letter": gt_letter,
            "correct": correct,
            "p_chosen": p_chosen,
            "entropy": ent_val,
            "max_prob": max_p_val,
            "pred_idx": argmax_i,
            "conf_mix": conf_mix,
            "conf_mix_with_correct": conf_mix_with_correct,
            "entropy_score": entropy_score,
            "question_type": qt,
            "is_long_qtype": is_long,
            "is_tail_qtype": is_tail,
            # 新版三层过滤模型指标
            "is_consistent": is_consistent,  # 语义一致性：生成答案与最大概率选项是否一致
            "prob_gap": prob_gap_val,  # 第一名与第二名的概率差距
            "h_norm": H_norm_val,  # 归一化熵
            "trigger_final": trigger_final_val,  # 最终决策指标（三层过滤模型）
            # 旧版trigger标记（保留用于兼容性分析）
            "trigger": trig,
            "trigger_high_entropy": trig_H,
            "trigger_low_confidence": trig_P,
            "trigger_low_confidence1": trig_P1,
            "trigger_ood": trig_ood,
            "trigger_rule_tail_qtype": bool(rule_flag[b_idx]),
            "trigger_uniform_dist": trig_uniform_dist,
            "trigger_small_gap": trig_small_gap,
            "trigger_entropy_extra": trig_entropy_extra,
            "trigger_combined": trig_combined,  # 旧版组合trigger（用于对比）
            "prob_std": prob_std,
            "prob_gap_old": prob_gap_old,  # 旧版计算的prob_gap（用于对比）
            "image_paths": input_['image_paths'],
            "question_id": input_['question_id'],
            "question_type_raw": input_['question_type'],
            "option_A": input_['option_A'],
            "option_B": input_['option_B'],
            "option_C": input_['option_C'],
            "option_D": input_['option_D'],
            "option_E": input_.get('option_E'),
            "option_F": input_.get('option_F'),
            "option_G": input_.get('option_G'),
            "option_H": input_.get('option_H'),
        }
        result_item.update(prob_dict)
        results.append(result_item)

        # 随机打印调试信息
        if random.random() < 0.5:
            print(f'问题类型: {qt}, is_long={is_long}, is_tail={is_tail}')
            print(f'答案: {answer_letter}, p_chosen={p_chosen}, max_prob={max_p_val:.3f}')
            print(f'概率分布: {prob_dict}')
            print(f'语义一致性: {is_consistent}, prob_gap={prob_gap_val:.3f}, H_norm={H_norm_val:.3f}')
            print(f'Trigger (新版): {trigger_final_val}, Trigger (旧版): {trig}, 熵={ent_val:.3f}\n')

# ===== 保存结果 =====
json.dump(results, open(data_save_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
print('推理完成！')
