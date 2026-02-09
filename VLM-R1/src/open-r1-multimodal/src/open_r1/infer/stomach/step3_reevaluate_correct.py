"""
================================================================================
步骤3: 重新评估correct
================================================================================
功能: 根据提取的答案重新计算correct字段
输入: new_eval_tsy_llm_extracted.json (步骤2的输出)
输出: new_eval_tsy_llm_final.json (更新 correct 字段)

说明:
- 将extracted_answer与gt_answer进行对比
- 计算准确率
- 添加更多分析字段
"""

import json
import os
import re

# ===== 配置参数 =====
data_name = '胃'
input_data_path = f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New1/new_eval_tsy_llm_extracted.json'
output_data_path = f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New1/new_eval_tsy_llm_final.json'

os.makedirs(os.path.dirname(output_data_path), exist_ok=True)

# ===== 工具函数 =====
def get_gt_letter(item: dict) -> str:
    """
    从item中提取ground truth的选项字母
    """
    gt_text = item.get("gt_answer", None)
    if gt_text is None:
        return None

    # 遍历所有选项，找到匹配的
    for letter in "ABCDEFGH":
        key = f"option_{letter}"
        if key in item and item[key] == gt_text:
            return letter

    return None


def normalize_letter(x: str) -> str:
    """
    标准化选项字母
    """
    if x is None:
        return None

    s = str(x).strip()

    # 处理 "option_B" 格式
    if s.lower().startswith("option_"):
        s = s.split("_", 1)[1]

    # 提取字母
    for ch in s:
        if ch.upper() in "ABCDEFGH":
            return ch.upper()

    return None


# ===== 主处理流程 =====
print(f"正在读取数据: {input_data_path}")
with open(input_data_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

print(f"数据集大小: {len(dataset)}")
print("开始重新评估correct...\n")

results = []
total_correct = 0
total_with_answer = 0
total_samples = len(dataset)

for i, item in enumerate(dataset):
    # 获取ground truth字母
    gt_letter = get_gt_letter(item)

    # 获取提取的答案
    extracted_answer = item.get('extracted_answer', None)
    norm_extracted = normalize_letter(extracted_answer)
    norm_gt = normalize_letter(gt_letter)

    # 计算correct
    if norm_extracted is not None and norm_gt is not None:
        correct = 1 if norm_extracted == norm_gt else 0
        total_with_answer += 1
        total_correct += correct
    else:
        correct = 0  # 无法提取答案视为错误

    # 添加/更新字段
    item['gt_letter'] = gt_letter
    item['extracted_answer_normalized'] = norm_extracted
    item['correct'] = correct

    # 计算p_chosen（提取答案对应的概率）
    if norm_extracted is not None:
        prob_key = f"p_{norm_extracted}"
        item['p_chosen'] = item.get(prob_key, None)
    else:
        item['p_chosen'] = None

    results.append(item)

    # 随机打印调试信息
    if i < 5:
        print(f"\n--- 样本 {i+1} ---")
        print(f"GT答案: {gt_letter} ({item.get('gt_answer', 'N/A')[:50]}...)")
        print(f"提取答案: {norm_extracted}")
        print(f"正确性: {correct}")
        print(f"p_chosen: {item.get('p_chosen')}")
        print("----------------\n")

# ===== 计算统计信息 =====
accuracy = total_correct / total_with_answer if total_with_answer > 0 else 0.0
coverage = total_with_answer / total_samples

print("\n" + "=" * 80)
print("评估统计:")
print(f"总样本数: {total_samples}")
print(f"成功提取答案: {total_with_answer} ({coverage*100:.1f}%)")
print(f"正确数量: {total_correct}")
print(f"准确率: {accuracy*100:.2f}% ({total_correct}/{total_with_answer})")
print("=" * 80 + "\n")

# ===== 保存结果 =====
print(f"正在保存结果到: {output_data_path}")
with open(output_data_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print('=' * 80)
print('步骤3完成！')
print(f'输出文件: {output_data_path}')
print('=' * 80)
