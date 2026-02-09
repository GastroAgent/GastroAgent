"""
================================================================================
完整版：医疗VQA答案提取与重新评估系统（保留Trigger机制）
功能：
1. 保留所有原始字段（包括trigger相关字段）
2. 使用大模型从generated_text中提取答案到extracted_answer
3. 根据extracted_answer与gt_letter比较重新计算correct
4. 重新分析trigger机制在新correct下的表现
================================================================================
"""

import json
import re
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
from modelscope import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# =========================
# 配置参数
# =========================
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,7'

# 模型配置
MODEL_NAME = "/mnt/inaisfs/data/home/tansy_criait/weights/Qwen2.5-32B-Instruct"

# 数据路径
DATA_PATH = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/doctor_食管/new_eval_tsy_llm.json'

# 有效选项
VALID_OPTIONS = set("ABCDEFGH")

# 正则表达式
answer_tag_re = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
answer_letter_re = re.compile(r"\b([A-H])\b", re.IGNORECASE)

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


# =========================
# 答案标准化函数
# =========================
def normalize_answer(value: Optional[str]) -> Optional[str]:
    """标准化答案格式"""
    if value is None:
        return None
    
    value = str(value).strip()
    if not value:
        return None
    
    # 处理option_X格式
    option_match = re.match(r"option[_\s]*([A-H])", value, re.IGNORECASE)
    if option_match:
        return option_match.group(1).upper()
    
    # 处理纯字母格式
    if len(value) == 1 and value.upper() in VALID_OPTIONS:
        return value.upper()
    
    # 尝试从字符串中提取字母
    letter_match = answer_letter_re.search(value)
    if letter_match:
        letter = letter_match.group(1).upper()
        if letter in VALID_OPTIONS:
            return letter
    
    return None


def extract_answer_from_text(text: str) -> Optional[str]:
    """从文本中提取<answer></answer>标签之间的内容"""
    if not text:
        return None
    
    # 首先尝试匹配<answer></answer>标签
    match = answer_tag_re.search(text)
    if match:
        return match.group(1).strip()
    
    # 如果没有找到标签，尝试直接匹配字母
    letter_match = answer_letter_re.search(text)
    if letter_match:
        return letter_match.group(1).strip()
    
    return None


# =========================
# 步骤1: 使用大模型提取答案（保留所有原始字段）
# =========================
def extract_answers_with_llm(
    input_path: str,
    output_path: str,
    model_name: str = MODEL_NAME,
    skip_existing: bool = True
):
    """
    使用大模型从generated_text中提取答案，保留所有原始字段
    """
    print(f"\n{'='*70}")
    print(f"步骤1: 使用大模型提取答案（保留所有原始字段）")
    print(f"{'='*70}\n")
    
    # 加载模型
    print(f"加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    print("模型加载完成！\n")
    
    # 加载数据
    print(f"加载数据: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"数据加载完成，共 {len(dataset)} 条记录\n")
    
    # 统计信息
    stats = {
        'total': len(dataset),
        'skipped': 0,
        'processed': 0,
        'success': 0,
        'failed': 0
    }
    
    new_dataset = []
    
    # 处理每条数据
    for idx, data in enumerate(tqdm(dataset, desc="提取答案")):
        # 如果已经有extracted_answer且skip_existing=True，则跳过
        if skip_existing and "extracted_answer" in data and data["extracted_answer"]:
            new_dataset.append(data)
            stats['skipped'] += 1
            continue
        
        stats['processed'] += 1
        
        # 获取generated_text和prompt_text
        text = data.get('generated_text', '')
        prompt_text = data.get('prompt_text', '')
        
        # 从prompt_text中提取question部分
        if '\nQuestion' in prompt_text:
            question = prompt_text.split('\nQuestion')[-1].replace('<|im_end|>\n<|im_start|>assistant\n', '')
        else:
            question = prompt_text.replace('<|im_end|>\n<|im_start|>assistant\n', '')
        
        # 构建prompt
        prompt = (
            f"Question: {question}\n\n"
            f"Model Response: {text}\n\n"
            "Your Task: Extract the final answer option from the model's response above. "
            "The answer should be a single letter (A-H). "
            "If the response already contains an <answer> tag, extract the content within it. "
            "Otherwise, identify the chosen option from the response. "
            "Format your output as: <answer>X</answer> where X is the option letter.\n\n"
            "Please provide your answer:"
        )
        
        messages = [{"role": "user", "content": prompt}]
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
        
        # 生成答案
        try:
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=False
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            
            # 解析thinking content
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
            
            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            
            # 保存extracted_answer（新增字段）
            data['extracted_answer'] = content
            data['extraction_thinking'] = thinking_content
            
            stats['success'] += 1
            
            # 随机打印
            if idx % 50 == 0:
                print(f"\n样本 {idx}:")
                print(f"  Generated text: {text[:100]}...")
                print(f"  Extracted answer: {content}")
        
        except Exception as e:
            print(f"\n警告：处理样本 {idx} 时出错: {e}")
            data['extracted_answer'] = None
            data['extraction_error'] = str(e)
            stats['failed'] += 1
        
        # 保留所有原始字段（包括trigger相关字段）
        new_dataset.append(data)
    
    # 保存结果
    print(f"\n保存提取结果到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_dataset, f, ensure_ascii=False, indent=4)
    
    # 打印统计
    print(f"\n{'='*70}")
    print(f"提取统计:")
    print(f"  总样本数: {stats['total']}")
    print(f"  跳过（已存在）: {stats['skipped']}")
    print(f"  处理样本数: {stats['processed']}")
    print(f"  提取成功: {stats['success']}")
    print(f"  提取失败: {stats['failed']}")
    print(f"{'='*70}\n")
    
    return output_path


# =========================
# 步骤2: 重新评估correct（保留trigger字段）
# =========================
def reevaluate_correct_with_trigger(
    input_path: str,
    output_path: str
):
    """
    根据extracted_answer重新计算correct，保留所有trigger字段
    """
    print(f"\n{'='*70}")
    print(f"步骤2: 重新评估correct（保留trigger机制）")
    print(f"{'='*70}\n")
    
    # 加载数据
    print(f"加载数据: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计信息
    stats = {
        "total": 0,
        "extracted_found": 0,
        "gt_found": 0,
        "both_found": 0,
        "matched": 0,
        "original_correct": 0,
        "new_correct": 0,
        "changed_to_correct": 0,
        "changed_to_incorrect": 0
    }
    
    correct_list = []
    
    # 处理每条数据
    for item in tqdm(data, desc="重新评估correct"):
        stats["total"] += 1
        
        # 保存原始correct值
        original_correct = item.get("correct", 0)
        stats["original_correct"] += original_correct
        
        # 获取gt_letter和extracted_answer
        #gt_letter_raw = item.get("gt_letter")
        gt_letter_raw = get_gt_option(item)
        extracted_answer_raw = item.get("extracted_answer", "")
        
        # 提取答案
        extracted_content = extract_answer_from_text(extracted_answer_raw)
        if extracted_content:
            stats["extracted_found"] += 1
        
        # 标准化
        pred_letter = normalize_answer(extracted_content)
        gt_letter_normalized = normalize_answer(gt_letter_raw)
        
        if gt_letter_normalized:
            stats["gt_found"] += 1
        
        if pred_letter and gt_letter_normalized:
            stats["both_found"] += 1
        
        # 计算新的correct值
        if pred_letter and gt_letter_normalized:
            new_correct = 1 if pred_letter == gt_letter_normalized else 0
            if new_correct:
                stats["matched"] += 1
        else:
            new_correct = 0
        
        # 统计变化
        if new_correct == 1:
            stats["new_correct"] += 1
        if original_correct == 0 and new_correct == 1:
            stats["changed_to_correct"] += 1
        if original_correct == 1 and new_correct == 0:
            stats["changed_to_incorrect"] += 1
        
        # 更新字段（新增字段）
        item["pred_letter"] = pred_letter
        item["correct_new"] = new_correct  # 新correct值
        item["correct_original"] = original_correct  # 保留原始值
        item["correct"] = new_correct  # 重新赋值correct为新值
        item["extracted_content"] = extracted_content
        item["gt_letter_normalized"] = gt_letter_normalized
        
        # 保留所有trigger相关字段（不修改）
        # trigger_final, confidence_score, trigger_score, 
        # max_prob, prob_gap, h_norm, is_consistent等字段都保持原样
        
        correct_list.append(new_correct)
    
    # 计算准确率
    total = len(correct_list)
    new_acc = sum(correct_list) / total if total > 0 else 0.0
    original_acc = stats["original_correct"] / total if total > 0 else 0.0
    
    # 打印统计
    print(f"\n{'='*70}")
    print(f"重新评估统计:")
    print(f"{'='*70}")
    print(f"总样本数: {total}")
    print(f"成功提取extracted_answer: {stats['extracted_found']} ({stats['extracted_found']/total*100:.2f}%)")
    print(f"成功标准化gt_letter: {stats['gt_found']} ({stats['gt_found']/total*100:.2f}%)")
    print(f"两者都成功: {stats['both_found']} ({stats['both_found']/total*100:.2f}%)")
    print(f"匹配成功: {stats['matched']} ({stats['matched']/total*100:.2f}%)")
    print(f"\n准确率对比:")
    print(f"  原始Accuracy: {original_acc:.2%} ({stats['original_correct']}/{total})")
    print(f"  新Accuracy: {new_acc:.2%} ({stats['new_correct']}/{total})")
    print(f"  变化: {(new_acc - original_acc)*100:+.2f}%")
    print(f"\n变化详情:")
    print(f"  从错误变为正确: {stats['changed_to_correct']}")
    print(f"  从正确变为错误: {stats['changed_to_incorrect']}")
    print(f"{'='*70}\n")
    
    # 保存结果
    print(f"保存最终结果到: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return output_path, stats


# =========================
# 步骤3: 重新分析Trigger机制在新correct下的表现
# =========================
def reanalyze_trigger_performance(
    input_path: str,
    output_report_path: str
):
    """
    基于新的correct值，重新分析trigger机制的表现
    """
    print(f"\n{'='*70}")
    print(f"步骤3: 重新分析Trigger机制表现（基于新correct）")
    print(f"{'='*70}\n")
    
    # 加载数据
    with open(input_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 检查是否有trigger字段
    has_trigger_fields = len(results) > 0 and 'trigger_final' in results[0]
    
    if not has_trigger_fields:
        print(f"警告: 数据中不包含trigger_final字段")
        print(f"      无法进行trigger性能分析")
        print(f"      如需trigger分析，请先运行trigger推理脚本\n")
        
        # 返回一个简单的报告
        simple_report = {
            "note": "数据中不包含trigger字段，无法进行trigger分析",
            "total_samples": len(results),
            "overall_accuracy": np.mean([r.get('correct', 0) for r in results])
        }
        
        with open(output_report_path, 'w', encoding='utf-8') as f:
            json.dump(simple_report, f, ensure_ascii=False, indent=2)
        
        return simple_report
    
    # 分组
    triggered = [r for r in results if r.get('trigger_final', False)]
    not_triggered = [r for r in results if not r.get('trigger_final', False)]
    
    n_total = len(results)
    n_triggered = len(triggered)
    n_not_triggered = len(not_triggered)
    
    # 基于新correct计算准确率
    acc_overall_new = np.mean([r.get('correct', 0) for r in results])
    acc_triggered_new = np.mean([r.get('correct', 0) for r in triggered]) if n_triggered > 0 else 0
    acc_not_triggered_new = np.mean([r.get('correct', 0) for r in not_triggered]) if n_not_triggered > 0 else 0
    
    # 基于原始correct计算准确率（对比）- 检查字段是否存在
    has_original_correct = 'correct_original' in results[0] if results else False
    
    if has_original_correct:
        acc_overall_old = np.mean([r.get('correct_original', 0) for r in results])
        acc_triggered_old = np.mean([r.get('correct_original', 0) for r in triggered]) if n_triggered > 0 else 0
        acc_not_triggered_old = np.mean([r.get('correct_original', 0) for r in not_triggered]) if n_not_triggered > 0 else 0
    else:
        # 如果没有correct_original，说明这是第一次运行，使用当前correct作为原始值
        acc_overall_old = acc_overall_new
        acc_triggered_old = acc_triggered_new
        acc_not_triggered_old = acc_not_triggered_new
    
    coverage = n_not_triggered / n_total if n_total > 0 else 0
    
    # 错误分析
    errors_not_triggered_new = [r for r in not_triggered if r.get('correct', 0) == 0]
    errors_triggered_new = [r for r in triggered if r.get('correct', 0) == 0]
    
    if has_original_correct:
        errors_not_triggered_old = [r for r in not_triggered if r.get('correct_original', 0) == 0]
        errors_triggered_old = [r for r in triggered if r.get('correct_original', 0) == 0]
    else:
        errors_not_triggered_old = errors_not_triggered_new
        errors_triggered_old = errors_triggered_new
    
    # 打印详细报告
    print(f"{'='*70}")
    print(f"Trigger机制性能分析（基于新correct）")
    print(f"{'='*70}\n")
    
    print(f"样本分布:")
    print(f"  总样本数: {n_total}")
    print(f"  未触发样本数: {n_not_triggered} ({coverage*100:.1f}%)")
    print(f"  触发样本数: {n_triggered} ({(1-coverage)*100:.1f}%)")
    
    print(f"\n准确率对比（新correct vs 原始correct）:")
    print(f"  整体准确率:")
    print(f"    新: {acc_overall_new*100:.2f}% | 原始: {acc_overall_old*100:.2f}% | 变化: {(acc_overall_new-acc_overall_old)*100:+.2f}%")
    print(f"  未触发样本准确率: ⭐ 【核心指标】")
    print(f"    新: {acc_not_triggered_new*100:.2f}% | 原始: {acc_not_triggered_old*100:.2f}% | 变化: {(acc_not_triggered_new-acc_not_triggered_old)*100:+.2f}%")
    print(f"  触发样本准确率:")
    print(f"    新: {acc_triggered_new*100:.2f}% | 原始: {acc_triggered_old*100:.2f}% | 变化: {(acc_triggered_new-acc_triggered_old)*100:+.2f}%")
    
    print(f"\n错误分析:")
    print(f"  未触发样本中的错误:")
    print(f"    新: {len(errors_not_triggered_new)} / {n_not_triggered} | 原始: {len(errors_not_triggered_old)} / {n_not_triggered}")
    print(f"  触发样本中的错误:")
    print(f"    新: {len(errors_triggered_new)} / {n_triggered} | 原始: {len(errors_triggered_old)} / {n_triggered}")
    
    # 特征统计（只有在字段存在时才计算）
    has_trigger_features = 'max_prob' in results[0] if results else False
    
    if has_trigger_features:
        if n_not_triggered > 0:
            avg_max_prob = np.mean([r.get('max_prob', 0) for r in not_triggered])
            avg_gap = np.mean([r.get('prob_gap', 0) for r in not_triggered])
            avg_entropy = np.mean([r.get('h_norm', 0) for r in not_triggered])
            
            print(f"\n未触发样本的平均特征:")
            print(f"  平均max_prob: {avg_max_prob:.3f}")
            print(f"  平均prob_gap: {avg_gap:.3f}")
            print(f"  平均h_norm: {avg_entropy:.3f}")
        
        if n_triggered > 0:
            avg_max_prob_t = np.mean([r.get('max_prob', 0) for r in triggered])
            avg_gap_t = np.mean([r.get('prob_gap', 0) for r in triggered])
            avg_entropy_t = np.mean([r.get('h_norm', 0) for r in triggered])
            
            print(f"\n触发样本的平均特征:")
            print(f"  平均max_prob: {avg_max_prob_t:.3f}")
            print(f"  平均prob_gap: {avg_gap_t:.3f}")
            print(f"  平均h_norm: {avg_entropy_t:.3f}")
    else:
        print(f"\n提示: 数据中不包含trigger特征字段（max_prob, prob_gap等）")
        print(f"      如需完整的trigger分析，请先运行trigger推理脚本")
    
    print(f"\n{'='*70}\n")
    
    # 生成详细报告
    report = {
        "sample_distribution": {
            "total": n_total,
            "not_triggered": n_not_triggered,
            "triggered": n_triggered,
            "coverage": coverage
        },
        "accuracy_comparison": {
            "overall": {
                "new": acc_overall_new,
                "original": acc_overall_old,
                "change": acc_overall_new - acc_overall_old
            },
            "not_triggered": {
                "new": acc_not_triggered_new,
                "original": acc_not_triggered_old,
                "change": acc_not_triggered_new - acc_not_triggered_old
            },
            "triggered": {
                "new": acc_triggered_new,
                "original": acc_triggered_old,
                "change": acc_triggered_new - acc_triggered_old
            }
        },
        "error_analysis": {
            "not_triggered": {
                "new": len(errors_not_triggered_new),
                "original": len(errors_not_triggered_old)
            },
            "triggered": {
                "new": len(errors_triggered_new),
                "original": len(errors_triggered_old)
            }
        }
    }
    
    # 保存报告
    with open(output_report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"详细报告已保存到: {output_report_path}")
    
    return report


# =========================
# 主函数
# =========================
def main(
    data_path: str = DATA_PATH,
    skip_extraction: bool = False,
    skip_existing: bool = True
):
    """
    主流程：提取答案 + 重新评估 + 重新分析trigger
    """
    print(f"\n{'='*70}")
    print(f"完整版：医疗VQA答案提取与重新评估系统")
    print(f"（保留所有Trigger机制字段）")
    print(f"{'='*70}\n")
    
    extracted_path = data_path.replace('.json', '_extracted.json')
    final_path = data_path.replace('.json', '_final.json')
    report_path = data_path.replace('.json', '_trigger_report.json')
    
    # 步骤1: 提取答案
    if not skip_extraction:
        extracted_path = extract_answers_with_llm(
            input_path=data_path,
            output_path=extracted_path,
            model_name=MODEL_NAME,
            skip_existing=skip_existing
        )
    else:
        print(f"\n跳过答案提取步骤，使用现有文件: {extracted_path}")
        if not Path(extracted_path).exists():
            raise FileNotFoundError(f"未找到提取文件: {extracted_path}")
    
    # 步骤2: 重新评估correct
    final_path, stats = reevaluate_correct_with_trigger(
        input_path=extracted_path,
        output_path=final_path
    )
    
    # 步骤3: 重新分析trigger性能
    trigger_report = reanalyze_trigger_performance(
        input_path=final_path,
        output_report_path=report_path
    )
    
    print(f"\n{'='*70}")
    print(f"全部完成！")
    print(f"  提取结果: {extracted_path}")
    print(f"  最终结果: {final_path}")
    print(f"  Trigger报告: {report_path}")
    print(f"{'='*70}\n")
    
    return final_path, trigger_report


# =========================
# 命令行入口
# =========================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='完整版：医疗VQA答案提取与重新评估系统（保留Trigger机制）')
    parser.add_argument('--data_path', type=str, default=DATA_PATH,
                       help='输入数据路径')
    parser.add_argument('--skip_extraction', action='store_true',
                       help='跳过答案提取步骤')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                       help='跳过已有extracted_answer的样本')
    parser.add_argument('--model_name', type=str, default=MODEL_NAME,
                       help='提取模型名称或路径')
    
    args = parser.parse_args()
    
    if args.model_name != MODEL_NAME:
        MODEL_NAME = args.model_name
    
    final_path, trigger_report = main(
        data_path=args.data_path,
        skip_extraction=args.skip_extraction,
        skip_existing=args.skip_existing
    )
    
    print("\n处理完成！")