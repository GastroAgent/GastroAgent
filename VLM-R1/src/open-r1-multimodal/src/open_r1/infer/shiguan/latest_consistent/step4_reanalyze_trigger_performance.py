"""
================================================================================
步骤4: 重新分析Trigger性能
================================================================================
功能: 基于最终数据分析Trigger策略的性能
输入: new_eval_tsy_llm_final.json (步骤3的输出)
输出: new_eval_tsy_llm_trigger_report.json (性能分析报告)

说明:
- 分析trigger_final策略的性能指标
- 分析trigger_final_v2策略的性能指标
- 对比旧版trigger策略
- 按问题类型分组分析
- 生成详细的性能报告
"""

import json
import os
from collections import defaultdict
from typing import Dict, List

# ===== 配置参数 =====
data_name = '食管'
input_data_path = f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/食管_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New-latest-consis/new_eval_tsy_llm_final.json'
output_report_path = f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/食管_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New-latest-consis/new_eval_tsy_llm_trigger_report.json'

os.makedirs(os.path.dirname(output_report_path), exist_ok=True)

# ===== 分析函数 =====
def analyze_trigger_performance(data: List[dict], trigger_key: str = 'trigger_final') -> dict:
    """
    分析trigger策略的性能指标
    """
    # 统计变量
    total = len(data)
    triggered = sum(1 for d in data if d.get(trigger_key, False))
    not_triggered = total - triggered

    # 触发样本的正确性
    triggered_correct = sum(1 for d in data if d.get(trigger_key, False) and d.get('correct', 0) == 1)
    triggered_wrong = triggered - triggered_correct

    # 未触发样本的正确性
    not_triggered_correct = sum(1 for d in data if not d.get(trigger_key, False) and d.get('correct', 0) == 1)
    not_triggered_wrong = not_triggered - not_triggered_correct

    # 计算指标
    trigger_rate = triggered / total if total > 0 else 0.0
    precision = triggered_wrong / triggered if triggered > 0 else 0.0  # 触发中实际有问题的比例
    recall = triggered_wrong / (triggered_wrong + not_triggered_wrong) if (triggered_wrong + not_triggered_wrong) > 0 else 0.0  # 所有错误中被捕获的比例

    # 未触发样本的准确率（安全性指标）
    safety_accuracy = not_triggered_correct / not_triggered if not_triggered > 0 else 0.0

    # 触发样本的准确率（用于分析）
    triggered_accuracy = triggered_correct / triggered if triggered > 0 else 0.0

    # 整体准确率
    overall_accuracy = (triggered_correct + not_triggered_correct) / total if total > 0 else 0.0

    return {
        'total_samples': total,
        'triggered_count': triggered,
        'not_triggered_count': not_triggered,
        'trigger_rate': trigger_rate,

        # 触发样本分析
        'triggered_correct': triggered_correct,
        'triggered_wrong': triggered_wrong,
        'triggered_accuracy': triggered_accuracy,

        # 未触发样本分析（关键安全指标）
        'not_triggered_correct': not_triggered_correct,
        'not_triggered_wrong': not_triggered_wrong,
        'safety_accuracy': safety_accuracy,  # 未触发样本的准确率

        # 错误捕获能力
        'precision': precision,  # 触发样本中的错误率
        'recall': recall,  # 所有错误中被捕获的比例

        # 整体性能
        'overall_accuracy': overall_accuracy,
    }


def has_valid_trigger_key(data: List[dict], trigger_key: str) -> bool:
    """
    判断是否存在有效的trigger字段
    """
    return any((trigger_key in d) and (d.get(trigger_key) is not None) for d in data)


def analyze_by_question_type(data: List[dict], trigger_key: str = 'trigger_final') -> dict:
    """
    按问题类型分组分析
    """
    grouped = defaultdict(list)
    for item in data:
        qt = item.get('question_type', 'Unknown')
        grouped[qt].append(item)

    results = {}
    for qt, items in grouped.items():
        results[qt] = analyze_trigger_performance(items, trigger_key)

    return results


def analyze_probability_metrics(data: List[dict]) -> dict:
    """
    分析概率相关指标
    """
    import statistics

    # 提取指标
    max_probs = [d['max_prob'] for d in data if 'max_prob' in d]
    prob_gaps = [d['prob_gap'] for d in data if 'prob_gap' in d]
    h_norms = [d['h_norm'] for d in data if 'h_norm' in d]
    entropy_scores = [d['entropy_score'] for d in data if 'entropy_score' in d]

    # 按correct分组
    correct_max_probs = [d['max_prob'] for d in data if d.get('correct') == 1 and 'max_prob' in d]
    wrong_max_probs = [d['max_prob'] for d in data if d.get('correct') == 0 and 'max_prob' in d]

    correct_prob_gaps = [d['prob_gap'] for d in data if d.get('correct') == 1 and 'prob_gap' in d]
    wrong_prob_gaps = [d['prob_gap'] for d in data if d.get('correct') == 0 and 'prob_gap' in d]

    return {
        'overall': {
            'max_prob': {
                'mean': statistics.mean(max_probs) if max_probs else 0,
                'median': statistics.median(max_probs) if max_probs else 0,
                'std': statistics.stdev(max_probs) if len(max_probs) > 1 else 0,
            },
            'prob_gap': {
                'mean': statistics.mean(prob_gaps) if prob_gaps else 0,
                'median': statistics.median(prob_gaps) if prob_gaps else 0,
                'std': statistics.stdev(prob_gaps) if len(prob_gaps) > 1 else 0,
            },
            'h_norm': {
                'mean': statistics.mean(h_norms) if h_norms else 0,
                'median': statistics.median(h_norms) if h_norms else 0,
                'std': statistics.stdev(h_norms) if len(h_norms) > 1 else 0,
            },
            'entropy_score': {
                'mean': statistics.mean(entropy_scores) if entropy_scores else 0,
                'median': statistics.median(entropy_scores) if entropy_scores else 0,
                'std': statistics.stdev(entropy_scores) if len(entropy_scores) > 1 else 0,
            },
        },
        'by_correctness': {
            'correct_samples': {
                'max_prob_mean': statistics.mean(correct_max_probs) if correct_max_probs else 0,
                'prob_gap_mean': statistics.mean(correct_prob_gaps) if correct_prob_gaps else 0,
            },
            'wrong_samples': {
                'max_prob_mean': statistics.mean(wrong_max_probs) if wrong_max_probs else 0,
                'prob_gap_mean': statistics.mean(wrong_prob_gaps) if wrong_prob_gaps else 0,
            },
        }
    }


def compare_trigger_strategies(data: List[dict]) -> dict:
    """
    对比不同trigger策略
    """
    strategies = {
        'trigger_final': '新版三层过滤策略',
        'trigger_final_v2': '新版三层过滤策略 v2',
        'trigger_old': '旧版trigger',
        'trigger_high_entropy': '仅高熵',
        'trigger_low_confidence': '仅低置信度',
    }

    comparison = {}
    for key, name in strategies.items():
        if any(key in d for d in data):
            comparison[name] = analyze_trigger_performance(data, key)

    return comparison


# ===== 主处理流程 =====
print(f"正在读取数据: {input_data_path}")
with open(input_data_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

print(f"数据集大小: {len(dataset)}")
print("开始分析Trigger性能...\n")

# ===== 执行分析 =====
# 1. 整体性能分析
print("1. 分析整体性能...")
overall_performance = analyze_trigger_performance(dataset, 'trigger_final')
overall_performance_v2 = analyze_trigger_performance(dataset, 'trigger_final_v2') if has_valid_trigger_key(dataset, 'trigger_final_v2') else {}

# 2. 按问题类型分析
print("2. 按问题类型分析...")
by_question_type = analyze_by_question_type(dataset, 'trigger_final')
by_question_type_v2 = analyze_by_question_type(dataset, 'trigger_final_v2') if has_valid_trigger_key(dataset, 'trigger_final_v2') else {}

# 3. 概率指标分析
print("3. 分析概率指标...")
probability_metrics = analyze_probability_metrics(dataset)

# 4. 策略对比
print("4. 对比不同策略...")
strategy_comparison = compare_trigger_strategies(dataset)

# 5. 分析语义一致性影响
print("5. 分析语义一致性...")
consistent_samples = [d for d in dataset if d.get('is_consistent', False)]
inconsistent_samples = [d for d in dataset if not d.get('is_consistent', False)]

consistency_analysis = {
    'consistent': {
        'count': len(consistent_samples),
        'accuracy': sum(1 for d in consistent_samples if d.get('correct') == 1) / len(consistent_samples) if consistent_samples else 0,
        'avg_max_prob': sum(d.get('max_prob', 0) for d in consistent_samples) / len(consistent_samples) if consistent_samples else 0,
    },
    'inconsistent': {
        'count': len(inconsistent_samples),
        'accuracy': sum(1 for d in inconsistent_samples if d.get('correct') == 1) / len(inconsistent_samples) if inconsistent_samples else 0,
        'avg_max_prob': sum(d.get('max_prob', 0) for d in inconsistent_samples) / len(inconsistent_samples) if inconsistent_samples else 0,
    }
}

# 6. 高风险样本分析
print("6. 分析高风险样本...")
high_risk_samples = [d for d in dataset if d.get('is_tail_qtype', False)]
high_risk_performance = analyze_trigger_performance(high_risk_samples, 'trigger_final') if high_risk_samples else {}
high_risk_performance_v2 = analyze_trigger_performance(high_risk_samples, 'trigger_final_v2') if high_risk_samples and has_valid_trigger_key(high_risk_samples, 'trigger_final_v2') else {}

# ===== 构建报告 =====
report = {
    'dataset_info': {
        'data_name': data_name,
        'total_samples': len(dataset),
        'question_types': list(by_question_type.keys()),
    },

    'overall_performance': overall_performance,
    'overall_performance_v2': overall_performance_v2,

    'by_question_type': by_question_type,
    'by_question_type_v2': by_question_type_v2,

    'probability_metrics': probability_metrics,

    'strategy_comparison': strategy_comparison,

    'consistency_analysis': consistency_analysis,

    'high_risk_analysis': high_risk_performance,
    'high_risk_analysis_v2': high_risk_performance_v2,

    'trigger_thresholds': {
        'THRES_MAX_P': 0.65,
        'THRES_GAP': 0.20,
        'THRES_ENTROPY': 0.40,
    },

    'recommendations': []
}

# ===== 生成建议 =====
recommendations = []

# 安全性检查
if overall_performance['safety_accuracy'] < 0.95:
    recommendations.append({
        'level': 'WARNING',
        'message': f"未触发样本的准确率为 {overall_performance['safety_accuracy']*100:.2f}%，低于95%安全线，建议调整阈值。"
    })
else:
    recommendations.append({
        'level': 'INFO',
        'message': f"未触发样本的准确率为 {overall_performance['safety_accuracy']*100:.2f}%，满足安全要求。"
    })

# 触发率检查
if overall_performance['trigger_rate'] > 0.5:
    recommendations.append({
        'level': 'INFO',
        'message': f"触发率为 {overall_performance['trigger_rate']*100:.2f}%，较高，可能需要人工审核资源较多。"
    })

# 错误捕获能力
if overall_performance['recall'] < 0.7:
    recommendations.append({
        'level': 'WARNING',
        'message': f"错误召回率为 {overall_performance['recall']*100:.2f}%，建议放宽触发条件以捕获更多错误。"
    })

# 语义一致性
inconsistency_rate = len(inconsistent_samples) / len(dataset) if dataset else 0
if inconsistency_rate > 0.1:
    recommendations.append({
        'level': 'WARNING',
        'message': f"语义不一致率为 {inconsistency_rate*100:.2f}%，模型可能存在内部矛盾。"
    })

report['recommendations'] = recommendations

# ===== 打印报告摘要 =====
print("\n" + "=" * 80)
print("Trigger性能分析报告")
print("=" * 80)
print(f"\n【整体性能】")
print(f"  总样本数: {overall_performance['total_samples']}")
print(f"  触发率: {overall_performance['trigger_rate']*100:.2f}%")
print(f"  整体准确率: {overall_performance['overall_accuracy']*100:.2f}%")
if overall_performance_v2:
    print(f"  [v2] 触发率: {overall_performance_v2['trigger_rate']*100:.2f}%")
    print(f"  [v2] 整体准确率: {overall_performance_v2['overall_accuracy']*100:.2f}%")
print(f"\n【安全性指标】")
print(f"  未触发样本数: {overall_performance['not_triggered_count']}")
print(f"  未触发样本准确率: {overall_performance['safety_accuracy']*100:.2f}%")
print(f"  未触发中的错误数: {overall_performance['not_triggered_wrong']}")
if overall_performance_v2:
    print(f"  [v2] 未触发样本数: {overall_performance_v2['not_triggered_count']}")
    print(f"  [v2] 未触发样本准确率: {overall_performance_v2['safety_accuracy']*100:.2f}%")
    print(f"  [v2] 未触发中的错误数: {overall_performance_v2['not_triggered_wrong']}")
print(f"\n【触发样本分析】")
print(f"  触发样本数: {overall_performance['triggered_count']}")
print(f"  触发样本准确率: {overall_performance['triggered_accuracy']*100:.2f}%")
print(f"  触发中的错误数: {overall_performance['triggered_wrong']}")
if overall_performance_v2:
    print(f"  [v2] 触发样本数: {overall_performance_v2['triggered_count']}")
    print(f"  [v2] 触发样本准确率: {overall_performance_v2['triggered_accuracy']*100:.2f}%")
    print(f"  [v2] 触发中的错误数: {overall_performance_v2['triggered_wrong']}")
print(f"\n【错误捕获能力】")
print(f"  Precision (触发中的错误率): {overall_performance['precision']*100:.2f}%")
print(f"  Recall (错误召回率): {overall_performance['recall']*100:.2f}%")
if overall_performance_v2:
    print(f"  [v2] Precision (触发中的错误率): {overall_performance_v2['precision']*100:.2f}%")
    print(f"  [v2] Recall (错误召回率): {overall_performance_v2['recall']*100:.2f}%")
print(f"\n【语义一致性】")
print(f"  一致样本: {consistency_analysis['consistent']['count']} (准确率: {consistency_analysis['consistent']['accuracy']*100:.2f}%)")
print(f"  不一致样本: {consistency_analysis['inconsistent']['count']} (准确率: {consistency_analysis['inconsistent']['accuracy']*100:.2f}%)")

print(f"\n【建议】")
for rec in recommendations:
    print(f"  [{rec['level']}] {rec['message']}")

print("\n" + "=" * 80)

# ===== 保存报告 =====
print(f"\n正在保存报告到: {output_report_path}")
with open(output_report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=4, ensure_ascii=False)

print('=' * 80)
print('步骤4完成！')
print(f'输出文件: {output_report_path}')
print('=' * 80)
