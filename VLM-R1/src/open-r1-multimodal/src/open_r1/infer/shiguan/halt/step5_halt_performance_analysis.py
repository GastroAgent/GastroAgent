"""
================================================================================
步骤5: HALT幻觉检测性能分析
================================================================================
功能: 分析HALT方法的幻觉检测性能，对比传统trigger策略
输入: new_eval_tsy_llm_final.json (包含correct字段和HALT检测结果)
输出: halt_performance_report.json (HALT性能分析报告)
"""

import json
import os
from collections import defaultdict
import numpy as np

# ===== 配置参数 =====
input_data_path = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/食管_doctor_exam/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-jmf-sft-tsy-cotRL-2000/halt_85v1/new_eval_tsy_llm_final.json'
output_report_path = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/食管_doctor_exam/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-jmf-sft-tsy-cotRL-2000/halt_85v1/halt_performance_report.json'

os.makedirs(os.path.dirname(output_report_path), exist_ok=True)

# ===== 加载数据 =====
print("=" * 80)
print("HALT幻觉检测性能分析")
print("=" * 80)
print(f"\n正在加载数据: {input_data_path}")

with open(input_data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"数据集大小: {len(data)}")

# ===== 性能指标计算 =====
def calculate_metrics(predictions, ground_truth):
    """
    计算分类指标
    predictions: 预测为高风险的标记 (True/False)
    ground_truth: 实际是否错误 (True=错误/幻觉, False=正确)
    """
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
    tn = sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)
    fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)

    total = len(predictions)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total': total
    }

# ===== 提取数据 =====
print("\n正在分析HALT性能...")

# 过滤掉没有HALT数据的样本
halt_data = [d for d in data if d.get('halt_risk_score') is not None]
print(f"包含HALT检测的样本数: {len(halt_data)}")

if len(halt_data) == 0:
    print("\n错误: 数据中没有HALT检测结果！")
    print("请确保step1_model_inference.py中启用了HALT检测（HALT_ENABLED=True）")
    exit(1)

# 提取标签和预测
ground_truth = [not d.get('correct', False) for d in halt_data]  # True=错误/幻觉
halt_predictions = [d.get('halt_high_risk', False) for d in halt_data]
trigger_old_predictions = [d.get('trigger_old', False) for d in halt_data]
trigger_final_predictions = [d.get('trigger_final', False) for d in halt_data]

# 提取风险分数
halt_risk_scores = [d.get('halt_risk_score', 0) for d in halt_data]

# ===== 计算性能指标 =====
print("\n计算性能指标...")

halt_metrics = calculate_metrics(halt_predictions, ground_truth)
trigger_old_metrics = calculate_metrics(trigger_old_predictions, ground_truth)
trigger_final_metrics = calculate_metrics(trigger_final_predictions, ground_truth)

# ===== 按问题类型分析 =====
print("\n按问题类型分析...")

qtypes = set(d['question_type'] for d in halt_data)
qtype_analysis = {}

for qtype in qtypes:
    qtype_samples = [d for d in halt_data if d['question_type'] == qtype]
    qtype_gt = [not d.get('correct', False) for d in qtype_samples]
    qtype_halt_pred = [d.get('halt_high_risk', False) for d in qtype_samples]

    qtype_analysis[qtype] = {
        'count': len(qtype_samples),
        'error_rate': sum(qtype_gt) / len(qtype_gt) if len(qtype_gt) > 0 else 0,
        'halt_metrics': calculate_metrics(qtype_halt_pred, qtype_gt)
    }

# ===== 风险分数分布分析 =====
print("\n分析风险分数分布...")

correct_samples = [d for d in halt_data if d.get('correct', False)]
incorrect_samples = [d for d in halt_data if not d.get('correct', False)]

correct_risk_scores = [d.get('halt_risk_score', 0) for d in correct_samples]
incorrect_risk_scores = [d.get('halt_risk_score', 0) for d in incorrect_samples]

risk_distribution = {
    'correct_samples': {
        'count': len(correct_risk_scores),
        'mean': float(np.mean(correct_risk_scores)) if correct_risk_scores else 0,
        'std': float(np.std(correct_risk_scores)) if correct_risk_scores else 0,
        'min': float(np.min(correct_risk_scores)) if correct_risk_scores else 0,
        'max': float(np.max(correct_risk_scores)) if correct_risk_scores else 0,
        'median': float(np.median(correct_risk_scores)) if correct_risk_scores else 0,
    },
    'incorrect_samples': {
        'count': len(incorrect_risk_scores),
        'mean': float(np.mean(incorrect_risk_scores)) if incorrect_risk_scores else 0,
        'std': float(np.std(incorrect_risk_scores)) if incorrect_risk_scores else 0,
        'min': float(np.min(incorrect_risk_scores)) if incorrect_risk_scores else 0,
        'max': float(np.max(incorrect_risk_scores)) if incorrect_risk_scores else 0,
        'median': float(np.median(incorrect_risk_scores)) if incorrect_risk_scores else 0,
    }
}

# ===== 阈值敏感性分析 =====
print("\n进行阈值敏感性分析...")

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
threshold_analysis = {}

for threshold in thresholds:
    threshold_predictions = [score > threshold for score in halt_risk_scores]
    threshold_metrics = calculate_metrics(threshold_predictions, ground_truth)
    threshold_analysis[f"threshold_{threshold}"] = threshold_metrics

# ===== 构建报告 =====
report = {
    'summary': {
        'total_samples': len(halt_data),
        'correct_samples': sum(1 for d in halt_data if d.get('correct', False)),
        'incorrect_samples': sum(1 for d in halt_data if not d.get('correct', False)),
        'error_rate': sum(ground_truth) / len(ground_truth) if ground_truth else 0,
    },

    'halt_performance': {
        'description': 'HALT幻觉检测性能（基于中间层隐藏状态）',
        'metrics': halt_metrics,
    },

    'baseline_comparison': {
        'trigger_old': {
            'description': '旧版Trigger策略（熵+置信度）',
            'metrics': trigger_old_metrics,
        },
        'trigger_final': {
            'description': '新版Trigger策略（三层过滤）',
            'metrics': trigger_final_metrics,
        },
    },

    'performance_comparison': {
        'halt_vs_trigger_old': {
            'accuracy_improvement': halt_metrics['accuracy'] - trigger_old_metrics['accuracy'],
            'precision_improvement': halt_metrics['precision'] - trigger_old_metrics['precision'],
            'recall_improvement': halt_metrics['recall'] - trigger_old_metrics['recall'],
            'f1_improvement': halt_metrics['f1'] - trigger_old_metrics['f1'],
        },
        'halt_vs_trigger_final': {
            'accuracy_improvement': halt_metrics['accuracy'] - trigger_final_metrics['accuracy'],
            'precision_improvement': halt_metrics['precision'] - trigger_final_metrics['precision'],
            'recall_improvement': halt_metrics['recall'] - trigger_final_metrics['recall'],
            'f1_improvement': halt_metrics['f1'] - trigger_final_metrics['f1'],
        },
    },

    'question_type_analysis': qtype_analysis,

    'risk_score_distribution': risk_distribution,

    'threshold_sensitivity': threshold_analysis,

    'key_findings': []
}

# ===== 生成关键发现 =====
findings = []

# 1. 整体性能
if halt_metrics['f1'] > trigger_old_metrics['f1']:
    findings.append(f"HALT方法的F1分数({halt_metrics['f1']:.3f})优于旧版Trigger策略({trigger_old_metrics['f1']:.3f})")
if halt_metrics['f1'] > trigger_final_metrics['f1']:
    findings.append(f"HALT方法的F1分数({halt_metrics['f1']:.3f})优于新版Trigger策略({trigger_final_metrics['f1']:.3f})")

# 2. 风险分数区分度
risk_diff = risk_distribution['incorrect_samples']['mean'] - risk_distribution['correct_samples']['mean']
findings.append(f"错误样本的平均风险分数({risk_distribution['incorrect_samples']['mean']:.3f})比正确样本({risk_distribution['correct_samples']['mean']:.3f})高{risk_diff:.3f}")

# 3. 最佳阈值
best_threshold = max(threshold_analysis.items(), key=lambda x: x[1]['f1'])
findings.append(f"最佳风险阈值为{best_threshold[0].replace('threshold_', '')}，F1分数为{best_threshold[1]['f1']:.3f}")

# 4. 问题类型差异
if qtype_analysis:
    best_qtype = max(qtype_analysis.items(), key=lambda x: x[1]['halt_metrics']['f1'])
    worst_qtype = min(qtype_analysis.items(), key=lambda x: x[1]['halt_metrics']['f1'])
    findings.append(f"HALT在'{best_qtype[0]}'类型问题上表现最好(F1={best_qtype[1]['halt_metrics']['f1']:.3f})")
    findings.append(f"HALT在'{worst_qtype[0]}'类型问题上表现较弱(F1={worst_qtype[1]['halt_metrics']['f1']:.3f})")

report['key_findings'] = findings

# ===== 保存报告 =====
print(f"\n正在保存报告到: {output_report_path}")
with open(output_report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=4, ensure_ascii=False)

# ===== 打印摘要 =====
print("\n" + "=" * 80)
print("HALT性能分析完成！")
print("=" * 80)

print("\n📊 整体性能:")
print(f"  总样本数: {report['summary']['total_samples']}")
print(f"  错误率: {report['summary']['error_rate']:.1%}")

print("\n🎯 HALT检测性能:")
print(f"  准确率: {halt_metrics['accuracy']:.3f}")
print(f"  精确率: {halt_metrics['precision']:.3f}")
print(f"  召回率: {halt_metrics['recall']:.3f}")
print(f"  F1分数: {halt_metrics['f1']:.3f}")

print("\n📈 对比基线:")
print(f"  vs 旧版Trigger: F1提升 {report['performance_comparison']['halt_vs_trigger_old']['f1_improvement']:+.3f}")
print(f"  vs 新版Trigger: F1提升 {report['performance_comparison']['halt_vs_trigger_final']['f1_improvement']:+.3f}")

print("\n💡 关键发现:")
for i, finding in enumerate(findings, 1):
    print(f"  {i}. {finding}")

print("\n" + "=" * 80)
print(f"详细报告已保存: {output_report_path}")
print("=" * 80)
