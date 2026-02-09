#!/usr/bin/env python3
"""
================================================================================
Logit Fusion Pipeline - 融合MLLM和Flow模型的预测
================================================================================
功能: 使用动态权重融合两个模型的概率分布，而非简单路由
方法: P_final = α * P_MLLM + (1-α) * P_Flow
      其中α基于MLLM的熵动态调整：熵低(自信)时α接近1，熵高(不自信)时α接近0
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


# ===== 类别映射 =====
# 这些将在加载数据后动态构建
CLASS_NAMES = []
CLASS_TO_IDX = {}
IDX_TO_CLASS = {}


def build_class_mapping(mllm_data: List[Dict], flow_data: List[Dict]):
    """从数据中构建类别映射"""
    global CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS

    class_set = set()

    # 从Flow数据中提取类别
    for item in flow_data:
        if 'gt' in item:
            class_set.add(item['gt'].strip())
        if 'pred' in item:
            class_set.add(item['pred'].strip())

    # 从MLLM数据中提取类别
    for item in mllm_data:
        for opt in ['A', 'B', 'C', 'D']:
            opt_key = f'option_{opt}'
            if opt_key in item:
                class_set.add(item[opt_key].strip())

    # 排序以保证一致性
    CLASS_NAMES = sorted(list(class_set))
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

    return len(CLASS_NAMES)


def normalize_class_name(name: str) -> str:
    """标准化类别名称"""
    return name.strip()


def compute_entropy(probs: np.ndarray, epsilon=1e-10) -> float:
    """计算概率分布的熵"""
    probs = np.array(probs)
    probs = np.clip(probs, epsilon, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)


def normalize_entropy(entropy: float, num_classes: int = 8) -> float:
    """
    将熵归一化到[0, 1]区间
    max_entropy = log(num_classes)
    normalized = entropy / max_entropy
    """
    max_entropy = np.log(num_classes)
    return entropy / max_entropy


def compute_alpha_from_entropy(entropy_normalized: float,
                               alpha_min: float = 0.3,
                               alpha_max: float = 0.9,
                               steepness: float = 10.0) -> float:
    """
    基于归一化熵计算动态权重α

    Args:
        entropy_normalized: 归一化熵 [0, 1]，越大表示越不确定
        alpha_min: α的最小值（熵最大时）
        alpha_max: α的最大值（熵最小时）
        steepness: sigmoid函数的陡峭度

    Returns:
        α ∈ [alpha_min, alpha_max]

    策略：
    - 当熵低（MLLM自信）时，α接近alpha_max，更信任MLLM
    - 当熵高（MLLM不自信）时，α接近alpha_min，更信任Flow
    """
    # 使用sigmoid函数实现平滑过渡
    # 将entropy_normalized从[0,1]映射到[-1,1]，然后通过sigmoid
    x = (0.5 - entropy_normalized) * steepness
    sigmoid = 1.0 / (1.0 + np.exp(-x))

    # 映射到[alpha_min, alpha_max]
    alpha = alpha_min + (alpha_max - alpha_min) * sigmoid
    return float(alpha)


def extract_flow_probs(flow_item: Dict) -> np.ndarray:
    """
    从Flow结果中提取概率分布
    Flow使用距离(dist)，需要转换为概率
    """
    num_classes = len(CLASS_NAMES)
    pred_class = normalize_class_name(flow_item['pred'])

    # 如果Flow只给出了预测类别，构造一个尖锐的分布
    if 'probs' in flow_item:
        # 如果有完整概率分布
        return np.array(flow_item['probs'])
    else:
        # 构造one-hot式的尖锐分布
        probs = np.zeros(num_classes)
        if pred_class in CLASS_TO_IDX:
            pred_idx = CLASS_TO_IDX[pred_class]
            # 给预测类别高概率，其他类别小概率
            probs[pred_idx] = 0.95
            probs += 0.05 / num_classes  # 平滑
            probs = probs / probs.sum()  # 归一化
        else:
            # 如果类别不在映射中，使用均匀分布
            probs = np.ones(num_classes) / num_classes

        return probs


def extract_mllm_probs(mllm_item: Dict) -> np.ndarray:
    """从MLLM结果中提取概率分布"""
    num_classes = len(CLASS_NAMES)

    # 尝试从p_A, p_B, p_C, p_D中提取
    if all(f'p_{opt}' in mllm_item for opt in ['A', 'B', 'C', 'D']):
        probs_4 = np.array([
            mllm_item['p_A'],
            mllm_item['p_B'],
            mllm_item['p_C'],
            mllm_item['p_D']
        ])

        # 获取选项对应的类别
        option_classes = []
        for opt in ['A', 'B', 'C', 'D']:
            opt_key = f'option_{opt}'
            if opt_key in mllm_item:
                class_name = normalize_class_name(mllm_item[opt_key])
                option_classes.append(class_name)
            else:
                option_classes.append(None)

        # 构造完整的概率分布
        probs_full = np.zeros(num_classes)
        for i, class_name in enumerate(option_classes):
            if class_name and class_name in CLASS_TO_IDX:
                idx = CLASS_TO_IDX[class_name]
                probs_full[idx] = probs_4[i]

        # 归一化
        if probs_full.sum() > 0:
            probs_full = probs_full / probs_full.sum()
        else:
            probs_full = np.ones(num_classes) / num_classes

        return probs_full
    else:
        # 如果没有概率信息，返回均匀分布
        return np.ones(num_classes) / num_classes


def fusion_predictions(mllm_item: Dict,
                       flow_item: Dict,
                       alpha_min: float = 0.3,
                       alpha_max: float = 0.9,
                       steepness: float = 10.0) -> Dict:
    """
    融合MLLM和Flow的预测

    Returns:
        包含融合结果的字典
    """
    num_classes = len(CLASS_NAMES)

    # 提取两个模型的概率分布
    p_mllm = extract_mllm_probs(mllm_item)
    p_flow = extract_flow_probs(flow_item)

    # 计算MLLM的熵
    entropy_mllm = compute_entropy(p_mllm, epsilon=1e-10)
    entropy_normalized = normalize_entropy(entropy_mllm, num_classes)

    # 计算动态权重α
    alpha = compute_alpha_from_entropy(
        entropy_normalized,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        steepness=steepness
    )

    # 融合概率分布
    p_fusion = alpha * p_mllm + (1 - alpha) * p_flow

    # 归一化（理论上已经归一化，但为了数值稳定性）
    p_fusion = p_fusion / p_fusion.sum()

    # 获取融合后的预测
    fusion_pred_idx = int(np.argmax(p_fusion))
    fusion_pred_class = IDX_TO_CLASS[fusion_pred_idx]
    fusion_max_prob = float(p_fusion[fusion_pred_idx])

    # 计算融合后的熵
    fusion_entropy = compute_entropy(p_fusion)
    fusion_entropy_normalized = normalize_entropy(fusion_entropy, num_classes)

    # 获取ground truth
    gt_class = normalize_class_name(flow_item['gt'])

    # 判断是否正确
    fusion_correct = (fusion_pred_class == gt_class)
    mllm_correct = mllm_item.get('correct', 0) == 1
    flow_correct = flow_item.get('correct', True)

    return {
        'image_path': flow_item['x0'],
        'gt': gt_class,

        # MLLM信息
        'mllm_pred': normalize_class_name(mllm_item.get('option_' + mllm_item.get('pred_letter', 'A'), '')),
        'mllm_correct': mllm_correct,
        'mllm_max_prob': float(mllm_item.get('max_prob', 0)),
        'mllm_entropy': float(mllm_item.get('entropy', 0)),
        'mllm_entropy_normalized': entropy_normalized,
        'mllm_probs': p_mllm.tolist(),

        # Flow信息
        'flow_pred': normalize_class_name(flow_item['pred']),
        'flow_correct': flow_correct,
        'flow_dist': float(flow_item.get('dist', 0)),
        'flow_probs': p_flow.tolist(),

        # 融合信息
        'alpha': alpha,
        'fusion_pred': fusion_pred_class,
        'fusion_correct': fusion_correct,
        'fusion_max_prob': fusion_max_prob,
        'fusion_entropy': fusion_entropy,
        'fusion_entropy_normalized': fusion_entropy_normalized,
        'fusion_probs': p_fusion.tolist(),

        # 对比信息
        'improvement_over_mllm': fusion_correct and not mllm_correct,
        'improvement_over_flow': fusion_correct and not flow_correct,
        'degradation_from_mllm': not fusion_correct and mllm_correct,
        'degradation_from_flow': not fusion_correct and flow_correct,
    }


def load_data(mllm_path: str, flow_path: str) -> Tuple[List[Dict], List[Dict]]:
    """加载MLLM和Flow的结果数据"""
    with open(mllm_path, 'r', encoding='utf-8') as f:
        mllm_data = json.load(f)

    with open(flow_path, 'r', encoding='utf-8') as f:
        flow_data = json.load(f)

    return mllm_data, flow_data


def match_samples(mllm_data: List[Dict], flow_data: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """
    匹配MLLM和Flow的样本
    基于图像路径匹配
    """
    # 构建Flow数据的索引
    flow_index = {}
    for flow_item in flow_data:
        path = flow_item['x0']
        # 提取文件名作为key
        filename = Path(path).name
        flow_index[filename] = flow_item

    # 匹配样本
    matched_pairs = []
    unmatched_mllm = []

    for mllm_item in mllm_data:
        # 从MLLM的image_paths中提取文件名
        if 'image_paths' in mllm_item and len(mllm_item['image_paths']) > 0:
            mllm_path = mllm_item['image_paths'][0]
            filename = Path(mllm_path).name

            if filename in flow_index:
                matched_pairs.append((mllm_item, flow_index[filename]))
            else:
                unmatched_mllm.append(filename)

    if unmatched_mllm:
        print(f"警告: {len(unmatched_mllm)} 个MLLM样本未找到对应的Flow结果")
        print(f"示例: {unmatched_mllm[:3]}")

    return matched_pairs


def compute_statistics(fusion_results: List[Dict]) -> Dict:
    """计算融合结果的统计信息"""
    total = len(fusion_results)

    # 准确率统计
    mllm_correct = sum(1 for r in fusion_results if r['mllm_correct'])
    flow_correct = sum(1 for r in fusion_results if r['flow_correct'])
    fusion_correct = sum(1 for r in fusion_results if r['fusion_correct'])

    # 改进统计
    improved_over_mllm = sum(1 for r in fusion_results if r['improvement_over_mllm'])
    improved_over_flow = sum(1 for r in fusion_results if r['improvement_over_flow'])
    degraded_from_mllm = sum(1 for r in fusion_results if r['degradation_from_mllm'])
    degraded_from_flow = sum(1 for r in fusion_results if r['degradation_from_flow'])

    # α统计
    alphas = [r['alpha'] for r in fusion_results]

    # 按α范围分组统计
    alpha_ranges = {
        'high_confidence_mllm (α>0.7)': [r for r in fusion_results if r['alpha'] > 0.7],
        'medium_confidence (0.5<α≤0.7)': [r for r in fusion_results if 0.5 < r['alpha'] <= 0.7],
        'low_confidence_mllm (α≤0.5)': [r for r in fusion_results if r['alpha'] <= 0.5],
    }

    stats = {
        'total_samples': total,
        'accuracy': {
            'mllm': mllm_correct / total if total > 0 else 0,
            'flow': flow_correct / total if total > 0 else 0,
            'fusion': fusion_correct / total if total > 0 else 0,
        },
        'improvement': {
            'over_mllm_count': improved_over_mllm,
            'over_mllm_rate': improved_over_mllm / total if total > 0 else 0,
            'over_flow_count': improved_over_flow,
            'over_flow_rate': improved_over_flow / total if total > 0 else 0,
        },
        'degradation': {
            'from_mllm_count': degraded_from_mllm,
            'from_mllm_rate': degraded_from_mllm / total if total > 0 else 0,
            'from_flow_count': degraded_from_flow,
            'from_flow_rate': degraded_from_flow / total if total > 0 else 0,
        },
        'alpha_statistics': {
            'mean': float(np.mean(alphas)),
            'std': float(np.std(alphas)),
            'min': float(np.min(alphas)),
            'max': float(np.max(alphas)),
            'median': float(np.median(alphas)),
        },
        'by_alpha_range': {}
    }

    # 按α范围的统计
    for range_name, range_results in alpha_ranges.items():
        if len(range_results) > 0:
            stats['by_alpha_range'][range_name] = {
                'count': len(range_results),
                'fusion_accuracy': sum(1 for r in range_results if r['fusion_correct']) / len(range_results),
                'mllm_accuracy': sum(1 for r in range_results if r['mllm_correct']) / len(range_results),
                'flow_accuracy': sum(1 for r in range_results if r['flow_correct']) / len(range_results),
            }

    return stats


def main():
    parser = argparse.ArgumentParser(description='Logit Fusion Pipeline')
    parser.add_argument(
        '--mllm-path',
        default='/Users/tanshuyue/Documents/ssh_code/agent_latest/result/new_eval_tsy_llm_final.json',
        help='MLLM结果文件路径'
    )
    parser.add_argument(
        '--flow-path',
        default='/Users/tanshuyue/Documents/ssh_code/agent_latest/flow_result.json',
        help='Flow结果文件路径'
    )
    parser.add_argument(
        '--output-dir',
        default='/Users/tanshuyue/Documents/ssh_code/agent_latest/result',
        help='输出目录'
    )
    parser.add_argument(
        '--alpha-min',
        type=float,
        default=0.3,
        help='α的最小值（MLLM不自信时）'
    )
    parser.add_argument(
        '--alpha-max',
        type=float,
        default=0.9,
        help='α的最大值（MLLM自信时）'
    )
    parser.add_argument(
        '--steepness',
        type=float,
        default=10.0,
        help='sigmoid函数的陡峭度'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Logit Fusion Pipeline - 融合MLLM和Flow模型")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  MLLM结果: {args.mllm_path}")
    print(f"  Flow结果: {args.flow_path}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  α范围: [{args.alpha_min}, {args.alpha_max}]")
    print(f"  陡峭度: {args.steepness}")
    print()

    # 加载数据
    print("加载数据...")
    mllm_data, flow_data = load_data(args.mllm_path, args.flow_path)
    print(f"  MLLM样本数: {len(mllm_data)}")
    print(f"  Flow样本数: {len(flow_data)}")

    # 构建类别映射
    print("\n构建类别映射...")
    num_classes = build_class_mapping(mllm_data, flow_data)
    print(f"  发现 {num_classes} 个类别:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"    {i}: {class_name}")

    # 匹配样本
    print("\n匹配样本...")
    matched_pairs = match_samples(mllm_data, flow_data)
    print(f"  成功匹配: {len(matched_pairs)} 对样本")

    if len(matched_pairs) == 0:
        print("错误: 没有匹配的样本!")
        return

    # 执行融合
    print("\n执行融合...")
    fusion_results = []
    for mllm_item, flow_item in matched_pairs:
        result = fusion_predictions(
            mllm_item,
            flow_item,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            steepness=args.steepness
        )
        fusion_results.append(result)

    # 计算统计信息
    print("\n计算统计信息...")
    stats = compute_statistics(fusion_results)

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fusion_output_path = output_dir / 'fusion_results.json'
    stats_output_path = output_dir / 'fusion_statistics.json'

    with open(fusion_output_path, 'w', encoding='utf-8') as f:
        json.dump(fusion_results, f, indent=2, ensure_ascii=False)

    with open(stats_output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存:")
    print(f"  融合结果: {fusion_output_path}")
    print(f"  统计信息: {stats_output_path}")

    # 打印关键统计
    print("\n" + "=" * 80)
    print("融合结果统计")
    print("=" * 80)
    print(f"\n准确率对比:")
    print(f"  MLLM:   {stats['accuracy']['mllm']:.4f} ({int(stats['accuracy']['mllm'] * stats['total_samples'])}/{stats['total_samples']})")
    print(f"  Flow:   {stats['accuracy']['flow']:.4f} ({int(stats['accuracy']['flow'] * stats['total_samples'])}/{stats['total_samples']})")
    print(f"  Fusion: {stats['accuracy']['fusion']:.4f} ({int(stats['accuracy']['fusion'] * stats['total_samples'])}/{stats['total_samples']})")

    print(f"\n改进情况:")
    print(f"  相比MLLM: +{stats['improvement']['over_mllm_count']} 样本, -{stats['degradation']['from_mllm_count']} 样本")
    print(f"  相比Flow:  +{stats['improvement']['over_flow_count']} 样本, -{stats['degradation']['from_flow_count']} 样本")

    print(f"\nα权重统计:")
    print(f"  均值: {stats['alpha_statistics']['mean']:.4f}")
    print(f"  标准差: {stats['alpha_statistics']['std']:.4f}")
    print(f"  范围: [{stats['alpha_statistics']['min']:.4f}, {stats['alpha_statistics']['max']:.4f}]")

    print(f"\n按α范围的准确率:")
    for range_name, range_stats in stats['by_alpha_range'].items():
        print(f"  {range_name}:")
        print(f"    样本数: {range_stats['count']}")
        print(f"    Fusion准确率: {range_stats['fusion_accuracy']:.4f}")
        print(f"    MLLM准确率: {range_stats['mllm_accuracy']:.4f}")
        print(f"    Flow准确率: {range_stats['flow_accuracy']:.4f}")

    print("\n" + "=" * 80)
    print("完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
