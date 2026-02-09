#!/usr/bin/env python3
"""
================================================================================
HALT V2 快速开始脚本
================================================================================
功能: 一键运行HALT V2的训练、评估和对比
使用: python quick_start.py [--mode train|eval|compare|all]
================================================================================
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path

def print_banner(text):
    """打印横幅"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")

def check_dependencies():
    """检查依赖"""
    print_banner("检查依赖")

    required_packages = [
        "torch",
        "transformers",
        "sklearn",
        "matplotlib",
        "seaborn",
        "tqdm",
        "numpy",
        "pandas"
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (缺失)")
            missing.append(package)

    if missing:
        print(f"\n缺少以下依赖: {', '.join(missing)}")
        print("请运行: pip install " + " ".join(missing))
        return False

    print("\n所有依赖已安装！")
    return True

def check_data():
    """检查数据文件"""
    print_banner("检查数据文件")

    data_dir = "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/halt/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-all-RL-2500/merge"
    required_files = [
        "train_split_with_hidden_states.json",
        "val_split_with_hidden_states.json"
    ]

    all_exist = True
    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename} (不存在)")
            all_exist = False

    if not all_exist:
        print(f"\n请确保数据文件存在于: {data_dir}")
        return False

    print("\n所有数据文件就绪！")
    return True

def train_model():
    """训练HALT V2模型"""
    print_banner("训练HALT V2模型")

    print("开始训练...")
    print("这可能需要一些时间，请耐心等待...\n")

    try:
        # 运行训练脚本
        result = subprocess.run(
            ["python", "train_halt_v2.py"],
            cwd="/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/stomach/halt/v2",
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("✓ 训练完成！")
            print(result.stdout)
            return True
        else:
            print("✗ 训练失败！")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"✗ 训练出错: {e}")
        return False

def evaluate_model():
    """评估HALT V2模型"""
    print_banner("评估HALT V2模型")

    print("开始评估...")

    try:
        # 运行评估脚本
        result = subprocess.run(
            ["python", "halt_v2_evaluation.py"],
            cwd="/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/stomach/halt/v2",
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("✓ 评估完成！")
            print(result.stdout)
            return True
        else:
            print("✗ 评估失败！")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"✗ 评估出错: {e}")
        return False

def compare_with_baseline():
    """与baseline对比"""
    print_banner("与Baseline对比")

    # 加载HALT V2结果
    v2_report_path = "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/stomach/halt/v2/evaluation_results/evaluation_report.json"
    if not os.path.exists(v2_report_path):
        print("✗ 未找到HALT V2评估报告")
        print("请先运行评估: python quick_start.py --mode eval")
        return False

    with open(v2_report_path, 'r') as f:
        v2_results = json.load(f)

    # 加载V1结果
    v1_report_path = "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/stomach/halt/v2/result/halt_performance_report.json"
    if not os.path.exists(v1_report_path):
        print("✗ 未找到HALT V1评估报告")
        return False

    with open(v1_report_path, 'r') as f:
        v1_results = json.load(f)

    # 对比
    print("\n性能对比:")
    print("-" * 80)
    print(f"{'指标':<20} {'HALT V1':<15} {'HALT V2':<15} {'改进':<15}")
    print("-" * 80)

    v1_metrics = v1_results["halt_performance"]["metrics"]
    v2_metrics = v2_results

    metrics = [
        ("准确率", "accuracy"),
        ("精确率", "precision"),
        ("召回率", "recall"),
        ("F1分数", "f1"),
        ("AUC-ROC", "auc_roc")
    ]

    for name, key in metrics:
        v1_value = v1_metrics.get(key, 0.0)
        v2_value = v2_metrics.get(key, 0.0)
        improvement = v2_value - v1_value
        improvement_pct = (improvement / v1_value * 100) if v1_value > 0 else 0.0

        print(f"{name:<20} {v1_value:<15.4f} {v2_value:<15.4f} {improvement:+.4f} ({improvement_pct:+.2f}%)")

    print("-" * 80)

    # 混淆矩阵对比
    print("\n混淆矩阵对比:")
    print("-" * 80)
    print(f"{'指标':<20} {'HALT V1':<15} {'HALT V2':<15} {'改进':<15}")
    print("-" * 80)

    v1_cm = v1_metrics
    v2_cm = v2_metrics["confusion_matrix"]

    cm_metrics = [
        ("真阳性 (TP)", "tp"),
        ("假阳性 (FP)", "fp"),
        ("真阴性 (TN)", "tn"),
        ("假阴性 (FN)", "fn")
    ]

    for name, key in cm_metrics:
        v1_value = v1_cm.get(key, 0)
        v2_value = v2_cm.get(key, 0)
        improvement = v2_value - v1_value

        print(f"{name:<20} {v1_value:<15} {v2_value:<15} {improvement:+}")

    print("-" * 80)

    # 关键改进
    print("\n关键改进:")
    if v2_cm["tn"] > 0 and v1_cm["tn"] == 0:
        print("✓ 解决了零真阴性问题！")
    if v2_cm["fp"] < v1_cm["fp"]:
        print(f"✓ 减少了 {v1_cm['fp'] - v2_cm['fp']} 个假阳性")
    if v2_metrics["f1"] > v1_metrics["f1"]:
        print(f"✓ F1分数提升了 {(v2_metrics['f1'] - v1_metrics['f1']) * 100:.2f}%")

    # 风险分数分布对比
    print("\n风险分数分布对比:")
    print("-" * 80)

    v1_dist = v1_results["risk_score_distribution"]
    v2_dist = v2_results["risk_distribution"]

    print(f"{'样本类型':<20} {'HALT V1 均值':<15} {'HALT V2 均值':<15} {'差异':<15}")
    print("-" * 80)

    v1_correct_mean = v1_dist["correct_samples"]["mean"]
    v1_incorrect_mean = v1_dist["incorrect_samples"]["mean"]
    v1_diff = abs(v1_incorrect_mean - v1_correct_mean)

    v2_correct_mean = v2_dist["correct_samples"]["mean"]
    v2_incorrect_mean = v2_dist["incorrect_samples"]["mean"]
    v2_diff = abs(v2_incorrect_mean - v2_correct_mean)

    print(f"{'正确样本':<20} {v1_correct_mean:<15.4f} {v2_correct_mean:<15.4f}")
    print(f"{'错误样本':<20} {v1_incorrect_mean:<15.4f} {v2_incorrect_mean:<15.4f}")
    print(f"{'分数差异':<20} {v1_diff:<15.4f} {v2_diff:<15.4f} {v2_diff - v1_diff:+.4f}")
    print("-" * 80)

    if v2_diff > v1_diff:
        improvement_factor = v2_diff / v1_diff if v1_diff > 0 else float('inf')
        print(f"\n✓ 特征区分度提升了 {improvement_factor:.1f} 倍！")

    return True

def main():
    parser = argparse.ArgumentParser(description="HALT V2 快速开始脚本")
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "compare", "all"],
        default="all",
        help="运行模式: train(训练), eval(评估), compare(对比), all(全部)"
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="跳过依赖和数据检查"
    )

    args = parser.parse_args()

    print_banner("HALT V2 快速开始")

    # # 检查
    # if not args.skip_checks:
    #     if not check_dependencies():
    #         return 1

    #     if not check_data():
    #         print("\n提示: 如果数据路径不同，请修改 halt_v2_config.py 中的 PATH_CONFIG")
    #         return 1

    # 执行
    success = True

    if args.mode in ["train", "all"]:
        if not train_model():
            success = False

    if args.mode in ["eval", "all"]:
        if not evaluate_model():
            success = False

    if args.mode in ["compare", "all"]:
        if not compare_with_baseline():
            success = False

    # 总结
    print_banner("完成")

    if success:
        print("✓ 所有任务成功完成！")
        print("\n生成的文件:")
        print("  - 模型权重: agent/halt/models/halt_v2_probe.pth")
        print("  - 评估报告: agent/halt/evaluation_results/evaluation_report.json")
        print("  - 可视化图表: agent/halt/evaluation_results/*.png")
        print("\n下一步:")
        print("  1. 查看评估报告了解详细性能")
        print("  2. 查看可视化图表分析结果")
        print("  3. 根据需要调整配置并重新训练")
        return 0
    else:
        print("✗ 部分任务失败，请查看上面的错误信息")
        return 1

if __name__ == "__main__":
    sys.exit(main())
