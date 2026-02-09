"""
================================================================================
HALT V2 评估和可视化工具
================================================================================
功能:
1. 全面的性能评估
2. 阈值敏感性分析
3. 错误案例分析
4. 可视化报告生成
5. 与baseline对比
================================================================================
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple
import pandas as pd
from collections import defaultdict

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False


class HALTV2Evaluator:
    """HALT V2评估器"""

    def __init__(self, output_dir: str = "./evaluation_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def evaluate(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5,
        question_types: List[str] = None
    ) -> Dict:
        """
        全面评估模型性能

        Args:
            predictions: 预测概率 (N,)
            labels: 真实标签 (N,)
            threshold: 分类阈值
            question_types: 问题类型列表

        Returns:
            评估结果字典
        """
        # 二值化预测
        pred_binary = (predictions >= threshold).astype(int)

        # 基础指标
        accuracy = accuracy_score(labels, pred_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, pred_binary, average='binary', zero_division=0
        )

        # 混淆矩阵
        tn, fp, fn, tp = confusion_matrix(labels, pred_binary).ravel()

        # AUC
        try:
            auc_roc = roc_auc_score(labels, predictions)
        except:
            auc_roc = 0.0

        # 基础结果
        results = {
            "threshold": threshold,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc_roc": float(auc_roc),
            "confusion_matrix": {
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn)
            },
            "total_samples": len(labels),
            "positive_samples": int(labels.sum()),
            "negative_samples": int((1 - labels).sum())
        }

        # 按问题类型评估
        if question_types is not None:
            results["per_question_type"] = self._evaluate_by_question_type(
                predictions, labels, pred_binary, question_types
            )

        # 风险分数分布
        results["risk_distribution"] = self._analyze_risk_distribution(
            predictions, labels
        )

        return results

    def _evaluate_by_question_type(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        pred_binary: np.ndarray,
        question_types: List[str]
    ) -> Dict:
        """按问题类型评估"""
        type_results = {}

        for qt in set(question_types):
            indices = [i for i, q in enumerate(question_types) if q == qt]
            if len(indices) == 0:
                continue

            qt_preds = pred_binary[indices]
            qt_labels = labels[indices]
            qt_probs = predictions[indices]

            # 计算指标
            acc = accuracy_score(qt_labels, qt_preds)
            prec, rec, f1, _ = precision_recall_fscore_support(
                qt_labels, qt_preds, average='binary', zero_division=0
            )

            try:
                auc = roc_auc_score(qt_labels, qt_probs)
            except:
                auc = 0.0

            type_results[qt] = {
                "count": len(indices),
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "auc_roc": float(auc)
            }

        return type_results

    def _analyze_risk_distribution(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """分析风险分数分布"""
        correct_mask = (labels == 0)
        incorrect_mask = (labels == 1)

        correct_scores = predictions[correct_mask]
        incorrect_scores = predictions[incorrect_mask]

        return {
            "correct_samples": {
                "count": int(correct_mask.sum()),
                "mean": float(correct_scores.mean()) if len(correct_scores) > 0 else 0.0,
                "std": float(correct_scores.std()) if len(correct_scores) > 0 else 0.0,
                "min": float(correct_scores.min()) if len(correct_scores) > 0 else 0.0,
                "max": float(correct_scores.max()) if len(correct_scores) > 0 else 0.0,
                "median": float(np.median(correct_scores)) if len(correct_scores) > 0 else 0.0
            },
            "incorrect_samples": {
                "count": int(incorrect_mask.sum()),
                "mean": float(incorrect_scores.mean()) if len(incorrect_scores) > 0 else 0.0,
                "std": float(incorrect_scores.std()) if len(incorrect_scores) > 0 else 0.0,
                "min": float(incorrect_scores.min()) if len(incorrect_scores) > 0 else 0.0,
                "max": float(incorrect_scores.max()) if len(incorrect_scores) > 0 else 0.0,
                "median": float(np.median(incorrect_scores)) if len(incorrect_scores) > 0 else 0.0
            }
        }

    def threshold_sensitivity_analysis(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        thresholds: List[float] = None
    ) -> Dict:
        """阈值敏感性分析"""
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.05)

        results = {}
        for threshold in thresholds:
            pred_binary = (predictions >= threshold).astype(int)

            acc = accuracy_score(labels, pred_binary)
            prec, rec, f1, _ = precision_recall_fscore_support(
                labels, pred_binary, average='binary', zero_division=0
            )
            tn, fp, fn, tp = confusion_matrix(labels, pred_binary).ravel()

            results[f"threshold_{threshold:.2f}"] = {
                "threshold": float(threshold),
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn)
            }

        return results

    def find_optimal_threshold(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        metric: str = "f1"
    ) -> Tuple[float, float]:
        """
        找到最优阈值

        Args:
            predictions: 预测概率
            labels: 真实标签
            metric: 优化指标 ("f1", "accuracy", "precision", "recall")

        Returns:
            (最优阈值, 最优指标值)
        """
        thresholds = np.arange(0.1, 1.0, 0.01)
        best_threshold = 0.5
        best_score = 0.0

        for threshold in thresholds:
            pred_binary = (predictions >= threshold).astype(int)

            if metric == "f1":
                _, _, score, _ = precision_recall_fscore_support(
                    labels, pred_binary, average='binary', zero_division=0
                )
            elif metric == "accuracy":
                score = accuracy_score(labels, pred_binary)
            elif metric == "precision":
                score, _, _, _ = precision_recall_fscore_support(
                    labels, pred_binary, average='binary', zero_division=0
                )
            elif metric == "recall":
                _, score, _, _ = precision_recall_fscore_support(
                    labels, pred_binary, average='binary', zero_division=0
                )
            else:
                raise ValueError(f"不支持的指标: {metric}")

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold, best_score

    def compare_with_baseline(
        self,
        halt_v2_results: Dict,
        baseline_results: Dict
    ) -> Dict:
        """与baseline对比"""
        comparison = {
            "halt_v2": halt_v2_results,
            "baseline": baseline_results,
            "improvements": {}
        }

        # 计算改进
        for metric in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
            halt_v2_value = halt_v2_results.get(metric, 0.0)
            baseline_value = baseline_results.get(metric, 0.0)
            improvement = halt_v2_value - baseline_value
            improvement_pct = (improvement / baseline_value * 100) if baseline_value > 0 else 0.0

            comparison["improvements"][metric] = {
                "absolute": float(improvement),
                "percentage": float(improvement_pct)
            }

        return comparison

    # ===== 可视化函数 =====

    def plot_confusion_matrix(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        threshold: float = 0.5,
        save_path: str = None
    ):
        """绘制混淆矩阵"""
        pred_binary = (predictions >= threshold).astype(int)
        cm = confusion_matrix(labels, pred_binary)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['正确', '错误'],
            yticklabels=['正确', '错误']
        )
        plt.title(f'混淆矩阵 (阈值={threshold:.2f})')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_risk_distribution(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        save_path: str = None
    ):
        """绘制风险分数分布"""
        correct_scores = predictions[labels == 0]
        incorrect_scores = predictions[labels == 1]

        plt.figure(figsize=(12, 5))

        # 直方图
        plt.subplot(1, 2, 1)
        plt.hist(correct_scores, bins=50, alpha=0.6, label='正确样本', color='green')
        plt.hist(incorrect_scores, bins=50, alpha=0.6, label='错误样本', color='red')
        plt.xlabel('风险分数')
        plt.ylabel('频数')
        plt.title('风险分数分布')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 箱线图
        plt.subplot(1, 2, 2)
        data = [correct_scores, incorrect_scores]
        plt.boxplot(data, labels=['正确样本', '错误样本'])
        plt.ylabel('风险分数')
        plt.title('风险分数箱线图')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_roc_curve(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        save_path: str = None
    ):
        """绘制ROC曲线"""
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        auc = roc_auc_score(labels, predictions)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'HALT V2 (AUC={auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
        plt.xlabel('假阳性率 (FPR)')
        plt.ylabel('真阳性率 (TPR)')
        plt.title('ROC曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_pr_curve(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        save_path: str = None
    ):
        """绘制PR曲线"""
        precision, recall, thresholds = precision_recall_curve(labels, predictions)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('召回率 (Recall)')
        plt.ylabel('精确率 (Precision)')
        plt.title('Precision-Recall曲线')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_threshold_sensitivity(
        self,
        threshold_results: Dict,
        save_path: str = None
    ):
        """绘制阈值敏感性分析"""
        thresholds = []
        accuracies = []
        precisions = []
        recalls = []
        f1s = []

        for key, metrics in sorted(threshold_results.items()):
            thresholds.append(metrics["threshold"])
            accuracies.append(metrics["accuracy"])
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])
            f1s.append(metrics["f1"])

        plt.figure(figsize=(12, 6))
        plt.plot(thresholds, accuracies, marker='o', label='准确率')
        plt.plot(thresholds, precisions, marker='s', label='精确率')
        plt.plot(thresholds, recalls, marker='^', label='召回率')
        plt.plot(thresholds, f1s, marker='d', label='F1分数', linewidth=2)

        plt.xlabel('阈值')
        plt.ylabel('指标值')
        plt.title('阈值敏感性分析')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_comparison_bar(
        self,
        comparison: Dict,
        save_path: str = None
    ):
        """绘制对比柱状图"""
        metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
        halt_v2_values = [comparison["halt_v2"].get(m, 0.0) for m in metrics]
        baseline_values = [comparison["baseline"].get(m, 0.0) for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, halt_v2_values, width, label='HALT V2', color='steelblue')
        plt.bar(x + width/2, baseline_values, width, label='Baseline', color='coral')

        plt.xlabel('指标')
        plt.ylabel('值')
        plt.title('HALT V2 vs Baseline 性能对比')
        plt.xticks(x, ['准确率', '精确率', '召回率', 'F1分数', 'AUC-ROC'])
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, (v1, v2) in enumerate(zip(halt_v2_values, baseline_values)):
            plt.text(i - width/2, v1 + 0.01, f'{v1:.3f}', ha='center', va='bottom')
            plt.text(i + width/2, v2 + 0.01, f'{v2:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5,
        question_types: List[str] = None,
        baseline_results: Dict = None
    ) -> Dict:
        """
        生成完整的评估报告

        Args:
            predictions: 预测概率
            labels: 真实标签
            threshold: 分类阈值
            question_types: 问题类型列表
            baseline_results: baseline结果（用于对比）

        Returns:
            完整的评估报告
        """
        print("=" * 80)
        print("HALT V2 评估报告")
        print("=" * 80)

        # 1. 基础评估
        print("\n1. 基础性能评估...")
        results = self.evaluate(predictions, labels, threshold, question_types)

        # 2. 阈值敏感性分析
        print("2. 阈值敏感性分析...")
        threshold_results = self.threshold_sensitivity_analysis(predictions, labels)
        results["threshold_sensitivity"] = threshold_results

        # 3. 找到最优阈值
        print("3. 寻找最优阈值...")
        optimal_threshold, optimal_f1 = self.find_optimal_threshold(predictions, labels, "f1")
        results["optimal_threshold"] = {
            "threshold": float(optimal_threshold),
            "f1": float(optimal_f1)
        }
        print(f"   最优阈值: {optimal_threshold:.3f} (F1={optimal_f1:.3f})")

        # 4. 与baseline对比
        if baseline_results is not None:
            print("4. 与Baseline对比...")
            comparison = self.compare_with_baseline(results, baseline_results)
            results["comparison"] = comparison

        # 5. 生成可视化
        print("\n5. 生成可视化...")
        self.plot_confusion_matrix(
            labels, predictions, threshold,
            os.path.join(self.output_dir, "confusion_matrix.png")
        )
        self.plot_risk_distribution(
            predictions, labels,
            os.path.join(self.output_dir, "risk_distribution.png")
        )
        self.plot_roc_curve(
            labels, predictions,
            os.path.join(self.output_dir, "roc_curve.png")
        )
        self.plot_pr_curve(
            labels, predictions,
            os.path.join(self.output_dir, "pr_curve.png")
        )
        self.plot_threshold_sensitivity(
            threshold_results,
            os.path.join(self.output_dir, "threshold_sensitivity.png")
        )

        if baseline_results is not None:
            self.plot_comparison_bar(
                comparison,
                os.path.join(self.output_dir, "comparison.png")
            )

        # 6. 保存报告
        report_path = os.path.join(self.output_dir, "evaluation_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n评估报告已保存: {report_path}")
        print(f"可视化图表已保存到: {self.output_dir}")

        # 7. 打印摘要
        print("\n" + "=" * 80)
        print("评估摘要")
        print("=" * 80)
        print(f"总样本数: {results['total_samples']}")
        print(f"正样本数: {results['positive_samples']} ({results['positive_samples']/results['total_samples']*100:.1f}%)")
        print(f"负样本数: {results['negative_samples']} ({results['negative_samples']/results['total_samples']*100:.1f}%)")
        print(f"\n当前阈值: {threshold:.3f}")
        print(f"准确率: {results['accuracy']:.4f}")
        print(f"精确率: {results['precision']:.4f}")
        print(f"召回率: {results['recall']:.4f}")
        print(f"F1分数: {results['f1']:.4f}")
        print(f"AUC-ROC: {results['auc_roc']:.4f}")
        print(f"\n混淆矩阵:")
        print(f"  TP: {results['confusion_matrix']['tp']}, FP: {results['confusion_matrix']['fp']}")
        print(f"  TN: {results['confusion_matrix']['tn']}, FN: {results['confusion_matrix']['fn']}")

        if baseline_results is not None:
            print(f"\n与Baseline对比:")
            for metric, improvement in comparison["improvements"].items():
                print(f"  {metric}: {improvement['absolute']:+.4f} ({improvement['percentage']:+.2f}%)")

        print("=" * 80)

        return results


if __name__ == "__main__":
    # 测试代码
    print("=== 测试HALT V2评估器 ===\n")

    # 生成模拟数据
    np.random.seed(42)
    n_samples = 100

    # 模拟预测和标签
    labels = np.random.randint(0, 2, n_samples)
    predictions = np.random.rand(n_samples)
    # 让预测与标签有一定相关性
    predictions[labels == 1] += 0.3
    predictions = np.clip(predictions, 0, 1)

    question_types = np.random.choice(['Disease Diagnosis', 'Anatomy Identification'], n_samples)

    # 创建评估器
    evaluator = HALTV2Evaluator(output_dir="./test_evaluation")

    # 生成报告
    results = evaluator.generate_report(
        predictions=predictions,
        labels=labels,
        threshold=0.5,
        question_types=question_types.tolist()
    )

    print("\n测试完成！")
