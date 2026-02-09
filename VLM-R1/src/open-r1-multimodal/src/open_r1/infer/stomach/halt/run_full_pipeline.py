"""
================================================================================
完整Pipeline执行脚本（集成HALT幻觉检测） - 批处理版
================================================================================
功能: 按顺序执行所有步骤，完成完整的推理、HALT幻觉检测和分析流程
使用方法: python run_full_pipeline.py [--mode full|inference|halt_only]
  --mode full: 执行完整流程（包括HALT训练和推理）
  --mode inference: 仅执行推理流程（跳过HALT训练）
  --mode halt_only: 仅执行HALT相关步骤

注意: 此版本已移除交互式确认，启动即运行。
"""

import subprocess
import sys
import os
import time
import argparse

# ===== 配置 =====
# 获取当前脚本所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# HALT相关步骤（可选）
HALT_STEPS = [
    {
        'name': '步骤0a: HALT数据准备',
        'script': os.path.join(CURRENT_DIR, 'prepare_halt_data.py'),
        'description': '提取中间层隐藏状态，生成HALT训练数据',
        'optional': True,
        'skip_if_exists': True,  # 如果已有训练数据则跳过
    },
    {
        'name': '步骤0b: HALT探针训练',
        'script': os.path.join(CURRENT_DIR, 'train_halt_probe.py'),
        'description': '训练轻量级探针模型用于幻觉检测',
        'optional': True,
        'skip_if_exists': True,  # 如果已有探针模型则跳过
    },
]

# 主推理和分析步骤
MAIN_STEPS = [
    {
        'name': '步骤1: 模型推理 + HALT幻觉检测',
        'script': '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/shiguan/halt/step1_model_inference.py',
        'description': '对输入数据进行模型推理，使用HALT方法检测幻觉风险',
    },
    {
        'name': '步骤2: LLM提取答案',
        'script': '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/shiguan/halt/step2_extract_answers_with_llm.py',
        'description': '使用LLM从生成文本中提取标准化答案',
    },
    {
        'name': '步骤3: 重新评估correct',
        'script': '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/shiguan/halt/step3_reevaluate_correct.py',
        'description': '根据提取的答案重新计算correct字段',
    },
    {
        'name': '步骤4: 分析Trigger性能',
        'script': '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/shiguan/halt/step4_reanalyze_trigger_performance.py',
        'description': '基于最终数据分析Trigger策略的性能',
    },
    {
        'name': '步骤5: HALT性能分析',
        'script': '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/shiguan/halt/step5_halt_performance_analysis.py',
        'description': '分析HALT幻觉检测性能，对比传统trigger策略',
        'optional': False,
    },
]

# ===== 执行函数 =====
def run_step(step_info: dict, step_num: int, skip_optional: bool = False) -> bool:
    """
    执行单个步骤
    """
    # 检查是否为可选步骤
    if step_info.get('optional', False) and skip_optional:
        print(f"\n⊘ 跳过可选步骤: {step_info['name']}")
        return True

    print("\n" + "=" * 80)
    print(f"开始执行: {step_info['name']}")
    print(f"描述: {step_info['description']}")
    print("=" * 80 + "\n")

    script_path = step_info['script']

    # 检查脚本是否存在
    if not os.path.exists(script_path):
        if step_info.get('optional', False):
            print(f"警告: 找不到可选脚本文件 {script_path}，跳过此步骤")
            return True
        else:
            print(f"错误: 找不到脚本文件 {script_path}")
            return False

    # 执行脚本
    start_time = time.time()
    try:
        # flush=True 确保日志在输出重定向时也能实时显示
        print(f"正在启动子进程: {script_path}", flush=True)
        
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,  # 实时显示输出
            text=True
        )
        elapsed = time.time() - start_time
        print(f"\n✓ {step_info['name']} 完成 (耗时: {elapsed:.1f}秒)", flush=True)
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {step_info['name']} 失败 (耗时: {elapsed:.1f}秒)", flush=True)
        print(f"错误信息: {e}", flush=True)

        # 可选步骤失败不中断流程
        if step_info.get('optional', False):
            print("这是可选步骤，继续执行后续步骤...", flush=True)
            return True
        return False

    except KeyboardInterrupt:
        print(f"\n\n用户中断执行", flush=True)
        return False


def main():
    """
    主函数：按顺序执行所有步骤
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='医疗VQA推理与HALT幻觉检测完整Pipeline')
    parser.add_argument('--mode', type=str, default='inference',
                        choices=['full', 'inference', 'halt_only'],
                        help='执行模式: full=完整流程(含HALT训练), inference=仅推理, halt_only=仅HALT')
    parser.add_argument('--skip-halt-training', action='store_true',
                        help='跳过HALT训练步骤（如果已有训练好的探针模型）')
    args = parser.parse_args()

    print("=" * 80)
    print("医疗VQA推理与HALT幻觉检测完整Pipeline (自动作业模式)")
    print("=" * 80)

    # 根据模式选择要执行的步骤
    if args.mode == 'halt_only':
        steps_to_run = HALT_STEPS
        print("\n执行模式: 仅HALT训练")
    elif args.mode == 'full':
        steps_to_run = HALT_STEPS + MAIN_STEPS
        print("\n执行模式: 完整流程（包括HALT训练和推理）")
    else:  # inference
        steps_to_run = MAIN_STEPS
        print("\n执行模式: 推理流程（跳过HALT训练）")

    print("\n流程概览:")
    for i, step in enumerate(steps_to_run, 1):
        optional_tag = " [可选]" if step.get('optional', False) else ""
        print(f"  {i}. {step['name']}{optional_tag}")
    print("\n" + "=" * 80)

    # --- 修改处：移除交互式确认，直接开始 ---
    print("\n作业自动开始执行...")
    # ------------------------------------

    # 记录开始时间
    total_start_time = time.time()

    # 逐步执行
    skip_optional = args.skip_halt_training
    for i, step in enumerate(steps_to_run, 1):
        success = run_step(step, i, skip_optional=skip_optional)

        if not success:
            print(f"\n{'='*80}")
            print(f"Pipeline在步骤{i}处停止")
            print(f"{'='*80}")
            sys.exit(1)

        # 在步骤之间短暂暂停
        if i < len(steps_to_run):
            time.sleep(1)

    # 全部完成
    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 80)
    print("🎉 完整Pipeline执行成功!")
    print(f"总耗时: {total_elapsed:.1f}秒 ({total_elapsed/60:.1f}分钟)")
    print("=" * 80)

    # 显示输出文件
    print("\n生成的文件:")
    if args.mode in ['full', 'halt_only']:
        print("  HALT相关:")
        print("    - train_with_hidden_states.json - HALT训练数据")
        print("    - halt_probe.pth - 训练好的探针模型")
    if args.mode in ['full', 'inference']:
        print("  推理结果:")
        print("    1. new_eval_tsy_llm_with_trigger.json - 推理结果 + HALT幻觉检测")
        print("    2. new_eval_tsy_llm_extracted.json - 添加LLM提取的答案")
        print("    3. new_eval_tsy_llm_final.json - 最终评估结果")
        print("    4. new_eval_tsy_llm_trigger_report.json - Trigger性能分析报告")
        print("    5. halt_performance_report.json - HALT性能分析报告")
    print("\n" + "=" * 80)
    print("作业结束", flush=True)


if __name__ == "__main__":
    main()