"""
================================================================================
完整Pipeline执行脚本
================================================================================
功能: 按顺序执行所有4个步骤，完成完整的推理和分析流程
使用方法: python run_full_pipeline.py
"""

import subprocess
import sys
import os
import time

# ===== 配置 =====
STEPS = [
    {
        'name': '步骤1: 模型推理 + Trigger策略分析',
        'script': '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/shierzhichang/step1_model_inference.py',
        'description': '对输入数据进行模型推理，计算概率分布和trigger指标',
    },
    {
        'name': '步骤2: LLM提取答案',
        'script': '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/shierzhichang/step2_extract_answers_with_llm_latest.py',
        'description': '使用LLM从生成文本中提取标准化答案',
    },
    {
        'name': '步骤3: 重新评估correct',
        'script': '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/shierzhichang/step3_reevaluate_correct.py',
        'description': '根据提取的答案重新计算correct字段',
    },
    {
        'name': '步骤4: 重新分析Trigger性能',
        'script': '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/shierzhichang/step4_reanalyze_trigger_performance.py',
        'description': '基于最终数据分析Trigger策略的性能',
    },
]

# ===== 执行函数 =====
def run_step(step_info: dict, step_num: int) -> bool:
    """
    执行单个步骤
    """
    print("\n" + "=" * 80)
    print(f"开始执行: {step_info['name']}")
    print(f"描述: {step_info['description']}")
    print("=" * 80 + "\n")

    script_path = step_info['script']

    # 检查脚本是否存在
    if not os.path.exists(script_path):
        print(f"错误: 找不到脚本文件 {script_path}")
        return False

    # 执行脚本
    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,  # 实时显示输出
            text=True
        )
        elapsed = time.time() - start_time
        print(f"\n✓ {step_info['name']} 完成 (耗时: {elapsed:.1f}秒)")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {step_info['name']} 失败 (耗时: {elapsed:.1f}秒)")
        print(f"错误信息: {e}")
        return False

    except KeyboardInterrupt:
        print(f"\n\n用户中断执行")
        return False


def main():
    """
    主函数：按顺序执行所有步骤
    """
    print("=" * 80)
    print("医疗VQA推理与分析完整Pipeline")
    print("=" * 80)
    print("\n流程概览:")
    for i, step in enumerate(STEPS, 1):
        print(f"  {i}. {step['name']}")
    print("\n" + "=" * 80)

    # 确认执行
    response = input("\n是否开始执行完整流程? (y/n): ").strip().lower()
    if response != 'y':
        print("已取消执行")
        return

    # 记录开始时间
    total_start_time = time.time()

    # 逐步执行
    for i, step in enumerate(STEPS, 1):
        success = run_step(step, i)

        if not success:
            print(f"\n{'='*80}")
            print(f"Pipeline在步骤{i}处停止")
            print(f"{'='*80}")
            sys.exit(1)

        # 在步骤之间短暂暂停
        if i < len(STEPS):
            time.sleep(1)

    # 全部完成
    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 80)
    print("🎉 完整Pipeline执行成功!")
    print(f"总耗时: {total_elapsed:.1f}秒 ({total_elapsed/60:.1f}分钟)")
    print("=" * 80)

    # 显示输出文件
    print("\n生成的文件:")
    print("  1. new_eval_tsy_llm_with_trigger.json - 推理结果 + trigger字段")
    print("  2. new_eval_tsy_llm_extracted.json - 添加LLM提取的答案")
    print("  3. new_eval_tsy_llm_final.json - 最终评估结果")
    print("  4. new_eval_tsy_llm_trigger_report.json - Trigger性能分析报告")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
