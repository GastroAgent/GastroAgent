"""
================================================================================
完整Pipeline执行脚本 (批处理自动版)
================================================================================
功能: 按顺序执行所有4个步骤，完成完整的推理和分析流程
特点: 无需人工交互，适合 sbatch/nohup 提交
"""

import subprocess
import sys
import os
import time

# ===== 配置 =====
STEPS = [
    {
        'name': '步骤1: 模型推理 + Trigger策略分析',
        'script': '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/shiguan/step1_model_inference.py',
        'description': '对输入数据进行模型推理，计算概率分布和trigger指标',
    },
    {
        'name': '步骤2: LLM提取答案',
        'script': '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/shiguan/step2_extract_answers_with_llm.py',
        'description': '使用LLM从生成文本中提取标准化答案',
    },
    {
        'name': '步骤3: 重新评估correct',
        'script': '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/shiguan/step3_reevaluate_correct.py',
        'description': '根据提取的答案重新计算correct字段',
    },
    {
        'name': '步骤4: 重新分析Trigger性能',
        'script': '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/shiguan/step4_reanalyze_trigger_performance.py',
        'description': '基于最终数据分析Trigger策略的性能',
    },
]

# ===== 执行函数 =====
def run_step(step_info: dict, step_num: int) -> bool:
    """
    执行单个步骤
    """
    print("\n" + "=" * 80, flush=True)
    print(f"开始执行: {step_info['name']}", flush=True)
    print(f"描述: {step_info['description']}", flush=True)
    print("=" * 80 + "\n", flush=True)

    script_path = step_info['script']

    # 检查脚本是否存在
    if not os.path.exists(script_path):
        print(f"错误: 找不到脚本文件 {script_path}", flush=True)
        return False

    # 执行脚本
    start_time = time.time()
    try:
        # 使用 sys.executable 确保使用当前环境的 python解释器
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,  # 设为False，让子脚本的输出直接打印到当前标准输出
            text=True
        )
        elapsed = time.time() - start_time
        print(f"\n✓ {step_info['name']} 完成 (耗时: {elapsed:.1f}秒)", flush=True)
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {step_info['name']} 失败 (耗时: {elapsed:.1f}秒)", flush=True)
        print(f"错误信息: {e}", flush=True)
        return False

    except KeyboardInterrupt:
        print(f"\n\n用户中断执行", flush=True)
        return False


def main():
    """
    主函数：按顺序执行所有步骤
    """
    print("=" * 80)
    print("医疗VQA推理与分析完整Pipeline (批处理作业模式)")
    print("=" * 80)
    print("\n流程概览:")
    for i, step in enumerate(STEPS, 1):
        print(f"  {i}. {step['name']}")
    print("\n" + "=" * 80)

    # --- 修改处：移除了 input() 交互，直接开始 ---
    print("\n[系统] 作业自动开始执行...", flush=True)
    # ----------------------------------------

    # 记录开始时间
    total_start_time = time.time()

    # 逐步执行
    for i, step in enumerate(STEPS, 1):
        success = run_step(step, i)

        if not success:
            print(f"\n{'='*80}", flush=True)
            print(f"Pipeline在步骤{i}处停止", flush=True)
            print(f"{'='*80}", flush=True)
            sys.exit(1) # 返回非零状态码，告知作业系统任务失败

        # 在步骤之间短暂暂停
        if i < len(STEPS):
            time.sleep(1)

    # 全部完成
    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 80, flush=True)
    print("🎉 完整Pipeline执行成功!", flush=True)
    print(f"总耗时: {total_elapsed:.1f}秒 ({total_elapsed/60:.1f}分钟)", flush=True)
    print("=" * 80, flush=True)

    # 显示输出文件
    print("\n生成的文件:", flush=True)
    print("  1. new_eval_tsy_llm_with_trigger.json - 推理结果 + trigger字段")
    print("  2. new_eval_tsy_llm_extracted.json - 添加LLM提取的答案")
    print("  3. new_eval_tsy_llm_final.json - 最终评估结果")
    print("  4. new_eval_tsy_llm_trigger_report.json - Trigger性能分析报告")
    print("\n" + "=" * 80, flush=True)


if __name__ == "__main__":
    main()