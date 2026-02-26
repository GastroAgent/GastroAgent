#!/usr/bin/env python3
"""
================================================================================
独立运行Logit Fusion融合
================================================================================
功能: 单独运行MLLM和Flow的融合，无需重新推理
使用方法: python run_fusion_only.py [--alpha-min 0.3] [--alpha-max 0.9]
"""

import subprocess
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='运行Logit Fusion融合')
    parser.add_argument(
        '--mllm-path',
        default='/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New5/new_eval_tsy_llm_final.json',
        help='MLLM结果文件路径（相对于agent_latest目录）'
    )
    parser.add_argument(
        '--flow-path',
        default='/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New5/kshot/eval_by_x0_clean.json',
        help='Flow结果文件路径（相对于agent_latest目录）'
    )
    parser.add_argument(
        '--output-dir',
        default='/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New5/fusion',
        help='输出目录（相对于agent_latest目录）'
    )
    parser.add_argument(
        '--alpha-min',
        type=float,
        default=0.1,
        help='α的最小值（MLLM不自信时，默认0.3）'
    )
    parser.add_argument(
        '--alpha-max',
        type=float,
        default=0.4,
        help='α的最大值（MLLM自信时，默认0.9）'
    )
    parser.add_argument(
        '--steepness',
        type=float,
        default=10.0,
        help='sigmoid函数的陡峭度（默认10.0）'
    )

    args = parser.parse_args()

    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建绝对路径
    mllm_path = os.path.join(script_dir, args.mllm_path)
    flow_path = os.path.join(script_dir, args.flow_path)
    output_dir = os.path.join(script_dir, args.output_dir)
    fusion_script = os.path.join(script_dir, 'fusion_pipeline.py')

    print("=" * 80)
    print("Logit Fusion - 独立运行模式")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  MLLM结果: {mllm_path}")
    print(f"  Flow结果: {flow_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  α范围: [{args.alpha_min}, {args.alpha_max}]")
    print(f"  陡峭度: {args.steepness}")
    print("\n" + "=" * 80 + "\n")

    # 检查文件是否存在
    if not os.path.exists(mllm_path):
        print(f"错误: MLLM结果文件不存在: {mllm_path}")
        sys.exit(1)

    if not os.path.exists(flow_path):
        print(f"错误: Flow结果文件不存在: {flow_path}")
        sys.exit(1)

    if not os.path.exists(fusion_script):
        print(f"错误: 融合脚本不存在: {fusion_script}")
        sys.exit(1)

    # 构建命令
    cmd = [
        sys.executable,
        fusion_script,
        '--mllm-path', mllm_path,
        '--flow-path', flow_path,
        '--output-dir', output_dir,
        '--alpha-min', str(args.alpha_min),
        '--alpha-max', str(args.alpha_max),
        '--steepness', str(args.steepness),
    ]

    # 执行融合
    try:
        result = subprocess.run(
            cmd,
            check=True,
            env=os.environ.copy()
        )
        print("\n" + "=" * 80)
        print("✓ 融合完成!")
        print("=" * 80)
        return 0

    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 80)
        print(f"✗ 融合失败，错误代码: {e.returncode}")
        print("=" * 80)
        return 1

    except KeyboardInterrupt:
        print("\n\n用户中断执行")
        return 1


if __name__ == "__main__":
    sys.exit(main())
