#!/bin/bash
#SBATCH -J tsy_vllmServer
#SBATCH -p gpu
#SBATCH -N 1                          # 节点个数
#SBATCH --ntasks-per-node=1           # ← 关键！每个节点只跑 1 个主进程
#SBATCH --gres=gpu:8                  # 每个节点 8 张 GPU（供 torchrun 使用）
#SBATCH --cpus-per-task=64            # 给足 CPU（如 4~8 倍 GPU 数）

#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/run_scripts/tsy/slurm_out/vllm.out


source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate vllm_serve
#     --max-num-seqs 24 \
python -m vllm.entrypoints.openai.api_server \
    --model /mnt/inaisfs/data/home/tansy_criait/weights/Qwen3-VL-32B-Instruct \
    --dtype bfloat16 \
    --max-num-batched-tokens 96000 \
    --tensor-parallel-size 8 \
    --max-model-len 12000 \
    --max-num-seqs 256 \
    --port 8000 \
    --served-model-name tsy-medical-validator \
    --gpu-memory-utilization 0.95