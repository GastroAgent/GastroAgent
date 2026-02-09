#!/bin/bash
#SBATCH -J vllmServer
#SBATCH -p gpu
#SBATCH -N 1                          # 节点个数
#SBATCH --ntasks-per-node=1           # ← 关键！每个节点只跑 1 个主进程
#SBATCH --gres=gpu:4                  # 每个节点 8 张 GPU（供 torchrun 使用）
#SBATCH --cpus-per-task=32            # 给足 CPU（如 4~8 倍 GPU 数）

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate vllm_serve

python -m vllm.entrypoints.openai.api_server \
    --model /mnt/inaisfs/data/home/tansy_criait/weights/Qwen3-VL-32B-Instruct \
    --dtype bfloat16 \
    --tensor-parallel-size 4 \
    --port 8000 \
    --served-model-name medical-validator \
    --gpu_memory_utilization 0.9 \
    --max_model_len 16384
