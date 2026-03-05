#!/bin/bash
#SBATCH -J GRPO
#SBATCH -p gpu
#SBATCH -N 8                          # 节点个数
#SBATCH --ntasks-per-node=1           # ← 关键！每个节点只跑 1 个主进程
#SBATCH --gres=gpu:8                  # 每个节点 8 张 GPU（供 torchrun 使用）
#SBATCH --cpus-per-task=64            # 给足 CPU（如 4~8 倍 GPU 数）

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate github

export PYTHONPATH=/mnt/inaisfs/data/home/tansy_criait/jmf/GitHub/GDPO-main:$PYTHONPATH
# export DEBUG_MODE="true"
export WANDB_API_KEY="xxx"
export WANDB_MODE="offline"

ACCELERATE_LOG_LEVEL=info \
    accelerate launch \
    --config_file /mnt/inaisfs/data/home/tansy_criait/GasAgent-main/VLM-R1/src/open-r1-multimodal/local_scripts/zero3.yaml \
    --main_process_port 29500 \
    /mnt/inaisfs/data/home/tansy_criait/GasAgent-main/VLM-R1/VLM-R1/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py \
    --config /mnt/inaisfs/data/home/tansy_criait/GasAgent-main/VLM-R1/src/open-r1-multimodal/run_scripts/config.yaml \
    --vllm_mode colocate

