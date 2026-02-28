#!/bin/bash
#SBATCH -J sft-AllDiseaseMLLM
#SBATCH -p gpu
#SBATCH -N 1                          # 节点个数
#SBATCH --ntasks-per-node=1           # ← 关键！每个节点只跑 1 个主进程
#SBATCH --gres=gpu:8                  # 每个节点 8 张 GPU（供 torchrun 使用）
#SBATCH --cpus-per-task=64            # 给足 CPU（如 4~8 倍 GPU 数）

#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/run_scripts/tsy/slurm_out/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-75-option1-add-sft.out

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate llm

export PYTHONUNBUFFERED=1
# export NCCL_DEBUG=INFO
export MASTER_PORT=12345
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)

# echo "MASTER_ADDR=$MASTER_ADDR"
# echo "SLURM_NNODES=$SLURM_NNODES"
# echo "SLURM_NODEID=$SLURM_NODEID"

# 只启动一次脚本（每个节点 1 次）
srun bash /mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/run_scripts/tsy/sft_data_v1/stage3_generate_lora_run2.sh