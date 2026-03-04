#!/bin/bash
#SBATCH -J vit-gan
#SBATCH -p gpu
#SBATCH -N 2                          # 节点个数
#SBATCH --ntasks-per-node=1           # ← 关键！每个节点只跑 1 个主进程
#SBATCH --gres=gpu:8                  # 每个节点 8 张 GPU（供 torchrun 使用）
#SBATCH --cpus-per-task=64            # 给足 CPU（如 4~8 倍 GPU 数）

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

export PYTHONUNBUFFERED=1
# export NCCL_DEBUG=INFO
export MASTER_PORT=12345
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)

# echo "MASTER_ADDR=$MASTER_ADDR"
# echo "SLURM_NNODES=$SLURM_NNODES"
# echo "SLURM_NODEID=$SLURM_NODEID"

srun bash /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/vae_train_dist_gan.sh