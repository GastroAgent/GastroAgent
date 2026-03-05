#!/bin/bash
#SBATCH -J tsy_rl
#SBATCH -p gpu
#SBATCH -N 4                          # number of nodes
#SBATCH --ntasks-per-node=1           # ← critical! only 1 main process per node
#SBATCH --gres=gpu:8                  # 8 GPUs per node (used by torchrun)
#SBATCH --cpus-per-task=64            # sufficient CPUs (e.g. 4~8x the number of GPUs)

#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/run_scripts/tsy/slurm_out/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-all-option-add-cot-sft-rl.out

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
srun bash /mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/run_scripts/tsy/rl/run_grpo_my_server_qwen_8gpu_nodes_run.sh