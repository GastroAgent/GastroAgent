#!/bin/bash
#SBATCH -J sft-AllDiseaseMLLM
#SBATCH -p gpu
#SBATCH -N 4                          # Number of nodes
#SBATCH --ntasks-per-node=1           # ← Key! Only 1 main process per node
#SBATCH --gres=gpu:8                  # 8 GPUs per node (used by torchrun)
#SBATCH --cpus-per-task=64            # Sufficient CPUs (e.g. 4~8x the number of GPUs)

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/GasAgent-main/miniconda3/bin/activate llm

export PYTHONUNBUFFERED=1
# export NCCL_DEBUG=INFO
export MASTER_PORT=12345
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)

# echo "MASTER_ADDR=$MASTER_ADDR"
# echo "SLURM_NNODES=$SLURM_NNODES"
# echo "SLURM_NODEID=$SLURM_NODEID"

# Launch the script once per node
srun bash VLM-R1/src/open-r1-multimodal/run_scripts/sft_data_v1/stage3_generate_lora_run.sh