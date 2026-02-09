#!/bin/bash
#cd src/open-r1-multimodal
#SBATCH --gpus=4
export PYTHONPATH=/home/dalhxwlyjsuo/criait_tansy/project/EndoViT:$PYTHONPATH
module load anaconda/2024.10
# sbatch -p vip_gpu_01 --gpus=2 /home/dalhxwlyjsuo/criait_tansy/project/EndoViT/src/vqae_train_dist.sh
module load gcc/10.3
module load cuda/12.4
source activate unsloth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#export CUDA_LAUNCH_BLOCKING=1

torchrun --nnodes=1 --nproc_per_node=2 --master_port=25001 /home/dalhxwlyjsuo/criait_tansy/project/EndoViT/src/vqae_train_dist.py
