#!/bin/bash
#cd src/open-r1-multimodal
#SBATCH --gpus=4
export PYTHONPATH=/home/dalhxwlyjsuo/criait_tansy/project/EndoViT:$PYTHONPATH
module load anaconda/2024.10
# sbatch -p vip_gpu_01 --gpus=3 /home/dalhxwlyjsuo/criait_tansy/project/EndoViT/src/train_clip.sh
module load gcc/10.3
module load cuda/12.4
source activate unsloth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#export CUDA_LAUNCH_BLOCKING=1

python /mnt/inaisfs/data/home/tansy_criait/GasAgent-main/train_clip/train_clip.py
