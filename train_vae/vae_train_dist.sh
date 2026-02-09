#!/bin/bash
#SBATCH -J vit-gan
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --gres=gpu:4

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=2
export CUDA_VISIBLE_DEVICES=2,3,4,5
torchrun --nnodes=1 --nproc_per_node=4 --master_port=25011 /mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/vae_src/vae_train_dist_gan.py
