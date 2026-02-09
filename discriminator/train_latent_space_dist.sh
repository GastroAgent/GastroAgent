#!/bin/bash
#SBATCH -J disc_vae
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --gres=gpu:8

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

# export CUDA_VISIBLE_DEVICES=4
torchrun --nproc_per_node=8 /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/discriminator/train_latent_space_dist.py