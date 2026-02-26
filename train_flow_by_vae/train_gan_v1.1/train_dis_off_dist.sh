#!/bin/bash
#SBATCH -J off_dis
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 48
#SBATCH --gres=gpu:4

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

torchrun --nproc_per_node=4 --master_port=29501 /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v1/train_dis_off_dist.py
