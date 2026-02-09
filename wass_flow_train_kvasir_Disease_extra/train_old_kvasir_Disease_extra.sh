#!/bin/bash
#SBATCH -J kvasir_Disease_s2
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

# export CUDA_VISIBLE_DEVICES=7
python /mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/wass_flow_train_kvasir_Disease_extra/train_old_kvasir_Disease_extra.py

