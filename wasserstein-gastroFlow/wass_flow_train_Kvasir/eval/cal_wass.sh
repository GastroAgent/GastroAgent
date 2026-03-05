#!/bin/bash
#SBATCH -J cal_wass
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

# export CUDA_VISIBLE_DEVICES=7
python /mnt/inaisfs/data/home/tansy_criait/GasAgent-main/wasserstein-gastroFlow/wass_flow_train_Kvasir/eval/cal_wass.py
