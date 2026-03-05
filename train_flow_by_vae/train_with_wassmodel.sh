#!/bin/bash
#SBATCH -J tr_k1_r2
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:3

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

python /mnt/inaisfs/data/home/tansy_criait/GasAgent-main/train_flow_by_vae/train_with_wassmodel.py



