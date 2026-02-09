#!/bin/bash
#SBATCH -J shiguan_CalGen
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:2

source /mnt/inaisfs/data/apps/cuda12.4.sh 
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

python /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/wass_flow_train/cal_wass.py
