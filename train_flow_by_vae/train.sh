#!/bin/bash
#SBATCH -J flow_s1
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --gres=gpu:3

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

python /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_十二指肠/train/train.py

