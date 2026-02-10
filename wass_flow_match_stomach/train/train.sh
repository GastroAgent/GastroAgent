#!/bin/bash
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/train_tsy/slurm_out/step0_base_support.out

#SBATCH -J tsy_flow_s1
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --gres=gpu:3

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

python /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/train_tsy/train.py

