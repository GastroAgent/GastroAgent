#!/bin/bash
#SBATCH -J shiguan_wass
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:3

#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/wass_flow_train/slurm_out/doctor_exam_step1_wass.out

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

# export CUDA_VISIBLE_DEVICES=7
python /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/wass_flow_train/train_new.py


