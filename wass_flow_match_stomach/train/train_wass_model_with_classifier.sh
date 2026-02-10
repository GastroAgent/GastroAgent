#!/bin/bash
#SBATCH -J tsy_stomach_tsy_wass
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:3
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/train_tsy/slurm_out/doctor_exam/batch8/loss/23.22/step2_wass_epoch25.out

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

# export CUDA_VISIBLE_DEVICES=7
python /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/train_tsy/train_wass_model_with_classifier.py

