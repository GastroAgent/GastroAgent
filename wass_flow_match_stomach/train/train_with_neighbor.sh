#!/bin/bash
#SBATCH -J stomach_flow_s3
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:3

#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/train_tsy/slurm_out/step4_data_doctor_exam.out

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

python /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/train_tsy/train_with_neighbor.py