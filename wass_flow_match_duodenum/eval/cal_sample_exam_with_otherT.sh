#!/bin/bash
#SBATCH -J jmfT_otest
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:2

source /mnt/inaisfs/data/apps/cuda12.4.sh 
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

python /mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/cal_wass/cal_sample_exam_with_otherT.py

