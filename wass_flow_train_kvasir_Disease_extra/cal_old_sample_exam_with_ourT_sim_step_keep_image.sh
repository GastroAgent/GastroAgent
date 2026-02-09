#!/bin/bash
#SBATCH -J CalGen
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1

source /mnt/inaisfs/data/apps/cuda12.4.sh 
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

python /mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/wass_flow_train_kvasir_Disease_extra/cal_old_sample_exam_with_ourT_sim_step_keep_image.py