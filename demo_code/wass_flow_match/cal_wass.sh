#!/bin/bash
#SBATCH -J tsy_CalGen
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 3
#SBATCH --gres=gpu:1

#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/train_tsy/slurm_out/kvasir_doctor/eval_kvasir_extra_doctor_exam_50000_noextra5_4.out

source /mnt/inaisfs/data/apps/cuda12.4.sh 
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

python /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/eval_tsy/cal_old_sample_exam_with_ourT_sim_step_keep_image.py
