#!/bin/bash
#SBATCH -J extract
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --gres=gpu:1

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

python /mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/llm_extract_answer.py