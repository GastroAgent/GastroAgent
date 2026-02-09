#!/bin/bash
#SBATCH -J shiguan_workflow_test
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 3
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/VLM-R1/results/slurm_out/shiguan_infer-claude-optim.out

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate llm

# export CUDA_VISIBLE_DEVICES=7

python /mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/test_med_llava_workflow_shiguan_claude.py \
    --mode optimize \
    --validation_data /mnt/inaisfs/data/home/tansy_criait/VLM-R1/results/doctor_食管_workflow_claude/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-75/食管_optimized.json
