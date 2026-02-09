#!/bin/bash
#SBATCH --gpus=1
module load anaconda/2024.10
module load gcc/10.3
module load cuda/12.4
source activate /home/dalhxwlyjsuo/criait_tansy/anaconda3/envs/unsloth

#export CUDA_VISIBLE_DEVICES=0

# bash /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/lora_infers/lora_infer2.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
# sbatch -p vip_gpu_01 --gpus=1 /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/lora_infer.sh
python /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/lora_infers/lora_infer2.py
