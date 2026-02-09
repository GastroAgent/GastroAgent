#!/bin/bash
#SBATCH --gpus=1
module load anaconda/2024.10
module load gcc/10.3
module load cuda/12.4
source activate /home/dalhxwlyjsuo/criait_tansy/anaconda3/envs/unsloth
export PYTHONPATH=/home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1:$PYTHONPATH

#export CUDA_VISIBLE_DEVICES=4

# bash /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/lora_infer.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
# sbatch -p vip_gpu_01 --gpus=1 /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/lora_infer.sh
python /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/tsy/lora_infer_grpo_dia.py \
    --model-id /home/dalhxwlyjsuo/criait_tansy/GRPO_weights/LlavaQwen2-GRPO-Tricks-Med \
    --question-file /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/data/eval_qwen.json \
    --answers-file /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/data/answer/answer-grpo.json \