#!/bin/bash
#SBATCH --gpus=2
# export PYTHONPATH=/home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2:$PYTHONPATH
module load /home/dalhxwlyjsuo/criait_tansy/anaconda3
source activate /home/dalhxwlyjsuo/criait_tansy/anaconda3/envs/llava-next
export CUDA_VISIBLE_DIVECES='0,5'
#python -m llava.eval.model_vqa_loader_mulimg \
python /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1/llava/eval/model_vqa_qwen_mul.py \
    --model-path /home/dalhxwlyjsuo/criait_tansy/GRPO_weights/LlavaQwen2-GRPO-Tricks-Total-CoT-6000 \
    --model-name LlavaQwen2-GRPO-Tricks-Total-CoT-6000 \
    --question-file /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/data/instruct_prompt_step_gen1.json \
    --answers-file /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/data/answer/answer-grpo-mul-prompt.json \
    #--conv-mode vicuna_v1