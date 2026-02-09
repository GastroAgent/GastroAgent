#!/bin/bash
#SBATCH --gpus=1
# export PYTHONPATH=/home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2:$PYTHONPATH
module load /home/dalhxwlyjsuo/criait_tansy/anaconda3
source activate /home/dalhxwlyjsuo/criait_tansy/anaconda3/envs/llava-next
#export CUDA_VISIBLE_DIVECES='0,1,5'
#python -m llava.eval.model_vqa_loader_mulimg \
python /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1/llava/eval/model_vqa_qwen_mul_promt_rag.py \
    --model-path /home/dalhxwlyjsuo/criait_tansy/GRPO_weights/LlavaQwen2-GRPO-Tricks-Total-CoT-6000 \
    --model-name LlavaQwen2-GRPO-Tricks-Total-CoT-6000 \
    --question-file /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/data/instrcut_dia_mul_test.json \
    --answers-file /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/data/answer/answer-grpo-mul-prompt-rag.json \
    #--conv-mode vicuna_v1