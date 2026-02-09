#!/bin/bash
#SBATCH --gpus=2
# export PYTHONPATH=/home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2:$PYTHONPATH
module load /home/dalhxwlyjsuo/criait_tansy/anaconda3
source activate /home/dalhxwlyjsuo/criait_tansy/anaconda3/envs/llava-next
#export CUDA_VISIBLE_DIVECES='1,4'
#python -m llava.eval.model_vqa_loader_mulimg \
python /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1/llava/eval/model_vqa_qwen_mul.py \
    --model-path /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/weight/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-next \
    --model-name Llava-Qwen2-7B-tune-med-v0-mmtag-mul-next \
    --question-file /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/data/instrcut_dia_mulimg_format2.json \
    --answers-file /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/data/answer/answer-tune-med-v0-mmtag-mul-next111111.json \
    #--conv-mode vicuna_v1