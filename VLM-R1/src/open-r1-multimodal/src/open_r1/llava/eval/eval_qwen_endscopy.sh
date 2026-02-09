#!/bin/bash
#SBATCH --gpus=2
#export PYTHONPATH=/home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2:$PYTHONPATH
#module load /home/dalhxwlyjsuo/criait_tansy/anaconda3
#source activate llava-next
export CUDA_VISIBLE_DEVICES=4
#python -m llava.eval.model_vqa_loader_mulimg \
python /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/llava/eval/model_vqa_qwen_endscopy.py \
    --model-path /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/weight/Llava-Qwen2-7B-sft-med-vitpretrain \
    --model-name Llava-Qwen2-7B-sft-med-vitpretrain \
    --question-file /home/dalhxwlyjsuo/criait_tansy/project/Multi-Modality-Arena-main/Multi-Modality-Arena-main/OmniMedVQA/OmniMedVQA/QA_information/Restricted-access/Kvasir1.json
    #--conv-mode vicuna_v1