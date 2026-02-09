#!/bin/bash
#SBATCH --gpus=1
#export PYTHONPATH=/home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1:$PYTHONPATH
#module load /home/dalhxwlyjsuo/criait_tansy/anaconda3
#source activate llava-next
#export CUDA_VISIBLE_DEVICES=4

# bash /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1/llava/eval/eval_qwen_endscopy_whit_pred.sh
#python -m llava.eval.model_vqa_loader_mulimg \
python /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1/llava/eval/model_vqa_qwen_endscopy_with_pred.py \
    --model-name LlavaQwen2-GRPO-Tricks-Med \
    --question-file /home/dalhxwlyjsuo/criait_tansy/jmf/ds-vl-on-policy-2/Kvasir-en.json \
    --answers_base_path /home/dalhxwlyjsuo/criait_tansy/jmf/GRPO_result/LlavaQwen2-GRPO-Med/Kvasir-en-False.json
    #--conv-mode vicuna_v1