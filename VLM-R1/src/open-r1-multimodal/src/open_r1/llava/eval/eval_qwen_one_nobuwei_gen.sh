#!/bin/bash
#SBATCH --gpus=2
# export PYTHONPATH=/home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2:$PYTHONPATH
module load /home/dalhxwlyjsuo/criait_tansy/anaconda3
source activate /home/dalhxwlyjsuo/criait_tansy/anaconda3/envs/llava-next
#export CUDA_VISIBLE_DIVECES='0,1'
#python -m llava.eval.model_vqa_loader_mulimg \
python /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1/llava/eval/model_vqa_qwen_one.py \
    --model-path /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/weight/Llava-Qwen2-7B-pretrain-med-v0-mmtag-nobuwei \
    --model-name Llava-Qwen2-7B-pretrain-med-v0-mmtag-nobuwei \
    --question-file /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/data/eval_qwen_test_nobuwei_gen.json \
    --answers-file /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/data/answer/answer_pretrain_one_nobuwei_gen.json \
    #--conv-mode vicuna_v1