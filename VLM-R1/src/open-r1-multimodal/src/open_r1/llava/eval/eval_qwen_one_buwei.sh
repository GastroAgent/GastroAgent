#!/bin/bash
#SBATCH --gpus=2
# export PYTHONPATH=/home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2:$PYTHONPATH
#module load /home/dalhxwlyjsuo/criait_tansy/anaconda3
#source activate /home/dalhxwlyjsuo/criait_tansy/anaconda3/envs/llava-next

#export PYTHONPATH=/home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1:$PYTHONPATH
#module load anaconda/2024.10
#
#module load gcc/10.3
#module load cuda/12.4
#source activate /home/dalhxwlyjsuo/criait_tansy/anaconda3/envs/unsloth
# bash /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1/llava/eval/eval_qwen_one_buwei.sh
#export CUDA_VISIBLE_DIVECES='0,1'
#python -m llava.eval.model_vqa_loader_mulimg \
# sbatch -p vip_gpu_01 --gpus=1 /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1/llava/eval/eval_qwen_one_buwei.sh
python /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1/llava/eval/model_vqa_qwen_one.py \
    --model-path /home/dalhxwlyjsuo/criait_tansy/GRPO_weights/LlavaQwen2-GRPO-Tricks-Total-CoT-6000 \
    --model-name LlavaQwen2-GRPO-Tricks-Total-CoT-6000 \
    --question-file /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/data/eval_qwen_test_buwei.json \
    --answers-file /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/data/answer/answer_grpo_one_buwei.json \
    #--conv-mode vicuna_v1