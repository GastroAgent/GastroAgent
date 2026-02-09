#!/bin/bash
#SBATCH --gpus=2
# export PYTHONPATH=/home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2:$PYTHONPATH
#module load /home/dalhxwlyjsuo/criait_tansy/anaconda3
#source activate /home/dalhxwlyjsuo/criait_tansy/anaconda3/envs/llava-next

export PYTHONPATH=/home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1:$PYTHONPATH
module load anaconda/2024.10

module load gcc/10.3
module load cuda/12.4
source activate /home/dalhxwlyjsuo/criait_tansy/anaconda3/envs/unsloth

# sbatch -p vip_gpu_01 --gpus=1 /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1/llava/eval/eval_qwen_one.sh

#export CUDA_VISIBLE_DIVECES=''

#python -m llava.eval.model_vqa_loader_mulimg \
python /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1/llava/eval/model_vqa_qwen_one.py \
    --model-path /home/dalhxwlyjsuo/criait_tansy/weight/Llava-Qwen2-7B-knowtune-med-v0-mmtag-mul-next-stage2-4 \
    --model-name Llava-Qwen2-7B-knowtune-med-v0-mmtag-mul-next-stage2-4 \
    --question-file /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/data/eval_qwen.json \
    --answers-file /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/data/answer/answer_stage2-4_one_buwei.json \
    #--conv-mode vicuna_v1