#!/bin/bash
#cd src/open-r1-multimodal
#SBATCH --gpus=8
export PYTHONPATH=/home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1:$PYTHONPATH
module load anaconda/2024.10

module load gcc/10.3
module load cuda/12.4
source activate /home/dalhxwlyjsuo/criait_tansy/anaconda3/envs/unsloth

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

export CUDA_VISIBLE_DEVICES=0

RUN_NAME="ft_qwen2_med_v0_mmtag_mul_next-Stage2-2"

export WANDB_API_KEY="046193756f0009d71636370009dbcee76f3bd3ac"
export WANDB_MODE="offline"

# /home/dalhxwlyjsuo/criait_tansy/project/LLaVA-npu-latest/LLaVA-npu/extend/knowtune/stage2_1.json
# sbatch -p vip_gpu_01 --gpus=1  /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/run_scripts/ft_qwen2_med_v0_mmtag_mul_next2.sh
# sbatch -p vip_gpu_01 --gpus=4  /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/run_scripts/ft_qwen2_med_v0_mmtag_mul_next2.sh
# python -m torch.distributed.run --nnodes=1 --nproc_per_node=3 --master_port=20001 llava/train/train_mem.py \
torchrun --nnodes=1 --nproc_per_node=4 --master_port=25001 \
    /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1/llava/train/train_qwen_mul_next.py \
    --model_name_or_path /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/weight/Llava-Qwen2-7B-pretrain-med-v0-mmtag \
    --version v0_mmtag \
    --data_path  /home/dalhxwlyjsuo/criait_tansy/jmf/data/stage2_2.json \
    --vision_tower /home/dalhxwlyjsuo/criait_tansy/project/llava-npu-gpu/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/weight/Llava-Qwen2-7B-pretrain-med-v0-mmtag/mm_projector.bin \
    --tune_mm_mlp_adapter false \
    --freeze_mm_mlp_adapter false \
    --freeze_vision_module false \
    --freeze_backbone true \
    --mm_projector_type mlp2x_gelu \
    --mm_patch_merge_type none \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /home/dalhxwlyjsuo/criait_tansy/weight/Llava-Qwen2-7B-knowtune-med-v0-mmtag-mul-next-stage2-2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --max_grad_norm 2.0 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $RUN_NAME \
    --deepspeed /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/local_scripts/deepspeed_zero3.json