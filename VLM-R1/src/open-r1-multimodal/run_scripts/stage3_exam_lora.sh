#!/bin/bash
#SBATCH -J SFT
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:8

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate llm

export PYTHONPATH=/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1:$PYTHONPATH

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export CUDA_LAUNCH_BLOCKING=1
RUN_NAME="Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-2048"

export WANDB_API_KEY="046193756f0009d71636370009dbcee76f3bd3ac"
export WANDB_MODE="offline"


torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    /mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/llava/train/train_qwen_mul_next.py \
    --model_name_or_path /mnt/inaisfs/data/home/tansy_criait/weights/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-75 \
    --version qwen2_vl \
    --data_path /mnt/inaisfs/data/home/tansy_criait/Open_source_med_domain/llava-med-zh-instruct-60k.json \
    --vision_tower /mnt/inaisfs/data/home/tansy_criait/weights/my-clip-vision \
    --pretrain_mm_mlp_adapter /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/weight/Llava-Qwen2-7B-knowtune-med-v0-mmtag-mul-next/mm_projector.bin \
    --lora_enable true \
    --lora_r 32 \
    --lora_alpha 16 \
    --tune_mm_mlp_adapter false \
    --freeze_mm_mlp_adapter false \
    --freeze_vision_module false \
    --freeze_backbone false \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /mnt/inaisfs/data/home/tansy_criait/VLM-R1/checkpoints/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-512 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --mm_patch_merge_type none \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 1e-3 \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.5 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --loss_denorm 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $RUN_NAME \
    --deepspeed /mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/local_scripts/deepspeed_zero3.json