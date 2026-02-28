#!/bin/bash
# 不再接收 $1 $2 $3... 参数

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate llm

export PYTHONPATH=/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1:$PYTHONPATH
export DEBUG_MODE="true"

export WANDB_API_KEY="046193756f0009d71636370009dbcee76f3bd3ac"
export WANDB_MODE="offline"
RUN_NAME="Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-all-RL-2500-option-add-cot-sft"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export CUDA_VISIBLE_DEVICES=7
# 从 SLURM 获取
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_NODEID
MASTER_ADDR=${MASTER_ADDR}  
MASTER_PORT=${MASTER_PORT:-12345}

echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

torchrun \
    --nproc_per_node=8 \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/llava/train/train_qwen_mul_next.py \
    --model_name_or_path /mnt/inaisfs/data/home/tansy_criait/weights/tsy/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-all-RL-2500 \
    --version qwen2_vl \
    --data_path /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/data_tsy/mllm/all_disease_vqa/option_add/all_cot_think_samples.json \
    --vision_tower /mnt/inaisfs/data/home/tansy_criait/weights/my-clip-vision \
    --pretrain_mm_mlp_adapter /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/weight/Llava-Qwen2-7B-knowtune-med-v0-mmtag-mul-next/mm_projector.bin \
    --lora_enable true \
    --lora_r 64 \
    --lora_alpha 32 \
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
    --output_dir /mnt/inaisfs/data/home/tansy_criait/VLM-R1/checkpoints/tsy/$RUN_NAME \
    --num_train_epochs 5 \
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