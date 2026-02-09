#!/bin/bash
#SBATCH --gpus=4
export PYTHONPATH=/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/VLM-R1/src/open-r1-multimodal/src/open_r1:$PYTHONPATH
module load anaconda/2024.10

module load gcc/10.3
module load cuda/12.4
#source activate /home/dalhxwlyjsuo/criait_tansy/anaconda3/envs/unsloth
source activate unsloth

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN_NAME="Llava-Qwen2-7B-tune-med"

export WANDB_API_KEY="046193756f0009d71636370009dbcee76f3bd3ac"
export WANDB_MODE="offline"
#export CUDA_VISIBLE_DEVICES=3,4,6,7

# sbatch -p vip_gpu_01 --gpus=4 /home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/VLM-R1/src/open-r1-multimodal/run_scripts/stage3_generate_lora_nodes.sh

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12389" \
    /home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/VLM-R1/src/open-r1-multimodal/src/open_r1/llava/train/train_qwen_mul_next.py \
    --model_name_or_path /home/dalhxwlyjsuo/criait_tansy/weights/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-75 \
    --version qwen2_exam \
    --data_path /home/dalhxwlyjsuo/criait_tansy/Node_Code/126/kavi/RL_train_SFT.json \
    --vision_tower /home/dalhxwlyjsuo/criait_tansy/weights/my-clip-vision \
    --pretrain_mm_mlp_adapter /home/dalhxwlyjsuo/criait_tansy/weights/mm_project-7B.bin \
    --lora_enable true \
    --vision_lora_enable false \
    --lora_r 16 \
    --lora_alpha 8 \
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
    --output_dir /home/dalhxwlyjsuo/criait_tansy/checkpoints/Llava-Qwen2-7B-tune-med \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --mm_patch_merge_type none \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 1e-3 \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 16384 \
    --loss_denorm -1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $RUN_NAME \
    --deepspeed /home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/VLM-R1/src/open-r1-multimodal/local_scripts/deepspeed_zero3.json