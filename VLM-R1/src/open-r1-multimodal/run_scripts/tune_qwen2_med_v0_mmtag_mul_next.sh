#!/bin/bash
#SBATCH -N 2
#SBATCH --qos=gpugpu
#SBATCH --gres=gpu:8
#SBATCH -p vip_gpu_01
export PYTHONPATH=/home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1:$PYTHONPATH
module load anaconda/2024.10

module load gcc/10.3
module load cuda/12.4
source activate /home/dalhxwlyjsuo/criait_tansy/anaconda3/envs/unsloth

RUN_NAME="tune_qwen2_med_v0_mmtag_mul_next-Stage3-Full"

export NCCL_ALGO=Ring
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export WANDB_API_KEY="046193756f0009d71636370009dbcee76f3bd3ac"
export WANDB_MODE="offline"

# sbatch -p vip_gpu_01 --gpus=1 /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/run_scripts/tune_qwen2_med_v0_mmtag_mul_next.sh
# python -m torch.distributed.run --nnodes=1 --nproc_per_node=3 --master_port=20001 llava/train/train_mem.py \
# sbatch -p vip_gpu_01 -N 2 --gres=gpu:8 --ntasks-per-node=48 --qos=gpugpu /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/run_scripts/tune_qwen2_med_v0_mmtag_mul_next.sh
torchrun --nnodes=2 \
    --node_rank=0 \
    --nproc_per_node=8 \
    --master_addr="g0008" \
    --master_port="29501" \
    --max_restarts=3 \
    /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/llava/train/train_mem_mul_next.py \
    --model_name_or_path /home/dalhxwlyjsuo/criait_tansy/weight/Llava-Qwen2-7B-knowtune-med-v0-mmtag-mul-next-stage2-4 \
    --version qwen_2 \
    --data_path /home/dalhxwlyjsuo/criait_tansy/jmf/share_huaxi/stage3_total.json \
    --vision_tower /home/dalhxwlyjsuo/criait_tansy/project/llava-npu-gpu/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/weight/Llava-Qwen2-7B-knowtune-med-v0-mmtag-mul-next/mm_projector.bin \
    --lora_enable false \
    --lora_r 64 \
    --lora_alpha 16 \
    --tune_mm_mlp_adapter false \
    --freeze_mm_mlp_adapter false \
    --freeze_vision_module true \
    --freeze_backbone false \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /home/dalhxwlyjsuo/criait_tansy/weight/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-next-Stage3-Full \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --mm_patch_merge_type none \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --max_grad_norm 2.0 \
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
