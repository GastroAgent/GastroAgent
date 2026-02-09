#!/bin/bash
#SBATCH --gpus=16
export PYTHONPATH=/home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2:$PYTHONPATH
module load /home/dalhxwlyjsuo/criait_tansy/anaconda3
source activate llava-next

export NCCL_ALGO=Ring
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0

# python -m torch.distributed.run --nnodes=1 --nproc_per_node=3 --master_port=20001 llava/train/train_mem.py \
python -m torch.distributed.launch \
    --nnodes=2 \
    --node_rank=0 \
    --nproc_per_node=8 \
    --master_addr="g0004" \
    --master_port="29501" \
    /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/llava/train/train_mem.py \
    --model_name_or_path /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/Qwen2.5-32B-Instruct \
    --lora_enable True \
    --bits 8 \
    --target_modules "q_proj, k_proj, v_proj, o_proj, gate_proj" \
    --version plain \
    --data_path /home/dalhxwlyjsuo/criait_tansy/project/LLaVA-npu-latest/LLaVA-npu/extend/pretrain/extend_pretrain_ciai_2004_2010_3.json \
    --vision_tower  /home/dalhxwlyjsuo/criait_tansy/project/llava-npu-gpu/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/weight/Llava-Qwen2-32B-pretrain-med-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none 
