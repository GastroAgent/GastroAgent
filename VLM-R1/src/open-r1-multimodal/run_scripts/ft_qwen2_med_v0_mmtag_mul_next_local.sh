#!/bin/bash
cd src/open-r1-multimodal
#SBATCH --gpus=2
#  export PYTHONPATH=/home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1:$PYTHONPATH
#  module load /home/dalhxwlyjsuo/criait_tansy/anaconda3
#  source activate /home/dalhxwlyjsuo/criait_tansy/anaconda3/envs/llava-next

# python -m torch.distributed.run --nnodes=1 --nproc_per_node=3 --master_port=20001 llava/train/train_mem.py \
torchrun --nnodes=1 --nproc_per_node=1 --master_port=25001 \
    /home/work/VLM-R1/src/open-r1-multimodal/src/open_r1/llava/train/train_qwen_mul_next.py \
    --model_name_or_path /home/work/code/model_match/local_llavaqwen2 \
    --version v0_mmtag \
    --data_path  /home/work/data/stage2_2.json \
    --vision_tower /home/work/code/model_match/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /home/work/code/model_match/mm_projector/checkpoint.bin \
    --lora_enable true \
    --tune_mm_mlp_adapter false \
    --freeze_mm_mlp_adapter false \
    --freeze_vision_module true \
    --freeze_backbone false \
    --mm_projector_type mlp2x_gelu \
    --mm_patch_merge_type none \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /home/work/weight/Llava-Qwen2-7B-knowtune-med-v0-mmtag-mul-next \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4\
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to none
    # --deepspeed /home/work/VLM-R1/src/open-r1-multimodal/local_scripts/zero3.json
