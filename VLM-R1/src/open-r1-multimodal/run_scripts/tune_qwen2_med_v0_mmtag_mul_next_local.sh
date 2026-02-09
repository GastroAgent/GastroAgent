#!/bin/bash
cd src/open-r1-multimodal
#SBATCH --gpus=2
#  export PYTHONPATH=/home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1:$PYTHONPATH
#  module load /home/dalhxwlyjsuo/criait_tansy/anaconda3
#  source activate /home/dalhxwlyjsuo/criait_tansy/anaconda3/envs/llava-next

# python -m torch.distributed.run --nnodes=1 --nproc_per_node=3 --master_port=20001 llava/train/train_mem.py \
torchrun --nnodes=1 --nproc_per_node=1 --master_port=25001 \
     /home/work/VLM-R1/src/open-r1-multimodal/src/open_r1/llava/train/train_qwen_mul_next.py  \
    --model_name_or_path /home/work/code/model_match/local_llavaqwen2 \
    --version direct \
    --data_path /home/work/vqa_train.jsonl \
    --vision_tower /home/work/code/model_match/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /home/work/code/model_match/mm_projector/checkpoint.bin \
    --tune_mm_mlp_adapter true \
    --freeze_mm_mlp_adapter false \
    --freeze_vision_module true \
    --freeze_backbone true \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /home/work/weight/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-next \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --mm_patch_merge_type none \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 1e-3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --loss_denorm 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to none \
    --deepspeed /home/work/VLM-R1/src/open-r1-multimodal/local_scripts/deepspeed_zero3.json \
