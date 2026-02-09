cd src/open-r1-multimodal

export DEBUG_MODE="true"
# export CUDA_VISIBLE_DEVICES=4,5,6,7

RUN_NAME="Qwen2.5-VL-3B-GRPO-GUI_multi-image"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_jsonl.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path /mnt/d/pythonProject2024spring/work/code/model_match/Qwen2.5-VL-7B-Instruct \
    --use_peft true \
    --dataset_name none \
    --image_folders /mnt/d/pythonProject2024spring/work/Images/ACRIMA \
    --data_file_paths /mnt/d/pythonProject2024spring/work/ds-vl-on-policy/ACRIMA.json \
    --freeze_vision_modules true \
    --max_prompt_length 256 \
    --num_generations 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --lora_target_modules "q_proj, v_proj" \
    --epsilon_high 0.3 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true