cd src/open-r1-multimodal

# Python 路径 是 grpo_jsonl.py 所在的路径？
export DEBUG_MODE="true"
# export CUDA_VISIBLE_DEVICES=4,5,6,7

RUN_NAME="Llava-Qwen2-GRPO"
export LOG_PATH="./debug_log_$RUN_NAME.txt"
    # --deepspeed local_scripts/zero3.json \
torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_jsonl.py \
    --deepspeed /home/work/VLM-R1/src/open-r1-multimodal/local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path /home/work/code/model_match/local_llavaqwen2 \
    --use_peft true \
    --reward_funcs accuracy format lang \
    --use_rslora true \
    --dataset_name none \
    --image_folders /home/work/Images \
    --data_file_paths /home/work/vqa_grpo_train.jsonl \
    --freeze_vision_modules true \
    --freeze_mm_mlp_adapter false \
    --resume_from_checkpoint none \
    --max_prompt_length 1024 \
    --max_completion_length 16 \
    --num_generations 16 \
    --use_llava_v1_conv true \
    --use_mix_prompts true \
    --max_grad_norm 1.0 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --alg grpo-unbias2 \
    --grpo_denorm_length 8192 \
    --lora_target_modules q_proj v_proj \
    --epsilon_high 0.2 \
    --prune_threshold 0.25 \
    --prune_ratio 0.1875 \
    --epsilon 0.1 \
    --num_iterations 4 \
    --is_message true \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to none \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 1000 \
    --save_only_model true