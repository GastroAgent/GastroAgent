#!/bin/bash
#SBATCH -J noref_cot
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:4

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate llm

export PYTHONPATH=/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1:$PYTHONPATH
export DEBUG_MODE="true"

export WANDB_API_KEY="046193756f0009d71636370009dbcee76f3bd3ac"
export WANDB_MODE="offline"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME="Med-Llava-Qwen2-7B-GRPOSample_NoRef_CoT"

OUTPUT_DIR="/mnt/inaisfs/data/home/tansy_criait/VLM-R1/checkpoints"
export LOG_PATH="/mnt/inaisfs/data/home/tansy_criait/VLM-R1/logs/debug_log_$RUN_NAME.txt"
# acc 
torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12381" \
    /mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py \
    --output_dir $OUTPUT_DIR/$RUN_NAME \
    --model_name_or_path /mnt/inaisfs/data/home/tansy_criait/weights/Med-Llava-Qwen2-7B-SFT \
    --use_peft true \
    --reward_funcs acc format lang \
    --use_rslora true \
    --use_vllm false \
    --dataset_name none \
    --image_folders /home/lab/data/OmniMedVQA/Images \
    --data_file_paths /mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/RL/RL_train_without_ref_with_train2.json \
    --resume_from_checkpoint none \
    --freeze_vision_modules true \
    --freeze_mm_mlp_adapter true \
    --deepspeed /mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/local_scripts/zero3.json \
    --max_prompt_length 8192 \
    --max_completion_length 1024 \
    --num_generations 24 \
    --per_device_train_batch_size 24 \
    --gradient_accumulation_steps 1 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --lora_r 64 \
    --lora_alpha 32 \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --use_llava_v1_conv false \
    --use_mix_prompts false \
    --epsilon_high 0.2 \
    --epsilon 0.1 \
    --beta 0.04 \
    --decouple_adv false \
    --alg grpo-unbias \
    --grpo_denorm_length 2048 \
    --bi_kl false \
    --learning_rate 1e-5 \
    --prune_threshold 0.0 \
    --prune_ratio 0.25 \
    --advantages_clip_up 4.0 \
    --advantages_clip_down 3.0 \
    --use_advantages_clip true \
    --num_iterations 4 \
    --bf16 \
    --is_message true \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 2000 \
    --save_only_model true


