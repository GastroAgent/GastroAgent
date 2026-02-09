#!/bin/bash
#SBATCH -J DmathQwen
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:8

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate llm
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1:$PYTHONPATH
export DEBUG_MODE="true"

export WANDB_API_KEY="046193756f0009d71636370009dbcee76f3bd3ac"
export WANDB_MODE="offline"

### vllm 不兼容下列设置
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME="Qwen2.5-7B-Formated-Off-MetaMathQA-DeUnbiasGRPO-4096"
# /mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/MATH/formated_off_7B_train_correct.json
# /mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/GSM8K/formated_off_7B_train_correct.json
# /mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/MetaMathQA/formated_off_7B_train_correct.json
OUTPUT_DIR="/mnt/inaisfs/data/home/tansy_criait/VLM-R1/checkpoints"
export LOG_PATH="/mnt/inaisfs/data/home/tansy_criait/VLM-R1/logs/debug_log_$RUN_NAME.txt"
# grpo-unbias 0.28 0.2 0.012
torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12335" \
    /mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py \
    --output_dir $OUTPUT_DIR/$RUN_NAME \
    --model_name_or_path /mnt/inaisfs/data/home/tansy_criait/weights/Med-Llava-Qwen2-7B-GSPO \
    --use_peft true \
    --reward_funcs acc format lang \
    --use_rslora true \
    --use_vllm false \
    --vllm_gpu_memory_utilization 0.3 \
    --dataset_name none \
    --image_folders /home/lab/data/OmniMedVQA/Images \
    --data_file_paths /mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/RL/RL_train_without_ref_with_train.json \
    --resume_from_checkpoint none \
    --freeze_vision_modules false \
    --freeze_mm_mlp_adapter false \
    --deepspeed /mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/local_scripts/zero3.json \
    --max_prompt_length 512 \
    --max_completion_length 2048 \
    --num_generations 12 \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 1 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --lora_r 64 \
    --lora_alpha 32 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj \
    --use_llava_v1_conv false \
    --use_mix_prompts false \
    --epsilon_high 0.28 \
    --epsilon 0.20 \
    --beta 0.02 \
    --decouple_adv false \
    --alg grpo-unbias2 \
    --grpo_denorm_length 4096 \
    --bi_kl false \
    --learning_rate 1e-5 \
    --prune_threshold 0.0 \
    --prune_ratio 0.25 \
    --advantages_clip_up 5.0 \
    --advantages_clip_down 2.5 \
    --use_advantages_clip true \
    --num_iterations 4 \
    --bf16 \
    --is_message true \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 500 \
    --save_only_model true

