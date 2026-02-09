#!/bin/bash
# 不再接收 $1 $2 $3... 参数

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate llm

export PYTHONPATH=/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1:$PYTHONPATH
# export DEBUG_MODE="true"
export WANDB_API_KEY="046193756f0009d71636370009dbcee76f3bd3ac"
export WANDB_MODE="offline"

RUN_NAME="Llava-Qwen2-7B-tune-med-SFT-RL-New2"
OUTPUT_DIR="/mnt/inaisfs/data/home/tansy_criait/VLM-R1/checkpoints"
export LOG_PATH="/mnt/inaisfs/data/home/tansy_criait/VLM-R1/logs/debug_log_$RUN_NAME.txt"

# 从 SLURM 获取
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_NODEID
MASTER_ADDR=${MASTER_ADDR}      # 来自父脚本 export
MASTER_PORT=${MASTER_PORT:-12345}
## /mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/RL_7B_train_correct.json
## /mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/GSM8K/formated_off_7B_train_correct.json

echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
# 若使用 vllm_reward，先启动 奖励模型 并检查对应的 openai_api_base 设置。

# export CUDA_VISIBLE_DEVICES=5
# NNODES=1
# NODE_RANK=0
# MASTER_ADDR="127.0.0.1"
# MASTER_PORT=12345

torchrun \
    --nproc_per_node=8 \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py \
    --output_dir $OUTPUT_DIR/$RUN_NAME \
    --model_name_or_path /mnt/inaisfs/data/home/tansy_criait/weights/Llava-Qwen2-7B-tune-med-SFT-RL-New \
    --use_peft true \
    --reward_funcs acc_nobox_nooption over_length_reward \
    --use_rslora true \
    --use_vllm false \
    --vllm_gpu_memory_utilization 0.3 \
    --dataset_name none \
    --image_folders /home/lab/data/OmniMedVQA/Images \
    --data_file_paths /mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/SFT/all_cot_think_samples.json \
    --resume_from_checkpoint none \
    --freeze_vision_modules false \
    --freeze_mm_mlp_adapter false \
    --deepspeed /mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --max_completion_length 1024 \
    --num_generations 16 \
    --per_device_train_batch_size 16 \
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
    --beta 0.012 \
    --decouple_adv false \
    --alg grpo-unbias \
    --grpo_denorm_length 4096 \
    --bi_kl false \
    --learning_rate 1e-5 \
    --prune_threshold 0.0 \
    --prune_ratio 0.0 \
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
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 500 \
    --save_only_model true