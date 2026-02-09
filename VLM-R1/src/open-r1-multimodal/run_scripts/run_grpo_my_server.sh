#!/bin/bash
#SBATCH --gpus=4
module load anaconda/2024.10
module load gcc/10.3
module load cuda/12.4

source activate unsloth

#source activate /home/dalhxwlyjsuo/criait_tansy/.conda/envs/unsloth
# sbatch -p vip_gpu_01 --gpus=4 /home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/VLM-R1/src/open-r1-multimodal/run_scripts/run_grpo_my_server.sh

export PYTHONPATH=/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/VLM-R1/src/open-r1-multimodal/src/open_r1:$PYTHONPATH
export DEBUG_MODE="true"
#export CUDA_VISIBLE_DEVICES=2,5,6,7
export WANDB_API_KEY="046193756f0009d71636370009dbcee76f3bd3ac"
export WANDB_MODE="offline"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800
RUN_NAME="Med-Llava-Qwen2-7B-GSPO_Decouple"

OUTPUT_DIR="/home/dalhxwlyjsuo/criait_tansy/checkpoints/GRPO_checkpoints"
export LOG_PATH="/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/VLM-R1/logs/debug_log_$RUN_NAME.txt"
# --lora_target_modules qkv gate_proj Qwen-VL 最好不使用 gradient_checkpointing ，否则保报错。
# acc_length == accuracy + length  Qwen/Qwen2.5-VL-3B-Instruct

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12364" \
    /home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/VLM-R1/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py \
    --output_dir $OUTPUT_DIR/$RUN_NAME \
    --model_name_or_path /home/dalhxwlyjsuo/criait_tansy/weights/Med-Llava-Qwen2-7B-SFT \
    --use_peft true \
    --reward_funcs acc_length format lang \
    --use_rslora true \
    --use_vllm false \
    --dataset_name none \
    --image_folders /home/lab/data/OmniMedVQA/Images \
    --data_file_paths /home/dalhxwlyjsuo/criait_tansy/Node_Code/126/kavi/RL_train_without_ref_with_train.json \
    --resume_from_checkpoint none \
    --freeze_vision_modules true \
    --freeze_mm_mlp_adapter false \
    --deepspeed /home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/VLM-R1/src/open-r1-multimodal/local_scripts/zero3.json \
    --max_prompt_length 8192 \
    --max_completion_length 512 \
    --num_generations 16 \
    --per_device_train_batch_size 16 \
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
    --beta 0.0 \
    --decouple_adv true \
    --alg gspo \
    --grpo_denorm_length 2048 \
    --bi_kl false \
    --learning_rate 1e-5 \
    --prune_threshold 0.0 \
    --prune_ratio 0.25 \
    --advantages_clip_up 5.0 \
    --advantages_clip_down 0.75 \
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
    --save_steps 5000 \
    --save_only_model true


