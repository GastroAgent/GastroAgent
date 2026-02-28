#!/bin/bash
#SBATCH -J tsy_rl
#SBATCH -p gpu
#SBATCH -N 1                          # 节点个数
#SBATCH --ntasks-per-node=1           # ← 关键！每个节点只跑 1 个主进程
#SBATCH --gres=gpu:8                  # 每个节点 8 张 GPU（供 torchrun 使用）
#SBATCH --cpus-per-task=32            # 给足 CPU（如 4~8 倍 GPU 数）

#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/run_scripts/tsy/rl/rl.out

export PYTHONPATH=/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1:$PYTHONPATH
export WANDB_API_KEY="046193756f0009d71636370009dbcee76f3bd3ac"
export WANDB_MODE="offline"
# 项目名称
RUN_NAME="Llava-Qwen2-7B-tune-med-RL"
# 保存 checkpoint 路径
OUTPUT_DIR="/mnt/inaisfs/data/home/tansy_criait/VLM-R1/checkpoints/tsy"
### torchrun 参数
# 模型权重
    --model_name_or_path /mnt/inaisfs/data/home/tansy_criait/weights/tsy/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-all \
# 奖励函数
    --reward_funcs acc_nobox_nooption new_format vllm_reward \
# 数据路径
    --data_file_paths /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/data_tsy/mllm/all_disease/all_medical_dialogue_dataset.json \
# 冻结相关层
    --freeze_vision_modules false \
    --freeze_mm_mlp_adapter false \
# 输入上下文大小
    --max_prompt_length 1024 \
# 输出上下文大小
    --max_completion_length 1024 \
# Rollout 大小, 确保 num_generations == per_device_train_batch_size 
    --num_generations 12 \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 1 \
# 最大梯度
    --max_grad_norm 1.0 \
# GRPO 的经典参数，不了解就不动。
    --epsilon_high 0.28 \
    --epsilon 0.20 \
    --beta 0.012 \
    --alg grpo-unbias \ # 还用 'grpo' 'grpo-unbias2' 
# 裁剪比例、若爆显存，除了调小 max_completion_length 外，加大裁剪比例。num_generations 已优化为 拿时间换空间 了。
    --prune_ratio 0.0 \
# 每次rollout后的训练次数，不懂不动。
    --num_iterations 4 \
# 设置wandb
    --report_to wandb \
# 训练轮数，因为 没有设置 教师Ref 和 Eval，所以建议小一点，自己额外评价后再次提交训练脚本。
    --num_train_epochs 2 \
# 保存间隔
    --save_steps 500 \
