#!/bin/bash

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export CUDA_VISIBLE_DEVICES=3
# export CUDA_VISIBLE_DEVICES=2,3,4,5
# torchrun --nnodes=1 --nproc_per_node=8 --master_port=25010 /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/vae_train_dist_gan.py

NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_NODEID
MASTER_ADDR=${MASTER_ADDR}      # 来自父脚本 export
MASTER_PORT=${MASTER_PORT:-12345}

torchrun \
    --nproc_per_node=8 \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/vae_train_dist_gan.py
