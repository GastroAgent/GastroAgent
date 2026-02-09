#!/bin/bash
#SBATCH --gpus=4
module load anaconda/2024.10
module load gcc/10.3
module load cuda/12.4

source activate unsloth

export PYTHONPATH=/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/VLM-R1/src/open-r1-multimodal/src/open_r1:$PYTHONPATH

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#export CUDA_LAUNCH_BLOCKING=1
# sbatch -p vip_gpu_01 --gpus=4 /home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/lora_infer_dist.sh
torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12365" \
    /home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/lora_infer_dist.py
