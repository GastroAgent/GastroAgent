#!/bin/bash
#SBATCH --gpus=1

#export PYTHONPATH=/home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1:$PYTHONPATH
#module load anaconda/2024.10
#
#module load gcc/10.3
#module load cuda/12.4
#source activate unsloth
# sbatch -p vip_gpu_01 --gpus=1 /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/run_scripts/vllm_serve.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

hostname -I

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
 --model /home/dalhxwlyjsuo/criait_tansy/weight/Qwen3-8B \
 --served-model-name Qwen3 \
 --max-model-len=8192 \
 --trust-remote-code \
 --gpu_memory_utilization=0.9 \
 --tensor_parallel_size=1
