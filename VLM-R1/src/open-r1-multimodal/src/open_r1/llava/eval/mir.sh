#!/bin/bash
#SBATCH --gpus=2
export PYTHONPATH=/home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2:$PYTHONPATH
module load /home/dalhxwlyjsuo/criait_tansy/anaconda3
source activate tsy

python mir.py \
  --model_path       /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/weight/Llava-Qwen2-7B-tune-med-mul \
  --base_llm         /home/dalhxwlyjsuo/criait_tansy/project/LLaVA-1/LLaVA-Med-1.0.0/Qwen2-7B-Instruct \
  --text_data_path   /home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/data/eval \
  --eval_num         100 \
  --mode             fast
