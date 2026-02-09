import argparse
from copy import copy
import functools
import gc
import json
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from tqdm import tqdm
sys.path.append('/mnt/inaisfs/data/home/tansy_criait/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1')
sys.path.append('/mnt/inaisfs/data/home/tansy_criait/jmf/VLM-R1/data/dataset')
from reward_func import match_answer
from math_utils import parse_answer
from mathdataset import get_gsm8k_questions

from functools import partial
import torch
from vllm.lora.request import LoRARequest
from vllm import LLM, SamplingParams
from peft import PeftModel, PeftConfig, PeftModelForCausalLM
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer

base_model_name = '/mnt/inaisfs/data/home/tansy_criait/weights/Med-Qwen2.5-VL-7B-UnbiasGRPO'
lora_model_path = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/checkpoints/Med-Qwen2.5-VL-7B-UnbiasGRPO-SFT'
saved_model_path = '/mnt/inaisfs/data/home/tansy_criait/weights/Med-Qwen2.5-VL-7B-UnbiasGRPO-SFT'
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_model_name)
model = PeftModelForCausalLM.from_pretrained(base_model, lora_model_path, )
tokenizer = AutoTokenizer.from_pretrained(base_model_name)     
print("model loaded")
    
# 合并 LoRA 权重
merged_model = model.merge_and_unload()
merged_model.save_pretrained(saved_model_path)
tokenizer.save_pretrained(saved_model_path)


