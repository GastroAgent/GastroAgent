import gc
import random
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import json
import PIL
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, GenerationConfig
import sys
import warnings

warnings.filterwarnings("ignore")
from safetensors.torch import load_file

sys.path.append('/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1')
from llava.model.llavaProcessor import LlavaProcessor
from llava.model.language_model.llava_qwen import LlavaQwen2ForCausalLM
from trl.data_utils import maybe_apply_chat_template
from llava.model.language_model.llava_qwen_mul import PloyLlavaLlamaForCausalLM, LlavaConfig

model_path = '/mnt/inaisfs/data/home/tansy_criait/weights/Llava-Qwen2-7B-tune-med-SFT-New'
model_save_path = '/mnt/inaisfs/data/home/tansy_criait/weights/Llava-Qwen2-7B-tune-med-SFT-RL-New'
lora_model_path = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/checkpoints/Llava-Qwen2-7B-tune-med-SFT-RL-New'
# lora_model_path = None
### /home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/model_merge.py
# non_lora_model_path = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/checkpoints/Med-Llava-Qwen2-7B-GSPO-1216/non_lora_trainables.bin'
non_lora_model_path = ''
model = PloyLlavaLlamaForCausalLM.from_pretrained(
    model_path,
    device_map='auto',
    torch_dtype=torch.bfloat16,
)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#         model_path, torch_dtype="bfloat16", device_map="auto",
#         attn_implementation="flash_attention_2"
# )
print(model)
if lora_model_path is not None and os.path.exists(lora_model_path):
    # 加载 LoRA 模型并合并权重
    try:
        lora_state_dict = torch.load(f'{lora_model_path}/lora_trainables.bin')
        # lora_state_dict = load_file(f'{lora_model_path}/adapter_model.safetensors')
    except:
        lora_state_dict = load_file(f'{lora_model_path}/adapter_model.safetensors')
    print('lora_state_dict:\n', list(lora_state_dict.keys()))
    print('=' * 100)

    model = PeftModel.from_pretrained(model, lora_model_path)
    model.load_state_dict(lora_state_dict, strict=False)
    print(model)
    if non_lora_model_path:
        non_lora_state_dict = torch.load(non_lora_model_path, map_location=torch.device('cpu'))
        print('non_lora_state_dict:\n', non_lora_state_dict)
        model.load_state_dict(non_lora_state_dict, strict=False)

    model = model.merge_and_unload()  # 合并 LoRA权重 并卸载 LoRA层 为普通 Model。

if non_lora_model_path and not lora_model_path:
    non_lora_state_dict = torch.load(non_lora_model_path, map_location=torch.device('cpu'))
    print('non_lora_state_dict:\n', non_lora_state_dict)
    model.load_state_dict(non_lora_state_dict, strict=False)

processing_class = LlavaProcessor.from_pretrained(model_path,
                                                  use_fast=True,
                                                  trust_remote_code=True,
                                                  patch_size=14)
processing_class.tokenizer.padding_side = 'left'

# model = None
# processing_class = AutoProcessor.from_pretrained(model_path, use_fast=True)

if model_save_path is not None and model_save_path:
    model.save_pretrained(model_save_path)
    processing_class.save_pretrained(model_save_path)
    print('Saved model to', model_save_path)

### /home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/model_merge.py



