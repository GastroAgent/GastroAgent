
import gc
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
import PIL
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, GenerationConfig
import sys
import warnings
warnings.filterwarnings("ignore")
from safetensors.torch import load_file

sys.path.append('/home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1') # 服务器调试

from llava.model.llavaProcessor import LlavaProcessor
from llava.model.language_model.llava_qwen import LlavaQwen2ForCausalLM
from trl.data_utils import maybe_apply_chat_template
from llava.model.language_model.llava_qwen_mul import PloyLlavaLlamaForCausalLM, LlavaConfig


model_id = '/home/dalhxwlyjsuo/criait_tansy/weights/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-buwei'

model = PloyLlavaLlamaForCausalLM.from_pretrained(
    model_id,
    device_map = 'auto',
    torch_dtype=torch.bfloat16,
)

processing_class = LlavaProcessor.from_pretrained(model_id,
                            use_fast=True,
                            trust_remote_code=True,
                            patch_size = 14)

processing_class.tokenizer.padding_side = 'left'
vision_model = model.model.vision_tower.vision_tower
image_processor = model.model.vision_tower.image_processor
print(vision_model)
for k, v in vision_model.state_dict().items():
    print(k, v)
vision_model.save_pretrained('/home/dalhxwlyjsuo/criait_tansy/weights/my-clip-vision')
image_processor.save_pretrained('/home/dalhxwlyjsuo/criait_tansy/weights/my-clip-vision')

