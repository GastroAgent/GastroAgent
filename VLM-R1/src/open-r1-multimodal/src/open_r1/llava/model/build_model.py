# import torch
# import sys
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,1,7"
# sys.path.append('/home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1')
# import transformers
#
# from transformers import Qwen2Config
# from llava.model.language_model.llava_qwen_mul import PloyLlavaLlamaForCausalLM
#
# model = PloyLlavaLlamaForCausalLM.from_pretrained('/home/dalhxwlyjsuo/criait_tansy/weights/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-buwei',
#                                                   attn_implementation="flash_attention_2",
#                                                   torch_dtype=torch.bfloat16)
# model.model.vision_tower.load_model()
# old_model_id = '/home/dalhxwlyjsuo/criait_tansy/weights/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-buwei'
# old_model = PloyLlavaLlamaForCausalLM.from_pretrained(
#     old_model_id,
#     device_map = 'auto',
#     torch_dtype=torch.bfloat16
# )
# state_dict = model.state_dict()
# for k, v in old_model.state_dict().items():
#     if 'vision_tower.' in k or 'mm_projector' in k:
#         print(k, state_dict[k])
#
# print(model)

import torch
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,1,0"
sys.path.append('/home/lab/work/VLM-R1/src/open-r1-multimodal/src/open_r1')
import transformers

from transformers import Qwen2Config, AutoTokenizer, AutoProcessor
from llava.model.language_model.llava_qwen_mul import PloyLlavaLlamaForCausalLM

# # 加载新模型
# new_model = PloyLlavaLlamaForCausalLM.from_pretrained(
#     '/home/lab/mllm_weight/Llava-Qwen2-32B-tune-med',
#     device_map='auto',
#     torch_dtype=torch.bfloat16
# )
# new_model.model.vision_tower.load_model()

new_model_id = '/home/lab/mllm_weight/Llava-Qwen2-32B-tune-med'
new_model = PloyLlavaLlamaForCausalLM.from_pretrained(
    new_model_id,
    device_map='auto',
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(new_model_id)
processor = AutoProcessor.from_pretrained(new_model_id)

# 加载旧模型
# old_model_id = '/home/dalhxwlyjsuo/criait_tansy/weights/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-buwei'
# old_model = PloyLlavaLlamaForCausalLM.from_pretrained(
    # old_model_id,
    # device_map='auto',
    # torch_dtype=torch.bfloat16
# )
# tokenizer = AutoTokenizer.from_pretrained(old_model_id)
# processor = AutoProcessor.from_pretrained(old_model_id)

# 获取新旧模型的 state_dict
new_state_dict = new_model.state_dict()
# old_state_dict = old_model.state_dict()
old_state_dict = {}
# mm_projector_path = '/home/dalhxwlyjsuo/criait_tansy/jmf/weight/my-mm_projector.pth'
# vision_model_path = '/home/dalhxwlyjsuo/criait_tansy/jmf/weight/my-clip_vision.pth'
mm_projector_path = '/home/lab/mllm_weight/my-mm_projector.pth'
vision_model_path = '/home/lab/mllm_weight/my-clip_vision.pth'

for k, v in torch.load(mm_projector_path).items():
    old_state_dict[k] = v

for k, v in torch.load(vision_model_path).items():
    old_state_dict[k] = v

# 复制 vision_tower 参数
for k, v in old_state_dict.items():
    if 'vision_tower' in k:
        # 确保键名匹配（检查新旧模型的键是否一致）
        if k in new_state_dict:
            print(f"Updating {k}...")
            # 将旧模型参数移动到新模型的设备上
            new_state_dict[k].copy_(v.to(new_state_dict[k].device))
        else:
            print(f"Key {k} not found in new model state_dict.")

# 将修改后的 state_dict 重新加载到新模型
new_model.load_state_dict(new_state_dict, strict=False)  # strict=False 允许部分参数不匹配

# 保存更新后的模型
save_path = '/home/lab/mllm_weight/Llava-Qwen2-32B-tune-med'
new_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
processor.save_pretrained(save_path)

# for k, v in new_model.state_dict().items():
#     if 'vision_tower' in k:
#         # 确保键名匹配（检查新旧模型的键是否一致）
#         print(v)