from safetensors.torch import load_file
import sys
sys.path.append('/home/work/VLM-R1/src/open-r1-multimodal/src/open_r1') # 本地调试
sys.path.append('/home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1') # 服务器调试
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from llava.model.llavaProcessor import LlavaProcessor
from llava.model.language_model.llava_qwen import LlavaQwen2ForCausalLM

model_id = '/home/dalhxwlyjsuo/criait_tansy/jmf/code/model_match/serve_llavaqwen2'
model_init_kwargs = {
    'attn_implementation': 'flash_attention_2',
    'torch_dtype': 'bfloat16', # float32, bfloat16 # flash attention 只支持bfloat16。
    'use_cache': True,
}

model = LlavaQwen2ForCausalLM.from_pretrained(
    model_id,
    device_map = 'auto',
    **model_init_kwargs
)

processing_class = LlavaProcessor.from_pretrained(model_id,
                            use_fast=True,
                            trust_remote_code=model_init_kwargs.get("trust_remote_code", None),
                            patch_size = 14)

print(processing_class.tokenizer.vocab_size)
for k,v in model.state_dict().items():
    print(k, v.shape)
print(processing_class.tokenizer.vocab_size)
# 假设你的LoRA模型权重保存在'lora_weights.safetensors'文件中
# weights_path = '/home/dalhxwlyjsuo/criait_tansy/jmf/GRPO_peft_weights/LlavaQwen2-GRPO-Tricks-Med/adapter_model.safetensors'
# model_save_path = None
#
# # 加载权重
# lora_weights = load_file(weights_path)
#
# # 打印权重
# for key in lora_weights.keys():
#     if '.vision_tower' in key:
#         continue
#     print(f"Layer: {key}")
#     # 打印每一层的具体权重值
#     # 注意：这里的权重可能是多维张量，直接打印可能非常大。
#     # 因此，这里只打印其shape信息作为示例。
#     print(f"Shape: {lora_weights[key].shape}\n")
#
#     # 如果你想查看具体的数值，可以取消下面这行的注释
#     print(lora_weights[key])
# print(2 * lora_weights['base_model.model.model.layers.27.self_attn.q_proj.lora_B.weight'] @ lora_weights['base_model.model.model.layers.27.self_attn.q_proj.lora_A.weight'])
# print((2 * lora_weights['base_model.model.model.layers.27.self_attn.q_proj.lora_B.weight'] @ lora_weights['base_model.model.model.layers.27.self_attn.q_proj.lora_A.weight']).max())
#
# from safetensors.torch import load_file
#
# state_dict = load_file(f'{lora_model_path}/adapter_model.safetensors')
# for k, v in state_dict.items():
#     if 'lora_B' in k:
#         print(f"{k}:")
#         print(v)
#         print('max:', v.max())
# del state_dict
# torch.cuda.empty_cache()
# gc.collect()
#
# key = 'model.mm_projector.0.weight'
# raw_state = model.state_dict()[key]
#
# # 加载 LoRA 模型并合并权重
# model = PeftModel.from_pretrained(model, lora_model_path)
# model = model.merge_and_unload()  # 合并 LoRA权重 并卸载 LoRA层 为普通 Model。
#
# if model_save_path is not None and model_save_path:
#     model.save_pretrained(model_save_path)
#
# new_state = model.state_dict()[key]
#
# print(new_state - raw_state)
# print((new_state - raw_state).sum())
# print((new_state - raw_state).max())
