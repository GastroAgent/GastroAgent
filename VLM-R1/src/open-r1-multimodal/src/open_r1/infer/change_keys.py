import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from safetensors.torch import load_file, save_file

path = "/home/dalhxwlyjsuo/criait_tansy/checkpoints/Llava-Qwen2-7B-knowtune-med-v0-mmtag-mul-next-stage2-2-redo/adapter_model.safetensors"
weights = load_file(path)
# base_model.model.model.vision_tower.vision_tower.vision_model.encoder.layers.23.self_attn.v_proj.lora_B.default.weight
# base_model.model.model.layers.20.self_attn.k_proj.lora_B.weight
# print(list(weights.keys()))
new_weights = {}

for key in weights.keys():
    if 'vision_tower.vision_tower' in key: # 替换 not 即可。
        new_weights[key] = weights[key]
print('='*100)
print('='*100)
print(list(new_weights.keys()))
save_file(new_weights, path)

# /home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/change_keys.py