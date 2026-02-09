import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import json
from vllm import LLM, SamplingParams
import torch
from tqdm import tqdm
import json
from glob import glob

files = ["/mnt/inaisfs/data/home/tansy_criait/VLM-R1/results/Kvasir/LlavaQwen2-GRPO-Tricks-Total-CoT-6000/Kvasir-en.json"]
llm = LLM(model="/mnt/inaisfs/data/home/tansy_criait/weights/Qwen2.5-14B-Instruct",
    gpu_memory_utilization = 0.95,
    tensor_parallel_size = 1,
    dtype = torch.bfloat16,
    max_model_len = 8192
)
sampling_params = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=1024, n=1)

for idx, data_path in enumerate(files):
    print(f"------------------{idx}/{len(files)}-----------------------")
    print(data_path)
    with open(data_path, 'r') as f:
        results = json.load(f)
    for data in tqdm(results):
        pred = data['generated_text']
        if 'extracted_answer' in data:
            continue

        prompt = f"Response: {pred}\n\nYour Task: Extract the final answer and place it within <answer></answer>.\nThe output might be a bit confusing. You need to standardize the final answer."
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        outputs = llm.chat(messages, sampling_params, use_tqdm=False)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            data['extracted_answer'] = generated_text
    print("处理完成。")
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

