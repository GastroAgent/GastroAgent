#!/usr/bin/env python3
import argparse
import gc
import random
import json
import os
import sys
import PIL
import shortuuid
import torch
from peft import PeftModel
from transformers import GenerationConfig
from llava.model.llavaProcessor import LlavaProcessor
from llava.model.language_model.llava_qwen import LlavaQwen2ForCausalLM
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates

# 本地和服务器调试路径
sys.path.append('/home/work/VLM-R1/src/open-r1-multimodal/src/open_r1')
sys.path.append('/home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1')

def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA Qwen2 图文推理脚本，支持可选 LoRA 权重")
    parser.add_argument('--question_file', type=str, required=True,
                        help="输入 JSON 文件（数组格式），每个元素包含 question_id/text/image 等字段")
    parser.add_argument('--answers_file', type=str, required=True,
                        help="输出 JSONL 文件路径，每行一个答复对象")
    parser.add_argument('--num_chunks', type=int, default=1,
                        help="总共拆分几块（默认为1）")
    parser.add_argument('--chunk_idx', type=int, default=0,
                        help="当前运行第几块（从0开始）")
    parser.add_argument('--model_id', type=str,
                        default='/home/dalhxwlyjsuo/criait_tansy/jmf/code/model_match/serve_llavaqwen2',
                        help="基础模型目录")
    parser.add_argument('--lora_model_path', type=str, default=None,
                        help="LoRA 权重目录，如果不使用则留空")
    parser.add_argument('--temperature', type=float, default=0.2,
                        help="采样温度")
    parser.add_argument('--top_p', type=float, default=0.9,
                        help="nucleus sampling 概率阈值")
    parser.add_argument('--num_beams', type=int, default=1,
                        help="beam 搜索数目")
    parser.add_argument('--conv_mode', type=str, default='v1',
                        help="对话模板模式 (如 v1)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 一次性加载整个 JSON 数据
    with open(os.path.expanduser(args.question_file), 'r', encoding='utf-8') as f:
        try:
            questions = json.load(f)
            print(f"[DEBUG] Loaded {len(questions)} questions from {args.question_file}")
        except json.JSONDecodeError as e:
            print(f"Error loading JSON file: {e}")
            return

    # 按块拆分
    def split_list(lst, n):
        chunk_size = (len(lst) + n - 1) // n
        return [lst[i * chunk_size:(i + 1) * chunk_size] for i in range(n)]
    chunks = split_list(questions, args.num_chunks)
    if args.chunk_idx < 0 or args.chunk_idx >= len(chunks):
        print(f"chunk_idx {args.chunk_idx} 超出范围 (0–{len(chunks)-1})")
        return
    dataset = chunks[args.chunk_idx]

    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.expanduser(args.answers_file)), exist_ok=True)
    ans_file = open(os.path.expanduser(args.answers_file), 'w', encoding='utf-8')

    # 加载基础模型
    model_init_kwargs = {
        'attn_implementation': 'flash_attention_2',
        'torch_dtype': 'bfloat16',
        'use_cache': True,
        'trust_remote_code': True,
    }
    model = LlavaQwen2ForCausalLM.from_pretrained(
        args.model_id,
        device_map='auto',
        **model_init_kwargs
    )
    # 加载并合并 LoRA 权重
    if args.lora_model_path:
        model = PeftModel.from_pretrained(model, args.lora_model_path)
        model = model.merge_and_unload()

    # 加载处理器
    processing_class = LlavaProcessor.from_pretrained(
        args.model_id,
        use_fast=True,
        trust_remote_code=model_init_kwargs.get('trust_remote_code'),
        patch_size=14
    )
    processing_class.tokenizer.padding_side = 'left'

    # 生成配置
    generation_config = GenerationConfig(
        max_new_tokens=1024,
        do_sample=True,
        temperature=args.temperature,
        pad_token_id=processing_class.tokenizer.pad_token_id,
    )

    # 推理循环
    for example in dataset:
        idx = example.get('question_id')
        image_path = example.get('image') or example.get('image_path')
        qs = example.get('text', '')

        # 构造 prompt 文本
        if getattr(model.config, 'mm_use_im_start_end', False):
            prompt_qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            prompt_qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        # 应用对话模板
        conv = conv_templates.get(args.conv_mode, conv_templates['v1']).copy()
        conv.append_message(conv.roles[0], prompt_qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        #print(f"[DEBUG] prompt: {prompt.replace(chr(10), '\\n')}")

        # 编码输入 ids
        input_ids = tokenizer_image_token(
            prompt,
            processing_class.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(model.device)

        # 读取并预处理图像
        image = PIL.Image.open(image_path)
        img_tensor = processing_class.preprocess(image, return_tensors='pt')['pixel_values'].to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=img_tensor.unsqueeze(0).half(),
                image_sizes=[image.size],
                generation_config=generation_config,
                top_p=args.top_p,
                num_beams=args.num_beams,
                use_cache=True
            )

        # 解码生成结果
        outputs = processing_class.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # 写入 JSONL
        record = {
            "question_id": idx,
            "prompt": prompt_qs,
            "text": outputs,
            "answer_id": shortuuid.uuid(),
            "model_id": os.path.basename(args.model_id.rstrip('/')),
            "metadata": {}
        }
        ans_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        ans_file.flush()

    ans_file.close()
    print("Infer Done! 输出保存在", args.answers_file)

if __name__ == '__main__':
    main()
