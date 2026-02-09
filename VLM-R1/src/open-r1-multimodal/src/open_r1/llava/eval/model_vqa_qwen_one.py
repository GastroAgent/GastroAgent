import argparse
import torch
import os
import gc
import json
from tqdm import tqdm
import shortuuid

import sys
sys.path.append('/home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1')
sys.path.append('/home/work/VLM-R1/src/open-r1-multimodal/src/open_r1') # 本地调试

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.model import *
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from transformers import GenerationConfig, AutoTokenizer
from llava.model.llavaProcessor import LlavaProcessor
from PIL import Image
import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, use_flash_attn=True)
    model = PloyLlavaLlamaForCausalLM.from_pretrained(model_path, device_map='auto', attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    image_processor = model.get_vision_tower().image_processor
    check = ['model.layers.15.self_attn.q_proj.weight', 'model.mm_projector.0.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.10.self_attn.k_proj.weight']
    state_dict = model.state_dict()
    for c in check:
        print(state_dict[c])
    del state_dict
    torch.cuda.empty_cache()
    gc.collect()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # special_tokens = {'additional_special_tokens': ['<image>', '<|Image_start|>', '<|Image_end|>', '<answer>', '</answer>', '<think>', '</think>']}
    # tokenizer.add_special_tokens(special_tokens)
    print(tokenizer)
    print(model.device)
    #questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]

    # 一次性加载整个 JSON 文件
    with open(os.path.expanduser(args.question_file), "r") as f:
        try:
            questions = json.load(f)  # 加载整个 JSON 文件，而不是逐行解析
            print(f"[DEBUG] Loaded {len(questions)} questions from {args.question_file}")
        except json.JSONDecodeError as e:
            print(f"Error loading JSON file: {e}")
            return

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        # 提示词已经有了 image token， 这里就不要再加了。
        # if model.config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        qs = qs.replace("<image>", "<|Image_start|><image><Image_end>")
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # prompt = f"""你是一个胃肠镜科室医生的人工智能助手，针对用户的问题提供有用、详细且礼貌的答案。
        #             USER: <image>对于该病例最准确的诊断是什么？
        #             请结合给定的医学图像、临床信息，生成诊断。
        #             请不要输出思考推理过程，直接输出诊断

        if 'grpo' in args.model_path.lower():
            prompt = prompt.replace("<|Image_start|>", "<Image>").replace("<|Image_end|>", "</Image>")
        cur_prompt = qs

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        print(prompt)
        print(tokenizer.pad_token_id, tokenizer.pad_token)
        print(tokenizer.eos_token_id, tokenizer.eos_token)
        print(input_ids)
        image = Image.open(image_file)
        # image_tensor = process_images([image], image_processor, model.config)[0]
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(torch.bfloat16)

        #image = Image.open(image_file).convert('RGB')
        #image_tensor = process_images([image], image_processor, model.config)[0]
        generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=True,
            temperature=0.5,
            top_p = 0.95,
            top_k = 30,
            pad_token_id=tokenizer.pad_token_id,
            use_cache = True,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        print(image_tensor.shape)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids.to(model.device), # [B, S]
                images=image_tensor.to(model.device), # [B, C, H, W] 或 list[N, C, H, W]
                generation_config=generation_config,
                num_images=[1]
            )
        print(output_ids)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}, ensure_ascii=False) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="v0_mmtag")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
