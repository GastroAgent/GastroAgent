import argparse
import torch
import os
import gc
import json
from tqdm import tqdm
import shortuuid
import sys
from transformers import GenerationConfig, AutoTokenizer
from llava.model.llavaProcessor import LlavaProcessor
from PIL import Image
import math
import torch
from modelscope import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    # Model
    model_path = args.model_path
    # specify the path to the model
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path, attn_implementation="flash_attention_2")
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path,
                                                                         trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    print(vl_gpt)

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

        try:image_file = line["image"]
        except:image_file = line['images']

        if isinstance(image_file, str):
            image_file = [image_file]
        if len(image_file) > args.max_num_images:
            continue

        qs = line["text"].replace('<image>', '<image_placeholder>')
        qs = f"""{qs}\n
建议或提示: 你可以通过检查食管、胃角形态、幽门、十二指肠等情况来做出更准确的诊断。
输出要求:
 - 1.分析过程放在<think>和</think>标记之间, 而最终诊断结果放在<answer>和</answer>标记之间。
 - 2.除了专业名词、缩写词、实体词和特殊标记(如<think>、</think>、<answer>和</answer>等)外，使用中文回答。"""
        sys_prompt = f'''A chat between a curious user and an medical artificial intelligence assistant. As the medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, and use natural language to assist users in various tasks.The visual content will be provided with the following format: <|Image_start|>visual content<|Image_end|>.'''
        conversation = [
            {
                "role": "System",
                "content": sys_prompt,
            },
            {
                "role": "User",
                "content": qs,
                "images": image_file
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.5,
            top_p = 0.95,
            use_cache=True
        )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        print(answer)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": conversation,
                                   "text": answer,
                                   "answer_id": ans_id,
                                   "model_id": model_path,
                                   "gpt4_answer": line['gpt4_answer'],
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
    parser.add_argument("--force_think", type=bool, default=False)
    parser.add_argument("--max_num_images", type=int, default=8)
    args = parser.parse_args()

    eval_model(args)
