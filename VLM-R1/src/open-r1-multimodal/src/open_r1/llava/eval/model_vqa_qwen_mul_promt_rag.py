import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

import sys
sys.path.append('/home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1')
sys.path.append('/home/work/VLM-R1/src/open-r1-multimodal/src/open_r1') # 本地调试

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder_mul import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

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
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

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
        image_file = line["images"]
        qs = line["text"]
        cur_prompt = qs
        
        # if model.config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print('prompt============',prompt)
        
        # context = ''        
        with open('/home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/cal_eval/实体频率统计.txt', 'r', encoding='utf-8') as f:
            context = f.read()
        
        prompt = f"""你是一个胃肠镜科室医生的人工智能助手，针对用户的问题提供有用、详细且礼貌的答案。
                    USER: <image>\n<image>\n<image>\n<image>\n对于该病例最准确的诊断是什么？
                    请结合参考资料和给定的一组多张医学图像、临床信息，生成诊断，参考资料会在下面提供。
                    输出格式:
                    请不要输出思考推理过程，直接输出诊断，诊断只可能是参考资料中疾病实体一种、两种或三种的组合

                    例如：问题：<image>\n<image>\n<image>\n对于该病例最准确的诊断是什么？
                    输出：食管癌术后 吻合口溃疡 慢性浅表性胃窦炎

                    相关资料：
                    {context}
                    该参考资料统计了常见的胃部疾病实体和对应出现的频率
                    请根据以上资料准确回答。

                 """

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # 直接从image_file加载图像
        image_tensors = []
         #=========读取的是列表
        for img_path in image_file:
            try:
                image = Image.open(img_path)
                #image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                #image_tensors.append(image_tensor)
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
                image_tensors.append(image_tensor.half().cuda())
            except Exception as e:
                print(f"[ERROR] Failed to load image {img_path}: {e}")
        # 将所有图片的 tensor 合并成一个批次
        #image_tensors = torch.stack(image_tensors)  # 只用第一张图像进行调试
        image_tensors = torch.cat(image_tensors, dim=0)

        #image = Image.open(image_file).convert('RGB')
        #image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                #images=image_tensor.unsqueeze(0).half().cuda(),
                images=image_tensors,
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        #ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   #"answer_id": ans_id,
                                   "model_id": model_name,
                                   #"prompt_template":prompt,
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
