import argparse
import gc
import random
import torch
import os

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
    # model = PloyLlavaLlamaForCausalLM.from_pretrained(model_path, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    image_processor = model.get_vision_tower().image_processor

    check = ['model.layers.15.self_attn.q_proj.weight', 'model.mm_projector.0.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.10.self_attn.k_proj.weight']
    state_dict = model.state_dict()
    for c in check:
        print(state_dict[c])
    del state_dict
    torch.cuda.empty_cache()
    gc.collect()

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
    example_file = args.example_file if args.example_file else None
    print('example_file: ', example_file)
    if example_file is not None:
        with open(example_file, "r") as f:
            examples = json.load(f)
    else:
        examples = None
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["images"]
        qs = line["text"]
        num_images = len(image_file)
        if num_images > args.max_num_images:
            continue

        # qs = qs.replace("<|Image_start|><image><Image_end>", "<image>")
        # qs = qs.replace("<image>", "<|Image_start|><image><Image_end>")
        print(qs)
        # conv = conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()
        system = ("You are a helpful Medical AI assistant and an expert in the area of the Digestive Tract and Human Stomach. "
         "As the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, and use natural language to assist users in various tasks. "
         "As the expert, you need to make sure that your response is logical and factual, and that it conveys the authority of the expert. "
         "The visual content will be provided with the following format: <|Image_start|>visual content<|Image_end|>.")
        if examples is None:
            prompt = f''' <|im_start|>system
{system}<|im_end|>
<|im_start|>user\n{qs}\n
建议或提示: 
  - 1.你可以通过检查食管、胃角形态、幽门、十二指肠等情况来做出更准确的胃部疾病诊断。
  - 2.你可以通过一步步分析每张图片的具体症状来帮助你的最终诊断。
输出要求:
 - 1.分析过程放在<think>和</think>标记之间, 而最终诊断结果放在<answer>和</answer>标记之间。
 - 2.除了专业名词、缩写词、实体词和特殊标记(如<think>、</think>、<answer>和</answer>等)外，使用中文回答。<|im_end|>\n<|im_start|>assistant\n'''.replace('<image>', '<|Image_start|><image><|Image_end|>')

        else:
            thinks_templete = [
                '',
                '检查所见',
                '根据影像表现为',
                "检测结果显示",
                "病理表现为",
                '根据影像内容，初步表现为',
            ]
            answers_templete = [
                '最终诊断结果为',
                '病理诊断为',
                '倾向诊断是',
                '最可能的诊断是',
                '考虑诊断为'
            ]
            is_valid = False
            while not is_valid:
                ref_data = random.choice(examples)
                if len(ref_data['image']) < 4 and '慢性浅表性' not in ref_data['镜下诊断']:
                    is_valid = True
            ref_data['镜下诊断'] = str(ref_data['镜下诊断']).strip()
            ref_data['镜下所见'] = ref_data['镜下所见'].strip()
            ref_num_images = len(ref_data['image'])
            ref_think_content = f'''<think>{str(ref_data['镜下所见']).replace("检查所见", random.choice(thinks_templete))}</think>'''
            ref_answer_content = f'''{random.choice(answers_templete)}: <answer>{str(ref_data['镜下诊断']).replace(" ", "，")}</answer>'''
            # answer_content = answer_content.replace('.。', '。').replace('。。', '。').replace('，。', '。').replace('？。', '。')
            # answer_content = re.sub(r'([\.，？！；,?]+\s*。)|。+', '。', answer_content)
            ref_images_token = "  ".join(["<image>" for _ in range(ref_num_images)])
            ref_question = random.choice(questions)
            # ref_human_content = f"""{ref_images_token}\n  {ref_question}"""
            # ref_gpt_content = f"""{ref_think_content}\n  {ref_answer_content}"""

            # 构造用户 Prompt 及 示例。
            cot_templete = f"""建议或提示: 
 - 1.检查食管、胃角形态、幽门、十二指肠等情况来做出更准确的胃部疾病诊断。
 - 2.一步步分析每张图片的具体症状来帮助你的最终诊断。
参考示例1:
    问题:
        {ref_images_token}
        {ref_question}
    诊断分析:
        {ref_think_content}
        {ref_answer_content}"""

            zh = f'''{cot_templete}\n\n仿照上述参考示例回答初始的用户问题，输出要求:\n - 1.分析过程放在<think>和</think>标记之间, 而最终诊断结果放在<answer>和</answer>标记之间。\n - 2.除了专业名词、缩写词、实体词和特殊标记(如<think>、</think>、<answer>和</answer>等)外，使用中文回答。'''
            # en = 'Please answer the above questions based on the information provided in the image, give the thinking process in as much detail as possible, and then summarize the process to get a concise and effective final answer. \n Output requirements: The reasoning process goes between <think> and </think> tags, and the final answer goes between <answer> and </answer> tags.'
            prompt = f''' <|im_start|>system
{system}<|im_end|>
<|im_start|>user\n{qs}\n{zh}<|im_end|>\n<|im_start|>assistant\n'''.replace(
                '<image>', '<|Image_start|><image><|Image_end|>')

        if args.force_think:
            prompt = prompt + '<think>'
        prompt = prompt.replace('<|Image_start|><image><|Image_end|>', '<image>')
        prompt = prompt.replace('<image>', '<|Image_start|><image><|Image_end|>')
        raw_question = '对于该病例最准确的诊断是什么？'
        new_question = '基于提供的图像资料，逐步分析可能的相关特征及考虑点，并汇总该分析过程，给出一个明确且简洁的诊断结果。'
        # new_question = '对于该病例最准确的诊断是什么？'
        prompt = prompt.replace(raw_question, new_question)

        if 'grpo' in args.model_path.lower():
            prompt = prompt.replace("<|Image_start|>", "<Image>").replace("<|Image_end|>", "</Image>")
        cur_prompt = qs
        print(prompt)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # 直接从image_file加载图像
        image_tensors = []
         #=========读取的是列表

        for img_path in image_file:
            try:
                image = Image.open(img_path)
                image = crop_right_square(image)
                #image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                #image_tensors.append(image_tensor)
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
                image_tensors.append(image_tensor.cuda())
            except Exception as e:
                print(f"[ERROR] Failed to load image {img_path}: {e}")
        # 将所有图片的 tensor 合并成一个批次
        #image_tensors = torch.stack(image_tensors)  # 只用第一张图像进行调试
        image_tensors = torch.cat(image_tensors, dim=0).to(torch.bfloat16)

        #image = Image.open(image_file).convert('RGB')
        #image_tensor = process_images([image], image_processor, model.config)[0]
        generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=True,
            temperature=0.2,
            top_p = 0.95,
            top_k = 30,
            pad_token_id=tokenizer.pad_token_id,
            use_cache = True,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        print(image_tensors.shape, num_images)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids.to(model.device), # [B, S]
                images=image_tensors.to(model.device), # [B, C, H, W] 或 list[N, C, H, W]
                generation_config=generation_config,
                num_images=[num_images]
            )
            # output_ids = model.generate(
            #     input_ids,
            #     #images=image_tensor.unsqueeze(0).half().cuda(),
            #     images=image_tensors,
            #     image_sizes=[image.size],
            #     do_sample=True if args.temperature > 0 else False,
            #     temperature=args.temperature,
            #     top_p=args.top_p,
            #     num_beams=args.num_beams,
            #     # no_repeat_ngram_size=3,
            #     max_new_tokens=1024,
            #     use_cache=True)
        # print(output_ids)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
        if args.force_think:
            outputs = '<think>' + outputs
        print(outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "prompt_template":prompt,
                                   "inputs_shape": input_ids.shape,
                                   "gpt4_answer": line['gpt4_answer'],
                                   "metadata": {}}, ensure_ascii=False) + "\n")
        ans_file.flush()
    ans_file.close()

def crop_right_square(image: Image.Image,
                      target_weight_size: int = 1280,
                      target_height_size=1080,
                      identy = False) -> Image.Image:
    """
    从图像右侧裁剪出一个正方形区域（target_size x target_size）。

    参数:
        image (Image.Image): 输入的 PIL 图像对象。
        target_size (int): 裁剪目标区域的边长（默认为 1080）。

    返回:
        Image.Image: 裁剪后的图像对象。
    """
    if identy:
        return image
    # 获取图像尺寸
    width, height = image.size

    # 检查图像尺寸是否满足裁剪需求
    if width < target_weight_size or height < target_height_size:
        return image

    # 计算裁剪区域（右上角的正方形）
    left = width - target_weight_size
    top = 0
    right = width
    bottom = target_height_size

    # 执行裁剪
    cropped_image = image.crop((left, top, right, bottom))

    return cropped_image

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
    parser.add_argument("--num_beams", type=int, default=1) # max_num_images
    parser.add_argument("--max_num_images", type=int, default=8)
    parser.add_argument("--force_think", type=bool, default=False)
    parser.add_argument("--example_file", type=str, default=None)
    args = parser.parse_args()

    eval_model(args)
