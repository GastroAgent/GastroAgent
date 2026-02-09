import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from copy import deepcopy

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import difflib

from PIL import Image
import math


def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()
 
def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    #print('1111111111111')
    #print('str_list',str_list)
    #print('target_str',target_str)
    most_similar_str = None
    most_similar_index = None
    highest_similarity = 0
    
    # Iterate through each string in the list
    for i, str in enumerate(str_list):
        #print('=====2222222')
        #print('str',str)
        # Calculate the similarity between the current string and the target string
        similarity = str_similarity(str, target_str)
        #print('similarity',similarity)
        
        # If the current string is more similar than the previous most similar string, update the variables
        if similarity > highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity
    
    # Return the index of the most similar string
    return most_similar_index


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]




def eval_model(args, question_file, answers_base_path):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    #print(f"Total Parameters: {total_params}")
    #print(f"Trainable Parameters: {trainable_params}")

    #questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = json.load(open(os.path.expanduser(question_file), "r"))
    # 一次性加载整个 JSON 文件
    # with open(os.path.expanduser(args.question_file), "r") as f:
    #     try:
    #         questions = json.load(f)  # 加载整个 JSON 文件，而不是逐行解析
    #         print(f"[DEBUG] Loaded {len(questions)} questions from {args.question_file}")
    #     except json.JSONDecodeError as e:
    #         print(f"Error loading JSON file: {e}")
    #         return


    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    #answers_file = os.path.expanduser(args.answers_file)
    answers_file = os.path.join(answers_base_path, 'tmp', os.path.dirname(question_file).replace('/', '_') + 'pred.jsonl')
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    print('start inference...', flush=True)
    res = []

    for line in tqdm(questions):
    #for i, line in enumerate(tqdm(questions)):

        question, gt_ans, image = preprocess_input(line)
        print('question',question)

        qs = question['value']

        qs = qs.replace('<image>', '').strip()
        cur_prompt = qs

        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        images = image_tensor.unsqueeze(0).half().cuda()


        # idx = line["question_id"]
        # image_file = line["images"]
        # qs = line["text"]
        #cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        #cur_prompt = cur_prompt + '\n' + '<image>'

        #if args.conv_mode == 'simple_legacy':
        #qs += '\n\n### Response:'

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # # 直接从image_file加载图像
        # image_tensors = []
        #  #=========读取的是列表
        # for img_path in image_file:
        #     try:
        #         image = Image.open(img_path)
        #         #image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        #         #image_tensors.append(image_tensor)
        #         image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        #         image_tensors.append(image_tensor.half().cuda())
        #     except Exception as e:
        #         print(f"[ERROR] Failed to load image {img_path}: {e}")
        # # 将所有图片的 tensor 合并成一个批次
        # #image_tensors = torch.stack(image_tensors)  # 只用第一张图像进行调试
        # image_tensors = torch.cat(image_tensors, dim=0)


        #image = Image.open(image_file).convert('RGB')
        #image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                #images=image_tensor.unsqueeze(0).half().cuda(),
                images=images,
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print('outputs',outputs)

        #if args.conv_mode == 'simple_legacy':
        # while True:
        #     cur_len = len(outputs)
        #     outputs = outputs.strip()
        #     for pattern in ['###', 'Assistant:', 'Response:']:
        #         if outputs.startswith(pattern):
        #             outputs = outputs[len(pattern):].strip()
        
        # try:
        #         index = outputs.index(conv.sep)
        # except ValueError:
        #         outputs += conv.sep
        #         index = outputs.index(conv.sep)

        # outputs = outputs[:index].strip()


        # if True: #args.answer_prompter:
        #         print('======lucky')
        #         outputs_reasoning = outputs
        #         inputs = tokenizer([prompt + outputs_reasoning + ' ###\nANSWER:'])

        #         input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        #         #keywords = ['###']
        #         #stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        #         with torch.inference_mode():
        #             output_ids = model.generate(
        #             input_ids,
        #             #images=image_tensor.unsqueeze(0).half().cuda(),
        #             images=images,
        #             image_sizes=[image.size],
        #             do_sample=True if args.temperature > 0 else False,
        #             temperature=args.temperature,
        #             top_p=args.top_p,
        #             num_beams=args.num_beams,
        #             # no_repeat_ngram_size=3,
        #             max_new_tokens=1024,
        #             use_cache=True)

        #         outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        #         print(outputs)

        #         if args.conv_mode == 'simple_legacy':
        #                 while True:
        #                     cur_len = len(outputs)
        #                     outputs = outputs.strip()
        #                     for pattern in ['###', 'Assistant:', 'Response:']:
        #                         if outputs.startswith(pattern):
        #                             outputs = outputs[len(pattern):].strip()
                
        #         try:
        #                 index = outputs.index(conv.sep)
        #         except ValueError:
        #                 outputs += conv.sep
        #                 index = outputs.index(conv.sep)

        #         outputs = outputs[:index].strip()
        #                 # outputs = outputs_reasoning + '\n The answer is ' + outputs



        save_dict = deepcopy(line)
        save_dict['model_pred'] = outputs
        #print('outputs', outputs)
        save_dict['prompt_question'] = cur_prompt


        ans_file.write(json.dumps(save_dict, ensure_ascii=False) + "\n")
        res.append(save_dict)
        #print('res',res)
        ans_file.flush()

        # ans_id = shortuuid.uuid()
        # ans_file.write(json.dumps({"question_id": idx,
        #                            "prompt": cur_prompt,
        #                            "text": outputs,
        #                            "answer_id": ans_id,
        #                            "model_id": model_name,
        #                            "metadata": {}}, ensure_ascii=False) + "\n")
        # ans_file.flush()
    #ans_file.close()

    pred_dict, correct_precentage = MedicalEval(res)
    print(correct_precentage)
    final_save_dict = {
        "model_name": model_name,
        "dataset_name": question_file,
        "correct_precentage" :correct_precentage,
        "pred_dict" : pred_dict
    }
    
    with open(os.path.join(answers_base_path, os.path.dirname(question_file).replace('/', '_') + '.json'), 'w') as f:
        json.dump(final_save_dict, f, indent=4, ensure_ascii=False)


def preprocess_input(entity) -> tuple:
    a,b,c,d = entity.get('option_A'), entity.get('option_B'), entity.get('option_C'), entity.get('option_D')
    #print(a)
    answer_list = [a, b]
    if c is not None:
        answer_list.append(c)
    if d is not None:
        answer_list.append(d)
    q_str = entity['question'] + f'Here are {len(answer_list)} candidate answers:' + str(answer_list)+' Only return what you think is the correct answer from the candidate answers, do not return any other irrelevant text!\n<image>'
    ans_str = entity['gt_answer']
    
    question = {
        'from': 'human',
        'value': q_str
    }
    
    
    answer = {
        "from": "gpt",
        "value": ans_str
    }
    
    image_url = entity.get('image_path')
    
    image = read_img_from_url(image_url)
    #print(image)
    return question, answer, image

    

from PIL import Image
import sys
from io import BytesIO

def read_img_from_url(url):
    acturl = '/home/dalhxwlyjsuo/criait_tansy/project/Multi-Modality-Arena-main/Multi-Modality-Arena-main/OmniMedVQA/OmniMedVQA/'  + url
    print(acturl)
    img = Image.open(acturl)
    #print('img',img)
    return img


def MedicalEval(pred_dict: list) -> tuple:
    tot = len(pred_dict)
    #print('tot',tot)
    succ = 0
    print('pred_dict',pred_dict)
    for data in pred_dict:
        print('======hello')
        try:
            a,b,c,d = data.get('option_A'), data.get('option_B'), data.get('option_C'), data.get('option_D')
            answer_list = [a, b]
            if c is not None:
                answer_list.append(c)
            if d is not None:
                answer_list.append(d)
            
            print(data['model_pred'])
            print(data['gt_answer'])
            print('answer_list',answer_list)
            print(find_most_similar_index(answer_list, data['model_pred']))
            if answer_list[find_most_similar_index(answer_list, data['model_pred'])] == data['gt_answer']:
                succ += 1
                data['is_correct'] = 'yes'
            else:
                data['is_correct'] = 'no'
        except:
            continue
        
    return pred_dict, succ/tot




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument('--answers_base_path', type=str, default="qwen_med_output_en_chestCT")
    args = parser.parse_args()

    print('start', flush=True)
    os.makedirs(args.answers_base_path, exist_ok=True)
    eval_model(args, args.question_file, args.answers_base_path)
    print('finish', flush=True)
