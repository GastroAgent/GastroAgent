import argparse
import glob
import random
import re
import torch
import os
import json
from tqdm import tqdm
# import shortuuid

import sys
# sys.path.append('/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1') # 本地调试
sys.path.append('/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1') # 服务器执行
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
    most_similar_index = 0
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
    questions = json.load(open(os.path.expanduser(question_file), "r"))
    res = json.load(open(answers_base_path, 'r'))

    if args.no_tag:
        if "True" in answers_base_path:
            pred_dict, correct_precentage = MedicalEvalNoTag(questions, res, tag=True)
        else:
            pred_dict, correct_precentage = MedicalEvalNoTag(questions, res, tag=False)
    else:
        if "Kvasir-zh" in answers_base_path:
            pred_dict, correct_precentage = MedicalEvalZh(questions, res)
        else:
            pred_dict, correct_precentage = MedicalEval(questions, res)
    print(correct_precentage)
    answers_base_path = answers_base_path.replace('.json', '') + f'-{args.no_tag}' + f'-{correct_precentage:.3f}-soft.json'
    with open(answers_base_path, 'w', encoding='utf-8') as f:
        json.dump(pred_dict, f, indent=4, ensure_ascii=False)

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
    
    image_url = entity.get('image_paths')[0]
    
    image = read_img_from_url(image_url)
    #print(image)
    return question, answer, image

from PIL import Image
import sys
from io import BytesIO

def read_img_from_url(url):
    acturl =  url
    print(acturl)
    img = Image.open(acturl)
    #print('img',img)
    return img

def extract_pred(content, **kwargs):
    # pattern = r'\\box{\[*(.*)\]*}'
    pattern = r'<answer>(.*?)</answer>'
    content_matches = re.findall(pattern, content, re.DOTALL)
    student_answer = content_matches[-1].strip() if content_matches else content.strip()
    return student_answer

def extract_choice(text):
    # 1. Clean and normalize text
    # text = text.upper()  # Convert to uppercase
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces

    # 2. Choice should not have uppercase letters before or after
    choices = re.findall(r'(?<![A-D])([A-D])(?=[\.\,\?\!\:\;]|$)', text)
    # choices = re.findall(r'([A-D])(?=[\.\,\?\!\:\;]|$)', text)

    if not choices:
        choice =  re.findall(r'([A-D]).*', text)
        return  choice[0] if len(choice) > 0 else ''

    # 3. If only one choice, return it directly
    if len(choices) == 1 and choices[0] in ['A', 'B', 'C', 'D']:
        return choices[0]
    # elif len(choices) > 1:
    #     return choices[random.randint(0, len(choices) - 1)] # 前闭后闭。
    else:
        return text.strip()

    # 4. If multiple choices, use heuristic rules
    choice_scores = {choice: 0 for choice in choices}

    # 4.1 Keywords around choices get points
    keywords = [
        '答案', '选择', '正确', '是', '对',
        'answer', 'correct', 'choose', 'select', 'right',
        '认为', '应该', '觉得', 'think', 'believe', 'should'
    ]

    # Get context for each choice (20 chars before and after)
    for choice in choices:
        pos = text.find(choice)
        context = text[max(0, pos-20):min(len(text), pos+20)]

        # Add points for keywords
        for keyword in keywords:
            if keyword.upper() in context:
                choice_scores[choice] += 1

        # Add points if choice is near the end (usually final answer)
        if pos > len(text) * 0.7:  # In last 30% of text
            choice_scores[choice] += 2

        # Add points if followed by punctuation
        if pos < len(text) - 1 and text[pos+1] in '。.!！,，':
            choice_scores[choice] += 1

    # Return highest scoring choice
    if not choice_scores:
        return ''
    return max(choice_scores.items(), key=lambda x: x[1])[0]

def extract_choice_zh(text):
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces

    # 2. Choice should not have uppercase letters before or after
    choices = re.findall(r'(?<![A-D])([A-D])(?=[\.\,\?\!\:\;]|$)', text)
    # choices = re.findall(r'([A-D])(?=[\.\,\?\!\:\;]|$)', text)

    if not choices:
        choice =  re.findall(r'选项([A-D]).*', text)
        return  choice[0] if len(choice) > 0 else ''

    # 3. If only one choice, return it directly
    if len(choices) == 1:
        return choices[0]
    elif len(choices) > 1:
        return choices[random.randint(0, len(choices) - 1)] # 前闭后闭。
    else:
        return text.strip()
    

def MedicalEval(data_dict:list, pred_dict: list, **kwargs) -> tuple:
    tot = len(pred_dict)
    #print('tot',tot)
    succ = 0
    # print('pred_dict',pred_dict)
    for idx, (options, data) in enumerate(zip(data_dict, pred_dict)):
        print(f'{idx}/{len(data_dict)}')
        if 'vote_answer' in data:
            if data['vote_answer'].replace(' ', '') == data['gt_answer'].replace(' ', ''):
                succ += 1
                data['is_correct'] = 'yes'
                continue
            else:
                data['is_correct'] = 'no'
            continue
        try:
            # 比较选项
            target_str = extract_pred(data['generated_text'])
            pred_choice = extract_choice(target_str)
            gt_answer = extract_choice(data['answer'])
            if pred_choice.lower().strip() == gt_answer.lower().strip():
                succ += 1
                data['is_correct'] = 'yes'
                continue
            else:
                data['is_correct'] = 'no'
                
            # 比较内容
            a,b,c,d = options.get('option_A'), options.get('option_B'), options.get('option_C'), options.get('option_D')
            answer_list = [a, b]
            if c is not None:
                answer_list.append(c)
            if d is not None:
                answer_list.append(d)

            most_similar_index = find_most_similar_index(answer_list, target_str)
            if answer_list[most_similar_index].replace(' ', '') == data['gt_answer'].replace(' ', ''):
                succ += 1
                data['is_correct'] = 'yes'
                continue
            else:
                data['is_correct'] = 'no'
            
            target_str = data['generated_text'].replace('<answer>', '').replace('</answer>', ':')
            most_similar_index = find_most_similar_index(answer_list, target_str)
            if answer_list[most_similar_index].replace(' ', '') == data['gt_answer'].replace(' ', ''):
                succ += 1
                data['is_correct'] = 'yes'
                continue
            else:
                data['is_correct'] = 'no'
            
        except:
            continue
        
    return pred_dict, succ/tot

def MedicalEvalZh(data_dict:list, pred_dict: list, **kwargs) -> tuple:
    tot = len(pred_dict)
    #print('tot',tot)
    succ = 0
    # print('pred_dict',pred_dict)
    for idx, (options, data) in enumerate(zip(data_dict, pred_dict)):
        print(f'{idx}/{len(data_dict)}')
        if 'vote_answer' in data:
            if data['vote_answer'].replace(' ', '') == data['gt_answer'].replace(' ', ''):
                succ += 1
                data['is_correct'] = 'yes'
                continue
            else:
                data['is_correct'] = 'no'
            continue
        
        try:
            # 比较选项
            target_str = extract_pred(data['generated_text'])
            pred_choice = extract_choice_zh(target_str)
            gt_answer = extract_choice_zh(data['answer'])
            if pred_choice.lower().strip() == gt_answer.lower().strip():
                succ += 1
                data['is_correct'] = 'yes'
                continue
            else:
                data['is_correct'] = 'no'
            
            # 比较内容
            a,b,c,d = options.get('option_A'), options.get('option_B'), options.get('option_C'), options.get('option_D')
            answer_list = [a, b]
            if c is not None:
                answer_list.append(c)
            if d is not None:
                answer_list.append(d)

            most_similar_index = find_most_similar_index(answer_list, target_str)
            if answer_list[most_similar_index].replace(' ', '') == data['gt_answer'].replace(' ', ''):
                succ += 1
                data['is_correct'] = 'yes'
                continue
            else:
                data['is_correct'] = 'no'
        
            target_str = data['generated_text'].replace('<answer>', '').replace('</answer>', ':')
            most_similar_index = find_most_similar_index(answer_list, target_str)
            if answer_list[most_similar_index].replace(' ', '') == data['gt_answer'].replace(' ', ''):
                succ += 1
                data['is_correct'] = 'yes'
                continue
            else:
                data['is_correct'] = 'no'
        except:
            continue
    return pred_dict, succ/tot

def MedicalEvalNoTag(data_dict:list, pred_dict: list, **kwargs) -> tuple:
    tot = len(pred_dict)
    # print('tot',tot)
    tag = kwargs.pop('tag', False)
    succ = 0
    # print('pred_dict',pred_dict)
    for idx, (options, data) in enumerate(zip(data_dict, pred_dict)):
        # print(f'{idx}/{len(data_dict)}')
        if 'vote_answer' in data:
            if data['vote_answer'].replace(' ', '') == data['gt_answer'].replace(' ', ''):
                succ += 1
                data['is_correct'] = 'yes'
                continue
            else:
                data['is_correct'] = 'no'
            
        try:
            a,b,c,d = options.get('option_A'), options.get('option_B'), options.get('option_C'), options.get('option_D')
            answer_list = [a, b]
            if c is not None:
                answer_list.append(c)
            if d is not None:
                answer_list.append(d)
            # 比较选项
            
            if 'extracted_answer' in data:
                if tag:
                    target_str = extract_pred(data['extracted_answer'])
                else:
                    target_str = data['extracted_answer'].replace('<answer>', '').replace('</answer>', '')
            else:
                if tag:
                    target_str = extract_pred(data['generated_text'])
                else:
                    target_str = data['generated_text'].replace('<answer>', '').replace('</answer>', '').strip()
            # target_str = data['extracted_answer']
            pred_choice = extract_choice(target_str)
            gt_answer = extract_choice(data['answer'])
            
            if pred_choice.upper().strip() == gt_answer.upper().strip():
                succ += 1
                data['is_correct'] = 'yes'
                continue
            else:
                data['is_correct'] = 'no'

            # if 'extracted_answer' in data:
            #     if '<answer>' not in data['extracted_answer']:
            #         target_str = data['extracted_answer']
            # else:
            #     if '<answer>' not in data['generated_text']:
            #         target_str = data['generated_text']
                    
            most_similar_index = find_most_similar_index(answer_list, target_str)
            if answer_list[most_similar_index] == data['gt_answer']:
                succ += 1
                data['is_correct'] = 'yes'
                continue
            else:
                data['is_correct'] = 'no'
            
            # # 无效内容。 有问题。
            # target_str = data['generated_text']
            # most_similar_index = find_most_similar_index(answer_list, target_str)
            # if answer_list[most_similar_index] == data['gt_answer']:
            #     succ += 1
            #     data['is_correct'] = 'yes'
            #     continue
            # else:
            #     data['is_correct'] = 'no'
            
            # if 'extracted_answer' in data:
            #     target_str = data['extracted_answer']
            # else:
            #     target_str = data['generated_text']
            # most_similar_index = find_most_similar_index(answer_list, target_str)
            # if answer_list[most_similar_index] == data['gt_answer']:
            #     succ += 1
            #     data['is_correct'] = 'yes'
            #     continue
            # else:
            #     data['is_correct'] = 'no'
                
        except:
            continue

    return pred_dict, succ / tot

def MedicalEvalNoTagZh(data_dict:list, pred_dict: list, **kwargs) -> tuple:
    tot = len(pred_dict)
    # print('tot',tot)
    succ = 0
    # print('pred_dict',pred_dict)
    for idx, (options, data) in enumerate(zip(data_dict, pred_dict)):
        print(f'{idx}/{len(data_dict)}')
        if 'vote_answer' in data:
            if data['vote_answer'].replace(' ', '') == data['gt_answer'].replace(' ', ''):
                succ += 1
                data['is_correct'] = 'yes'
                continue
            else:
                data['is_correct'] = 'no'
            
        try:
            # 比较选项
            target_str = data['generated_text'].replace('<answer>', '').replace('</answer>', ':')
            pred_choice = extract_choice_zh(target_str)
            gt_answer = extract_choice_zh(data['answer'])
            if pred_choice.lower().strip() == gt_answer.lower().strip():
                succ += 1
                data['is_correct'] = 'yes'
                continue
            else:
                data['is_correct'] = 'no'
            
            # 比较内容
            a,b,c,d = options.get('option_A'), options.get('option_B'), options.get('option_C'), options.get('option_D')
            answer_list = [a, b]
            if c is not None:
                answer_list.append(c)
            if d is not None:
                answer_list.append(d)

            most_similar_index = find_most_similar_index(answer_list, target_str)
            if answer_list[most_similar_index].replace(' ', '') == data['gt_answer'].replace(' ', ''):
                succ += 1
                data['is_correct'] = 'yes'
            else:
                data['is_correct'] = 'no'
        except:
            continue
    return pred_dict, succ/tot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/work/code/model_match/local_llavaqwen2")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default='LlavaQwen2-GRPO-Med-CRC100k')
    parser.add_argument("--image-folder", type=str, default="/home/work/Images/ACRIMA")
    parser.add_argument("--question-file", type=str, default="/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/Kvasir/Kvasir-en.json")
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--no_tag", type=bool, default=True)
    parser.add_argument('--answers_base_path', type=str, 
                        default="/mnt/inaisfs/data/home/tansy_criait/VLM-R1/results/Kvasir/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest/Kvasir-en.json")
    args = parser.parse_args()
    eval_model(args, args.question_file, args.answers_base_path)
    
    # answer_files = glob.glob('/home/work/GRPO_result/*/*.json')
    # reject_answer_files = glob.glob('/home/work/GRPO_result/*/*0.*.json')
    # # eval_model(args, args.question_file, args.answers_base_path)
    # for answer_file in answer_files:
    #     if answer_file in reject_answer_files:
    #         continue
    #     eval_model(args, args.question_file, answer_file)
    #     print('file name:', answer_file)
        
    print('finish', flush=True)



