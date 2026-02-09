import re
from typing import List
from math_utils import parse_answer

def extract_xml_answer(text: str, **kwargs) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# Reward functions
def correctness_reward_func(completions, **kwargs) -> list[float]:
    answer = kwargs.pop('answer')
    if isinstance(completions[0], str):
        responses = completions
    else:
        responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [1.0 if r == a else 0. for r, a in zip(extracted_responses, answer)]

def correctness_boxreward_func(completions,  **kwargs) -> list[float]:
    model = kwargs.pop('model', None)
    answers = kwargs.pop('answer')
    if model is not None:
        raise NotImplementedError('Not Implemented.')
        return
    if isinstance(completions[0], str):
        responses = completions
    else:
        responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # for prompt, ans, response, extracted_response in zip(prompts, answer, responses, extracted_responses):
    #     q = prompt[-1]['content']
        # print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{ans}", f"\nResponse:\n{response}", f"\nExtracted:\n{extracted_response}")
    return [1.0 if match_answer(parse_answer(r), a) else 0 for r, a in zip(extracted_responses, answers)]
 
def match_answer(pred: str, gt: str) -> bool:
    if gt == pred:
        return True
    pred = pred.strip().lower()
    gt = gt.strip().lower()
    prex = ['$', '[','{']
    post = [']', '}','%', '.', ',']
    for repl in prex + post:
        pred = pred.replace(repl, '')
        gt = gt.replace(repl, '')
        if gt == pred:
            return True
    try:
        pred = float(pred)
        gt = float(gt)
        return gt == pred
    except:
        pass
    return pred == gt

def int_reward_func(completions, **kwargs) -> list[float]:
    if isinstance(completions[0], str):
        responses = completions
    else:
        responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.2 if r.isdigit() else 0.0 for r in extracted_responses]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    spec_format = kwargs.pop('kwargs', 'default')
    if spec_format == 'no_format':
        return [ 0.0 for _ in completions]
    # pattern = r"<reasoning>(.*?)</reasoning>(.*?)<answer>(.*?)</answer>"
    pattern = r"<answer>(.*?)</answer>"
    if isinstance(completions[0], str):
        responses = completions
    else:
        responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses] # match 是开头就要匹配
    return [0.5 if match else 0.0 for match in matches]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r'\\box[ed]*{(.*)?}' 
    if isinstance(completions[0], str):
        responses = completions
    else:
        responses = [completion[0]["content"] for completion in completions]
    matches = [re.findall(pattern, r) for r in responses]
    return [0.5 if not len(match) == 0 else 0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>") == 1:
        count += 0.25
    if text.count("</reasoning>") == 1:
        count += 0.25
    if text.count("<answer>") == 1:
        count += 0.125
    #     # count -= len(text.split("</answer>")[-1]) * 0.001
    if text.count("</answer>") == 1:
        count += 0.125
    if text.count("</answer>") > 1:
        count -= (len(text.split("</answer>")[0]) - 1) * 0.005 # 结束后依然生成的 惩罚！！！
    
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    if isinstance(completions[0], str):
        contents = completions
    else:
        contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]