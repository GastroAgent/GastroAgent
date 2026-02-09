
import gc
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import random
import json
import PIL
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, GenerationConfig
import re
import torch.nn.functional as F

import sys

sys.path.append('/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1') # 服务器调试
from llava.model.llavaProcessor import LlavaProcessor
from llava.model.language_model.llava_qwen_mul_workflow import PloyLlavaLlamaForCausalLM
from trl.data_utils import maybe_apply_chat_template

model_id = '/mnt/inaisfs/data/home/tansy_criait/weights/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-75'

data_name = '十二指肠'
batch_size = 1 
model_name = 'Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-75' # 2: is_message=True


#data_path = f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/Kvasir/Kvasir-en.json'
data_path = f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/十二指肠/eval_all_llm2.json'

# lora_model_path = f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/checkpoints/{model_name}'
lora_model_path = None
model_save_path = None

data_save_path = f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/results/十二指肠_workflow1/{model_name}/{data_name}.json'
os.makedirs(f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/results/十二指肠_workflow1/{model_name}', exist_ok=True)

model_init_kwargs = {
    'attn_implementation': 'flash_attention_2',
    'torch_dtype': 'bfloat16', # float32, bfloat16 # flash attention 只支持bfloat16。
    'use_cache': True,
}

model = PloyLlavaLlamaForCausalLM.from_pretrained(
    model_id, 
    device_map = 'auto',
    **model_init_kwargs
)

if lora_model_path is not None and lora_model_path:
    # 加载 LoRA 模型并合并权重
    model = PeftModel.from_pretrained(model, lora_model_path)
    model = model.merge_and_unload()  # 合并 LoRA权重 并卸载 LoRA层 为普通 Model。

processing_class = LlavaProcessor.from_pretrained(model_id,
                            use_fast=True,
                            trust_remote_code=model_init_kwargs.get("trust_remote_code", None),
                            patch_size = 14)
print(processing_class.tokenizer.vocab_size)
processing_class.tokenizer.padding_side = 'left'
processing_class.chat_template = processing_class.tokenizer.chat_template

if model_save_path is not None and model_save_path:
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    processing_class.save_pretrained(model_save_path)


LONG_QTYPES = {
    "Modality Recognition",
    "Anatomy Identification",
}

TAIL_QTYPES = {
    "Disease Diagnosis",
}



# ====== 选项字母的 token id（A~H） ======
choice_tokens = ["A", "B", "C", "D"]
choice_ids = []
for ch in choice_tokens:
    ids = processing_class.tokenizer(ch, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        print(f"[WARN] 选项 {ch} 分成了多 token: {ids}")
    choice_ids.append(ids[0])
choice_ids = torch.tensor(choice_ids, device=model.device)  # [C], C<=8
print("choice_ids:", choice_ids.tolist())



import re

def extract_answer_letter(text: str):
    """
    从模型输出中提取答案选项字母：
    - 支持 <answer>option_D</answer>
    - 支持 <answer>Option D: Echocardiogram</answer>
    - 支持 <answer>D</answer> 或 <answer> D : ...</answer>
    大小写均可，仅匹配 A~D。
    """
    # ① 优先匹配 option + 下划线/空格 + 字母
    #    例如：
    #    <answer>option_D</answer>
    #    <answer>Option D: Echocardiogram</answer>
    m = re.search(
        r"<answer>\s*option[_\s]*([A-D])\b[^<]*</answer>",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    # ② 兜底：直接在 <answer> 标签内找首个 A~D 字母
    #    允许后面跟冒号/文字，例如：
    #    <answer>D</answer>
    #    <answer>D: something</answer>
    m = re.search(
        r"<answer>\s*([A-D])\b[^<]*</answer>",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    return None



def get_gt_option(example):
    """
    根据 gt_answer 的文本内容，反推出正确选项是 option_A / option_B / option_C / option_D。
    gt_answer 形如 "Dyed and lifted polyps"，
    在 option_A~D 中找到值相同的那个，返回对应 key（如 'option_A'）。
    """
    gt_text = example.get("gt_answer", None)
    if gt_text is None:
        return None

    for letter in "ABCDEFGHIJ":
        key = f"option_{letter}"
        if key in example and example[key] == gt_text:
            return key  # 注意：这里返回的是 'option_A' 这种形式

    # 没找到就返回 None
    return None

def normalize_letter(x: str | None):
    if x is None:
        return None
    s = str(x).strip()

    # 兼容 "option_B" 这种形式
    if s.lower().startswith("option_"):
        s = s.split("_", 1)[1]  # 取后半部分 "B"

    # 再兼容只给 "B" 或者带别的字符但包含 A~H 的情况
    for ch in s:
        if ch.upper() in "ABCDEFGH":
            return ch.upper()

    return None



dataset = json.load(open(data_path,'r'))

# 处理图像
def get_key_from_inputs(x: dict, key: str):
    ele = x.get(key, None)
    if isinstance(ele, list):
        return ele
    else:
        return [ele]

generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=True,
            temperature=0.2,
            pad_token_id= processing_class.tokenizer.pad_token_id,
)
results = []
print('data save path is:', data_save_path)
for i in range(0, len(dataset), batch_size):
    inputs = dataset[i:i+batch_size]
    print(f"Process: {i}/{len(dataset)}")

    system_prompt = ("You are a helpful Medical AI assistant and authoritative expert in the medical field. You have a solid foundation in medicine and are proficient in answering various visual questions related to medical topics. "
           "As the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, and use natural language to assist users in various tasks. "
           "As the expert, you need to make sure that your response is logical and factual, and that it conveys the authority of the expert. "
           "The visual content (most about medical imaging) will be provided with the following format: <|Image_start|>visual content<|Image_end|>.")

    for example in inputs:
        example['prompt'] = [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': example['formatted_text'], # Qwen2 纯文本提示词
            }
        ]
    prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]

    images = []
    for x in inputs:
        if "image_paths" in x and x["image_paths"] is not None:
            imgs = [PIL.Image.open(p).convert('RGB') for p in get_key_from_inputs(x, "image_paths")]
        else:
            imgs = []

        for img in imgs:
            try:
                # Ensure minimum dimensions of 28 pixels
                w, h = img.size
                if w < 28 or h < 28:
                # Calculate new dimensions maintaining aspect ratio
                    if w < h:
                        new_w = 28
                        new_h = int(h * (28/w))
                    else:
                        new_h = 28
                        new_w = int(w * (28/h))
                    img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
            except Exception as e:
                print(e)
                pass
            images.append(img)

    if len(images) > 0:
        prompt_inputs = processing_class(
            text=prompts_text,
            images=images,
            return_tensors='pt',
            padding=True,
            padding_side='left',
            add_special_tokens=False
        )
    else:
        prompt_inputs = processing_class(
            text=prompts_text,
            return_tensors='pt',
            padding=True,
            padding_side='left',
            add_special_tokens=False
        )
    prompt_inputs = prompt_inputs.to(model.device)


    # generate_returned_result = model.generate(
    #     **prompt_inputs,
    #     generation_config=generation_config
    # )
    # prompt_ids = prompt_inputs["input_ids"]
    # prompt_length =  prompt_inputs["input_ids"].size(1)

    # ========= 1) 根据题目 + 图像算 p_G(c|q) =========
    with torch.no_grad():
        # ========= 1) 先算 p_G(c|q) =========
        p_G, outputs = model.get_pG_from_inputs(
            candidate_ids=choice_ids,   # A~H 的 token id
            **prompt_inputs
        )   # [B, C]

        # ========= 2) 根据 question_type 标记 long / tail =========
        # inputs 是你后面 zip(completions, inputs, prompts_text) 用的那个列表
        is_long_list = []
        is_tail_list = []
        for inp in inputs:
            qt = inp["question_type"]
            is_long_list.append(qt in LONG_QTYPES)
            is_tail_list.append(qt in TAIL_QTYPES)

        is_long_qtype = torch.tensor(is_long_list, dtype=torch.bool, device=p_G.device)  # [B]
        is_tail_qtype = torch.tensor(is_tail_list, dtype=torch.bool, device=p_G.device)  # [B]

        # ========= 3) 调 Trigger policy（按 question_type 的 tail 来触发 rule） =========
        # 这里我们不再用 candidate 的 tail_indices，直接设为 None
        theta_u = 1.2   # 熵阈值（示例，你可以之后调）  大于这个阈值视为high_entropy
        theta_p = 0.5  # 置信度阈值（示例）  小于这个阈值视为low_confidence

        policy = model.trigger_policy_from_pG(
            p_G=p_G,
            tail_indices=None,          # ✅ 不按选项定义 tail，而是按 question_type
            theta_u=theta_u,
            theta_p=theta_p,
            ood_distance=None,          # 暂时不用 OOD，可以后面加
            theta_ood=None,
            rule_based_flag=is_tail_qtype,  # ✅ Disease Diagnosis 样本作为“规则触发”
        )

        # ========= 4) 搬到 CPU，方便后面写 json =========
        p_G_cpu        = p_G.detach().float().cpu()
        trigger_mask   = policy["trigger_mask"].detach().cpu().numpy()
        high_entropy   = policy["high_entropy"].detach().cpu().numpy()

        #max_prob = float(policy["max_prob"])
        low_confidence = policy["low_confidence"].detach().cpu().numpy()
        ood_flag       = policy["ood_flag"].detach().cpu().numpy()
        rule_flag      = policy["rule_flag"].detach().cpu().numpy()   # <== 现在就是 is_tail_qtype

        entropy_cpu    = policy["entropy"].detach().float().cpu().tolist()
        max_prob_cpu   = policy["max_prob"].detach().float().cpu().tolist()
        argmax_idx_cpu = policy["argmax_idx"].detach().cpu().tolist()


    # ========= 2) 正常生成答案（你原来的逻辑） =========
    generate_returned_result = model.generate(
        **prompt_inputs,
        generation_config=generation_config
    )
    prompt_ids = prompt_inputs["input_ids"]
    prompt_length =  prompt_inputs["input_ids"].size(1)


    completions = processing_class.batch_decode(generate_returned_result, skip_special_tokens=True)
    # print(completions[0])
    # for generated_text, input_, prompt_text in zip(completions, inputs, prompts_text):
    #     input_['generated_text'] = generated_text
    #     results.append({
    #         'generated_text': generated_text,
    #         'prompt_text': prompt_text,
    #         "gt_answer": input_['gt_answer'],
    #         "image_paths": input_['image_paths'],
    #         "question_id": input_['question_id'],
    #         "question_type": input_['question_type'],
    #         "option_A": input_['option_A'],
    #         "option_B": input_['option_B'],
    #         "option_C": input_['option_C'],
    #         "option_D": input_['option_D'],
    #         "option_E": input_['option_E'] if 'option_E' in input_ else None,
    #         "option_F": input_['option_F'] if 'option_F' in input_ else None,
    #         "option_G": input_['option_G'] if 'option_G' in input_ else None,
    #         "option_H": input_['option_H'] if 'option_H' in input_ else None,
    #     })
    #     if random.random() < 0.5:
    #         print(f'prompt_text: {prompt_text}')
    #         print(f'generated_text: {generated_text}\n')

    for b_idx, (generated_text, input_, prompt_text) in enumerate(zip(completions, inputs, prompts_text)):
        input_['generated_text'] = generated_text

        # ---- 当前样本 p_G(c|q) ----
        probs = p_G_cpu[b_idx].tolist()
        prob_dict = {}
        for j, ch in enumerate(choice_tokens):
            if j < len(probs):
                prob_dict[f"p_{ch}"] = float(probs[j])

       # ---- 从 generated_text 里抽取最终答案字母 ----
        answer_letter = extract_answer_letter(generated_text)  # 'A'~'D' 或 None

        # ==== 由 gt_answer 文本反推正确选项字母 ====
        gt_letter = get_gt_option(input_)   # 也是 'A'~'D' 或 None

        # ==== 是否答对 ====
        norm_answer = normalize_letter(answer_letter)
        norm_gt     = normalize_letter(gt_letter)

        correct = 1 if (
            norm_answer is not None
            and norm_gt is not None
            and norm_answer == norm_gt
        ) else 0

        # ==== p_chosen: 模型选的那个选项在 p_G 里的概率 ====
        p_chosen = None
        if answer_letter is not None:
            # 这里假设 choice_letters = ["A","B","C","D"]
            if answer_letter in choice_tokens:
                idx = choice_tokens.index(answer_letter)
                if idx < len(probs):
                    p_chosen = float(probs[idx])

       
        # ---- Trigger policy & question_type long/tail 标记 ----
        trig      = bool(trigger_mask[b_idx])
        trig_H    = bool(high_entropy[b_idx])
        trig_P    = bool(low_confidence[b_idx])
        
        trig_ood  = bool(ood_flag[b_idx])

        # 这两个直接由我们前面 is_long_qtype / is_tail_qtype 得到
        qt = input_["question_type"]
        is_long = (qt in LONG_QTYPES)
        is_tail = (qt in TAIL_QTYPES)
        # rule_flag[b_idx] 理论上等于 is_tail

        ent_val   = float(entropy_cpu[b_idx])
        max_p_val = float(max_prob_cpu[b_idx])
        argmax_i  = int(argmax_idx_cpu[b_idx])

         # ==== 合成置信度：max_prob + 正确性 ====
        alpha, beta = 0.7, 0.3   # 你可以自己调权重
        conf_mix = alpha * max_p_val + beta * correct
        low_confidence1 = conf_mix < theta_p
        trig_P1    = bool(low_confidence1)

        result_item = {
            'generated_text': generated_text,
            'prompt_text': prompt_text,
            "gt_answer": input_['gt_answer'],

            # === p_G 相关 ===
            "answer_letter": answer_letter,   # 模型生成的选项字母
            "gt_letter": gt_letter,           # 由 gt_answer 文本反推的正确字母
            "correct": correct,               # 0/1 是否答对
            "p_chosen": p_chosen,             # p_G(模型选的那一项)
            "entropy": ent_val,
            "max_prob": max_p_val,
            "pred_idx": argmax_i,
            "conf_mix":conf_mix,

            # === question_type 分组 ===
            "question_type": qt,
            "is_long_qtype": is_long,   # True: Modality/Anatomy
            "is_tail_qtype": is_tail,   # True: Disease Diagnosis

            # === Trigger policy ===
            "trigger": trig,                       # 是否触发 WG-Flow
            "trigger_high_entropy": trig_H,
            "trigger_low_confidence": trig_P,
            "trigger_low_confidence1": trig_P1,
            "trigger_ood": trig_ood,
            "trigger_rule_tail_qtype": bool(rule_flag[b_idx]),  # 按 Disease Diagnosis 触发

            "image_paths": input_['image_paths'],
            "question_id": input_['question_id'],
            "question_type_raw": input_['question_type'],  # 保留原始字段也可以
            "option_A": input_['option_A'],
            "option_B": input_['option_B'],
            "option_C": input_['option_C'],
            "option_D": input_['option_D'],
            "option_E": input_.get('option_E'),
            "option_F": input_.get('option_F'),
            "option_G": input_.get('option_G'),
            "option_H": input_.get('option_H'),
            "option_I": input_.get('option_I'),
            "option_J": input_.get('option_J'),
        }

        # 把各个选项的概率写进去
        result_item.update(prob_dict)
        results.append(result_item)

        if random.random() < 0.5:
            print(f'prompt_text: {prompt_text}')
            print(f'generated_text: {generated_text}')
            print(f'answer_letter: {answer_letter}, p_chosen={p_chosen}')
            print(f'choice probs: {prob_dict}')
            print(
                f'qt={qt}, is_long={is_long}, is_tail={is_tail}, '
                f'trigger={trig}, H={ent_val:.3f}, max_p={max_p_val:.3f}\n'
            )




json.dump(results, open(data_save_path,'w', encoding='utf-8'), indent=4, ensure_ascii=False)
print('Infer Done!')