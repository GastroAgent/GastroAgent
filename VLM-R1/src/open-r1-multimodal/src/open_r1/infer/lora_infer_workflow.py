
import gc
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import random
import json
import PIL
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, GenerationConfig
import sys

sys.path.append('/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1') # 服务器调试
from llava.model.llavaProcessor import LlavaProcessor
from llava.model.language_model.llava_qwen_mul_workflow import PloyLlavaLlamaForCausalLM
from trl.data_utils import maybe_apply_chat_template

model_id = '/mnt/inaisfs/data/home/tansy_criait/weights/LlavaQwen2-GRPO-Tricks-Total-CoT-6000'

data_name = 'Kvasir-en'
batch_size = 1 
model_name = 'LlavaQwen2-GRPO-Tricks-Total-CoT-6000' # 2: is_message=True
is_message = True
llava_v1_prompt_post = ''
use_llava_v1_prompt = False

if use_llava_v1_prompt:
    is_message = False
    llava_v1_prompt_post = '-no-tag'

data_path = f'/mnt/inaisfs/data/home/tansy_criait/wass_flow_match/data/{data_name}.json'

# lora_model_path = f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/checkpoints/{model_name}'
lora_model_path = None
model_save_path = None

data_save_path = f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/results/kvasir_workflow2/{model_name}/{data_name}.json'
os.makedirs(f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/results/kvasir_workflow2/{model_name}', exist_ok=True)

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
if is_message:
    processing_class.chat_template = processing_class.tokenizer.chat_template

if model_save_path is not None and model_save_path:
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    processing_class.save_pretrained(model_save_path)

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

def get_gt_option(example):
    """
    根据 gt_answer 的文本内容，反推出正确选项是 option_A / option_B / option_C / option_D。
    gt_answer 形如 "Dyed and lifted polyps"，
    在 option_A~D 中找到值相同的那个，返回对应 key（如 'option_A'）。
    """
    gt_text = example.get("gt_answer", None)
    if gt_text is None:
        return None

    for letter in "ABCD":
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


print('data save path is:', data_save_path)
for i in range(0, len(dataset), batch_size):
    inputs = dataset[i:i+batch_size]
    print(f"Process: {i}/{len(dataset)}")

    system_prompt = (("You are a helpful Medical AI assistant and an authoritative expert in the area of the Upper Digestive Tract and Digestive System, skilled in medical imaging analysis and disease diagnosis. "
           "\nAs the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, and use natural language to assist users in various tasks. "
           "\nAs the expert, you need to make sure that your response is logical and factual, and that it conveys the authority of the expert. "
           "\nThe visual content (most about medical imaging) will be provided with the following format: <|Image_start|>visual content<|Image_end|>.")
                     if 'easy' not in data_name else "You are a helpful medical AI assistant.")

    # system_prompt = (("You are a helpful Medical AI assistant and authoritative expert in the medical field. You have a solid foundation in medicine and are proficient in answering various visual questions related to medical topics. "
    #        "As the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, and use natural language to assist users in various tasks. "
    #        "As the expert, you need to make sure that your response is logical and factual, and that it conveys the authority of the expert. "
    #        "The visual content (most about medical imaging) will be provided with the following format: <|Image_start|>visual content<|Image_end|>.")
    #                  if 'easy' not in data_name else "You are a helpful medical AI assistant.")
    if is_message:
        for example in inputs:
            example['prompt'] = [
                {'role': 'system',
                 'content': system_prompt
                 },
                {
                'role': 'user',
                'content': example['formatted_text'], # Qwen2 纯文本提示词
                }
            ]
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
    elif use_llava_v1_prompt:
        prompts_text = [example['llava_v1_prompt_text'] for example in inputs]
    else:
        prompts_text = [example['prompt_text'] for example in inputs]

    images = []
    for x in inputs:
        if "image" in x and x["image"] is not None:
            imgs = [PIL.Image.open(p).convert('RGB') for p in get_key_from_inputs(x, "image")]
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
                    images.append(img)
            except Exception as e:
                print(e)
                pass
            

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
    # raw_prompt_inputs = prompt_inputs.clone()
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

    generate_returned_result = model.generate(
        **prompt_inputs,
        generation_config=generation_config,
        batch_num_images=[len(images)]
    )
    prompt_ids = prompt_inputs["input_ids"]
    prompt_length =  prompt_inputs["input_ids"].size(1)

    completions = processing_class.batch_decode(generate_returned_result, skip_special_tokens=True)
    # print(completions[0])

    for b_idx, (generated_text, input_, prompt_text) in enumerate(zip(completions, inputs, prompts_text)):
        input_['generated_text'] = generated_text

        # ---- 当前样本 p_G(c|q) ----
        probs = p_G_cpu[b_idx].tolist()
        prob_dict = {}
        for j, ch in enumerate(choice_tokens):
            if j < len(probs):
                prob_dict[f"p_{ch}"] = float(probs[j])

      

        # ==== 由 gt_answer 文本反推正确选项字母 ====
        gt_letter = get_gt_option(input_)   # 也是 'A'~'D' 或 None
        # ==== 是否答对 ====
        norm_gt     = normalize_letter(gt_letter)


       
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
        #conf_mix = alpha * max_p_val + beta * correct
        #low_confidence1 = conf_mix < theta_p
        #trig_P1    = bool(low_confidence1)


        result_item = {
            'generated_text': generated_text,
            'prompt_text': prompt_text,
            "answer": input_['answer'],
            "gt_answer": input_['gt_answer'],
            # === p_G 相关 ===
        
            "gt_letter": gt_letter,           # 由 gt_answer 文本反推的正确字母
            
            
            "entropy": ent_val,
            "max_prob": max_p_val,
            "pred_idx": argmax_i,
            #"conf_mix":conf_mix,

            # === question_type 分组 ===
            "question_type": qt,
            "is_long_qtype": is_long,   # True: Modality/Anatomy
            "is_tail_qtype": is_tail,   # True: Disease Diagnosis

            # === Trigger policy ===
            "trigger": trig,                       # 是否触发 WG-Flow
            "trigger_high_entropy": trig_H,
            "trigger_low_confidence": trig_P,
            #"trigger_low_confidence1": trig_P1,
            "trigger_ood": trig_ood,
            "trigger_rule_tail_qtype": bool(rule_flag[b_idx]),  # 按 Disease Diagnosis 触发

            "image_paths": input_['image_paths'],
            "question_id": input_['question_id'],
            "question_type_raw": input_['question_type'],  # 保留原始字段也可以
            "option_A": input_['option_A'],
            "option_B": input_['option_B'],
            "option_C": input_['option_C'],
            "option_D": input_['option_D'],
        }

        # 把各个选项的概率写进去
        result_item.update(prob_dict)
        results.append(result_item)

        if random.random() < 0.05:
            print(f'prompt_text: {prompt_text}')
            print(f'generated_text: {generated_text}')
            print(f'choice probs: {prob_dict}')
            print(
                f'qt={qt}, is_long={is_long}, is_tail={is_tail}, '
                f'trigger={trig}, H={ent_val:.3f}, max_p={max_p_val:.3f}\n'
            )

json.dump(results, open(data_save_path,'w', encoding='utf-8'), indent=4, ensure_ascii=False)
print('Infer Done!')