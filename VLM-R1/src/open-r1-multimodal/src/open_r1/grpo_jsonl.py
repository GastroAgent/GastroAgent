# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from glob import glob
import os
import random
import re
import pathlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from babel.numbers import parse_decimal
from math_verify import parse, verify
from utils.math import compute_score
from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

# import debugpy

# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9505))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

from math_verify import parse, verify

from trainer import VLMGRPOTrainer, GRPOConfig

from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
import PIL
from Levenshtein import ratio
from utils.pycocotools.coco import COCO
from utils.pycocotools.cocoeval import COCOeval
import json
import math
from json_repair import repair_json
from typing import Dict, List, Optional, Union
from vlm_modules import *
import base64
from typing import Tuple, Union, List, Optional
from transformers.utils import logging
from transformers import AutoProcessor, AutoTokenizer
from llava.eval.model_vqa_qwen_endscopy import *

from openai import BadRequestError, OpenAI

logger = logging.get_logger(__name__)

openai_api_key = 'xxxx'
openai_api_base = "http://gpu34:8000/v1"

client = OpenAI(
    base_url=openai_api_base,
    api_key=openai_api_key,
)

tokenizer = None

def initialize_tokenizer(model_path):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    reward_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "Choose reward method: 'default', 'mcp', ..."
        },
    )
    task_type: Optional[str] = field(
        default=None,
        metadata={"help": "Choose task type: 'default', 'gui', ..."},
    )
    is_reward_customized_from_vlm_module: bool = field(
        default=False,
        metadata={"help": "Whether to use a customized reward from vlm module"},
    )
    is_message: bool = field(
        default=False,
        metadata={"help": "Use Message and Chat templete For Prompt."},
    )
    use_advantages_clip: bool = field(
        default=False,
        metadata={"help": "use advantages clip."},
    )
    advantages_clip_up: Optional[float] = field(
        default=0,
        metadata={"help": "uo for use advantages clip."},
    )
    advantages_clip_down: Optional[float] = field(
        default=0,
        metadata={"help": "down for advantages clip."},
    )
    alg: Optional[str] = field(
        default='grpo',
        metadata={"help": "down for advantages clip."},
    )
    grpo_denorm_length: Optional[int] = field(
        default=2048,
        metadata={"help": "down for advantages clip."},
    )
    bi_kl: bool = field(
        default=False,
        metadata={"help": "use advantages clip."},
    )
    prune_threshold: Optional[float] = field(
        default=0,
        metadata={"help": "down for advantages clip."},
    )
    prune_ratio: Optional[float] = field(
        default=0,
        metadata={"help": "down for advantages clip."},
    )
    use_llava_v1_conv: Optional[bool] = field(
        default=False,
        metadata={"help": "prompt for no tag."},
    )
    use_mix_prompts: Optional[bool] = field(
        default=False,
        metadata={"help": "prompt for no tag."},
    )
    decouple_adv: Optional[bool] = field(
        default=False,
        metadata={"help": "Decouple Advantages."},
    )
    high_entropy: Optional[bool] = field(
        default=False,
        metadata={"help": "只优化 高熵Token."},
    )
    use_neg_clamp: Optional[bool] = field(
        default=False,
        metadata={"help": "取消 Min 操作. "},
    )
    vllm_mode: Optional[str] = field(
        default="colocate",
        metadata={"help": "vllm的使用方式，仅支持 'colocate' 和 'server'."},
    )
    vllm_tensor_parallel_size: Optional[int] = field(
        default=1,
        metadata={"help": "每个组的 vllm 并行大小。"},
    )

def extract_choice(text):
    # 1. Clean and normalize text
    text = text.upper()  # Convert to uppercase
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces

    # 2. Choice should not have uppercase letters before or after
    choices = re.findall(r'(?<![A-D])([A-D])(?=[\.\,\?\!\:\;]|$)', text)

    # 3. If only one choice, return it directly
    if len(choices) == 1:
        return choices[0]
    # elif len(choices) > 1:
    #     return choices[random.randint(0, len(choices) - 1)] # 前闭后闭。
    else:
        return ''

def _encode_image_to_data_url(image_path: str) -> str:
    """将本地图像文件转为 data URL，供多模态模型使用。"""
    if image_path.startswith(("http://", "https://")):
        return image_path  # 直接返回 URL
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with open(image_path, "rb") as f:
        mime_type = "image/jpeg" if image_path.lower().endswith(".jpg") or image_path.lower().endswith(".jpeg") else "image/png"
        encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

def medical_diagnosis_score_generate_multimodal(
    problem: str,
    response: str,
    ground_truth: str,
    images: Optional[List[str]] = None,  # 支持多个本地路径或 URL
    scoring_criteria: str = None,
    model_name: str = "medical-validator",  # or your served Qwen-VL model name
    max_tokens: int = 512,
    temperature: float = 0.0,
    n: int = 1,
    timeout: int = 30,
    parse_float: bool = True,
    **extra_kwargs
) -> Union[float, List[float], str]:
    """
    Score a medical diagnostic response that may involve multiple images (e.g., radiographs, skin lesion photos).
    
    Args:
        problem: Clinical text description (e.g., "Patient presents with...").
        response: Model's diagnostic answer (text only).
        ground_truth: Reference diagnosis.
        images: Optional list of paths or URLs to medical images.
        ... (other args same as before)
    """
    problem = problem.replace("<|Image_start|><image><|Image_end|>\n", "")
    system_prompt = (
        "You are a board-certified physician and medical educator. "
        "Your task is to grade diagnostic responses that may include one or more medical images. "
        "Evaluate based on clinical accuracy, reasoning, safety, and whether the response appropriately interprets each provided image. "
        "Be strict: never reward hallucinated findings or missed critical visual signs."
    )

    if scoring_criteria is None:
        scoring_criteria = """You are grading a multimodal medical diagnostic response (text + potentially multiple images). Evaluate on four criteria:

0. **Diagnostic Accuracy (Prerequisite)**:
   - Final diagnosis must align with both clinical description AND visible findings in all images.
   - If any image shows a clear abnormality but response ignores it → score ≤ 0.3.
   - Hallucinating findings not present in images → severe penalty → 0.0.

1. **Clinical Reasoning Quality ("Symptoms → Diagnosis")**:
   - The reasoning must reflect sound clinical logic: consider key symptoms, risk factors, red flags, and generate appropriate differentials.
   - Critical requirements:
     • Must mention **differential diagnoses** when uncertainty exists (not just one answer).
     • Must address **red-flag symptoms** (e.g., chest pain, neurological deficits, acute bleeding) with urgency.
     • If the response outlines a diagnostic plan (e.g., "First assess..., then order..."), it must align with actual steps taken.
   - Deduct 0.2 for:
     • Missing critical differentials (e.g., ignoring MI in chest pain),
     • Overconfident single diagnosis without justification,
     • Plan–execution mismatch (stated workup not followed).
   - Minor gaps (e.g., omitting a rare but possible cause) → deduct up to 0.1.

2. **Safety & Ethical Compliance**:
   - Must emphasize: **"This is not a substitute for professional care"** or equivalent disclaimer when appropriate.
   - Must **strongly advise seeking in-person medical evaluation** for urgent/emergent symptoms.
   - Never recommend self-treatment for serious conditions (e.g., antibiotics without prescription, ignoring cancer signs).
   - Violation (e.g., says "you probably have cancer" without nuance, or "just take ibuprofen for chest pain") → deduct 0.3, possibly score ≤ 0.2.

3. **Communication Clarity & Professionalism**:
   - Use accurate medical terminology while remaining understandable to a layperson.
   - Avoid alarmist language, speculation, or absolute certainty unless justified.
   - Well-structured: clear separation of assessment, differentials, recommendations.
   - Poor structure, jargon overload, or confusing advice → deduct up to 0.1.

Scoring Rules:
- Full score (1.0): Accurate diagnosis + robust reasoning + safety-compliant + clear communication.
- Deduct cumulatively:
   • –0.3 for safety/ethical violation,
   • –0.1 to –0.2 for reasoning flaws,
   • –0.0 to –0.1 for poor communication.
- Incorrect or dangerous diagnosis → score ≤ 0.3.
- Hallucinated drugs, fake studies, or life-threatening advice → 0.0.

❗Note: A correct-sounding diagnosis with poor reasoning (e.g., ignores key symptom) or unsafe advice does NOT earn full credit.

Output ONLY a single float number between 0.0 and 1.0. Do not explain."""

    user_content = []

    text_part = (
        f"Clinical Question: {problem}\n"
        f"Model's Diagnostic Response: {response}\n"
        f"Reference Diagnosis / Gold Standard: {ground_truth}\n"
        f"Scoring Criteria: {scoring_criteria}\n\n"
        f"Note: {'Multiple images are provided below.' if images else 'No images are provided.'}"
    )
    user_content.append({"type": "text", "text": text_part})

    if images is not None:
        for image in images:
            try:
                image_url = _encode_image_to_data_url(image)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            except Exception as e:
                print(f"Warning: Failed to load image '{image}': {e}")
                pass

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    for attempt_max_tokens in [max_tokens, 32, 16]:
        try:
            # 假设client已配置好并兼容OpenAI API
            chat_outputs = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=attempt_max_tokens,
                temperature=temperature,
                n=n,
                timeout=timeout,
                extra_body=extra_kwargs.get("extra_body", {})
            )
            break
        except Exception as e:
            if "context length" in str(e).lower() and attempt_max_tokens > 16:
                continue
            else:
                raise e
    else:
        raise RuntimeError("Failed to generate after token fallback.")

    raw_responses = [choice.message.content.strip() for choice in chat_outputs.choices]
    if random.random() < 0.01:
        print("LLM as Judge Score (Multimodal Medical):", raw_responses[0])

    if not parse_float:
        return raw_responses[0] if n == 1 else raw_responses

    def extract_score(text: str) -> float:
        match = re.search(r"(\d*\.?\d+)", text.replace(",", ""))
        if match:
            try:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            except ValueError:
                return 0.0
        return 0.0

    scores = [extract_score(r) for r in raw_responses]
    return scores[0] if n == 1 else scores

def medical_diagnosis_score_generate(
    problem: str,
    response: str,
    ground_truth: str,
    scoring_criteria: str = None,  # will be overridden by medical-specific criteria
    model_name: str = "medical-validator",  # 建议使用医学微调模型，如 "meditron-7b" 或 "clinical-camel"
    max_tokens: int = 64,
    temperature: float = 0.0,  # deterministic for reliable scoring
    n: int = 1,
    timeout: int = 30,
    parse_float: bool = True,
    **extra_kwargs
) -> Union[float, List[float], str]:
    """
    Use LLM to score a medical diagnostic response against a reference standard based on clinical criteria.
    
    Args:
        problem: The patient's clinical scenario or chief complaint.
        response: The model/doctor's diagnostic answer.
        ground_truth: The correct/reference diagnosis or differential list (for context).
        scoring_criteria: (Optional) Custom criteria; if None, uses built-in medical rubric.
        model_name: Name of the served LLM (preferably medically fine-tuned).
        max_tokens: Small output (just a score).
        temperature: Set to 0.0 for deterministic grading.
        n: Number of parallel generations (for ensemble scoring).
        parse_float: If True, extract float(s); else return raw text.
    
    Returns:
        If n == 1 and parse_float: float (e.g., 0.8)
        If n > 1 and parse_float: List[float]
        Otherwise: raw model output string(s)
    """
    problem = problem.replace("<|Image_start|><image><|Image_end|>\n", "")
    # === Medical-Specific System Prompt ===
    system_prompt = (
        "You are a board-certified physician and medical educator. "
        "Your task is to grade diagnostic responses with strict adherence to clinical guidelines, patient safety, and ethical standards. "
        "Be unforgiving of hallucinations, overconfidence, missed red flags, or unsafe recommendations. "
        "Prioritize harm prevention over politeness."
    )

    # === Medical-Specific Scoring Criteria (overrides any default) ===
    if scoring_criteria is None:
        scoring_criteria = (
            """You are a senior clinical physician grading a diagnostic response to a patient's medical query. Evaluate strictly on four criteria:

0. **Diagnostic Accuracy (Prerequisite)**:
   - The final diagnosis or differential diagnosis must be medically plausible and consistent with current clinical guidelines.
   - If the response provides an incorrect primary diagnosis that contradicts standard medical knowledge → score ≤ 0.3.
   - Completely irrelevant, hallucinated, or dangerous advice (e.g., "ignore chest pain") → 0.0.

1. **Clinical Reasoning Quality ("Symptoms → Diagnosis")**:
   - The reasoning must reflect sound clinical logic: consider key symptoms, risk factors, red flags, and generate appropriate differentials.
   - Critical requirements:
     • Must mention **differential diagnoses** when uncertainty exists (not just one answer).
     • Must address **red-flag symptoms** (e.g., chest pain, neurological deficits, acute bleeding) with urgency.
     • If the response outlines a diagnostic plan (e.g., "First assess..., then order..."), it must align with actual steps taken.
   - Deduct 0.2 for:
     • Missing critical differentials (e.g., ignoring MI in chest pain),
     • Overconfident single diagnosis without justification,
     • Plan–execution mismatch (stated workup not followed).
   - Minor gaps (e.g., omitting a rare but possible cause) → deduct up to 0.1.

2. **Safety & Ethical Compliance**:
   - Must emphasize: **"This is not a substitute for professional care"** or equivalent disclaimer when appropriate.
   - Must **strongly advise seeking in-person medical evaluation** for urgent/emergent symptoms.
   - Never recommend self-treatment for serious conditions (e.g., antibiotics without prescription, ignoring cancer signs).
   - Violation (e.g., says "you probably have cancer" without nuance, or "just take ibuprofen for chest pain") → deduct 0.3, possibly score ≤ 0.2.

3. **Communication Clarity & Professionalism**:
   - Use accurate medical terminology while remaining understandable to a layperson.
   - Avoid alarmist language, speculation, or absolute certainty unless justified.
   - Well-structured: clear separation of assessment, differentials, recommendations.
   - Poor structure, jargon overload, or confusing advice → deduct up to 0.1.

Scoring Rules:
- Full score (1.0): Accurate diagnosis + robust reasoning + safety-compliant + clear communication.
- Deduct cumulatively:
   • –0.3 for safety/ethical violation,
   • –0.1 to –0.2 for reasoning flaws,
   • –0.0 to –0.1 for poor communication.
- Incorrect or dangerous diagnosis → score ≤ 0.3.
- Hallucinated drugs, fake studies, or life-threatening advice → 0.0.

❗Note: A correct-sounding diagnosis with poor reasoning (e.g., ignores key symptom) or unsafe advice does NOT earn full credit.

Output ONLY a single float number between 0.0 and 1.0. Do not explain."""
        )

    # === User Prompt Template ===
    user_prompt = (
        f"Clinical Scenario: {problem}\n"
        f"Model's Diagnostic Response: {response}\n"
        f"Reference Diagnosis / Gold Standard: {ground_truth}\n"
        f"Scoring Criteria: {scoring_criteria}\n\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Try with original max_tokens, fallback if context too long
    for attempt_max_tokens in [max_tokens, 32, 16]:
        try:
            chat_outputs = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=attempt_max_tokens,
                temperature=temperature,
                n=n,
                timeout=timeout,
                extra_body=extra_kwargs.get("extra_body", {})
            )
            break
        except BadRequestError as e:
            if "context length" in str(e).lower() and attempt_max_tokens > 16:
                continue
            else:
                raise e
    else:
        raise RuntimeError("Failed to generate due to context length even after fallback.")

    raw_responses = [choice.message.content.strip() for choice in chat_outputs.choices]
    if random.random() < 0.01:  # occasional debug print
        print("LLM as Judge Score (Medical):", raw_responses[0])
        
    if not parse_float:
        return raw_responses[0] if n == 1 else raw_responses

    def extract_score(text: str) -> float:
        # Extract first floating-point number, clamp to [0.0, 1.0]
        match = re.search(r"(\d*\.?\d+)", text.replace(",", ""))
        if match:
            try:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            except ValueError:
                return 0.0
        else:
            return 0.0  # default penalty for unparseable output

    scores = [extract_score(r) for r in raw_responses]

    return scores[0] if n == 1 else scores

def vllm_accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is correct using symbolic verification, exact string matching, or fuzzy matching."""
    solution = kwargs.pop('answer', '')  # 答案文本
    contents = [completion[0]["content"] for completion in completions]
    questions = kwargs.get('question', None) or  kwargs.get('formatted_text', None)
    images = kwargs.get('image', None) or  kwargs.get('image_paths', None)
    rewards = []
    solution += [solution[0]] * (len(contents) - len(solution))
    accu_reward_method = kwargs.get("accu_reward_method", "default")
    accu_reward_method += [accu_reward_method[0]] * (len(contents) - len(accu_reward_method))
    if images and len(images) == len(questions):
        for idx, (question, content, image, sol, accu_reward_method) in enumerate(zip(questions, contents, images, solution, accu_reward_method)):
            reward = medical_diagnosis_score_generate_multimodal(question, content, sol, image)
            if isinstance(reward, list):
                rewards.append(sum(reward) / len(reward))
            else:
                rewards.append(reward)
    else:
        for idx, (question, content, sol, accu_reward_method) in enumerate(zip(questions, contents, solution, accu_reward_method)):
            reward = medical_diagnosis_score_generate(question, content, sol)
            if isinstance(reward, list):
                rewards.append(sum(reward) / len(reward))
            else:
                rewards.append(reward)

    return rewards

def llm_reward(content, sol, **kwargs):
    # Extract answer from content if it has think/answer tags
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

    # Extract answer from content if it has think/answer tags
    content_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_matches[-1].strip() if content_matches else content.strip()
    return evaluate_answer_similarity(student_answer, ground_truth)

def cosine_reward(content, tokenizer, acc_reward, **kwargs):
    # https://arxiv.org/abs/2502.03373
    min_len_value_wrong = 0.0
    max_len_value_wrong = - 0. # 错了就不优化，"50步笑百步"是有害的。
    min_len_value_correct = acc_reward
    max_len_value_correct = acc_reward - 0.25
    cosine_max_len = 512
    cosine_min_len = 16
    # processing_class = AutoProcessor.from_pretrained(model_path)
    # tokenizer = processing_class.tokenizer
    if isinstance(content, str):
        content = [content]

    gen_len = len(tokenizer.encode(content[0]))
    is_correct = acc_reward >= 0.25

    if is_correct:
        # Swap min/max for correct answers
        min_value = max_len_value_correct
        max_value = min_len_value_correct
    else:
        min_value = min_len_value_wrong
        max_value = max_len_value_wrong

    if gen_len < cosine_max_len and gen_len > cosine_min_len:
        reward = acc_reward
    else:
        # 定义衰减的尺度，用于控制余弦衰减的速度
        scale = cosine_max_len - cosine_min_len
        if gen_len > cosine_max_len:
            # gen_len 超过最大值，计算超出部分的角度
            angle = (gen_len - cosine_max_len) * math.pi / scale
        else:
            # gen_len 小于最小值，计算不足部分的角度
            angle = (cosine_min_len - gen_len) * math.pi / scale
        reward = max_value - (max_value - min_value) * (1 - math.cos(angle)) / 2
        # reward = max_value - (max_value - min_value) * (1 - math.cos(gen_len * math.pi / cosine_max_len)) / 2

    if cosine_max_len < gen_len:
        if random.random() < 0.1:
            print("Warning: Generated length exceeds cosine_max_len: ", gen_len)
    return reward

def repetition_reward(content, **kwargs):
    max_penalty = -1.0

    if content == '':
        return 0.0

    # First, try to extract explicitly marked JSON sections
    pattern = r'```json(.*?)```'
    json_match = re.search(pattern, content, re.DOTALL)

    if json_match:
        bbox_json = json_match.group(1).strip()
    else:
        # If no explicitly marked JSON is found, try to find any possible JSON sections
        pattern = r'```(.*?)```'
        json_match = re.search(pattern, content, re.DOTALL)
        bbox_json = json_match.group(1).strip() if json_match else None

        # If still not found, try to find possible JSON array sections
        if not bbox_json:
            pattern = r'\[\s*{.*?"bbox_2d".*?"label".*?}\s*\]'
            json_match = re.search(pattern, content, re.DOTALL)
            bbox_json = json_match.group(0) if json_match else None

    # Try to parse JSON data
    if bbox_json:
        try:
            # Try direct parsing
            data = json.loads(bbox_json)
        except json.JSONDecodeError:
            try:
                # If direct parsing fails, try using json_repair to repair
                repaired_json = repair_json(bbox_json)
                data = json.loads(repaired_json)
            except:
                # If repair also fails, switch to plain text processing
                data = None
        if data and isinstance(data, list):
            # Ensure data is in list format
            try:
                # For JSON data, set ngram_size to 1
                ngram_size = 1
                # Combine 'bbox_2d' and 'label' of each object into a string
                items = []
                for item in data:
                    if 'bbox_2d' in item and 'label' in item:
                        items.append(f"{item['bbox_2d']}_{item['label']}")

                @staticmethod
                def zipngram(text: list, ngram_size: int):
                    return zip(*[text[i:] for i in range(ngram_size)])

                ngrams = set()
                total = 0

                for ng in zipngram(items, ngram_size):
                    ngrams.add(ng)
                    total += 1

                if total == 0:
                    return 0.0

                scaling = 1 - len(ngrams) / total
                reward = scaling * max_penalty

                return reward
            except KeyError:
                # If necessary keys are missing, switch to plain text processing
                pass

    # If no JSON section is found or JSON processing fails, treat as plain text
    ngram_size = 6

    if len(content.split()) < ngram_size:
        return 0.0

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    ngrams = set()
    total = 0

    for ng in zipngram(content, ngram_size):
        ngrams.add(ng)
        total += 1

    scaling = 1 - len(ngrams) / total
    reward = scaling * max_penalty

    return reward

def repetition_rewards(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        reward = repetition_reward(content)
        rewards.append(reward)

        # if os.getenv("DEBUG_MODE") == "true" and random.random() < 0.5:
        #     log_path = os.getenv("LOG_PATH")
        #     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        #     image_path = kwargs.get("image_paths")[0] if "image_paths" in kwargs else None
        #     problem = kwargs.get("problem")[0]
        #     if reward <= 0.0:  # this condition can be changed for debug
        #         with open(log_path + "_repetition.txt", "a", encoding='utf-8') as f:
        #             f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
        #             f.write(f"image_path: {image_path}\n")
        #             f.write(f"problem: {problem}\n")
        #             f.write(f"Content: {content}\n")
        #             f.write(f"Solution: {sol}\n")
    return rewards

def acc_rewards_nobox_nooption(completions, **kwargs):
    contents = [completion[0]['content'] for completion in completions]
    rewards = accuracy_reward_nobox_nooption(completions, **kwargs)

    num_refs = len(kwargs.get('ref_responses', [[]])[0])
    for i in range(len(rewards)):
        if i >= (len(rewards) - num_refs):
            rewards[i] = rewards[i] * 1
    return rewards

def acc_rewards(completions, **kwargs):
    contents = [completion[0]['content'] for completion in completions]
    rewards = accuracy_reward(completions, **kwargs)

    num_refs = len(kwargs.get('ref_responses', [[]])[0])
    for i in range(len(rewards)):
        if i >= (len(rewards) - num_refs):
            rewards[i] = rewards[i] * 1
    return rewards

def cosine_rewards(completions, **kwargs):
    contents = [completion[0]['content'] for completion in completions]
    acc_rewards = accuracy_reward(completions, **kwargs)
    rewards = []
    for content, acc_reward in zip(contents, acc_rewards):
        reward = cosine_reward(content, tokenizer, acc_reward)
        rewards.append(reward)
    num_refs = len(kwargs.get('ref_responses', [[]])[0])
    for i in range(len(rewards)):
        if i >= (len(rewards) - num_refs):
            rewards[i] = rewards[i] * 0.85
    return rewards

def numeric_reward(content, sol, **kwargs):
    content = clean_text(content)
    sol = clean_text(sol)
    try:
        content, sol = float(content), float(sol)
        return 1.0 if content == sol else 0.0
    except:
        return None

def math_reward(content, sol, **kwargs):
    content = clean_text(content)
    sol = clean_text(sol)
    return compute_score(content, sol)

def clean_text(text, exclue_chars=['\n', '\r']):
    # Extract content between <answer> and </answer> if present
    answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_matches:
        # Use the last match
        text = answer_matches[-1]

    for char in exclue_chars:
        if char in ['\n', '\r']:
            # If there is a space before the newline, remove the newline
            text = re.sub(r'(?<=\s)' + re.escape(char), '', text)
            # If there is no space before the newline, replace it with a space
            text = re.sub(r'(?<!\s)' + re.escape(char), ' ', text)
        else:
            text = text.replace(char, ' ')

    # Remove leading and trailing spaces and convert to lowercase
    return text.strip().rstrip('.').lower()

def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    most_similar_index = -1
    highest_similarity = 0.5  # 设立阈值。
    target_str = target_str.replace('option_', '')
    # Iterate through each string in the list
    for i, str_ in enumerate(str_list):
        similarity = str_similarity(str_, target_str)
        # print('similarity',similarity)

        # If the current string is more similar than the previous most similar string, update the variables
        if similarity > highest_similarity:
            most_similar_index = i
            highest_similarity = similarity

    # Return the index of the most similar string
    return most_similar_index

def all_match_reward(content, sol, **kwargs):
    content = clean_text(content)
    sol = clean_text(sol)
    return 1.0 if content == sol else 0.0

def default_accuracy_reward(content, sol, **kwargs):
    no_tag = kwargs.get("no_tag", False)
    if no_tag:
        kwargs['model_pred'] = content if content is not None else ''
        kwargs['gt_answer'] = sol
        reward = 0.0
        target_str = content
        pred_choice = extract_choice(target_str)
        gt_answer = extract_choice(sol)
        if pred_choice.lower().strip() == gt_answer.lower().strip():
            return 1.0

        a, b, c, d = kwargs.get('option_A'), kwargs.get('option_B'), kwargs.get('option_C'), kwargs.get('option_D')
        answer_list = [a, b, c, d, '']
        if answer_list[find_most_similar_index(answer_list, kwargs['model_pred'])] == kwargs['gt_answer']:
            reward = 1.0
        return reward
    # pattern = r'\\box{\[*(.*)\]*}'
    pattern = r'<answer>(.*?)</answer>'
    reward = 0.0
    # Extract answer from solution if it has think/answer tags
    try:
        sol_match = re.search(pattern, sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
    except:
        print('sol:', sol)
    # Extract answer from content if it has think/answer tags
    content_matches = re.findall(pattern, content, re.DOTALL)
    # student_answer = content_matches[-1].strip() if content_matches else content.strip().replace('<answer>', '').replace('</answer>', '')
    student_answer = content_matches[-1].strip() if content_matches else ''
    if 'within the <answer> and </answer> tags.' not in kwargs['prompts_text'] and (not student_answer):
        student_answer = content.strip().replace('<answer>', '').replace('</answer>', '').strip()

    # # Try symbolic verification first for numeric answers
    # try:
    #     answer = parse(student_answer)
    #     if float(verify(answer, parse(ground_truth))) > 0:
    #         reward = 1.0
    # except Exception:
    #     pass  # Continue to next verification method if this fails

    # If symbolic verification failed, try string matching or fuzzy matching
    if reward == 0.0:
        try:
            # Check if ground truth contains numbers
            has_numbers = bool(re.search(r'\d', ground_truth))
            # Check if it's a multiple choice question
            has_choices = extract_choice(ground_truth)

            if has_numbers:
                # For numeric answers, use exact matching
                reward = numeric_reward(student_answer, ground_truth)
                if reward is None:
                    reward = ratio(clean_text(student_answer), clean_text(ground_truth))
            elif has_choices:
                # For multiple choice, extract and compare choices
                correct_choice = has_choices.upper()
                student_choice = extract_choice(student_answer).upper()
                if student_choice and (content_matches or 'within the <answer> and </answer> tags.' not in kwargs['prompts_text']):
                    if kwargs.pop('decouple_adv', False):
                        reward = 1.0 if student_choice == correct_choice else 0.0
                        # reward = 0.5 if student_choice == correct_choice else 0.0
                    else:
                        reward = 1 if student_choice == correct_choice else 0.0

                ### 同时，考虑内容。 参考 "/home/dalhxwlyjsuo/criait_tansy/jmf/VLM-R1/src/open-r1-multimodal/src/open_r1/llava/eval/model_vqa_qwen_endscopy_with_pred.py"
                if reward == 0:
                    a, b, c, d = kwargs.get('option_A'), kwargs.get('option_B'), kwargs.get('option_C'), kwargs.get(
                        'option_D')
                    answer_list = [a, b, c, d, '']
                    str2int = {'A': 0, 'B': 1, 'C': 2, 'D': 3, }
                    gt_content = answer_list[str2int[correct_choice]]
                    most_similar_index = find_most_similar_index(answer_list, student_answer)
                    if answer_list[most_similar_index] == gt_content:
                        if kwargs.pop('decouple_adv', False):
                            # reward = 0.5
                            reward = 1.0
                        else:
                            reward = 1.0
                else:
                    a, b, c, d = kwargs.get('option_A'), kwargs.get('option_B'), kwargs.get('option_C'), kwargs.get(
                        'option_D')
                    answer_list = [a, b, c, d, '']
                    str2int = {'A': 0, 'B': 1, 'C': 2, 'D': 3, }
                    if answer_list[str2int[student_choice]] == answer_list[str2int[correct_choice]]:
                        pass
                    else:
                        reward = 0.0
            else:
                # For text answers, use fuzzy matching
                reward = ratio(clean_text(student_answer), clean_text(ground_truth))
        except Exception as e:
            print(e)
            pass  # Keep reward as 0.0 if all methods fail

    return reward

def default_accuracy_reward_nobox_nooption(content, sol, **kwargs):
    # pattern = r'\\box{\[*(.*)\]*}'
    pattern = r'<answer>(.*?)</answer>'
    reward = 0.0
    # 处理标准答案
    if '<answer>' in sol or '</answer>' in sol:
        sol_match = re.search(pattern, sol)
        ground_truth = sol_match.group(1).strip()
    else:
        ground_truth = sol.strip()
    # Extract answer from answer tags
    content_matches = re.findall(pattern, content, re.DOTALL)
    student_answer = content_matches[-1].strip() if content_matches else ''

    if reward == 0.0:
        try:
            if student_answer.lower().strip() == ground_truth.lower():
                reward = 1
            
            ### 严格的语法奖励
            if reward == 0:
                score = str_similarity(student_answer.lower().strip(), ground_truth.lower())
                # reward = score
                ## or
                reward = int(score > 0.9) * 0.25
                
        except Exception as e:
            print(e)
            pass  # Keep reward as 0.0 if all methods fail

    return reward


def accuracy_reward_nobox_nooption(completions, **kwargs):
    """Reward function that checks if the completion is correct using symbolic verification, exact string matching, or fuzzy matching."""
    solution = kwargs.pop('answer', []) 
    gt_answer = kwargs.pop('gt_answer', [])  
    solution = solution or gt_answer
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    solution += [solution[0]] * (len(contents) - len(solution))
    accu_reward_method = kwargs.get("accu_reward_method", "default")
    accu_reward_method += [accu_reward_method[0]] * (len(contents) - len(accu_reward_method))
    for idx, (content, sol, accu_reward_method) in enumerate(zip(contents, solution, accu_reward_method)):
        if accu_reward_method == 'default':
            single_kwargs = {}
            for key, value in kwargs.items():
                single_kwargs[key] = value[0]
            reward = default_accuracy_reward_nobox_nooption(content, sol, idx=idx, gt_answer=gt_answer, **single_kwargs)
            rewards.append(reward)

    return rewards

def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is correct using symbolic verification, exact string matching, or fuzzy matching."""
    solution = kwargs.pop('answer', '')  # 答案文本
    gt_answer = kwargs.pop('gt_answer', '')  # 答案选项
    solution = solution if solution is not None else gt_answer
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    solution += [solution[0]] * (len(contents) - len(solution))
    accu_reward_method = kwargs.get("accu_reward_method", "default")
    accu_reward_method += [accu_reward_method[0]] * (len(contents) - len(accu_reward_method))
    for idx, (content, sol, accu_reward_method) in enumerate(zip(contents, solution, accu_reward_method)):
        # if accu_reward_method is defined, use the corresponding reward function, otherwise use the default reward function
        if accu_reward_method == 'llm':
            reward = llm_reward(content, sol)
        elif accu_reward_method == 'math':
            reward = math_reward(content, sol)
        elif accu_reward_method == 'weighted_sum':
            clean_content = clean_text(content)
            sol = clean_text(sol)
            if sol == "none":
                if clean_content == "none":
                    reward = 1.0
                else:
                    reward = 0.0
        elif accu_reward_method == 'all_match':
            reward = all_match_reward(content, sol)
        else:
            single_kwargs = {}
            for key, value in kwargs.items():
                single_kwargs[key] = value[0]
            reward = default_accuracy_reward(content, sol, no_tag=False, idx=idx, gt_answer=gt_answer, **single_kwargs)
        rewards.append(reward)

        # if os.getenv("DEBUG_MODE") == "true" and random.random() < 0.5:
        #     log_path = os.getenv("LOG_PATH")
        #     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        #     image_path = kwargs.get("image_paths")[0] if "image_paths" in kwargs else None
        #     problem = kwargs.get("problem")[0]
        #     prompt_text = kwargs.get("prompt_text")[0]
        #     if reward <= 1.0:  # this condition can be changed for debug
        #         with open(log_path, "a", encoding='utf-8') as f:
        #             f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
        #             f.write(f"accu_reward_method: {accu_reward_method}\n")
        #             f.write(f"image_path: {image_path}\n")
        #             f.write(f"problem: {problem}\n")
        #             f.write(f"prompt_text: {prompt_text}\n")
        #             f.write(f"Content: {content}\n")
        #             f.write(f"Solution: {sol}\n")
    return rewards

def accuracy_reward_no_tag(completions, **kwargs):
    """Reward function that checks if the completion is correct using symbolic verification, exact string matching, or fuzzy matching."""
    solution = kwargs.pop('answer', '')  # 答案选项
    gt_answer = kwargs.pop('gt_answer', '')  # 答案文本

    solution = gt_answer
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for idx, (content, sol, accu_reward_method) in enumerate(zip(contents, solution, kwargs.get("accu_reward_method"))):
        # if accu_reward_method is defined, use the corresponding reward function, otherwise use the default reward function
        if accu_reward_method == "mcq":
            raise NotImplementedError
            reward = mcq_reward(content, sol)
        elif accu_reward_method == 'yes_no':
            raise NotImplementedError
            reward = yes_no_reward(content, sol)
        elif accu_reward_method == 'llm':
            raise NotImplementedError
            reward = llm_reward(content, sol)
        elif accu_reward_method == 'map':
            raise NotImplementedError
            reward = map_reward(content, sol)
        elif accu_reward_method == 'math':
            raise NotImplementedError
            reward = math_reward(content, sol)
        elif accu_reward_method == 'weighted_sum':
            raise NotImplementedError
            clean_content = clean_text(content)
            sol = clean_text(sol)
            if sol == "none":
                if clean_content == "none":
                    reward = 1.0
                else:
                    reward = 0.0
            else:
                reward = detection_score(clean_content, sol)
        elif accu_reward_method == 'od_ap':
            raise NotImplementedError
            reward = od_reward(content, sol)
        elif accu_reward_method == 'od_ap50':
            raise NotImplementedError
            reward = od_reward(content, sol, score_type=1)
        elif accu_reward_method == 'odLength':
            raise NotImplementedError
            reward = odLength_reward(content, sol)
        elif accu_reward_method == 'all_match':
            raise NotImplementedError
            reward = all_match_reward(content, sol)
        else:
            single_kwargs = {}
            for key, value in kwargs.items():
                single_kwargs[key] = value[idx]
            reward = default_accuracy_reward(content, sol, no_tag=True, idx=idx, **single_kwargs)
        rewards.append(reward)

        # if os.getenv("DEBUG_MODE") == "true" and random.random() < 0.5:
        #     log_path = os.getenv("LOG_PATH")
        #     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        #     image_path = kwargs.get("image_paths")[0] if "image_paths" in kwargs else None
        #     problem = kwargs.get("problem")[0]
        #     prompt_text = kwargs.get("prompt_text")[0]
        #     if reward <= 1.0:  # this condition can be changed for debug
        #         with open(log_path, "a", encoding='utf-8') as f:
        #             f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
        #             f.write(f"accu_reward_method: {accu_reward_method}\n")
        #             f.write(f"image_path: {image_path}\n")
        #             f.write(f"problem: {problem}\n")
        #             f.write(f"prompt_text: {prompt_text}\n")
        #             f.write(f"Content: {content}\n")
        #             f.write(f"Solution: {sol}\n")
    return rewards

def over_length_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r'<answer>(.*?)</answer>'
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [True if len(re.findall(pattern, content, re.DOTALL)) == 1 else False for content in completion_contents]
    reward = 1.0
    rewards = [reward if match else 0.0 for match in matches]

    ### </answer> 结束惩罚
    for idx, completion in enumerate(completions):
        include_answer = '<answer>' in completion[0]['content'] and '</answer>' in completion[0]['content']
        if include_answer:
            neg_text = completion[0]['content'].split('</answer>')[1:]
            neg_length = ''.join(neg_text)
            neg_reward = min(len(neg_length) * 0.01, 1)
            rewards[idx] = rewards[idx] - neg_reward
        else:
            rewards[idx] = rewards[idx] - 2
    return rewards

def new_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    reward = 1.0

    pattern = r'<answer>(.*?)</answer>'
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [True if len(re.findall(pattern, content, re.DOTALL)) == 1 else False for content in completion_contents]

    rewards = [reward if match else 0.0 for match in matches]

    ### </answer> 结束惩罚
    for idx, completion in enumerate(completions):
        neg_text = completion[0]['content'].split('</answer>')[1:]
        neg_length = ''.join(neg_text)
        neg_reward = len(neg_length) / (len(completion[0]['content']) + 1e-4)
        rewards[idx] = rewards[idx] - neg_reward * reward

    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = r'\\box{\[*(.*)\]*}'
    if kwargs.pop('decouple_adv', [False])[0]:
        reward = 1.0
    else:
        # reward = 0.5
        reward = 1.0

    pattern = r'<answer>(.*?)</answer>'
    completion_contents = [completion[0]["content"] for completion in completions]
    if '<answer>' not in kwargs['prompts_text'][0]:
        return [reward if '<answer>' not in x and '</answer>' not in x else 0 for x in completion_contents]
    matches = [True if len(re.findall(pattern, content, re.DOTALL)) > 0 else False for content in completion_contents]

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = [reward if match else 0.0 for match in matches]

    for idx, completion in enumerate(completions):
        neg_text = completion[0]['content'].split('</answer>')[1:]
        neg_length = ''.join(neg_text)
        neg_reward = len(neg_length) / (len(completion[0]['content']) + 1e-4)
        rewards[idx] = rewards[idx] - neg_reward * reward

    return rewards


# def length_reward(completions, target_language='en', **kwargs):
#     if kwargs.pop('decouple_adv', [False])[0]:
#         reward = 0.5
#     else:
#         reward = 0.5
#     return [reward if len(x) < 5000 else 0 for x in completions]

def lang_reward(completions, target_language='en', **kwargs):
    if kwargs.pop('decouple_adv', [False])[0]:
        reward = 1.0
    else:
        # reward = 0.5
        reward = 1.0

    # 根据目标语言设置检测规则
    if target_language == 'en':
        pattern = r'[\u4e00-\u9fff]'  # 检测中文字符（基本汉字）
    elif target_language == 'zh':
        pattern = r'[a-zA-Z]'  # 检测英文字符（拉丁字母）
    else:
        # 其他语言暂不支持，全部返回 0.0
        return [0.0 for _ in completions]

    # 提取所有 completion 的内容
    completion_contents = [completion[0]["content"] for completion in completions]

    # 判断每个内容是否包含不应有的语言
    matches = [re.search(pattern, content) is not None for content in completion_contents]

    # 计算奖励值（符合要求得 1，否则 0.0）
    rewards = [reward if not match else 0.0 for match in matches]
    return rewards

from reward_func import xmlcount_reward_func, soft_format_reward_func, strict_format_reward_func, int_reward_func, correctness_boxreward_func
reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "acc": acc_rewards,
    "acc_nobox_nooption": acc_rewards_nobox_nooption,
    "format": format_reward,
    'over_length_reward': over_length_reward,
    "new_format": new_format_reward,
    "lang": lang_reward,
    "acc_length": cosine_rewards,
    "repetition": repetition_rewards,
    "accuracy_no_tag": accuracy_reward_no_tag,
    "xml": xmlcount_reward_func,
    'soft_format': soft_format_reward_func,
    "strict_format": strict_format_reward_func,
    'int': int_reward_func,
    'correct': correctness_boxreward_func,
    'vllm_reward': vllm_accuracy_reward
}


@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False
    freeze_mm_mlp_adapter: bool = False
    freeze_backbone: bool = False
    backbone_layer_id: int = -1

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def get_vlm_module(model_name_or_path):
    if "llavaqwen" in model_name_or_path.lower():
        return LLAVAQwenModule
    elif "llava-qwen" in model_name_or_path.lower():
        return LLAVAQwenModule
    elif "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

def main(script_args, training_args, model_args):
    print('=' * 100)
    print('Base Model is: ', model_args.model_name_or_path)
    print('Model Output is: ', training_args.output_dir)
    print('=' * 100)
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)

    # Get reward functions
    if script_args.is_reward_customized_from_vlm_module:
        reward_funcs = [vlm_module_cls.select_reward_func(func, script_args.task_type) for func in
                        script_args.reward_funcs]
    else:
        reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the JSONL datasets
    import json
    from datasets import Dataset

    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")

    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")

    if script_args.reward_method is None:
        accu_reward_methods = ["default"] * len(data_files)
    else:
        accu_reward_methods = script_args.reward_method.split(":")
        assert len(accu_reward_methods) == len(
            data_files), f"Number of reward methods must match number of data files: {len(accu_reward_methods)} != {len(data_files)}"

    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")

    all_data = []
    files = []
    for data_file in data_files:
        if os.path.isdir(data_file):
            files.extend(glob(os.path.join(data_file, '*.json')))
        else:
            files.append(data_file)
    for file in files:
        with open(file, 'r') as f:
            all_data.extend(json.load(f))

    dataset = Dataset.from_list(all_data)
    def make_conversation_from_jsonl(example, old=False):
        if 'image' in example:
            example['image_paths'] = example['image']
        content = example['formatted_text']
        if old:
            return {
                'image_paths': [p for p in example['image_paths']],  # Store path instead of loaded image
                'problem': example['question'],
                'answer': example['answer'],
                "gt_answer": example['gt_answer'],
                'accu_reward_method': example['accu_reward_method'],
                'prompt': [
                    {'role': 'system',
                     'content': "You are a helpful medical AI assistant.",
                     },
                    {
                        'role': 'user',
                        'content': content,
                        # 'content': [
                        #     *({'type': 'image', 'text': example['image_paths'][i]} for i in range(len(example['image_paths']))),
                        #     {'type': 'text', 'text': example['formatted_text']}
                        # ]
                    }]
            }

        if 'image' in example and example['image'] is not None:
            # assert all(os.path.exists(p) for p in example['image_paths']), f"Image paths do not exist: {example['image_paths']}"
            # Don't load image here, just store the path
            if isinstance(example['image'], str):
                example['image'] = [example['image']]
            return {
                'image_paths': [p for p in example['image']],  # Store path instead of loaded image
                'problem': example.get('question', content),
                'answer': example['answer'],
                "gt_answer": example['gt_answer'],
                'accu_reward_method': example.get('accu_reward_method', 'default'),
                'prompt': [
                    {
                        'role': 'system',
                        'content': (
                            "You are a helpful Medical AI assistant and authoritative expert in the medical field. You have a solid foundation in medicine and are proficient in answering various visual questions related to medical topics. "
                            "\nAs the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, and use natural language to assist users in various tasks. "
                            "\nAs the expert, you need to make sure that your response is logical and factual, and that it conveys the authority of the expert. "
                            "\nThe visual content (most about medical imaging) will be provided with the following format: <|Image_start|>visual content<|Image_end|>.")
                    },
                    {
                        'role': 'user',
                        'content': content,
                    }]
            }

        elif 'image_paths' in example and example['image_paths'] is not None:
            # assert all(os.path.exists(p) for p in example['image_paths']), f"Image paths do not exist: {example['image_paths']}"
            # Don't load image here, just store the path
            if isinstance(example['image_paths'], str):
                example['image_paths'] = [example['image_paths']]
            return {
                'image_paths': [p for p in example['image_paths']],  # Store path instead of loaded image
                'problem': example['question'],
                'answer': example['answer'],
                "gt_answer": example['gt_answer'],
                'accu_reward_method': example['accu_reward_method'],
                'prompt': [
                    {
                        'role': 'system',
                        'content': (
                            "You are a helpful Medical AI assistant and authoritative expert in the medical field. You have a solid foundation in medicine and are proficient in answering various visual questions related to medical topics. "
                            "\nAs the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, and use natural language to assist users in various tasks. "
                            "\nAs the expert, you need to make sure that your response is logical and factual, and that it conveys the authority of the expert. "
                            "\nThe visual content (most about medical imaging) will be provided with the following format: <|Image_start|>visual content<|Image_end|>.")
                    },
                    {
                        'role': 'user',
                        'content': content,
                    }]
            }

        # elif 'image_paths' in example and example['image_paths'] is not None and 'llavaqwen' in model_args.model_name_or_path:
        #     # assert all(os.path.exists(p) for p in example['image_paths']), f"Image paths do not exist: {example['image_paths']}"
        #     # Don't load image here, just store the path
        #     return {
        #         'image_paths': [p for p in example['image_paths']],  # Store path instead of loaded image
        #         'problem': example['question'],
        #         'answer': example['answer'],
        #         "gt_answer": example['gt_answer'],
        #         'accu_reward_method': example['accu_reward_method'],
        #         'prompt': [
        #             {'role': 'system',
        #              'content': ("You are a helpful Medical AI assistant and an expert in the area of the Digestive Tract and Human Stomach. "
        #  "As the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, and use natural language to assist users in various tasks. "
        #  "As the expert, you need to make sure that your response is logical and factual, and that it conveys the authority of the expert. "
        #  "The visual content will be provided with the following format: <|Image_start|>visual content<|Image_end|>.")
        #              },
        #             {
        #             'role': 'user',
        #             'content': content
        #             # [
        #             #     # *({'type': 'image', 'text': example['image_paths'][i]} for i in range(len(example['image_paths']))),
        #             #     {'type': 'text', 'text': example['formatted_text']}
        #             # ]
        #         }]
        #     }
        # else:
        #     return {
        #         'problem': example['question'],
        #         'answer': example['answer'],
        #         "gt_answer": example['gt_answer'],
        #         'accu_reward_method': example['accu_reward_method'],
        #         'prompt': [
        #             {'role': 'system',
        #              'content': ("You are a helpful Medical AI assistant and an expert in the area of the Digestive Tract and Human Stomach. "
        #  "As the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, and use natural language to assist users in various tasks. "
        #  "As the expert, you need to make sure that your response is logical and factual, and that it conveys the authority of the expert. "
        #  "The visual content will be provided with the following format: <|Image_start|>visual content<|Image_end|>.")
        #              },
        #             {
        #             'role': 'user',
        #             'content': content
        #             # 'content': [
        #             #     {'type': 'text', 'text': example['formatted_text']}
        #             # ]
        #         }]
        #     }

    # Map the conversations
    dataset = dataset.map(make_conversation_from_jsonl)

    # Split dataset for validation if requested
    splits = {'train': dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio
        )
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']

    # Select trainer class based on vlm_trainer argument
    trainer_cls = VLMGRPOTrainer
    print("using trainer:", trainer_cls.__name__)
    initialize_tokenizer(model_args.model_name_or_path)
    extra_kwargs = {
        'alg': getattr(script_args, 'alg', 'grpo'),
        'grpo_denorm_length': getattr(script_args, 'grpo_denorm_length', 4096),  # G x O
        'bi_kl': getattr(script_args, 'bi_kl', False),
        'patch_size': getattr(script_args, 'patch_size', 14),
        'prune_threshold': getattr(script_args, 'prune_threshold', 0),
        'prune_ratio': getattr(script_args, 'prune_ratio', 0),
        'is_message': getattr(script_args, 'is_message', False),
        "advantages_clip_up": getattr(script_args, 'advantages_clip_up', 0),
        "advantages_clip_down": getattr(script_args, 'advantages_clip_down', 0),
        "use_advantages_clip": getattr(script_args, 'use_advantages_clip', False),
        "decouple_adv": getattr(script_args, 'decouple_adv', False),
        "high_entropy": getattr(script_args, 'high_entropy', False),
        "vllm_tensor_parallel_size": getattr(script_args, 'vllm_tensor_parallel_size', 1),
        "vllm_mode": getattr(script_args, 'vllm_mode', 'colocate'),
        # "decouple_adv": False
    }

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        freeze_mm_mlp_adapter=model_args.freeze_mm_mlp_adapter,
        freeze_backbone=model_args.freeze_backbone,
        backbone_layer_id=model_args.backbone_layer_id,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        model_args=model_args,
        torch_dtype="bfloat16",
        resume_from_checkpoint=training_args.resume_from_checkpoint,
        **extra_kwargs
    )

    ### Train and push the model to the Hub
    if list(pathlib.Path(training_args.resume_from_checkpoint).glob("checkpoint-*")) or os.path.exists(
            training_args.resume_from_checkpoint):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    print('训练完成')
    model = trainer.model
    model.config.use_cache = True

    # model.config.save_pretrained(training_args.output_dir)
    # # 判断是否是 PEFT 模型

    # processing_class = trainer.processing_class
    # processing_class.save_pretrained(training_args.output_dir)

    if True:
        # state_dict = get_peft_state_maybe_zero_3(
        #     model.named_parameters(), 'none'
        # )
        # non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
        #     model.named_parameters()
        # )
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), 'all'
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(state_dict, os.path.join(training_args.output_dir, 'lora_trainables.bin'))
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))

        # # Save and push to hub
        # trainer.save_model(training_args.output_dir)
        # # trainer.save_state(training_args.output_dir)
        # if training_args.push_to_hub:
        #     trainer.push_to_hub()

    print('Done.')


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if training_args.deepspeed and "zero3" in training_args.deepspeed:
        print("zero3 is used, qwen2_5vl forward monkey patch is applied")
        # monkey_patch_qwen2_5vl_forward()
    main(script_args, training_args, model_args)
