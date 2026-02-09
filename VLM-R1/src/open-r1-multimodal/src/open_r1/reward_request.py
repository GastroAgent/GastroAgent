import re
import json
from typing import Dict, List, Optional, Union
from openai import OpenAI, BadRequestError
import numpy as np

# 假设你已定义好 client
openai_api_key = 'xxxx'
openai_api_base = "http://gpu36:8000/v1"

client = OpenAI(
    base_url=openai_api_base,
    api_key=openai_api_key,
)

def math_score_generate(
    problem: str,
    response: str,
    ground_truth: str,
    scoring_criteria: str = (
        "Score 1.0 if the student's answer is mathematically equivalent to the correct answer. "
        "Score 0.0 otherwise. Only output a single float number between 0.0 and 1.0."
    ),
    model_name: str = "math-validator",
    max_tokens: int = 64,
    temperature: float = 0.0,  # deterministic for scoring
    n: int = 1,
    timeout: int = 30,
    parse_float: bool = True,
    **extra_kwargs
) -> Union[float, List[float], str]:
    """
    Use LLM to score a math response against ground truth based on given criteria.
    
    Args:
        problem: The math question.
        response: The student/model's answer.
        ground_truth: The correct answer.
        scoring_criteria: Instructions for how to score.
        model_name: Model name served by vLLM.
        max_tokens: Small (e.g., 64) since output is just a number.
        temperature: Usually 0.0 for deterministic scoring.
        n: Number of parallel responses (for ensemble scoring).
        parse_float: If True, try to extract and return float(s); else return raw text.
    
    Returns:
        If n == 1 and parse_float: float (e.g., 1.0)
        If n > 1 and parse_float: List[float]
        Otherwise: raw model output string(s)
    """
    system_prompt = "You are an expert math grader. Follow instructions precisely."
    scoring_criteria = (
        "You are grading a response to a math problem. Evaluate it on two criteria:\n\n"
        "1. **Correctness**: Is the mathematical answer correct or equivalent to the ground truth?\n"
        "2. **Language Compliance**: \n"
        "   - If the user's question is in English, the response must be entirely in English (no Chinese characters allowed).\n"
        "   - If the user's question is in Chinese, the response may include English terms (e.g., math symbols, variables like 'x', 'f(x)', 'derivative'), but should primarily be in Chinese and must not contain irrelevant Chinese when the question is English.\n\n"
        "Scoring rule:\n"
        "- Full score (1.0): Correct answer AND language follows the above rules.\n"
        "- Partial penalty: Deduct 0.4 if language violates the rule (e.g., Chinese appears in an English-question response), even if math is correct.\n"
        "- Incorrect math: Score ≤ 0.3 regardless of language.\n"
        "- Completely wrong or irrelevant: 0.0.\n\n"
        "Output ONLY a single float number between 0.0 and 1.0. Do not explain."
    )
    user_prompt = (
        f"Problem: {problem}\n"
        f"Student's Answer: {response}\n"
        f"Correct Answer: {ground_truth}\n"
        f"Scoring Criteria: {scoring_criteria}\n\n"
        "Output ONLY a single float number between 0.0 and 1.0. Do not explain."
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

    if not parse_float:
        return raw_responses[0] if n == 1 else raw_responses

    def extract_score(text: str) -> float:
        # 尝试提取第一个浮点数
        match = re.search(r"(\d*\.?\d+)", text.replace(",", ""))
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))  # clamp to [0,1]
        else:
            return 0.0  # default penalty

    scores = [extract_score(r) for r in raw_responses]

    return scores[0] if n == 1 else scores

if __name__ == '__main__':
    score = math_score_generate(
        problem="Solve for x: 2x + 3 = 7",  # 英文问题
        response="解为 x=2。",                # 错误：用了中文！
        ground_truth="2",
        model_name="math-validator",
        temperature=0.0
    )
    # 预期得分 ≈ 0.8（数学对但语言违规）

    score = math_score_generate(
        problem="求函数 f(x) = x^2 的导数",  # 中文问题
        response="The derivative is 2x.",     # 允许英文术语
        ground_truth="2x",
        model_name="math-validator"
    )
    # 预期得分 = 1.0（正确 + 语言合规）
