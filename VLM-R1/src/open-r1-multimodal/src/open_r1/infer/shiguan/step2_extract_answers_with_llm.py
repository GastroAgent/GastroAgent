"""
================================================================================
步骤2: LLM提取答案
================================================================================
功能: 使用LLM从生成的文本中提取标准化答案
输入: new_eval_tsy_llm_with_trigger.json (步骤1的输出)
输出: new_eval_tsy_llm_extracted.json (添加 extracted_answer 字段)

说明:
- 使用更强的LLM（如GPT-4/Gemini）来理解复杂的生成文本
- 提取标准化的答案字母（A/B/C/D）
- 处理各种边缘情况和格式变化
"""

import json
import os
import re
from typing import Optional
import time

# ===== 配置参数 =====
data_name = '食管'
input_data_path = f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/食管_doctor_exam/Llava-Qwen2-7B-tune-med-SFT/new_eval_tsy_llm_with_trigger.json'
output_data_path = f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/食管_doctor_exam/Llava-Qwen2-7B-tune-med-SFT/new_eval_tsy_llm_extracted.json'

# LLM API配置（根据实际使用的API调整）
USE_LLM_API = True  # 设置为False则使用规则提取
LLM_API_TYPE = "gemini"  # 可选: "openai", "gemini", "anthropic"
LLM_API_KEY = os.getenv("GEMINI_API_KEY")  # 从环境变量读取
LLM_MODEL = "gemini-1.5-flash"  # 或 "gpt-4o-mini", "claude-3-haiku"

os.makedirs(os.path.dirname(output_data_path), exist_ok=True)

# ===== LLM客户端初始化 =====
if USE_LLM_API and LLM_API_TYPE == "gemini":
    try:
        import google.generativeai as genai
        genai.configure(api_key=LLM_API_KEY)
        llm_model = genai.GenerativeModel(LLM_MODEL)
        print(f"已初始化 Gemini 模型: {LLM_MODEL}")
    except Exception as e:
        print(f"Gemini初始化失败: {e}")
        print("将使用规则提取方法")
        USE_LLM_API = False

elif USE_LLM_API and LLM_API_TYPE == "openai":
    try:
        from openai import OpenAI
        client = OpenAI(api_key=LLM_API_KEY)
        print(f"已初始化 OpenAI 模型: {LLM_MODEL}")
    except Exception as e:
        print(f"OpenAI初始化失败: {e}")
        print("将使用规则提取方法")
        USE_LLM_API = False

# ===== 答案提取函数 =====
def extract_answer_by_rule(text: str) -> Optional[str]:
    """
    基于规则的答案提取（后备方案）
    """
    # 优先提取 <answer> 标签内的内容
    m = re.search(r"<answer>\s*option[_\s]*([A-H])\b[^<]*</answer>", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"<answer>\s*([A-H])\b[^<]*</answer>", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 尝试提取最后出现的选项字母
    matches = re.findall(r"\b([A-H])\b", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    return None


def extract_answer_with_llm(generated_text: str, question_text: str, options: dict) -> Optional[str]:
    """
    使用LLM提取答案（主要方法）
    """
    if not USE_LLM_API:
        return extract_answer_by_rule(generated_text)

    # 构建选项文本
    options_text = "\n".join([f"{k.split('_')[1]}: {v}" for k, v in options.items() if v is not None])

    # 构建提示词
    prompt = f"""Please analyze the following medical VQA model response and extract the answer option letter.

Question: {question_text}

Available Options:
{options_text}

Model Response:
{generated_text}

Requirements:
1. Extract ONLY the option letter (A, B, C, or D) that the model selected as the final answer
2. Look for explicit answer markers like <answer>, "The answer is", "I choose", etc.
3. If multiple letters appear, prioritize the one in the answer section
4. If no clear answer is found, return "UNKNOWN"
5. Return ONLY the letter, nothing else

Extracted Answer:"""

    try:
        if LLM_API_TYPE == "gemini":
            response = llm_model.generate_content(prompt)
            answer_text = response.text.strip()

        elif LLM_API_TYPE == "openai":
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a precise answer extraction assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )
            answer_text = response.choices[0].message.content.strip()

        else:
            return extract_answer_by_rule(generated_text)

        # 提取字母
        match = re.search(r"\b([A-H])\b", answer_text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

        if "UNKNOWN" in answer_text:
            return None

        return extract_answer_by_rule(generated_text)

    except Exception as e:
        print(f"LLM提取失败: {e}, 使用规则提取")
        return extract_answer_by_rule(generated_text)


# ===== 主处理流程 =====
print(f"正在读取数据: {input_data_path}")
with open(input_data_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

print(f"数据集大小: {len(dataset)}")
print("开始提取答案...\n")

results = []
success_count = 0
fail_count = 0

for i, item in enumerate(dataset):
    if (i + 1) % 50 == 0:
        print(f"进度: {i+1}/{len(dataset)}")

    # 准备选项字典
    options = {
        k: item.get(k)
        for k in ['option_A', 'option_B', 'option_C', 'option_D',
                  'option_E', 'option_F', 'option_G', 'option_H']
        if item.get(k) is not None
    }

    # 提取答案
    generated_text = item.get('generated_text', '')
    question_text = item.get('formatted_text', '')

    if USE_LLM_API:
        extracted_answer = extract_answer_with_llm(generated_text, question_text, options)
        time.sleep(0.1)  # 避免API限流
    else:
        extracted_answer = extract_answer_by_rule(generated_text)

    # 记录结果
    if extracted_answer:
        success_count += 1
    else:
        fail_count += 1

    # 添加字段
    item['extracted_answer'] = extracted_answer
    results.append(item)

    # 随机打印调试信息
    if i < 5 or (extracted_answer is None and fail_count <= 10):
        print(f"\n--- 样本 {i+1} ---")
        print(f"生成文本片段: {generated_text[:200]}...")
        print(f"提取答案: {extracted_answer}")
        print(f"Ground Truth: {item.get('gt_answer', 'N/A')}")
        print("----------------\n")

# ===== 保存结果 =====
print(f"\n正在保存结果到: {output_data_path}")
with open(output_data_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print('=' * 80)
print('步骤2完成！')
print(f'成功提取: {success_count}/{len(dataset)} ({success_count/len(dataset)*100:.1f}%)')
print(f'提取失败: {fail_count}/{len(dataset)} ({fail_count/len(dataset)*100:.1f}%)')
print(f'输出文件: {output_data_path}')
print('=' * 80)
