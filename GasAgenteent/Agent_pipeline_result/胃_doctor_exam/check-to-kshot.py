import json
import os
from pathlib import Path
from collections import Counter

# ===== 1. 配置输入输出文件对照表 =====
# 格式: (输入文件完整路径, 输出文件完整路径)
# 这样你可以处理不在同一个文件夹下的任何文件
FILE_PAIRS = [
    (
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New/new_eval_tsy_llm_final.json",
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New/kshot/new_eval_tsy_llm_trig_final_true.json"
    ),
    (
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New1/new_eval_tsy_llm_final.json",
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New1/kshot/new_eval_tsy_llm_trig_final_true.json"
    ),
    (
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New2/new_eval_tsy_llm_final.json",
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New2/kshot/new_eval_tsy_llm_trig_final_true.json"
    ),
    (
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New3/new_eval_tsy_llm_final.json",
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New3/kshot/new_eval_tsy_llm_trig_final_true.json"
    ),
    (
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New4/new_eval_tsy_llm_final.json",
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New4/kshot/new_eval_tsy_llm_trig_final_true.json"
    ),
    (
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New5/new_eval_tsy_llm_final.json",
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New5/kshot/new_eval_tsy_llm_trig_final_true.json"
    ),
    # 你可以根据需要继续添加...
]

# ======================================

def process_single_pair(input_path_str, output_path_str):
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)

    # 1. 检查输入是否存在
    if not input_path.exists():
        print(f"跳过：找不到输入文件 -> {input_path}")
        return

    # 2. 确保输出文件夹存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. 读取数据
    with input_path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"跳过：读取 {input_path.name} 失败: {e}")
            return

    # 兼容 dict 或 list 格式
    records = data if isinstance(data, list) else data.get("results", list(data.values()))

    # 4. 核心逻辑：过滤 trigger_final 为真的样本
    rows = [r for r in records if "trigger_final" in r and "question_id" in r]
    
    counter_trigger_true = Counter()
    trigger_samples = []

    for r in rows:
        label = r.get("gt_answer", "Unknown")
        # 只要 trigger_final 是 True/1 等真值
        if r.get("trigger_final"):
            counter_trigger_true[label] += 1
            trigger_samples.append({
                "question_id": r.get("question_id"),
                "gt_answer": label,
                "image_paths": r.get("image_paths", []),
            })

    # 5. 写入输出
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(trigger_samples, f, ensure_ascii=False, indent=2)

    print(f"成功处理: {input_path.name}")
    print(f"    -> 提取了 {len(trigger_samples)} 条触发样本至: {output_path}")
    if counter_trigger_true:
        print(f"    -> 统计: {dict(counter_trigger_true)}")
    print("-" * 50)

def main():
    print(f"开始批量处理，共计 {len(FILE_PAIRS)} 个任务...")
    for in_p, out_p in FILE_PAIRS:
        process_single_pair(in_p, out_p)
    print("所有指定的任务已完成。")

if __name__ == "__main__":
    main()