#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import Counter

# ===== 1. 路径自己改这里 =====
JSON_PATH = Path(
    "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/食管_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New3/new_eval_tsy_llm_final.json"
)

OUT_JSON = Path(
   "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/食管_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New3/kshot/new_eval_tsy_llm_final_true.json"
)
# ======================================

with JSON_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)

# 顶层可能是 dict 或 list，简单兼容一下
if isinstance(data, dict):
    records = data.get("results", list(data.values()))
else:
    records = data

# 只保留有 conf_mix2 & trigger_low_confidence2 & question_id 的样本
rows = [
    r for r in records
    if "trigger_final" in r and "question_id" in r
]

print(f"有效样本数: {len(rows)}")

# ===== 2. 统计每个 label 在 trigger_low_confidence2 == True 时的数量 =====
counter_trigger_true = Counter()
trigger_samples = []   # 用于写到新 json 的简化对象列表

for r in rows:
    label = r.get("gt_answer", "Unknown")
    flag = bool(r.get("trigger_final"))

    if flag:
        counter_trigger_true[label] += 1

        # 只保留你关心的字段（可以按需增减）
        trigger_samples.append({
            "question_id": r.get("question_id"),
            "gt_answer": label,
            "image_paths": r.get("image_paths", []),
        
        })

# ===== 3. 打印统计结果 =====
print("\n各 label 中 trigger_final == True 的数量：")
for label, cnt in counter_trigger_true.most_common():
    print(f"{label}: {cnt}")

print("\n总计触发数:", sum(counter_trigger_true.values()))

# ===== 4. 写出触发样本到新的 JSON =====
with OUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(trigger_samples, f, ensure_ascii=False, indent=2)

print(f"\n已将 trigger_final == True 的样本（共 {len(trigger_samples)} 条）")
print(f"写入到: {OUT_JSON}")
