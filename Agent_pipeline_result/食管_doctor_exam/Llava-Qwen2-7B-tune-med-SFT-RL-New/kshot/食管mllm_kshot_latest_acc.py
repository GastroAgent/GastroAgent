#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

# ===== 自己改这里 =====
JSON1 = Path("/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/食管_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New/kshot/new_eval_tsy_llm_final_true.json")   # 含 question_id / image_paths 的文件
JSON2 = Path("/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/食管_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New/kshot/eval_by_x0_clean_fixed.json")  # 含 x0 的文件
OUT   = Path("/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/食管_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New/kshot/食管mllm_kshot_latest_acc.json") # 输出匹配后的新文件
# =====================



def load_as_list(path: Path):
    """把顶层为 list 或 dict 的 json 统一转成 list[dict]."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    elif isinstance(data, dict):
        # 如果是 dict，就把 values 当成对象列表
        return [v for v in data.values() if isinstance(v, dict)]
    else:
        raise TypeError(f"{path} 顶层结构必须是 list 或 dict")

data1 = load_as_list(JSON1)
data2 = load_as_list(JSON2)

# ---- 1. 从第二个 json 中收集所有 x0 → 对象 列表 ----
x0_map = {}
for obj in data2:
    x0 = obj.get("x0")
    if not x0:
        continue
    # 允许理论上一条 x0 对应多个对象，所以用 list 存
    x0_map.setdefault(x0, []).append(obj)

print(f"第二个文件中可用 x0 数量: {len(x0_map)}")

# ---- 2. 遍历第一个 json，按 image_paths 与 x0 匹配并合并 ----
merged = []
for obj1 in data1:
    paths = obj1.get("image_paths", [])

    # 兼容 image_paths 既可能是 list 也可能是 str
    if isinstance(paths, str):
        candidate_paths = [paths]
    elif isinstance(paths, list):
        candidate_paths = paths
    else:
        candidate_paths = []

    # 遍历该样本的所有 image_paths，看有没有在 x0_map 里
    for p in candidate_paths:
        if p in x0_map:
            for obj2 in x0_map[p]:
                # 合并两个 json 对象：先放 obj1，再覆盖/追加 obj2 的字段
                merged_obj = dict(obj1)
                # 如果你不想让 obj2 覆盖已有字段，可以挑选性 update
                for k, v in obj2.items():
                    # 这里保留 question_id / gt_answer 等，不被覆盖
                    if k in ("question_id", "gt_answer", "image_paths", "conf_mix2"):
                        continue
                    merged_obj[k] = v

                # 为了明确，这里再放一个匹配使用的路径
                merged_obj["matched_x0"] = p

                merged.append(merged_obj)
            # 一般一个 image_paths 对应一个 x0，这里可以 break 也可以不 break
            # 如果可以一个 image_paths 对应多个 x0，就不要 break
            break  # 如果只要第一个匹配，就保留 break

print(f"匹配并合并后的对象数量: {len(merged)}")

# ---- 3. 写出新的 json 文件 ----
with OUT.open("w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"已将合并结果写入: {OUT}")
