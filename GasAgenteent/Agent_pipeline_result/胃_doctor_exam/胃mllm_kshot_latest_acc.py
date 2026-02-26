#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

# ===== 1. 配置批量任务列表 =====
# 格式: (触发样本JSON, 包含x0的JSON, 输出合并JSON)
TASKS = [
    (
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New/kshot/new_eval_tsy_llm_trig_final_true.json",
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New/kshot/eval_by_x0_clean.json",
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New/胃mllm_kshot_latest_acc.json"
    ),
    (
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New1/kshot/new_eval_tsy_llm_trig_final_true.json",
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New1/kshot/eval_by_x0_clean.json",
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New1/胃mllm_kshot_latest_acc.json"
    ),
    (
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New2/kshot/new_eval_tsy_llm_trig_final_true.json",
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New2/kshot/eval_by_x0_clean.json",
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New2/胃mllm_kshot_latest_acc.json"
    ),
    (
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New3/kshot/new_eval_tsy_llm_trig_final_true.json",
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New3/kshot/eval_by_x0_clean.json",
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New3/胃mllm_kshot_latest_acc.json"
    ),
    (
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New4/kshot/new_eval_tsy_llm_trig_final_true.json",
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New4/kshot/eval_by_x0_clean.json",
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New4/胃mllm_kshot_latest_acc.json"
    ),
    (
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New5/kshot/new_eval_tsy_llm_trig_final_true.json",
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New5/kshot/eval_by_x0_clean.json",
        "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/胃_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New5/胃mllm_kshot_latest_acc.json"
    )
    # (
    #     "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/kvasir_doctor_exam/cot-419-v5/kshot/new_eval_tsy_llm_trig_final_true.json",
    #     "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/kvasir_doctor_exam/cot-419-v5/kshot/eval_by_x0_clean.json",
    #     "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/kvasir_doctor_exam/cot-419-v5/kshot/十二指肠mllm_kshot_latest_acc.json"
    # ),
    # 您可以在此继续添加更多组
    # ( "/path/to/trig_B.json", "/path/to/x0_B.json", "/path/to/merged_B.json" ),
]
# =============================

def load_as_list(path: Path):
    """把顶层为 list 或 dict 的 json 统一转成 list[dict]."""
    if not path.exists():
        print(f"警告: 找不到文件 {path}")
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    elif isinstance(data, dict):
        return [v for v in data.values() if isinstance(v, dict)]
    return []

def run_merge_task(json1_path, json2_path, out_path):
    p1, p2, po = Path(json1_path), Path(json2_path), Path(out_path)
    
    data1 = load_as_list(p1)
    data2 = load_as_list(p2)
    
    if data1 is None or data2 is None:
        return

    # ---- 1. 建立 x0 映射 ----
    x0_map = {}
    for obj in data2:
        x0 = obj.get("x0")
        if x0:
            x0_map.setdefault(x0, []).append(obj)

    # ---- 2. 匹配合并 ----
    merged = []
    for obj1 in data1:
        paths = obj1.get("image_paths", [])
        candidate_paths = [paths] if isinstance(paths, str) else (paths if isinstance(paths, list) else [])

        for p in candidate_paths:
            if p in x0_map:
                for obj2 in x0_map[p]:
                    merged_obj = dict(obj1)
                    # 更新字段，排除需要保留的核心字段
                    for k, v in obj2.items():
                        if k in ("question_id", "gt_answer", "image_paths", "conf_mix2"):
                            continue
                        merged_obj[k] = v
                    merged_obj["matched_x0"] = p
                    merged.append(merged_obj)
                break 

    # ---- 3. 自动创建目录并保存 ----
    po.parent.mkdir(parents=True, exist_ok=True)
    with po.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    
    print(f"成功: {p1.name} + {p2.name} -> {po.name} (合并数: {len(merged)})")

def main():
    print(f"开始执行批量合并任务，共 {len(TASKS)} 组...")
    for j1, j2, out in TASKS:
        try:
            run_merge_task(j1, j2, out)
        except Exception as e:
            print(f"失败: 处理 {j1} 时发生错误: {e}")
    print("所有任务处理完毕。")

if __name__ == "__main__":
    main()