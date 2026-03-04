#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Any

# ================== 1) 路径配置：改这里 ==================
IN_JSON  = Path("./wass_flow_match_tsy/data_tsy1/食管_doctor/final_doctor_exam.json")
OUT_JSON = Path("./wass_flow_match_tsy/data_tsy1/食管_doctor/final_doctor_exam_flat.json")

# 若你希望把路径前缀整体替换成新的目录结构（可选）
# 例如把 data_tsy/final_eval_img -> data/十二指肠/final_eval1
# 例如把 data_tsy/support      -> data/十二指肠/support
PATH_REPLACE = [
    # ("/mnt/.../data_tsy/final_eval_img", "/mnt/.../data/十二指肠/final_eval1"),
    # ("/mnt/.../data_tsy/support",       "/mnt/.../data/十二指肠/support"),
]

# 兜底随机选择 support 图像时是否固定随机种子（方便复现实验）
RANDOM_SEED = 0

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# =========================================================

def apply_replace(p: str) -> str:
    for a, b in PATH_REPLACE:
        p = p.replace(a, b)
    return p

def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def list_images_cached(dir_path: str, cache: Dict[str, List[str]]) -> List[str]:
    if dir_path in cache:
        return cache[dir_path]
    d = Path(dir_path)
    if not d.is_dir():
        cache[dir_path] = []
        return cache[dir_path]
    imgs = [str(p) for p in d.iterdir() if is_img(p)]
    imgs.sort()
    cache[dir_path] = imgs
    return imgs

def main():
    random.seed(RANDOM_SEED)

    data = json.loads(IN_JSON.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("输入 JSON 顶层不是 list，请检查文件格式。")

    out: List[Dict[str, Any]] = []
    img_cache: Dict[str, List[str]] = {}

    warn_missing = 0
    warn_len = 0

    for item in data:
        x0 = item["x0"]
        qid = item.get("question_id")

        label_A = item.get("label_A")
        label_A_id = item.get("label_A_id")

        x1_dirs = item.get("x1_dirs", [])
        x1_labels = item.get("x1_labels", [])
        captions = item.get("caption", [])
        label_B_ids = item.get("label_B_ids", None)  # 有些文件可能没有

        # 统一把 caption 视为 list[str]
        if isinstance(captions, str):
            captions = [captions] * len(x1_labels)

        n = min(len(x1_dirs), len(x1_labels), len(captions))
        if n == 0:
            continue
        if not (len(x1_dirs) == len(x1_labels) == len(captions)):
            warn_len += 1

        # x0 文件名
        x0_base = os.path.basename(x0)

        for i in range(n):
            label_B = x1_labels[i]
            cap = captions[i]
            x1_dir = x1_dirs[i]

            # label_B_id
            if isinstance(label_B_ids, list) and i < len(label_B_ids):
                label_B_id = label_B_ids[i]
            else:
                label_B_id = item.get("ys", [None]*n)[i] if isinstance(item.get("ys"), list) and i < len(item["ys"]) else None

            # 默认：support 里用同名文件
            x1 = os.path.join(x1_dir, x0_base)

            # 如果不存在，兜底：随机挑一张 support 图
            if not os.path.exists(x1):
                cand = list_images_cached(x1_dir, img_cache)
                if cand:
                    x1 = random.choice(cand)
                    warn_missing += 1
                else:
                    # 该目录没有图片，直接跳过这一条
                    warn_missing += 1
                    continue

            # 可选：路径替换
            x0_new = apply_replace(x0)
            x1_new = apply_replace(x1)

            out.append({
                "question_id": qid,
                "x0": x0_new,
                "label_A": label_A,
                "label_A_id": label_A_id,
                "label_B": label_B,
                "label_B_id": label_B_id,
                "caption": cap,
                "x1": x1_new,
                "hint_path": x1_new,   # 按你要求：hint_path = x1
            })

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] in : {IN_JSON}")
    print(f"[DONE] out: {OUT_JSON}")
    print(f"[INFO] original records: {len(data)}")
    print(f"[INFO] new records     : {len(out)}")
    if warn_len:
        print(f"[WARN] {warn_len} records had length mismatch among x1_dirs/x1_labels/caption; used min length.")
    if warn_missing:
        print(f"[WARN] {warn_missing} pairs: x1 same-name file missing (used random fallback or skipped).")

if __name__ == "__main__":
    main()
