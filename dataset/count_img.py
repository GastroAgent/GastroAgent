#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

# ===== 1) 改成你的根目录 =====
ROOT = Path("/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/data_tsy1/suppport_img")   # 例如：/mnt/data_tsy/final_eval_img

# ===== 2) 图片后缀 =====
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def count_imgs_in_dir(d: Path, recursive: bool = True) -> int:
    it = d.rglob("*") if recursive else d.iterdir()
    return sum(1 for p in it if is_img(p))

def main():
    root = ROOT.resolve()
    if not root.exists():
        raise FileNotFoundError(f"ROOT not found: {root}")

    subdirs = [p for p in root.iterdir() if p.is_dir()]

    stats = []
    total = 0
    for d in subdirs:
        c = count_imgs_in_dir(d, recursive=True)  # 改 False 则只统计该文件夹的第一层
        stats.append((d.name, c))
        total += c

    stats.sort(key=lambda x: x[1], reverse=True)

    print(f"Root: {root}")
    print(f"Subfolders: {len(subdirs)}")
    print("-" * 60)
    for name, c in stats:
        print(f"{name}\t{c}")
    print("-" * 60)
    print(f"TOTAL\t{total}")

if __name__ == "__main__":
    main()
