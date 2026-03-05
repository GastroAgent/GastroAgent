import json
import os
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# 你的原始 result.json
in_path = '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/result_tsy/doctor/image_hint_胃_step4_75000_doctor_exam_wass6/result.json'
with open(in_path, 'r', encoding='utf-8') as f:
    wass_dataset = json.load(f)

# 1) 按 x0_path 分组
grouped = defaultdict(list)
for item in tqdm(wass_dataset, desc="group by x0_path"):
    x0_path = item.get("x0_path") or item.get("x0")  # 你这个文件叫 x0_path
    if x0_path is None:
        # 没有就跳过
        continue

    label_A = item.get("label_A")
    label_B = item.get("label_B")
    if label_A is None or label_B is None:
        continue
      
    # 这里就是你之前的“相邻帧去偏求和”
    #nb_all = item["Neighbor_distances_sinkhorn_w2_latent_mapped_all"] 
    #bias_all = item["distances_sinkhorn_w2_latent_mapped_bias1"] 
    #dist_val = sum([x - y for x, y in zip(nb_all, bias_all)]) 
    #dist_val = sum(nb_all) 
    #dist_val = item['distances_sinkhorn_w2_image'] 
    #dist_val = item['Neighbor_distances_sinkhorn_w2_image'] 
    dist_val = item['distances_sinkhorn_w2_latent_mapped'] 

    grouped[x0_path].append({
        "label_A": label_A,
        "label_B": label_B,
        "dist": dist_val,
    })

# 2) 对每个 x0 选距离均值最小的 label_B
overall_correct = 0
overall_total = 0
per_label = defaultdict(lambda: {"correct": 0, "total": 0})
records = []

for x0, items in grouped.items():
    # 同一张图的 GT
    gt = items[0]["label_A"]
    if not os.path.exists(x0):
        continue
    # 同一张图下，同一个 label_B 可能出现多次 → 先求均值
    tmp = defaultdict(list)
    for it in items:
        tmp[it["label_B"]].append(it["dist"])
    labelB_mean = {lb: float(np.min(dlist)) for lb, dlist in tmp.items()}

    # 选距离最小的那个 label_B
    pred_label, pred_dist = min(labelB_mean.items(), key=lambda x: x[1])

    is_correct = int(pred_label == gt)
    overall_total += 1
    overall_correct += is_correct

    per_label[gt]["total"] += 1
    per_label[gt]["correct"] += is_correct

    records.append({
        "x0": x0,
        "gt": gt,
        "pred": pred_label,
        "dist": pred_dist,
        "correct": bool(is_correct),
    })

# 3) 打印结果
acc = overall_correct / overall_total if overall_total else 0.0
print(f"Overall accuracy: {acc:.4f} ({overall_correct}/{overall_total})")

print("\nPer-label accuracy:")
per_label_acc = {}
for lab, c in sorted(per_label.items(), key=lambda x: x[0]):
    la = c["correct"] / c["total"] if c["total"] else 0.0
    per_label_acc[lab] = la
    print(f"{lab:30s} {la:.4f} ({c['correct']}/{c['total']})")

# 4) 可选：保存一份
save_dir = '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/result_tsy/doctor/image_hint_胃_step4_75000_doctor_exam_wass6/tar_img'
os.makedirs(save_dir, exist_ok=True)
with open(os.path.join(save_dir, 'eval_by_x0_clean.json'), 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
with open(os.path.join(save_dir, 'summary_by_x0_clean.json'), 'w', encoding='utf-8') as f:
    json.dump(
        {
            "overall": {
                "accuracy": acc,
                "correct": overall_correct,
                "total": overall_total,
            },
            "per_label": {
                lab: {
                    "accuracy": per_label_acc[lab],
                    "correct": per_label[lab]["correct"],
                    "total": per_label[lab]["total"],
                }
                for lab in per_label_acc
            },
        },
        f,
        ensure_ascii=False,
        indent=2
    )
