import json
import os

def merge_specific_pairs(tasks):
    """
    按需合并指定的文件对。
    
    Args:
        tasks: 列表，每个元素为 (file1, file2, output) 的元组
    """
    for i, (f1, f2, out) in enumerate(tasks):
        print(f"\n任务 [{i+1}/{len(tasks)}]:")
        
        # 检查文件是否存在
        if not os.path.exists(f1) or not os.path.exists(f2):
            print(f"  跳过：文件路径不存在\n  F1: {f1}\n  F2: {f2}")
            continue

        try:
            # 读取数据
            with open(f1, 'r', encoding='utf-8') as file1:
                data1 = json.load(file1)
            with open(f2, 'r', encoding='utf-8') as file2:
                data2 = json.load(file2)

            # 合并逻辑
            if isinstance(data1, list) and isinstance(data2, list):
                merged = data1 + data2
                mode = "List拼接"
            elif isinstance(data1, dict) and isinstance(data2, dict):
                merged = data1.copy()
                merged.update(data2)
                mode = "Dict合并"
            else:
                merged = [data1, data2]
                mode = "混合结构（转列表）"

            # 自动创建输出目录
            os.makedirs(os.path.dirname(out), exist_ok=True)

            # 保存
            with open(out, 'w', encoding='utf-8') as f_out:
                json.dump(merged, f_out, ensure_ascii=False, indent=4)
            
            print(f"  成功！模式: {mode}")
            print(f"  保存至: {out}")

        except Exception as e:
            print(f"  处理失败: {e}")

# ================= 配置区域 =================

# 在这里手动指定你想合并的 Pair
# 格式: (文件1, 文件2, 输出文件)
tasks_to_run = [
    (
        '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/result_tsy/kvasir_419_doctor/disease_self_exam_json_endovit_image_hint_resnetmodel_neighbor_nfree-tsy1/eval_by_x0_clean.json',
        '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/result_tsy/kvasir_doctor/image_hint_oldattn_Disease_new_extra_50000_tsy_final_img/eval_by_x0_clean.json',
        '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/result_tsy/kvasir_all_doctor/eval_by_x0/95/eval_by_x0_clean_all.json'
    ),
    (
        '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/result_tsy/kvasir_419_doctor/disease_self_exam_json_endovit_image_hint_resnetmodel_neighbor_nfree-tsy2/eval_by_x0_clean.json',
        '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/result_tsy/kvasir_doctor/image_hint_oldattn_Disease_new_extra_50000_tsy_final_img1/eval_by_x0_clean.json',
        '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/result_tsy/kvasir_all_doctor/eval_by_x0/94/eval_by_x0_clean_all.json'
    ),
    (
        '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/result_tsy/kvasir_419_doctor/disease_self_exam_json_endovit_image_hint_resnetmodel_neighbor_nfree-tsy4/eval_by_x0_clean.json',
        '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/result_tsy/kvasir_doctor/image_hint_oldattn_Disease_new_extra_50000_tsy_final_img2/eval_by_x0_clean.json',
        '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/result_tsy/kvasir_all_doctor/eval_by_x0/94_1/eval_by_x0_clean_all.json'
    ),
    (
        '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/result_tsy/kvasir_419_doctor/disease_self_exam_json_endovit_image_hint_resnetmodel_neighbor_nfree-tsy5/eval_by_x0_clean.json',
        '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/result_tsy/kvasir_doctor/image_hint_oldattn_Disease_new_extra_50000_tsy_final_img4/eval_by_x0_clean.json',
        '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/result_tsy/kvasir_all_doctor/eval_by_x0/94_2/eval_by_x0_clean_all.json'
    ),
    (
        '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/result_tsy/kvasir_419_doctor/disease_self_exam_json_endovit_image_hint_resnetmodel_neighbor_nfree-tsy6/eval_by_x0_clean.json',
        '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/result_tsy/kvasir_doctor/image_hint_oldattn_Disease_new_extra_50000_tsy_final_img5/eval_by_x0_clean.json',
        '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/result_tsy/kvasir_all_doctor/eval_by_x0/94_3/eval_by_x0_clean_all.json'
    ),
]

if __name__ == "__main__":
    merge_specific_pairs(tasks_to_run)