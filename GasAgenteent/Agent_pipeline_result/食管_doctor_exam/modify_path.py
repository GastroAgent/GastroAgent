import json
import os

def update_single_json_path(input_file, output_file):
    """
    修改单个 JSON 文件中的路径字段。
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # 读取指定文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 遍历处理路径
        for item in data:
            if 'x0' in item:
                path_parts = item['x0'].split('/')
                # 逻辑：倒数第3层替换为 "食管_doctor"
                if len(path_parts) >= 3:
                    path_parts[-3] = "食管_doctor"
                    item['x0'] = "/".join(path_parts)

        # 写入指定的输出路径
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"处理成功！文件已保存至: {output_file}")

    except Exception as e:
        print(f"处理失败: {e}")

# --- 配置路径 ---
input_path = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/食管_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New5/kshot/eval_by_x0_clean5.json'
output_path = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/食管_doctor_exam/Llava-Qwen2-7B-tune-med-SFT-RL-New5/kshot/eval_by_x0_clean_fixed5.json'

# 执行
update_single_json_path(input_path, output_path)

