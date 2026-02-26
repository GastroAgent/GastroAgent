import json
import os

def process_json_add_gt_letter(input_path, output_path):
    """
    读取JSON文件，根据 gt_answer 匹配 option_X，并添加 gt_letter 字段。
    """
    # 1. 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误: 找不到输入文件 {input_path}")
        return

    # 2. 读取 JSON 文件
    print(f"正在读取文件: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取JSON失败: {e}")
        return

    # 3. 统一数据格式（处理单个对象或对象列表）
    if isinstance(data, dict):
        items = [data]  # 如果是单个对象，包装成列表处理
        is_list = False
    else:
        items = data    # 如果已经是列表，直接使用
        is_list = True

    # 4. 遍历处理
    count = 0
    total = len(items)
    
    for i, item in enumerate(items):
        gt_answer = item.get('gt_answer')
        
        # 如果没有 gt_answer，跳过该条数据
        if not gt_answer:
            continue
            
        found = False
        # 遍历该对象中所有的 key，寻找以 "option_" 开头的 key
        # 这里使用 list(item.keys()) 是为了避免在迭代时修改字典
        for key in list(item.keys()):
            # 检查 key 是否是 option_A, option_B... 格式，且值不为空
            if key.startswith("option_") and item[key] is not None:
                # 对比内容 (转为字符串并去除首尾空格，防止格式差异)
                val_str = str(item[key]).strip()
                gt_str = str(gt_answer).strip()
                
                if val_str == gt_str:
                    # 提取字母。例如 "option_D" -> split("_") -> ["option", "D"] -> 取 "D"
                    letter = key.split("_")[-1]
                    item['gt_letter'] = letter
                    found = True
                    break # 找到匹配项后，停止遍历当前对象的其他 option
        
        if found:
            count += 1
        else:
            # 如果没找到匹配项，可以在控制台打印警告（可选）
            # print(f"[警告] ID {item.get('question_id', i)}: 未找到与 '{gt_answer}' 匹配的选项")
            pass

    # 5. 还原数据结构并保存
    final_data = items if is_list else items[0]

    try:
        # 自动创建输出目录（如果不存在）
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
        
        print("-" * 30)
        print(f"处理完成！")
        print(f"总数据量: {total}")
        print(f"成功匹配并添加 gt_letter: {count}")
        print(f"结果已保存至: {output_path}")
        
    except Exception as e:
        print(f"保存文件失败: {e}")

# ================= 配置区域 =================

if __name__ == "__main__":
    # 1. 请将此处修改为您真实的 JSON 输入文件路径
    input_json_file = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/食管_doctor_exam/new_eval_tsy_llm.json'

    # 2. 请设置处理后的文件保存路径
    output_json_file = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/食管_doctor_exam/new_eval_tsy_llm_with_letter.json'

    # 运行
    process_json_add_gt_letter(input_json_file, output_json_file)