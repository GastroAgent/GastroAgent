
import json
import os
import random

# ----------------------- 原始定义 -----------------------
dataset = []
data_id = 0
question_id = 0
images_dir = '/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match_2十二指肠/simple_data_checked/十二指肠球部/train'
labels = os.listdir(images_dir)

label_map = json.load(open('/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/utils/label_map.json', 'r'))
data_template = {
    'question_id': None,
    'x0': None,
    'caption': None,
    # "mask_hint_path": None
}

# ----------------------- 新增参数 -----------------------
question_id = 0
min_k = 5
max_k = len(labels)
dataset = []
# ----------------------- 构造 dataset -----------------------
for label in labels * 20:
    x0_dir = os.path.join(images_dir, label)
    x1_dir = os.path.join(images_dir, label)
    x0_files = os.listdir(x0_dir)
    x0_paths = [os.path.join(x0_dir, f) for f in x0_files]
    for x0_path in x0_paths:
        test_dirs = []
        test_labels = [] 
        test_captions = []
        k = random.choice(list(range(min_k, max_k)))
        random.shuffle(labels)
        for i, test_label in enumerate(labels):
            if i >= k:
                break
            else:
                if test_label not in label_map:
                    label_map[test_label] = len(label_map)
                if test_label in test_labels:
                    continue

                test_labels.append(test_label)
                label_dir = os.path.join(images_dir, test_label)
                if not os.path.exists(label_dir):
                    raise ValueError
                test_dirs.append(label_dir)
                test_captions.append(f'将当前未知的医疗图像病症 转化为 {test_label} 等相关病症图像。')
                if label not in test_labels and random.random() > 0.8:
                    test_labels.append(label)
                    label_dir = os.path.join(images_dir, label)
                    test_dirs.append(label_dir)
                    test_captions.append(f'将当前未知的医疗图像病症 转化为 {label} 等相关病症图像。')
        if label not in test_labels:
            test_labels.append(label)
            label_dir = os.path.join(images_dir, label)
            test_dirs.append(label_dir)
            test_captions.append(f'将当前未知的医疗图像病症 转化为 {label} 等相关病症图像。')
        data = data_template.copy()
        data['data_id'] = data_id
        data['question_id'] = question_id
        data['x0'] = x0_path
        data['x1_dirs'] = test_dirs
        data['x1_labels'] = test_labels
        data['caption'] = test_captions
        data['label_B_ids'] = [label_map[x] for x in test_labels]
        data['label_A_id'] = label_map[label]
        data['label_A'] = label    
        data['ys'] = [label_map[x] for x in test_labels]
        data['question_id'] = question_id
        question_id += 1
        dataset.append(data)

### ----------------------- 可选：保存为 JSON 文件 -----------------------
print("Dataset Size: ", len(dataset))
with open('/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match_2十二指肠/simple_data_checked/train_十二指肠球部.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)
