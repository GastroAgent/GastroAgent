import json
import os
import random
dataset = []
data_id = 0
question_id = 0
images_dir = './wass_flow_match_tsy/data_tsy1/食管_doctor'
labels = os.listdir(images_dir)
saved_dir = './wass_flow_match_tsy/data_tsy1/suppport_img_raw' # 支撑集
label_map = json.load(open('./wass_flow_match_tsy/data_tsy1/label_map.json', 'r'))
data_template = {
    'question_id': None,
    'x0': None,
    'caption': None,
    # "mask_hint_path": None
}
# cp -r ./wass_flow_match_十二指肠/data/十二指肠/eval_region/十二指肠降部炎/* ./wass_flow_match_十二指肠/data/十二指肠/final_eval/十二指肠降部炎
# ----------------------- 新增参数 -----------------------
question_id = 0
min_k = 5
max_k = len(labels)
min_k = min([min_k, max_k])

dataset = []
# ----------------------- 构造 dataset -----------------------
for label in labels:
    x0_dir = os.path.join(images_dir, label)
    x1_dir = os.path.join(images_dir, label)
    x0_files = os.listdir(x0_dir)
    x0_paths = [os.path.join(x0_dir, f) for f in x0_files]
    for x0_path in x0_paths:
        test_dirs = []
        test_labels = [] 
        test_captions = []
        # k = random.choice(list(range(min_k, max_k + 1)))
        for i, test_label in enumerate(labels):
            if False:
                break
            else:
                if test_label not in label_map:
                    label_map[test_label] = len(label_map)
                if test_label in test_labels:
                    continue

                test_labels.append(test_label)
                label_dir = os.path.join(images_dir, test_label).replace(images_dir, saved_dir)
                if not os.path.exists(label_dir):
                    raise ValueError
                test_dirs.append(label_dir)
                
                test_captions.append(f'{test_label}')
                if label not in test_labels and random.random() > 0.8:
                    test_labels.append(label)
                    label_dir = os.path.join(images_dir, label).replace(images_dir, saved_dir)
                    test_dirs.append(label_dir)
                    test_captions.append(f'{label}')
                    
        if label not in test_labels:
            test_labels.append(label)
            label_dir = os.path.join(images_dir, label).replace(images_dir, saved_dir)
            test_dirs.append(label_dir)
            test_captions.append(f'{label}')
        
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
with open('./wass_flow_match_tsy/data_tsy1/食管_doctor/final_doctor_exam.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)