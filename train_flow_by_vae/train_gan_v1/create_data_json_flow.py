import json
import os
import random

# ----------------------- 原始定义 -----------------------
dataset = []
data_id = 0
question_id = 0
images_dir = masks_dir = '/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/data_tsy1/images_dia'
# masks_dir = "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/data_tsy1/images-region"

labels = os.listdir(images_dir)
label_map = json.load(open('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/data_tsy1/label_map.json', 'r'))
data_template = {
    'id': None,
    'question_id': None,
    'x0': None,
    'x1': None,
    'caption': None,
    'y': None,
    'hint_path': None,
    "mask_hint_path": None
}

# ----------------------- 新增参数 -----------------------
m = 200
k = 100  # 每个 x0 配对 k 个不同的 x1

# ----------------------- 构造 dataset -----------------------
for label in labels:
    dataset = []
    x0_dir = os.path.join(images_dir, label)
    x1_dir = os.path.join(images_dir, label)
    x0_files = os.listdir(x0_dir)
    x0_paths = [os.path.join(x0_dir, f) for f in x0_files]
    x1_files = os.listdir(x1_dir)
    mask_dir = os.path.join(masks_dir, label)
    mask_files = os.listdir(mask_dir)

    if label not in label_map:
        label_map[label] = len(label_map)
    random.shuffle(x0_paths)
    random.shuffle(mask_files)
    for x0_path in x0_paths[:m]:
        # 随机选择 k 个 x1 文件
        idx = 0
        for mask_file in mask_files[:k]:
            # ----------------------- 获取图像文件列表 -----------------------
            # x1_file = mask_file.replace("_masked", "").replace("_crop", "").replace(".png", ".jpg")
            x1_file = mask_file
            x1_path = os.path.join(x1_dir, x1_file)
            x1_mask_path = os.path.join(mask_dir, mask_file)
            
            if x1_path == x0_path:
                continue
            if os.path.exists(x1_path) and os.path.exists(x1_mask_path):
                pass
            else:
                continue
            data = data_template.copy()
            data['data_id'] = data_id
            data['question_id'] = question_id
            data['x0'] = x0_path
            data['x1'] = x1_path
             # data['caption'] = f'内窥镜医疗图像'  # 可自定义
            data['caption'] = f'{label}'  # 可自定义
            data['label_B_id'] = label_map[label]
            data['label_A_id'] = label_map[label]
            data['label_A'] = label
            data['label_B'] = label
            data['hint_path'] = x1_path
            data['y'] = label_map[label]
            data['mask_hint_path'] = x1_mask_path
            dataset.append(data)
            idx += 1
            data_id += 1
        question_id += 1 # 同一个x0

    ### ----------------------- 可选：保存为 JSON 文件 -----------------------
    if len(dataset) == 0:
        continue
    print(f"{label} Dataset Size: ", len(dataset))
    os.makedirs('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v1/2004-2010_data_pairs_dia', exist_ok=True)
    with open(f'/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v1/2004-2010_data_pairs_dia/{label}.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
with open('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/data_tsy1/label_map.json', 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)
