import json
import os
import glob
import re
from PIL import Image
from tqdm import tqdm
import numpy as np
# from skimage.metrics import structural_similarity as ssim

####################### 数据筛选的 第一步 #######################
pattern_step = r'step(\d+)_'
pattern_score = r'_([0-9]*\.[0-9]+)_'

try:
    processed_files = json.load(open('./discriminator/data/generated_all_data/processed_files.json', 'r'))
except:
    processed_files = []

def parse_image_name(image_name: str):
    if '/' in image_name:
        image_name = image_name.split('/')[-1]
    if 'step' not in image_name:
        step = 0
    else:
        step = re.findall(pattern_step, image_name)
        
        if len(step) > 0:
            step = int(step[0])
            image_name = image_name.replace('step' + str(step), '')
        else:
            step = 0

    match = re.search(pattern_score, image_name)
    if match:
        score = match.group(1)  # 提取捕获组内容
        image_name = image_name.replace('_' + score + '_', '')
        score = float(score)
    else:
        score = float(0)   

    return step, score, image_name
    
image_root = "./discriminator/data/generated_all_data/image_hint_dinov3"
new_image_root = "./discriminator/data/similarity_images_draw/Dinov3"
new_image_root_all = './discriminator/data/images_draw/Dinov3'
image_roots = glob.glob("./discriminator/data/generated_all_data/*dinov3*") + \
     glob.glob("./discriminator/data/generated_all_data/*qwen*")

os.makedirs(new_image_root, exist_ok=True)

label2labels_dirs = os.listdir(image_root)
label2labels_dirs = [os.path.join(image_root, x) for x in label2labels_dirs]
processed_files.append(
    {
        "file": image_root,
        "sim_model_type": "Qwen",
    }
)

for image_dir in tqdm(label2labels_dirs):
    image_names = os.listdir(image_dir)
    label2label = image_dir.split('/')[-1]
    image_dict = {}
    dataset = []
    for image_name in image_names:
        step, score, image_name = parse_image_name(image_name)
        if image_name not in image_dict:
            image_dict[image_name] = [(step, score)]
        else:
            image_dict[image_name] += [(step, score)]
            
    for image_name in tqdm(image_dict):
        image_dict[image_name] = sorted(image_dict[image_name], key=lambda x: x[0])
        scores = [x[1] for x in image_dict[image_name]]
        steps = [x[0] for x in image_dict[image_name]]
        q1 = np.percentile(scores, 10)  # 下四分位数（10th percentile）
        q3 = np.percentile(scores, 90)  # 上四分位数（90th percentile）
        s1 = int(np.percentile(steps, 10))  # 下四分位数（15th percentile）
        s3 = int(np.percentile(steps, 90))  # 上四分位数（5th percentile）
        
        for step, score in image_dict[image_name]:
            if '###' in image_name:
                image_path = image_name.split('###')[-1]
            elif '#to#' in image_name:
                image_path = image_name.split('#to#')[-1]
            else:
                image_path = image_name.split('to')[-1]
            if step == 0:
                new_image_dir = new_image_root + f'/anchor/{label2label}'
                os.makedirs(new_image_dir, exist_ok=True)
                new_image_path = os.path.join(new_image_dir, image_path)
                if not os.path.exists(new_image_path):
                    image = Image.open(os.path.join(image_dir, image_path)).convert("RGB")
                    image.save(new_image_path)
                continue
            
            image_path = f'step{step}_{score:.4f}_{image_name}'
            if score > q3 and score > 0.9 and step >= s3:
                new_image_dir = new_image_root + f'/positive/{label2label}/{image_name[:-4]}'
                os.makedirs(new_image_dir, exist_ok=True)
                new_image_path = os.path.join(new_image_dir, image_path)
            elif score < q1 and step <= s1:
                new_image_dir = new_image_root + f'/negitive/{label2label}/{image_name[:-4]}'
                os.makedirs(new_image_dir, exist_ok=True)
                new_image_path = os.path.join(new_image_dir, image_path)
            else:
                new_image_dir = new_image_root + f'/determined/{label2label}/{image_name[:-4]}'
                os.makedirs(new_image_dir, exist_ok=True)
                new_image_path = os.path.join(new_image_dir, image_path)
            
            if not os.path.exists(new_image_path):
                image = Image.open(os.path.join(image_dir, image_path)).convert("RGB")
                image.save(new_image_path)
            else:
                image = None
            
            new_image_dir = new_image_root_all + f'/{label2label}/{image_name[:-4]}'
            os.makedirs(new_image_dir, exist_ok=True)
            new_image_path = os.path.join(new_image_dir, image_path)
            if not os.path.exists(new_image_path):
                if image is None:
                    image = Image.open(os.path.join(image_dir, image_path)).convert("RGB")
                image.save(new_image_path)
        
with open("./discriminator/data/generated_all_data/processed_files.json", 'w') as f:
    json.dump(processed_files, f, indent=4, ensure_ascii=False)
