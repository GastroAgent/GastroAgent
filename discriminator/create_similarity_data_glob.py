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
    
new_image_root = "/mnt/inaisfs/data/home/tansy_criait/whole_wass_flow_match/discriminator/data/similarity_images/Dinov3" # 仅输出符合要求的数据
new_image_root_all = '/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/discriminator/data/images/Dinov3' # 输出路径，输出全部
image_roots = glob.glob("/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/discriminator/data/generated_all_data/*dinov3*") # 输入路径

os.makedirs(new_image_root, exist_ok=True)

for image_root in image_roots:
    try:
        processed_files = json.load(open('/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/generated_all_data/processed_files_draw.json', 'r'))
    except:
        processed_files = []
    label2labels_dirs = os.listdir(image_root)
    label2labels_dirs = [os.path.join(image_root, x) for x in label2labels_dirs]
    processed_files.append(
        {
            "file": image_root,
            "sim_model_type": "Dinov3",
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
            if len(scores) == 1:
                continue
            q1 = np.percentile(scores, 75)  # 下四分位数（10th percentile）
            q3 = np.percentile(scores, 90)  # 上四分位数（90th percentile）
            s1 = int(np.percentile(steps, 60))  # 下四分位数（15th percentile）
            s3 = int(np.percentile(steps, 80))  # 上四分位数（5th percentile）
            
            for step, score in image_dict[image_name]:
                if '###' in image_name:
                    image_path = image_name.split('###')[-1]
                elif '#to#' in image_name:
                    image_path = image_name.split('#to#')[-1]
                elif 'to' in image_name:
                    image_path = image_name.split('to')[-1]
                else:
                    continue
                if step <= 1:
                    new_image_dir = new_image_root + f'/anchor/{label2label}'
                    os.makedirs(new_image_dir, exist_ok=True)
                    new_image_path = os.path.join(new_image_dir, image_path)
                    if not os.path.exists(new_image_path):
                        try:
                            image = Image.open(os.path.join(image_dir, image_path)).convert("RGB")
                            image.save(new_image_path)
                        except:
                            pass
                    continue
                
                image_path = f'step{step}_{score:.4f}_{image_name}'
                if score > q3 and score > 0.8 and step >= s3:
                    new_image_dir = new_image_root + f'/positive/{label2label}/{image_name[:-4]}'
                    os.makedirs(new_image_dir, exist_ok=True)
                    new_image_path = os.path.join(new_image_dir, image_path)
                elif score < q1 and step <= s1:
                    new_image_dir = new_image_root + f'/negitive/{label2label}/{image_name[:-4]}'
                    os.makedirs(new_image_dir, exist_ok=True)
                    new_image_path = os.path.join(new_image_dir, image_path)
                else:
                    new_image_dir = new_image_root_all + f'_determined/{label2label}/{image_name[:-4]}'
                    os.makedirs(new_image_dir, exist_ok=True)
                    new_image_path = os.path.join(new_image_dir, image_path)
                
                if not os.path.exists(new_image_path):
                    try:
                        image = Image.open(os.path.join(image_dir, image_path)).convert("RGB")
                        image.save(new_image_path)
                    except:
                        image = None
                else:
                    image = None
                
                new_image_dir = new_image_root_all + f'/{label2label}/{image_name[:-4]}'
                os.makedirs(new_image_dir, exist_ok=True)
                new_image_path = os.path.join(new_image_dir, image_path)
                if not os.path.exists(new_image_path):
                    try:
                        if image is None:
                            image = Image.open(os.path.join(image_dir, image_path)).convert("RGB")
                        image.save(new_image_path)
                    except:
                        pass
