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
    
new_image_root = "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/step_score_generated/similarity_images_draw"

image_roots = glob.glob("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/discriminator/data/generated_all_data/*dinov3*") + \
        glob.glob("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/discriminator/data/generated_all_data/*qwen*") + \
        glob.glob("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match/generated_all_data/*dinov3*") + \
        glob.glob("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match/generated_all_data/*Dinov3*") + \
        glob.glob("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match/generated_all_data/*gme*") + \
        glob.glob("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match/generated_all_data/*GME*") + \
        glob.glob("/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/discriminator/data/generated_all_data/*gme*") + \
        glob.glob("/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/discriminator/data/generated_all_data/*dinov3*")
gt_iamge_roots = glob.glob("/mnt/inaisfs/data/home/tansy_criait/new_wass_flow_match/discriminator/data/generated_all_data/*GT*")
image_roots = [x for x in image_roots if x not in gt_iamge_roots and 'GT' not in x]
# allowed_name = os.listdir('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/data_tsy1/images_dia')
# allowed_name += ['胃体']
# unallowed_name = []

allowed_name = json.load(open('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/allowed_name.json', 'r'))
unallowed_name = json.load(open('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/unallowed_name.json', 'r'))

os.makedirs(new_image_root, exist_ok=True)
for image_root in image_roots:
    label2labels_dirs = os.listdir(image_root)
    label2labels_dirs = [os.path.join(image_root, x) for x in label2labels_dirs]

    for image_dir in tqdm(label2labels_dirs):
        image_names = os.listdir(image_dir)
        label2label = image_dir.split('/')[-1]
        if 'to' in label2label:
            label = label2label.split('to')[-1]
        elif '#to#' in label2label:
            label = label2label.split('#to#')[-1]
        elif '###' in label2label:
            label = label2label.split('###')[-1]
        else:
            label = label2label.split('2')[-1]
        if label.replace("_", " ") not in allowed_name:
            if label not in unallowed_name:
                unallowed_name.append(label)
            continue
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
            q1 = np.percentile(scores, 25)  # 下四分位数（10th percentile）
            q3 = np.percentile(scores, 90)  # 上四分位数（90th percentile）
            s1 = int(np.percentile(steps, 25))  # 下四分位数（15th percentile）
            s3 = int(np.percentile(steps, 90))  # 上四分位数（5th percentile）
            s2 = int(np.percentile(steps, 60)) 
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
                        try:
                            image = Image.open(os.path.join(image_dir, image_path)).convert("RGB")
                            image.save(new_image_path)
                        except:
                            pass
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
                elif score < q1 and step >= s2:
                    new_image_dir = new_image_root + f'/determined_fake/{label2label}/{image_name[:-4]}'
                    os.makedirs(new_image_dir, exist_ok=True)
                    new_image_path = os.path.join(new_image_dir, image_path)
                else:
                    new_image_dir = new_image_root + f'/unknown/{label2label}/{image_name[:-4]}'
                    os.makedirs(new_image_dir, exist_ok=True)
                    new_image_path = os.path.join(new_image_dir, image_path)
                try:
                    if not os.path.exists(new_image_path):
                        image = Image.open(os.path.join(image_dir, image_path)).convert("RGB")
                        image.save(new_image_path)
                    else:
                        image = None
                except:
                    pass
                
with open('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/unallowed_name.json', 'w', encoding='utf-8') as f:
    json.dump(unallowed_name, f, ensure_ascii=False, indent=4)

with open('/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/allowed_name.json', 'w', encoding='utf-8') as f:
    json.dump(allowed_name, f, ensure_ascii=False, indent=4)