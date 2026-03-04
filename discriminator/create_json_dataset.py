import json
import os
from glob import glob
import random

try:
    dataset = json.load(open("./discriminator/data/triplet_all_dataset.json", "r"))
    # dataset = []
except:
    dataset = []

image_root = './discriminator/data/similarity_images_draw/Dinov3' # GME
labels = os.listdir(image_root + '/positive')

for label in labels:
    label_path = os.path.join(image_root + '/positive', label)
    image_dir_names = os.listdir(label_path)
    image_dir_paths = [os.path.join(label_path, x) for x in image_dir_names]
    for image_dir_path in image_dir_paths:
        image_names = os.listdir(image_dir_path)
        x0, x1 = image_dir_path.split('/')[-1].split('to')
        x0 = x0.replace('#', '')
        x1 = x1.replace('#', '')
        neg_image_dir_path = image_dir_path.replace('positive', 'negitive')
        for image_name in image_names:
            image_path = os.path.join(image_dir_path, image_name)
            anchor_path = image_root + f'/anchor/{label}/{x1}.jpg'
            if not os.path.exists(anchor_path):
                continue
            if not os.path.exists(neg_image_dir_path):
                continue
            neg_image_names = os.listdir(neg_image_dir_path)
            for neg_image_name in neg_image_names:
                neg_image_path = os.path.join(neg_image_dir_path, neg_image_name)
                if not os.path.exists(neg_image_path):
                    continue

                data = {
                    'positive_path': image_path,
                    'anchor_path': anchor_path,
                    'negative_path': neg_image_path
                }
                dataset.append(data)

random.shuffle(dataset)
eval_size = min([int(len(dataset) * 0.01), 3000])
train_dataset = dataset[:-eval_size]
eval_dataset = dataset[-eval_size:]
print(f'Dataset Size: {len(dataset)}')
print(f'Train Dataset Size: {len(train_dataset)}')
print(f'Eval Dataset Size: {len(eval_dataset)}')
with open('./discriminator/data/eval_triplet_all_dataset.json', 'w') as f:
    json.dump(eval_dataset, f, indent=4, ensure_ascii=False)
with open('./discriminator/data/train_triplet_all_dataset.json', 'w') as f:
    json.dump(train_dataset, f, indent=4, ensure_ascii=False)
with open('./discriminator/data/triplet_all_dataset.json', 'w') as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)