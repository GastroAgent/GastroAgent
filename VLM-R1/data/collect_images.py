import json
from PIL import Image
import os
import tqdm

dataset = json.load(open("/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/RL/RL_train_without_ref_with_train.json", "r"))

image_keys = ["image", "image_paths"]
for data in tqdm.tqdm(dataset):
    for key in image_keys:
        if key in data:
            image_paths = data[key]
            if isinstance(image_paths, str):
                image_paths = [image_paths]

            for i, image_path in enumerate(image_paths):
                new_path_dir = os.path.join("/mnt/inaisfs/data/home/tansy_criait/VLM-R1/RL_Images", image_path.split("/")[-2])
                os.makedirs(new_path_dir, exist_ok=True)
                new_path = os.path.join(new_path_dir, image_path.split("/")[-1])
                if not os.path.exists(new_path):
                    image = Image.open(image_path)
                    image = image.convert('RGB')
                    image.save(new_path)
                image_paths[i] = new_path

with open('/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/RL/RL_train_without_ref_with_train2.json', 'w') as outfile:
    json.dump(dataset, outfile, indent=4, ensure_ascii=False)




