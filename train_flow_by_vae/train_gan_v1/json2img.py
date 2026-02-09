import json
from PIL import Image
import os
import glob

json_paths = glob.glob("/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/data_tsy1/train_json/All_data_pairs_dia/*.json")
save_dir = "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_tsy/train/train_gan_v2/new_cropped-2004-2010"
for json_path in json_paths:
    dataset = json.load(open(json_path))

    for data in dataset:
        x0 = data['x0']
        name = x0.split('/')[-1]
        new_path = save_dir + '/' + name
        if os.path.exists(new_path):
            continue
        print(x0)
        img = Image.open(x0)
        name = x0.split('/')[-1]
        new_path = save_dir + '/' + name
        img.save(new_path)