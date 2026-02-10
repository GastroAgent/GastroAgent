import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from loguru import logger
import json
import numpy as np
import random

class MedicalTripletJsonDataset(Dataset):
    def __init__(self,
                 path='/mnt/inaisfs/data/home/tansy_criait/wass_flow_match/discriminator/data/triplet_dataset.json',
                 transform=None):
        self.dataset = json.load(open(path, 'r'))
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        positive_path = data['positive_path']
        negative_path = data['negative_path']
        anchor_path = data['anchor_path']
        positive = self.transform(Image.open(positive_path).convert("RGB"))
        negative = self.transform(Image.open(negative_path).convert("RGB"))
        anchor = self.transform(Image.open(anchor_path).convert("RGB"))
        return {
            'positive': positive,
            'negative': negative,
            'anchor': anchor,
        }


