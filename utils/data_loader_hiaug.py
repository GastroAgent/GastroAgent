import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from loguru import logger
import json
import numpy as np
import random
import pickle


class MedicalCLIPTinyAUGDataset(Dataset):
    def __init__(self, data_path='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/data/disease_data/diffusion_dataset_sup_tiny_hiaug.pkl',
                 label_path='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/data/disease_data/diffusion_label2id_sup_tiny_hiaug.json'):
        self.dataset = pickle.load(open(data_path, 'rb'))

        self.labels2id = json.load(open(label_path, 'r'))
        self.id2labels = {v:k for k, v in self.labels2id.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        batch = dict(x1=data['x1'], caption=data['prompt'], x0=data['x0'], class_id=torch.tensor(data['y']).long())
        return batch

class MedicalCLIPTinyDiseaseDataset(Dataset):
    def __init__(self, data_path='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/diffusion_dataset_sup_tiny_hidis.json',
                 label_path='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/diffusion_label2id_sup_tiny_hidis.json',
                 transform=None):
        self.dataset = json.load(open(data_path, 'r'))
        self.transform = transform
        self.labels2id = json.load(open(label_path, 'r'))
        self.id2labels = {v:k for k, v in self.labels2id.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        x0_path = data['x0']
        x1_path = data['x1']
        x0 = self.transform(Image.open(x0_path).convert('RGB'))
        x1 = self.transform(Image.open(x1_path).convert('RGB'))
        y = data['label']
        caption = data['caption']
        batch = dict(x1=x1, caption=caption, x0=x0, class_id=torch.tensor(y).long())
        return batch

class MedicalCLIPTinyA2BDataset(Dataset):
    def __init__(self, data_path_A='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_All_Cropped/A级',
                 data_path_B='/home/dalhxwlyjsuo/criait_tansy/Node_Code/126/work/Diffusion/flow-based-models/test_data/食管_All_Cropped/C级',
                 label_A='食管炎-A级', label_B='食管炎-C级', y = 2, shuffle_AB = False,
                 transform=None, transform_A = None, transform_B=None):
        self.images_A = [os.path.join(data_path_A, x) for x in os.listdir(data_path_A)]
        self.images_B = [os.path.join(data_path_B, x) for x in os.listdir(data_path_B)]
        self.transform = transform
        self.label_A = label_A
        self.label_B = label_B
        self.y = y
        self.transform_A = transform_A
        self.transform_B = transform_B
        self.shuffle_AB = shuffle_AB

    def __len__(self):
        return min([len(self.images_A), len(self.images_B)])

    def __getitem__(self, idx):
        if len(self.images_A) == len(self.images_B) and not self.shuffle_AB:
            x0_path = self.images_A[idx]
            x1_path = self.images_B[idx]
        else:
            x0_path = random.choice(self.images_A)
            x1_path = random.choice(self.images_B)
            
        if self.transform_A is not None:
            x0 = self.transform_A(Image.open(x0_path).convert('RGB'))
        else:
            x0 = self.transform(Image.open(x0_path).convert('RGB'))

        if self.transform_B is not None:
            x1 = self.transform_B(Image.open(x1_path).convert('RGB'))
        else:
            x1 = self.transform(Image.open(x1_path).convert('RGB'))
        y = self.y
        caption = f'{self.label_A} 转化为 {self.label_B}。'
        batch = dict(x1=x1, caption=caption, x0=x0, class_id=torch.tensor(y).long())
        return batch


if __name__ == '__main__':
    MedicalCLIPTinyAUGDataset()