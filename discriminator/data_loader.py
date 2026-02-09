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


if __name__ == "__main__":

    # Example image transformations
    transform = transforms.Compose([
        # transforms.Resize((256, 256)),   # Resize images to 256x256
        transforms.CenterCrop((128, 256)),  # Crop the center 256x256
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
    ])

    # Initialize the dataset
    image_dir = '/disk1/BharatDiffusion/kohya_ss/experimental_sricpts/anime_images'  # Specify your image directory
    dataset = MedicalCLIPTestDataset(image_dir=image_dir, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=32,  # Adjust the batch size as needed
        shuffle=True,  # Shuffle the data at the beginning of each epoch
        num_workers=4  # Number of subprocesses to use for data loading
    )

    for images in dataloader:
        logger.info(f"Shape of images: {images.shape}")
