# JW 
#   Dragonball
#       data
#           Dataset.py

#!/usr/bin/env python
# coding: utf-8
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from glob import glob
import torch
import torch.nn as nn
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import glob
from PIL import Image


seed = 42


class CrackDataset(Dataset):
    def __init__(self, split = 'train'):
        super().__init__()
        path = os.path.join(os.path.dirname(__file__), 'crack_segmentation_dataset')
        path = os.path.join(path, split, 'images')
        self.img_path = glob.glob(path + '/*.jpg')
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.image_size = (448, 448)
        
    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, idx):
        image_path = self.img_path[idx]
        label_path = image_path.replace('images', 'masks')
        
        image = Image.open(image_path)      
        mask = Image.open(label_path)

        image = self.transforms(image)
        mask = self.transforms(mask)

        return image, mask

