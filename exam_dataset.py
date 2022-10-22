import torch
import numpy as np
from PIL import Image
import pytorch_lightning as pl
import random
import os
import torch
import torchvision
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from pathlib import Path


DIR_IMG  = '/home/jovyan/Datasets/crack_segmentation_dataset/train/images/'
DIR_MASK = '/home/jovyan/Datasets/crack_segmentation_dataset/train/masks/'
img_names  = [path.name for path in Path(DIR_IMG).glob('*.jpg')]
mask_names = [path.name for path in Path(DIR_MASK).glob('*.jpg')]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class ImgDataSet(Dataset):
    def __init__(self, img_transform, mask_transform):
        super().__init__()
        self.img_dir = DIR_IMG
        self.img_fnames = img_names
        self.img_transform = img_transform

        self.mask_dir = DIR_MASK
        self.mask_fnames = mask_names
        self.mask_transform = mask_transform

        self.seed = np.random.randint(2147483647)
        
    def __getitem__(self, i):
        fname = self.img_fnames[i]
        fpath = os.path.join(self.img_dir, fname)
        img = Image.open(fpath)
        if self.img_transform is not None:
            random.seed(self.seed)
            img = self.img_transform(img)
            #print('image shape', img.shape)

        mname = self.mask_fnames[i]
        mpath = os.path.join(self.mask_dir, mname)
        mask = Image.open(mpath)
        #print('khanh1', np.min(test[:]), np.max(test[:]))
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
            #print('mask shape', mask.shape)
            #print('khanh2', np.min(test[:]), np.max(test[:]))

        return img, mask #torch.from_numpy(np.array(mask, dtype=np.int64))

    def __len__(self):
        return len(self.img_fnames)





############'
class CustomeDataset(Dataset):
    def __init__(self, c_data, m_data) -> None:
        super().__init__()
        self.dpath = '/home/jovyan/Datasets/crack_segmentation_dataset/images/'
        self.lpath = '/home/jovyan/Datasets/crack_segmentation_dataset/masks/'
        self.data = c_data
        self.label = m_data
        # self.file = file.load()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return super().__getitem__(index)