import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape, rand_crop_w=4320, rand_crop_h=4320):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.RandomCrop(size=(rand_crop_w, rand_crop_h))
                transforms.Resize((hr_height // 8, hr_height // 8), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        #img = Image.open(self.files[index % len(self.files)])

        #Read EXR file with CV2
        img = cv2.imread(self.files[index % len(self.files)],\
              cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        
        #Convert to PIL Image
        img_hr = Image.fromarray(img, 'RGB')
        
        #Donwsample 
        img_lr = self.lr_transform(img_hr)
        
        #img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)
