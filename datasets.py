import glob
import random
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from random import randrange
import torchvision.transforms.functional as TF


# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape

        SAMPLING = [Image.BICUBIC, Image.BILINEAR, Image.NEAREST]

        # Transforms for low resolution images and high resolution images
        self.crop_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(hr_shape),
                transforms.ToTensor()
            ]
        )
        
        self.lr_transform = transforms.Compose(
            [   
                transforms.ToPILImage(),
                transforms.Resize((hr_height // 2, hr_width // 2), SAMPLING[randrange(0,len(SAMPLING))]),
                transforms.Resize((hr_height // 4, hr_width // 4), SAMPLING[randrange(0,len(SAMPLING))]),
                transforms.Resize((hr_height // 8, hr_width // 8), SAMPLING[randrange(0,len(SAMPLING))]),
                transforms.ToTensor()
            ]
        )
        
        self.hr_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor()
            ]
        )

        images = []
        for dir in root:
            images += glob.glob(dir+"/*.*")
        self.files = sorted(images)

    def __getitem__(self, index):

        #Read EXR file with CV2
        img = cv2.imread(self.files[index],\
              cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        #print("Original image shape: "+str(img.shape))

        #Downsample
        img = self.crop_transform(transforms.ToTensor()(img))
        print("Cropped image shape: "+str(img.shape))
        #print("Cropped image shape after permutation: "+str(img_lr.shape))
        img_lr = self.lr_transform(img)
        #img_lr = img_lr.permute(1,2,0)
        print("LR image shape: "+str(img_lr.size()))
        #img_hr = self.hr_transform(img)
        img_hr = img
        print("HR image shape: "+str(img_hr.size()))

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)
