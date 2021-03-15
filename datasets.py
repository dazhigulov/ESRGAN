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
    def __init__(self, root, hr_shape, rand_crop_w=4320, rand_crop_h=4320):
        hr_height, hr_width = hr_shape
        SAMPLING = [Image.BICUBIC, Image.BILINEAR, Image.NEAREST]

        # Transforms for low resolution images and high resolution images
        self.crop_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(512),
            ]
        )
        
        self.lr_transform = transforms.Compose(
            [   
                transforms.Resize((rand_crop_w // 2, rand_crop_h // 2), SAMPLING[randrange(0,len(SAMPLING))]),
                transforms.Resize((rand_crop_w // 2, rand_crop_h // 2), SAMPLING[randrange(0,len(SAMPLING))]),
                transforms.Resize((rand_crop_w // 2, rand_crop_h // 2), SAMPLING[randrange(0,len(SAMPLING))]),
                transforms.ToTensor()
            ]
        )
        
        self.hr_transform = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )

        images = []
        for dir in root:
            images += glob.glob(dir+"/*.*")
        self.files = sorted(images)

    def __getitem__(self, index):
        #img = Image.open(self.files[index % len(self.files)])

        #Read EXR file with CV2
        img = cv2.imread(self.files[index % len(self.files)],\
              cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        print("Input shape: "+str(img.shape))
        #Downsample
        print(torch.Tensor(img).shape)
        print(transforms.ToPILImage()(transforms.ToTensor()(img)).size)
        img_cropped = self.crop_transform(transforms.ToTensor()(img))
        img_lr = self.lr_transform(img_cropped)
        img_hr = self.hr_transform(img_cropped)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)
