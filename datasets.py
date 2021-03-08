import glob
import random
import os
import numpy as np
import cv2 #***Added***#
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from random import randrange
import torchvision.transforms.functional as TF


# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
SAMPLING = [Image.BICUBIC, Image.BILINEAR, Image.NEAREST]

def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape, rand_crop_w=512, rand_crop_h=512):
        hr_height, hr_width = hr_shape #can ignore this, not used
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((rand_crop_w // 2, rand_crop_h // 2), SAMPLING[randrange(0,len(SAMPLING))]),
                transforms.Resize((rand_crop_w // 2, rand_crop_h // 2), SAMPLING[randrange(0,len(SAMPLING))]),
                transforms.Resize((rand_crop_w // 2, rand_crop_h // 2), SAMPLING[randrange(0,len(SAMPLING))]),
                #transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                #transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                #transforms.Normalize(mean, std),
            ]
        )
        images = []
        for dir in root:
            images += glob.glob(dir+"/*.*")
        self.files = sorted(images)

    def __getitem__(self, index):
        #img = Image.open(self.files[index % len(self.files)])
        img_orig = cv2.imread(self.files[index % len(self.files)],\
              cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        width = randrange(0,len(img_orig[0,:])-512)
        height = randrange(0,len(img_orig[:,0])-512)
        final_img = img_orig[height:height+512,width:width+512]
        #print("ORIG SIZE X: "+str(len(img_orig[:][0])))
        #print("ORIG SIZE Y: "+str(len(img_orig[0][:])))
        #print(final_img.shape)
        #print("x = "+str(x)+", y = "+str(y))
        #print("\n\nORIG SIZE: "+str(img_orig.shape)+". POINT: "+str(x)+","+str(y)+". IMAGE SIZE: "+str(final_img.shape)+"\n\n")
        img_lr = self.lr_transform(final_img)
        img_hr = self.hr_transform(final_img)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)
