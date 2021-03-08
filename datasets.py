import glob
import random
import os
import numpy as np
<<<<<<< HEAD
import cv2 #***Added***#
=======
import random

>>>>>>> 44b224cd0afae16d9253937a2ddca5ad1f18ffbd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from random import randrange
import torchvision.transforms.functional as TF


# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
<<<<<<< HEAD
SAMPLING = [Image.BICUBIC, Image.BILINEAR, Image.NEAREST]
=======

#Downsampling Container
DOWN_SAMPLING_CONTAINER = [Image.BICUBIC, Image.BILNEAR]

>>>>>>> 44b224cd0afae16d9253937a2ddca5ad1f18ffbd

def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


class ImageDataset(Dataset):
<<<<<<< HEAD
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
=======
    def __init__(self, root, hr_shape):
        # Generate random down sampling methods if you add more methods please change the random generator
        random_method = random.randint(0, 1)
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 8, hr_height // 8), DOWN_SAMPLING_CONTAINER[random_method]),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
>>>>>>> 44b224cd0afae16d9253937a2ddca5ad1f18ffbd
            ]
        )
        self.hr_transform = transforms.Compose(
            [
<<<<<<< HEAD
                transforms.ToTensor(),
                #transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                #transforms.Normalize(mean, std),
=======
                transforms.Resize((hr_height, hr_height), DOWN_SAMPLING_CONTAINER[random_method]),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
>>>>>>> 44b224cd0afae16d9253937a2ddca5ad1f18ffbd
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        #img = Image.open(self.files[index % len(self.files)])
        img_orig = cv2.imread(self.files[index % len(self.files)],\
              cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        x = randrange(0,4320-512)
        y = randrange(0,7680-512)
        xx = x+512
        yy = y+512
        final_img = img_orig[x:xx,y:yy]
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
