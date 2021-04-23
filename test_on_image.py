from models import GeneratorRRDB
from datasets import denormalize, mean, std
import torch
from torch.autograd import Variable
import argparse
import os
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from random import randrange
import cv2
import glob

SAMPLING = [Image.BICUBIC, Image.BILINEAR, Image.NEAREST]

parser = argparse.ArgumentParser()
#parser.add_argument("--image_path", type=str, required=True, help="Path to image")
parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--residual_blocks", type=int, default=23, help="Number of residual blocks in G")
parser.add_argument("--dataset_path", type=str, required=True, help="path to dataset")
parser.add_argument("--hr_height", type=int, default=7680, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=4320, help="high res. image width")
opt = parser.parse_args()
print(opt)

PATH = opt.dataset_path
dev_paths = [PATH + "c06_Drama_standingup_8K",
             PATH + "c12_Volleyball_fixed_8K",
             PATH + "c15_Paddock_fixed_8K"]
images = []
for dir in dev_paths:
    images += glob.glob(dir+"/*.*")
images = sorted(images)
hr_height = opt.hr_height
hr_width = opt.hr_width
lr_transform = transforms.Compose(
            [   
            	transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize((hr_height // 2, hr_width // 2), SAMPLING[randrange(0,len(SAMPLING))]),
                transforms.Resize((hr_height // 4, hr_width // 4), SAMPLING[randrange(0,len(SAMPLING))]),
                transforms.Resize((hr_height // 8, hr_width // 8), SAMPLING[randrange(0,len(SAMPLING))]),
                transforms.ToTensor()
            ]
        )



os.makedirs("images/outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model and load model checkpoint
#EDIT: added num_upsample argument explicitly
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks, num_upsample=3).to(device)
generator.load_state_dict(torch.load(opt.checkpoint_model))
generator.eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

# Prepare input
#image_tensor = Variable(transform(Image.open(opt.image_path))).to(device).unsqueeze(0)
for img in images:
	image_tensor = Variable(lr_transform(cv2.imread(img,\
	              cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))).to(device).unsqueeze(0)

	# Upsample image
	with torch.no_grad():
		#EDIT: removed denormalize
	    sr_image = generator(image_tensor).cpu()

	# Save image
	#fn = opt.image_path.split("/")[-1]
	fn = img.split("/")[-1]
	print("img: "+img)
	print("fn: "+fn)
	filename = "images/outputs/sr-"+fn
	cv2.imwrite(filename, sr_image)
	#save_image(sr_image, f"images/outputs/sr-{fn}")
