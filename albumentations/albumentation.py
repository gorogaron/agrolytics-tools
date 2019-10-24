import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt
from random import uniform
import os
from os import listdir
from os.path import isfile, join
import shutil

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,    
    RandomGamma    
)

#ARGUMENTS:
#imgdir   - directory of images
#maskdir  - directory of corresponding masks
#quantity - number of images to be generated
#size     - TODO

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--imgdir', type=str, required=True)
parser.add_argument('-m', '--maskdir', type=str, required=True)
parser.add_argument('-s', '--size', type=int, required=True)
parser.add_argument('-q', '--quantity', type=int, required=True)

args = parser.parse_args()

imgdir = args.imgdir
maskdir = args.maskdir
size = args.size
quantity = args.quantity

def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18
    
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))
        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
        
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)


image_filenames = [f for f in listdir(imgdir) if isfile(join(imgdir, f))]

output_dir = './output'
if os.path.exists(output_dir):
    print('Deleting already existing output directory: ' + output_dir)
    shutil.rmtree(output_dir)
os.makedirs(output_dir)
os.makedirs(output_dir + '/images')
os.makedirs(output_dir + '/masks')

print('Creating ' + str(quantity) + ' images into: ./output/images, ./output/masks...')

i=0
while i < quantity:
    img_name = image_filenames[i%quantity]
    mask_name =  img_name[0:-4] + '.png'

    image = cv2.imread(os.path.join(imgdir,img_name))
    mask = cv2.imread(os.path.join(maskdir,mask_name))

    original_height, original_width = mask.shape[:2]

    #Cropping random area
    x_min = (int(original_width*uniform(0,0.2))) 
    y_min = (int(original_height*uniform(0,0.2))) 

    x_max = (int(original_width*uniform(0.8,1)))
    y_max = (int(original_height*uniform(0.8,1)))

    aug = Compose([
        Crop(p=1, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.33),
        GridDistortion(p=0.5),
        RandomBrightnessContrast(p = 0.75, brightness_limit=0.3, contrast_limit=0.3)
    ])
    augmented = aug(image=image, mask=mask)

    image_augmented = augmented['image']
    mask_augmented = augmented['mask']*250

    cv2.imwrite('./output/images/augmented_' + str(i) + '.png', image_augmented)
    cv2.imwrite('./output/masks/augmented_' + str(i) + '.png', mask_augmented)
    i = i + 1
    

#TODO: Appropriate parametrization
#aug = ElasticTransform(p=1, alpha=original_width*0.03, sigma=original_width*0.004, alpha_affine=original_width * 0.02)
#aug = OpticalDistortion(p=1, distort_limit=1.1, shift_limit=0.5)