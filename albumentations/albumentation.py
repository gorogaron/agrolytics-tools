import cv2
from tqdm import tqdm
import argparse
from random import uniform
import os
from os.path import isfile, join
import shutil
from padding_resize import padding_resize
import re

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
#imgdir      - directory of images
#maskdir     - directory of corresponding masks
#imgdestdir  - output directory of augmented images
#maskdestdir - output directory of augmented masks
#quantity    - number of images to be generated
#size        - width and height of generated images



parser = argparse.ArgumentParser()
parser.add_argument('-i', '--imgdir', type=str, required=True)
parser.add_argument('-m', '--maskdir', type=str, required=True)
parser.add_argument('-id', '--imgdestdir', type=str, required=True)
parser.add_argument('-md', '--maskdestdir', type=str, required=True)
parser.add_argument('-s', '--size', type=str, required=True)
parser.add_argument('-q', '--quantity', type=int, required=True)
parser.add_argument('-p', '--padding', type=bool)

args = parser.parse_args()

#Getting arguments
imgdir = args.imgdir
maskdir = args.maskdir
quantity = args.quantity
imgdestdir = os.path.join(os.getcwd(),args.imgdestdir)
maskdestdir = os.path.join(os.getcwd(),args.maskdestdir)

#Getting size argument
size_pattern = re.compile("^\([0-9]{1,}\,\s?[0-9]{1,}\)$", re.M)
padding_needed = 0
if (not re.match(size_pattern, args.size)):
    quit('ERROR: Format for size (-s) is not correct. Use the following format: --s=(w, h)')
else:
    size = re.findall('\d+', args.size)
    if (size[0] == size[1]):
        if args.padding is None:
            quit('ERROR: --padding argument has to be added if w==h')
        else:
            padding_needed = args.padding

image_filenames = [f for f in os.listdir(imgdir) if isfile(join(imgdir, f))]

if not (os.path.exists(imgdestdir) or os.path.exists(maskdestdir)):
    print('Creating new directory: ' + imgdestdir)
    print('Creating new directory: ' + maskdestdir)
    os.makedirs(imgdestdir)
    os.makedirs(maskdestdir)
print('Creating ' + str(quantity) + ' images into: ' + imgdestdir + ' , ' + maskdestdir)

for i in tqdm(range(quantity)):
    img_name = image_filenames[i%quantity]
    mask_name =  img_name[0:-4] + '.png'

    image = cv2.imread(os.path.join(imgdir,img_name))
    mask = cv2.imread(os.path.join(maskdir,mask_name))

    if (image is None or mask is None):
        quit('ERROR: Source images or masks not found.')
    original_height, original_width = mask.shape[:2]

    #Parameters for cropping random area
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
        RandomBrightnessContrast(p = 0.75, brightness_limit=0.3, contrast_limit=0.3),
        OpticalDistortion(p=0.5, distort_limit=0.75, shift_limit=0)
    ])
    augmented = aug(image=image, mask=mask)

    image_augmented = augmented['image']
    mask_augmented = augmented['mask']

    if (padding_needed):
        image_augmented_resized, mask_augmented_resized = padding_resize(image_augmented, mask_augmented, int(size[0]))
    else:
        image_augmented_resized = cv2.resize(image_augmented, (int(size[0]), int(size[1])))
        mask_augmented_resized = cv2.resize(mask_augmented, (int(size[0]), int(size[1])))

    cv2.imwrite(imgdestdir + '/augmented_' + str(i) + '.png', image_augmented_resized)
    cv2.imwrite(maskdestdir + '/augmented_' + str(i) + '.png', mask_augmented_resized)

#TODO: Appropriate parametrization
#aug = ElasticTransform(p=1, alpha=original_width*0.03, sigma=original_width*0.004, alpha_affine=original_width * 0.02)