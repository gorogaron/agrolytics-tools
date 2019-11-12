import cv2
import argparse
import os
import shutil
import sys
from tqdm import tqdm

sys.path.append("../albumentations")
from padding_resize import padding_resize
#from albumentation import albumentation

def create_dir(directory):
    path, folder = os.path.split(trainimg_dir)

    if not os.path.exists(os.path.join(os.getcwd(), path)):
        sys.exit("ERROR: Path does not exist: " + path)

    if os.path.exists(os.path.join(os.getcwd(), directory)):
        print('Overwriting already existing output directory: ' + os.path.join(os.getcwd(), directory))
        shutil.rmtree(os.path.join(os.getcwd(), directory))
    else:
        print('Creating new directory: ' + os.path.join(os.getcwd(), directory))
    os.makedirs(os.path.join(os.getcwd(), directory))


parser = argparse.ArgumentParser()
parser.add_argument('-ti', '--trainimg', type=str, required=True)
parser.add_argument('-tm', '--trainmask', type=str, required=True)
parser.add_argument('-vi', '--valimg', type=str, required=True)
parser.add_argument('-vm', '--valmask', type=str, required=True)
parser.add_argument('-s', '--size', type=int, required=True)
args = parser.parse_args()

trainimg_dir = args.trainimg
trainmask_dir = args.trainmask
valimg_dir = args.valimg
valmask_dir = args.valmask
size = args.size

create_dir(trainimg_dir)
create_dir(trainmask_dir)
create_dir(valimg_dir)
create_dir(valmask_dir)

img_path = '../../dataset/images'
mask_path = '../../dataset/masks'

f = open('train.txt')
i = 0
for line in f:
    img_src = os.path.join(img_path, line.rstrip())
    img_dst = os.path.join(os.getcwd(),trainimg_dir,line.rstrip())

    mask_src = os.path.join(mask_path, os.path.splitext(line.rstrip())[0] + '.png')
    mask_dst = os.path.join(os.getcwd(),trainmask_dir,os.path.splitext(line.rstrip())[0] + '.png')

    img = cv2.imread(img_src)
    mask = cv2.imread(mask_src)
    img, mask = padding_resize(img, mask, size)

    cv2.imwrite(img_dst, img)
    cv2.imwrite(mask_dst, mask)
    i = i + 1
    print('Original training img/mask copied: ' + str(i), end = '\r')
print('Training data has been copied and resized.' + str(i))

i = 0
f = open('val.txt')
for line in f:
    img_src = os.path.join(img_path, line.rstrip())
    img_dst = os.path.join(os.getcwd(),valimg_dir,line.rstrip())

    mask_src = os.path.join(mask_path, os.path.splitext(line.rstrip())[0] + '.png')
    mask_dst = os.path.join(os.getcwd(),valmask_dir,os.path.splitext(line.rstrip())[0] + '.png')

    img = cv2.imread(img_src)
    mask = cv2.imread(mask_src)
    img, mask = padding_resize(img, mask, size)

    cv2.imwrite(img_dst, img)
    cv2.imwrite(mask_dst, mask)
    i = i + 1
    print('Original validation img/mask copied: ' + str(i), end = '\r')
print('Validation data has been copied and resized.' + str(i))