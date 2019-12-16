import cv2
import argparse
import os
import shutil
import sys
import re
from tqdm import tqdm

sys.path.append("../albumentations")
from padding_resize import padding_resize

#Copies list of files in 'f' from dataset to desired training and valid dirs
def copy_data(f, method):
    i = 0
    for line in f:
        if (method == 'train'):
            imgdir = trainimg_dir
            maskdir = trainmask_dir
        elif (method == 'valid'):
            imgdir = valimg_dir
            maskdir = valmask_dir

        img_src = os.path.join(img_path, line.rstrip())
        img_dst = os.path.join(os.getcwd(),imgdir,line.rstrip())

        mask_src = os.path.join(mask_path, os.path.splitext(line.rstrip())[0] + '.png')
        mask_dst = os.path.join(os.getcwd(),maskdir,os.path.splitext(line.rstrip())[0] + '.png')

        img = cv2.imread(img_src)
        mask = cv2.imread(mask_src)
        
        if (padding_needed):
            img, mask = padding_resize(img, mask, int(size[0]))
        else:
            img = cv2.resize(img, (int(size[0]), int(size[1])))
            mask = cv2.resize(mask, (int(size[0]), int(size[1])))

        cv2.imwrite(img_dst, img)
        cv2.imwrite(mask_dst, mask)
        i = i + 1
        if (method == 'train'):
            print('Original training img/mask copied: ' + str(i), end = '\r')
        elif (method == 'valid'):
            print('Original valid img/mask copied: ' + str(i), end = '\r')
    return i

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
parser.add_argument('-s', '--size', type=str, required=True)
parser.add_argument('-p', '--padding', type=bool)

args = parser.parse_args()

trainimg_dir = args.trainimg
trainmask_dir = args.trainmask
valimg_dir = args.valimg
valmask_dir = args.valmask

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

create_dir(trainimg_dir)
create_dir(trainmask_dir)
create_dir(valimg_dir)
create_dir(valmask_dir)

img_path = '../../dataset/images'
mask_path = '../../dataset/masks'

f = open('train.txt')
num = copy_data(f,'train')
print('Training data has been copied and resized.' + str(num))

f = open('val.txt')
num = copy_data(f,'valid')
print('Validation data has been copied and resized.' + str(num))