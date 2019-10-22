import cv2
import argparse
import os
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--imgdir', type=str, required=True)
parser.add_argument('-m', '--maskdir', type=str, required=True)
#parser.add_argument('-s', '--size', type=int, required=True)
args = parser.parse_args()

#desired_size = args.size
imagedir = args.imgdir
maskdir = args.maskdir

for r, d, f in os.walk(args.imgdir):
    for imagename in f:
        maskname = os.path.splitext(imagename)[0] + '.png'
        print(os.path.join(imagedir, imagename))
        im = cv2.imread(os.path.join(imagedir, imagename))
        mask = cv2.imread(os.path.join(maskdir, maskname))*255

        height, width = im.shape[:2]
        #mask_width = mask.shape[:2][0]
        
        # Create mask image
        foreground = np.zeros((height,width,3), np.uint8)
        foreground[:,:] = (0, 0, 255)
        background = im

        #mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        alpha = mask/2

        foreground = cv2.resize(foreground, ( width , height )).astype(float)
        background = cv2.resize(background, ( width , height )).astype(float)
        alpha = alpha.astype(float)/255
        foreground = cv2.multiply(alpha, foreground)
        background = cv2.multiply(1.0 - alpha, background)
        outImage = cv2.add(foreground, background).astype(np.uint8)
        cv2.imwrite(maskname, cv2.resize(outImage, (640, 480)))