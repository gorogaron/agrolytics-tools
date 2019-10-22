########################################################################
#This is just a sketch version for converting labelme json files to png.
########################################################################

import cv2
import argparse
import os
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--imgdir', type=str, required=True)
parser.add_argument('-m', '--maskdir', type=str, required=True)
args = parser.parse_args()

imagedir = args.imgdir
maskdir = args.maskdir

for r, d, f in os.walk(args.imgdir):
    for imagename in f:
        maskname = os.path.splitext(imagename)[0] + '.png'

        im = cv2.imread(os.path.join(imagedir, imagename))
        mask = cv2.imread(os.path.join(maskdir, maskname))

        im_width = im.shape[:2][0]
        mask_width = mask.shape[:2][0]
        
        if mask_width < im_width:
            #print(os.path.isfile('../../dataset/etc/labelme/json/' + os.path.splitext(imagename)[0] + '.json'))
            print('../../dataset/etc/labelme/json/' + os.path.splitext(imagename)[0] + '.json')
            img = cv2.imread(os.path.join(imagedir, imagename))
            height, width = img.shape[:2]
            blank_img = np.zeros((height, width), dtype=np.int32) 
            jsonFilename = '../../dataset/etc/labelme/json/' + os.path.splitext(imagename)[0] + '.json'
            with open(jsonFilename) as json_file:
                json_data = json.load(json_file)
                numOfObjects = np.asarray(json_data['shapes']).shape
                for element in json_data['shapes']:
                    if (element['label'] == 'fa'):
                        pointArray = np.asarray(element['points'], dtype=np.int32)
                        pointArray[:,0] = pointArray[:,0]
                        pointArray[:,1] = pointArray[:,1]
                        contours = np.round(pointArray)
                        cv2.fillPoly(blank_img, [pointArray], color=(255, 255, 255))
                cv2.imwrite('./' + os.path.splitext(imagename)[0] + '.png', blank_img/255)