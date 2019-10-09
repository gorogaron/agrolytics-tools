#!/usr/bin/env python
import glob
import numpy as np
import json
import argparse
import cv2
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputdir', type=str, required=True)
parser.add_argument('-o', '--outputdir', type=str, required=True)
args = parser.parse_args()

JSON_DIR = os.path.normpath(os.path.join(os.getcwd(), args.inputdir))
PNG_DIR = os.path.normpath(os.path.join(os.getcwd(), args.outputdir))

for input_file in tqdm(glob.glob(os.path.join(JSON_DIR, '*.json')), ascii=True, desc='Converting'):
    file_name = os.path.splitext(os.path.basename(input_file))[0]

    with open(input_file) as json_file:
        json_data = json.load(json_file)

    width, height = json_data['output']['image']['width'], json_data['output']['image']['height']
    image_arr = np.zeros((height, width), dtype=np.int32) 
    for obj_elem in np.asarray(json_data['output']['objects']):
        contours = np.round(np.asarray(obj_elem['points']['exterior'], dtype=np.int32))
        cv2.fillPoly(image_arr, [contours], color=(1, 1, 1))
    cv2.imwrite(os.path.join(PNG_DIR, file_name + '.png'), image_arr)

print('Converting done!')