#!/usr/bin/env python
import numpy as np
import argparse
import cv2
import os
import glob
import Augmentor

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputdir', type=str, required=True)
parser.add_argument('-m', '--maskdir', type=str, required=True)
args = parser.parse_args()

INPUT_PATH = os.path.normpath(os.path.join(os.getcwd(), args.inputdir))
MASK_PATH = os.path.normpath(os.path.join(os.getcwd(), args.maskdir))
print(INPUT_PATH)
print(MASK_PATH)

p = Augmentor.Pipeline(INPUT_PATH)

p.ground_truth(MASK_PATH)
p.skew(0.3,0.2)
p.zoom(0.3,1.01,1.1)
p.rotate(0.2, max_left_rotation=10, max_right_rotation=10)
p.rotate_without_crop(0.1,180,180)
p.flip_left_right(0.5)
p.flip_top_bottom(0.25)
p.random_distortion(0.3, 10, 10, 10)
p.shear(0.3, 18, 18)
p.random_contrast(0.3,0.5,1.7)
p.random_color(0.3,0.5,2.5)
p.random_brightness(0.3,0.5,1.5)
p.histogram_equalisation(0.1)


p.sample(50)