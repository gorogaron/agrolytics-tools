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
p.rotate(0.6, max_left_rotation=25, max_right_rotation=25)
p.flip_top_bottom(0.6)
p.flip_random(0.6)
p.skew_tilt(0.6)
p.random_distortion(0.6, 8, 8, 8)
p.shear(0.6, 25, 25)
p.zoom_random(0.6, percentage_area=0.6)

p.sample(924)
