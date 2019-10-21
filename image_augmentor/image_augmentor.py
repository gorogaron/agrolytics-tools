#!/usr/bin/env python
import numpy as np
import argparse
import cv2
import os
import glob
import shutil
import Augmentor

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputdir', type=str, required=True)
parser.add_argument('-m', '--maskdir', type=str, required=True)
parser.add_argument('-s', '--sample', type=int, required=True)
args = parser.parse_args()

INPUT_PATH = os.path.normpath(os.path.join(os.getcwd(), args.inputdir))
MASK_PATH = os.path.normpath(os.path.join(os.getcwd(), args.maskdir))

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
p.random_color(0.3,0.5,2.5)
p.histogram_equalisation(0.1)

p.sample(args.sample)

## Moving files to another directory and rename them
i = 0
for input_file, mask_file in zip(sorted(glob.glob(os.path.join(INPUT_PATH, 'output', 'train_images_original*.png'))),
                                 sorted(glob.glob(os.path.join(INPUT_PATH, 'output', '_groundtruth_(1)_train_images*.png')))):

    shutil.move(input_file, INPUT_PATH)
    shutil.move(mask_file, MASK_PATH)

    new_input_filename, new_mask_filename = os.path.basename(input_file), os.path.basename(mask_file)
    new_input_path, new_mask_path = os.path.join(INPUT_PATH, new_input_filename), os.path.join(MASK_PATH, new_mask_filename)

    os.rename(new_input_path, os.path.join(INPUT_PATH, 'augmented_' + str(i) + '.png'))
    os.rename(new_mask_path, os.path.join(MASK_PATH, 'augmented_' + str(i) + '.png'))
    i = i + 1

## Remove the original output directory
os.rmdir(os.path.join(INPUT_PATH, 'output'))