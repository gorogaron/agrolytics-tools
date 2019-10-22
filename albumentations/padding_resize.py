import cv2
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--imgdir', type=str, required=True)
parser.add_argument('-m', '--maskdir', type=str, required=True)
parser.add_argument('-s', '--size', type=int, required=True)
parser.add_argument('-o', '--outdir', type=str, required=True)

args = parser.parse_args()

desired_size = args.size
imagedir = args.imgdir
maskdir = args.maskdir
outdir = args.outdir
os.makedirs(outdir)
os.makedirs(outdir + '/images')
os.makedirs(outdir + '/masks')

for _, _, f in os.walk(args.imgdir):
    for imagename in f:
        print(imagename)
        maskname = os.path.splitext(imagename)[0] + '.png'
        im = cv2.imread(os.path.join(imagedir, imagename))
        mask = cv2.imread(os.path.join(maskdir, maskname))*255

        old_size = im.shape[:2] # old_size is in (height, width) format

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        im = cv2.resize(im, (new_size[1], new_size[0]))
        mask = cv2.resize(mask, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)
        new_mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)
        cv2.imwrite(os.path.join(outdir, 'masks/', maskname), new_mask)
        cv2.imwrite(os.path.join(outdir, 'images/', maskname), new_im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()