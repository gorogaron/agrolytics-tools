# agrolytics-tools
Tools for Agrolytics are contained here.


## albumentations
Augmentation [library](https://github.com/albu/albumentations) for segmentation dataset.

Arguments:
<pre>
--imgdir      Directory of original sized images
--maskdir     Directory of original sized masks
--imgdestdir  Output directory of augmented images
--maskdestdir Output director of augmented masks
--size        Size of output images. Output images will be resized to (w x w) with padding_resize.py
--quantity    Number of augmented images to be generated
</pre>




## create_dataset
Contains script to create train and validation directories with images and masks for training. The "dataset" directory must be placed next to agrolytics-tools.

 * train.txt: Training image filenames
 * val.txt: Validation image filenames

Arguments:
<pre>
--trainimg    Destination of training images
--trainmask   Destination of training masks
--valimg      Destination of validation images
--valmask     Destination of validation masks
--size        Size of output images. Output images will be resized to (w x w) with padding_resize.py
</pre>

Run **create_dataset.sh** to automatically create and augment dataset for the training scripts. Modify parameters inside the shell script.

## visualizer
The script draws the masks onto the corresponding images.

Arguments:
<pre>
--imgdir    Directory of images
--maskdir   Directory of masks
</pre>
