import os
import glob


i = 0
for input_file, mask_file in zip(sorted(glob.glob('./train_images/train_images_original*.png')),
                                 sorted(glob.glob('./train_labels/_groundtruth_(1)_train_images*.png'))):

    os.rename(input_file, './train_images/augmented_' + str(i) + '.png')
    os.rename(mask_file, './train_labels/augmented_' + str(i) + '.png')
    i = i + 1