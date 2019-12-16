IMG_SRC='../../dataset/images'
MASK_SRC='../../dataset/masks'

TRAIN_IMG_DST='train_images'
TRAIN_MASK_DST='train_labels'
VAL_IMG_DST='val_images'
VAL_MASK_DST='val_labels'
NUM_OF_AUGMENT=10
SIZE='(640, 480)'
PADDING=1

python create_dataset.py -ti=$TRAIN_IMG_DST -tm=$TRAIN_MASK_DST -vi=$VAL_IMG_DST -vm=$VAL_MASK_DST -s="$SIZE" -p=$PADDING
python ./../albumentations/albumentation.py -i=$IMG_SRC -m=$MASK_SRC -id=$TRAIN_IMG_DST -md=$TRAIN_MASK_DST -s="$SIZE" -q=$NUM_OF_AUGMENT -p=$PADDING