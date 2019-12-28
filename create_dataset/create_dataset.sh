IMG_SRC='../../dataset/images'
MASK_SRC='../../dataset/masks'

TRAIN_IMG_DST='../../agrolytics-models/vggsegnet/train_images'
TRAIN_MASK_DST='../../agrolytics-models/vggsegnet/train_labels'
VAL_IMG_DST='../../agrolytics-models/vggsegnet/val_images'
VAL_MASK_DST='../../agrolytics-models/vggsegnet/val_labels'
NUM_OF_AUGMENT=128
SIZE='(640, 480)'
PADDING=0

python create_dataset.py -ti=$TRAIN_IMG_DST -tm=$TRAIN_MASK_DST -vi=$VAL_IMG_DST -vm=$VAL_MASK_DST -s="$SIZE" -p=$PADDING
python ./../albumentations/albumentation.py -i=$IMG_SRC -m=$MASK_SRC -id=$TRAIN_IMG_DST -md=$TRAIN_MASK_DST -s="$SIZE" -q=$NUM_OF_AUGMENT -p=$PADDING