#!/usr/bin/env python3

from models import unet
from data_prep import train_generator
from keras.callbacks import TensorBoard, ModelCheckpoint

import logging
import os
import random
import shutil

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args():
    parser = argparse.ArgumentParser(description="Run the Training Pipeline. Currently supports only Unet.")
    parser.add_argument('--data-dir',
                        required=True,
                        help="The path to the data directory containing the patches and masks directories. "
                             "The masks directory is assumed to have names of corresponding patches with suffix - '_mask'.")
    parser.add_argument('--patches-dirname',
                        default="patches",
                        help="The name of the data sub-dir containing patches.")
    parser.add_argument('--masks-dirname',
                        default="masks",
                        help="The name of the data sub-dir containing masks.")
    parser.add_argument('--val-split',
                        type=float,
                        default=0.20,
                        help="The validation split ratio. Default: 0.20.")
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help="The number of epochs to be run. Default: 100.")
    parser.add_argument('--batch-size',
                        type=int,
                        default=32,
                        help="The batch_size for training. Default: 32.")
    parser.add_argument('--outfile',
                        required=True,
                        help="The path to the output file (HDF5 file).")
    parser.add_argument('--tensorboard-logdir',
                        help="The path to the Tensorboard log directory. If not provided, tensorboard logs will not be written.")
    parser.add_argument('--image-ext',
                        default='png',
                        help="The image file extension. Default: png.")

    return parser.parse_args()


def get_train_val_split(data_dir, val_split, patches_dirname, masks_dirname, img_ext):
    if val_split > 0.5:
        raise ValueError("The validation split of0.5 and above are not accepted.")

    train_dir = data_dir + "_train"
    if os.path.exists(train_dir):
        raise ValueError("The train_dir already exists: %s" % train_dir)
    os.makedirs(train_dir)

    train_patches_dir = os.path.join(train_dir, patches_dirname)
    train_masks_dir   = os.path.join(train_dir, masks_dirname)
    os.makedirs(train_patches_dir)
    os.makedirs(train_masks_dir)

    val_dir = data_dir + "_val"
    if os.path.exists(val_dir):
        raise ValueError("The val_dir already exists: %s" % val_dir)
    os.makedirs(val_dir)

    val_patches_dir = os.path.join(val_dir, patches_dirname)
    val_masks_dir   = os.path.join(val_dir, masks_dirname)
    os.makedirs(val_patches_dir)
    os.makedirs(val_masks_dir)

    orig_patches_dir = os.path.join(data_dir, patches_dirname)
    orig_masks_dir   = os.path.join(data_dir, masks_dirname)

    patch_names = os.listdir(orig_patches_dir)
    random.shuffle(patch_names)

    num_patches = len(patch_names)
    train_size  = int(num_patches * (1 - val_split))

    c = 0
    val_size = 0
    for patch_name in patch_names:
        c += 1

        mask_name = patch_name.rstrip(".%s" % img_ext) + "_mask.%s" % img_ext

        src_patch_path = os.path.join(orig_patches_dir, patch_name)
        src_mask_path  = os.path.join(orig_masks_dir, mask_name)

        if c <= train_size:
            dest_patch_path = os.path.join(train_patches_dir, patch_name)
            dest_mask_path  = os.path.join(train_masks_dir, mask_name)
        else:
            val_size += 1
            dest_patch_path = os.path.join(val_patches_dir, patch_name)
            dest_mask_path  = os.path.join(val_masks_dir, mask_name)

        shutil.copy(src_patch_path, dest_patch_path)
        shutil.copy(src_mask_path, dest_mask_path)

    return train_dir, val_dir, train_size, val_size


if __name__ == "__main__":
    args = get_args()
    augmentation_args = dict(rotation_range=0.2,
                             width_shift_range=0.05,
                             height_shift_range=0.05,
                             shear_range=0.05,
                             zoom_range=0.1,
                             horizontal_flip=True)

    train_dir, val_dir, train_size, val_size = get_train_val_split(args.data_dir,
                                                                   args.val_split,
                                                                   args.patches_dirname,
                                                                   args.masks_dirname,
                                                                   args.img_ext)

    train_gen = train_generator(args.batch_size, train_dir, args.patches_dirname, args.masks_dirname, augmentation_args)
    val_gen   = train_generator(args.batch_size, val_dir, args.patches_dirname, args.masks_dirname, {})

    model = unet()
    model_checkpoint = ModelCheckpoint(args.outfile, monitor='val_loss', mode='auto', verbose=1, save_best_only=True)
    callbacks = [model_checkpoint]
    if args.tensorboard_logdir:
        tensorboard = TensorBoard(log_dir=args.tensorboard_logdir)
        logging.info("To view tensorboard, run: 'tensorboard --logdir=%s'" % args.tensorboard_logdir)
        callbacks.append(tensorboard)

    model.fit_generator(train_gen,
                        steps_per_epoch=train_size/args.batch_size,
                        validation_data=val_gen,
                        validation_steps=val_size/args.batch_size,
                        epochs=args.epochs,
                        shuffle=True,
                        callbacks=callbacks)

    """
    testGene = testGenerator("data/membrane/test")
    results = model.predict_generator(testGene, 30, verbose=1)
    saveResult("data/membrane/test", results)
    """
