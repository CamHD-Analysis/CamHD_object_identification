#!/usr/bin/env python3

"""
Train a U-Net segmentation model.

TODO: Currently the selected models are persisted as a model_config after the training done using this script.

"""

from models import unet
from data_prep import train_generator
from keras.callbacks import TensorBoard, ModelCheckpoint
from skimage import io

import argparse
import logging
import numpy as np
import os
import random
import shutil

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args():
    parser = argparse.ArgumentParser(description="Run the Training Pipeline. Currently supports only Unet.")
    parser.add_argument('--func',
                        required=True,
                        help="Specify the function to be called. The available list of functions: ['train_unet', 'test_unet'].")
    parser.add_argument('--data-dir',
                        help="The path to the data directory containing the patches and masks directories. "
                             "The masks directory is assumed to have names of corresponding patches with suffix - '_mask'."
                             "Valid for functions: 'train_unet'.")
    parser.add_argument('--patches-dirname',
                        default="patches",
                        help="The name of the data sub-dir containing patches."
                             "Valid for functions: 'train_unet'.")
    parser.add_argument('--masks-dirname',
                        default="masks",
                        help="The name of the data sub-dir containing masks."
                             "Valid for functions: 'train_unet'.")
    parser.add_argument('--val-split',
                        type=float,
                        default=0.20,
                        help="The validation split ratio. Default: 0.20."
                             "Valid for functions: 'train_unet'.")
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help="The number of epochs to be run. Default: 100."
                             "Valid for functions: 'train_unet'.")
    parser.add_argument('--batch-size',
                        type=int,
                        default=32,
                        help="The batch_size for training. Default: 32."
                             "Valid for functions: 'train_unet'.")
    parser.add_argument('--model-outfile',
                        required=True,
                        help="The path to the model output file (HDF5 file)."
                             "Valid for functions: 'train_unet', 'test_unet'.")
    parser.add_argument('--tensorboard-logdir',
                        help="The path to the Tensorboard log directory. If not provided, tensorboard logs will not be written."
                             "Valid for functions: 'train_unet'.")
    parser.add_argument('--image-ext',
                        dest="img_ext",
                        default='png',
                        help="The image file extension. Default: png."
                             "Valid for functions: 'train_unet', 'test_unet'.")
    parser.add_argument('--test-dir',
                        help="The path to the test data directory containing the test patches. "
                             "Valid for functions: 'test_unet'.")
    parser.add_argument('--test-output-dir',
                        help="The path to the output directory where the predictions need to be written."
                             "Valid for functions: 'test_unet'.")
    parser.add_argument("--log",
                        default="INFO",
                        help="Specify the log level. Default: INFO.")

    return parser.parse_args()


def _get_mask_name(patch_name, img_ext):
    return patch_name.rstrip(".%s" % img_ext) + "_mask.%s" % img_ext


def get_train_val_split(data_dir, val_split, patches_dirname, masks_dirname, img_ext):
    if val_split > 0.5:
        raise ValueError("The validation split of 0.5 and above are not accepted.")

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

        mask_name = _get_mask_name(patch_name, img_ext)

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


def train_unet(args):
    if not args.data_dir:
        raise ValueError("The data-dir has not to been provided for the function 'train_unet'.")

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
    model_checkpoint = ModelCheckpoint(args.model_outfile, monitor='val_loss', mode='auto', verbose=1, save_best_only=True)
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

    logging.info("The trained model has been saved in %s" % args.model_outfile)


def test_unet(args):
    if not args.test_dir:
        raise ValueError("The test-dir has not to been provided for the function 'test_unet'.")

    if not os.path.exists(args.test_dir):
        raise ValueError("The test-dir does not exist: %s" % args.test_dir)

    if not args.test_output_dir:
        raise ValueError("The test-output-dir has not to been provided for the function 'test_unet'.")

    if not os.path.exists(args.test_output_dir):
        os.makedirs(args.test_output_dir)

    model = unet()
    model.load_weights(args.model_outfile)

    for patch_name in os.listdir(args.test_dir):
        test_patch_path = os.path.join(args.test_dir, patch_name)
        pred_mask_path = os.path.join(args.test_output_dir, _get_mask_name(patch_name, args.img_ext))

        img = io.imread(test_patch_path)
        img = img / 255
        img = np.reshape(img, (1, ) + img.shape)

        pred_mask = model.predict(img)
        # TODO: Take the mask_index from the model_config.
        pred_mask = pred_mask[0] * 255
        pred_mask = pred_mask.astype(np.uint8)
        pred_mask = np.reshape(pred_mask, pred_mask.shape[:-1])
        io.imsave(pred_mask_path, pred_mask)

    logging.info("The predictions have been written to %s" % args.test_output_dir)


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=args.log.upper())

    if   args.func == "train_unet":
        train_unet(args)
    elif args.func == "test_unet":
        test_unet(args)
    else:
        raise ValueError("The given function is not supported: %s" % args.func)
