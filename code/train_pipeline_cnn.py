#!/usr/bin/env python3

"""
Train a CNN classification model.

# Ref: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# Ref: https://keras.io/applications/#vgg16

"""

from models import cnn_amphipod

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator

import argparse
import json
import logging
import os
import random
import shutil


# Modify Keras Session:
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# TODO: This could be generalized once the script allows multiple types of models by reading form model_config.
TARGET_SIZE = (256, 256, 3)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args():
    parser = argparse.ArgumentParser(description="Run the Training Pipeline for training CNN models.")
    parser.add_argument('--func',
                        required=True,
                        help="Specify the function to be called. The available list of functions: ['train_cnn', 'test_cnn'].")
    parser.add_argument('--data-dir',
                        help="The path to the data directory containing the images corresponding to each class label. "
                             "The images of each class label must be organized into separate directory having the "
                             "the name of the corresponding class label."
                             "Valid for functions: 'train_cnn'.")
    parser.add_argument('--classes',
                        required=True,
                        help="The set of classes to be considered. Provide comma separated string.")
    parser.add_argument('--val-split',
                        type=float,
                        default=0.20,
                        help="The validation split ratio. Default: 0.20."
                             "Valid for functions: 'train_cnn'.")
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help="The number of epochs to be run. Default: 100."
                             "Valid for functions: 'train_cnn'.")
    parser.add_argument('--batch-size',
                        type=int,
                        default=32,
                        help="The batch_size for training. Default: 32."
                             "Valid for functions: 'train_cnn'.")
    parser.add_argument('--model-outfile',
                        required=True,
                        help="The path to the model output file (HDF5 file)."
                             "Valid for functions: 'train_cnn', 'test_cnn'.")
    parser.add_argument('--tensorboard-logdir',
                        help="The path to the Tensorboard log directory. If not provided, tensorboard logs will not be written."
                             "Valid for functions: 'train_cnn'.")
    parser.add_argument('--image-ext',
                        dest="img_ext",
                        default='png',
                        help="The image file extension. Default: png."
                             "Valid for functions: 'train_cnn', 'test_cnn'.")
    parser.add_argument('--test-dir',
                        help="The path to the test data directory containing the test patches. "
                             "Valid for functions: 'test_cnn'.")
    parser.add_argument('--test-output-path',
                        help="The path to the output file where the predictions (csv) need to be written."
                             "Valid for functions: 'test_cnn'.")
    parser.add_argument("--log",
                        default="INFO",
                        help="Specify the log level. Default: INFO.")

    return parser.parse_args()


# Stratified splitting.
def get_train_val_split(data_dir, val_split, allow_exist=True):
    split_exists = False
    if val_split > 0.5:
        raise ValueError("The validation split of 0.5 and above are not accepted.")

    train_dir = data_dir + "_train"
    if os.path.exists(train_dir):
        if not allow_exist:
            raise ValueError("The train_dir already exists: %s" % train_dir)
        split_exists = True
    else:
        os.makedirs(train_dir)

    val_dir = data_dir + "_val"
    if os.path.exists(val_dir):
        if not allow_exist:
            raise ValueError("The val_dir already exists: %s" % val_dir)
    else:
        if split_exists:
            raise ValueError("The train_dir already exists but val_dir doesn't exist.")

        os.makedirs(val_dir)

    if split_exists:
        train_size = 0
        val_size = 0

        labels = os.listdir(data_dir)
        for label in labels:
            train_label_dir_path = os.path.join(train_dir, label)
            train_size += len(os.listdir(train_label_dir_path))

            val_label_dir_path = os.path.join(val_dir, label)
            val_size += len(os.listdir(val_label_dir_path))

        return train_dir, val_dir, train_size, val_size

    # Create new splits.
    train_size = 0
    val_size = 0

    # Each of these labels must be directories.
    labels = os.listdir(data_dir)
    for label in labels:
        orig_label_dir_path = os.path.join(data_dir, label)
        if not os.path.isdir(orig_label_dir_path):
            raise ValueError("The data doesn't seem to have been correctly organized.")

        train_label_dir_path = os.path.join(train_dir, label)
        os.makedirs(train_label_dir_path)
        val_label_dir_path = os.path.join(val_dir, label)
        os.makedirs(val_label_dir_path)

        img_names = os.listdir(orig_label_dir_path)
        random.shuffle(img_names)

        num_imgs = len(img_names)
        cur_train_size = int(num_imgs * (1 - val_split))

        c = 0
        for img_name in img_names:
            c += 1

            src_img_path = os.path.join(orig_label_dir_path, img_name)

            # Stratified sub-sampling with respect to cur_train_size.
            if c <= cur_train_size:
                train_size += 1
                dest_img_path = os.path.join(train_label_dir_path, img_name)
            else:
                val_size += 1
                dest_img_path = os.path.join(val_label_dir_path, img_name)

            shutil.copy(src_img_path, dest_img_path)

    return train_dir, val_dir, train_size, val_size


def train_cnn(args):
    # Data preparation
    train_dir, val_dir, train_size, val_size = get_train_val_split(args.data_dir, args.val_split)

    # Load the model and compile.
    model = cnn_amphipod(len(args.classes), input_size=TARGET_SIZE, batch_norm=True)

    # Setup data loading.
    # This is the augmentation configuration we will use for training.
    # TODO: Allow specification of different augmentations through config.
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=0.2,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.1,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    # This is the augmentation configuration we will use for testing: only rescaling.
    test_datagen = ImageDataGenerator(rescale=1./255)

    # This is a generator that will read pictures found in subfolders of 'data/train', and indefinitely generate
    # batches of augmented image data.
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=TARGET_SIZE[:-1],
                                                        batch_size=args.batch_size,
                                                        color_mode="rgb",
                                                        class_mode='categorical',
                                                        classes=args.classes,
                                                        seed=1)

    # This is a similar generator, for validation data.
    validation_generator = test_datagen.flow_from_directory(val_dir,
                                                            target_size=TARGET_SIZE[:-1],
                                                            batch_size=args.batch_size,
                                                            color_mode="rgb",
                                                            class_mode='categorical',
                                                            classes=args.classes,
                                                            seed=1)

    # Setup training.
    model_checkpoint = ModelCheckpoint(args.model_outfile, monitor='val_acc', mode='auto', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
    callbacks = [model_checkpoint, early_stopping]
    if args.tensorboard_logdir:
        tensorboard = TensorBoard(log_dir=args.tensorboard_logdir)
        logging.info("To view tensorboard, run: 'tensorboard --logdir=%s'" % args.tensorboard_logdir)
        callbacks.append(tensorboard)

    model.fit_generator(train_generator,
                        steps_per_epoch=train_size/args.batch_size,
                        validation_data=validation_generator,
                        validation_steps=val_size/args.batch_size,
                        epochs=args.epochs,
                        shuffle=True,
                        callbacks=callbacks)

    logging.info("The trained model has been saved in %s" % args.model_outfile)

    # Create a model config for the trained model.
    model_config = {
        "model_name": "%s" % os.path.basename(args.data_dir), # Creating model_name from the basename of the data dir.
        "train_data_desc": "Path: %s." % args.data_dir,
        "model_path": "%s" % os.path.abspath(args.model_outfile),
        "type": "classification",
        "input_shape": [256, 256, 3], # To be checked by user.
        "rescale": True, # To be checked by user.
        "valid_class": "amphipod",
        "classes": [
            "amphipod",
            "nonamphipod"
        ],
        "prob_thresholds": {
            "amphipod": 0.7,
            "nonamphipod": 0.3
        },
        "adjust_patch_size": True # To be checked by user depending on the train data provided and the analysis pipeline.
    }

    model_config_path = os.path.splitext(args.model_outfile)[0] + ".json"
    with open(model_config_path, 'w') as fp:
        json.dump(model_config, fp, sort_keys=True, indent=4)

    logging.info("The model config for the train model has been saved in %s. Please verify and update description." % model_config_path)


def test_cnn(args):
    pass


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=args.log.upper())
    args.classes = args.classes.split(",")
    if args.func == "train_cnn":
        train_cnn(args)
    elif args.func == "test_cnn":
        test_cnn(args)
    else:
        raise ValueError("The given function is not supported: %s" % args.func)
