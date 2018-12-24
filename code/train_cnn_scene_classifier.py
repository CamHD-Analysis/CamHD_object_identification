#!/usr/bin/env python3

"""
# Ref: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# Ref: https://keras.io/applications/#vgg16

"""

from keras.applications.vgg16 import VGG16
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

import argparse
import logging
import os
import random
import shutil


TARGET_SIZE = (224, 224) # TODO: This could be generalized once the script allows multiple types of models.
SCENE_TAGS_CLASSIFICATION_SPEC_STRING = "SCENE_TAGS"

# Standard CamHD scene_tags.
SCENE_TAGS = [
    'p0_z0',
    'p0_z1',
    'p0_z2',
    'p1_z0',
    'p1_z1',
    'p2_z0',
    'p2_z1',
    'p3_z0',
    'p3_z1',
    'p3_z2',
    'p4_z0',
    'p4_z1',
    'p4_z2',
    'p5_z0',
    'p5_z1',
    'p5_z2',
    'p6_z0',
    'p6_z1',
    'p6_z2',
    'p7_z0',
    'p7_z1',
    'p8_z0',
    'p8_z1'
]

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args():
    parser = argparse.ArgumentParser(description="Run the Training Pipeline. Currently supports only Unet.")
    parser.add_argument('--func',
                        required=True,
                        help="Specify the function to be called. The available list of functions: ['train_cnn', 'test_cnn'].")
    parser.add_argument('--data-dir',
                        help="The path to the data directory containing the patches and masks directories. "
                             "The masks directory is assumed to have names of corresponding patches with suffix - '_mask'."
                             "Valid for functions: 'train_cnn'.")
    parser.add_argument('--classes',
                        required=True,
                        help="The set of classes to be considered. Provide comma separated string. "
                             "Specify '%s' to classify the standard scene_tags in CamHD."
                             % SCENE_TAGS_CLASSIFICATION_SPEC_STRING)
    parser.add_argument('--deployment',
                        help="Must be provided if classes is '%s'. Specify the deployment version to be "
                             "prefixed to the standard scene_tags."
                             % SCENE_TAGS_CLASSIFICATION_SPEC_STRING)
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
def get_train_val_split(data_dir, val_split):
    if val_split > 0.5:
        raise ValueError("The validation split of0.5 and above are not accepted.")

    train_dir = data_dir + "_train"
    if os.path.exists(train_dir):
        raise ValueError("The train_dir already exists: %s" % train_dir)
    os.makedirs(train_dir)

    val_dir = data_dir + "_val"
    if os.path.exists(val_dir):
        raise ValueError("The val_dir already exists: %s" % val_dir)
    os.makedirs(val_dir)

    train_size = 0
    val_size = 0

    # Each of these labels must be directories.
    labels = os.listdir(data_dir)
    for label in labels:
        orig_label_dir_path = os.path.join(data_dir, label)
        if not os.isdir(orig_label_dir_path):
            raise ValueError("The data doesn't seem to have been correctly organized.")

        train_label_dir_path = os.path.join(train_dir, label)
        os.makedirs(train_label_dir_path)
        val_label_dir_path = os.path.join(val_dir, label)
        os.makedirs(val_label_dir_path)

        img_names = os.listdir(orig_label_dir_path)
        random.shuffle(img_names)

        num_imgs = len(img_names)
        train_size = int(num_imgs * (1 - val_split))

        c = 0
        for img_name in img_names:
            c += 1

            src_img_path = os.path.join(orig_label_dir_path, img_name)

            if c <= train_size:
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
    # TODO: Modify to allow different kinds of model to be loaded.
    model = VGG16(include_top=True, weights='imagenet', classes=len(args.classes))
    model.compile(optimizer=Adam(lr = 1e-4), loss=categorical_crossentropy, metrics=['accuracy'])

    # Setup data loading.
    # This is the augmentation configuration we will use for training.
    # TODO: Allow specification of different augmentations.
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2)

    # This is the augmentation configuration we will use for testing: only rescaling.
    test_datagen = ImageDataGenerator(rescale=1./255)

    # This is a generator that will read pictures found in subfolders of 'data/train', and indefinitely generate
    # batches of augmented image data.
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=TARGET_SIZE,
                                                        batch_size=args.batch_size,
                                                        color_mode="rgb",
                                                        class_mode='categorical',
                                                        classes=args.classes,
                                                        seed=1)

    # This is a similar generator, for validation data.
    validation_generator = test_datagen.flow_from_directory(val_dir,
                                                            target_size=TARGET_SIZE,
                                                            batch_size=args.batch_size,
                                                            color_mode="rgb",
                                                            class_mode='categorical',
                                                            classes=args.classes,
                                                            seed=1)

    # Setup training.
    model_checkpoint = ModelCheckpoint(args.model_outfile, monitor='val_loss', mode='auto', verbose=1, save_best_only=True)
    callbacks = [model_checkpoint]
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


def test_cnn(args):
    pass


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=args.log.upper())

    if args.classes == SCENE_TAGS_CLASSIFICATION_SPEC_STRING:
        if not args.deployment:
            raise ValueError("The --deployment needs to be provided since the classes provided is %s"
                             % SCENE_TAGS_CLASSIFICATION_SPEC_STRING)

        args.classes = ["%s_%s" % (args.deployment, x) for x in SCENE_TAGS_CLASSIFICATION_SPEC_STRING]

    if args.func == "train_cnn":
        train_cnn(args)
    elif args.func == "test_cnn":
        test_cnn(args)
    else:
        raise ValueError("The given function is not supported: %s" % args.func)
