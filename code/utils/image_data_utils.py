#!/usr/bin/env python3

"""
This file contains data utils which can be used for image data manipulation.
TODO: Specific parts of this script can be run manually for specific use cases.
"""

from skimage import io
from skimage.transform import resize

import numpy as np
import os


def resize_images(src_data_dir, dest_data_dir, shape=(256, 256), classification_data=True):
    def _resize_images_cur_dir(src_dir, dest_dir):
        for img_name in os.listdir(src_dir):
            img = io.imread(os.path.join(src_dir, img_name))
            resized_image = resize(img, shape) * 255
            resized_image = resized_image.astype(np.uint8)
            io.imsave(os.path.join(dest_dir, img_name), resized_image)


    if os.path.exists(dest_data_dir):
        raise ValueError("The dest_data_dir already exists: %s" % dest_data_dir)

    os.makedirs(dest_data_dir)

    if not classification_data:
        _resize_images_cur_dir(src_data_dir, dest_data_dir)
        return

    # The source data dir contains image classification data.
    labels = os.listdir(src_data_dir)
    for label in labels:
        src_label_dir_path = os.path.join(src_data_dir, label)
        if not os.path.isdir(src_label_dir_path):
            raise ValueError("The data doesn't seem to have been correctly organized.")

        dest_label_dir_path = os.path.join(dest_data_dir, label)
        os.makedirs(dest_label_dir_path)

        _resize_images_cur_dir(src_label_dir_path, dest_label_dir_path)
