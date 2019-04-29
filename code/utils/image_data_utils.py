#!/usr/bin/env python3

"""
This file contains data utils which can be used for image data manipulation.
TODO: Specific parts of this script can be run manually for specific use cases.
"""

from scipy.spatial import distance
from skimage import io
from skimage import measure
from skimage.filters import threshold_otsu
from skimage.transform import resize

import cv2
import numpy as np
import os
import random


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


def center_crop_pad_mask(patch_dir, mask_dir, center_crop_shape=(128, 128)):
    """
    Note: Call this on a copied data_dir before running this as this would overwrite the patches.
    This could be used to create positive patches data for classification model from the segmentation (U-Net) labeled
    patches and masks data.

    :param patch_dir: The directory containing patches containing the objects.
    :param mask_dir: The directory containing masks corresponding to each patch in the patch_dir with suffix '_mask'.
    :param center_crop_shape: The thresholds to decide if the patch needs to be cropped and padded.

    """
    def center_crop_pad(img, cropx, cropy):
        y, x = img.shape[:2]
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        img_cropped = img[starty:starty + cropy, startx:startx + cropx]
        pad_size = x - cropx
        img_padded  = cv2.copyMakeBorder(img_cropped,
                                         top=pad_size,
                                         bottom=pad_size,
                                         left=pad_size,
                                         right=pad_size,
                                         borderType=cv2.BORDER_CONSTANT,
                                         value=0)
        return img_padded


    def _should_crop(raw_mask):
        binary_mask = raw_mask > threshold_otsu(raw_mask / 255)
        label_image = measure.label(binary_mask)
        connected_regions = measure.regionprops(label_image)
        num_connected_regions = len(connected_regions)
        if num_connected_regions == 0:
            raise ValueError("No connected region found.")
        elif num_connected_regions == 1:
            chosen_region = connected_regions[0]
        else:
            y, x = raw_mask.shape[:2]
            center_coord = (x / 2, y / 2)
            chosen_region_index = -1
            min_dist = np.inf
            for region_index, region in enumerate(connected_regions):
                cur_centroid = region.centroid
                cur_dist = distance.euclidean(center_coord, cur_centroid)
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    chosen_region_index = region_index

            chosen_region = connected_regions[chosen_region_index]

        minr, minc, maxr, maxc = chosen_region.bbox
        bb_length = maxc - minc
        bb_width = maxr - minr
        if bb_length < center_crop_shape[1] and bb_width < center_crop_shape[0]:
            return True
        else:
            return False


    for img_file in os.listdir(patch_dir):
        img = io.imread(os.path.join(patch_dir, img_file))
        mask_file = "%s_mask%s" % os.path.splitext(img_file)
        mask = io.imread(os.path.join(mask_dir, mask_file))
        if _should_crop(mask):
            cropped_img = center_crop_pad(img, center_crop_shape[1], center_crop_shape[0])
            # It overwrites the patch.
            io.imsave(os.path.join(patch_dir, img_file), cropped_img)


def random_center_crop_pad(patch_dir, center_crop_shape=(128, 128), prob=0.5):
    # Note: Make a copy of data_dir before running this.
    def center_crop_pad(img, cropx, cropy):
        y, x = img.shape[:2]
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        img_cropped = img[starty:starty + cropy, startx:startx + cropx]
        pad_size = x - cropx
        img_padded  = cv2.copyMakeBorder(img_cropped,
                                         top=pad_size,
                                         bottom=pad_size,
                                         left=pad_size,
                                         right=pad_size,
                                         borderType=cv2.BORDER_CONSTANT,
                                         value=0)
        return img_padded


    for img_file in os.listdir(patch_dir):
        img = io.imread(os.path.join(patch_dir, img_file))
        if random.random() < prob:
            cropped_img = center_crop_pad(img, center_crop_shape[1], center_crop_shape[0])
            new_img_name = "%s_cropped%s" % os.path.splitext(img_file)
            io.imsave(os.path.join(patch_dir, new_img_name), cropped_img)
