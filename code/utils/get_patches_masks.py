#!/usr/bin/env python3

"""
Extracts patches and masks from given annotation_work_dir (directory containing frames and their annotations files.)

Note:
    1. It writes the output to the output dir under separate directories 'patches' and 'masks' with mask name
having suffix '_mask'. If the patches and mask directories exist the output dit, then files will be added to
the existing directories.
    2. The size of the patch is will be square and will be fixed corresponding to the given label in this script.
    3. The corresponding annotation file path is assumed to have the same file name as the frame with .json extension.
    4. The annotation polygons must be disjoint.

python code/utils/get_patches_masks.py
    --anno-work-dir <anno_work_dir_path>
    --label <label_of_interest>
    --output <output_dir>
    --backup-raw <The path of the backup annotations directory>
    --backup-annotations <The path of the backup annotations directory>

"""

from skimage.measure import label, regionprops

import argparse
import cv2
import glob
import json
import logging
import numpy as np
import os
import shutil

LABEL_PATCH_SIZE_MAP = {
    "amphipod": 256,
    "star": 256,
    "crab": 256,
    "other": 256,
}

def get_args():
    parser = argparse.ArgumentParser(description=
    """
    Extracts patches and masks from given annotation_work_dir (directory containing frames and their annotations files.)

    Note: It writes the output to the output dir under separate directories 'patches' and 'masks' with mask name
    having suffix '_mask'. If the patches and mask directories exist the output dit, then files will be added to
    the existing directories.
    
    Note: The size of the patch is will be square and will be fixed corresponding to the given label in this script.
    """)
    parser.add_argument('--anno-work-dir',
                        required=True,
                        help="The path to the annotation_work_dir containing the frames and"
                             "corresponding labelme annotation files.")
    parser.add_argument('--label',
                        required=True,
                        help="The label of interest for which the masks need to be extracted.")
    parser.add_argument('--output',
                        required=True,
                        help="The path of the directory where the patches and masks need to be output.")
    parser.add_argument('--backup-raw',
                        help="The path of the backup raw frames directory. If not entered, backup is skipped.")
    parser.add_argument('--backup-annotations',
                        help="The path of the backup annotations directory. If not entered, backup is skipped.")
    parser.add_argument('--image-ext',
                        dest='img_ext',
                        default='png',
                        help="The image file extension. Default: png.")
    parser.add_argument("--log",
                        default="INFO",
                        help="Specify the log level. Default: INFO.")

    return parser.parse_args()


# TODO: Put this logic at one place, since it is also used in the analysis_pipeline.
def get_patch(img, center_coord, patch_size):
    pad_size = int(patch_size / 2)
    padded_img = cv2.copyMakeBorder(img,
                                    top=pad_size,
                                    bottom=pad_size,
                                    left=pad_size,
                                    right=pad_size,
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=0)
    padded_center_coord = (center_coord[0] + pad_size, center_coord[1] + pad_size)

    patch_x = int(padded_center_coord[0] - patch_size / 2)
    patch_y = int(padded_center_coord[1] - patch_size / 2)
    patch = padded_img[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]
    return patch


def get_patches_masks(anno_work_dir, obj_label, output_dir, backup_dir_raw, backup_dir_annotations, img_ext):
    if not os.path.exists(anno_work_dir):
        raise ValueError("The provided anno-work-dir does not exist: %s" % anno_work_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    patches_dir = os.path.join(output_dir, "patches")
    masks_dir = os.path.join(output_dir, "masks")

    for cur_dir in [patches_dir, masks_dir]:
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)

    for cur_dir in [backup_dir_raw, backup_dir_annotations]:
        if cur_dir and not os.path.exists(cur_dir):
            os.makedirs(cur_dir)

    for frame_path in glob.iglob(os.path.join(anno_work_dir, "*.%s" % img_ext)):
        anno_path = frame_path.rstrip(".%s" % img_ext) + ".json"
        if not os.path.exists(anno_path):
            continue

        # Copy to backup
        if backup_dir_raw:
            shutil.copy(frame_path, os.path.join(backup_dir_raw, os.path.basename(frame_path)))

        if backup_dir_annotations:
            shutil.copy(anno_path, os.path.join(backup_dir_annotations, os.path.basename(anno_path)))

        with open(anno_path) as fp:
            anno_dict = json.load(fp)

        annos = anno_dict["shapes"]
        if len(annos) < 1:
            return

        frame_img = cv2.imread(frame_path)
        frame_img = frame_img.astype(np.uint8)
        frame_mask = np.zeros(shape=(frame_img.shape[0], frame_img.shape[1]), dtype=np.uint8)

        cur_label_anno_count = 0
        for anno in annos:
            if anno["label"] != obj_label:
                continue

            contours = np.asarray(anno["points"])
            cv2.fillPoly(frame_mask, pts=[contours], color=(255, 255, 255))
            cur_label_anno_count += 1

        # Get connected components and extract patch and mask.
        frame_mask_bin = frame_mask > 0
        labeled_mask = label(frame_mask_bin)

        regions = regionprops(labeled_mask)

        if len(regions) != cur_label_anno_count:
            raise ValueError("The number of regions is not equal to number of annotations: %s "
                             "(number of regions found: %s, number of annotations: %s)"
                             % (anno_path, len(regions), cur_label_anno_count))

        for region in regions:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            center_coord = (int((minc + maxc) / 2), int((minr + maxr) / 2))

            patch_name = "_".join((os.path.basename(frame_path).rstrip(".%s" % img_ext),
                                   obj_label,
                                   str(center_coord[0]),
                                   str(center_coord[1]))) \
                         + ".%s" % img_ext
            mask_name = patch_name.rstrip(".%s" % img_ext) + "_mask" + ".%s" % img_ext

            patch = get_patch(frame_img, center_coord, LABEL_PATCH_SIZE_MAP[obj_label])
            mask  = get_patch(frame_mask_bin.astype(np.uint8), center_coord, LABEL_PATCH_SIZE_MAP[obj_label])

            cv2.imwrite(os.path.join(patches_dir, patch_name), patch)
            cv2.imwrite(os.path.join(masks_dir, mask_name), mask * 255)

    logging.info("All the annotation files in the anno-work-dir %s have been processed." % anno_work_dir)


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=args.log.upper())

    get_patches_masks(args.anno_work_dir,
                      args.label,
                      args.output,
                      args.backup_raw,
                      args.backup_annotations,
                      args.img_ext)
