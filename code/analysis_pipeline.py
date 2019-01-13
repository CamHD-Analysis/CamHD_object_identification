#!/usr/bin/env python3

import pycamhd.lazycache as camhd
import pycamhd.motionmetadata as mmd

import models

from collections import defaultdict
from keras.models import load_model
from skimage import io
from skimage.measure import label, regionprops
from skimage.morphology import opening, closing
from skimage.morphology import disk

import argparse
import cv2
import json
import logging
import numpy as np
import os

FRAME_RESOLUTION = (1080, 1920)

# TODO: Try the thresholds for each scene and enter the thresholds here.
SCENE_TAG_TO_SHARPNESS_SCORE_THRESHOLD_DICT = {

}

LABEL_TO_COLOR_DICT = {
    "amphipod": (255, 0, 0),    # RED
    "star": (255, 255, 0)       # Yellow
}

# TODO: Generalize the patch_size.
PATCH_SIZE = 256

# TODO: This puts a restriction that this script must be run from repository root dir.
TRAINED_MODELS_DIR = "./trained_models"

def get_args():
    parser = argparse.ArgumentParser(description=
    """
    Run Analysis on a given video regions_file and output the analysis report.
    Intermediate analysis files will be written to provided analysis-work-dir in
    debug mode (currently set to default).

    Sample analysis config:
    {
        "analysis_version": "prototype-1",
        "labels": ["amphipod", "star"],
        "scene_tags": {
            "amphipod": ["d5A_p1_z1", "d5A_p5_z1", "d5A_p6_z1", "d5A_p0_z1", "d5A_p7_z1", "d5A_p8_z1"],
            "star": ["d5A_p7_z1"]
        },
        "segmentation_trained_models": {
            "amphipod": "/home/bhuvan/Projects/CamHD_object_identification/trained_models/amphipod_unet_1.hdf5",
            "star": "/home/bhuvan/Projects/CamHD_object_identification/trained_models/star_unet_1.hdf5"
        }
    }
    """)
    parser.add_argument('--config',
                        help="The analysis config (JSON). Please refer to the sample analyser config.")
    parser.add_argument('--regions-file',
                        help="The path to the video regions_file on which the analysis needs to be run.")
    parser.add_argument('--analysis-work-dir',
                        help="The path to the directory where intermediate analysis files can be written.")
    parser.add_argument('--outfile',
                        help="The path at which the analysis report needs to be saved.")

    parser.add_argument('--extract-patches-only',
                        action="store_true",
                        help="Only extracts the patches sent for classifier in the provided patches-output-dir.")
    parser.add_argument('--input-data-dir',
                        help="The path to the directory containing the input data frames organized by scene_tags.")
    parser.add_argument('--patches-output-dir',
                        help="The path at which the patches sent for classifier need to be saved."
                             "It is required if 'extract-patches-only' flag is set.")

    parser.add_argument('--image-ext',
                        dest='img_ext',
                        default='png',
                        help="The image file extension. Default: png.")
    parser.add_argument('--lazycache-url',
                        dest='lazycache',
                        default=os.environ.get("LAZYCACHE_URL", None),
                        help='URL to Lazycache repo server. Default: Environment variable LAZYCACHE_URL.')
    parser.add_argument("--log",
                        default="DEBUG",
                        help="Specify the log level. Default: DEBUG.")

    return parser.parse_args()


def _is_sharp(img, scene_tag):
    def _get_sharpness_score(img):
        pass


    if _get_sharpness_score(img) >= SCENE_TAG_TO_SHARPNESS_SCORE_THRESHOLD_DICT[scene_tag]:
        return True

    return False


class RegionSize(object):
    def __init__(self, area, bb_length, bb_width):
        self.area = area
        self.bb_length = bb_length
        self.bb_width = bb_width


# TODO: Keep this logic at one place.
def get_patch(img, center_coord, patch_size, padding_check=True):
    if padding_check:
        pad_size = int(patch_size / 2)
        img = cv2.copyMakeBorder(img,
                                 top=pad_size,
                                 bottom=pad_size,
                                 left=pad_size,
                                 right=pad_size,
                                 borderType=cv2.BORDER_CONSTANT,
                                 value=0)
        center_coord = (center_coord[0] + pad_size, center_coord[1] + pad_size)

    patch_x = int(center_coord[0] - patch_size / 2)
    patch_y = int(center_coord[1] - patch_size / 2)
    patch = img[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]
    return patch


def _invoke_unet(patch, model, mask_index):
    patch = np.reshape(patch, (1,) + patch.shape)
    pred_mask = model.predict(patch)
    pred_mask = pred_mask[mask_index] * 255
    pred_mask = pred_mask.astype(np.uint8)
    pred_mask = np.reshape(pred_mask, pred_mask.shape[:-1])
    return pred_mask


# Utility functions of analyse_frame function.
def _get_raw_mask(frame, segmentation_model_config):
    def _crop_center(img, cropx, cropy):
        y, x = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty + cropy, startx:startx + cropx]


    if frame.shape[:-1] != FRAME_RESOLUTION:
        raise ValueError("The frame resolution needs to be %s, but the current frame resolution is %s"
                         % (str(FRAME_RESOLUTION), str(frame.shape[:-1])))

    # Assuming a square patch.
    patch_size = segmentation_model_config["input_shape"][:-1][0]
    stride_size = patch_size // 2

    if frame.shape[0] % stride_size == 0:
        height_adjustment = 0
    else:
        height_adjustment = stride_size - (frame.shape[0] % stride_size)

    if frame.shape[1] % stride_size == 0:
        width_adjustment = 0
    else:
        width_adjustment = stride_size - (frame.shape[1] % stride_size)

    # Check for even value of adjustment paddings.
    if not any([x % 2 != 2 for x in (height_adjustment, width_adjustment)]):
        raise ValueError("The frame dimensions couldn't be adjusted for stride_size: %s" % stride_size)

    adjusted_frame = cv2.copyMakeBorder(frame,
                                        top=height_adjustment // 2,
                                        bottom=height_adjustment // 2,
                                        left=width_adjustment // 2,
                                        right=width_adjustment // 2,
                                        borderType=cv2.BORDER_CONSTANT,
                                        value=0)

    # Padding for taking the strided patches.
    padded_frame = cv2.copyMakeBorder(adjusted_frame,
                                      top=stride_size,
                                      bottom=stride_size,
                                      left=stride_size,
                                      right=stride_size,
                                      borderType=cv2.BORDER_CONSTANT,
                                      value=0)

    row_ordered_coords = []
    for ri in range(stride_size, padded_frame.shape[0] - stride_size + 1, stride_size):
        row_i = []
        for ci in range(stride_size, padded_frame.shape[1] - stride_size + 1, stride_size):
            row_i.append((ci, ri))

        row_ordered_coords.append(row_i)

    # TODO: Standardize the model definition.
    # Invoke and stitch mask
    if "load_model_arch" in segmentation_model_config:
        model = getattr(models, segmentation_model_config["load_model_arch"])()
        model.load_weights(os.path.join(TRAINED_MODELS_DIR, segmentation_model_config["model_path"]))
    else:
        model = load_model(os.path.join(TRAINED_MODELS_DIR, segmentation_model_config["model_path"]))

    mask_index = segmentation_model_config["mask_index"]

    row_stitched_masks = []
    for ri_coords in row_ordered_coords:
        ri_masks = []
        for coord in ri_coords:
            patch = get_patch(padded_frame, coord, patch_size, padding_check=False)
            if segmentation_model_config.get("rescale", True):
                patch = patch * (1.0 / 255)

            cur_mask = _invoke_unet(patch, model, mask_index)
            cur_mask_center_crop = _crop_center(cur_mask, stride_size, stride_size)
            ri_masks.append(cur_mask_center_crop)

        ri_stitched_mask = np.hstack(ri_masks)
        row_stitched_masks.append(ri_stitched_mask)

    padded_stitched_mask = np.vstack(row_stitched_masks)

    adjusted_stitched_mask = _crop_center(padded_stitched_mask,
                                          adjusted_frame.shape[1],
                                          adjusted_frame.shape[0])

    # Re-adjust the mask to match the frame resolution.
    raw_mask = _crop_center(adjusted_stitched_mask,
                            frame.shape[1],
                            frame.shape[0])
    io.imsave("./raw_mask.png", raw_mask * 255)
    return raw_mask


def _get_patch_coord_to_label_size_dict(label_to_postprocessed_mask_dict):
    patch_coord_to_label_size_dict = {}
    for label, mask in label_to_postprocessed_mask_dict.items():
        label_image = label(mask)
        for region in regionprops(label_image):
            minr, minc, maxr, maxc = region.bbox
            bb_length = maxc - minc
            bb_width = maxr - minr
            region_size = RegionSize(region.area, bb_length, bb_width)
            patch_coord = tuple([int(x) for x in region.centroid])
            patch_coord_to_label_size_dict[patch_coord] = (label, region_size)

    return patch_coord_to_label_size_dict


def _get_marked_image(frame, patch_coord_to_label_size_dict, label_to_color_dict):
    # return marked_image
    pass


# Utility functions of analyse_frame function: Postprocess functions:
def _postprocess_mask_1(raw_mask):
    prob_thresh_1 = 0.5
    binary_mask = raw_mask > prob_thresh_1
    # After this all are binary (bool ndarray) images.
    opened_mask = opening(binary_mask, disk(6))
    closed_mask = closing(opened_mask, disk(6))
    label_image = label(closed_mask)

    for region in regionprops(label_image):
        minr, minc, maxr, maxc = region.bbox
        bb_length = maxc - minc
        bb_width = maxr - minr
        # Apply min and max bounding box restrictions.
        # TODO: There could be better metric to put restrictions on.
        # We don't know the min and max size of amphipods. So putting restrictions based on
        # the analysis pipeline structure, where square patch extraction is the next step.
        # Currently, restricting the working objects to stay within a bounding box of 64 to 256.
        if (bb_length > 256 or bb_length < 64 or
            bb_width > 256 or bb_width < 64):
            # Clear the region by setting the coords to False.
            for coord in region.coords:
                closed_mask[coord] = False

    io.imsave("./postprocessed_mask.png", closed_mask * 255)
    return closed_mask


def _postprocess_mask_2(raw_mask):
    #return postprocessed_mask
    pass

LABEL_TO_POSTPROCESS_FUNC_DICT = {
    "amphipod": _postprocess_mask_1,
    "star": _postprocess_mask_2
}


def _extract_labeled_patches(frame, patch_coord_to_label_size_dict, patch_size=256):
    label_to_coord_patches_dict = defaultdict(list)
    for patch_coord, label_size in patch_coord_to_label_size_dict.items():
        label, region_size = label_size
        if (region_size.bb_length < patch_size // 2 or
            region_size.bb_width < patch_size // 2):
            unpadded_patch = get_patch(frame, patch_coord, patch_size // 2, padding_check=True)
            pad_size = patch_size // 2
            patch = cv2.copyMakeBorder(unpadded_patch,
                                       top=pad_size,
                                       bottom=pad_size,
                                       left=pad_size,
                                       right=pad_size,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=0)
        else:
            patch = get_patch(frame, patch_coord, patch_size, padding_check=True)

        label_to_coord_patches_dict[label].append((patch_coord, patch))

    return label_to_coord_patches_dict


def _validate_by_classification(frame,
                                patch_coord_to_label_size_dict,
                                label_to_classification_model_dict):
    pass
    # return validated_patch_coord_to_label_size_dict


def analyse_frame(scene_tag,
                  frame_path,
                  label_to_segmentation_model_config_dict,
                  label_to_classification_model_config_dict,
                  img_ext="png",
                  patches_output_dir=None):
    work_dir, frame_name = os.path.split(frame_path)
    frame_base_name = frame_name.split(".%s" % img_ext)[0]

    frame = io.imread(frame_path)

    label_to_raw_mask_dict = {}
    for label, segmentation_model_config in label_to_segmentation_model_config_dict.items():
        raw_mask = _get_raw_mask(frame, segmentation_model_config)
        label_to_raw_mask_dict[label] = raw_mask
        io.imsave(os.path.join(work_dir, "%s_mask_%s.%s"
                               % (frame_base_name, label, img_ext)), raw_mask * 255)

    label_to_postprocessed_mask_dict = {}
    for label, raw_mask in label_to_raw_mask_dict.items():
        postprocessed_mask = LABEL_TO_POSTPROCESS_FUNC_DICT[label](raw_mask)
        label_to_postprocessed_mask_dict[label] = postprocessed_mask
        io.imsave(os.path.join(work_dir, "%s_postprocessed_mask_%s.%s"
                               % (frame_base_name, label, img_ext)), postprocessed_mask * 255)

    # This contains list of centroids mapped to labels
    patch_coord_to_label_size_dict = _get_patch_coord_to_label_size_dict(label_to_postprocessed_mask_dict)

    # Extract Patches and Classify the patches to get validated_patch_coord_to_label_size_dict.
    # TODO: Update after training classifier.
    # If 'patches_output_dir' has been provided, then just return as only the patches were required.
    if patches_output_dir:
        label_to_coord_patches_dict = _extract_labeled_patches(frame,
                                                               patch_coord_to_label_size_dict,
                                                               patch_size=256)
        for label, coord_patches in label_to_coord_patches_dict.items():
            for coord_patch in coord_patches:
                coord, patch = coord_patch
                patch_name = "%s_%s_%s_%s.%s" % (frame_base_name, label, coord[0], coord[1])
                io.imsave(patch_name, patch)

        return

    validated_patch_coord_to_label_size_dict = _validate_by_classification(frame,
                                                                           patch_coord_to_label_size_dict,
                                                                           label_to_classification_model_config_dict)

    marked_image = _get_marked_image(frame, validated_patch_coord_to_label_size_dict, LABEL_TO_COLOR_DICT)
    io.imsave(os.path.join(work_dir, "%s_marked.%s" % (frame_base_name, img_ext)), marked_image)

    # Format the result.
    label_to_location_sizes_dict = defaultdict(dict)
    for patch_coord, label_size in validated_patch_coord_to_label_size_dict.items():
        label, size = label_size
        label_to_location_sizes_dict[label][patch_coord] = size

    label_to_counts_dict = {}
    for label, location_sizes in label_to_location_sizes_dict.items():
        label_to_counts_dict[label] = len(location_sizes)

    result_dict = {
        "frame": frame_base_name,
        "scene_tag": scene_tag,
        "frame_res": json.dumps(FRAME_RESOLUTION),
        "counts": label_to_counts_dict,
        "location_sizes": label_to_location_sizes_dict
    }
    with open(os.path.join(work_dir, "%s_report.json" % scene_tag), "w") as fp:
        json.dump(result_dict, fp)

    return result_dict


def analyse_regions_file(args):
    if not args.lazycache:
        raise ValueError("The lazycache-url could not be found.")

    pass


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=args.log.upper())

    # TODO: This assumes that the script is being run from the repository root directory.
    with open("./trained_models/amphipod_unet_1.json") as fp:
        amphipod_segmentation_model_config_dict = json.load(fp)

    label_to_segmentation_model_config_dict = {
        "amphipod": amphipod_segmentation_model_config_dict
    }

    if args.extract_patches_only:
        if not args.patches_output_dir:
            raise ValueError("The patches-output-dir must be provided when extract-patches-only flag is set.")

        for scene_tag in os.listdir(args.input_data_dir):
            for frame_file in os.listdir(os.path.join(args.input_data_dir, scene_tag)):
                frame_path = os.path.join(args.input_data_dir, scene_tag, frame_file)
                analyse_frame(scene_tag,
                              frame_path,
                              label_to_segmentation_model_config_dict,
                              label_to_classification_model_config_dict=None,
                              img_ext="png",
                              patches_output_dir=args.patches_output_dir)
    else:
        analyse_regions_file(args)
