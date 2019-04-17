#!/usr/bin/env python3

"""
Run Analysis on a given video regions_file and output the analysis report.

sample_analysis_config_1:
{
    "analysis_version": "prototype-1",
    "labels": ["amphipod"],
    "scene_tags": {
        "amphipod": ["d5A_p1_z1", "d5A_p5_z1", "d5A_p6_z1", "d5A_p0_z1", "d5A_p7_z1", "d5A_p8_z1"]
    },
    "label_to_segmentation_model_config": {
        "amphipod": "/home/bhuvan/Projects/CamHD_object_identification/trained_models/amphipod_unet_1.json"
    },
    "label_to_classification_model_config": {
        "amphipod": "/home/bhuvan/Projects/CamHD_object_identification/trained_models/amphipod_cnn-v0.1.json"
    }
}

sample_analysis_config_2:
{
    "analysis_version": "prototype-2",
    "labels": ["amphipod", "star"],
    "scene_tags": {
        "amphipod": ["d5A_p1_z1", "d5A_p5_z1", "d5A_p6_z1", "d5A_p0_z1", "d5A_p7_z1", "d5A_p8_z1"],
        "star": ["d5A_p7_z1"]
    },
    "label_to_segmentation_model_config": {
        "amphipod": "/home/bhuvan/Projects/CamHD_object_identification/trained_models/amphipod_unet_1.json",
        "star": "/home/bhuvan/Projects/CamHD_object_identification/trained_models/star_unet_1.json"
    },
    "label_to_classification_model_config": {
        "amphipod": "/home/bhuvan/Projects/CamHD_object_identification/trained_models/amphipod_cnn-v0.1.json",
        "star": "/home/bhuvan/Projects/CamHD_object_identification/trained_models/star_cnn-v0.1.json"
    }
}

Note: All the trained models must be present in the 'trained_models' directory.
Also, the model_configs must have the model_path relative to the 'trained_models' directory.

"""

import pycamhd.lazycache as camhd
import pycamhd.motionmetadata as mmd

import models

from collections import defaultdict, OrderedDict
from keras.models import load_model
# from scipy.ndimage.morphology import binary_fill_holes
from skimage import io
from skimage import measure
from skimage.morphology import dilation
from skimage.morphology import disk

import argparse
import cv2
import json
import logging
import numpy as np
import os
import pickle

# Use below code to force execute on CPU instead of GPU.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

FRAME_RESOLUTION = (1080, 1920)

# TODO: Avoid defaulting deployment once this information is available from the regions files.
DEFAULT_DEPLOYMENT = "d5A"

# TODO: Try the thresholds for each scene and enter the thresholds here. Currently, set to an arbitrary value.
SCENE_TAG_TO_SHARPNESS_SCORE_THRESHOLD_DICT = {
    "d5A_p1_z1": 0.5,
    "d5A_p5_z1": 0.5,
    "d5A_p6_z1": 0.5,
    "d5A_p0_z1": 0.5,
    "d5A_p7_z1": 0.5,
    "d5A_p8_z1": 0.5
}

LABEL_TO_COLOR_DICT = {
    "amphipod": (255, 0, 0),    # RED
    "star": (255, 255, 0)       # Yellow
}

LABEL_TO_CLASS_ID_DICT = {
    "amphipod": 1               # RED
}

# TODO: Generalize the patch_size.
PATCH_SIZE = 256

# TODO: This puts a restriction that this script must be run from repository root dir.
TRAINED_MODELS_DIR = "./trained_models"

# Stores the model_path mapped to the loaded model, eliminating reloading the model.
MODEL_CACHE = {}

def get_args():
    parser = argparse.ArgumentParser(description=
    """
    Run Analysis on a given video regions_file and output the analysis report.

    """)
    parser.add_argument('--config',
                        required=True,
                        help="The analysis config (JSON). Please refer to the sample analyser config.")
    parser.add_argument('--regions-file',
                        help="The path to the video regions_file on which the analysis needs to be run. "
                             "If it is not provided, a set of test frames can be provided using '--input-data-dir'.")
    parser.add_argument('--input-data-dir',
                        required=True,
                        help="The path to the directory containing the input data frames organized by scene_tags. "
                             "If '--regions-file' is provided, this directory would be created and frames from the "
                             "regions files from static regions will be extracted into this directory.")
    parser.add_argument('--output-dir',
                        help="The directory path to which the intermediate results for each frame need to be written. "
                             "If not provided, the intermediate results for each frame will be written to the "
                             "directory in which the respective frame is present in the input-data-dir.")
    parser.add_argument('--outfile',
                        required=True,
                        help="The path at which the analysis report needs to be saved.")
    parser.add_argument('--coco-result-path',
                        help="The path to which the results in the coco-format need to be output as a pickle file.")
    parser.add_argument('--patches-output-dir',
                        help="The path at which the patches sent for classifier need to be saved. "
                             "If not provided, the patches sent for classifier will not be saved.")
    parser.add_argument('--masks-output-dir',
                        help="The path to the directory where the patch-level masks of the detected objects "
                             "need to be saved. If not provided, the patch-level masks will not be saved.")
    parser.add_argument('--extract-patches-only',
                        action="store_true",
                        help="Only extracts the patches sent for classifier in the provided patches-output-dir.")
    parser.add_argument('--no-write',
                        action="store_true",
                        help="If this flag is set, the intermediate frame-level mask, marked image and report"
                             "will not be written.")
    parser.add_argument('--restrict-gpu',
                        help="The value to be set for 'CUDA_VISIBLE_DEVICES' if GPU cores need to be restricted. "
                             "If not provided, the 'CUDA_VISIBLE_DEVICES' will not be set by this script.")
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

    args = parser.parse_args()

    if args.patches_output_dir and not os.path.exists(args.patches_output_dir):
        os.makedirs(args.patches_output_dir)

    if args.masks_output_dir and not os.path.exists(args.masks_output_dir):
        os.makedirs(args.masks_output_dir)

    return args


def _is_sharp(img, scene_tag):
    def _get_sharpness_score(img):
        # TODO: This needs to be implemented when sharpness based to frame selection is incorporated.
        return 1

    # TODO: Chose the appropriate default sharpness value after analysis.
    if _get_sharpness_score(img) >= SCENE_TAG_TO_SHARPNESS_SCORE_THRESHOLD_DICT.get(scene_tag, 0.5):
        return True

    return False


def get_json_serializable(o):
    if isinstance(o, np.int64): return int(o)
    raise TypeError


class RegionInfo(object):
    def __init__(self, region, orig_img_shape, pred_score=None):
        self.region = region
        self.pred_score = pred_score
        self.orig_img_shape = orig_img_shape
        self._extract_region_info()

    def _extract_region_info(self):
        self.area = self.region.area

        minr, minc, maxr, maxc = self.region.bbox
        self.bb_length = maxc - minc
        self.bb_width = maxr - minr

        region_mask = np.full(self.orig_img_shape, False, dtype=bool)
        region_mask[tuple(self.region.coords.T)] = True
        self.mask = region_mask


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


def _get_tf_model(model_config):
    model_path = os.path.join(TRAINED_MODELS_DIR, model_config["model_path"])
    if model_path not in MODEL_CACHE:
        if "load_model_arch" in model_config:
            model = getattr(models, model_config["load_model_arch"])()
            model.load_weights(model_path)
            MODEL_CACHE[model_path] = model
        else:
            model = load_model(model_path)
            MODEL_CACHE[model_path] = model

    return MODEL_CACHE[model_path]


def _invoke_unet(patch, model, mask_index):
    patch = np.reshape(patch, (1,) + patch.shape)
    pred_mask = model.predict(patch)
    pred_mask = pred_mask[mask_index] * 255
    pred_mask = pred_mask.astype(np.uint8)
    pred_mask = np.reshape(pred_mask, pred_mask.shape[:-1])
    return pred_mask


def _invoke_cnn(patch, model, class_labels):
    patch = np.reshape(patch, (1,) + patch.shape)
    pred_probas = model.predict(patch)
    pred_classes = np.argmax(pred_probas, axis=1)
    # We have only one input image.
    pred_proba = max(pred_probas[0])
    pred_class = pred_classes[0]
    pred_class_label = class_labels[pred_class]
    return pred_class_label, pred_proba


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
    model = _get_tf_model(segmentation_model_config)

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
    return raw_mask


# Utility functions of analyse_frame function: Postprocess functions:
def _postprocess_mask_1(raw_mask):
    raw_mask = raw_mask / 255
    prob_thresh_1 = 0.80
    binary_mask = raw_mask > prob_thresh_1
    # After this, all are binary (bool ndarray) images.
    #p_mask = binary_fill_holes(binary_mask)
    #p_mask = opening(binary_mask, disk(3))
    p_mask = dilation(binary_mask, disk(3))
    final_mask = p_mask

    label_image = measure.label(final_mask)
    for region in measure.regionprops(label_image):
        minr, minc, maxr, maxc = region.bbox
        bb_length = maxc - minc
        bb_width = maxr - minr
        # Apply min and max bounding box restrictions.
        # TODO: There could be better metric to put restrictions on.
        # We don't know the min and max size of amphipods. So putting restrictions based on
        # the analysis pipeline structure, where square patch extraction is the next step.
        # Currently, restricting the working objects to stay within a bounding box of 64 to 256.
        if (bb_length > 256 or bb_length < 48 or
            bb_width > 256 or bb_width < 48):
            # Clear the region by setting the coords to False.
            for coord in region.coords:
                final_mask[coord[0], coord[1]] = False

    return final_mask * 255


def _postprocess_mask_2(raw_mask):
    #return postprocessed_mask
    pass

LABEL_TO_POSTPROCESS_FUNC_DICT = {
    "amphipod": _postprocess_mask_1,
    "star": _postprocess_mask_2
}


def _get_patch_coord_to_label_region_info_dict(label_to_postprocessed_mask_dict):
    # This contains list of centroids mapped to (label, region_info) tuples.
    patch_coord_to_label_region_info_dict = {}
    for label, mask in label_to_postprocessed_mask_dict.items():
        orig_img_shape = mask.shape
        mask = mask >= 255
        label_image = measure.label(mask)
        for region in measure.regionprops(label_image):
            region_info = RegionInfo(region, orig_img_shape)
            patch_coord = tuple([int(x) for x in region.centroid])
            patch_coord_to_label_region_info_dict[patch_coord] = (label, region_info)

    return patch_coord_to_label_region_info_dict


def _extract_labeled_patches(frame,
                             label_to_postprocessed_mask_dict,
                             patch_size=256,
                             adjust_patch_size=False,
                             label_to_model_config_dict=None):
    patch_coord_to_label_region_info_dict = _get_patch_coord_to_label_region_info_dict(label_to_postprocessed_mask_dict)

    # The patch_size and adjust_patch_size provided will be overridden if label_to_model_config_dict is provided.
    label_to_coord_patch_masks_dict = defaultdict(list)
    for patch_coord, label_region_info in patch_coord_to_label_region_info_dict.items():
        label, region_info = label_region_info
        if label_to_model_config_dict:
            patch_size = label_to_model_config_dict[label]["input_shape"][:-1][0]
            adjust_patch_size = label_to_model_config_dict[label]["adjust_patch_size"]

        if (adjust_patch_size and
            region_info.bb_length < patch_size // 2 and
            region_info.bb_width < patch_size // 2):
            unpadded_patch = get_patch(frame, (patch_coord[1], patch_coord[0]), patch_size // 2, padding_check=True)
            pad_size = (patch_size // 2) // 2
            patch = cv2.copyMakeBorder(unpadded_patch,
                                       top=pad_size,
                                       bottom=pad_size,
                                       left=pad_size,
                                       right=pad_size,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=0)
        else:
            patch = get_patch(frame, (patch_coord[1], patch_coord[0]), patch_size, padding_check=True)

        mask = get_patch(label_to_postprocessed_mask_dict[label],
                         (patch_coord[1], patch_coord[0]),
                         patch_size,
                         padding_check=True)

        label_to_coord_patch_masks_dict[label].append((patch_coord, patch, mask))

    return label_to_coord_patch_masks_dict, patch_coord_to_label_region_info_dict


def _validate_by_classification(frame,
                                label_to_postprocessed_mask_dict,
                                label_to_model_config_dict,
                                frame_base_name,
                                patches_output_dir=None,
                                masks_output_dir=None,
                                img_ext="png"):
    validated_patch_coord_to_label_region_info_dict = {}
    label_to_coord_patch_masks_dict, patch_coord_to_label_region_info_dict = \
        _extract_labeled_patches(frame,
                                 label_to_postprocessed_mask_dict,
                                 label_to_model_config_dict=label_to_model_config_dict)

    for label, coord_patch_masks in label_to_coord_patch_masks_dict.items():
        cur_model_config = label_to_model_config_dict[label]
        cur_model = _get_tf_model(cur_model_config)
        cur_class_labels = cur_model_config["classes"]
        for coord_patch_mask in coord_patch_masks:
            coord, patch, mask = coord_patch_mask
            if cur_model_config.get("rescale", True):
                patch = patch * (1.0 / 255)

            patch_name = "%s_%s_%s_%s.%s" % (frame_base_name, label, coord[0], coord[1], img_ext)
            if patches_output_dir:
                patch_path = os.path.join(patches_output_dir, patch_name)
                try:
                    io.imsave(patch_path, patch)
                except:
                    # TODO: Try to find the reason and avoid this case.
                    logging.exception("Couldn't save: %s" % patch_path)

            pred_label, pred_proba = _invoke_cnn(patch, cur_model, cur_class_labels)
            logging.info("Classification for %s: %s (%.2f)" % (patch_name, pred_label, pred_proba))

            # Add the predicted probability to the region_info.
            label, region_info = patch_coord_to_label_region_info_dict[coord]
            region_info.pred_score = pred_proba

            valid_label = cur_model_config["valid_class"]
            if pred_label == valid_label and pred_proba >= cur_model_config["prob_thresholds"][valid_label]:
                validated_patch_coord_to_label_region_info_dict[coord] = (label, region_info)

                mask_name = "%s_mask.%s" % os.path.splitext(patch_name)
                if masks_output_dir:
                    mask_path = os.path.join(masks_output_dir, mask_name)
                    try:
                        io.imsave(mask_path, mask)
                    except:
                        # TODO: Try to find the reason and avoid this case.
                        logging.exception("Couldn't save: %s" % mask_path)

    logging.info("The patches have been validated using the provided classification models.")
    return validated_patch_coord_to_label_region_info_dict


def _get_marked_image(frame,
                      patch_coord_to_label_region_info_dict,
                      label_to_color_dict,
                      thickness=2):
    # The patch_size and adjust_patch_size provided will be overridden if label_to_model_config_dict is provided.
    for patch_coord, label_region_info in patch_coord_to_label_region_info_dict.items():
        label, region_info = label_region_info
        color = label_to_color_dict[label]
        minr, minc, maxr, maxc = region_info.region.bbox
        top_left_coord = (minc, minr)
        bottom_right_coord = (maxc, maxr)
        cv2.rectangle(frame, top_left_coord, bottom_right_coord, color, thickness)

        # Add the predicted probability as a text near the bounding box.
        if region_info.pred_score:
            prob_text = "%.2f" % region_info.pred_score
            # Adjust the text for bounding box thickness.
            cv2.putText(frame, prob_text, (minc, minr - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, lineType=cv2.LINE_AA)

    return frame


def analyse_frame(scene_tag,
                  frame_path,
                  label_to_segmentation_model_config_dict,
                  label_to_classification_model_config_dict,
                  img_ext="png",
                  patches_output_dir=None,
                  masks_output_dir=None,
                  extract_patches_only=False,
                  coco_format=False,
                  output_dir=None,
                  no_write=False):
    frame_dir, frame_name = os.path.split(frame_path)
    if not output_dir:
        output_dir = frame_dir
        if not no_write:
            logging.info("Writing the intermediate outputs for the frame %s to the frame dir: %s"
                         % (frame_name, frame_dir))

    frame_base_name = frame_name.split(".%s" % img_ext)[0]
    frame = io.imread(frame_path)

    label_to_raw_mask_dict = {}
    for label, segmentation_model_config in label_to_segmentation_model_config_dict.items():
        raw_mask = _get_raw_mask(frame, segmentation_model_config)
        label_to_raw_mask_dict[label] = raw_mask
        if not no_write:
            io.imsave(os.path.join(output_dir, "%s_mask_%s.%s"
                                   % (frame_base_name, label, img_ext)), raw_mask)

    label_to_postprocessed_mask_dict = {}
    for label, raw_mask in label_to_raw_mask_dict.items():
        postprocessed_mask = LABEL_TO_POSTPROCESS_FUNC_DICT[label](raw_mask)
        label_to_postprocessed_mask_dict[label] = postprocessed_mask
        if not no_write:
            io.imsave(os.path.join(output_dir, "%s_postprocessed_mask_%s.%s"
                                   % (frame_base_name, label, img_ext)), postprocessed_mask)

    # If 'extract_patches_only' and 'patches_output_dir' has been provided, then return after extracting patches.
    if extract_patches_only:
        if not os.path.exists(patches_output_dir):
            raise ValueError("The patches-output-dir must be provided when extract-patches-only argument is provided.")

        logging.info("Extracting the patches only as the 'extract-patches-only' argument has been provided. "
                     "The extracted patches will not have patch_size = 256, and adjust_patch_size = False.")
        label_to_coord_patches_dict, _ = _extract_labeled_patches(frame,
                                                                  label_to_postprocessed_mask_dict,
                                                                  patch_size=256,
                                                                  adjust_patch_size=False)
        for label, coord_patches in label_to_coord_patches_dict.items():
            for coord_patch in coord_patches:
                coord, patch = coord_patch
                patch_name = "%s_%s_%s_%s.%s" % (frame_base_name, label, coord[0], coord[1], img_ext)
                patch_path = os.path.join(patches_output_dir, patch_name)
                try:
                    io.imsave(patch_path, patch)
                except:
                    # TODO: Try to find the reason and avoid this case.
                    logging.exception("Couldn't save: %s" % patch_path)

        logging.info("The patches have been extracted from: %s" % frame_path)
        return

    # Extract Patches and Classify the patches to get validated_patch_coord_to_label_region_info_dict.
    validated_patch_coord_to_label_region_info_dict = \
        _validate_by_classification(frame,
                                    label_to_postprocessed_mask_dict,
                                    label_to_classification_model_config_dict,
                                    frame_base_name,
                                    patches_output_dir=patches_output_dir,
                                    masks_output_dir=masks_output_dir,
                                    img_ext=img_ext)

    marked_image = _get_marked_image(frame,
                                     validated_patch_coord_to_label_region_info_dict,
                                     LABEL_TO_COLOR_DICT)
    if not no_write:
        io.imsave(os.path.join(output_dir, "%s_marked.%s" % (frame_base_name, img_ext)), marked_image)

    # Format the result.
    # Create the detected_object_info dict.
    label_to_detected_object_info_dict = defaultdict(dict)
    for patch_coord, label_region_info in validated_patch_coord_to_label_region_info_dict.items():
        label, region_info = label_region_info
        # XXX: Converting patch coord to string as JSON cannot handle tuple in dictionary keys.
        # XXX: To convert back to tuple, use: `t = tuple([int(x) for x in t_s.lstrip("(").rstrip(")").split(",")])`
        label_to_detected_object_info_dict[label][str(patch_coord)] = {
            "area": region_info.area,
            "bbox": json.dumps(region_info.region.bbox), # TODO: Check if the order is right?
            "pred_score": float(region_info.pred_score)
        }
        # TODO: Add more region info to the report if needed.

    label_to_counts_dict = {}
    for label, detected_object_info_dict in label_to_detected_object_info_dict.items():
        label_to_counts_dict[label] = len(detected_object_info_dict)

    result_dict = {
        "frame": frame_base_name,
        "scene_tag": scene_tag,
        "frame_res": json.dumps(FRAME_RESOLUTION),
        "counts": label_to_counts_dict,
        "detected_object_info": label_to_detected_object_info_dict
    }

    if not no_write:
        with open(os.path.join(output_dir, "%s_report.json" % frame_base_name), "w") as fp:
            print(result_dict)
            json.dump(result_dict, fp, indent=4, sort_keys=True, default=get_json_serializable)

    logging.info("The frame has been analysed: %s" % frame_path)

    if not coco_format:
        return result_dict

    # Format result as required for Coco evaluation.
    # Refer to https://github.com/matterport/Mask_RCNN/blob/master/samples/coco/coco.py.
    logging.info("Formatting the output suitable for Coco evaluation: %s" % frame_path)

    class_ids = []
    rois = []
    scores = []
    masks = []

    for patch_coord, label_region_info in validated_patch_coord_to_label_region_info_dict.items():
        label, region_info = label_region_info
        class_id = LABEL_TO_CLASS_ID_DICT[label]
        class_ids.append(class_id)
        rois.append(region_info.region.bbox)
        scores.append(float(region_info.pred_score))
        masks.append(region_info.mask)

    coco_result = {
        'class_ids': np.array(class_ids),
        'rois': np.array(rois),
        'scores': np.array(scores),
        'masks': np.dstack(masks) if len(masks) > 0 else np.array([])
    }
    result_dict['coco_result'] = coco_result
    return result_dict


def prepare_input_data_dir(regions_file, input_data_dir, required_scene_tags, lazycache, img_ext="png"):
    if not lazycache:
        raise ValueError("The lazycache-url could not be found.")

    qt = camhd.lazycache(args.lazycache)

    for scene_tag in required_scene_tags:
        os.makedirs(os.path.join(input_data_dir, scene_tag))

    for region in regions_file.static_regions():
        if region.scene_tag not in required_scene_tags:
            continue

        url = regions_file.mov
        # TODO: Check for sharpness and select the frame.
        sample_frame = region.start_frame + 0.5 * (region.end_frame - region.start_frame)

        img_path = os.path.join(input_data_dir, region.scene_tag)
        sample_frame_path = os.path.join(img_path, "%s_%d.%s"
                                         % (os.path.splitext(os.path.basename(url))[0], sample_frame, img_ext))

        # The script doesn't allow for an already existing input_data_dir.
        # This is added as an additional check to avoid retrieving the frame again.
        if os.path.exists(sample_frame_path):
            logging.info("Already exists: %s" % sample_frame_path)
            continue

        logging.info("Fetching frame %d from %s for contact sheet" % (sample_frame, os.path.basename(url)))
        img = qt.get_frame(url, sample_frame, format=img_ext)
        img.save(sample_frame_path)

    logging.info("The input-data-dir has been created at '%s' by extracting frames from static regions"
                 "of the regions file: %s" % (input_data_dir, regions_file.mov))


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=args.log.upper())

    if args.restrict_gpu:
        logging.info("Setting CUDA_VISIBLE_DEVICES=%s" % str(args.restrict_gpu))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.restrict_gpu

    with open(args.config) as fp:
        analysis_config = json.load(fp)

    required_scene_tags = []
    for label, cur_scene_tags in analysis_config["scene_tags"].items():
        required_scene_tags.extend(cur_scene_tags)

    label_to_segmentation_model_config_dict = {}
    label_to_classification_model_config_dict = {}
    for label, model_config_path in analysis_config["label_to_segmentation_model_config"].items():
        with open(model_config_path) as fp:
            label_to_segmentation_model_config_dict[label] = json.load(fp)

    for label, model_config_path in analysis_config["label_to_classification_model_config"].items():
        with open(model_config_path) as fp:
            label_to_classification_model_config_dict[label] = json.load(fp)

    regions_file_report = None
    if args.regions_file:
        if not os.path.exists(args.regions_file):
            raise ValueError("The regions-file does not exist: %s" % args.regions_file)

        if os.path.exists(args.input_data_dir):
            raise ValueError("The provided input-data-dir already exists: %s. While processing the regions-file, "
                             "the input-data-dir will be created by the script." % args.input_data_dir)

        os.makedirs(args.input_data_dir)

        logging.info("Analysing the regions file: %s" % args.regions_file)
        regions_file = mmd.RegionFile.load(args.regions_file)
        date_time = regions_file.basename.split("-")[1].split("T")
        regions_file_report = {
            "video": regions_file.basename,
            "deployment": DEFAULT_DEPLOYMENT, # TODO: Need to be updated once regions files include deployment info.
            "date": date_time[0],
            "time": date_time[1]
        }
        prepare_input_data_dir(regions_file,
                               args.input_data_dir,
                               required_scene_tags,
                               args.lazycache,
                               img_ext=args.img_ext)

    if args.extract_patches_only:
        if not args.patches_output_dir:
            raise ValueError("The patches-output-dir must be provided when extract-patches-only flag is set.")

    all_frame_reports = []
    all_coco_results = OrderedDict()
    for scene_tag in sorted(os.listdir(args.input_data_dir)):
        for frame_file in sorted(os.listdir(os.path.join(args.input_data_dir, scene_tag))):
            frame_path = os.path.join(args.input_data_dir, scene_tag, frame_file)
            frame_report = analyse_frame(scene_tag,
                                         frame_path,
                                         label_to_segmentation_model_config_dict,
                                         label_to_classification_model_config_dict,
                                         img_ext="png",
                                         patches_output_dir=args.patches_output_dir,
                                         masks_output_dir=args.masks_output_dir,
                                         extract_patches_only=args.extract_patches_only,
                                         coco_format=True if args.coco_result_path else False,
                                         output_dir=args.output_dir,
                                         no_write=args.no_write)
            all_coco_results[frame_report['frame']] = frame_report.pop('coco_result', None)
            all_frame_reports.append(frame_report)

    if regions_file_report:
        regions_file_report["frame_reports"] = all_frame_reports
        with open(args.outfile, "w") as fp:
            json.dump(regions_file_report, fp, indent=4, sort_keys=True, default=get_json_serializable)
    else:
        with open(args.outfile, "w") as fp:
            json.dump(all_frame_reports, fp, indent=4, sort_keys=True, default=get_json_serializable)

    if args.coco_result_path:
        with open(args.coco_result_path, "wb") as fp:
            pickle.dump(all_coco_results, fp)
            print("The results in the coco evaluation format have been written to %s" % args.coco_result_path)
