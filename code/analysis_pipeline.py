#!/usr/bin/env python3

import pycamhd.lazycache as camhd
import pycamhd.motionmetadata as mmd

from collections import defaultdict
from skimage import io

import argparse
import json
import logging
import os

FRAME_RESOLUTION = (1920, 1080)

# TODO: Try the thresholds for each scene and enter the thresholds here.
SCENE_TAG_TO_SHARPNESS_SCORE_THRESHOLD_DICT = {

}

LABEL_TO_COLOR_DICT = {
    "amphipod": (255, 0, 0),    # RED
    "star": (255, 255, 0)       # Yellow
}

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
                        required=True,
                        help="The analysis config (JSON). Please refer to the sample analyser config.")
    parser.add_argument('--regions-file',
                        required=True,
                        help="The path to the video regions_file on which the analysis needs to be run.")
    parser.add_argument('--analysis-work-dir',
                        required=True,
                        help="The path to the directory where intermediate analysis files can be written.")
    parser.add_argument('--outfile',
                        required=True,
                        help="The path at which the analysis report needs to be saved.")
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


# Utility functions of analyse_frame function.
def _get_raw_mask(frame, label_to_segmentation_model_dict):
    # return stitched_mask
    pass


def _get_patch_coord_to_label_size_dict(label_to_postprocessed_mask_dict):
    # return patch_coord_to_label_size_dict
    pass


def _get_marked_image(frame, patch_coord_to_label_size_dict, label_to_color_dict):
    # return marked_image
    pass


# Utility functions of analyse_frame function: Postprocess functions:
def _postprocess_mask_1(raw_mask):
    #return postprocessed_mask
    pass


def _postprocess_mask_2(raw_mask):
    #return postprocessed_mask
    pass

LABEL_TO_POSTPROCESS_FUNC_DICT = {
    "amphipod": _postprocess_mask_1,
    "star": _postprocess_mask_2
}

# TODO: Add parameter - label_to_classification_model_dict
def analyse_frame(scene_tag, frame_path, label_to_segmentation_model_dict, img_ext="png"):
    work_dir, frame_name = os.path.splitext(frame_path)
    frame_base_name = frame_name.split(".%s" % img_ext)[0]

    frame = io.imread(frame_path)

    label_to_raw_mask_dict = _get_raw_mask(frame, label_to_segmentation_model_dict)
    for label, raw_mask in label_to_raw_mask_dict.items():
        io.imsave(os.path.join(work_dir, "%s_mask_%s.%s"
                               % (frame_base_name, img_ext)), raw_mask, label)

    label_to_postprocessed_mask_dict = {}
    for label, raw_mask in label_to_raw_mask_dict.items():
        label_to_postprocessed_mask_dict[label] = LABEL_TO_POSTPROCESS_FUNC_DICT[label](raw_mask)

    for label, postprocessed_mask in label_to_postprocessed_mask_dict.items():
        io.imsave(os.path.join(work_dir, "%s_postprocessed_mask.%s"
                               % (frame_base_name, img_ext)), postprocessed_mask, label)

    # This contains list of centroids mapped to labels
    patch_coord_to_label_size_dict = _get_patch_coord_to_label_size_dict(label_to_postprocessed_mask_dict)

    # TODO: Extract Patches and Classify the patches to get validated patch_coord_to_label_dict.
    # patch_coord_to_label_size_dict = _validate_by_classification(frame,
    #                                                              patch_coord_to_label_size_dict,
    #                                                              label_to_classification_model_dict)

    marked_image = _get_marked_image(frame, patch_coord_to_label_size_dict, LABEL_TO_COLOR_DICT)
    io.imsave(os.path.join(work_dir, "%s_marked.%s" % (frame_base_name, img_ext)), marked_image)

    # Format the result.
    label_to_location_sizes_dict = defaultdict(dict)
    for patch_coord, label_size in patch_coord_to_label_size_dict.items():
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

    analyse_regions_file(args)
