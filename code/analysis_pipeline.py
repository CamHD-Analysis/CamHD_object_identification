#!/usr/bin/env python3

import pycamhd.lazycache as camhd
import pycamhd.motionmetadata as mmd

import argparse
import json
import logging
import os


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


def analyse_regions_file(args):
    if not args.lazycache:
        raise ValueError("The lazycache-url could not be found.")

    pass


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=args.log.upper())

    analyse_regions_file(args)
