#!/usr/bin/env python3

"""
Sample the frames of the video from given set of scenes from the given regions files
and store them in the given output path.

python code/utils/sample_random_data.py ../CamHD_motion_metadata/RS03ASHS/PN03B/06-CAMHDA301/2018/0[78]/*
    --scenes d5A_p1_z1,d5A_p5_z1,d5A_p6_z1,d5A_p0_z1,d5A_p7_z1,d5A_p8_z1
    --output data/unet_annotation_data/20180711_20180831

"""

import pycamhd.lazycache as camhd
import pycamhd.motionmetadata as mmd

import argparse
import glob
import logging
import os
import random


def get_args():
    parser = argparse.ArgumentParser(description="Get random sample frames from the given regions files for each scene.")
    parser.add_argument('input',
                        metavar='N',
                        nargs='*',
                        help='Files or paths to process.')
    parser.add_argument('--scenes',
                        required=True,
                        help="The set of scenes for which random frames need to be sampled. Provide comma separated string.")
    parser.add_argument('--prob',
                        type=float,
                        default=0.5,
                        help="The probability of a static region being selected. Default: 0.5.")
    parser.add_argument('--output',
                        required=True,
                        help="The path to the folder where the sampled frames need to be written.")
    parser.add_argument('--image-ext',
                        dest='img_ext',
                        default='png',
                        help="The image file extension. Default: png.")
    parser.add_argument('--lazycache-url', dest='lazycache',
                        default=os.environ.get("LAZYCACHE_URL", None),
                        help='URL to Lazycache repo server (only needed if classifying)')
    parser.add_argument("--log",
                        default="INFO",
                        help="Specify the log level. Default: INFO.")

    args = parser.parse_args()

    # Set up the Lazycache connection.
    args.qt = camhd.lazycache(args.lazycache)

    # Scene parsing.
    args.scenes = [x.strip() for x in args.scenes.split(",")]

    if os.path.exists(args.output):
        raise ValueError("The given output path already exists. Please provide a new output path: %s" % args.output)

    return args



def _get_random_frames(regions_file, img_path, qt, img_ext, scene_tag, prob):
    def _sample_prob(prob):
        """
        Returns True or False based on the given probability. Bernoille trial with given probability.

        """
        r = random.uniform(0, 1)
        # TODO: Check if this is correctly sampling.
        if r <= prob:
            return True

        return False


    for region in regions_file.static_regions():
        if region.scene_tag != scene_tag:
            continue

        if not _sample_prob(prob):
            continue

        url = regions_file.mov

        sample_frame = region.start_frame + 0.5 * (region.end_frame - region.start_frame)

        sample_frame_path = os.path.join(img_path, "%s_%d.%s"
                                         % (os.path.splitext(os.path.basename(url))[0], sample_frame, img_ext))
        logging.info("Fetching frame %d from %s for contact sheet" % (sample_frame, os.path.basename(url)))
        img = qt.get_frame(url, sample_frame, format=img_ext)
        img.save(sample_frame_path)


def sample_random_frames(args):
    blacklist_days = ["01", "02", "03", "10", "20"]
    def _process(infile):
        day = os.path.basename(infile).split("-")[1][6:8]
        if day in blacklist_days:
            return

        logging.info("Sampling the frames from regions file: {}".format(infile))
        regions_file = mmd.RegionFile.load(infile)

        for scene in args.scenes:
            output_path = os.path.join(args.output, scene)
            os.makedirs(output_path)
            _get_random_frames(regions_file, output_path, args.qt, args.img_ext, scene, args.prob)


    for pathin in args.input:
        for infile in glob.iglob(pathin):
            if os.path.isdir(infile):
                infile = os.path.join(infile, "*_regions.json")
                for f in glob.iglob(infile):
                    _process(f)
            else:
                _process(infile)


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=args.log.upper())
    sample_random_frames(args)