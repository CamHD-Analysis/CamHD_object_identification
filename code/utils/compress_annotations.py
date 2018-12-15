#!/usr/bin/env python3

"""
Compress the annotations inside a given root directory - a directory under which all annotation (label) JSON files
need to be compressed (recursive). Note: Version of Labelme supported: "3.5.0".

python code/utils/compress_annotations.py --anno-work-dir <root directory> --output <output dir>

"""

import argparse
import glob
import json
import os

LABELME_VESION = "3.5.0"

def get_args():
    parser = argparse.ArgumentParser(description="Compress the annotations inside a given root directory."
                                                 "root directory - a directory under which all annotation (label)"
                                                 "JSON files need to be compressed (recursive)."
                                                 "Note: Version of Labelme supported: '3.5.0'.")
    parser.add_argument('--anno-work-dir',
                        dest="anno_work_dir",
                        required=True,
                        help="The path to the root directory.")
    parser.add_argument('--output',
                        help="The path to the output directory. Default: Overwrite the existing annotation file.")

    return parser.parse_args()


def _compress_labelme_annotation_file(anno_file, outfile):
    with open(anno_file) as fp:
        anno = json.load(fp)

    if anno["version"] != LABELME_VESION:
        return

    with open(outfile, "w") as fp:
        json.dump({"shapes": anno["shapes"]}, fp)


def compress_annotations(anno_work_dir, output_dir):
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    recursive_json_files = os.path.join(anno_work_dir, "**", "*.json")
    for anno_file in glob.iglob(recursive_json_files, recursive=True):
        outfile = os.path.join(output_dir, anno_file) if output_dir else anno_file
        _compress_labelme_annotation_file(anno_file, outfile)


if __name__ == "__main__":
    args = get_args()
    compress_annotations(args.anno_work_dir, args.output)
