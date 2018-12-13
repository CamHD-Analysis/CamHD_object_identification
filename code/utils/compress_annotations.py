#!/usr/bin/env python3

"""
Compress the annotations inside a given root directory - a directory under which all annotation (label) JSON files
need to be compressed (recursive). Note: Version of Labelme supported: "3.5.0".

python code/utils/compress_annotations.py --rootdir <root directory>

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
    parser.add_argument('--rootdir',
                        required=True,
                        help="The path to the root directory.")

    return parser.parse_args()


def _compress_labelme_annotation_file(anno_file):
    with open(anno_file) as fp:
        anno = json.load(fp)

    if anno["version"] != LABELME_VESION:
        return

    with open(anno_file, "w") as fp:
        json.dump(anno["shapes"], fp)


def compress_annotations(root_dir):
    recursive_json_files = os.path.join(root_dir, "**", "*.json")
    for anno_file in glob.iglob(recursive_json_files, recursive=True):
        _compress_labelme_annotation_file(anno_file)


if __name__ == "__main__":
    args = get_args()
    compress_annotations(args.rootdir)
