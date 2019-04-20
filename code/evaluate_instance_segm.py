#!/usr/bin/env python3

"""
Evaluate an analysis framework's output report (from analysis_pipeline.py) against the ground truth test set.

Evaluates using the Coco Evaluation method both for "segm" and "bbox".

Input:
- Path to directory containing ground truth raw images, each having name: <image_name>.<image_ext>.
- Path to directory containing ground truth annotations from labelme, each having name: <image_name>.json.
- Path to the result pickle file containing the predicted outputs in coco format, which is a list of dictionaries.
- Class name map.

"""

import pycocotools
