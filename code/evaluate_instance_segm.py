#!/usr/bin/env python3

"""
Evaluate an analysis framework's output report (from analysis_pipeline.py) against the ground truth test set.

Evaluates using the Coco Evaluation method both for "segm" and "bbox".

Input:
- Path to directory containing ground truth raw images, each having name: <image_name>.<image_ext>.
- Path to directory containing ground truth annotations from labelme, each having name: <image_name>.json.
- Path to the result pickle file containing the predicted outputs in coco format, which is a list of dictionaries.
- Path to the JSON file containing the class_id to class_label map.


Note: This requires few changes to be patched to the "cocoapi" library. The patch diff can be found in
the 'code' directory of this repository (code/cocoapi_diff.txt).

"""

from mrcnn import visualize
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from skimage import measure

import argparse
import cv2
import json
import logging
import numpy as np
import os
import pickle
import re


IGNORE_SUFFIX = '_hard'

def get_args():
    parser = argparse.ArgumentParser(description=
    """
    Run Analysis on a given video regions_file and output the analysis report.

    """)
    parser.add_argument('--images-dir',
                        required=True,
                        help="The path to the directory containing the input test frames.")
    parser.add_argument('--gt-anno-dir',
                        required=True,
                        help="The path to the directory containing the ground truth annotations (labelme format) "
                             "for each input frame. The suffix '_hard' for labels are considered as 'ignore' labels.")
    parser.add_argument('--dt-result-file',
                        required=True,
                        help="The path to the detection results pickle containing the predictions in coco format.")
    parser.add_argument('--class-map',
                        help="The class id to label map JSON file.")
    parser.add_argument('--include-ignore',
                        action='store_true',
                        help="Flag to indicate if the ignore (hard) cases must be considered as dont-case conditions.")
    parser.add_argument('--display-instances',
                        action='store_true',
                        help="Flag to indicate if the instances need to be displayed in marked images.")
    parser.add_argument('--image-ext',
                        dest='img_ext',
                        default='png',
                        help="The image file extension. Default: png.")
    parser.add_argument("--log",
                        default="DEBUG",
                        help="Specify the log level. Default: DEBUG.")

    args = parser.parse_args()
    logging.basicConfig(level=args.log.upper())

    return args


def _build_coco_results(image_ids, rois, class_ids, scores, masks):
    """
    Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
            }
            results.append(result)

    return results


def get_res_coco_obj(resutls, image_ids, class_id_label_map):
    """
    Get a coco object for loading results in the coco eval format.

    """
    coco = COCO()
    coco.dataset["images"] = [{"id": id} for id in image_ids]
    coco.dataset["categories"] = [{'id': id, 'name': label} for id, label in class_id_label_map.items()]
    # XXX: Hack to patch the img ids.
    coco.imgs = {id: {"id": id} for id in image_ids}
    coco_obj = coco.loadRes(resutls)
    return coco_obj


def evaluate_coco(gt_coco_obj, dt_coco_obj, image_ids, eval_type="segm"):
    cocoEval = COCOeval(gt_coco_obj, dt_coco_obj, eval_type)
    cocoEval.params.imgIds = image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def evaluate(images_dir,
             gt_anno_dir,
             dt_results,
             class_id_label_map,
             include_ignore=False,
             display_instances=False,
             img_ext="png"):
    class_label_id_map = {v:k for k, v in class_id_label_map.items()}
    allowed_classes = list(class_id_label_map.values())
    if include_ignore:
        ignore_classes = ["%s%s" % (x, IGNORE_SUFFIX) for x in allowed_classes]
        allowed_classes.extend(ignore_classes)

    print("Evaluating class labels: %s (ignore suffix: %s)" % (allowed_classes, IGNORE_SUFFIX))

    image_ids = list(dt_results.keys())

    # Build Coco GT object.
    all_gt_results = []
    for image_id in image_ids:
        anno_file = os.path.join(gt_anno_dir, "%s.json" % image_id)
        with open(anno_file) as fp:
            anno_dict = json.load(fp)
            del anno_dict['imageData']

        annos = anno_dict["shapes"]
        if len(annos) < 1:
            raise ValueError("No annotations found for %s" % image_id)

        # Get the results for the image by processing each anno.
        # TODO: Format the output directly without using build_coco_results.
        cur_image_res = []
        for anno in annos:
            cur_anno_res = {"image_id": image_id}

            class_label = anno["label"]
            if class_label not in allowed_classes:
                continue

            if IGNORE_SUFFIX in class_label:
                cur_anno_res['ignore'] = 1
                class_label = re.sub('%s$' % IGNORE_SUFFIX, '', class_label)

            class_id = class_label_id_map[class_label]
            cur_anno_res['category_id'] = class_id

            # Get the mask.
            # TODO: The frame size is hard coded.
            frame_mask = np.zeros(shape=(1080, 1920), dtype=np.uint8)
            contours = np.asarray(anno["points"])
            cv2.fillPoly(frame_mask, pts=[contours], color=1)
            cur_anno_res['segmentation'] = maskUtils.encode(np.asfortranarray(frame_mask))

            regions = measure.regionprops(measure.label(frame_mask > 0))
            if len(regions) < 1:
                raise ValueError("No connected region found: %s" % image_id)

            if len(regions) > 1:
                raise ValueError("The length of annotated connected regions is greater than 1: %s" % image_id)

            # Get the bbox.
            bbox = regions[0].bbox
            cur_anno_res["bbox"] = [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]]

            cur_image_res.append(cur_anno_res)

        all_gt_results.extend(cur_image_res)

    gt_coco_obj = get_res_coco_obj(all_gt_results, image_ids, class_id_label_map)
    print("Loaded GT coco object.")

    # Build Coco DT object.
    all_dt_results = []
    for image_id, r in dt_results.items():
        res_formatted = _build_coco_results([image_id], r["rois"], r["class_ids"], r["scores"], r["masks"])
        all_dt_results.extend(res_formatted)

    dt_coco_obj = get_res_coco_obj(all_dt_results, image_ids, class_id_label_map)
    print("Loaded DT coco object.")

    # Evaluate using CocoEval.
    print("\nRunning Evaluations:")
    evaluate_coco(gt_coco_obj, dt_coco_obj, image_ids, eval_type="segm")
    evaluate_coco(gt_coco_obj, dt_coco_obj, image_ids, eval_type="bbox")

    if display_instances:
        for frame_name, r in dt_results.items():
            image = np.array(Image.open(os.path.join(images_dir, "%s.%s" % (frame_name, img_ext))))
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_id_label_map, r['scores'])

    logging.info("Evaluation complete.")


if __name__ == "__main__":
    args = get_args()

    with open(args.dt_result_file, "rb") as fp:
        dt_results = pickle.load(fp)

    with open(args.class_map) as fp:
        class_id_label_map = json.load(fp)
        class_id_label_map = {int(k): v for k, v in class_id_label_map.items()}

    evaluate(args.images_dir,
             args.gt_anno_dir,
             dt_results,
             class_id_label_map,
             include_ignore=args.include_ignore,
             display_instances=args.display_instances,
             img_ext=args.img_ext)
