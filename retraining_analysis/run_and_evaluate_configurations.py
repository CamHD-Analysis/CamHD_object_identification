#!/usr/bin/env python3

"""
Runs and evaluates the analysis pipeline configurations for various unet and vgg models for retraining analysis.

"""

import copy
import json
import os
from subprocess import call
import sys
import time

MY_ENV = os.environ.copy()
MY_ENV["CUDA_VISIBLE_DEVICES"] = "1"

analysis_config_template = {
    "analysis_version": "analysis-eval-3-3", # placeholder example
    "labels": ["amphipod"],
    "scene_tags": {
        "amphipod": ["d5A_p0_z1", "d5A_p1_z1", "d5A_p6_z1"]
    },
    "label_to_segmentation_model_config": {
        "amphipod": "/home/bhuvan/Projects/CamHD_object_identification/trained_models/amphipod_unet-v0.4.json" # placeholder example
    },
    "label_to_classification_model_config": {
        "amphipod": "/home/bhuvan/Projects/CamHD_object_identification/trained_models/amphipod_cnn-v0.6.json" # placeholder example
    }
}

ROOT_DIR = "/home/bhuvan/Projects/CamHD_object_identification/retraining_analysis"

run_number = sys.argv[1]
run_dir = os.path.join(ROOT_DIR, run_number)
print("Running evaluations for %s, and the outputs would be present in %s." % (run_number, os.path.abspath(run_dir)))

TRAINED_MODELS_DIR   = os.path.join(run_dir, "trained_models")
ANALYSIS_CONFIGS_DIR = os.path.join(run_dir, "analysis_configs")
ANALYSIS_OUTPUT_DIR  = os.path.join(run_dir, "test_bed_outputs")
LOGS_DIR             = os.path.join(run_dir, "training_logs")
EVALUATIONS_DIR      = os.path.join(run_dir, "evaluations")


def run_cmd(cmd_list, logfile, no_write=False):
    print("\nRunning cmd: %s, and it would be logged at %s." % (" ".join(cmd_list), logfile))
    if not no_write:
        with open(logfile, "w") as fp:
            return_code = call(cmd_list, env=MY_ENV, stdout=fp, stderr=fp)
            print("Command completed with return_code %s" % return_code)


def run_analysis(unet_model_name, vgg_model_name, no_write=False):
    start_time = time.time()
    print("\n\nRunning analysis for (%s, %s)"
          % (os.path.basename(unet_model_name), os.path.basename(vgg_model_name)))

    analysis_name = "analysis-%s_%s" % (unet_model_name, vgg_model_name)

    # Prepare analysis config.
    cur_analysis_config = copy.copy(analysis_config_template)
    cur_analysis_config["analysis_version"] = analysis_name
    cur_analysis_config["label_to_segmentation_model_config"]["amphipod"]   = os.path.join(TRAINED_MODELS_DIR, "%s.json" % unet_model_name)
    cur_analysis_config["label_to_classification_model_config"]["amphipod"] = os.path.join(TRAINED_MODELS_DIR, "%s.json" % vgg_model_name)

    analysis_config_path = os.path.join(ANALYSIS_CONFIGS_DIR, "%s.json" % analysis_name)
    with open(analysis_config_path, 'w') as fp:
        json.dump(cur_analysis_config, fp, sort_keys=True, indent=4)

    analysis_out_dir = os.path.join(ANALYSIS_OUTPUT_DIR, analysis_name)
    analysis_coco_file = os.path.join(analysis_out_dir, "our_results_coco_format.pickle")

    # Run analysis pipeline.
    analysis_cmd = [
        "/home/bhuvan/anaconda3/envs/cvenv/bin/python",
        "/home/bhuvan/Projects/CamHD_object_identification/code/analysis_pipeline.py",
        "--input-data-dir",
        "/home/bhuvan/Projects/camhd_project_details/analysis_pipeline_test_dir/amphipod_testbed_analysis/test_set_frames",
        "--config",
        analysis_config_path,
        "--output-dir",
        os.path.join(analysis_out_dir, "our_frame_output_dir"),
        "--outfile",
        os.path.join(analysis_out_dir, "our_report.json"),
        "--coco-result-path",
        analysis_coco_file,
    ]
    run_cmd(analysis_cmd, os.path.join(LOGS_DIR, "%s.log" % analysis_name), no_write=no_write)

    # Run evaluation
    evaluation_cmd = [
        "/home/bhuvan/anaconda3/envs/cvenv/bin/python",
        "/home/bhuvan/Projects/CamHD_object_identification/code/evaluate_instance_segm.py",
        "--images-dir",
        "/home/bhuvan/Projects/camhd_project_details/analysis_pipeline_test_dir/amphipod_testbed_analysis/input_frames",
        "--gt-anno-dir",
        "/home/bhuvan/Projects/camhd_project_details/analysis_pipeline_test_dir/amphipod_testbed_analysis/ground_truth_annos",
        "--class-map",
        "/home/bhuvan/Projects/CamHD_object_identification/code/class_id_label_map.json",
        "--dt-result-file",
        analysis_coco_file
    ]
    run_cmd(evaluation_cmd, os.path.join(EVALUATIONS_DIR, "%s.txt" % analysis_name), no_write=no_write)

    end_time = time.time()
    time_taken = end_time - start_time
    print("Time taken for analysis: %s seconds" % time_taken)


if __name__ == "__main__":
    unet_models = ["unet-100", "unet-200", "unet-300", "unet-400", "unet-500"]
    vgg_models = [
        "vgg-100_RCP-0_CNP",
        "vgg-200_RCP-0_CNP",
        "vgg-300_RCP-0_CNP",
        "vgg-400_RCP-0_CNP",
        "vgg-500_RCP-0_CNP",

        "vgg-500_RCP-100_CNP",
        "vgg-500_RCP-200_CNP",
        "vgg-500_RCP-300_CNP",
        "vgg-500_RCP-400_CNP",
        "vgg-500_RCP-500_CNP",
    ]

    for unet_model in unet_models:
        for vgg_model in vgg_models:
            run_analysis(unet_model, vgg_model)
