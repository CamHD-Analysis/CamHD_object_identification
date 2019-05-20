#!/usr/bin/env python3

"""
Schedule a given list of commands to be run one after the other.

Please note that all the paths mentioned in the commands must be absolute paths.

"""

import argparse
import logging
import os
import subprocess
import sys


def get_args():
    parser = argparse.ArgumentParser(description="Schedule set of commands.")
    parser.add_argument("--log",
                        default="INFO",
                        help="Specify the log level. Default: INFO.")

    args = parser.parse_args()

    logging.basicConfig(level=args.log.upper())

    return args


def _run(cmd_list, logfile, py_script=False, restrict_gpu=None, no_write=False, allow_error=False):
    """
    Executes a command through python subprocess and logs the STDOUT and STDERR to the logfile provided.

    """
    cmd = []
    custom_env = os.environ.copy()
    if restrict_gpu is not None:
        logging.info("Setting CUDA_VISIBLE_DEVICES=%s" % str(restrict_gpu))
        custom_env["CUDA_VISIBLE_DEVICES"] = str(restrict_gpu)

    if py_script:
        cmd.append(sys.executable)

    cmd.extend(cmd_list)
    cmd_str = " ".join(cmd)

    with open(logfile, "a") as outfile:
        if not no_write:
            logging.info("Executing command: %s" % cmd_str)
            logging.info("Logging to file: %s" % logfile)
            error_code = subprocess.call(cmd, stdout=outfile, stderr=outfile, env=custom_env)
            if error_code != 0:
                error_msg = "The cmd failed at runtime with error_code %s: %s" % (error_code, cmd_str)
                if not allow_error:
                    raise RuntimeError(error_msg)
                else:
                    logging.error(error_msg)

            return error_code
        else:
            logging.info("Executing command (no-write): %s" % cmd_str)


if __name__ == "__main__":
    args = get_args()

    # TODO: parametrize this.
    # Manually add the parameter values.

    # Running analysis pipeline on testbed for different algorithms.
    analysis_config_dir = "/home/bhuvan/Projects/CamHD_object_identification/analysis_configs"
    test_out_dir = "/home/bhuvan/Projects/camhd_project_details/analysis_pipeline_test_dir/amphipod_testbed_analysis/res_dirs"
    log_dir = "/home/bhuvan/Projects/camhd_project_details/analysis_pipeline_test_dir/amphipod_testbed_analysis/res_dirs/analysis_logs"

    analysis_configs = [
        "analysis_eval-1-1",
        "analysis_eval-1-2",
        "analysis_eval-1-3",
        "analysis_eval-2-1",
        "analysis_eval-2-2",
        "analysis_eval-2-3",
        "analysis_eval-3-1",
        "analysis_eval-3-2",
        "analysis_eval-3-3",
    ]

    commands_logfile_list = []
    for analysis_config in analysis_configs:
        cur_analysis_name = "res-%s" % analysis_config
        cur_outdir = os.path.join(test_out_dir, cur_analysis_name)
        cur_cmd = [
            "/home/bhuvan/Projects/CamHD_object_identification/code/analysis_pipeline.py",
            "--input-data-dir", "/home/bhuvan/Projects/camhd_project_details/analysis_pipeline_test_dir/amphipod_testbed_analysis/test_set_frames",
            "--config",           os.path.join(analysis_config_dir, "%s.json" % analysis_config),
            "--output-dir",       os.path.join(cur_outdir, "our_frame_output_dir"),
            "--outfile",          os.path.join(cur_outdir, "our_report.json"),
            "--coco-result-path", os.path.join(cur_outdir, "our_results_coco_format.pickle")
        ]
        cur_logfile = os.path.join(log_dir, "%s.log" % cur_analysis_name)
        commands_logfile_list.append((" ".join(cur_cmd), cur_logfile))

        # Evaluation script.
        cur_cmd = [
            "/home/bhuvan/Projects/CamHD_object_identification/code/evaluate_instance_segm.py",
            "--images-dir",  "/home/bhuvan/Projects/camhd_project_details/analysis_pipeline_test_dir/amphipod_testbed_analysis/input_frames",
            "--gt-anno-dir", "/home/bhuvan/Projects/camhd_project_details/analysis_pipeline_test_dir/amphipod_testbed_analysis/ground_truth_annos",
            "--class-map", "/home/bhuvan/Projects/CamHD_object_identification/code/class_id_label_map.json",
            "--dt-result-file", os.path.join(cur_outdir, "our_results_coco_format.pickle"),
        ]
        cur_logfile = os.path.join(cur_outdir, "evaluation_report.txt")
        commands_logfile_list.append((" ".join(cur_cmd), cur_logfile))

    restrict_gpu = "1"

    commands_logfile_list = commands_logfile_list
    for i, cmd_logfile in enumerate(commands_logfile_list):
        cmd, cur_logfile = cmd_logfile
        error_code = _run(cmd.split(" "), cur_logfile, py_script=True, restrict_gpu=restrict_gpu)
        print("\nCommand - %s, executed with error code: %s" % (cmd, error_code))


###
# Old command lists:
"""
    # Running U-Net training.
    commands_logfile_list = [
        ("/home/bhuvan/Projects/CamHD_object_identification/code/train_pipeline_unet.py --func train_unet --data-dir /home/bhuvan/Projects/CamHD_object_identification/data/amphipod/amphipod_segmentation/set_1 "
         "--batchnorm --val-split 0.1 --epochs 1000 --batch-size 2 --lr 0.001 "
         "--model-outfile /home/bhuvan/Projects/CamHD_object_identification/trained_models/amphipod_unet-v0.1.hdf5", "/home/bhuvan/Projects/CamHD_object_identification/training_logs/amphipod_unet-v0.1.log"),

        ("/home/bhuvan/Projects/CamHD_object_identification/code/train_pipeline_unet.py --func train_unet --data-dir /home/bhuvan/Projects/CamHD_object_identification/data/amphipod/amphipod_segmentation/set_1_p016 "
         "--batchnorm --val-split 0.1 --epochs 1000 --batch-size 2 --lr 0.001 "
         "--model-outfile /home/bhuvan/Projects/CamHD_object_identification/trained_models/amphipod_unet-v0.2.hdf5", "/home/bhuvan/Projects/CamHD_object_identification/training_logs/amphipod_unet-v0.2.log"),

        ("/home/bhuvan/Projects/CamHD_object_identification/code/train_pipeline_unet.py --func train_unet --data-dir /home/bhuvan/Projects/CamHD_object_identification/data/amphipod/amphipod_segmentation/set_1_2 "
         "--batchnorm --val-split 0.1 --epochs 1000 --batch-size 2 --lr 0.001 "
         "--model-outfile /home/bhuvan/Projects/CamHD_object_identification/trained_models/amphipod_unet-v0.3.hdf5", "/home/bhuvan/Projects/CamHD_object_identification/training_logs/amphipod_unet-v0.3.log"),

        ("/home/bhuvan/Projects/CamHD_object_identification/code/train_pipeline_unet.py --func train_unet --data-dir /home/bhuvan/Projects/CamHD_object_identification/data/amphipod/amphipod_segmentation/set_1_2_p016 "
         "--batchnorm --val-split 0.1 --epochs 1000 --batch-size 2 --lr 0.001 "
         "--model-outfile /home/bhuvan/Projects/CamHD_object_identification/trained_models/amphipod_unet-v0.4.hdf5", "/home/bhuvan/Projects/CamHD_object_identification/training_logs/amphipod_unet-v0.4.log")
    ]
"""
# Old command lists
