Retraining Analysis

The data match the descriptions in WACV paper:
unet-patches_masks: 537 patches-masks from Set-1 and Set-2 from p016 (scenes 0, 1 and 6).
vgg-withoutCNP: 537 postive + 392 negative (RCP) + 
vgg-withCNP: 537 positive + 1387 negative (392 RCP + 995 CPN)


# Getting extra RCP.

import os
import glob
import random
import shutil

max_count = 100
src_rcp_dir = "/media/bhuvan/706f17dd-eaf1-4c3b-aef2-494e232321ae/bhuvan/projects/CamHD_object_identification/data/amphipod_classification/random_negative_unet_data/patches_output_dir_rand_4000"
old_rcp_dir = "/home/bhuvan/Projects/CamHD_object_identification/retraining_analysis/data_RCP_added/vgg-RCPwithoutCNP/nonamphipod"
extra_rcp_out_dir = "/home/bhuvan/Projects/CamHD_object_identification/retraining_analysis/data_RCP_added/vgg-RCPwithoutCNP/extra_rcp"
p016_dirs = [
    "/media/bhuvan/706f17dd-eaf1-4c3b-aef2-494e232321ae/bhuvan/projects/CamHD_object_identification/annotations_work_dir/amphipod/amphipod_segmentation/set_1/201807_201808/d5A_p0_z1",
    "/media/bhuvan/706f17dd-eaf1-4c3b-aef2-494e232321ae/bhuvan/projects/CamHD_object_identification/annotations_work_dir/amphipod/amphipod_segmentation/set_1/201807_201808/d5A_p1_z1",
    "/media/bhuvan/706f17dd-eaf1-4c3b-aef2-494e232321ae/bhuvan/projects/CamHD_object_identification/annotations_work_dir/amphipod/amphipod_segmentation/set_1/201807_201808/d5A_p6_z1",
    "/media/bhuvan/706f17dd-eaf1-4c3b-aef2-494e232321ae/bhuvan/projects/CamHD_object_identification/annotations_work_dir/amphipod/amphipod_segmentation/set_2/201809_201810/d5A_p0_z1",
    "/media/bhuvan/706f17dd-eaf1-4c3b-aef2-494e232321ae/bhuvan/projects/CamHD_object_identification/annotations_work_dir/amphipod/amphipod_segmentation/set_2/201809_201810/d5A_p1_z1",
    "/media/bhuvan/706f17dd-eaf1-4c3b-aef2-494e232321ae/bhuvan/projects/CamHD_object_identification/annotations_work_dir/amphipod/amphipod_segmentation/set_2/201809_201810/d5A_p6_z1"
]

src_rcp_images = glob.glob(os.path.join(src_rcp_dir, "*.png"))
random.shuffle(src_rcp_images)

old_rcp_images = glob.glob(os.path.join(old_rcp_dir, "*.png"))
old_rcp_basenames = list(map(os.path.basename, old_rcp_images))

p016_frames = []
for cur_p016_dir in p016_dirs:
    p016_frames.extend(glob.glob(os.path.join(cur_p016_dir, "*.png")))
p016_frame_basenames = list(map(os.path.basename, p016_frames))


cur_count = 0
for cur_rcp_image in src_rcp_images:
    cur_rcp_basename = os.path.basename(cur_rcp_image)
    if cur_count >= max_count:
        break

    if cur_rcp_basename in old_rcp_basenames:
        continue

    cur_frame_basename = '_'.join(cur_rcp_basename.split('_')[:2]) + ".png"
    if cur_frame_basename not in p016_frame_basenames:
        continue

    cur_count += 1
    shutil.copy(cur_rcp_image, os.path.join(extra_rcp_out_dir, cur_rcp_basename))


# Create 128px square cropped versions.

import cv2
import glob
import os

src_dir = "./extra_rcp/"
rcp_images = glob.glob(os.path.join(src_dir, "*.png"))

for rcp_image_file in rcp_images:
    rcp_image = cv2.imread(rcp_image_file)
    offset = 128 - 64 # center - (new_patch_size / 2)
    center_crop = rcp_image[offset:offset + 128, offset:offset+128]
    pad_size = 64
    cropped_rcp = cv2.copyMakeBorder(center_crop,
                                     top=pad_size,
                                     bottom=pad_size,
                                     left=pad_size,
                                     right=pad_size,
                                     borderType=cv2.BORDER_CONSTANT,
                                     value=0)
    cropped_file = '_cropped'.join(os.path.splitext(rcp_image_file))
    cv2.imwrite(cropped_file, cropped_rcp)

# Exrta RCP - 200 created


# Creating data chunks for retraining analysis after copying unet orig data andd vgg orig data with RCP added. 
# Run from root_dir/retraining_analysis/data-retraining-analysis/run_<run_number>.

import os
import glob
import random
import shutil

def unet_selection(src_dir, dest_dir, count):
    src_patches_dir = os.path.join(src_dir, "patches")
    src_masks_dir = os.path.join(src_dir, "masks")

    dest_patches_dir = os.path.join(dest_dir, "patches")
    dest_masks_dir = os.path.join(dest_dir, "masks")
    os.makedirs(dest_patches_dir)
    os.makedirs(dest_masks_dir)

    src_patches = glob.glob(os.path.join(src_patches_dir, "*.png"))
    random.shuffle(src_patches)

    selected_patches = src_patches[:count]

    for patch in selected_patches:
        shutil.copy(patch, os.path.join(dest_patches_dir, os.path.basename(patch)))

        mask = os.path.join(src_masks_dir, '_mask'.join(os.path.splitext(os.path.basename(patch))))
        shutil.copy(mask, os.path.join(dest_masks_dir, os.path.basename(mask)))


def vgg_selection(src_dir, dest_dir, count):
    src_pos_dir = os.path.join(src_dir, "amphipod")
    src_neg_dir = os.path.join(src_dir, "nonamphipod")

    dest_pos_dir = os.path.join(dest_dir, "amphipod")
    dest_neg_dir = os.path.join(dest_dir, "nonamphipod")
    os.makedirs(dest_pos_dir)
    os.makedirs(dest_neg_dir)

    for cur_src_dir, cur_dest_dir in [(src_pos_dir, dest_pos_dir), (src_neg_dir, dest_neg_dir)]:
        src_patches = glob.glob(os.path.join(cur_src_dir, "*.png"))
        random.shuffle(src_patches)
        selected_patches = src_patches[:count]
        for patch in selected_patches:
            shutil.copy(patch, os.path.join(cur_dest_dir, os.path.basename(patch)))



def add_cnp(cnp_src_dir, src_data_dir, dest_dir, count):
    shutil.copytree(src_data_dir, dest_dir)

    dest_neg_dir = os.path.join(dest_dir, "nonamphipod")

    cnp_patches = glob.glob(os.path.join(cnp_src_dir, "*.png"))
    random.shuffle(cnp_patches)

    selected_cnp_patches = cnp_patches[:count]

    for patch in selected_cnp_patches:
        shutil.copy(patch, os.path.join(dest_neg_dir, os.path.basename(patch)))


unet_selection("../unet-orig", "unet/unet-100", 100)
unet_selection("../unet-orig", "unet/unet-200", 200)
unet_selection("../unet-orig", "unet/unet-300", 300)
unet_selection("../unet-orig", "unet/unet-400", 400)
unet_selection("../unet-orig", "unet/unet-500", 500)


vgg_selection("../vgg-orig_RCP-0_CNP", "vgg/vgg-100_RCP-0_CNP", 100)
vgg_selection("../vgg-orig_RCP-0_CNP", "vgg/vgg-200_RCP-0_CNP", 200)
vgg_selection("../vgg-orig_RCP-0_CNP", "vgg/vgg-300_RCP-0_CNP", 300)
vgg_selection("../vgg-orig_RCP-0_CNP", "vgg/vgg-400_RCP-0_CNP", 400)
vgg_selection("../vgg-orig_RCP-0_CNP", "vgg/vgg-500_RCP-0_CNP", 500)

add_cnp("../CNP-orig", "vgg/vgg-500_RCP-0_CNP", "vgg/vgg-500_RCP-100_CNP", 100)
add_cnp("../CNP-orig", "vgg/vgg-500_RCP-0_CNP", "vgg/vgg-500_RCP-200_CNP", 200)
add_cnp("../CNP-orig", "vgg/vgg-500_RCP-0_CNP", "vgg/vgg-500_RCP-300_CNP", 300)
add_cnp("../CNP-orig", "vgg/vgg-500_RCP-0_CNP", "vgg/vgg-500_RCP-400_CNP", 400)
add_cnp("../CNP-orig", "vgg/vgg-500_RCP-0_CNP", "vgg/vgg-500_RCP-500_CNP", 500)


### Individual Model Training.
# Run from root dir.

### Run: run_1
# UNet training.
CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_unet.py --func train_unet --data-dir retraining_analysis/data-retraining-analysis/run_1/unet/unet-100 --batchnorm --val-split 0.1 --epochs 1000 --batch-size 4 --model-outfile retraining_analysis/run_1/trained_models/unet-100.hdf5 > retraining_analysis/run_1/training_logs/unet-100.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_unet.py --func train_unet --data-dir retraining_analysis/data-retraining-analysis/run_1/unet/unet-200 --batchnorm --val-split 0.1 --epochs 1000 --batch-size 4 --model-outfile retraining_analysis/run_1/trained_models/unet-200.hdf5 > retraining_analysis/run_1/training_logs/unet-200.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_unet.py --func train_unet --data-dir retraining_analysis/data-retraining-analysis/run_1/unet/unet-300 --batchnorm --val-split 0.1 --epochs 1000 --batch-size 4 --model-outfile retraining_analysis/run_1/trained_models/unet-300.hdf5 > retraining_analysis/run_1/training_logs/unet-300.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_unet.py --func train_unet --data-dir retraining_analysis/data-retraining-analysis/run_1/unet/unet-400 --batchnorm --val-split 0.1 --epochs 1000 --batch-size 4 --model-outfile retraining_analysis/run_1/trained_models/unet-400.hdf5 > retraining_analysis/run_1/training_logs/unet-400.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_unet.py --func train_unet --data-dir retraining_analysis/data-retraining-analysis/run_1/unet/unet-500 --batchnorm --val-split 0.1 --epochs 1000 --batch-size 4 --model-outfile retraining_analysis/run_1/trained_models/unet-500.hdf5 > retraining_analysis/run_1/training_logs/unet-500.log 2>&1

# VGG training.
CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_1/vgg/vgg-100_RCP-0_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_1/trained_models/vgg-100_RCP-0_CNP.hdf5 > retraining_analysis/run_1/training_logs/vgg-100_RCP-0_CNP.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_1/vgg/vgg-200_RCP-0_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_1/trained_models/vgg-200_RCP-0_CNP.hdf5 > retraining_analysis/run_1/training_logs/vgg-200_RCP-0_CNP.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_1/vgg/vgg-300_RCP-0_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_1/trained_models/vgg-300_RCP-0_CNP.hdf5 > retraining_analysis/run_1/training_logs/vgg-300_RCP-0_CNP.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_1/vgg/vgg-400_RCP-0_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_1/trained_models/vgg-400_RCP-0_CNP.hdf5 > retraining_analysis/run_1/training_logs/vgg-400_RCP-0_CNP.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_1/vgg/vgg-500_RCP-0_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_1/trained_models/vgg-500_RCP-0_CNP.hdf5 > retraining_analysis/run_1/training_logs/vgg-500_RCP-0_CNP.log 2>&1


CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_1/vgg/vgg-500_RCP-100_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_1/trained_models/vgg-500_RCP-100_CNP.hdf5 > retraining_analysis/run_1/training_logs/vgg-500_RCP-100_CNP.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_1/vgg/vgg-500_RCP-200_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_1/trained_models/vgg-500_RCP-200_CNP.hdf5 > retraining_analysis/run_1/training_logs/vgg-500_RCP-200_CNP.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_1/vgg/vgg-500_RCP-300_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_1/trained_models/vgg-500_RCP-300_CNP.hdf5 > retraining_analysis/run_1/training_logs/vgg-500_RCP-300_CNP.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_1/vgg/vgg-500_RCP-400_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_1/trained_models/vgg-500_RCP-400_CNP.hdf5 > retraining_analysis/run_1/training_logs/vgg-500_RCP-400_CNP.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_1/vgg/vgg-500_RCP-500_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_1/trained_models/vgg-500_RCP-500_CNP.hdf5 > retraining_analysis/run_1/training_logs/vgg-500_RCP-500_CNP.log 2>&1

### Running analysis on test bed.
/home/bhuvan/Projects/CamHD_object_identification/retraining_analysis/run_and_evaluate_configurations.py run_1



### Run: run_2
# UNet training.
CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_unet.py --func train_unet --data-dir retraining_analysis/data-retraining-analysis/run_2/unet/unet-100 --batchnorm --val-split 0.1 --epochs 1000 --batch-size 4 --model-outfile retraining_analysis/run_2/trained_models/unet-100.hdf5 > retraining_analysis/run_2/training_logs/unet-100.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_unet.py --func train_unet --data-dir retraining_analysis/data-retraining-analysis/run_2/unet/unet-200 --batchnorm --val-split 0.1 --epochs 1000 --batch-size 4 --model-outfile retraining_analysis/run_2/trained_models/unet-200.hdf5 > retraining_analysis/run_2/training_logs/unet-200.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_unet.py --func train_unet --data-dir retraining_analysis/data-retraining-analysis/run_2/unet/unet-300 --batchnorm --val-split 0.1 --epochs 1000 --batch-size 4 --model-outfile retraining_analysis/run_2/trained_models/unet-300.hdf5 > retraining_analysis/run_2/training_logs/unet-300.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_unet.py --func train_unet --data-dir retraining_analysis/data-retraining-analysis/run_2/unet/unet-400 --batchnorm --val-split 0.1 --epochs 1000 --batch-size 4 --model-outfile retraining_analysis/run_2/trained_models/unet-400.hdf5 > retraining_analysis/run_2/training_logs/unet-400.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_unet.py --func train_unet --data-dir retraining_analysis/data-retraining-analysis/run_2/unet/unet-500 --batchnorm --val-split 0.1 --epochs 1000 --batch-size 4 --model-outfile retraining_analysis/run_2/trained_models/unet-500.hdf5 > retraining_analysis/run_2/training_logs/unet-500.log 2>&1

# VGG training - RCP without CNP.
CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_2/vgg/vgg-100_RCP-0_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_2/trained_models/vgg-100_RCP-0_CNP.hdf5 > retraining_analysis/run_2/training_logs/vgg-100_RCP-0_CNP.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_2/vgg/vgg-200_RCP-0_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_2/trained_models/vgg-200_RCP-0_CNP.hdf5 > retraining_analysis/run_2/training_logs/vgg-200_RCP-0_CNP.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_2/vgg/vgg-300_RCP-0_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_2/trained_models/vgg-300_RCP-0_CNP.hdf5 > retraining_analysis/run_2/training_logs/vgg-300_RCP-0_CNP.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_2/vgg/vgg-400_RCP-0_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_2/trained_models/vgg-400_RCP-0_CNP.hdf5 > retraining_analysis/run_2/training_logs/vgg-400_RCP-0_CNP.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_2/vgg/vgg-500_RCP-0_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_2/trained_models/vgg-500_RCP-0_CNP.hdf5 > retraining_analysis/run_2/training_logs/vgg-500_RCP-0_CNP.log 2>&1

# VGG training - RCP with CNP.
CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_2/vgg/vgg-500_RCP-100_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_2/trained_models/vgg-500_RCP-100_CNP.hdf5 > retraining_analysis/run_2/training_logs/vgg-500_RCP-100_CNP.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_2/vgg/vgg-500_RCP-200_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_2/trained_models/vgg-500_RCP-200_CNP.hdf5 > retraining_analysis/run_2/training_logs/vgg-500_RCP-200_CNP.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_2/vgg/vgg-500_RCP-300_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_2/trained_models/vgg-500_RCP-300_CNP.hdf5 > retraining_analysis/run_2/training_logs/vgg-500_RCP-300_CNP.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_2/vgg/vgg-500_RCP-400_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_2/trained_models/vgg-500_RCP-400_CNP.hdf5 > retraining_analysis/run_2/training_logs/vgg-500_RCP-400_CNP.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_2/vgg/vgg-500_RCP-500_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_2/trained_models/vgg-500_RCP-500_CNP.hdf5 > retraining_analysis/run_2/training_logs/vgg-500_RCP-500_CNP.log 2>&1


### Running analysis on test bed.
python /home/bhuvan/Projects/CamHD_object_identification/retraining_analysis/run_and_evaluate_configurations.py run_2 > retraining_analysis/run_2/training_logs/analysis-master.log 2>&1


### Run: run_3
# UNet training.
CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_unet.py --func train_unet --data-dir retraining_analysis/data-retraining-analysis/run_3/unet/unet-100 --batchnorm --val-split 0.1 --epochs 1000 --batch-size 4 --model-outfile retraining_analysis/run_3/trained_models/unet-100.hdf5 > retraining_analysis/run_3/training_logs/unet-100.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_unet.py --func train_unet --data-dir retraining_analysis/data-retraining-analysis/run_3/unet/unet-200 --batchnorm --val-split 0.1 --epochs 1000 --batch-size 4 --model-outfile retraining_analysis/run_3/trained_models/unet-200.hdf5 > retraining_analysis/run_3/training_logs/unet-200.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_unet.py --func train_unet --data-dir retraining_analysis/data-retraining-analysis/run_3/unet/unet-300 --batchnorm --val-split 0.1 --epochs 1000 --batch-size 4 --model-outfile retraining_analysis/run_3/trained_models/unet-300.hdf5 > retraining_analysis/run_3/training_logs/unet-300.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_unet.py --func train_unet --data-dir retraining_analysis/data-retraining-analysis/run_3/unet/unet-400 --batchnorm --val-split 0.1 --epochs 1000 --batch-size 4 --model-outfile retraining_analysis/run_3/trained_models/unet-400.hdf5 > retraining_analysis/run_3/training_logs/unet-400.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_unet.py --func train_unet --data-dir retraining_analysis/data-retraining-analysis/run_3/unet/unet-500 --batchnorm --val-split 0.1 --epochs 1000 --batch-size 4 --model-outfile retraining_analysis/run_3/trained_models/unet-500.hdf5 > retraining_analysis/run_3/training_logs/unet-500.log 2>&1

# VGG training - RCP without CNP.
CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_3/vgg/vgg-100_RCP-0_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_3/trained_models/vgg-100_RCP-0_CNP.hdf5 > retraining_analysis/run_3/training_logs/vgg-100_RCP-0_CNP.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_3/vgg/vgg-200_RCP-0_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_3/trained_models/vgg-200_RCP-0_CNP.hdf5 > retraining_analysis/run_3/training_logs/vgg-200_RCP-0_CNP.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_3/vgg/vgg-300_RCP-0_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_3/trained_models/vgg-300_RCP-0_CNP.hdf5 > retraining_analysis/run_3/training_logs/vgg-300_RCP-0_CNP.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_3/vgg/vgg-400_RCP-0_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_3/trained_models/vgg-400_RCP-0_CNP.hdf5 > retraining_analysis/run_3/training_logs/vgg-400_RCP-0_CNP.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_3/vgg/vgg-500_RCP-0_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_3/trained_models/vgg-500_RCP-0_CNP.hdf5 > retraining_analysis/run_3/training_logs/vgg-500_RCP-0_CNP.log 2>&1

# VGG training - RCP with CNP.
CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_3/vgg/vgg-500_RCP-100_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_3/trained_models/vgg-500_RCP-100_CNP.hdf5 > retraining_analysis/run_3/training_logs/vgg-500_RCP-100_CNP.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_3/vgg/vgg-500_RCP-200_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_3/trained_models/vgg-500_RCP-200_CNP.hdf5 > retraining_analysis/run_3/training_logs/vgg-500_RCP-200_CNP.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_3/vgg/vgg-500_RCP-300_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_3/trained_models/vgg-500_RCP-300_CNP.hdf5 > retraining_analysis/run_3/training_logs/vgg-500_RCP-300_CNP.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_3/vgg/vgg-500_RCP-400_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_3/trained_models/vgg-500_RCP-400_CNP.hdf5 > retraining_analysis/run_3/training_logs/vgg-500_RCP-400_CNP.log 2>&1

CUDA_VISIBLE_DEVICES=0 python code/train_pipeline_cnn.py --func train_cnn --data-dir retraining_analysis/data-retraining-analysis/run_3/vgg/vgg-500_RCP-500_CNP --classes "amphipod,nonamphipod" --val-split 0.1 --epochs 300 --batch-size 8 --model-outfile retraining_analysis/run_3/trained_models/vgg-500_RCP-500_CNP.hdf5 > retraining_analysis/run_3/training_logs/vgg-500_RCP-500_CNP.log 2>&1


### Running analysis on test bed.
python /home/bhuvan/Projects/CamHD_object_identification/retraining_analysis/run_and_evaluate_configurations.py run_3 > retraining_analysis/run_3/training_logs/analysis-master.log 2>&1


# Checking trained models: update the run_<number> accordingly.
# Checking val_mean_iou for train unet models:
tail -1 retraining_analysis/run_3/training_logs/unet-100.log

tail -1 retraining_analysis/run_3/training_logs/unet-200.log

tail -1 retraining_analysis/run_3/training_logs/unet-300.log

tail -1 retraining_analysis/run_3/training_logs/unet-400.log

tail -1 retraining_analysis/run_3/training_logs/unet-500.log

# Checking val_acc for vgg models:

tail -1 retraining_analysis/run_3/training_logs/vgg-100_RCP-0_CNP.log

tail -1 retraining_analysis/run_3/training_logs/vgg-200_RCP-0_CNP.log

tail -1 retraining_analysis/run_3/training_logs/vgg-300_RCP-0_CNP.log

tail -1 retraining_analysis/run_3/training_logs/vgg-400_RCP-0_CNP.log

tail -1 retraining_analysis/run_3/training_logs/vgg-500_RCP-0_CNP.log

tail -1 retraining_analysis/run_3/training_logs/vgg-500_RCP-100_CNP.log

tail -1 retraining_analysis/run_3/training_logs/vgg-500_RCP-200_CNP.log

tail -1 retraining_analysis/run_3/training_logs/vgg-500_RCP-300_CNP.log

tail -1 retraining_analysis/run_3/training_logs/vgg-500_RCP-400_CNP.log

tail -1 retraining_analysis/run_3/training_logs/vgg-500_RCP-500_CNP.log


# Collate the results in the google sheets: https://docs.google.com/spreadsheets/d/1pDnlNZXFtmPMJ12V6ZRcO62kTKDfmNye_-iV0I7XuM8/edit?usp=sharing.
