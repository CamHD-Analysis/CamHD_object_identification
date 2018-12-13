### Data Directory
This directory contains large image files hence not committed to GitHub currently.
Once the data is ready, it would be hosted on a remote drive and link will be shared here. <br/><br/>

_Note: The frames are extracted from the videos at zoom Z1 (scenes: d5A_p1_z1, d5A_p5_z1, d5A_p6_z1, d5A_p0_z1, d5A_p7_z1, d5A_p8_z1).
Each frame is 1920x1080 pixels. The coordinates in the patch names are with respect these dimensions._

##### File Format Specification
_patch name format_: <src_frame_name_<object_type>_X_Y.png
(Where (X, Y) is the centroid of the patch, and the patch size is defined for each object type) <br/><br/>

_sample src_frame_name (VideoName_Frame)_: "CAMHDA301-20180711T001500_1274.png" (this is a frame retrieved from LazyCache) <br/><br/>
_file extension_:* "png" <br/><br/>

- Object Types:
    - amphipode (patch size: 128x128 (_TODO: finalize the patch size_))

##### Directory Structure:
- raw (downloaded src_frames from LazyCache with the above mentioned src_frame_name format) <br/><br/>
- annotations (json files containing the manually segmented annotations using 'labelme' on the raw frames) <br/><br/>
- amphipod _(Similar structure for labels: amphipod, star, crab etc.)_
    - amphipod_segmentation
        - train
            - patches
            - masks

    - amphipod_classification (class_labels: "amphipod", "nonamphipod")
        - train
            - patches
            - annotation_file ("patch_name,class_label")

    - testbed (randomly sampled frames from several videos across days)
        - testbed_src_frame_names
        - analysis_output _(debug_mode)_
            - raw_stitched_mask
            - postprocessed_mask
            - marked_frame (src frame on which the predicted bounding boxes are drawn)
            - classification_output ("patch_name,class_label")
            - report.json
        - analysis_report (consolidation of report.json files from each of the test frame images.)

##### Report format (report.json):
###### Frame level object format (frame_report):
```json
{
    "frame": "CAMHDA301-20180711T001500_1274.png",
    "scene_tag": "d5A_p1_z1",
    "frame_res": (1920, 1080),
    "counts": {
        "amphipod": 4,
        "crab": 2,
        "other": 2
    },
    "location_sizes": {
        "amphipod": {
            (X_wrt_frame, Y_wrt_frame): num_pixels,
            (324, 1452): 65,
            ...
        }
        "crab": {
            (x, y): pixels,
            ...
        }
        "other": {
            (x, y): pixels,
            ...
        }
    }
}
```
###### Video level object format (frame_report):
```json
{
    "video": "CAMHDA301-20180711T001500",
    "deployment": "d5A",
    "date": "20180711",
    "time": "00:15",
    "frames": [
        frame_report_1,
        frame_report_2,
        ...
    ]
}
```
