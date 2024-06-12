from test_simple import parse_args
from test_simple import test_simple 

import numpy as np
import cv2
import sys
import os

import shutil

STEREO_SCALE_FACTOR = 5.4

if __name__ == '__main__':

    image_dir = os.path.join("data_ass3", "Task1_3", "images")

    placeholder = ["assign3_task1.py"]

    # Comment out when predicting single files
    image_path = ["--image_path", image_dir]
    search_for = ["--ext", "png"]
    #
    # Use th following for single file prediction
    #image_file = "2011_10_03_drive_0047_sync_image_0000000005_image_03.png"
    #image_path = ["--image_path", os.path.join(image_dir, image_file)]
    # 

    model_name = ["--model_name", "mono+stereo_640x192"]
    #predict_depth = ["--pred_metric_depth"]

    # -------------------------------------------------------------------------

    sys.argv = placeholder + image_path + model_name  + search_for #+ predict_depth
    args = parse_args() 
    test_simple(args)

    # -------------------------------------------------------------------------

    eval_fname = "2011_10_03_drive_0047_sync_image_0000000005_image_03"
    disp_img_fname = os.path.join(image_dir, "{}_disp.jpeg".format(eval_fname)) 
    shutil.copy(disp_img_fname, os.path.join("fileoutput", "{}_disp.jpeg".format(eval_fname))) 