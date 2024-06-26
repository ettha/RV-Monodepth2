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

    #model_name = ["--model_name", "mono+stereo_640x192"]
    #model_name = ["--model_name", "finetuned_mono+stereo_640x192_epoch_9"]
    predict_depth = ["--pred_metric_depth"]

    # -------------------------------------------------------------------------

    sys.argv = placeholder + image_path + model_name  + search_for #+ predict_depth
    args = parse_args() 
    test_simple(args)

    # -------------------------------------------------------------------------

    image_dir_files = os.listdir(image_dir)
        
        # Filter out directories and non-png files, keep only .png files
    image_paths = [f for f in image_dir_files 
                    if os.path.isfile(os.path.join(image_dir, f))
                    and f.lower().endswith('.png')]
    
    # -------------------------------------------------------------------------

    average_rmse = 0 

    for i in range(len(image_paths)): 

        image_pattern = f'_sync_image_'
        groundtruth_pattern = f'_sync_groundtruth_depth_'

        image_path = image_paths[i]
        file_name, file_extension = os.path.splitext(image_path)

        gt_image_dir = os.path.join("data_ass3", "Task1_3", "groundtruth")
        gt_image_file = image_path.replace(image_pattern, groundtruth_pattern)

        gt_image = cv2.imread(os.path.join(gt_image_dir, gt_image_file), cv2.IMREAD_UNCHANGED)

        if gt_image is None:
            print("Error: Could not load image from")
            print(os.path.join(gt_image_dir, gt_image_file))
            exit()

        disp_npy_fname = os.path.join(image_dir, "{}_depth.npy".format(file_name)) 
        disp_pred = np.load(disp_npy_fname)

        if file_name == "2011_10_03_drive_0047_sync_image_0000000005_image_03":
            eval_fname = "2011_10_03_drive_0047_sync_image_0000000005_image_03"
            disp_img_fname = os.path.join(image_dir, "{}_disp.jpeg".format(eval_fname)) 
            shutil.copy(disp_img_fname, os.path.join("fileoutput", "{}_disp.jpeg".format(eval_fname))) 

        if not disp_pred.any():
            print("Error: Could not load image from")
            print(disp_npy_fname) 
            exit()

        # Postprocessing

        gt_height, gt_width = gt_image.shape[:2]

        #scaled_depth_pred = STEREO_SCALE_FACTOR * np.squeeze(disp_pred)
        scaled_depth_pred = np.squeeze(disp_pred)
        scaled_depth_pred = (cv2.resize(scaled_depth_pred, (gt_width, gt_height)))

        scaled_gt = ((gt_image / 256))

        mask = scaled_gt > 0
        scaled_gt = scaled_gt[mask]
        scaled_depth_pred = scaled_depth_pred[mask]

        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80

        scaled_depth_pred[scaled_depth_pred < MIN_DEPTH] = MIN_DEPTH
        scaled_depth_pred[scaled_depth_pred > MAX_DEPTH] = MAX_DEPTH

        # 

        # Debug
        # print(np.max(scaled_gt))    
        # print(np.min(scaled_gt))    
        # print(np.max(depth_pred))    
        # print(np.min(depth_pred))    

        # print(np.sum(diff))
        # cv2.imshow("Prediction", scaled_depth_pred)
        # cv2.waitKey(0)
        
        # cv2.imshow("Ground Truth", scaled_gt)
        # cv2.waitKey(0)

        # cv2.imshow("Difference", np.abs(scaled_depth_pred - scaled_gt))
        # cv2.waitKey(0)
        #

        rmse = (scaled_gt - scaled_depth_pred) ** 2
        rmse = np.sqrt(rmse.mean())

        print(rmse)
        average_rmse += rmse


    average_rmse /= len(image_paths)
    print(average_rmse)