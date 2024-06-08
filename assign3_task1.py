from test_simple import parse_args
from test_simple import test_simple 

import numpy as np
import cv2
import sys
import os

STEREO_SCALE_FACTOR = 5.4

if __name__ == '__main__':

    image_dir = os.path.join("data_ass3", "Task1_3", "images")
    image_file = "2011_10_03_drive_0047_sync_image_0000000005_image_03.png"

    placeholder = ["assign3_task1.py"]
    image_path = ["--image_path", os.path.join(image_dir, image_file)]
    #image_path = ["--image_path", image_dir]
    model_name = ["--model_name", "mono+stereo_640x192"]
    search_for = ["--ext", "png"]

    # -------------------------------------------------------------------------

    sys.argv = placeholder + image_path + model_name #+ search_for
    args = parse_args() 
    test_simple(args)

    # -------------------------------------------------------------------------

    gt_image_dir = os.path.join("data_ass3", "Task1_3", "groundtruth")
    gt_image_file = "2011_10_03_drive_0047_sync_groundtruth_depth_0000000005_image_03.png"

    gt_image = cv2.imread(os.path.join(gt_image_dir, gt_image_file))

    if gt_image is None:
        print("Error: Could not load image from")
        print(os.path.join(gt_image_dir, image_file))
        exit()

    file_name, file_extension = os.path.splitext(image_file)
    file_name_wo_extension = os.path.basename(file_name)
    depth_output_fname = os.path.join(image_dir, "{}_disp.jpeg".format(file_name_wo_extension)) 

    depth_image = cv2.imread(depth_output_fname)

    if depth_image is None:
        print("Error: Could not load image from")
        print(depth_output_fname)
        exit()

    # -------------------------------------------------------------------------

    gt_image = gt_image.astype(float)

    mask = gt_image > 0
    gt_image[mask] = 1 / gt_image[mask]

    print(gt_image.shape)
    #print(gt_image)
    print(np.max(gt_image))
    print(np.min(gt_image[mask]))

    # -------------------------------------------------------------------------

    depth_image = depth_image.astype(float)
    mask = depth_image > 0
    depth_image[mask] = 1 / (depth_image[mask] * STEREO_SCALE_FACTOR)

    print(depth_image.shape)
    #print(depth_image)
    print(np.max(depth_image))
    print(np.min(depth_image[mask]))

    # -------------------------------------------------------------------------

    errors = np.abs(gt_image - depth_image)
    mask = errors > 0
    errors[mask] = (80 * errors[mask])

    errors = np.round(errors)
    print(errors)
    # print(result_image.shape)
    # cv2.imshow(result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()