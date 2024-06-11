# Part of this code is taken from the following repository:
# https://github.com/nianticlabs/monodepth2/blob/master/train.py
# Last accessed: June 10, 2024

# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions

import sys
import os


#file_dir = os.path.dirname(__file__) # the directory that this file resides in
                           
def set_cmd_parameters(): 

    placeholder = ["assign3_task1.py"]

    # PATHS
    data_path = ["--data_path", "kitti_data"]
    log_path = ["--log_dir", "train_log"]

    # TRAINING options
    save_model = ["--model_name", "finetuned_mono_stereo"] # the name of the folder to save the model in
    split = ["--split", "small"]
    dataset = ["--dataset", "small"]
    png = ["--png"]

    # OPTIMIZATION options
    epochs = ["--num_epochs", "10"]

    # LOADING options
    load_model = ["--load_weights_folder", "mono+stereo_640x192"]

    # EVALUATION options
    evaluate_mode = ["--eval_stereo"]
    evaluate_split = ["--eval_split", "small"]

    # -------------------------------------------------------------------------

    sys.argv = placeholder + data_path + log_path + save_model + split + dataset + png + epochs + load_model + evaluate_mode + evaluate_split
    
    # -------------------------------------------------------------------------

if __name__ == "__main__":

    set_cmd_parameters()

    options = MonodepthOptions()
    opts = options.parse()

    trainer = Trainer(opts)
    trainer.train()
