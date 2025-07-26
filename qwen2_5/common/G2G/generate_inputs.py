# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import argparse
import os
import shutil

import numpy as np


# Load model inputs
def load_inputs(src, model, model_dir, input_file):
    src_inputs = src[0]['inputs']
    with open(input_file, "w+") as f:
        for key in src_inputs:
            prefix = f"{key}:="
            entry_name = f"{model_dir}/{model}_{key}_input.raw"
            if len(src_inputs[key].shape) == 4:
                src_inputs[key].transpose(0,2,3,1).astype(np.float32).tofile(entry_name)
            else:
                src_inputs[key].astype(np.float32).tofile(entry_name)
            # write to input_list.txt
            f.write(prefix + entry_name + " ")

def input_list_add_lora_alpha(lora_alpha_file, model_dir, input_file):
    # Copy lora_alpha input file to inpus_list dir
    lora_dst_path = os.path.join(model_dir, 'lora_alpha.raw')
    shutil.copy(lora_alpha_file, lora_dst_path)

    with open(input_file, 'a+') as f:
        f.write("lora_alpha:=" + lora_dst_path + " ")

def generate_inputs(model_name, pickle_path, working_dir, lora_alpha_file=""):
    # Create inputs_list directory
    os.makedirs(working_dir, exist_ok=True)

    # create output dir for model
    model_dir = os.path.join(working_dir, f"{model_name}_input_vectors")
    os.makedirs(model_dir, exist_ok=True)
    input_file = os.path.join(model_dir, f"{model_name}_input_list.txt")

    # load sd lora model numpy array object
    input_data = np.load(pickle_path, allow_pickle=True)

    print(f"Load {model_name} inputs data")
    load_inputs(input_data, model_name, model_dir, input_file)

    # lora alpha input file path provided, then add to input_list file
    if lora_alpha_file:
        input_list_add_lora_alpha(lora_alpha_file, model_dir, input_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process provided numpy pickel to create inputs to submodel of stable diffusion with lora model.")
    parser.add_argument("-m", "--model_name", help="Name of model input list is being generated for.", required=True)
    parser.add_argument("-i", "--model_pickle_path", help="PATH to numpy array object pickle generated in AIMET work-flow for single model", required=True)
    parser.add_argument("-wd", "--working_dir", help="PATH for creating qnn_assets directory", default="./inputs_list")
    parser.add_argument("-l", "--lora_alpha_file", help="PATH to lora_alpha input file", default="")

    # parse aruments
    args = parser.parse_args()
    model_name = args.model_name
    pickle_path = args.model_pickle_path
    working_dir = args.working_dir
    lora_input = args.lora_alpha_file

    generate_inputs(model_name, pickle_path, working_dir, lora_input)
