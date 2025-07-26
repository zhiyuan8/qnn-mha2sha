# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

r"""Runner module for converting MHAs to SHAs.

This module is the high level entry point into the conversion of a model with MHAs into SHAs. From here, the users
provided flags into the program are parsed and used to load the model, convert the model, propagate encodings, and save
the new model/encodings. View the README.md for more information.

Basic usage
-----------

>>> converter = MHA2SHAConverter(
        model_name="llama2",
        sha_export_path="./exports",
        model_or_path="./llama2.onnx",
        **kwargs  # Where kwargs are flags for the converter to parse
    )
>>> converter.convert()
"""

from collections import OrderedDict
import datetime
import json
import os
from packaging import version
from pathlib import Path
from safetensors.numpy import save_file
import shutil
import sys
import time
from typing import Dict, List, Optional, Text, Tuple, Union

import numpy as np
import onnx
from onnx.onnx_pb import ModelProto

from mha2sha.defs.print_colors import Colors
from mha2sha.defs.tensor_info import TensorInfo, get_input_info, get_output_info
from mha2sha.transformations.o_proj_optimizer import OProjOptimzier
from mha2sha.prequant_adaption import PreQuantAdaption, PREQUANT_ENCODING_MAP_PATH
from mha2sha.utils import onnx as ou
from mha2sha.utils.arg_parser import (
    filter_args,
    pretty_print_args,
)
from mha2sha.utils.auto_mha_finder import auto_attention_finder
from mha2sha.utils.attention_patterns import attention_patterns
from mha2sha.utils.logger import (
    log_assert,
    log_debug,
    log_error,
    log_info,
    log_warning,
    setup_logging,
)
from mha2sha.utils.lora_adaptor_converter import (
    LoraAdaptor,
    parse_lora_adaptor_list,
    save_init_to_safetensor,
    LORA_SHA_SAFETENSOR_PATH
)
from mha2sha.optimizer import MHA2SHAOptimizer
from mha2sha.encoding_mapper import MHA_TO_SHA_NAME_MAPPING_PATH
from mha2sha.optimizer_extension.lora_extension import LoraVersion

_ONNX_MIN_VERSION_NEEDED = "1.14.1"

ALL_STAGES_NAME_MAPPING_PATH = "all_stages_encodings_mapping.json"
SHA_LORA_TENSOR_NAME_PATH = "sha_lora_tensor_names.txt"

class MHA2SHAConverter:
    """Converts models with MHA to SHA.

    Runner class for converting models from MHA to SHA. Additional conversions can be added based on the flags passed in.

    Attributes:
        model:
            ModelProto object.
        model_path:
            Model path form the model loader.
        model_name:
            Name of model to be export, assign a name base on model path if not provided.
        sha_export_path:
            Path for exporting the converted model.
        exported_model_encoding_path:
            Path of the encodings from the original model.
        prepared_model:
            Model name to generat auto generate one if nor peovided.
        handle_rope_ops:
            Flag for whether this model has RoPE operations to be handled.
        handle_past_key_value:
            Flag for whether this model has past key values to be handled.
        replace_linear_with_conv:
            Flag for replacing Linear operations with Convs.
        disable_auto_attn_finder:
            Flag to turn off the auto attention module finder feature.
        position_ids:
            Flag for if position ids are in the model.
        mha_conv:
            Flag for if mha is in Conv.
        nchw_aligned:
            Flag for if mha input is aligned to nchw format.
        lora_model:
            Flag for if mha has lora adaptor.
        lora_adaptor_list:
            Path to list of lora adaptor safetensor and encodings.
        create_input_lists:
            Flag for if to create input_list.
        no_verification:
            Flag to skip verification step.
        log_level:
            log level: 'warn', 'verbose', 'info', 'error', 'fatal.
        strict:
            Whether to strictly enforce golden RoPE pattern.
        build_ar:
            AR value to produce model of.
        ar_num:
            Manually specified AR num.
    """
    def __init__(
        self,
        model_name: str,
        sha_export_path: str,
        model_or_path: Union[ModelProto, str],
        log_level: str = "info",
        **passed_in_args
    ) -> None:
        """Creates a converter instance.

        Instance creation dependent on the flags passed in
        via the 'passed_in_args' positional args.
        """

        MHA2SHAConverter._verify_can_run()

        setup_logging(log_level)

        self._model_name = model_name
        self._sha_export_path = Path(sha_export_path)
        self._start_time = time.time()

        if isinstance(model_or_path, ModelProto):
            log_info("Step 1: Loaded model\n")
            self._model = ou.assign_names_to_empty_nodes(model_or_path)
            if not self._model_name:
                self._model_name = "mha2sha_model"
                log_info(f"'model_name' not passed in. Setting to {self._model_name}")
        else:
            log_info("Step 1: Loading model\n")
            self._model, self._model_path = ou.load_model(model_or_path)
            self._mha_model_path = model_or_path

        # add log level back into passed args for pretty print
        self._set_args(**passed_in_args | {"log_level": log_level})

    @staticmethod
    def _verify_can_run():
        """Verifies that the script meets requirements to run.

        List of Checks:
            - Checks for ONNX Minimum version is >=_ONNX_MIN_VERSION_NEEDED (see converter.py)
            - Checks for Python >=3.10
        """
        user_onnx_version = version.parse(onnx.version.version)
        minimum_onnx_version = version.parse(_ONNX_MIN_VERSION_NEEDED)
        cond_to_msg = [
            (
                user_onnx_version >= minimum_onnx_version,
                f"ONNX version must be >={_ONNX_MIN_VERSION_NEEDED}, got version: {onnx.version.version}",
            ),
            (
                sys.version_info >= (3, 10),
                f"Python version must be >= 3.10, got version: {sys.version}",
            ),
        ]
        for cond, msg in cond_to_msg:
            log_assert(cond=cond, msg=msg)

        if user_onnx_version == minimum_onnx_version:
            log_warning(
                f"Got ONNX version: {user_onnx_version}. This version of ONNX will be deprecated in future releases. "
                "Please migrate to ONNX 1.16.1"
            )

    def _set_args(self, **passed_in_args):
        filtered_args = filter_args(**passed_in_args)
        for arg_name, arg_value in filtered_args.items():
            setattr(self, f"_{arg_name}", arg_value)

        pretty_print_args(filtered_args, self._model.ByteSize())

    def convert(self) -> Tuple[ModelProto, Optional[bool]]:
        r"""Entry call for conversion to take place.

        Entry to start conversion of the MHA model. By default, logging will show each major step of the process.

        Returns:
            Converted MHA2SHA model and an optional verification status of the model.
            Verification status is only returned if --no-verification is set to False.
        """

        self._exported_model_encoding_path = Path(self._exported_model_encoding_path) if self._exported_model_encoding_path else None

        # Condition means we have a ModelProto that was passed in
        # and we need to save the model for running on ONNXRuntime.
        if not hasattr(self, "_model_path") and not self._no_verification:
            log_debug("ModelProto passed in, saving to verify MHA2SHA")
            self._model_path = (self._sha_export_path / "mha_export"  / "mha_model_proto.onnx").as_posix()
            ou.qairt_save_model(self._model, self._model_path)  # QAIRT save due to external data

        # Sanity check
        if self._lora_model:
            log_assert(self._mha_conv, f"lora model only support mha-conv models.")

        if self._lora_alpha_from_input:
            log_assert(self._lora_model, f"lora_alpha_from_input expects lora model = True, but got {self._lora_model}")

        if self._lora_adaptor_list_path is not None:
            log_assert(self._lora_model, f"lora_adaptor_list expects lora model = True, but got {self._lora_model}")
            self.lora_adaptor_list = parse_lora_adaptor_list(self._lora_adaptor_list_path)
        else:
            self.lora_adaptor_list = None

        if self._mha_conv and not self._nchw_aligned:
            log_warning("Got not nchw aligned mha model.")

        if self._exported_model_encoding_path is not None:
            assert os.path.exists(self._exported_model_encoding_path), f"Encoding file {self._exported_model_encoding_path} not exists."

        if self._build_ar and not self._no_verification:
            log_warning(
                "--build-ar flag used. Currently there is no verification of MHA vs SHA."
            )
        if self._optimize_o_proj:
            log_assert(self._mha_conv, "--optimize_o_proj only supports mha-conv model.")


        import onnxruntime
        onnxruntime.SessionOptions.use_deterministic_compute = True
        np.random.seed(42)

        # Check the correctness of model object
        log_info("-" * 20)
        if self._no_verification:
            log_info(
                ("Step 2: `--no-verification` set. "
                 "Skipping checking the correctness of `model` object\n")
            )
        else:
            log_info("Step 2: Checking the correctness of `model` object\n")
            self._check_original_model()

        # Generate model inputs and outputs
        log_info("-" * 20)
        if self._no_verification:
            log_info("Step 3: `--no-verification` set. Skipping Generating model inputs and outputs.")
        else:
            log_info("Step 3: Generating model inputs and outputs\n")
            np_inputs, output_names, golden_outputs = (
                self._generate_golden_inputs_outputs()
            )

        # Run pattern matcher
        log_info("-" * 20)
        log_info("Step 4: Running auto pattern matcher on model object.\n")
        pattern, pattern_start_node_names, pattern_end_node_names = self._pattern_match()

        # Apply mha2sha optimization
        log_info("-" * 20)
        log_info("Step 5: Applying MHA2SHA optimization on model object\n")
        mha2sha_model, onnx_output_filename = self._run_optimizations(
            pattern, pattern_start_node_names, pattern_end_node_names
        )

        # Validate mha2sha model by running it on ONNXRT
        log_info("-" * 20)
        verification_status = None  # None to start for if there is no verification
        if self._no_verification:
            log_info(
                ("Step 6: `--no-verification` set. "
                 "Skipping comparing MHA2SHA model to Original by running on ONNXRT\n")
            )
        else:
            log_info("Step 6: Comparing MHA2SHA model to Original by running on ONNXRT\n")
            verification_status = self._compare_goldens_to_converted(
                onnx_output_filename, np_inputs, output_names, golden_outputs
            )

        runtime_seconds =  time.time() - self._start_time
        log_info(f"Total Runtime ----- {str(datetime.timedelta(seconds=runtime_seconds)).split('.')[0]} -----")
        if verification_status is not None:
            verification_str = (
                f"{Colors.OKGREEN if verification_status else Colors.FAIL}"
                f"{'OK' if verification_status else 'FAIL'}{Colors.ENDC}"
            )
            log_info(f"Verification Status ----- {verification_str} -----")
            if not verification_status:
                log_info("To see a detailed list of comparisions, rerun with '--log-level debug'")

        return mha2sha_model, verification_status

    def _check_original_model(self) -> None:
        """Checks the original model with QAIRT `native_checker`.

        Will log and error and exit the program if an error is found during checking.
        """

        if not ou.native_checker(self._model):
            log_error("Model-Checker failed after model export. Exiting...")
            exit()

    def _save_inputs_goldens_and_input_lists(
        self,
        np_inputs: Dict[str, np.ndarray],
        golden_outputs: List[np.ndarray]
    ) -> None:
        """Creates two input lists for Linux/On device. Saves inputs and goldens.

        Saves the random initialized numpy inputs and the goldens produced from the original model to the directory
        specified by the `self.sha_export_path` flag. The inputs are save under `inputs_from_mha2sha` where each raw
        file has the input tensors name. The golden outputs are save under `goldens_from_mha2sha` and are enumerated.

        Additionally a input_list.txt is created for Linux and on device.

        Linux input_list.txt is saved as `self.sha_export_path`/input_list.txt.
        On Device input_list.txt is saved as `self.sha_export_path`/on_device_input_list.txt.
          Internally, the files are under /data/local/tmp/mha2sha.

        Args:
            np_inputs:
                NumPY inputs to save.
            golden_outputs:
                Golden outputs to save.
        """

        log_info("Saving inputs and goldens. Creating input lists")

        input_path = (self._sha_export_path / "mha_input_vectors/").absolute()
        input_list_txt_path = (self._sha_export_path / "mha_input_vectors/input_list.txt").absolute()
        input_shape_list_txt_path = (self._sha_export_path / "mha_input_vectors/input_shape_list.txt").absolute()
        on_device_input_list_txt_path = (self._sha_export_path / "on_device_input_list.txt").absolute()
        goldens_path = (self._sha_export_path / "golden_output_from_mha/").absolute()
        on_device_mha2sha_path = Path("/data/local/tmp/mha2sha")

        # Clear dir and set them up
        for path in [input_path, goldens_path]:
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True)

        for path in [input_list_txt_path, input_shape_list_txt_path, on_device_input_list_txt_path]:
            if path.exists():
                path.unlink()
            path.touch()

        for input_name, data in np_inputs.items():
            input_name = input_name.replace("/", "_")
            child = f"{input_name}.raw"
            path = Path(input_path) / child
            on_device_path = on_device_mha2sha_path / child

            # Writes path for Linux machine to raw files
            with open(input_list_txt_path, "a") as f:
                f.write(f"{path.as_posix()} ")

            # Writes path for on device raw files
            with open(on_device_input_list_txt_path, "a") as f:
                f.write(f"{on_device_path.as_posix()} ")

            # Writes the actual raw data
            with open(path, "wb") as f:
                data.astype(np.float32).tofile(f)

        # Writes input shape for input lists.
        with open(input_shape_list_txt_path, "a") as f:
            for tensor in np_inputs.values():
                f.write(f"{list(tensor.shape)} ")

        log_debug(f"Successfully saved inputs to model at: {input_path}")
        log_debug(f"Successfully saved input_list.txt to model at: {input_list_txt_path}")
        log_debug(f"Successfully saved on_device_input_list.txt to model at: {on_device_input_list_txt_path}")

        for idx, data in enumerate(golden_outputs):
            path = Path(goldens_path) / f"golden_output_{idx}.raw"
            with open(path, "wb") as f:
                data.tofile(f)

        log_debug(f"Successfully saved golden outputs of model at: {goldens_path}")

    def _get_models_inputs_output_names(
        self,
    ) -> Tuple[Dict[str, np.ndarray], List[str]]:
        r""""""

        # TODO Generate LVM and LLM input based on self.handle_past_key_value
        try:
            inp_specs = get_input_info(self._model)
            out_specs = get_output_info(self._model)
            if not self._handle_past_key_value:
                inputs = self._generate_random_test_data(inp_specs)
            else:
                inputs = self._generate_llama_test_data(inp_specs)

            output_names = []
            for key in out_specs.keys():
                output_names.append(key)
        except Exception as e:
            log_error(f"Generation of inputs and outputs failed due to: {e}")
            exit()

        return inputs, output_names

    def _generate_golden_inputs_outputs(
        self,
    ) -> Tuple[Dict[str, np.ndarray], List[Text], List[np.ndarray]]:
        """Generates random inputs and golden outputs of the original model.

        Creates random numpy inputs that then run through the model to produce golden outputs to compare against the
        converted model later on. Additionally, the output tensor names are also captured for direct comparison.

        Returns:
            Tensor names to randomly generated numpy inputs, names of the output tensors, and the golden inputs produced.
        """

        np_inputs, output_names = self._get_models_inputs_output_names()

        try:
            _, golden_outputs = ou.run_model_on_ort(
                self._model_path, np_inputs, output_names
            )
        except Exception as e:
            log_error(f"Model run on ONNXRT failed due to: {e}")
            exit()

        if self._create_input_lists:
            _, on_device_goldens = ou.run_model_on_ort(
                self._model_path,
                np_inputs,
                [name for name in output_names if "past" not in name]  # We don't care about past value/key on device
            )
            self._save_inputs_goldens_and_input_lists(np_inputs, on_device_goldens)

        return np_inputs, output_names, golden_outputs


    def _pattern_match(self) -> Tuple[List[str], List[str], List[str]]:
        """Finds all MHA's within the original model.

        Will either use the Auto Finder for finding MHAs or fallback to the explicitly listed pattern types if no
        patterns are found by Auto Finder or if Auto Finder is disabled.
        """

        pattern = None
        if not self._disable_auto_attn_finder:
            use_quick_auto_finder = (self._lora_model or self._gqa_model)
            pattern, pattern_start_node_names, pattern_end_node_names = auto_attention_finder(
                self._model, self._mha_conv, use_quick_auto_finder
            )

        if pattern is None:
            log_info("Step 4.2: Cannot find pattern with auto pattern finder, search again using pre-defined patterns.\n")
            pattern, pattern_start_node_names, pattern_end_node_names = ou.get_pattern_start_end_nodes(
                self._model, attention_patterns
            )

        log_info("Running pattern matcher - pattern matched:")
        log_info(" ".join([str(elem) for elem in pattern]))
        log_info(f"found_matched_pattern: {len(pattern_start_node_names)}")


        return pattern, pattern_start_node_names, pattern_end_node_names

    def _merge_encodings_mappings(self, mapping1, mapping2):
        '''
        mapping1: a->[b1,b2]
        mapping2: b1->[c1,c2,c3]
                    b2->[c4]
        out_mapping: a->[c1,c2,c3,c4]
        '''
        new_mapping = {}
        for enc_type in ("activation_encodings", "param_encodings"):
            reversed_mapping1 = {}
            new_mapping[enc_type] = {}
            for key, map1_names in mapping1[enc_type].items():
                for map1_n in map1_names:
                    if map1_n not in reversed_mapping1:
                        reversed_mapping1[map1_n] = []
                    if key not in reversed_mapping1[map1_n]:
                        reversed_mapping1[map1_n].append(key)

            for key, map2_names in mapping2[enc_type].items():
                # map2_names may not be unique
                map2_names = list(dict.fromkeys(map2_names)) # elements are unique and order is kept
                if key in reversed_mapping1:
                    # key is a mapping1 value
                    origin_keys = reversed_mapping1[key]
                else:
                    # key is not a mapping1 value
                    origin_keys = [key]

                for origin_k in origin_keys:
                    if origin_k not in new_mapping[enc_type]:
                        new_mapping[enc_type][origin_k] = []
                    new_mapping[enc_type][origin_k] += map2_names

            # handle the case when value of mapping1 is not in mapping2
            for map1_name, origin_names in reversed_mapping1.items():
                if map1_name not in mapping2[enc_type]:
                    for origin_k in origin_names:
                        if origin_k not in new_mapping[enc_type]:
                            new_mapping[enc_type][origin_k] = []
                        new_mapping[enc_type][origin_k].append(map1_name)

        return new_mapping

    def _merge_encodings_mapping_files(self, mapping1_file, mapping2_file, out_mappint_file):
        with open(mapping1_file, "r") as f:
            mapping1 = json.load(f)
        with open(mapping2_file, "r") as f:
            mapping2 = json.load(f)
        out_mapping = self._merge_encodings_mappings(mapping1, mapping2)
        with open(out_mappint_file, "w") as f:
            json.dump(out_mapping, f, indent=4)

    def _run_optimizations(
        self,
        pattern: List[str],
        pattern_start_node_names: List[str],
        pattern_end_node_names: List[str]
    ) -> Tuple[ModelProto, Path]:
        """Performs MHA2SHA optimizations and saves a new model.

        Runs the optimizations for each pattern that matches an MHA. Additional optimization such as Linear to Conv
        may also be done based on the inputed argument.

        Args:
            pattern:
                List of Op Type's in pattern.
            pattern_start_node_names
                Starting node names for each pattern.
            pattern_end_node_names:
                Ending node names for each pattern.

        Returns:
           ModelProto optimized and path to the saved converted model.
        """

        self._sha_export_path.mkdir(parents=True, exist_ok=True)

        mha_encodings = None
        if self._exported_model_encoding_path:
            with open(self._exported_model_encoding_path, "r") as f:
                mha_encodings = json.load(f)

        log_info("Step 5.1: Apply prequant model adaption.\n")
        prequant_opt = PreQuantAdaption(
            self._model,
            self._lora_model,
            self._lora_alpha_from_input,
        )
        self._model, mha_encodings = prequant_opt.optimize(mha_encodings,
                                                           self._sha_export_path)

        if self._lora_alpha_from_input:
            self._lora_alpha_value_list = prequant_opt.lora_alpha_value_list # For fp eval.

        if self._lora_model:
            self._lora_adaptor_set_info_dict = prequant_opt.lora_adaptor_set_info_dict
        else:
            # Dummy _lora_adaptor_set_info_dict when not lora model
            self._lora_adaptor_set_info_dict = {"0": None}

        if self._skip_mha2sha:
            log_info("Step 5.2: Skip mha2sha model adaption. Doesn't support LoRA v1 models.\n")
            mha2sha_model = self._model
            out_model_init_dict = {value.name:value for value in self._model.graph.initializer}
        else:
            log_info("Step 5.2: Apply mha2sha model adaption.\n")
            if self._ar_num:
                log_info(f"Skip auto detect ar num and use --ar-num = {self._ar_num}.\n")
            mha_opt = MHA2SHAOptimizer(
                self._model,
                pattern,
                pattern_start_node_names,
                pattern_end_node_names,
                self._handle_rope_ops,
                self._handle_past_key_value,
                self._prepared_model,
                self._replace_linear_with_conv,
                self._position_ids,
                self._mha_conv,
                self._nchw_aligned,
                self._handle_r3_matrix,
                self._strict,
                self._lora_model,
                self._lora_alpha_from_input,
                self._llm_model,
                self._gqa_model,
                self._build_ar,
                self._handle_alibi,
                self._handle_internal_rmsnorm,
                self._lora_adaptor_set_info_dict,
                self._ar_num
            )
            mha2sha_model = mha_opt.optimize()
            out_model_init_dict = mha_opt.get_initializer_by_name
            self.lora_version = mha_opt.lora_version

            if self._mha_conv and self._optimize_o_proj:
                log_info("Optimize head concat to o_proj pattern for mha-conv models...")
                o_proj_opt = OProjOptimzier(
                                mha2sha_model,
                                mha_opt.qkv_head_concat_node_list,
                            )
                mha2sha_model = o_proj_opt.optimize()

        log_info("Saving lora weights to safetensor...")
        # Filter on lora_tensor_name_to_init_dict: We don't want "lora_alpha_pads" and "lora_alpha_pad_value"
        lora_tensor_name_to_init_dict = {}
        for name, value in out_model_init_dict.items():
            if "lora" in name and name not in ("lora_alpha_pads", "lora_alpha_pad_value"):
                lora_tensor_name_to_init_dict[name] = value

        # Add gather indices to safetensor
        # Look for all the gathers with lora_alpha (model input) -> Pad -> Gather pattern.
        sha_model_input_names = [tensor.name for tensor in mha2sha_model.graph.input]
        if "lora_alpha" in sha_model_input_names:
            _pad_node = mha_opt.get_node_by_input_name["lora_alpha"][0]
            if _pad_node.op_type == "Pad":
                _gather_node_list = mha_opt.get_node_by_input_name[_pad_node.output[0]]
                for _gather_node in _gather_node_list:
                    assert _gather_node.op_type == "Gather", f"Expect Pad -> Gather pattern but for lora v3, but got\
{_pad_node.name}: {_pad_node.op_type} -> {_gather_node.name}: {_gather_node.op_type}"
                    _indice_name = _gather_node.input[1]
                    lora_tensor_name_to_init_dict[_indice_name] = out_model_init_dict[_indice_name]

        if lora_tensor_name_to_init_dict:
            save_init_to_safetensor(lora_tensor_name_to_init_dict, self._sha_export_path)

        if self._model_name is None:
            output_model_name = self._model_path.split(".")[-2].split('/')[-1]
            if self._replace_linear_with_conv or self._mha_conv:
                output_model_name += "_conv"
            if not self._skip_mha2sha and mha_opt.kv_cache:
                output_model_name += "_kvcache"
            output_model_name += "_sha"
        else:
            output_model_name = self._model_name

        log_info("-" * 20)
        if self._skip_mha2sha:
            log_info("Step 5.3: Skip create sha encodings.\n")
            if mha_encodings is not None:
                # lora alpha extraction may change encodings, so we need to save new encodings
                export_encoding_path = self._sha_export_path / f"{output_model_name}.encodings"
                with open(export_encoding_path, 'w') as f:
                    json.dump(mha_encodings, f, indent=4)
                log_info(f"MHA encodings saved at {export_encoding_path}")
                shutil.copy(self._sha_export_path / PREQUANT_ENCODING_MAP_PATH,
                            self._sha_export_path / ALL_STAGES_NAME_MAPPING_PATH)
        else:
            log_info("Step 5.3: Create sha encodings.\n")
            if mha_encodings is not None:
                sha_encoding = mha_opt.mha_to_sha_encoding_mapping(
                    mha_encodings,
                    self._sha_export_path,
                )
                sha_encoding_path = self._sha_export_path / f"{output_model_name}.encodings"
                with open(sha_encoding_path, 'w') as f:
                    json.dump(sha_encoding, f, indent=4)
                log_info(f"SHA encodings saved at {sha_encoding_path}")

                # merge prequant and mha2sha encodings mappings
                self._merge_encodings_mapping_files(
                    self._sha_export_path / PREQUANT_ENCODING_MAP_PATH,
                    self._sha_export_path / MHA_TO_SHA_NAME_MAPPING_PATH,
                    self._sha_export_path / ALL_STAGES_NAME_MAPPING_PATH
                )
            else:
                log_warning("exported_model_encoding_path is None. Skiping create SHA encoding.")

        log_info("Step 5.4: Saving converted model...\n")
        onnx_output_filename = self._sha_export_path / f"{output_model_name}.onnx"

        ou.save_model(mha2sha_model, onnx_output_filename, save_as_external_data=True)
        log_info(f"MHA2SHA model saved at: {onnx_output_filename}")

        # At hoc for lora. Will be deprecated.
        if self.lora_adaptor_list is not None:
            log_info("Step 5.5: Convert and save lora adaptor encodings and safetensors...\n")
            self._convert_lora_adaptors(
                output_model_name,
                lora_tensor_name_to_init_dict,
                sha_encoding_path)

        if self._mha_lora_tensor_path:
            log_info("Step 5.6: Map MHA lora tensor names to SHA lora tensor names...\n")
            self._map_mha_lora_updatable_tensor(
                mha2sha_model
            )

        return mha2sha_model, onnx_output_filename

    def _convert_lora_adaptors(
            self,
            output_model_name,
            lora_tensor_name_to_init_dict,
            sha_encoding_path
        ):
        mha_to_sha_encodings_names = self._sha_export_path / ALL_STAGES_NAME_MAPPING_PATH
        with open(mha_to_sha_encodings_names, "r") as f:
            mha_to_sha_encodings_names = json.load(f)

        with open(sha_encoding_path, "r") as f:
            sha_encoding = json.load(f)

        LoraAdaptor.mha_to_sha_encodings_names = mha_to_sha_encodings_names
        LoraAdaptor.mha_conv = self._mha_conv
        LoraAdaptor.base_sha_lora_keys = lora_tensor_name_to_init_dict.keys()
        LoraAdaptor.base_sha_encoding = sha_encoding

        for lora_adaptor in self.lora_adaptor_list:
            lora_sha_safetensor_dict, lora_sha_encodings = lora_adaptor.map_encoding_and_slice_safetensor()

            # Save lora adaptor encodings
            lora_adaptor_encodings_path = self._sha_export_path/ f"{lora_adaptor.name}_{output_model_name}.encodings"
            with open(lora_adaptor_encodings_path, 'w') as f:
                json.dump(lora_sha_encodings, f, indent=4)
            log_info(f"Lora adaptor {lora_adaptor.name} encodings saved at {lora_adaptor_encodings_path.absolute()}")

            # Save lora adaptor safetensor
            lora_adaptor_safetensor_path = self._sha_export_path/ f"{lora_adaptor.name}_{LORA_SHA_SAFETENSOR_PATH}"
            save_file(lora_sha_safetensor_dict, lora_adaptor_safetensor_path)
            log_info(f"Lora adaptor {lora_adaptor.name} safetensor saved at {lora_adaptor_safetensor_path.absolute()}")

    def _map_mha_lora_updatable_tensor(self, sha_model):
        """
        Use all_stage_name_mapping_path to map mha lora tensor name to sha lora tensor name and write
        updated sha lora tensor name to sha_lora_updatable_tensor_path.
        """
        with open(self._sha_export_path / ALL_STAGES_NAME_MAPPING_PATH, "r") as f:
            mapping_dict = json.load(f)
        mixed_mapping_dict = mapping_dict["activation_encodings"] | mapping_dict["param_encodings"]

        mha_lora_tensors = []
        with open(self._mha_lora_tensor_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                mha_lora_tensors.append(line.strip())

        # Get node name and tesnor name in the model.
        sha_node_tensor_name_set = set()
        for node in sha_model.graph.node:
            for _nodename_input_output in (node.name, node.input, node.output):
                for _name in _nodename_input_output:
                    if _name not in sha_node_tensor_name_set:
                        sha_node_tensor_name_set.add(_name)

        sha_lora_tensors = []
        for _mha_tensor in mha_lora_tensors:
            if _mha_tensor in mixed_mapping_dict.keys():
                # Update mha to sha tensor
                _sha_tesnor = mixed_mapping_dict[_mha_tensor]
                if isinstance(_sha_tesnor, list):
                    sha_lora_tensors.extend(_sha_tesnor)
                else:
                    sha_lora_tensors.append(_sha_tesnor)
            else:
                # Check _mha_tensor in sha model. Append only MHA tensor that also exists in SHA model.
                if _mha_tensor in sha_node_tensor_name_set:
                    sha_lora_tensors.append(_mha_tensor)
                else:
                    log_debug(f"MHA updatable tensor: {_mha_tensor} not found in SHA model.")

        with open(self._sha_export_path / SHA_LORA_TENSOR_NAME_PATH, "w") as f:
            for _tensor_name in sha_lora_tensors:
                f.write(_tensor_name+"\n")

        log_info(f"SHA Lora tensor names txt saved at {(self._sha_export_path / SHA_LORA_TENSOR_NAME_PATH).absolute()}")

    def _compare_goldens_to_converted(
        self,
        onnx_output_filename: Path,
        np_inputs: Dict[str, np.ndarray],
        output_names: List[Text],
        golden_outputs: List[np.ndarray]
    ) -> bool:
        """Compares the outputs of the original model and the converted on.

        Using the randomly generated numpy inputs, the outputs of the original model and the converted model are compared
        and there MAD is logged out.

        Args:
            onnx_output_filename:
                The path of where the converted model was saved.
            np_inputs:
                Randomly generated numpy inputs.
            output_names:
                Output names of the original model.
            golden_outputs:
                Outputs of the origina model.
        Returns:
            Verification status of the comparision.
        """

        log_info(f"Load and run SHA model from {onnx_output_filename}")

        try:
            log_info("Running SHA model with ORT")

            if self._build_ar:
                log_warning("Running ORT with new inputs due to '--build-ar' flag.")

            if self._skip_mha2sha or (self._lora_alpha_from_input and self.lora_version in [LoraVersion.V2, LoraVersion.V3]):
                _np_inputs, _ = self._get_models_inputs_output_names()
                for _, _info_dict in self._lora_adaptor_set_info_dict.items():
                    model_input_name = _info_dict.lora_alpha_model_input
                    lora_alpha_value = _info_dict.lora_alpha

                    _shape = lora_alpha_value.shape
                    _dtype = _np_inputs[model_input_name].dtype

                    # scalar value returns shape as empty tuple.
                    np_inputs[model_input_name] = np.ones(_shape if _shape else 1, dtype=_dtype) * lora_alpha_value

            if self._handle_past_key_value:
                log_debug("Permuting inputs for past key/value inputs")
                np_inputs = {
                    k: t.transpose(1, 0, 2, 3) if "past" in k else t
                    for k, t in np_inputs.items()
                }

            _, converted_model_outputs = ou.run_model_on_ort(
                onnx_output_filename.as_posix(), np_inputs, output_names
            )

        except Exception as e:
            log_error(f"Model run on ONNXRT failed due to: {e}")
            exit()

        if self._create_input_lists:
            test_vector_dir = self._sha_export_path / "sha_test_vectors/"
            os.makedirs(test_vector_dir, exist_ok=True)

            with open(os.path.join(test_vector_dir, 'input_list.txt'), 'w') as f:
                for name, tensor in np_inputs.items():
                    name = name.replace("/", "_")
                    filename = f'{name}.raw'
                    tensor.astype(np.float32).tofile(os.path.join(test_vector_dir, filename))

                    f.write(f"{name}:={(test_vector_dir/filename).absolute()} ")

            with open(os.path.join(test_vector_dir, 'input_shape_list.txt'), 'w') as f:
                for tensor in np_inputs.values():
                    f.write(f"{list(tensor.shape)} ")

            log_info(f"Write sha input list and test vector to: {test_vector_dir.absolute()}")

        if self._build_ar:
            log_warning(
                "'--build-ar' flag used, skipping comparison of MHA and SHA logits."
            )
            return

        # Validating the outputs of original v/s mha2sha model.
        output_verification_status = [False] * len(golden_outputs)

        for i in range(len(converted_model_outputs)):
            _golden_outputs = golden_outputs[i]
            _converted_model_outputs = converted_model_outputs[i]
            log_debug(f"{output_names[i]=} {_golden_outputs.shape=} {_converted_model_outputs.shape=}")

            if i>0:
                # only run tensor shape alignment on 4d tensor.
                if len(golden_outputs[i].shape) == 4:
                    # align transpose_key_cache
                    if golden_outputs[i].shape[-2] != _converted_model_outputs.shape[-2]:
                        _converted_model_outputs = np.transpose(_converted_model_outputs, [0, 1, 3, 2])

                    # align concat_past_key_value_to_batch
                    if golden_outputs[i].shape[0] != _converted_model_outputs.shape[0]:
                        _converted_model_outputs = np.transpose(_converted_model_outputs, [1, 0, 2, 3])

            output_verification_status[i] = np.allclose(
                _golden_outputs,
                _converted_model_outputs,
                atol=5e-4
            )


            log_debug(f"For {output_names[i]} : MAD = {str(np.abs(_golden_outputs - _converted_model_outputs).max())}\n")

        return all(output_verification_status)

    # Helper functions
    def _generate_llama_test_data(self, input_info_dict: Dict[Text, TensorInfo]) -> Dict[str, np.ndarray]:
        """Generate the test inputs based on given shape and data type (LLaMA).

        Args:
            input_info_dict:
                A dict with mapping from input name to another dict having info regarding input shape and input dtype.

        Returns:
        A dict with mapping from input name to test data of the input in np.array format.
        """

        final_inputs = OrderedDict()
        input_info_dict_list = list(input_info_dict.items())
        input_id_name, input_id_tensor = input_info_dict_list[0]
        final_inputs[input_id_name] = np.random.randint(1, 500, input_id_tensor.shape).astype(input_id_tensor.dtype)
        for input_name, tensor in input_info_dict_list[1:]:
            input_shape = tensor.shape
            input_dtype = tensor.dtype

            if len(tensor.shape) == 0:
                final_inputs[input_name] = np.random.rand(1,).astype(input_dtype)
            else:
                input_shape = tensor.shape
                final_inputs[input_name] = np.random.rand(*input_shape).astype(input_dtype)


        return final_inputs


    def _generate_random_test_data(self, input_info_dict: Dict[Text, TensorInfo]) -> Dict[str, np.ndarray]:
        """Generate the test inputs based on given shape and data type (Regular).

        Args:
            input_info_dict:
                A dict with mapping from input name to another dict having info regarding input shape and input dtype.

        Returns:
        A dict with mapping from input name to test data of the input in np.array format.
        """

        final_inputs = OrderedDict()
        for input_name, tensor in input_info_dict.items():
            input_shape = tensor.shape
            input_dtype = tensor.dtype
            final_inputs[input_name] = np.random.rand(*input_shape).astype(input_dtype)
        return final_inputs
