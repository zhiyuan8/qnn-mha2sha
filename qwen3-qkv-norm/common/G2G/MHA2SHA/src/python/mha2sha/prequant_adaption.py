# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from onnx.onnx_pb import ModelProto, NodeProto, TensorProto
from onnx import helper
import json
import os, copy


from mha2sha.utils.clean import clean_model, topological_sort

from mha2sha.utils.onnx import (
    get_initializer_mappings,
    get_node_by_input_name,
    get_node_by_output_name,
    get_node_mappings,
    get_value_info_proto_mapping,
)

from mha2sha.utils.logger import log_debug, log_error, log_info, log_warning
from mha2sha.utils.utils import update_all_mapping_dicts

PREQUANT_ENCODING_MAP_PATH="prequant_encodings_map.json"



class PreQuantAdaption:
    """
    PreQuantAdaption class is responsible for making pre-quant model adaptions up to mha2sha.
    """
    def __init__(
        self,
        model: ModelProto,
        lora_model: bool,
        lora_alpha_from_input: bool,
    ):
        """
        Initialization

        :param model: ModelProto
        :param lora_model: Is lora model
        :param lora_alpha_from_input: Is lora alpha from model input.
        :param mha_conv: is mha use conv instead of matmul in projection
        """

        self.model = model
        self.lora_model = lora_model
        self.lora_alpha_from_input = lora_alpha_from_input
        self.encodings_map = {
            "activation_encodings": {},
            "param_encodings": {}
        }
        self._update_all_mapping_dicts()  # Initial mapping

    def _update_all_mapping_dicts(self):
        """Helper function to update mappings to nodes.

        Updates all the mapping dictionaries such as `get_initializer_by_name`. These need
        to be updated as nodes are added to the graph and are not yet know.
        """
        update_all_mapping_dicts(self)

    def optimize(self, mha_encodings, enc_map_save_dir):
        if self.lora_model:
            from mha2sha.transformations.lora_model_adaption import LoraModelAdpation
            log_info("Running model adaption for --lora-alpha-from-input outside mha.")

            # Apply lora model adaptions
            self._update_all_mapping_dicts()
            self.lora_model_adaption = LoraModelAdpation(self)
            # Set lora_adaptor_set_info_dict and add_lora_alpha_from_input (optional)
            self.lora_model_adaption.run_lora_model_adaption()

            # Add lora model adaption objects to PreQuantAdaption self instance.
            self.lora_alpha_value_list = self.lora_model_adaption.lora_alpha_value_list
            self.lora_adaptor_set_info_dict = self.lora_model_adaption.lora_adaptor_set_info_dict
            self.encodings_map = self.lora_model_adaption.encodings_map

            if mha_encodings is not None:
                mha_encodings = self.lora_model_adaption.transform_encodings(
                                    mha_encodings
                                )

        self.model = topological_sort(clean_model(self.model))
        self._update_all_mapping_dicts()

        self.save_encodings_map(enc_map_save_dir)

        return self.model, mha_encodings

    def save_encodings_map(self, dir_path):
        with open(os.path.join(dir_path, PREQUANT_ENCODING_MAP_PATH), "w") as f:
            json.dump(self.encodings_map, f, indent=4)
