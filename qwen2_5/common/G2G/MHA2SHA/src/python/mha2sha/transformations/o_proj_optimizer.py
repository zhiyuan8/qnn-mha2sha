# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from mha2sha.utils.clean import clean_model, topological_sort
from mha2sha.utils.onnx import (get_next_node_up_based_on_cond,
                                get_next_node_down_based_on_cond,
                                get_constant_node_value,
                                get_initializer_value)
from mha2sha.utils.op_factory import OpFactory
from mha2sha.utils.utils import update_all_mapping_dicts

class OProjOptimzier:
    """ Post mha2sha model adaption to clean up transpoes reshape in o_proj. """
    def __init__(self, sha_model, qkv_head_concat_node_list) -> None:
        """Initalizes an instance based on the MHA2SHA optimizer provided.

        Args:
            mha2sha_optim:
                MHA2SHAOptimizer instance holding the model loader and model info.
            qkv_head_concat_node:
                list of qkv_head_concat_node in created in sha.
        """
        self.model = sha_model
        self.qkv_head_concat_node_list = qkv_head_concat_node_list
        self._update_all_mapping_dicts()
        self._op_factory = OpFactory(
            self.tensor_name_set,
            self.model,
            self.node_name_mapping_dict,
            self.mha_model_input_names_index_dict,
            None
        )

    def _update_all_mapping_dicts(self):
        """Helper function to update mappings to nodes.

        Updates all the mapping dictionaries such as `get_initializer_by_name`. These need
        to be updated as nodes are added to the graph and are not yet know.
        """
        update_all_mapping_dicts(self)

    def get_lora_convs(self, split_nodes):
        o_proj_conv = []
        for node in split_nodes:
            _conv_node = get_next_node_down_based_on_cond(
                            node,
                            self.get_node_by_input_name,
                            node_found_cond=lambda n: n.op_type == "Conv"
                        )
            o_proj_conv.append(_conv_node)
        return o_proj_conv

    def _cleanup_o_proj(self, qkv_concat):
        _has_lora = False
        o_proj_conv = get_next_node_down_based_on_cond(
                        qkv_concat,
                        self.get_node_by_input_name,
                        node_found_cond=lambda n: n.op_type == "Conv"
                     )
        node = qkv_concat
        while node != o_proj_conv:
            node_outputs = self.get_node_by_input_name[node.output[0]]
            if len(node_outputs) == 2:
                _has_lora = True
                break
            node = node_outputs[0]

        if _has_lora:
            o_proj_conv = self.get_lora_convs(node_outputs)
        else:
            o_proj_conv = [o_proj_conv]

        # qkv_concat output shape: [B, 1, H*W, C] -> Transpose -> [B, C, 1, H*W]
        transpose_node = self._op_factory.get_transpose_op(qkv_concat, [0, 3, 1, 2])
        self.model.graph.node.append(transpose_node)

        for node in o_proj_conv:
            node.input[0] = transpose_node.output[0]

    def optimize(self):
        for concat_node in self.qkv_head_concat_node_list:
            self._cleanup_o_proj(concat_node)

        self.model = topological_sort(clean_model(self.model))
        self._update_all_mapping_dicts()
        return self.model
