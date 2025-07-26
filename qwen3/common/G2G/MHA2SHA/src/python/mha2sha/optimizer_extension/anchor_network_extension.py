# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from onnx import helper
from onnx.onnx_pb import NodeProto, TensorProto

from mha2sha.utils.logger import log_debug, log_error, log_info, log_warning

from mha2sha.utils.onnx import (
    get_next_node_up_based_on_cond,
)
from mha2sha.utils.encoding_mapper_utils import (
    create_activation_node_mapping_dict,
    update_sha_tensor_to_node_mapping_dict
)

mha2sha_hf_model_optimizer = Any  # Causes circular import

ANCHOR_ACTIVATION_ENCODING_KEYS = ["anchor_matmul"] # "anchor_mul" and "anchor_add" doesn't require encoding mapping

@dataclass
class AnchorNode:
    """ Defines nodes to record for anchor network models. """
    anchor_matmul: Optional[Union[NodeProto, List[NodeProto]]] = None
    anchor_mul: Optional[Union[NodeProto, List[NodeProto]]] = None
    anchor_add: Optional[Union[NodeProto, List[NodeProto]]] = None

    def __str__(self):
        # useful for printing and debuging
        return f"AnchorNode(\n" \
               f"    anchor_matmul: '{self.anchor_matmul.name if self.anchor_matmul else None}',\n" \
               f"    anchor_mul: '{self.anchor_mul.name if self.anchor_mul else None}',\n" \
               f"    anchor_add: '{self.anchor_add.name if self.anchor_add else None}',\n" \
               f")"

    def __repr__(self):
        return str(self)

def create_anchor_network_encoding_mapping_dict(
        encoding_mapping_dict,
        mha_anchor_node
    ):
    """ create Dict[str, NodeMappingDict] for anchor nodes. """
    encoding_mapping_dict.anchor_network = {
        f"anchor_matmul": create_activation_node_mapping_dict(
                mha_anchor_node.anchor_matmul.input[0] if mha_anchor_node.anchor_matmul is not None else None,
                mha_anchor_node.anchor_matmul.input[1] if mha_anchor_node.anchor_matmul is not None else None,
                mha_anchor_node.anchor_matmul.output[0] if mha_anchor_node.anchor_matmul is not None else None
            ),
        f"anchor_mul": create_activation_node_mapping_dict(
                mha_anchor_node.anchor_mul.input[0] if mha_anchor_node.anchor_mul is not None else None,
                mha_anchor_node.anchor_mul.input[1] if mha_anchor_node.anchor_mul is not None else None,
                mha_anchor_node.anchor_mul.output[0] if mha_anchor_node.anchor_mul is not None else None
            ),
        f"anchor_add": create_activation_node_mapping_dict(
                mha_anchor_node.anchor_add.input[0] if mha_anchor_node.anchor_add is not None else None,
                mha_anchor_node.anchor_add.input[1] if mha_anchor_node.anchor_add is not None else None,
                mha_anchor_node.anchor_add.output[0] if mha_anchor_node.anchor_add is not None else None
            ),
        }

def update_anchor_sha_encoding_name_to_anchor_encoding_mapping_dict(
        anchor_encoding_mapping_dict,
        anchor_sha_node,
    ):
    """
    Update sha Anchor network encoding names to EncodingMappingDict.
    :param anchor_encoding_mapping_dict: encoding_mapping_dict.anchor_network: Dict[str, NodeMappingDict]
    :param sha_anchor_node: A name dict for sha nodes created when optimizer split and create nodes for sha
    """
    for anchor_node_name in ANCHOR_ACTIVATION_ENCODING_KEYS:
        update_sha_tensor_to_node_mapping_dict(
            node_mapping_dict = anchor_encoding_mapping_dict[anchor_node_name],
            sha_node_list = getattr(anchor_sha_node, anchor_node_name)
        )


class AnchorExtension:
    """ Extenstion helpers for mha2sha_optimzer to work on Anchor Network.  """
    def __init__(self, mha2sha_optim: mha2sha_hf_model_optimizer) -> None:
        """Initalizes an instance based on the MHA2SHA optimizer provided.

        Args:
            mha2sha_optim:
                MHA2SHAOptimizer instance holding the model loader and model info.
        """
        self.mha2sha_optim = mha2sha_optim

        # valid_token_mask_model_input: [1, 1, 1, seq_len]
        valid_token_mask_model_input = mha2sha_optim.model.graph.input[mha2sha_optim.mha_model_input_names_index_dict["valid_token_mask"]]
        self.valid_token_mask_name = valid_token_mask_model_input.name

        self.reset_anchor_node()

        anchor_matmul_list = self.mha2sha_optim.get_node_by_input_name["valid_token_mask"]
        if anchor_matmul_list[0].input[0] == "valid_token_mask":
            k_proj_input_idx = 1 # row 13
        else:
            k_proj_input_idx = 0 # row 12

        if self.mha2sha_optim.get_node_by_output_name[anchor_matmul_list[0].input[k_proj_input_idx]].op_type == "Transpose":
            self.transpose_k_proj = True
        else:
            self.transpose_k_proj = False

        mha_k_proj_node_list = [
            get_next_node_up_based_on_cond(
                self.mha2sha_optim.get_node_by_output_name[matmul_node.input[k_proj_input_idx]],
                self.mha2sha_optim.get_node_by_output_name,
                node_found_cond=lambda n: n.op_type == "Conv"
            ) for matmul_node in anchor_matmul_list
        ]
        self.mha_k_proj_name_matmal_dict = {k_proj.name: matmul.name for k_proj, matmul in zip(mha_k_proj_node_list, anchor_matmul_list) }

    def reset_anchor_node(self):
        self.mha_node = AnchorNode()
        self.sha_node = AnchorNode()

    def attach_anchor_network(self, info_dict, key_inp_list):
        """
        anchor_network: (valid_token_mask @ key) + (anchor_buffer_in * scale) = anchor_buffer_out
        :param key_inp_list: list of key_inp [1, 1, head_dim, seq_len]

        self.valid_token_mask_name
        valid_token_mask [1, 1, 1, seq_len]
        key_inp [1, 1, head_dim, seq_len]
        anchor_buffer_in [1, head_num, 1, head_dim]
        """
        self.reset_anchor_node()

        mha_conv_node = info_dict["key"]["matmul_node"]
        matmul_name = self.mha_k_proj_name_matmal_dict[mha_conv_node.name]
        matmul_node = self.mha2sha_optim.get_node_by_node_name[matmul_name]
        # Get anchor buffer
        mha_add_node = self.mha2sha_optim.get_node_by_input_name[matmul_node.output[0]][0]
        anchor_buffer_out_name = mha_add_node.output[0]
        anchor_buffer_in_node = self.mha2sha_optim.get_node_by_output_name[mha_add_node.input[1]] # [1, head_num, 1, head_dim]

        # Recored AnchorNode for MHA
        self.mha_node = AnchorNode(
                            anchor_matmul=matmul_node,
                            anchor_add=mha_add_node,
                            anchor_mul=None
                        )

        # Init sha node
        self.sha_node = AnchorNode(
                            anchor_matmul=[],
                            anchor_add=[],
                            anchor_mul=[]
                        )
        anchor_buffer_out_list = [] # List of
        for _, key_inp in enumerate(key_inp_list):
            key_inp = self.mha2sha_optim._op_factory.get_transpose_op(
                key_inp, [0, 1, 3, 2]
            ) # [1, 1, head_dim, seq_len] -> [1, 1, seq_len, head_dim]

            anchor_matmul_node = self.mha2sha_optim._op_factory.get_matmul_op(
                self.valid_token_mask_name,
                key_inp,
                propose_op_name = "AnchorMatMul"
            ) # [1, 1, 1, head_dim]

            self.sha_node.anchor_matmul.append(anchor_matmul_node)
            self.mha2sha_optim.model.graph.node.extend([key_inp, anchor_matmul_node])
            anchor_buffer_out_list.append(anchor_matmul_node)

        concat_buffer_out_node = self.mha2sha_optim._op_factory.get_concat_op(
            anchor_buffer_out_list, 1
        )  # Concate on [1, 1, head_dim, 1] -> [1, head_num, head_dim, 1]

        self.mha2sha_optim.model.graph.node.append(concat_buffer_out_node)

        # Remove exist anchor_buffer_out tensor from model
        for i, _output_tensor in enumerate(self.mha2sha_optim.model.graph.output):
            if _output_tensor.name == anchor_buffer_out_name:
                anchor_buffer_out_index = i
                break
        anchor_buffer_out = self.mha2sha_optim.model.graph.output[anchor_buffer_out_index]
        anchor_buffer_out_shape = [anchor_buffer_out.type.tensor_type.shape.dim[i].dim_value for i in range(4)]
        del self.mha2sha_optim.model.graph.output[anchor_buffer_out_index]

        # Create new model out tensor with same name
        anchor_buffer_out_tensor = helper.make_tensor_value_info(
            anchor_buffer_out_name, TensorProto.FLOAT, anchor_buffer_out_shape
        )
        self.mha2sha_optim.model.graph.output.append(anchor_buffer_out_tensor)

        anchor_buffer_add_node = self.mha2sha_optim._op_factory.get_add_op(
            concat_buffer_out_node, anchor_buffer_in_node, anchor_buffer_out_name
        )
        self.mha2sha_optim.model.graph.node.append(anchor_buffer_add_node)

    def create_sha_conv_lora_rope(self,
                                  info_dict,
                                  ns,
                                  head_num,
                                  head_dim,
                                  query_matmul_inp,
                                  key_matmul_inp,
                                  value_matmul_inp,
                                  sha_encoding_name_dict,
                                  **extenstion_kwargs):
        return self.mha2sha_optim._mha_conv_extension.create_sha_conv_with_rope(
                    info_dict,
                    ns,
                    head_num,
                    head_dim,
                    query_matmul_inp,
                    key_matmul_inp,
                    value_matmul_inp,
                    sha_encoding_name_dict,
                    **extenstion_kwargs
                )

    def create_sha_conv_lora(self,
                            info_dict,
                            ns,
                            head_num,
                            head_dim,
                            query_matmul_inp,
                            key_matmul_inp,
                            value_matmul_inp,
                            sha_encoding_name_dict,
                            **extension_kwargs):
        return self.mha2sha_optim.create_sha(
                    info_dict,
                    ns,
                    head_num,
                    head_dim,
                    query_matmul_inp,
                    key_matmul_inp,
                    value_matmul_inp,
                    sha_encoding_name_dict,
                    **extension_kwargs)
