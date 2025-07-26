# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" Utils to map base attention nodes encodings. """
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from onnx.onnx_pb import NodeProto, TensorProto

from mha2sha.utils.encoding_mapper_utils import (NodeMappingDict,
                                                 create_activation_node_mapping_dict,
                                                 update_sha_tensor_to_node_mapping_dict,
                                                 update_multi_input_concat_sha_tensor_to_node_mapping_dict)

ACTIVATION_ENCODING_KEYS = [
    "q_proj_activation",
    "k_proj_activation",
    "v_proj_activation",
    "qk_matmul",
    "qk_scale",
    "qk_mask_add",
    "qk_alibi_add",
    "qk_softmax",
    "qkv_matmul",
    "qkv_head_concat",
]
PARAM_ENCODING_KEYS = ["q_proj_param",
                       "k_proj_param",
                       "v_proj_param"]


@dataclass
class BaseAttnNode:
    """ Store base atten NodeProtos """
    qk_matmul:  Optional[Union[NodeProto, List[NodeProto]]]
    qk_scale:  Optional[Union[NodeProto, List[NodeProto]]]
    qk_softmax:  Optional[Union[NodeProto, List[NodeProto]]]
    qk_mask_add:  Optional[Union[NodeProto, List[NodeProto]]]
    qk_alibi_add:  Optional[Union[NodeProto, List[NodeProto]]]
    qkv_matmul: Union[NodeProto, List[NodeProto]]
    qkv_head_concat: Union[NodeProto, List[NodeProto]]
    q_matmul: Union[NodeProto, List[NodeProto]]
    k_matmul: Union[NodeProto, List[NodeProto]]
    v_matmul: Union[NodeProto, List[NodeProto]]

def create_base_attn_mapping_dict(mha_base_attn_node)->Dict[str, NodeMappingDict]:
    """
    Create NodeMappingDict for each node in base attention.
    :param mha_base_attn_node: BaseAttnNode for mha.
    :return tensor_name to NodeMappingDict dict
    """
    return {
        "qk_matmul": create_activation_node_mapping_dict(
                mha_base_attn_node.qk_matmul.input[0],
                mha_base_attn_node.qk_matmul.input[1],
                mha_base_attn_node.qk_matmul.output[0]
            ),
        "qk_scale": create_activation_node_mapping_dict(
            input_1_name=None,
            input_2_name=mha_base_attn_node.qk_scale.input[1] if mha_base_attn_node.qk_scale is not None else None,
            output_name=mha_base_attn_node.qk_scale.output[0] if mha_base_attn_node.qk_scale is not None else None
        ),
        "qk_mask_add": create_activation_node_mapping_dict(
            input_1_name=None,
            input_2_name=None,
            output_name=mha_base_attn_node.qk_mask_add.output[0] if mha_base_attn_node.qk_mask_add is not None else None
        ),
        "qk_alibi_add": create_activation_node_mapping_dict(
            input_1_name=None,
            input_2_name=mha_base_attn_node.qk_alibi_add.input[1] if mha_base_attn_node.qk_alibi_add is not None else None,
            output_name=mha_base_attn_node.qk_alibi_add.output[0] if mha_base_attn_node.qk_alibi_add is not None else None
        ),
        "qk_softmax": create_activation_node_mapping_dict(
            input_1_name=None,
            input_2_name=None,
            output_name=mha_base_attn_node.qk_softmax.output[0]
        ),
        "qkv_matmul": create_activation_node_mapping_dict(
            input_1_name=mha_base_attn_node.qkv_matmul.input[0],
            input_2_name=mha_base_attn_node.qkv_matmul.input[1],
            output_name=mha_base_attn_node.qkv_matmul.output[0]
        ),
        "qkv_head_concat": create_activation_node_mapping_dict(
            input_1_name=mha_base_attn_node.qkv_matmul.output[0],
            output_name=mha_base_attn_node.qkv_matmul.output[0]
        ),
        "q_proj_activation": create_activation_node_mapping_dict(
            input_1_name = None,
            input_2_name = None,
            output_name = mha_base_attn_node.q_matmul.output[0]
        ),
        "k_proj_activation": create_activation_node_mapping_dict(
            input_1_name = None,
            input_2_name = None,
            output_name = mha_base_attn_node.k_matmul.output[0]
        ),
        "v_proj_activation": create_activation_node_mapping_dict(
            input_1_name = None,
            input_2_name = None,
            output_name = mha_base_attn_node.v_matmul.output[0]
        ),
        "q_proj_param": NodeMappingDict(
            mha_mapping_name_list = ["mha_param_name"],
            sha_mapping_name_list = ["sha_param_name"],
            mapping_name_dict = {
                "mha_param_name": mha_base_attn_node.q_matmul.input[1],
                "sha_param_name": None
            }
        ),
        "k_proj_param": NodeMappingDict(
            mha_mapping_name_list = ["mha_param_name"],
            sha_mapping_name_list = ["sha_param_name"],
            mapping_name_dict = {
                "mha_param_name": mha_base_attn_node.k_matmul.input[1],
                "sha_param_name": None
            }
        ),
        "v_proj_param": NodeMappingDict(
            mha_mapping_name_list = ["mha_param_name"],
            sha_mapping_name_list = ["sha_param_name"],
            mapping_name_dict = {
                "mha_param_name": mha_base_attn_node.v_matmul.input[1],
                "sha_param_name": None
            }
        ),
    }


def update_base_attn_sha_encoding_name_to_base_attn_encoding_mapping_dict(base_attn_mapping_dict, sha_base_attn_node_list):
    """
    Update get_base_sha_encoding_name_dict collected from each sha creation to EncodingMappingDict
    """
    base_mapping_dict_keys_node_attr_pair = [
        ("q_proj_activation", "q_matmul"),
        ("k_proj_activation", "k_matmul"),
        ("v_proj_activation", "v_matmul"),
        ("qk_matmul", "qk_matmul"),
        ("qk_scale", "qk_scale"),
        ("qk_mask_add", "qk_mask_add"),
        ("qk_alibi_add", "qk_alibi_add"),
        ("qk_softmax", "qk_softmax"),
        ("qkv_matmul", "qkv_matmul"),
        ("q_proj_param", "q_matmul"),
        ("k_proj_param", "k_matmul"),
        ("v_proj_param", "v_matmul"),
        ("qkv_head_concat", "qkv_head_concat")
    ]

    for _key, _attr in base_mapping_dict_keys_node_attr_pair:
        if _key == "qkv_head_concat":
            update_multi_input_concat_sha_tensor_to_node_mapping_dict(
                base_attn_mapping_dict[_key],
                getattr(sha_base_attn_node_list, _attr)[0]
            )
        else:
            update_sha_tensor_to_node_mapping_dict(
                base_attn_mapping_dict[_key],
                getattr(sha_base_attn_node_list, _attr)
            )
