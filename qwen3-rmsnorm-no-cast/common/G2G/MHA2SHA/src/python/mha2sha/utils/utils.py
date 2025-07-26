# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" Helping functions for mha2sha optimizer """
from enum import Enum

from mha2sha.utils.onnx import (
    get_initializer_mappings,
    get_node_by_input_name,
    get_node_by_output_name,
    get_node_mappings,
    get_value_info_proto_mapping,
)

class BranchType(Enum):
    Q = 1
    K = 2
    V = 3

class ExtensionTypes(str, Enum):
    BASE_ATTN = 'base_attn'
    GQA = 'gqa'
    LORA = 'lora'
    PAST_KEY_VALUE = 'past_key_value'
    ROPE = 'rope'
    RMSNORM = 'rmsnorm'
    ANCHOR_NETWORK = 'anchor_network'

def sha_node_name_basis(attn_num, head_num):
    """
    attn_num: attention number in pattern.
    head_num: head number in mha.
    """
    propose_sha_name = f"attn_{attn_num}_head_{head_num}"
    propose_sha_query_name = propose_sha_name+"_query"
    propose_sha_key_name = propose_sha_name+"_key"
    propose_sha_value_name = propose_sha_name+"_value"

    return propose_sha_name, propose_sha_query_name, propose_sha_key_name, propose_sha_value_name

def update_all_mapping_dicts(optimizer):
    """Helper function to update mappings to nodes.
    Updates all the mapping dictionaries such as `get_initializer_by_name`. These need
    to be updated as nodes are added to the graph and are not yet know.
    """
    optimizer.get_node_by_node_name = get_node_mappings(optimizer.model)
    optimizer.get_initializer_by_name = get_initializer_mappings(optimizer.model)
    optimizer.get_initializer_idx_by_name = {
        n.name: idx for idx, n in enumerate(optimizer.model.graph.initializer)
    }
    optimizer.get_value_info_by_name = get_value_info_proto_mapping(
        optimizer.model
    )

    optimizer.get_node_by_input_name = get_node_by_input_name(optimizer.model)
    optimizer.get_node_by_output_name = get_node_by_output_name(optimizer.model)

    optimizer.node_name_mapping_dict = {}
    optimizer.tensor_name_set = {
        node_op
        for node in optimizer.model.graph.node
        for node_op in node.output
    }
    optimizer.tensor_name_set.update(*(
        set([node_inp for node_inp in node.input]) for node in optimizer.model.graph.node
    ))

    optimizer.mha_model_input_names = [_input.name for _input in optimizer.model.graph.input]
    optimizer.mha_model_output_names = [_output.name for _output in optimizer.model.graph.output]
    optimizer.mha_model_input_names_index_dict = {_input.name: i for i, _input in enumerate(optimizer.model.graph.input)}
