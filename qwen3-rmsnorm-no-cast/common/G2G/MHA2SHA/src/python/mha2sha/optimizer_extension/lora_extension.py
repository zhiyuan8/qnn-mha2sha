# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from collections import defaultdict
from dataclasses import dataclass
import enum
import numpy as np
from typing import Any, Dict, Tuple, List, Optional, Union
from onnx import numpy_helper
from onnx.onnx_pb import NodeProto, TensorProto

from mha2sha.utils.logger import log_debug, log_error, log_info, log_warning

from mha2sha.utils.onnx import (
    NodeNotFoundError,
    get_next_node_up_based_on_cond,
    get_next_node_down_based_on_cond,
    get_least_commom_ancestor_with_verified_pathway,
    get_mul_value,
)
from mha2sha.utils.utils import sha_node_name_basis, BranchType, ExtensionTypes
from mha2sha.utils.encoding_mapper_utils import (
    NodeMappingDict,
    create_activation_node_mapping_dict,
    update_sha_tensor_to_node_mapping_dict
)

mha2sha_hf_model_optimizer = Any  # Causes circular import

LORA_BRANCH_PREFIX = [ExtensionTypes.LORA+"_"+qkv for qkv in ["q", "k", "v"]]
LORA_ACTIVATION_ENCODING_KEYS = [
    "lora_b",
    "lora_alpha",
    "lora_add",
]
LORA_V1_ACTIVATION_ENCODING_KEYS = [
    "lora_add",
]
LORA_PARAM_ENCODING_KEYS=[
    "lora_b_param"
]

class LoraVersion(str, enum.Enum):
    """ Represents the LORA versions """
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"

@dataclass
class LoraNode:
    """ Defines nodes to record for lora models. """
    lora_a: Optional[Union[NodeProto, List[NodeProto]]] = None
    lora_b: Optional[Union[NodeProto, List[NodeProto]]] = None
    lora_add: Optional[Union[NodeProto, List[NodeProto]]] = None
    lora_alpha: Optional[Union[NodeProto, List[NodeProto]]] = None
    base_linear: Optional[Union[NodeProto, List[NodeProto]]] = None
    next_node: Optional[Any] = None

    def _get_nested_lora_a(self):
        _str = ""
        _lora_node = self.next_node
        while _lora_node:
            try:
                _str += f"-> {_lora_node.lora_a.name} "
            except:
                # Get lora_b if a SHA node. Sha node doesn't record lora_a, and lora_b is a list of NodeProto.
                _str += f"-> {_lora_node.lora_b[0].name} "
            _lora_node = _lora_node.next_node

        return _str

    def __str__(self):
        # useful for printing and debuging
        def get_name(lora_node_obj):
            if lora_node_obj:
                if isinstance(lora_node_obj, List):
                    return [getattr(_obj, "name") for _obj in lora_node_obj]
                else:
                    return getattr(lora_node_obj, "name")
            else:
                return None

        return f"LoraNode(\n" \
               f"    lora_a: '{get_name(self.lora_a)}',\n" \
               f"    lora_b: '{get_name(self.lora_b)}',\n" \
               f"    lora_add: '{get_name(self.lora_add)}',\n" \
               f"    lora_alpha: '{get_name(self.lora_alpha)}',\n" \
               f"    base_linear: '{get_name(self.base_linear)}'\n" \
               f"    next_node lora_a: '{self._get_nested_lora_a() if self.next_node else None}'\n" \
               f")"

    def __repr__(self):
        return str(self)

@dataclass
class LoraAdaptorSetInfo:
    """ Lora adaptor set info """
    num: Optional[str] = None
    lora_alpha: Optional[float] = None
    lora_alpha_model_input: Optional[TensorProto] = None
    lora_node_list: Optional[List[LoraNode]] = None
    lora_a_name_set: Optional[set[str]] = None

def get_lora_set_num(lora_node, split_string="lora_A"):
    """
    Use lora_node.lora_a's suffix after lora_A to identify lora adapter
    E.g.
        lora_a: 'down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_v_{lora_A}_0_Conv' -> adapter #0
        lora_a: 'down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_v_{lora_A}_1_Conv' -> adapter #1
        lora_a: 'down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_v_{lora_A}_2_Conv' -> adapter #2

        down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.1.Conv
    """
    if split_string.lower() == "lora_alpha":
        # for lora v1 sha get the lora num using alpha node input
        lora_node_name = getattr(lora_node, split_string.lower())[0].input[0]
    elif isinstance(getattr(lora_node, split_string.lower()), List):
        # sha case
        lora_node_name = getattr(lora_node, split_string.lower())[0].name
    else:
        lora_node_name = getattr(lora_node, split_string.lower()).name
    # lora_node_name = lora_node.lora_a.name
    try:
        # lora_a_name.split("lora_A") -> ['down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_v_', '_0_Conv']
        # '_0_Conv'.split(_) -> ['', '0', 'Conv']
        set_num_Conv = lora_node_name.split(split_string)[-1]
        if "_" in set_num_Conv:
            return [int(s) for s in set_num_Conv.split("_") if s.isdigit()][0]
        else:
            return [int(s) for s in set_num_Conv.split(".") if s.isdigit()]
            # return int(lora_node_name.split(split_string)[-1].split("_")[1])

    except Exception as e:
        return 0

def create_branch_lora_encoding_mapping_dict(mha_lora_node: LoraNode, prefix):
    """ create Dict[str, NodeMappingDict] for lora nodes. """
    num = get_lora_set_num(mha_lora_node)

    return {
        f"{prefix}_lora_b_{num}": create_activation_node_mapping_dict(
                input_1_name = mha_lora_node.lora_b.input[0],
                output_name = mha_lora_node.lora_b.output[0]
            ),
        f"{prefix}_lora_add_{num}": create_activation_node_mapping_dict(
                mha_lora_node.lora_add.input[0],
                mha_lora_node.lora_add.input[1],
                mha_lora_node.lora_add.output[0]
            ),
        f"{prefix}_lora_alpha_{num}": create_activation_node_mapping_dict(
                mha_lora_node.lora_alpha.input[0],
                mha_lora_node.lora_alpha.input[1],
                mha_lora_node.lora_alpha.output[0]
            ),
        f"{prefix}_lora_b_param_{num}": NodeMappingDict(
                mha_mapping_name_list = ["mha_param_name"],
                sha_mapping_name_list = ["sha_param_name"],
                mapping_name_dict = {
                    "mha_param_name": mha_lora_node.lora_b.input[1],
                    "sha_param_name": None
                    }
            )
        }

def create_lora_encoding_mapping_dict(mha_lora_node_list: List[LoraNode]):
    """
    Create a Dict[str, NodeMappingDict] for lora nodes which will be used in EncodingMapper.
    :param mha_lora_node_list: List of MHA LoRA nodes is q, k, v order. LoRA node can be None.
    :return LoraEncodingMappingDict:
    """
    # mha_rope_dict can be None when MHA ROPE pattern match fail.
    lora_dict = {}
    for mha_lora_node, prefix in zip(mha_lora_node_list, LORA_BRANCH_PREFIX):
        while mha_lora_node is not None:
            _lora_dict = create_branch_lora_encoding_mapping_dict(mha_lora_node, prefix)
            lora_dict.update(_lora_dict)
            mha_lora_node = mha_lora_node.next_node

    return lora_dict


def update_lora_sha_encoding_name_to_lora_encoding_mapping_dict(
        lora_encoding_mapping_dict,
        q_sha_lora_node,
        k_sha_lora_node,
        v_sha_lora_node
    ):
    """
    Update sha LORA encoding names to EncodingMappingDict.
    :param lora_encoding_mapping_dict: encoding_mapping_dict.lora: Dict[str, NodeMappingDict]
    :param q_sha_lora_node: A name dict for sha nodes created when optimizer split and create nodes for sha
    :param k_sha_lora_node: A name dict for sha nodes created when optimizer split and create nodes for sha
    :param v_sha_lora_node: A name dict for sha nodes created when optimizer split and create nodes for sha
    """
    for lora_branch_prefix, lora_sha_node in zip(LORA_BRANCH_PREFIX, [q_sha_lora_node, k_sha_lora_node, v_sha_lora_node]):
        if lora_sha_node and not lora_sha_node.lora_b and lora_sha_node.lora_add:
            set_num = get_lora_set_num(lora_sha_node, "lora_alpha")
            for lora_node_name in LORA_V1_ACTIVATION_ENCODING_KEYS:
                dict_key = "_".join([lora_branch_prefix, lora_node_name, str(set_num)])
                update_sha_tensor_to_node_mapping_dict(
                    node_mapping_dict=lora_encoding_mapping_dict[dict_key],
                    sha_node_list=getattr(lora_sha_node, lora_node_name)
                )

        # Check lora b for exsitance. Empty lora_b list should retuen False.
        while lora_sha_node and lora_sha_node.lora_b:
            # Handle activation encodings
            set_num = get_lora_set_num(lora_sha_node, "lora_b")
            for lora_node_name in LORA_ACTIVATION_ENCODING_KEYS:
                dict_key = "_".join([lora_branch_prefix, lora_node_name, str(set_num)])
                update_sha_tensor_to_node_mapping_dict(
                    node_mapping_dict = lora_encoding_mapping_dict[dict_key],
                    sha_node_list = getattr(lora_sha_node, lora_node_name)
                )

            # Handle param encodings, use lora_node.lora_b NodeProto and write to LORA_PARAM_ENCODING_KEYS
            for lora_node_name in LORA_PARAM_ENCODING_KEYS:
                dict_key = "_".join([lora_branch_prefix, lora_node_name, str(set_num)])
                update_sha_tensor_to_node_mapping_dict(
                    node_mapping_dict = lora_encoding_mapping_dict[dict_key],
                    sha_node_list = getattr(lora_sha_node, "lora_b")
                )

            lora_sha_node = lora_sha_node.next_node

def get_lora_version(info_dict, get_node_by_output_name, mha_model_input_names) -> LoraVersion:
    """ Check info dict and return lora version. """
    input_lora_b = True
    next_lora = True
    found_lora = False

    for branch in ["query", "key", "value"]:
        if (input_lora_b or next_lora) and "mha_lora_node" in info_dict[branch]:
            found_lora = True

            if not (info_dict[branch]["mha_lora_node"].lora_b.input[1] in get_node_by_output_name or \
                    info_dict[branch]["mha_lora_node"].lora_b.input[1] in mha_model_input_names):
                input_lora_b = False

            if not info_dict[branch]["mha_lora_node"].next_node:
                next_lora = False

    assert found_lora, "No Lora Adaptor found Q, K, V branch. Can't get LoRA version."

    if input_lora_b:
        # for lora v1 lora b weights should be a model input
        log_debug(f"Found lora {LoraVersion.V1}")
        return LoraVersion.V1
    elif not next_lora:
        # for lora v2 lora b weights must be a constant initializer input
        log_debug(f"Found lora {LoraVersion.V2}")
        return LoraVersion.V2
    else:
        # for lora v3 there must be concurrent lora adaptors
        log_debug(f"Found lora {LoraVersion.V3}")
        return LoraVersion.V3


class LoraExtension:
    """ Extenstion helpers for mha2sha_optimzer to bridge Morpheus pipeline code base and v1.0.0 release.  """
    def __init__(self, mha2sha_optim: mha2sha_hf_model_optimizer) -> None:
        """Initalizes an instance based on the MHA2SHA optimizer provided.

        Args:
            mha2sha_optim:
                MHA2SHAOptimizer instance holding the model loader and model info.
        """
        self.mha2sha_optim = mha2sha_optim
        self.map_lora_encoding = True

    def _get_empty_sha_lora_node(self):
        return LoraNode(
                    lora_b=[],
                    lora_add=[],
                    lora_alpha=[],
                    next_node = None,
                )

    def reset_sha_encoding_name_list(self):
        """
        Reset mha sha names for LORA tensors.
        """
        self.map_lora_encoding = True
        self.q_sha_lora_node = self._get_empty_sha_lora_node()
        self.k_sha_lora_node = self._get_empty_sha_lora_node()
        self.v_sha_lora_node = self._get_empty_sha_lora_node()

    def _update_sha_lora_node(self,
                              sha_lora_node,
                              lora_b,
                              lora_alpha,
                              lora_add):
        """
        Recusivly call _update_sha_lora_node to check if we are one the right linked node.
        If lora_alpha has same input[1], they are we are updating the correct SHA lora node.
        Else, we move to the next linked node and do it again.
        """
        # Make sure lora_alpha.input[1] are the same for the same SHA lora node
        # If not, we go the next node and check again
        if len(sha_lora_node.lora_b) > 0 and (lora_alpha.input[1] != sha_lora_node.lora_alpha[-1].input[1]):
            # make sure lora_alpha.input[1] are the same for the same SHA lora node
            if not sha_lora_node.next_node:
                sha_lora_node.next_node = self._get_empty_sha_lora_node()
            self._update_sha_lora_node(sha_lora_node.next_node,
                                       lora_b,
                                       lora_alpha,
                                       lora_add)
        else:
            if lora_b is None:
                sha_lora_node.lora_add.append(lora_add)
                sha_lora_node.lora_alpha.append(lora_alpha)
            else:
                sha_lora_node.lora_b.append(lora_b)
                sha_lora_node.lora_add.append(lora_add)
                sha_lora_node.lora_alpha.append(lora_alpha)

    def update_sha_lora_node(self,
                             branch_type,
                             lora_b,
                             lora_b_split,
                             lora_alpha,
                             lora_add):
        """ Update sha node """
        if branch_type == BranchType.Q:
            sha_lora_node = self.q_sha_lora_node
        elif branch_type == BranchType.K:
            sha_lora_node = self.k_sha_lora_node
        elif branch_type == BranchType.V:
            sha_lora_node = self.v_sha_lora_node

        if lora_b_split is not None:
            self._update_sha_lora_node(sha_lora_node=sha_lora_node,
                                       lora_b=None,
                                       lora_alpha=lora_alpha,
                                       lora_add=lora_add)

        else:
            self._update_sha_lora_node(sha_lora_node,
                                       lora_b,
                                       lora_alpha,
                                       lora_add)

    def verify_and_capture_lora_structure(self, add_lora):
        """
        Find base linear, lora_b and lora_b.
            x-----------
            |           |
            |         lora_a
            |           |
        base linear   Mul alpha
            |           |
            |         lora_b
            |           |
            Add--------/

        the lora-structure will be verified.
        If the verification failed, None will be returned.

        combine "lora verification" and "lora nodes matching" into one function
        since the majority of their logic is identical.
        """

        matmul_op_type = "Conv"

        # add_lora's inputs should come from two different nodes
        prev_node_0 = self.mha2sha_optim.get_node_by_output_name.get(
                                add_lora.input[0], None)
        prev_node_1 = self.mha2sha_optim.get_node_by_output_name.get(
                                add_lora.input[1], None)
        if prev_node_0 is None or prev_node_1 is None:
            return None
        if prev_node_0.name == prev_node_1.name:
            return None

        # we only allow transpose / reshape
        # between "Add" and matmul_op_type
        # Allow Add Op for multiple lora adaptor setting
        allow_types = set(["Add", "Transpose", "Reshape"])
        allow_types.add(matmul_op_type)

        try:
            temp_base_linear = get_next_node_up_based_on_cond(
                self.mha2sha_optim.get_node_by_output_name[add_lora.input[0]],
                self.mha2sha_optim.get_node_by_output_name,
                node_found_cond=lambda n: n.op_type == matmul_op_type,
                node_end_search_cond=lambda n: n.op_type not in allow_types,
                search_input_0_path_only=True
            )
        except NodeNotFoundError:
            return None

        try:
            temp_lora_b = get_next_node_up_based_on_cond(
                self.mha2sha_optim.get_node_by_output_name[add_lora.input[1]],
                self.mha2sha_optim.get_node_by_output_name,
                node_found_cond=lambda n: n.op_type == matmul_op_type or n.op_type == "MatMul",
                node_end_search_cond=lambda n: n.op_type not in allow_types
            )
        except NodeNotFoundError:
            return None

        temp_base_linear_init = numpy_helper.to_array(
                self.mha2sha_optim.get_initializer_by_name[temp_base_linear.input[1]])

        if temp_lora_b.op_type == "Conv":
            temp_lora_b_init = numpy_helper.to_array(
                    self.mha2sha_optim.get_initializer_by_name[temp_lora_b.input[1]])
            temp_lora_b_init_shape = temp_lora_b_init.shape

            # base linear input should have higher rank then lora_b input
            # ONNX conv weight is [O, I, kH, kW]
            if temp_base_linear_init.shape[1] > temp_lora_b_init_shape[1]:
                base_linear = temp_base_linear
                lora_b = temp_lora_b
            else:
                base_linear = temp_lora_b
                lora_b = temp_base_linear
        else:
            base_linear = temp_base_linear
            lora_b = temp_lora_b

        # we only allow transpose / reshape / Mul
        # between matmul_op_type and matmul_op_type
        allow_types = set(["Add", "Mul", "Transpose", "Reshape"])
        allow_types.add(matmul_op_type)

        if lora_b.input[0] not in self.mha2sha_optim.get_node_by_output_name.keys():
            return None

        try:
            lora_a = get_next_node_up_based_on_cond(
                self.mha2sha_optim.get_node_by_output_name[lora_b.input[0]],
                self.mha2sha_optim.get_node_by_output_name,
                node_found_cond=lambda n: n.op_type == matmul_op_type or n.op_type == "MatMul",
                node_end_search_cond=lambda n: n.op_type not in allow_types
            )
        except NodeNotFoundError:
            return None

        # verify lora_a and base_linear has a LCA (least common ancestor)
        # we only allow transpose / reshape
        # - between LCA and lora_a
        # - between LCA and base_linear
        # that is we only allow transpose / reshape as pathway nodes
        allow_types = set(["Add", "Transpose", "Reshape", "Cast"])
        lca = get_least_commom_ancestor_with_verified_pathway(
                    self.mha2sha_optim.get_node_by_output_name[lora_a.input[0]],
                    self.mha2sha_optim.get_node_by_output_name[base_linear.input[0]],
                    self.mha2sha_optim,
                    pathway_nodes_verifier=lambda n:n.op_type in allow_types,
                    search_input_0_path_only=True,
            )
        if lca is None:
            log_debug(
                "Ignoring candiate lora structure, reason: LCA not found\n    "
                f"lora_add: '{add_lora.name}', "
                f"base_linear:'{base_linear.name}', "
                f"lora_a:'{lora_a.name}', "
                f"lora_b:'{lora_b.name}'")
            return None

        lora_node = LoraNode(
            base_linear=base_linear,
            lora_a=lora_a,
            lora_b=lora_b,
            lora_add=add_lora
        )

        # caputre lora alpha
        lora_alpha = get_next_node_down_based_on_cond(
            lora_a,
            self.mha2sha_optim.get_node_by_input_name,
            node_found_cond=lambda n: n.op_type == "Mul",
            node_end_search_cond=lambda n: n == lora_b
        )

        if lora_alpha:
            lora_node.lora_alpha = lora_alpha
            # verify there is only one Mul between lora_a and lora_b
            lora_alpha_up = get_next_node_up_based_on_cond(
                self.mha2sha_optim.get_node_by_output_name[lora_b.input[0]],
                self.mha2sha_optim.get_node_by_output_name,
                node_found_cond=lambda n: n.op_type == "Mul"
            )

            if lora_alpha is not lora_alpha_up:
                log_warning(
                    "Ignoring candiate lora structure, "
                    "reason: more than one lora-Mul\n    "
                    f"base_linear:'{base_linear.name}', "
                    f"lora_a:'{lora_a.name}', "
                    f"lora_b:'{lora_b.name}'")
                return None

            # verify one input of this Mul is constant or input
            def is_tensor_cst_or_input(tensor_name):
                if tensor_name in self.mha2sha_optim.mha_model_input_names:
                    return True # input
                producer = self.mha2sha_optim.get_node_by_output_name.get(tensor_name, None)
                if producer and (producer.op_type in ("Constant", "Identity")):
                    return True # Constant or Identity
                if tensor_name in self.mha2sha_optim.get_initializer_by_name.keys():
                    return True # initializer (same as consant)
                return False

            # Lora Mul for max rank input can be Constant, Model input, or Gather -> reshape.
            if np.array(
                    [is_tensor_cst_or_input(x) for x in lora_alpha.input]
                ).sum() != 1 and \
                (_producer := self.mha2sha_optim.get_node_by_output_name.get(lora_alpha.input[1], None)) and \
                _producer.op_type != "Reshape":

                log_warning(
                    "Ignoring candiate lora structure, "
                    "reason: one of lora-Mul is neither constant nor input\n    "
                    f"base_linear:'{base_linear.name}', "
                    f"lora_a:'{lora_a.name}', "
                    f"lora_b:'{lora_b.name}'")

                return None

        return lora_node

    def get_qkv_lora_structure(self, qk_matmul_node, qkv_matmul_node):
        """
        Find Add op adds lora base linear and lora adaptor. Search up from qk_matmul or qkv_matmul
        for any Conv/MatMul, then search down for and Add between Conv and qk/qkv_matmul. Varify
        founded lora adds.
        :return q_add_lora: lora_add if verifed, proj Conv/MatMul if not a lora branch
        :return k_add_lora: lora_add if verifed, proj Conv/MatMul if not a lora branch
        :return v_add_lora: lora_add if verifed, proj Conv/MatMul if not a lora branch
        """
        def is_elementwise_Add(node):
            # element-wise add
            return node.op_type == "Add" and node.input[1] not in self.mha2sha_optim.get_initializer_by_name.keys()

        proj_op_type = "Conv" if self.mha2sha_optim.mha_conv else "MatMul"

        # - when lora exists, q_conv_candidate may be lora's conv or base_linear's conv
        #   but that dosen't matter
        # - when lora dosen't exist, q_conv_candidate is the base_linear's conv
        qkv_convs = dict()
        # defaultdict in the case we never set the lora nodes, we get None at the end
        qkv_lora_nodes = defaultdict(lambda: None)
        qkv_branch_types = [BranchType.Q, BranchType.K, BranchType.V]

        for branch_type in qkv_branch_types:
            found_lora_node = False
            qk_matmul_input_index = 0 if branch_type == BranchType.Q else 1
            qk_or_qkv_matmul_node = (
                qk_matmul_node
                if branch_type in [BranchType.Q, BranchType.K]
                else qkv_matmul_node
            )

            try:
                # Assumption:
                # Add - input 0 -> from linear_base / concerrent lora Add
                # Add - input 1 -> from lora branch
                if self.mha2sha_optim.llm_model: # gqa have to be llm model.
                    # Find last lora adapter conv.
                    conv_candidate = get_next_node_up_based_on_cond(
                        self.mha2sha_optim.get_node_by_output_name[
                            qk_or_qkv_matmul_node.input[qk_matmul_input_index]
                        ],
                        self.mha2sha_optim.get_node_by_output_name,
                        node_found_cond=lambda n: n.op_type == proj_op_type,
                    )

                    # Find last lora adapter Add.
                    add_lora = get_next_node_down_based_on_cond(
                        conv_candidate,
                        self.mha2sha_optim.get_node_by_input_name,
                        node_found_cond=is_elementwise_Add,
                        node_end_search_cond=lambda n: n == qk_or_qkv_matmul_node,
                    )

                    # Use last lora adapter Add to search for lora linear base
                    search_up_start_node = add_lora
                else:
                    search_up_start_node = self.mha2sha_optim.get_node_by_output_name[
                        qk_or_qkv_matmul_node.input[qk_matmul_input_index]
                    ]

                conv_candidate = get_next_node_up_based_on_cond(
                    search_up_start_node,
                    self.mha2sha_optim.get_node_by_output_name,
                    node_found_cond=lambda n: n.op_type == proj_op_type,
                    search_input_0_path_only=True
                )

                add_lora = get_next_node_down_based_on_cond(
                    conv_candidate,
                    self.mha2sha_optim.get_node_by_input_name,
                    node_found_cond=is_elementwise_Add,
                    node_end_search_cond=lambda n: n == qk_or_qkv_matmul_node,
                )

                lora_node = self.verify_and_capture_lora_structure(add_lora)
                qkv_lora_nodes[branch_type] = (
                    lora_node  # will be None or the LoRA node
                )

                if lora_node is not None:
                    found_lora_node = True

            except NodeNotFoundError:
                # found_lora_node starts as False so we don't need to do anything for conv
                # and defaultdict handles lora_node to default to None when we return
                ...

            if not found_lora_node:
                log_warning(f"No lora adaptor found on branch {branch_type.name}")
                # Search for conv_candidate without setting search_input_0_path_only to True
                # because we've know there's no LoRA adaptor found on that branch.
                _qkv_conv = get_next_node_up_based_on_cond(
                    self.mha2sha_optim.get_node_by_output_name[
                        qk_or_qkv_matmul_node.input[qk_matmul_input_index]
                    ],
                    self.mha2sha_optim.get_node_by_output_name,
                    node_found_cond=lambda n: n.op_type == proj_op_type,
                )
                qkv_convs[branch_type] = _qkv_conv
            else:
                qkv_convs[branch_type] = lora_node.base_linear

        return [qkv_convs[b] for b in qkv_branch_types] + [qkv_lora_nodes[b] for b in qkv_branch_types]

    def _get_next_node(self, lora_node, lora_add_candidate):
        """
        If lora_add_candidate is verified, add next_node to lora_node.next_node.
        Recursively call next_node to create linked LoraNode.
        """
        next_node = self.verify_and_capture_lora_structure(lora_add_candidate)
        if next_node:
            lora_node.next_node = next_node
            if (lora_add_cand := self.mha2sha_optim.get_node_by_input_name[next_node.lora_add.output[0]][0]).op_type == "Add":
                self._get_next_node(next_node, lora_add_cand)

    def _register_next_node(self, lora_node):
        """
        Check lora node's lora_add output to find concurent lora adaptor.
        If exist add it to LoraNode.next_node
        """
        # If lora_node.lora_add's output is Add Op, it implies there can have multiple lora.
        if (lora_add_cand := self.mha2sha_optim.get_node_by_input_name[lora_node.lora_add.output[0]][0]).op_type == "Add":
            self._get_next_node(lora_node, lora_add_cand)

    def update_dqkv_for_lora(self, dqkv, lora_node):
        """
        Get dquery, dkey, dvalue in info dict for lora.
        """
        lora_b = lora_node.lora_b
        if lora_b.input[1] in self.mha2sha_optim.get_initializer_by_name:
            lora_b_init = self.mha2sha_optim._mha_conv_extension.get_conv_weight_in_OI(lora_b)
            dqkv["lora_b_init"] = lora_b_init
        dqkv["mha_lora_node"] = lora_node

    def get_qkv_info(
            self,
            qk_matmul_node: NodeProto,
            qkv_matmul_node: NodeProto,
        ) -> Tuple[Dict, Dict, Dict]:
        """
            Function responsible for collecting QKV information.

            :param qk_matmul_node: The MatMul of where the Query and Key branches join.
            :param qkv_matmaul_node: The MatMul of where the Query, Key, and Key branches join.

            :return dquery: Dict - it contains matmul (node, initializers)
                    and add (node, initializers).
            :return dkey: Dict - it contains matmul (node, initializers)
                    and add (node, initializers).
            :return dvalue: Dict - it contains matmul (node, initializers)
                    and add (node, initializers).
            """
        # Find Add op that adds up baselinear and lora output
        assert self.mha2sha_optim.mha_conv, "Support mha-conv lora at the moment"
        q_conv, k_conv, v_conv, \
            q_lora_node, k_lora_node, v_lora_node \
                = self.get_qkv_lora_structure(qk_matmul_node, qkv_matmul_node)

        # Search and register next_node if exists.
        for lora_node in [q_lora_node, k_lora_node, v_lora_node]:
            if lora_node is not None:
                self._register_next_node(lora_node)

        # Update dquery, dkey, dvalue with lora info
        d_qkv = []
        for qkv_conv, qkv_lora_node in zip([q_conv, k_conv, v_conv], [q_lora_node, k_lora_node, v_lora_node]):
            _d_qkv = self.mha2sha_optim._mha_conv_extension.get_dqkv_for_qkv_info(qkv_conv)
            if qkv_lora_node is not None:
                self.update_dqkv_for_lora(_d_qkv, qkv_lora_node)
            d_qkv.append(_d_qkv)

        dquery, dkey, dvalue = d_qkv

        return dquery, dkey, dvalue

    def get_single_qkv_lora_input(self, dqkv):
        """ if lora_a in dqkv, add lora alpha and return alpha mul, else return None. """
        if "mha_lora_node" in dqkv.keys():
            mha_lora_node = dqkv["mha_lora_node"]
            lora_inp = mha_lora_node.lora_a

            # Reuse lora input [1] if input [1] is reshape. Max rank model has lora_alpha from model
            # input and have lora_alpha -> Pad -> Gather -> Reshape -> Mul pattern
            if (mha_lora_node.lora_alpha.input[1] in self.mha2sha_optim.mha_model_input_names) or \
                ((_producer := self.mha2sha_optim.get_node_by_output_name.get(mha_lora_node.lora_alpha.input[1], None)) and \
                 _producer.op_type == "Reshape"):
                lora_inp = self.mha2sha_optim._op_factory.get_element_mul_op(
                            lora_inp,
                            mha_lora_node.lora_alpha.input[1]
                        )
            else:
                mul_value = get_mul_value(mha_lora_node.lora_alpha,
                             self.mha2sha_optim.get_initializer_by_name,
                             self.mha2sha_optim.get_node_by_output_name
                        )

                # Got vector mul -> lora v3 concat.
                if len(mul_value.shape) > 1 or mul_value.shape[0] > 1:
                    mul_value = mul_value.reshape(1, -1, 1, 1)
                else:
                    mul_value = mul_value[0]

                lora_inp, lora_alpha_init = self.mha2sha_optim._op_factory.get_mul_op(
                                                lora_inp,
                                                mul_value
                                            )
                self.mha2sha_optim.model.graph.initializer.extend(lora_alpha_init)

            self.mha2sha_optim.model.graph.node.append(lora_inp)
            return lora_inp

        return None

    def get_qkv_lora_input(self, info_dict):
        """ Return query_lora_inp """
        query_lora_inp = self.get_single_qkv_lora_input(info_dict["query"])
        key_lora_inp = self.get_single_qkv_lora_input(info_dict["key"])
        value_lora_inp = self.get_single_qkv_lora_input(info_dict["value"])

        return query_lora_inp, key_lora_inp, value_lora_inp

    def _attach_concurrent_lora_adaptors(self,
                                         inp,
                                         lora_node,
                                         ns,
                                         head_num,
                                         head_dim,
                                         branch_type,
    ):
        """ Attach concurrent lora adaptors when lora_node is not None. """
        if lora_node is None:
            return inp

        lora_num = get_lora_set_num(lora_node)
        _info_set = self.mha2sha_optim.lora_adaptor_set_info_dict[lora_num]

        assert lora_node.lora_a.name in _info_set.lora_a_name_set, "Got unexpected lora node"
        conv_weight_init = self.mha2sha_optim._mha_conv_extension.get_conv_weight_in_OI(lora_node.lora_b)
        lora_inp = lora_node.lora_alpha

        lora_out = self.mha2sha_optim._mha_conv_extension.create_single_conv(
                    conv_weight_init,
                    ns,
                    head_num,
                    head_dim,
                    lora_inp,
                    suffix=f"lora_b_{lora_num}",
                    branch_type=branch_type,
                    # lora has no bias
                )

        # Add inp (from previous lora Add) and lora_out (from current lora_b)
        lora_add = self.mha2sha_optim._op_factory.get_add_op(inp, lora_out)
        self.mha2sha_optim.model.graph.node.append(lora_add)
        inp = lora_add

        self.update_sha_lora_node(
            branch_type=branch_type,
            lora_b=lora_out,
            lora_b_split=None,
            lora_alpha=lora_inp,  # lora input is alpha
            lora_add=lora_add,
        )

        inp = self._attach_concurrent_lora_adaptors(
                inp,
                lora_node.next_node,
                ns,
                head_num,
                head_dim,
                branch_type
            )

        return inp

    def create_sha_lora_add(self,
                            ns,
                            head_num,
                            qkv_matmul_input,
                            lora_split_input,
                            suffix=None,
                            branch_type=None,
                           ):
        sha_qkv_name = sha_node_name_basis(ns, head_num)[branch_type.value]

        sha_lora_add = self.mha2sha_optim._op_factory.get_add_op(qkv_matmul_input,
                                                                 lora_split_input,
                                                                 propose_op_name=sha_qkv_name+"_Add" if suffix is None else sha_qkv_name+f"_{suffix}",)
        return sha_lora_add

    def _attach_single_lora_adaptor(
            self,
            branch_info_dict,
            ns,
            head_num,
            head_dim,
            inp,
            lora_inp,
            branch_type,
            lora_split=None
        ):
        """ Attach lora adaptor from lora out to one of qkv conv
        Args:
            :param branch_info_dict: Dict containing corresponding branch information
            :param ns: number of start node (attention layer num)
            :param head_num: head number in heads
            :param head_dim: vector dim for each head
            :param inp: qkv input to lora-add node
            :param lora_inp: input to lora-b node
            :param branch_type: query/key/value branch type
            :param lora_split: split node if any input to lora-add node
        """
        lora_out = None
        init_name = "lora_b_init"
        if lora_split is None and init_name in branch_info_dict and \
                (conv_weight_init := branch_info_dict[init_name]) is not None:
            lora_num = get_lora_set_num(branch_info_dict["mha_lora_node"])
            lora_out = self.mha2sha_optim._mha_conv_extension.\
                        create_single_conv(
                            conv_weight_init,
                            ns,
                            head_num,
                            head_dim,
                            lora_inp,
                            suffix=f"lora_b_{lora_num}",
                            branch_type=branch_type,
                            # lora has no bias
                        )

        if lora_split is not None:
            # lora v1 case
            lora_add = self.create_sha_lora_add(ns,
                                                head_num,
                                                inp,
                                                lora_split.output[head_num],
                                                suffix=f"lora_add",
                                                branch_type=branch_type)
            self.mha2sha_optim.model.graph.node.append(lora_add)
            inp = lora_add

        elif lora_out is None:
            return inp

        else:
            # lora v2 case
            lora_add = self.create_sha_lora_add(ns,
                                                head_num,
                                                inp,
                                                lora_out,
                                                suffix=f"lora_add",
                                                branch_type=branch_type)
            self.mha2sha_optim.model.graph.node.append(lora_add)
            inp = lora_add

        self.update_sha_lora_node(
                    branch_type=branch_type,
                    lora_b=lora_out,
                    lora_b_split=lora_split,
                    lora_alpha=lora_inp,  # lora input is alpha
                    lora_add=lora_add
        )

        if lora_out is not None:
            inp = self._attach_concurrent_lora_adaptors(
                    inp,
                    branch_info_dict["mha_lora_node"].next_node,
                    ns,
                    head_num,
                    head_dim,
                    branch_type
                )

        return inp

    def _split_lora_b_adaptor_output(
            self,
            branch_info_dict,
            num_heads,
            head_dim,
        ):
        """
        Split lora B adaptor output into specified number of heads
        Args:
            branch_info_dict: Dict containing corresponding branch information
            num_heads: number of heads
            head_dim: each head dim
        """
        lora_node = branch_info_dict["mha_lora_node"]

        # transpose lora B output [B, seq_len, num_head*head_dim] -> [B, num_head*head_dim, seq_len]
        key_transpose = self.mha2sha_optim._op_factory.get_transpose_op(lora_node.lora_b, [0, 2, 1])
        self.mha2sha_optim.model.graph.node.append(key_transpose)

        reshape_node, reshape_init = self.mha2sha_optim._op_factory.get_reshape_op(key_transpose, [1, head_dim*num_heads, 1, -1])
        self.mha2sha_optim.model.graph.node.append(reshape_node)
        self.mha2sha_optim.model.graph.initializer.extend(reshape_init)

        lora_split_node, lora_split_init = self.mha2sha_optim._op_factory.get_split_op(reshape_node, 1,
                                                                                       [head_dim]*num_heads,
                                                                                       num_heads)
        self.mha2sha_optim.model.graph.node.append(lora_split_node)
        self.mha2sha_optim.model.graph.initializer.extend(lora_split_init)

        return lora_split_node

    def attach_single_lora_adaptor(
            self,
            branch_info_dict,
            ns,
            head_num,
            head_dim,
            inp,
            lora_inp,
            branch_type,
            lora_split=None,
        ):
        """ Attach lora adaptor from lora out to one of qkv conv
        Args:
            :param branch_info_dict: Dict containing corresponding branch information
            :param ns: number of start node (attention layer num)
            :param head_num: head number in heads
            :param head_dim: vector dim for each head
            :param inp: qkv input to lora-add node
            :param lora_inp: input to lora-b node
            :param branch_type: query/key/value branch type
            :param lora_split: split node if any input to lora-add node
        """
        inp = self._attach_single_lora_adaptor(
                branch_info_dict,
                ns,
                head_num,
                head_dim,
                inp,
                lora_inp,
                branch_type,
                lora_split,
            )
        return inp

    def split_lora_b_adaptor_output(
            self,
            branch_info_dict,
            num_heads,
            head_dim,
        ):
        """ Split lora B output into specified number of heads """
        inp = self._split_lora_b_adaptor_output(
                branch_info_dict,
                num_heads,
                head_dim,
            )

        return inp

    def attach_lora_adaptor(self,
                            info_dict,
                            ns,
                            head_num,
                            head_dim,
                            sha_base_attn_node_list,
                            query_inp,
                            key_inp,
                            value_inp,
                            query_lora_inp,
                            key_lora_inp,
                            value_lora_inp):
        """ Attach lora adpator from lora out to qkv conv """
        query_inp = self.attach_single_lora_adaptor(info_dict["query"],
                                                    ns, head_num, head_dim,
                                                    query_inp, query_lora_inp,
                                                    BranchType.Q)
        key_inp = self.attach_single_lora_adaptor(info_dict["key"],
                                                  ns, head_num, head_dim,
                                                  key_inp, key_lora_inp,
                                                  BranchType.K)
        value_inp = self.attach_single_lora_adaptor(info_dict["value"],
                                                    ns, head_num, head_dim,
                                                    value_inp, value_lora_inp,
                                                    BranchType.V)

        return query_inp, key_inp, value_inp

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
