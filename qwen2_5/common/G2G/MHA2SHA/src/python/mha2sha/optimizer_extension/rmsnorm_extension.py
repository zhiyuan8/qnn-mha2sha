# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
""" Extension for handling RMSNorm Op inside the Attention Module """

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict

from onnx import numpy_helper
from onnx.onnx_pb import NodeProto

from mha2sha.utils.logger import log_error
from mha2sha.utils.onnx import (
    NodeNotFoundError,
    get_next_node_down_based_on_cond,
    get_initializer_value,
)
from mha2sha.utils.op_factory import create_tensor_name


from mha2sha.utils.utils import BranchType, ExtensionTypes
from mha2sha.utils.encoding_mapper_utils import (
    NodeMappingDict,
    create_activation_node_mapping_dict,
    update_sha_tensor_to_node_mapping_dict
)

mha2sha_hf_model_optimizer = Any  # Causes circular import
RMSNORM_BRANCH_PREFIX = [ExtensionTypes.RMSNORM+"_"+qk for qk in ["q", "k"]]
RMSNORM_ACTIVATION_ENCODING_KEYS = [
    "pow_",
    "reduce_mean",
    "add",
    "sqrt",
    "div",
    "mul",
    "out_mul",
]

RMSNORM_PARAM_ENCODING_KEYS = ["out_mul_bias"]


@dataclass
class RMSNormAttnNode:
    """Stores Information of RMSNorm in the Attention Module"""

    # pow_ due to name mangling
    pow_: Optional[Union[NodeProto, List[NodeProto]]]
    reduce_mean: Optional[Union[NodeProto, List[NodeProto]]]
    add: Optional[Union[NodeProto, List[NodeProto]]]
    sqrt: Optional[Union[NodeProto, List[NodeProto]]]
    div: Optional[Union[NodeProto, List[NodeProto]]]
    mul: Optional[Union[NodeProto, List[NodeProto]]]
    out_mul: Optional[Union[NodeProto, List[NodeProto]]]

def create_branch_rmsnorm_mapping_dict(
    mha_rmsnorm_attn_node: Dict,
    prefix: str
) -> Dict[str, NodeMappingDict]:
    """
    mha_rope_dict: default dict
    """
    return {
        f"{prefix}_pow_": create_activation_node_mapping_dict(
            input_1_name=None,
            input_2_name=mha_rmsnorm_attn_node.pow_.input[1],
            output_name=mha_rmsnorm_attn_node.pow_.output[0],
        ),
        f"{prefix}_reduce_mean": create_activation_node_mapping_dict(
            input_1_name=None,
            input_2_name=None,
            output_name=mha_rmsnorm_attn_node.reduce_mean.output[0],
        ),
        f"{prefix}_add": create_activation_node_mapping_dict(
            input_1_name=None,
            input_2_name=mha_rmsnorm_attn_node.add.input[1],
            output_name=mha_rmsnorm_attn_node.add.output[0],
        ),
        f"{prefix}_sqrt": create_activation_node_mapping_dict(
            input_1_name=None,
            input_2_name=None,
            output_name=mha_rmsnorm_attn_node.sqrt.output[0],
        ),
        f"{prefix}_div": create_activation_node_mapping_dict(
            input_1_name=None,
            input_2_name=mha_rmsnorm_attn_node.div.input[1],
            output_name=mha_rmsnorm_attn_node.div.output[0],
        ),
        f"{prefix}_mul": create_activation_node_mapping_dict(
            input_1_name=mha_rmsnorm_attn_node.mul.input[0],
            input_2_name=mha_rmsnorm_attn_node.mul.input[1],
            output_name=mha_rmsnorm_attn_node.mul.output[0],
        ),
        f"{prefix}_out_mul": create_activation_node_mapping_dict(
            input_1_name=None,
            input_2_name=mha_rmsnorm_attn_node.out_mul.input[1],
            output_name=mha_rmsnorm_attn_node.out_mul.output[0],
        ),
    }

def create_rmsnorm_encoding_mapping_dict(mha_rmsnorm_node_list: List[RMSNormAttnNode]):
    rmsnorm_dict = {}
    for mha_rope_dict, prefix in zip(mha_rmsnorm_node_list, RMSNORM_BRANCH_PREFIX):
        if mha_rope_dict:
            _rmsnorm_dict = create_branch_rmsnorm_mapping_dict(mha_rope_dict, prefix)
            rmsnorm_dict.update(_rmsnorm_dict)
    return rmsnorm_dict

def update_rmsnorm_sha_encoding_name_to_rmsnorm_encoding_mapping_dict(
        rmsnorm_encoding_mapping_dict,
        query_pattern_dict,
        key_pattern_dict,
    ):
    """
    Update sha internal RMSNORM encoding names to EncodingMappingDict.
    :param lora_encoding_mapping_dict: encoding_mapping_dict.lora: Dict[str, NodeMappingDict]
    :param query_pattern_dict: A name dict for sha nodes created when optimizer split and create nodes for sha
    :param key_pattern_dict: A name dict for sha nodes created when optimizer split and create nodes for sha
    """
    for branch_prefix, rmsnodes in zip(RMSNORM_BRANCH_PREFIX, [query_pattern_dict, key_pattern_dict]):

        # Handle activation encodings
        for node_name in RMSNORM_ACTIVATION_ENCODING_KEYS:
            if node_name == "out_mul":
                node_mapping_dict = rmsnorm_encoding_mapping_dict[branch_prefix+"_"+node_name]
                sha_node_list = getattr(rmsnodes, "out_mul")

                # RMS Norm mha has input 0 = activation, input 1 = weight
                # RMS Norm sha has input 0 = weight, input 1 = activation
                # mha input 0 will map to sha_input_2_activation_name
                # mha input 1 will map to sha_input_1_activation_name
                if "sha_input_1_activation_name" in node_mapping_dict.sha_mapping_name_list:
                    node_mapping_dict.mapping_name_dict["sha_input_1_activation_name"] = [node.input[1] for node in sha_node_list]

                if "sha_input_2_activation_name" in node_mapping_dict.sha_mapping_name_list:
                    node_mapping_dict.mapping_name_dict["sha_input_2_activation_name"] = [node.input[0] for node in sha_node_list]

                if "sha_output_activation_name" in node_mapping_dict.sha_mapping_name_list:
                    node_mapping_dict.mapping_name_dict["sha_output_activation_name"] = [node.output[0] for node in sha_node_list]

            else:
                update_sha_tensor_to_node_mapping_dict(
                    node_mapping_dict = rmsnorm_encoding_mapping_dict[branch_prefix+"_"+node_name],
                    sha_node_list = getattr(rmsnodes, node_name)
                )

class RmsnormExtension:
    """Extension helpers for mha2sha_optimizer for RMSNorm's internal to attentions"""

    def __init__(self, mha2sha_optim: mha2sha_hf_model_optimizer) -> None:
        """Initalizes an instance based on the MHA2SHA optimizer provided.

        Args:
            mha2sha_optim:
                MHA2SHAOptimizer instance holding the model loader and model info.
        """
        self.mha2sha_optim = mha2sha_optim
        self.reset_sha_encoding_name_list()

    def reset_sha_encoding_name_list(self):
        """ Reset q and k RMSNormAttnNode for sha pattern """
        self.q_rmsnorm_node = RMSNormAttnNode(
            pow_=[],
            reduce_mean=[],
            add=[],
            sqrt=[],
            div=[],
            mul=[],
            out_mul=[],
            )
        self.k_rmsnorm_node = RMSNormAttnNode(
            pow_=[],
            reduce_mean=[],
            add=[],
            sqrt=[],
            div=[],
            mul=[],
            out_mul=[],
            )

    def capture_rmsnorm(self, d_qkv: dict, qk_matmul_node: NodeProto):
        self.query_pattern_dict = defaultdict(dict)
        self.key_pattern_dict = defaultdict(dict)

        for dict_key in ["query", "key"]:
            q_or_k_conv = d_qkv[dict_key]["matmul_node"]
            try:
                mha_pattern_dict = getattr(self, f"{dict_key}_pattern_dict")
                rmsnorm_pow = get_next_node_down_based_on_cond(
                    q_or_k_conv,
                    self.mha2sha_optim.get_node_by_input_name,
                    node_found_cond=lambda n: n.op_type == "Pow",
                    node_end_search_cond=lambda n: n == qk_matmul_node,
                )
                mha_pattern_dict["pow_val"] = (
                    get_initializer_value(
                        self.mha2sha_optim.get_initializer_by_name[rmsnorm_pow.input[1]]
                    )
                )

                rmsnorm_reduce_mean = get_next_node_down_based_on_cond(
                    rmsnorm_pow,
                    self.mha2sha_optim.get_node_by_input_name,
                    node_found_cond=lambda n: n.op_type == "ReduceMean",
                    node_end_search_cond=lambda n: n == qk_matmul_node,
                )
                mha_pattern_dict["reduce_mean"] = {
                    "axes": rmsnorm_reduce_mean.attribute[0].ints,
                    "keepdims": rmsnorm_reduce_mean.attribute[1].i,
                }

                rmsnorm_add = get_next_node_down_based_on_cond(
                    rmsnorm_reduce_mean,
                    self.mha2sha_optim.get_node_by_input_name,
                    node_found_cond=lambda n: n.op_type == "Add",
                    node_end_search_cond=lambda n: n == qk_matmul_node,
                )
                mha_pattern_dict["add_val"] = (
                    get_initializer_value(
                        self.mha2sha_optim.get_initializer_by_name[rmsnorm_add.input[1]]
                    )
                )

                # We don't need really any info from this, it's more a gut check it's here
                rmsnorm_sqrt = get_next_node_down_based_on_cond(
                    rmsnorm_add,
                    self.mha2sha_optim.get_node_by_input_name,
                    node_found_cond=lambda n: n.op_type == "Sqrt",
                    node_end_search_cond=lambda n: n == qk_matmul_node,
                )

                rmsnorm_div = get_next_node_down_based_on_cond(
                    rmsnorm_sqrt,
                    self.mha2sha_optim.get_node_by_input_name,
                    node_found_cond=lambda n: n.op_type == "Div",
                    node_end_search_cond=lambda n: n == qk_matmul_node,
                )

                # Plamo was dividing by 1 - no idea why you would, ¯\_(ツ)_/¯
                mha_pattern_dict["div_val"] = (
                    get_initializer_value(
                        self.mha2sha_optim.get_initializer_by_name[rmsnorm_div.input[0]]
                    )
                )

                # Again not info needed and here as a gut check for the pattern
                rmsnorm_mul = get_next_node_down_based_on_cond(
                    rmsnorm_div,
                    self.mha2sha_optim.get_node_by_input_name,
                    node_found_cond=lambda n: n.op_type == "Mul",
                    node_end_search_cond=lambda n: n == qk_matmul_node,
                )

                rmsnorm_out_mul = get_next_node_down_based_on_cond(
                    rmsnorm_mul,
                    self.mha2sha_optim.get_node_by_input_name,
                    node_found_cond=lambda n: n.op_type == "Mul" and n != rmsnorm_mul,
                    node_end_search_cond=lambda n: n == qk_matmul_node,
                )

                mha_pattern_dict["out_mul_weight"] = (
                    get_initializer_value(
                        self.mha2sha_optim.get_initializer_by_name[
                            rmsnorm_out_mul.input[1]
                        ]
                    )
                )
                d_qkv[dict_key]["mha_rmsnorm_node"] = RMSNormAttnNode(
                    pow_=rmsnorm_pow,
                    reduce_mean=rmsnorm_reduce_mean,
                    add=rmsnorm_add,
                    sqrt=rmsnorm_sqrt,
                    div=rmsnorm_div,
                    mul=rmsnorm_mul,
                    out_mul=rmsnorm_out_mul,
                )

            except NodeNotFoundError:
                log_error("Unable to match RMSNorm pattern")
                exit(1)

    def update_sha_names(self,
                        branch_type,
                        rmsnorm_pow,
                        rmsnorm_reduce_mean,
                        rmsnorm_add,
                        rmsnorm_sqrt,
                        rmsnorm_div,
                        rmsnorm_mul,
                        rmsnorm_out_mul,
                        ):
        """
        Add new sha tensor names by appending to existing sha name list
        """

        if branch_type == BranchType.Q:
            sha_node = self.q_rmsnorm_node
        else:
            sha_node = self.k_rmsnorm_node

        sha_node.pow_.append(rmsnorm_pow)
        sha_node.reduce_mean.append(rmsnorm_reduce_mean)
        sha_node.add.append(rmsnorm_add)
        sha_node.sqrt.append(rmsnorm_sqrt)
        sha_node.div.append(rmsnorm_div)
        sha_node.mul.append(rmsnorm_mul)
        sha_node.out_mul.append(rmsnorm_out_mul)

    def create_sha_rmsnorm(
        self,
        qk_inp: NodeProto,
        branch_type: BranchType,
        head_num: int,
        head_dim: int
    ) -> List[NodeProto]:
        if branch_type == BranchType.Q:
            pattern_dict_key = "query"
        elif branch_type == BranchType.K:
            pattern_dict_key = "key"
        else:
            raise ValueError(f"Invalid branch type {branch_type}")

        qk_pattern_dict = getattr(self, f"{pattern_dict_key}_pattern_dict")

        first_transpose = self.mha2sha_optim._op_factory.get_transpose_op(
            qk_inp, perm=[0, 2, 3, 1]
        )
        self.mha2sha_optim.model.graph.node.append(first_transpose)

        reshape_node, reshape_init = self.mha2sha_optim._op_factory.get_reshape_op(
            first_transpose, shape=[1, -1, 1, head_dim]
        )
        self.mha2sha_optim.model.graph.node.append(reshape_node)
        self.mha2sha_optim.model.graph.initializer.extend(reshape_init)

        second_transpose = self.mha2sha_optim._op_factory.get_transpose_op(
            reshape_node, perm=[0, 2, 1, 3]
        )
        self.mha2sha_optim.model.graph.node.append(second_transpose)

        rmsnorm_pow, rmsnorm_pow_init = self.mha2sha_optim._op_factory.get_pow_op(
            second_transpose, qk_pattern_dict["pow_val"]
        )
        self.mha2sha_optim.model.graph.node.append(rmsnorm_pow)
        self.mha2sha_optim.model.graph.initializer.extend(rmsnorm_pow_init)

        rmsnorm_reduce_mean, _ = (
            self.mha2sha_optim._op_factory.get_reduce_mean_op(
                rmsnorm_pow,
                qk_pattern_dict["reduce_mean"]["axes"],
                qk_pattern_dict["reduce_mean"]["keepdims"],
            )
        )
        self.mha2sha_optim.model.graph.node.append(rmsnorm_reduce_mean)
        print("self.mha2sha_optim.tensor_name_set Before", self.mha2sha_optim.tensor_name_set)

        add_tensor_name, self.mha2sha_optim.tensor_name_set = create_tensor_name(
            "Add_constant", self.mha2sha_optim.tensor_name_set
        )
        print("self.mha2sha_optim.tensor_name_set After", self.mha2sha_optim.tensor_name_set)

        add_init = numpy_helper.from_array(
            qk_pattern_dict["add_val"],
            add_tensor_name,
        )
        rmsnorm_add = self.mha2sha_optim._op_factory.get_add_op(
            rmsnorm_reduce_mean,
            add_tensor_name
        )

        self.mha2sha_optim.model.graph.node.append(rmsnorm_add)
        self.mha2sha_optim.model.graph.initializer.extend([add_init])

        rmsnorm_sqrt, _ = self.mha2sha_optim._op_factory.get_sqrt_op(rmsnorm_add)
        self.mha2sha_optim.model.graph.node.append(rmsnorm_sqrt)

        rmsnorm_div, rmsnorm_div_init = self.mha2sha_optim._op_factory.get_div_op(
            rmsnorm_sqrt,
            qk_pattern_dict["div_val"],
            divide_by_value=False,
        )
        self.mha2sha_optim.model.graph.node.append(rmsnorm_div)
        self.mha2sha_optim.model.graph.initializer.extend(rmsnorm_div_init)

        rmsnorm_mul, _ = self.mha2sha_optim._op_factory.get_mul_op(
            rmsnorm_div, second_transpose
        )
        self.mha2sha_optim.model.graph.node.append(rmsnorm_mul)

        out_mul_weight = qk_pattern_dict["out_mul_weight"]
        # TODO: really needs to be if MQA
        if self.mha2sha_optim.gqa_model and pattern_dict_key == "key":
            out_mul_weight_for_head_num = out_mul_weight
        else:
            out_mul_weight_for_head_num = out_mul_weight[
                :, head_num : head_num + 1, ...
            ]

        # RMS Norm requires input 0 = weight and input 1 = activation
        rmsnorm_out_mul, rmsnorm_out_mul_init = (
            self.mha2sha_optim._op_factory.get_mul_op(
                out_mul_weight_for_head_num, rmsnorm_mul, propose_op_name="rmsnormMul"
            )
        )
        self.mha2sha_optim.model.graph.node.append(rmsnorm_out_mul)
        self.mha2sha_optim.model.graph.initializer.extend(rmsnorm_out_mul_init)

        self.update_sha_names(
            BranchType.Q if pattern_dict_key=='query' else BranchType.K,
            rmsnorm_pow,
            rmsnorm_reduce_mean,
            rmsnorm_add,
            rmsnorm_sqrt,
            rmsnorm_div,
            rmsnorm_mul,
            rmsnorm_out_mul,
        )
        return rmsnorm_out_mul