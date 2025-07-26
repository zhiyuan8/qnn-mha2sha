# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from functools import partial
from operator import is_not
from onnx import numpy_helper, helper
from onnx.onnx_pb import NodeProto, TensorProto

from mha2sha.utils.logger import log_debug, log_warning, log_info

from mha2sha.utils.onnx import (
    get_next_node_up_based_on_cond,
    get_next_node_down_based_on_cond,
    get_parent,
    get_children,
    get_slice_info
)
from mha2sha.utils.utils import BranchType, ExtensionTypes
from mha2sha.utils.encoding_mapper_utils import (
    create_activation_node_mapping_dict,
    update_sha_tensor_to_node_mapping_dict,
    NodeMappingDict
)

mha2sha_hf_model_optimizer = Any  # Causes circular import
ROPE_BRANCH_PREFIX = [ExtensionTypes.ROPE+"_"+qk for qk in ["q", "k"]]
ROPE_ACTIVATION_ENCODING_KEYS = [
    "Mul_rope_cos_1",
    "Mul_rope_sin_1",
    "Mul_rope_cos_2",
    "Mul_rope_sin_2",
    "rope_sub",
    "rope_add",
    "rope_concat",
    'rope_r3_conv',
]

@dataclass
class RopeNode:
    """ Defines nodes to recode for ROPE """
    slice_1: Optional[Union[NodeProto, List[NodeProto]]] = None
    slice_2: Optional[Union[NodeProto, List[NodeProto]]] = None
    Mul_rope_cos_1: Optional[Union[NodeProto, List[NodeProto]]] = None
    Mul_rope_sin_1: Optional[Union[NodeProto, List[NodeProto]]] = None
    Mul_rope_cos_2: Optional[Union[NodeProto, List[NodeProto]]] = None
    Mul_rope_sin_2: Optional[Union[NodeProto, List[NodeProto]]] = None
    rope_sub: Optional[Union[NodeProto, List[NodeProto]]] = None
    rope_add: Optional[Union[NodeProto, List[NodeProto]]] = None
    rope_concat: Optional[Union[NodeProto, List[NodeProto]]] = None
    rope_r3_conv: Optional[Union[NodeProto, List[NodeProto]]] = None


def create_branch_rope_encoding_mapping_dict(mha_rope_dict: RopeNode, prefix=None) -> Dict[str, NodeMappingDict]:
    """ Create Dict[ste, NodeMappingDict] for Rope. """
    return {
        f"{prefix}_Mul_rope_cos_1": create_activation_node_mapping_dict(
            mha_rope_dict.Mul_rope_cos_1.input[0],
            mha_rope_dict.Mul_rope_cos_1.input[1],
            mha_rope_dict.Mul_rope_cos_1.output[0]
        ),
        f"{prefix}_Mul_rope_sin_1": create_activation_node_mapping_dict(
            mha_rope_dict.Mul_rope_sin_1.input[0],
            mha_rope_dict.Mul_rope_sin_1.input[1],
            mha_rope_dict.Mul_rope_sin_1.output[0]
        ),
        f"{prefix}_Mul_rope_cos_2": create_activation_node_mapping_dict(
            mha_rope_dict.Mul_rope_cos_2.input[0],
            mha_rope_dict.Mul_rope_cos_2.input[1],
            mha_rope_dict.Mul_rope_cos_2.output[0]
        ),
        f"{prefix}_Mul_rope_sin_2": create_activation_node_mapping_dict(
            mha_rope_dict.Mul_rope_sin_2.input[0],
            mha_rope_dict.Mul_rope_sin_2.input[1],
            mha_rope_dict.Mul_rope_sin_2.output[0]
        ),
        f"{prefix}_rope_sub": create_activation_node_mapping_dict(
            mha_rope_dict.rope_sub.input[0],
            mha_rope_dict.rope_sub.input[1],
            mha_rope_dict.rope_sub.output[0]
        ),
        f"{prefix}_rope_add": create_activation_node_mapping_dict(
            mha_rope_dict.rope_add.input[0],
            mha_rope_dict.rope_add.input[1],
            mha_rope_dict.rope_add.output[0]
        ),
        f"{prefix}_rope_concat": create_activation_node_mapping_dict(
            mha_rope_dict.rope_concat.input[0],
            mha_rope_dict.rope_concat.input[1],
            mha_rope_dict.rope_concat.output[0]
        ),
        f"{prefix}_rope_r3_conv": create_activation_node_mapping_dict(
            mha_rope_dict.rope_r3_conv.input[0] if mha_rope_dict.rope_r3_conv is not None else None,
            mha_rope_dict.rope_r3_conv.input[1] if mha_rope_dict.rope_r3_conv is not None else None,
            mha_rope_dict.rope_r3_conv.output[0] if mha_rope_dict.rope_r3_conv is not None else None
        ),
    }


def create_rope_encoding_mapping_dict(
    mha_rope_dict_list: List[RopeNode],
) -> Dict[str, NodeMappingDict]:
    """
    Create a dictionary for EncodingMapper.
    :param mha_rope_dict_list: Collected MHA rope input output tensor names from ROPE extension optimizer. In sequence of q -> k.
    :return: dictionary of key to NodeMappingDict. keys are in ROPE_ACTIVATION_ENCODING_KEYS.
    """
    # mha_rope_dict can be None when MHA ROPE pattern match fail.
    rope_dict = {}
    for mha_rope_dict, prefix in zip(mha_rope_dict_list, ROPE_BRANCH_PREFIX):
        if mha_rope_dict:
            _rope_dict = create_branch_rope_encoding_mapping_dict(mha_rope_dict, prefix)
            rope_dict.update(_rope_dict)
    return rope_dict

def update_rope_sha_encoding_name_to_rope_encoding_mapping_dict(
        rope_encoding_mapping_dict,
        q_sha_rope_node,
        k_sha_rope_node
    ):
    """
    Update sha ROPE encoding names to EncodingMappingDict.
    :param rope_encoding_mapping_dict: encoding_mapping_dict.rope
    :param q_sha_rope_node: List of sha RopeNode
    :param k_sha_rope_node: List of sha RopeNode
    """
    for rope_branch_prefix, sha_rope_node in zip(ROPE_BRANCH_PREFIX, [q_sha_rope_node, k_sha_rope_node]):
        for rope_node_name in ROPE_ACTIVATION_ENCODING_KEYS:
            if (ROPE_BRANCH_PREFIX[1] == rope_branch_prefix) and (f"{rope_branch_prefix}_{rope_node_name}" not in rope_encoding_mapping_dict):
                log_debug(f"Missing {rope_branch_prefix}_{rope_node_name} in RopeNode. Expecting BERT model")
                continue

            update_sha_tensor_to_node_mapping_dict(
                rope_encoding_mapping_dict[f"{rope_branch_prefix}_{rope_node_name}"],
                getattr(sha_rope_node, rope_node_name),
            )

class RopeExtension:
    """ Extenstion helpers for mha2sha_optimzer to bridge Morpheus pipeline code base and v1.0.0 release.  """
    def __init__(self, mha2sha_optim: mha2sha_hf_model_optimizer) -> None:
        """Initalizes an instance based on the MHA2SHA optimizer provided.

        Args:
            mha2sha_optim:
                MHA2SHAOptimizer instance holding the model loader and model info.
        """
        self.mha2sha_optim = mha2sha_optim
        self.map_rope_encoding = True
        self.interleave_rope = None

    def reset_sha_encoding_name_list(self):
        """
        Reset mha sha names for ROPE tensors.
        """
        self.q_sha_rope_node = RopeNode(
            slice_1=[],
            slice_2=[],
            Mul_rope_cos_1=[],
            Mul_rope_sin_1=[],
            Mul_rope_cos_2=[],
            Mul_rope_sin_2=[],
            rope_sub=[],
            rope_add=[],
            rope_concat=[],
            rope_r3_conv=[],
        )
        self.k_sha_rope_node = RopeNode(
            slice_1=[],
            slice_2=[],
            Mul_rope_cos_1=[],
            Mul_rope_sin_1=[],
            Mul_rope_cos_2=[],
            Mul_rope_sin_2=[],
            rope_sub=[],
            rope_add=[],
            rope_concat=[],
            rope_r3_conv=[],
        )

    def update_sha_names(self,
                        branch_type,
                        _1st_cos_mul,
                        _1st_sin_mul,
                        _2nd_cos_mul,
                        _2nd_sin_mul,
                        embed_1st_sub,
                        embed_2nd_add,
                        inp_concat_node,
                        r3_conv=None):
        """
        Add new sha tensor names in ROPE by appending to existing sha name list
        """

        if branch_type == BranchType.Q:
            sha_rope_node = self.q_sha_rope_node
        else:
            sha_rope_node = self.k_sha_rope_node

        sha_rope_node.Mul_rope_cos_1.append(_1st_cos_mul)
        sha_rope_node.Mul_rope_sin_1.append(_1st_sin_mul)
        sha_rope_node.Mul_rope_cos_2.append(_2nd_cos_mul)
        sha_rope_node.Mul_rope_sin_2.append(_2nd_sin_mul)
        sha_rope_node.rope_sub.append(embed_1st_sub)
        sha_rope_node.rope_add.append(embed_2nd_add)
        sha_rope_node.rope_concat.append(inp_concat_node)
        if r3_conv is not None:
            sha_rope_node.rope_r3_conv.append(r3_conv)

    def create_llama_rope_node(self,
                               inp: NodeProto,
                               rope_cos: NodeProto,
                               rope_sin: NodeProto,
                               head_dim: int,
                               branch_type: BranchType,
                               info_dict):
        """
        Implement prepared llama's ROPE pattern. Mathmetically equivilent to get_llama_rope_node.
        :param inp: query or key input, shape [1, 1, seq_len, head_dim] or [1, head_dim, 1, seq_len] if NCHW aligned
        :param rope_cos: cos values, shape [1, 1, seq_len, head_dim//2]
        :param rope_sin: sin values, shape [1, 1, seq_len, head_dim//2]
        :param head_dim: dim for each head
        :return inp_concat_node: rope embed node = [1st_half*cos - 2nd_half*sin, 1st_half*sin + 2nd_half*cos]
                                 shape: [1, 1, seq_len, head_dim]
        Create rope nodes to apply rope embeddings for llama:
                     ------     inp    --------
                    /                          \
                Slice_1st                  Slice_2nd
               /        \                  /          \
        Mul rope_cos  Mul rope_sin    Mul rope_cos  Mul rope_sin
              |           \               /         /
             Sub ----------\-------------/---------/
              |             \---Add ----/
              |                  |
             Concat ------------/
              |
        rope_embed_query_inp
        """
        assert branch_type in (branch_type.Q, branch_type.K)

        if self.mha2sha_optim.nchw_aligned:
            axis = 3 if self.mha2sha_optim.handle_internal_rmsnorm else 1
            if self.interleave_rope:
                _1st_half_slice, _1st_slice_init_list = self.mha2sha_optim._op_factory.get_slice_op(inp, start=0, end=self.interleave_rope_end, axis=axis, steps=2)
                _2nd_half_slice, _2nd_slice_init_list = self.mha2sha_optim._op_factory.get_slice_op(inp, start=1, end=self.interleave_rope_end, axis=axis, steps=2)
            else:
                _1st_half_slice, _1st_slice_init_list = self.mha2sha_optim._op_factory.get_slice_op(inp, start=0, end=head_dim//2, axis=axis)
                _2nd_half_slice, _2nd_slice_init_list = self.mha2sha_optim._op_factory.get_slice_op(inp, start=head_dim//2, end=head_dim, axis=axis)

            self.mha2sha_optim.model.graph.initializer.extend(_1st_slice_init_list + _2nd_slice_init_list)
            self.mha2sha_optim.model.graph.node.extend([_1st_half_slice, _2nd_half_slice])

            if not self.mha2sha_optim.handle_internal_rmsnorm:
                _1st_half_transpose = self.mha2sha_optim._op_factory.get_transpose_op(_1st_half_slice, [0, 2, 3, 1])
                _2nd_half_transpose = self.mha2sha_optim._op_factory.get_transpose_op(_2nd_half_slice, [0, 2, 3, 1])
                self.mha2sha_optim.model.graph.node.extend([_1st_half_transpose, _2nd_half_transpose])
                _1st_half_slice = _1st_half_transpose
                _2nd_half_slice = _2nd_half_transpose
        else:
            if self.interleave_rope:
                _1st_half_slice, _1st_slice_init_list = self.mha2sha_optim._op_factory.get_slice_op(inp, start=0, end=self.interleave_rope_end, axis=axis, steps=2)
                _2nd_half_slice, _2nd_slice_init_list = self.mha2sha_optim._op_factory.get_slice_op(inp, start=1, end=self.interleave_rope_end, axis=axis, steps=2)
            else:
                _1st_half_slice, _1st_slice_init_list = self.mha2sha_optim._op_factory.get_slice_op(inp, start=0, end=head_dim//2, axis=3)
                _2nd_half_slice, _2nd_slice_init_list = self.mha2sha_optim._op_factory.get_slice_op(inp, start=head_dim//2, end=head_dim, axis=3)

            self.mha2sha_optim.model.graph.initializer.extend(_1st_slice_init_list+_2nd_slice_init_list)
            self.mha2sha_optim.model.graph.node.extend([_1st_half_slice, _2nd_half_slice])

        _1st_cos_mul = self.mha2sha_optim._op_factory.get_element_mul_op(_1st_half_slice, rope_cos)
        _2nd_sin_mul = self.mha2sha_optim._op_factory.get_element_mul_op(_2nd_half_slice, rope_sin)
        embed_1st_sub = self.mha2sha_optim._op_factory.get_sub_op(_1st_cos_mul, _2nd_sin_mul)

        # Embedded 2nd half: 1st_half*sin + 2nd_half*cos
        _1st_sin_mul = self.mha2sha_optim._op_factory.get_element_mul_op(_1st_half_slice, rope_sin)
        _2nd_cos_mul = self.mha2sha_optim._op_factory.get_element_mul_op(_2nd_half_slice, rope_cos)
        embed_2nd_add = self.mha2sha_optim._op_factory.get_add_op(_1st_sin_mul, _2nd_cos_mul)

        inp_concat_node = self.mha2sha_optim._op_factory.get_concat_op([embed_1st_sub, embed_2nd_add], 3)

        self.mha2sha_optim.model.graph.node.extend([ _1st_cos_mul, _2nd_sin_mul, embed_1st_sub,\
                                                                   _1st_sin_mul, _2nd_cos_mul, embed_2nd_add, inp_concat_node])

        if self.mha2sha_optim.handle_r3_matrix:
            # 1) First Transpose: change shape from [1, 1, seq_len, head_dim] to [1, 1, head_dim, seq_len]
            r3_transpose_1 = self.mha2sha_optim._op_factory.get_transpose_op(
                inp_concat_node,
                perm=[0, 3, 2, 1],
            )
            self.mha2sha_optim.model.graph.node.append(r3_transpose_1)

            # 2) Obtain the shared Rope R3 Conv weight directly from MHA by reusing it
            mha_rope_node = info_dict[f"{branch_type.name.lower()}_rope_mha_node"]
            r3_conv_weight_name = mha_rope_node.rope_r3_conv.input[1]

            # 3) Create the R3_Conv node using the shared weight
            r3_conv = self.mha2sha_optim._op_factory.get_conv_op(
                input_node=r3_transpose_1.output[0],
                weight_tensor_name=r3_conv_weight_name,
                bias_tensor_name=None,
                kernel_shape=[1, 1],
                padding=[0, 0, 0, 0],
                strides=[1, 1],
            )
            self.mha2sha_optim.model.graph.node.append(r3_conv)

            # 4) Second Transpose: revert shape back to [1, 1, seq_len, head_dim]
            r3_transpose_2 = self.mha2sha_optim._op_factory.get_transpose_op(
                r3_conv.output[0],
                perm=[0, 3, 2, 1],
            )
            self.mha2sha_optim.model.graph.node.append(r3_transpose_2)

            # Optionally update encoding mapping with new nodes:
            if self.map_rope_encoding:
                self.update_sha_names(
                    branch_type,
                    _1st_cos_mul,
                    _1st_sin_mul,
                    _2nd_cos_mul,
                    _2nd_sin_mul,
                    embed_1st_sub,
                    embed_2nd_add,
                    inp_concat_node,
                    r3_conv=r3_conv,
                )
            return r3_transpose_2
        else:
            # Update sha_encodings_names
            if self.map_rope_encoding:
                self.update_sha_names(
                    branch_type,
                    _1st_cos_mul,
                    _1st_sin_mul,
                    _2nd_cos_mul,
                    _2nd_sin_mul,
                    embed_1st_sub,
                    embed_2nd_add,
                    inp_concat_node
                )

            return inp_concat_node

    def get_llama_rope_node_without_positional_embedding(self,
                            query_inp: NodeProto,
                            key_inp: NodeProto,
                            rope_cos: NodeProto,
                            rope_sin: NodeProto,
                            head_dim: int):
        """
        if rope_cos shape is [1, 1,  seq_len, head_dim//2], concat cos to itself -> [rope_cos, rope_cos]
        :param query_inp: input node from query branch, shape [1, 1, seq_len, head_dim]
        :param key_inp: input node from key branch, shape [1, 1, seq_len, head_dim]
        :param rope_cos: pre-computed cos values, shape [1, 1,  seq_len, head_dim] or [1, 1,  seq_len, head_dim//2]
        :param rope_sin :pre-computed sin values, shape [1, 1,  seq_len, head_dim] or [1, 1,  seq_len, head_dim//2]
        :param head_dim: dim for each head
        :return rope_embed_query_inp: rope embed node for query branch
        :return rope_embed_key_inp: rope embed node for key branch

        Create rope nodes to apply rope embeddings for llama:
        rope_cos- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            |      query_inp           rope_sin               key_inp          |
            |    /     |      \            |             /       |       \     |
            |   |   Slice_2nd Slice_1st    |        Slice_2nd  Slice_1st  |    |
            |   |      |       |           |            |        |        |    |
            |   |   Negtive    |           |         Negtive     |        |    |
            |   |      \      /            |             \       /        |    |
            |   |       Concat             |               Concat         |    |
            |  /          |               / \                 |           |   /
            Mul         Mul--------------      ------------- Mul          Mul
             |           /                                    |           /
            Add - - - -                                      Add - - - - -
             |                                                |
            rope_embed_query_inp                             rope_embed_key_inp
        """
        # If rope_cos/rope_sin shape is [1, 1,  seq_len, head_dim//2], concat cos/sin to itself -> [rope_cos, rope_cos]

        # Prepare model rope cos and rope sin are model input with shape [1, 1,  seq_len, head_dim//2]
        if self.mha2sha_optim.use_position_embedding:
            input_cos_tensor = self.mha2sha_optim.model.graph.input[self.mha2sha_optim.mha_model_input_names_index_dict[rope_cos]]
            if input_cos_tensor.type.tensor_type.shape.dim[-1].dim_value == head_dim//2:
                cos_concat_node = self.mha2sha_optim._op_factory.get_concat_op([rope_cos, rope_cos], -1)
            else:
                raise ValueError(f"Expecting prerared llama cos have dim: {head_dim//2} on it's last dim but got {[input_cos_tensor.type.tensor_type.shape.dim[-1].dim_value]}")

            input_sin_tensor = self.mha2sha_optim.model.graph.input[self.mha2sha_optim.mha_model_input_names_index_dict[rope_cos]]
            if input_sin_tensor.type.tensor_type.shape.dim[-1].dim_value == head_dim//2:
                sin_concat_node = self.mha2sha_optim._op_factory.get_concat_op([rope_sin, rope_sin], -1)
            else:
                raise ValueError(f"Expecting prerared llama sin have dim: {head_dim//2} on it's last dim but got {[input_sin_tensor.type.tensor_type.shape.dim[-1].dim_value]}")

            self.mha2sha_optim.model.graph.node.extend([cos_concat_node, sin_concat_node])
            rope_cos = cos_concat_node
            rope_sin = sin_concat_node

        # Rope on query
        q_1st_half, q_1st_slice_init_list = self.mha2sha_optim._op_factory.get_slice_op(query_inp, start=0, end=head_dim//2, axis=3)
        q_2nd_half, q_2nd_slice_init_list = self.mha2sha_optim._op_factory.get_slice_op(query_inp, start=head_dim//2, end=head_dim, axis=3)
        q_2nd_half_neg = self.mha2sha_optim._op_factory.get_neg_op(q_2nd_half)
        q_concat_node = self.mha2sha_optim._op_factory.get_concat_op([q_2nd_half_neg, q_1st_half], -1)
        q_cos_mul = self.mha2sha_optim._op_factory.get_element_mul_op(query_inp, rope_cos)
        q_sin_mul = self.mha2sha_optim._op_factory.get_element_mul_op(q_concat_node, rope_sin)
        rope_embed_query_inp = self.mha2sha_optim._op_factory.get_add_op(q_cos_mul, q_sin_mul)

        self.mha2sha_optim.model.graph.initializer.extend(q_1st_slice_init_list)
        self.mha2sha_optim.model.graph.initializer.extend(q_2nd_slice_init_list)
        self.mha2sha_optim.model.graph.node.extend([q_1st_half, q_2nd_half, q_2nd_half_neg, q_concat_node, q_cos_mul, q_sin_mul, rope_embed_query_inp])

        # Rope on keys
        k_1st_half, k_1st_slice_init_list = self.mha2sha_optim._op_factory.get_slice_op(key_inp, start=0, end=head_dim//2, axis=3)
        k_2nd_half, k_2nd_slice_init_list = self.mha2sha_optim._op_factory.get_slice_op(key_inp, start=head_dim//2, end=head_dim, axis=3)
        k_2nd_half_neg = self.mha2sha_optim._op_factory.get_neg_op(k_2nd_half)
        k_concat_node = self.mha2sha_optim._op_factory.get_concat_op([k_2nd_half_neg, k_1st_half], -1)
        k_cos_mul = self.mha2sha_optim._op_factory.get_element_mul_op(key_inp, rope_cos)
        k_sin_mul = self.mha2sha_optim._op_factory.get_element_mul_op(k_concat_node, rope_sin)
        rope_embed_key_inp = self.mha2sha_optim._op_factory.get_add_op(k_cos_mul, k_sin_mul)

        self.mha2sha_optim.model.graph.initializer.extend(k_1st_slice_init_list)
        self.mha2sha_optim.model.graph.initializer.extend(k_2nd_slice_init_list)
        self.mha2sha_optim.model.graph.node.extend([k_1st_half, k_2nd_half, k_2nd_half_neg, k_concat_node, k_cos_mul, k_sin_mul, rope_embed_key_inp])

        return rope_embed_query_inp, rope_embed_key_inp


    def get_position_ids(self, qk_matmul_node: NodeProto) -> List[NodeProto]:
        """
        Searches the QK MatMul node to find the RoPE's position ids

        :param qk_matmul_node: NodeProto - QK MatMul node to search form.
        :return: List[NodeProto, NodeProto] - The cos and sin postion ids
        """

        input_position_ids_not_implemented = NotImplementedError(
            f"Adding position ids through inputs is currently not supported."
        )

        if self.mha2sha_optim.position_ids:
            raise input_position_ids_not_implemented

        # The position cos and sin go to each branch, we only need to search up one.
        _, q_or_k_parent = get_parent(qk_matmul_node, self.mha2sha_optim.get_node_by_output_name)

        position_ids = None

        # First we check if the keys exist as LLaMA tends to keep these names.
        # TODO: Need to figure out what to handle with no Node returned
        position_keys = ["position_ids_cos", "position_ids_sin"]

        if set(position_keys) <= self.mha2sha_optim.mha_model_input_names_index_dict.keys():
            self.mha2sha_optim.use_position_embedding = True
            return position_keys
        # else:
        #     raise ValueError("Expect positional keys are model input with input name: 'position_ids_cos', 'position_ids_sin' when model is prepared.")

        if set(position_keys) <= self.mha2sha_optim.get_node_by_input_name.keys():
            position_ids = [self.mha2sha_optim.get_node_by_input_name[k] for k in position_keys]

        # Second we search for position ids with cos and sin in them.
        if not position_ids:
            position_ids = list(filter(partial(is_not, None), [
                get_next_node_up_based_on_cond(
                    q_or_k_parent,
                    self.mha2sha_optim.get_node_by_output_name,
                    node_found_cond=lambda node: trig in node.name,
                    node_end_search_cond=lambda node: node.op_type == "MatMul"  # End search if we hit a MatMul (Q or K)
                )
                for trig in ("cos", "sin")
            ]))

        # Special case for models with a different implementation.
        if not position_ids:
            log_debug("No position ids with names cos and sin found. Searching for rotary names instead.")

            # If the sin and cos are not found, we may have model based on the HuggingFace baseline which instead
            # has unsqueeze operations with the names `rotary`. The rotary without a `#` is considered cos and the
            # other sin. The implementaiton can be found here:
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L184-L208
            position_ids =  [
                get_next_node_up_based_on_cond(
                    q_or_k_parent,
                    self.mha2sha_optim.get_node_by_output_name,
                    node_found_cond=lambda node: rotary in node.name and node.op_type == "Unsqueeze",
                    node_end_search_cond=lambda node: node.op_type == "MatMul"  # End search if we hit a MatMul (Q or K)
                )
                for rotary in ("rotary_emb", "rotary_emb#")
            ]

        assert position_ids is not None, "Unable to find position ids."
        assert len(position_ids) == 2, f"Expected only two position ids, Got -> {len(position_ids)}"
        position_id_types = [isinstance(p, NodeProto) for p in position_ids]
        assert all(position_id_types), f"Expected position ids to be NodeProto. Got -> {position_id_types=}"
        return position_ids


    def get_ropes_gather_output(self, rope_trig_node: NodeProto) -> str:
        """
        Searches for a specific Gather along the RoPE (cos/sin) branch that has 2 inputs and one outputs.

                              Other Gather
                                   |
                                   v
                               +--------+
                               | Gather |<---- Position Ids (typically)
                               +--------+
                                   |
                                   |
                                   v

        :param rope_trig_node: NodeProto - The cos or sin node to search down to the gather.
        :return: NodeProto - The Gather input node.
        """

        def is_gather(node: NodeProto):
            nodes_children = get_children(node, self.mha2sha_optim.get_node_by_input_name)
            return node.op_type == "Gather" and len(nodes_children) == 1 and nodes_children[0].op_type == "Unsqueeze"

        node = get_next_node_down_based_on_cond(rope_trig_node, self.mha2sha_optim.get_node_by_input_name, node_found_cond=is_gather)
        assert len(node.output) == 1, f"Expected to find 1 output. Instead got {node.output=}"
        return node.output[0]  # Only one output


    def get_mha_rope_nodes(self, linear_node, qk_matmul_node) -> Optional[RopeNode]:
        """
        :param linear_node: q or k linear node
        :paeam qk_matmul_node: qk matmul node
        :return mha_rope_node_dict:

        Capture rope patten from linear_node's next slice to next concat before qk_matmul_node.
        In specific, it looks into this pattern:
                                linear_node
                                     |
                         -----------/ \---------------
                        /                             \
                  Slice_1st                          Slice_2nd
               /            \                       /          \
        Mul_rope_cos_1    Mul_rope_sin_1     Mul_rope_cos_2 Mul_rope_sin_2
              |               \                  /            /
             Sub --------------\----------------/------------
              |                 \              /
              |                   -----------Add
              |                               |
             Concat --------------------------
              |
        rope_embed_inp
        """
        try:
            next_slice = get_next_node_down_based_on_cond(
                linear_node,
                self.mha2sha_optim.get_node_by_input_name,
                node_found_cond=lambda n: n.op_type == "Slice",
                node_end_search_cond=lambda n: n == qk_matmul_node
            )
            end_concat = get_next_node_down_based_on_cond(
                linear_node,
                self.mha2sha_optim.get_node_by_input_name,
                node_found_cond=lambda n: n.op_type == "Concat",
                node_end_search_cond=lambda n: n == qk_matmul_node
            )

            slice_list = self.mha2sha_optim.get_node_by_input_name[next_slice.input[0]]
            assert len(slice_list) == 2
            temp_slice_1, temp_slice_2 = slice_list
            # check which temp slice 1 and temp slice 2 are actual slice_1 and slice_2
            slice_info = get_slice_info(temp_slice_1,
                                        self.mha2sha_optim.get_node_by_output_name,
                                        self.mha2sha_optim.get_initializer_by_name)
            start = slice_info.starts
            if start == 0:
                slice_1, slice_2 = temp_slice_1, temp_slice_2
            else:
                slice_2, slice_1 = temp_slice_1, temp_slice_2

            if not self.interleave_rope and slice_info.steps == 2:
                # self.interleave_rope is reset at optimizer.get_attention_subgraph_info right before get_mha_rope_nodes.
                self.interleave_rope = True
                self.interleave_rope_end = slice_info.ends
                log_info("Use interleave slice on RoPE")

            # get Mul_rope_cos_1 and Mul_rope_sin_1
            temp_Mul_rope_cos_1, temp_Mul_rope_sin_1 = self.mha2sha_optim.get_node_by_input_name[slice_1.output[0]]
            if "cos" in temp_Mul_rope_cos_1.input[0] or "cos" in temp_Mul_rope_cos_1.input[1]:
                Mul_rope_cos_1, Mul_rope_sin_1 = temp_Mul_rope_cos_1, temp_Mul_rope_sin_1
            elif "cos" in temp_Mul_rope_sin_1.input[0] or "cos" in temp_Mul_rope_sin_1.input[1]:
                Mul_rope_sin_1, Mul_rope_cos_1 = temp_Mul_rope_cos_1, temp_Mul_rope_sin_1
            else:
                raise ValueError(f"Except cos/sin input to 1st slice Rope Mul, but get {temp_Mul_rope_sin_1.input}, {temp_Mul_rope_cos_1.input}")

            # get Mul_rope_cos_2 Mul_rope_sin_2
            temp_Mul_rope_cos_2, temp_Mul_rope_sin_2 = self.mha2sha_optim.get_node_by_input_name[slice_2.output[0]]
            if "cos" in temp_Mul_rope_cos_2.input[0] or "cos" in temp_Mul_rope_cos_2.input[1]:
                Mul_rope_cos_2, Mul_rope_sin_2 = temp_Mul_rope_cos_2, temp_Mul_rope_sin_2
            elif "cos" in temp_Mul_rope_sin_2.input[0] or "cos" in temp_Mul_rope_sin_2.input[1]:
                Mul_rope_sin_2, Mul_rope_cos_2 = temp_Mul_rope_cos_2, temp_Mul_rope_sin_2
            else:
                raise ValueError(f"Except cos/sin input to 2nd slice Rope Mul, but get {temp_Mul_rope_cos_2.input}, {temp_Mul_rope_sin_2.input}")

            rope_sub = self.mha2sha_optim.get_node_by_input_name[Mul_rope_cos_1.output[0]][0]
            rope_add = self.mha2sha_optim.get_node_by_input_name[Mul_rope_cos_2.output[0]][0]

            rope_concat = self.mha2sha_optim.get_node_by_input_name[rope_sub.output[0]][0]

            # Sanity to check we are capturing the required/correct pattern
            assert rope_sub == self.mha2sha_optim.get_node_by_input_name[Mul_rope_sin_2.output[0]][0], "Failed to get rope_sub"
            assert rope_sub.op_type == "Sub", f"rope_sub got op type {rope_sub.op_type}"
            assert rope_add == self.mha2sha_optim.get_node_by_input_name[Mul_rope_sin_1.output[0]][0], "Failed to get rope_add"
            assert rope_add.op_type == "Add", f"rope_add got op type {rope_add.op_type}"
            assert rope_concat == self.mha2sha_optim.get_node_by_input_name[rope_add.output[0]][0], "Failed to get rope_concat"
            assert rope_concat.op_type == "Concat", f"rope_concat got op type {rope_concat.op_type}"
            assert end_concat == rope_concat, "Rope end node is not rope concat"
            assert rope_concat.input[0] == rope_sub.output[0], f"Rope concat input 1 is not rope_sub, got {rope_concat.input[0]}"
            assert rope_concat.input[1] == rope_add.output[0], f"Rope concat input 2 is not rope_add, got {rope_concat.input[1]}"
            mha_rope_nodes = RopeNode(
                                slice_1=slice_1,
                                slice_2=slice_2,
                                Mul_rope_cos_1=Mul_rope_cos_1,
                                Mul_rope_sin_1=Mul_rope_sin_1,
                                Mul_rope_cos_2=Mul_rope_cos_2,
                                Mul_rope_sin_2=Mul_rope_sin_2,
                                rope_sub=rope_sub,
                                rope_add=rope_add,
                                rope_concat=rope_concat,
                                rope_r3_conv=None,
                            )
            if self.mha2sha_optim.handle_r3_matrix:
                try:
                    r3_transpose_1 = get_next_node_down_based_on_cond(
                        rope_concat,
                        self.mha2sha_optim.get_node_by_input_name,
                        node_found_cond=lambda n: n.op_type == "Transpose",
                        node_end_search_cond=lambda n: n == qk_matmul_node
                    )
                    r3_conv = get_next_node_down_based_on_cond(
                        r3_transpose_1,
                        self.mha2sha_optim.get_node_by_input_name,
                        node_found_cond=lambda n: n.op_type == "Conv",
                        node_end_search_cond=lambda n: n == qk_matmul_node
                    )
                    r3_transpose_2 = get_next_node_down_based_on_cond(
                        r3_conv,
                        self.mha2sha_optim.get_node_by_input_name,
                        node_found_cond=lambda n: n.op_type == "Transpose",
                        node_end_search_cond=lambda n: n == qk_matmul_node
                    )
                    mha_rope_nodes.rope_r3_conv = r3_conv
                except Exception as new_e:
                    log_info(f"R3 ROPE pattern (Transpose->Conv->Transpose) not detected: {new_e}")
            self.map_rope_encoding = True
            return mha_rope_nodes
        except Exception as e:
            log_warning(f"Failed to verify MHA ROPE patten with expected pattern with exception \"{e}\". Will NOT map encodings in ROPE for {linear_node.name}.")
            if self.mha2sha_optim.strict:
                log_warning(f"Fail script MHA2SHA conversion because MHA in ROPE doesn't match the golden ROPE we support.\
Set --strict to False to continue MHA2SHA without ROPE encoding mapping. See readme for more about golden ROPE.")
                exit(-1)

            self.map_rope_encoding = False
        return None


    def create_sha_conv_with_rope(
            self,
            info_dict,
            ns,
            head_num,
            head_dim,
            query_matmul_inp,
            key_matmul_inp,
            value_matmul_inp,
            sha_encoding_name_dict,
            **extension_kwargs):

        return self.mha2sha_optim._mha_conv_extension.create_sha_conv_with_rope(
                    info_dict,
                    ns,
                    head_num,
                    head_dim,
                    query_matmul_inp,
                    key_matmul_inp,
                    value_matmul_inp,
                    sha_encoding_name_dict,
                    **extension_kwargs)

    def create_sha_with_rope(
            self,
            info_dict,
            ns,
            head_num,
            head_dim,
            query_matmul_inp,
            key_matmul_inp,
            value_matmul_inp,
            sha_encoding_name_dict,
            extension_kwargs):

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
