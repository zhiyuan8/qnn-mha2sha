# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import onnx
from onnx.onnx_pb import ModelProto, NodeProto, TensorProto
from onnx import numpy_helper, helper


from mha2sha.utils.clean import clean_model, topological_sort

from mha2sha.utils.onnx import (
    NodeNotFoundError,
    get_constant_node_value,
    get_initializer_value,
    get_model_size,
    get_next_node_up_based_on_cond,
    get_next_node_down_based_on_cond,
    get_node_input_constant_op_value,
    get_parent,
    get_parent_at_any_level,
    get_shape_from_value_info_proto,
    native_checker
)

from mha2sha.utils.op_factory import OpFactory, create_tensor_name
from mha2sha.utils.base_attn_encoding_mapper import (
    BaseAttnNode,
    update_base_attn_sha_encoding_name_to_base_attn_encoding_mapping_dict,
)
from mha2sha.utils.utils import sha_node_name_basis, BranchType, update_all_mapping_dicts
from mha2sha.utils.logger import log_assert, log_debug, log_error, log_info, log_warning
from mha2sha.htp_optimizations.linear_to_conv import LinearToConv

from mha2sha.optimizer_extension.alibi_extension import AlibiExtension
from mha2sha.optimizer_extension.rmsnorm_extension import (
    RmsnormExtension,
    update_rmsnorm_sha_encoding_name_to_rmsnorm_encoding_mapping_dict
)
from mha2sha.optimizer_extension.rope_extension import (
    RopeExtension,
    update_rope_sha_encoding_name_to_rope_encoding_mapping_dict,
)
from mha2sha.optimizer_extension.past_key_value_extension import (
    PastKeyValueExtension,
    PastKeyValueConcat,
    update_past_key_value_sha_encoding_name_to_encoding_mapping_dict,
)
from mha2sha.optimizer_extension.mha_conv_extension import MhaConvExtension
from mha2sha.optimizer_extension.lora_extension import (
    LoraExtension,
    update_lora_sha_encoding_name_to_lora_encoding_mapping_dict,
    LoraAdaptorSetInfo,
    LoraVersion,
    get_lora_version,
)
from mha2sha.optimizer_extension.gqa_extension import GqaExtension
from mha2sha.optimizer_extension.anchor_network_extension import (
    AnchorExtension,
    update_anchor_sha_encoding_name_to_anchor_encoding_mapping_dict,
    create_anchor_network_encoding_mapping_dict
)
from mha2sha.encoding_mapper import MHA2SHAEncodingMapper, get_encoding_mapping_dict
from mha2sha.transformations.ar_builder import ArBuilder


class MHA2SHAOptimizer:
    """
    MHA2SHAOptimizer class is responsible for:
    1. calling the pattern matcher.
    2. gathering the attention information (initializers, subgraph structure, etc.)
    3. split multi-head attention to single-head attention.
    4. addition and removal of nodes.
    """

    def __init__(
        self,
        model: ModelProto,
        pattern: List,
        pattern_start_node_names: List,
        pattern_end_node_names: List,
        handle_rope_ops: bool,
        handle_past_key_value: bool,
        prepared_model: bool,
        replace_linear_with_conv: bool,
        position_ids: Optional[str],
        mha_conv: bool,
        nchw_aligned: bool,
        handle_r3_matrix: bool,
        strict: bool,
        lora_model: bool,
        lora_alpha_from_input: bool,
        llm_model: bool,
        gqa_model: bool,
        ar_value: Optional[int],
        handle_alibi: bool,
        handle_internal_rmsnorm: bool,
        lora_adaptor_set_info_dict: Dict[int, LoraAdaptorSetInfo],
        ar_num: int,
    ):
        """Initialization"""

        self.gqa_model = gqa_model
        self.handle_past_key_value = handle_past_key_value
        self.handle_rope_ops = handle_rope_ops
        self.llm_model = llm_model
        self.lora_alpha_from_input = lora_alpha_from_input
        self.lora_model = lora_model
        self.mha_conv = mha_conv
        self.model = model
        self.nchw_aligned = nchw_aligned
        self.handle_r3_matrix = handle_r3_matrix
        self.pattern = pattern
        self.pattern_end_node_names = pattern_end_node_names
        self.pattern_start_node_names = pattern_start_node_names
        self.position_ids = position_ids
        self.prepared_model = prepared_model
        self.replace_linear_with_conv = replace_linear_with_conv
        self.strict = strict
        self.handle_alibi = handle_alibi
        self.handle_internal_rmsnorm = handle_internal_rmsnorm
        self.lora_adaptor_set_info_dict = lora_adaptor_set_info_dict
        self.ar_num = ar_num


        self._run_sanity_checks()

        self.kv_cache = False
        self.use_position_embedding = False
        self.return_new_key_value_only = True
        self.anchor_network = False

        self.lora_version = None

        self.mha_sha_encoding_mapping_dict = {}

        self._update_all_mapping_dicts()  # Initial mapping

        if not self.ar_num:
            # Auto detect seq_len when ar-num is not specified.
            # LLM Models input[0] have shape [1, seq_len, vector_dim]
            #           output[0] have shape [1, seq_len, vocab_size]
            if self.llm_model:
                self.seq_len = self.model.graph.output[0].type.tensor_type.shape.dim[1].dim_value
            # LVM Models have shape [1, C, H, W], seq_len for LVM are H*W
            else:
                H = self.model.graph.input[0].type.tensor_type.shape.dim[-2].dim_value
                W = self.model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
                self.seq_len = H*W
        else:
            self.seq_len = self.ar_num

        # Transformations
        self._ar_builder = ArBuilder(self, ar_value)

        # For creating ops.
        self._op_factory = OpFactory(
            self.tensor_name_set,
            self.model,
            self.node_name_mapping_dict,
            self.mha_model_input_names_index_dict,
            self._ar_builder
        )

        # Extensions
        self._alibi_extension = AlibiExtension(self)
        self._rope_extension = RopeExtension(self)
        self._past_kv_extension = PastKeyValueExtension(self)
        self._mha_conv_extension = MhaConvExtension(self)
        self._lora_extension = LoraExtension(self)
        self._gqa_extension = GqaExtension(self)
        self._encoding_mapper = MHA2SHAEncodingMapper(self)
        self._rmsnorm_extension = RmsnormExtension(self)

        if "valid_token_mask" in self.mha_model_input_names:
            self.anchor_network = True
            self._anchor_extension = AnchorExtension(self)

        # HTP Specific Optimizations
        self._linear_to_conv = LinearToConv(self)
        self.qkv_head_concat_node_list = []


    def _run_sanity_checks(self):
        """Series of sanity checks run before running the optimizer"""

        assert not ((self.mha_conv and self.handle_rope_ops) and not self.nchw_aligned), "mha-conv with rope only support NCHW aligned model."
        if self.handle_past_key_value:
            assert self.llm_model, "--handle-past-key-value is currently only supported for LLM models. If this is a LLM model, please use --llm-model."
        if self.handle_rope_ops:
            assert self.handle_past_key_value, "--handle-rope-ops is turned on, --handle-past-key-value should also be turned on."
        if self.handle_internal_rmsnorm:
            assert self.mha_conv, "--handle-internal-rmsnorm is only supported with --mha-conv"

    def _update_all_mapping_dicts(self):
        """Helper function to update mappings to nodes.

        Updates all the mapping dictionaries such as `get_initializer_by_name`. These need
        to be updated as nodes are added to the graph and are not yet know.
        """
        update_all_mapping_dicts(self)

    def get_qkv_matmuls(
        self, qk_matmul_node: NodeProto, qkv_matmul_node: NodeProto
    ) -> List[NodeProto]:
        """
        Q = Query, K = Key, V = Value

        Function to search up the the branches of the QK MatMul to find the QK's origin MatMul's and search the QKV
        MatMul to find V's origin MatMul.

                +----------+      +----------+
                | Q MatMul |      | K MatMul |
                +----------+      +----------+
                    |                 |
                    |                 |
                    \               /
                    input 0       input 1
                        +-----------+
                        | QK MatMul |
                        +-----------+
        +----------+           |
        | V MatMul |           |
        +----------+       intput 0
            |          +------------+
            | _input 1_| QKV MatMul |
                       +------------+


        :param NodeProto qk_matmul_node: The MatMul node where the Q and K branches meet.
        :param NodeProto qkv_matmul_node: The MatMul node where the Q, K, and V branches meet.
        :return List[NodeProto]: The Q, K, and V origin MatMuls.
        """

        # May need to update as more models are added
        prefixes = ["to_"]
        suffixes = ["_proj"]

        def generate_qkv_names(qkv_names: Dict, words: List[str], append_as_prefix: bool = False):
            for word in words:
                for letter in "qkv":
                    key = f"{letter}_names"
                    qkv_names[key] = qkv_names.get(key, []) + [
                        f"{word}{letter}" if append_as_prefix else f"{letter}{word}"
                    ]

        qkv_names = {}
        generate_qkv_names(qkv_names, prefixes, append_as_prefix=True)
        generate_qkv_names(qkv_names, suffixes)

        qk_matmul_parents = get_parent(qk_matmul_node, self.get_node_by_output_name)
        assert len(qk_matmul_parents) == 2, \
            f"Could not find the parent nodes of the QK MatMul. Got: {qk_matmul_parents}"

        qkv_matmul_parents = get_parent(qkv_matmul_node, self.get_node_by_output_name)
        assert len(qkv_matmul_parents) == 2, \
            f"Could not find the parent nodes of the QKV MatMul. Got: {qkv_matmul_parents}"

        # Traverse up and find the first MatMul for each branch, then sort based on the name
        # Each name should be the same except with some variation of "q", "k", and "v".
        k_matmul, q_matmul = sorted([
            get_next_node_up_based_on_cond(
                node,
                self.get_node_by_output_name,
                node_found_cond=lambda n: n.op_type == "MatMul"
            )
            for node in qk_matmul_parents
        ], key=lambda n: n.name)

        # TODO remove search through regular expression, use graph instead.
        # Check to make sure the name's of the Q and K MatMul are correct.
        names_to_check = [
           (q_matmul.name, qkv_names["q_names"]),
           (k_matmul.name, qkv_names["k_names"]),
        ]

        try:
            for matmul_name, potential_names in names_to_check:
                assert any(name in matmul_name for name in potential_names), \
                    f"MatMul name does not match up. MatMul name: {matmul_name}, potential names: {potential_names}"
        except:
            log_warning("MatMul name does not match up with prefix and suffix name, use simplified method on q_matmul and k_matmul.")
            q_matmul = get_next_node_up_based_on_cond(
            self.get_node_by_output_name[qk_matmul_node.input[0]],
                self.get_node_by_output_name,
                node_found_cond=lambda n: n.op_type == "MatMul"
            )
            k_matmul = get_next_node_up_based_on_cond(
                    self.get_node_by_output_name[qk_matmul_node.input[1]],
                    self.get_node_by_output_name,
                    node_found_cond=lambda n: n.op_type == "MatMul"
                )

        # Traverse up both of the branches for the QKV MatMul and find the next occurrance of a MatMul.
        # After obtaining that list, we filter by the potential names of the V MatMul to select to correct one.
        # We then convert the filter object to a list.
        v_matmuls = list(
            filter(lambda n: any(v in n.name for v in qkv_names["v_names"]), [
                get_next_node_up_based_on_cond(
                    node,
                    self.get_node_by_output_name,
                    node_found_cond=lambda n: n.op_type == "MatMul"
                )
                for node in qkv_matmul_parents
            ])
        )

        try:
            assert len(v_matmuls) == 1, \
                ("There should only be one V MatMul found after filtering.",
                "Got: {}, Filtered with: {}".format([n.name for n in v_matmuls], qkv_names["v_names"]))
            v_matmul = v_matmuls[0]  # We know the V MatMul is correct because we grab this during filtering.
        except:
            log_warning("MatMul name does not match up with prefix and suffix name, use simplified method on v_matmul.")
            v_matmul = get_next_node_up_based_on_cond(
                    self.get_node_by_output_name[qkv_matmul_node.input[1]],
                    self.get_node_by_output_name,
                    node_found_cond=lambda n: n.op_type == "MatMul"
                )

        return [q_matmul, k_matmul, v_matmul]

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
        dquery = {}
        dkey = {}
        dvalue = {}

        query_matmul, key_matmul, value_matmul = self.get_qkv_matmuls(qk_matmul_node, qkv_matmul_node)

        if not (query_matmul and key_matmul and value_matmul):
            log_debug("Cannot find QKV Matmuls.")
        else:
            log_debug(
                f"QKV MatMuls : {query_matmul.name}, {key_matmul.name}, {value_matmul.name}"
            )

        dquery["matmul_node"] = query_matmul
        dkey["matmul_node"] = key_matmul
        dvalue["matmul_node"] = value_matmul

        query_initializer = None
        key_initializer = None
        value_initializer = None
        if query_matmul and key_matmul and value_matmul:
            query_initializer = numpy_helper.to_array(
                self.get_initializer_by_name[query_matmul.input[1]]
            )
            key_initializer = numpy_helper.to_array(
                self.get_initializer_by_name[key_matmul.input[1]]
            )
            value_initializer = numpy_helper.to_array(
                self.get_initializer_by_name[value_matmul.input[1]]
            )
        dquery["matmul_init"] = query_initializer
        dkey["matmul_init"] = key_initializer
        dvalue["matmul_init"] = value_initializer

        return dquery, dkey, dvalue

    def add_scale_on_q_branch(self, info_dict, input_node, propose_sha_name, sha_base_attn_node_list):
        """ Lvm feature, qk_scale can be on q_branch  """
        scale_op_type = self._scale_on_q_branch.op_type
        scale_value = info_dict["scale"]
        if scale_op_type == "Div": # scale
            scale_node, scale_init = self._op_factory.get_div_op(
                input_node,
                scale_value,
                propose_sha_name+"_Div"
            )
        elif scale_op_type == "Mul": # scale
            scale_node, scale_init = self._op_factory.get_mul_op(
                input_node,
                scale_value,
                propose_sha_name+"_Mul"
            )
        self.model.graph.initializer.extend(scale_init)
        self.model.graph.node.extend([scale_node])
        sha_base_attn_node_list.qk_scale.append(scale_node)

        return scale_node

    def lora_b_query_split(self, info_dict, num_heads, head_dim):
        init_name = "lora_b_init"
        lora_query_node = info_dict["query"]["mha_lora_node"]
        query_lora_split = None
        if init_name not in info_dict["query"] and lora_query_node.lora_b.op_type == "MatMul":
            query_lora_split = self._lora_extension.split_lora_b_adaptor_output(info_dict["query"],
                                                                                num_heads,
                                                                                head_dim,
                                                                                )
        return query_lora_split

    def get_head_num_for_conv_model(self, matmul_input_shape, inter_transpose):
        # conv -> [B, head_num*head_dim, 1, seq_len]/[B, C, H, W]
        if inter_transpose:
            permute_axis = inter_transpose.attribute[0].ints
            if permute_axis[1] == 1:
                # permute -> [B,  C, ...]
                # reshape -> [B, head_num, head_dim, seq_len]
                number_of_heads = matmul_input_shape[-3]
            elif permute_axis[2] == 1:
                # permute -> [B, ..., C, ...]
                # reshape -> [B, seq_len, head_num, head_dim] or [B, head_num, head_dim, seq_len]
                if permute_axis[1] == 2: # Case [B, 1, C, seq_len] -> [B, head_num, head_dim, seq_len]
                    number_of_heads = matmul_input_shape[-3]
                elif permute_axis[3] == 2: # Case [B, seq_len, dim, 1] -> [B, seq_len, head_num, head_dim]
                    number_of_heads = matmul_input_shape[-2]
                else:
                    raise ValueError(f"Got Transpose({permute_axis}) -> Reshape({matmul_input_shape}) pattern,\
and expect when transposing C to dim=2 will have H in NCHW transposed to dim = 1 or 3. Can't determine head_num.")

            elif permute_axis[3] == 1:
                # permute -> [B,  ..., C]
                # reshape -> [B, seq_len, head_num, head_dim]
                number_of_heads = matmul_input_shape[-2] # [B, head_num, head_dim, seq_len]
            else:
                raise ValueError(f"expecting permute C in NCHW to axis 1, 2, or 3, but got {permute_axis}")
        else:
            # reshape -> [B, head_num, head_dim, seq_len]
            number_of_heads = matmul_input_shape[-3] # [B, head_num, head_dim, seq_len]

        return number_of_heads


    def get_q_k_reshape_outputs(self, query_matmul: NodeProto, key_matmul: NodeProto) -> List[str]:
        """
        Searches for Q and K MatMul's Reshape.

        :param q_matmul: NodeProto - Query MatMul Node
        :param k_matmul: NodeProto - Key MatMul Node
        :return: Tuple - The names of the Query and Key Reshapes
        """

        def is_reshape(node: NodeProto):
            return node.op_type == "Reshape" and self.get_node_by_output_name[node.input[0]].op_type == "MatMul"

        return [
            get_next_node_down_based_on_cond(n, self.get_node_by_input_name, node_found_cond=is_reshape).output[0]
            for n in [query_matmul, key_matmul]
        ]


    def get_dquery_dkey_dvalue(self, qk_matmul_node, qkv_matmul_node):
        """
        Get dquery, dkey, dvalue.
        """
        # extract k_proj_end_node, v_proj_end_node for search_past_key_past_value_name
        if self.lora_model:
            dquery, dkey, dvalue = self._lora_extension.get_qkv_info(qk_matmul_node, qkv_matmul_node)
        else:
            if self.mha_conv:
                dquery, dkey, dvalue = self._mha_conv_extension.get_qkv_info(qk_matmul_node, qkv_matmul_node)
            else:
                dquery, dkey, dvalue = self.get_qkv_info(qk_matmul_node, qkv_matmul_node)

        k_proj_end_node = dkey["mha_lora_node"].lora_add if "mha_lora_node" in dkey else dkey["matmul_node"]
        v_proj_end_node = dvalue["mha_lora_node"].lora_add if "mha_lora_node" in dvalue else dvalue["matmul_node"]

        # Get past_key_name, past_value_name if past key/value option is on, by searching down stream from
        # k_proj_end_node, v_proj_end_node endpoints.
        if self.handle_past_key_value:
            past_key_name, past_value_name = (
                self._past_kv_extension.search_past_key_past_value_name(
                    k_proj_end_node,
                    v_proj_end_node,
                    qk_matmul_node,
                    qkv_matmul_node
                )
            )
            dkey["past_output_name"] = past_key_name
            dvalue["past_output_name"] = past_value_name

        return dquery, dkey, dvalue

    def set_scale_from_init(self, node: NodeProto, d_qkv: Dict, key_to_set: str):
        initializer = numpy_helper.to_array(
            self.get_initializer_by_name[node.input[1]]
        )
        d_qkv["scale"] = (  # want only the value not the array
            initializer[0]
            if initializer.ndim > 1 else
            initializer
        )
        d_qkv.update({key_to_set: {"init": initializer}})

    def set_scale_from_const_input(self, d_qkv, const_node):
        """Sets the scale of the info dict from a Constant Op.

        Args:
            d_qkv:
                Dictionary to add to.
            const_node:
                Constant Op to get scale from
        """
        const_tensor_value = get_constant_node_value(const_node)
        d_qkv["scale"] = const_tensor_value  # no [0] because tensor is shaped correctly


    def set_scale_from_identity(self, d_qkv, identity_node):
        """Sets the scale of the info dict from a Identity Op.

        Args:
            d_qkv:
                Dictionary to add to.
            identity_node:
                Identity Op to get scale from
        """
        scale_init = self.get_initializer_by_name[identity_node.input[0]]
        const_tensor_value = get_initializer_value(scale_init)
        if scale_init.dims[0] == 1 and len(const_tensor_value) == 1:
            d_qkv["scale"] = const_tensor_value[0]  # only want scale not array


    def get_mha_pattern_info(self, d_qkv, qk_matmul_node, qkv_matmul_node):
        """ Get info in mha pattern: qk_matmul to qkv_matmul  """
        mha_base_attn_node = BaseAttnNode(
            qk_matmul = qk_matmul_node,
            qk_scale = None,
            qk_softmax = None,
            qk_mask_add = None,
            qk_alibi_add = None,
            qkv_matmul = qkv_matmul_node,
            qkv_head_concat = qkv_matmul_node,
            q_matmul = d_qkv["query"]['matmul_node'],
            k_matmul = d_qkv["key"]['matmul_node'],
            v_matmul = d_qkv["value"]['matmul_node'],
        )
        nodes_list = []
        nodes_list.append(qk_matmul_node)
        for j in range(1, len(self.pattern) - 1):

            # Div/Mul for scale
            if self.pattern[j] == "Div" or self.pattern[j] == "Mul":
                div_or_mul_node = self.get_node_by_input_name[nodes_list[-1].output[0]][0]
                nodes_list.append(div_or_mul_node)
                if div_or_mul_node.input[1] in self.get_initializer_by_name.keys():
                    self.set_scale_from_init(div_or_mul_node, d_qkv, self.pattern[j])

                elif (scale_identity := self.get_node_by_output_name[div_or_mul_node.input[1]]) is not None and scale_identity.op_type == "Identity":
                    self.set_scale_from_identity(d_qkv, scale_identity)

                elif (div_or_mul_node_input := self.get_node_by_output_name.get(div_or_mul_node.input[1])) is not None and div_or_mul_node_input.op_type == "Constant":
                    self.set_scale_from_const_input(d_qkv, div_or_mul_node_input)

                mha_base_attn_node.qk_scale = nodes_list[-1]


            elif self.pattern[j] == "Add":
                add_node = self.get_node_by_input_name[nodes_list[-1].output[0]][0]
                if nodes_list[-1].op_type == "Add":
                    key = "Add_1"
                else:
                    key = "Add"

                if self.get_initializer_by_name.get(add_node.input[1]) is not None:
                    add_initializer = onnx.numpy_helper.to_array(
                        self.get_initializer_by_name[add_node.input[1]]
                    )
                    d_add = {key: {"init": add_initializer, "parent": None}}
                elif add_node.input[1] in self.mha_model_input_names_index_dict.keys():
                    d_add = {key: {"init": None, "parent": add_node.input[1]}}
                else:
                    add_node_parents = get_parent_at_any_level(
                        add_node.output[0], self.get_node_by_output_name, 1
                    )
                    add_parent = None
                    for parent in add_node_parents:
                        if parent in nodes_list:
                            pass
                        else:
                            add_parent = parent
                    d_add = {key: {"init": None, "parent": add_parent}}
                nodes_list.append(add_node)
                d_qkv.update(d_add)
                # First Add is always attention mask for ALiBi
                if self.handle_alibi and mha_base_attn_node.qk_mask_add is not None:
                    mha_base_attn_node.qk_alibi_add = nodes_list[-1]
                else:
                    mha_base_attn_node.qk_mask_add = nodes_list[-1]

            elif self.pattern[j] == "Where":
                where_node = self.get_node_by_input_name[nodes_list[-1].output[0]][0]
                where_C = where_node.input[0]
                where_X = where_node.input[1]
                where_Y = where_node.input[2]
                nodes_list.append(where_node)
                d_where = {"Where": {"C": where_C, "X": where_X, "Y": where_Y}}
                d_qkv.update(d_where)

            elif self.pattern[j] == "Softmax":
                softmax_node = self.get_node_by_input_name[nodes_list[-1].output[0]][0]
                axis = softmax_node.attribute[0].i
                nodes_list.append(softmax_node)
                d_softmax = {"Softmax": {"axis": axis}}
                d_qkv.update(d_softmax)
                mha_base_attn_node.qk_softmax = nodes_list[-1]

            elif self.pattern[j] == "Cast":
                cast_node = self.get_node_by_input_name[nodes_list[-1].output[0]][0]
                nodes_list.append(cast_node)
            elif self.pattern[j] == "Transpose":
                transpose_node = self.get_node_by_input_name[nodes_list[-1].output[0]][0]
                permute_seq = transpose_node.attribute[0].ints

                # Only deal with transpose that permute from masks to fit in to mha shape and the premute
                # to transpose back, these nodes will be ignored when doing mha to sha split.
                # qk matmul shape: [B, head_num, q_seq_len, k_seq_len] ->
                # Transpose to [B, q_seq_len, k_seq_len, head_num] ->
                # Add mask: [1, q_seq_len, k_seq_len, 1] ->
                # Transpose back to [B, head_num, q_seq_len, k_seq_len]
                if self.get_node_by_input_name[transpose_node.output[0]][0].op_type == "Add":
                    if permute_seq == [0, 2, 3, 1]: # [B, head_num, q_seq_len, k_seq_len] to [B, q_seq_len, k_seq_len, head_num]
                        nodes_list.append(transpose_node)
                    else:
                        raise ValueError(
                            f"Expect Tranpose op before Add mask op has permute [0, 2, 3, 1]. but got {permute_seq}."
                        )
                elif nodes_list[-1].op_type == "Add":
                    if permute_seq == [0, 3, 1, 2]: # [B, q_seq_len, k_seq_len, head_num] to [B, head_num, q_seq_len, k_seq_len]
                        nodes_list.append(transpose_node)
                    else:
                        raise ValueError(
                            f"Expect Tranpose op after Add mask op has permute [0, 3, 2, 1]. but got {permute_seq}."
                        )
                else:
                    raise ValueError("expect only Tranpose op before or after Add mask op.")

            else:
                log_error(f"Undefined op_type: {self.pattern[j]}. Exiting")

        node_list_pattern_set = {node.op_type for node in nodes_list}
        self._scale_on_q_branch = False
        if ("Mul" not in node_list_pattern_set) and ("Div" not in node_list_pattern_set):
            # Seach between q_matmal and qk_matmul for Mul scale for not LLM models
            if not self.llm_model:
                try:
                    temp_scale_op = get_next_node_down_based_on_cond(d_qkv["query"]["matmul_node"],
                        self.get_node_by_input_name,
                        node_found_cond=lambda n: n.op_type in ("Mul", "Div"),
                        node_end_search_cond=lambda n: n == qk_matmul_node
                    )
                    mha_base_attn_node.qk_scale = temp_scale_op

                    if temp_scale_op.input[1] in self.get_initializer_by_name.keys():
                        self._scale_on_q_branch = temp_scale_op
                        d_qkv["scale"] = numpy_helper.to_array(
                            self.get_initializer_by_name[temp_scale_op.input[1]]
                        )[0]  # only want scale not array

                    elif (scale_identity := self.get_node_by_output_name[temp_scale_op.input[1]]).op_type == "Identity":
                        self.set_scale_from_identity(d_qkv, scale_identity)
                        if "scale" in d_qkv:  # the above function only sets the "scale" if it succeeds
                            self._scale_on_q_branch = temp_scale_op

                    elif (scale_node := self.get_node_by_output_name[temp_scale_op.input[1]]).op_type == "Constant":
                        self._scale_on_q_branch = temp_scale_op
                        self.set_scale_from_const_input(d_qkv, scale_node)

                    if self._scale_on_q_branch:
                        log_warning("Found scale Mul/Div between q_proj and qk_matmul. Will insert scale on q branch if not found scale between qk_matmul and qkv_matmul.")
                except NodeNotFoundError as e:
                    log_debug(f"{e.__str__()}")

            if not self._scale_on_q_branch:
                log_warning("Did not find Mul or Div in pattern, no qk scale in model. Make sure scale is folded into model or doesn't exist.")

        final_Matmul_output_node_list = self.get_node_by_input_name[qkv_matmul_node.output[0]]

        return final_Matmul_output_node_list, nodes_list, mha_base_attn_node

    def get_attention_subgraph_info(self, start_node_name: str, end_node_name: str) -> Dict:
        """
        Function responsible for storing the sub-graph information in a dictionary.
        Sub-graph is the sequence of the nodes in the matched pattern.

        :param start_node_name: str - starting node of a pattern
        :param end_node_name: str - ending node of a pattern
        :return d_qkv: Dict - stores the sub-graph information in a dictionary.
        """

        qk_matmul_node = self.get_node_by_node_name[start_node_name]
        qkv_matmul_node = self.get_node_by_node_name[end_node_name]
        dquery, dkey, dvalue = self.get_dquery_dkey_dvalue(qk_matmul_node, qkv_matmul_node)

        d_qkv = {
            "query": dquery,
            "key": dkey,
            "value": dvalue
        }

        if self.lora_model:
            d_qkv["q_lora_mha_node"] = dquery["mha_lora_node"] if "mha_lora_node" in dquery else None
            d_qkv["k_lora_mha_node"] = dkey["mha_lora_node"] if "mha_lora_node" in dkey else None
            d_qkv["v_lora_mha_node"] = dvalue["mha_lora_node"] if "mha_lora_node" in dvalue else None
        else:
            d_qkv["q_lora_mha_node"] = None
            d_qkv["k_lora_mha_node"] = None
            d_qkv["v_lora_mha_node"] = None

        if self.handle_rope_ops:
            self._rope_extension.interleave_rope = None # Reset interleave_rope for every attention
            d_qkv["q_rope_mha_node"] = self._rope_extension.get_mha_rope_nodes(dquery["matmul_node"], qk_matmul_node)
            d_qkv["k_rope_mha_node"] = self._rope_extension.get_mha_rope_nodes(dkey["matmul_node"], qk_matmul_node)
        else:
            d_qkv["q_rope_mha_node"] = None
            d_qkv["k_rope_mha_node"] = None

        d_qkv["past_key_value_concat"] = None
        if self.handle_past_key_value:
            # handled later on in get_kv_cache
            d_qkv["past_key_value_concat"] = PastKeyValueConcat()


        # Get the unsequeezes at the end of the rope cos/sin branches
        if self.handle_rope_ops:
            rope_cos, rope_sin = self._rope_extension.get_position_ids(qk_matmul_node)
            if self.use_position_embedding:
                d_qkv["rope_cos_model_input"] = rope_cos
                d_qkv["rope_sin_model_input"] = rope_sin

            else:
                rope_cos_gather_output, rope_sin_gather_output = [
                    self._rope_extension.get_ropes_gather_output(n)
                    for n in [rope_cos, rope_sin]
                ]

                rope_cos_unsqueeze, rope_sin_unsqueeze = [
                    self.get_node_by_input_name[rope_gather_output]
                    for rope_gather_output in [rope_cos_gather_output, rope_sin_gather_output]
                ]

                assert(len(rope_cos_unsqueeze) == 1 and len(rope_sin_unsqueeze) == 1), \
                "Expected only one node when gathering ropes cos/sin unsqueeze, " \
                    f"instead got {len(rope_cos_unsqueeze)=}, {len(rope_sin_unsqueeze)=}"

                d_qkv["rope_cos_unsqueeze"] = rope_cos_unsqueeze[0]
                d_qkv["rope_sin_unsqueeze"] = rope_sin_unsqueeze[0]

        final_Matmul_output_node_list, nodes_list, mha_base_attn_node = (
            self.get_mha_pattern_info(d_qkv, qk_matmul_node, qkv_matmul_node)
        )

        if self.lora_model and ("mha_lora_node" in d_qkv["query"]):
            q_proj_end_node = d_qkv["query"]["mha_lora_node"].lora_add
        else:
            q_proj_end_node = d_qkv["query"]["matmul_node"]

        # Get number of heads
        # Use the first reshape node after q matmul before qk_matmul
        reshape_node = get_next_node_down_based_on_cond(q_proj_end_node,
                                                        self.get_node_by_input_name,
                                                        node_found_cond=lambda n: n.op_type == "Reshape",
                                                        node_end_search_cond=lambda n: n == qk_matmul_node)
        matmul_input_shape = get_node_input_constant_op_value(reshape_node,
                                                              self.get_node_by_output_name,
                                                              self.get_initializer_by_name)

        if self.mha_conv and self.nchw_aligned:
            # collecting transpose between q_input and q_reshape
            inter_transpose = None
            try:
                inter_transpose = get_next_node_down_based_on_cond(
                    d_qkv["query"]["matmul_node"],
                    self.get_node_by_input_name,
                    node_found_cond=lambda n: n.op_type == "Transpose",
                    node_end_search_cond=lambda n: n == reshape_node
                )

            except NodeNotFoundError:
                ... # inter_transpose is None by default and handled below

            number_of_heads = self.get_head_num_for_conv_model(matmul_input_shape, inter_transpose)

        else:
          number_of_heads = matmul_input_shape[-2] # [B, seq_len, head_num, head_dim]

        if self.gqa_model:
            if self.lora_model and ("mha_lora_node" in d_qkv["key"]):
                k_proj_end_node = d_qkv["key"]["mha_lora_node"].lora_add
            else:
                k_proj_end_node = d_qkv["key"]["matmul_node"]
            self._gqa_extension.kv_head_num = \
                    self._gqa_extension.get_kv_group_head_num(
                        k_proj_end_node,
                        d_qkv["key"]["matmul_node"],
                        qk_matmul_node)

        if self.handle_past_key_value:
            # Utilize rope search if it's already happened.
            if self.handle_rope_ops:
                d_qkv["past_key_value_concat"].past_key_concat_out = d_qkv["k_rope_mha_node"].rope_concat

            # Use lora-add's encodings is there's lora adaptor on v branch
            if self.lora_model and "mha_lora_node" in dvalue:
                # Use last lora node_add
                if (_lora_node := dvalue["mha_lora_node"].next_node):
                    while _lora_node:
                        d_qkv["past_key_value_concat"].past_value_concat_out = _lora_node.lora_add
                        _lora_node = _lora_node.next_node
                else:
                    d_qkv["past_key_value_concat"].past_value_concat_out = dvalue["mha_lora_node"].lora_add

            else:
                d_qkv["past_key_value_concat"].past_value_concat_out = dvalue["matmul_node"]

        if self.handle_internal_rmsnorm:
            self._rmsnorm_extension.capture_rmsnorm(d_qkv, qk_matmul_node)

        if self.handle_internal_rmsnorm:
            d_qkv["q_rmsnorm"] = dquery["mha_rmsnorm_node"]
            d_qkv["k_rmsnorm"] = dkey["mha_rmsnorm_node"]
        else:
            d_qkv["q_rmsnorm"] = None
            d_qkv["q_rmsnorm"] = None

        k_concat = None
        v_concat = None
        if self.handle_past_key_value:
            self.kv_cache, k_concat, v_concat = self._past_kv_extension.get_kv_cache(
                dvalue['matmul_node'], qkv_matmul_node, qk_matmul_node
            )

            if self.kv_cache:
                d_qkv["past_key_value_concat"].past_key_concat = k_concat
                d_qkv["past_key_value_concat"].past_value_concat = v_concat

                # Scatter Op has inputs [past_in, indices, current_kv]
                # Concat Op has inputs [past_in, current_kv]
                if (_input := k_concat.input[0]) in self.mha_model_input_names_index_dict.keys():
                    d_qkv["key"]["past_input_name"] = _input
                else:
                    raise ValueError("Expect k_concat input[0] is past_key_in,\
                                     but k_concat.input[0] is not in model input list.")

                if( _input := v_concat.input[0]) in self.mha_model_input_names_index_dict.keys():
                    d_qkv["value"]["past_input_name"] = _input
                else:
                    raise ValueError("Expect v_concat input[0] is past_value_in,\
                                     but v_concat.input[0] is not in model input list.")

        d_rest = {
            "Matmul_1": qk_matmul_node,
            "num_heads": number_of_heads,
            "final_Matmul_output_node_list": final_Matmul_output_node_list,
            "nodes_list": nodes_list,
            "k_concat": k_concat,
            "v_concat": v_concat,
        }
        d_qkv["mha_base_attn_node"] = mha_base_attn_node
        d_qkv.update(d_rest)
        return d_qkv

    def align_attn_matmul_input_4d(self, info_dict):
        """
        Add or replace qkv_inp reshape to align sha input to 4D [1, 1, seq_len, vector_dim].
        :info_dict
        """
        # Create reshape -> unsqueeze to prepare query key value branches
        # change hidden_state shape to [1, seq_len, vector_dim] -> unsqueeze [1, 1, seq_len, vector_dim]
        query_matmul_inp = info_dict["qkv_input"][0].output[0]
        key_matmul_inp = info_dict["qkv_input"][1].output[0]
        value_matmul_inp = info_dict["qkv_input"][2].output[0]

        matmul_inp_shape = [1, 1, self.seq_len, info_dict["key"]["matmul_init"].shape[0]]
        if info_dict["qkv_input"][0] == info_dict["qkv_input"][1] == info_dict["qkv_input"][2]:
            # If q, k and v have shared input node, then share reshape node
            qkv_input_reshape_node, init_list_qkv_shared_reshape = self._op_factory.get_reshape_op(
                query_matmul_inp, matmul_inp_shape
            )
            self.model.graph.initializer.extend(init_list_qkv_shared_reshape)
            self.model.graph.node.append(qkv_input_reshape_node)

            # Update matmul_inp
            query_matmul_inp = qkv_input_reshape_node
            key_matmul_inp = qkv_input_reshape_node
            value_matmul_inp = qkv_input_reshape_node

        else:
            reshape_init_list = []
            reshape_node_list = []

            if info_dict["qkv_input"][0].op_type == "Reshape":
                query_matmul_inp = info_dict["qkv_input"][0].input[0]

            query_matmul_inp, init_list_q_reshape = self._op_factory.get_reshape_op(
                query_matmul_inp, [1, 1, -1, info_dict["query"]["matmul_init"].shape[0]])
            reshape_node_list.append(query_matmul_inp)
            reshape_init_list.extend(init_list_q_reshape)

            if info_dict["qkv_input"][1].op_type == "Reshape":
                key_matmul_inp = info_dict["qkv_input"][1].input[0]

            key_matmul_inp, init_list_k_reshape = self._op_factory.get_reshape_op(
                key_matmul_inp, [1, 1, -1, info_dict["key"]["matmul_init"].shape[0]])
            reshape_node_list.append(key_matmul_inp)
            reshape_init_list.extend(init_list_k_reshape)

            if info_dict["qkv_input"][2].op_type == "Reshape":
                value_matmul_inp = info_dict["qkv_input"][2].input[0]
            value_matmul_inp, init_list_v_reshape = self._op_factory.get_reshape_op(
                value_matmul_inp, [1, 1, -1, info_dict["value"]["matmul_init"].shape[0]])
            reshape_node_list.append(value_matmul_inp)
            reshape_init_list.extend(init_list_v_reshape)

            self.model.graph.initializer.extend(reshape_init_list)
            self.model.graph.node.extend(reshape_node_list)
        return query_matmul_inp, key_matmul_inp, value_matmul_inp

    def get_qkv_input(self, info_dict):
        # qkv_input are the input nodes for q k v proj matmul/conv
        qkv_input = []
        for qkv_name in ["query", "key", "value"]:
            for name in info_dict[qkv_name]["matmul_node"].input:
                if name not in self.get_initializer_by_name.keys():
                    matmul_input_node = self.get_node_by_output_name[name]
            qkv_input.append(matmul_input_node)

        assert len(qkv_input)==3, f'Expecting len(qkv_input) == 3, but got {len(qkv_input)}'
        return qkv_input

    def create_sha_qkv_matmuls(self,
                               info_dict,
                               ns,
                               head_num,
                               head_dim,
                               sha_base_attn_node_list,
                               query_matmul_inp,
                               key_matmul_inp,
                               value_matmul_inp,
                               init_name = "matmul_init",
                               suffix=None):
        """
        Create MatMuls with initializer sliced by head.
        :return query_inp: query linear node
        :return key_inp: key linear node
        :return value_inp: value linear node
        """
        _, propose_sha_query_name, propose_sha_key_name, propose_sha_value_name = sha_node_name_basis(ns, head_num)
        sha_query_weight_name = propose_sha_query_name+".weight" if suffix is None else propose_sha_query_name+f"_{suffix}.weight"
        sha_key_weight_name = propose_sha_key_name+".weight" if suffix is None else propose_sha_query_name+f"_{suffix}.weight"
        sha_value_weight_name = propose_sha_value_name+".weight" if suffix is None else propose_sha_query_name+f"_{suffix}.weight"

        # Create weight names
        if info_dict["query"][init_name] is not None:
            query_mm_init_name, self.tensor_name_set = create_tensor_name(
                sha_query_weight_name, self.tensor_name_set
            )
            assert sha_query_weight_name == query_mm_init_name, f"sha_query_weight_name and created init name are differnet"

            key_mm_init_name, self.tensor_name_set = create_tensor_name(
                sha_key_weight_name, self.tensor_name_set
            )
            assert sha_key_weight_name == key_mm_init_name, f"sha_key_weight_name and created init name are differnet"

            value_mm_init_name, self.tensor_name_set = create_tensor_name(
                sha_value_weight_name, self.tensor_name_set
            )
            assert sha_value_weight_name == value_mm_init_name, f"sha_value_weight_name and created init name are differnet"

            # Create q, k, v matmul weight
            query_mm_init = numpy_helper.from_array(
                info_dict["query"][init_name][
                    :, head_num * head_dim : (head_num + 1) * head_dim
                ],
                query_mm_init_name,
            )
            key_mm_init = numpy_helper.from_array(
                info_dict["key"][init_name][
                    :, head_num * head_dim : (head_num + 1) * head_dim
                ],
                key_mm_init_name,
            )
            value_mm_init = numpy_helper.from_array(
                info_dict["value"][init_name][
                    :, head_num * head_dim: (head_num + 1) * head_dim
                ],
                value_mm_init_name,
            )

            # Create q, k, v matmul op
            query_matmul = self._op_factory.get_matmul_op(
                query_matmul_inp,
                query_mm_init_name,
                propose_op_name = propose_sha_query_name+"_MatMul" if suffix is None else propose_sha_query_name+f"_{suffix}"
            )
            key_matmul = self._op_factory.get_matmul_op(
                key_matmul_inp,
                key_mm_init_name,
                propose_op_name = propose_sha_key_name+"_MatMul" if suffix is None else propose_sha_key_name+f"_{suffix}"
            )
            value_matmul = self._op_factory.get_matmul_op(
                value_matmul_inp,
                value_mm_init_name,
                propose_op_name = propose_sha_value_name+"_MatMul" if suffix is None else propose_sha_value_name+f"_{suffix}"
            )
            self.model.graph.initializer.extend(
                [query_mm_init, key_mm_init, value_mm_init]
            )
            self.model.graph.node.extend(
                [query_matmul, key_matmul, value_matmul]
            )

            sha_base_attn_node_list.q_matmul.append(query_matmul)
            sha_base_attn_node_list.k_matmul.append(key_matmul)
            sha_base_attn_node_list.v_matmul.append(value_matmul)

            query_inp = query_matmul
            key_inp = key_matmul
            value_inp = value_matmul
        else:
            raise ValueError("Missing initializer for q, k, and v projection")

        return query_inp, key_inp, value_inp

    def create_sha_pattern(self, info_dict, ns, head_num, value_inp, sha_nodes_list, sha_base_attn_node_list):

        propose_sha_name, _, _, _ = sha_node_name_basis(ns, head_num)
        attention_mask_add_added = False

        for j in range(1, len(self.pattern) - 1):
            if self.pattern[j] == "Div": # scale
                div_value = info_dict["scale"]
                div_node, div_init = self._op_factory.get_div_op(
                    sha_nodes_list[-1],
                    div_value,
                    propose_sha_name+"_Div"
                )
                sha_nodes_list.append(div_node)
                self.model.graph.initializer.extend(div_init)
                self.model.graph.node.extend([div_node])
                sha_base_attn_node_list.qk_scale.append(div_node)

            elif self.pattern[j] == "Mul": # scale
                mul_value = info_dict["scale"]
                mul_node, mul_init = self._op_factory.get_mul_op(
                    sha_nodes_list[-1],
                    mul_value,
                    propose_sha_name+"_Mul"
                )
                sha_nodes_list.append(mul_node)
                self.model.graph.initializer.extend(mul_init)
                self.model.graph.node.extend([mul_node])
                sha_base_attn_node_list.qk_scale.append(mul_node)

            # Attention mask
            elif self.pattern[j] == "Add":
                r"""
                ALiBi pattern, we have 2 different Add's. One for the attention mask
                and the other for the position ids.

                +----------------+              +-----+                     +--------------+
                | attention_mask |              | Div |                     | position_ids |
                +----------------+              +-----+                     +--------------+
                        |                          |                               |
                        |              +-----+     |                               |
                        +------------->| Add |<----+                               |
                                       +-----+                                     |
                                          |                                        |
                                          |          +-------+                     |
                                          +--------->| Add_1 |<--------------------+
                                                     +-------+
                                                        |
                                                        V
                """
                if self.handle_alibi and attention_mask_add_added:
                    attn_mask_or_alibi_input = self._alibi_extension.get_curr_head_slice_inp(info_dict["Add_1"].get("parent"), head_num)
                else:
                    attn_mask_or_alibi_input = info_dict["Add"].get("parent")
                    # Remove transpose to attention mask to align attention mask shape with MHA.
                    # Patten observed in prepared-llama-v2
                    if isinstance(attn_mask_or_alibi_input, NodeProto) and attn_mask_or_alibi_input.op_type == "Transpose":
                        assert attn_mask_or_alibi_input.attribute[0].ints == [0, 2, 3, 1]
                        if attn_mask_or_alibi_input.input[0] in self.get_node_by_output_name.keys():
                            attn_mask_or_alibi_input = self.get_node_by_output_name[attn_mask_or_alibi_input.input[0]]
                        elif attn_mask_or_alibi_input.input[0] in self.mha_model_input_names_index_dict.keys():
                            attn_mask_or_alibi_input = attn_mask_or_alibi_input.input[0]


                add_node = self._op_factory.get_add_op(sha_nodes_list[-1], attn_mask_or_alibi_input)
                sha_nodes_list.append(add_node)
                self.model.graph.node.extend([add_node])
                if self.handle_alibi and attention_mask_add_added:
                    sha_base_attn_node_list.qk_alibi_add.append(add_node)
                else:
                    sha_base_attn_node_list.qk_mask_add.append(add_node)
                    attention_mask_add_added = True

            elif self.pattern[j] == "Where":
                if info_dict["Where"].get("C") is not None:
                    inp1 = info_dict["Where"]["C"]

                    if (
                        self.get_value_info_by_name.get(info_dict["Where"]["C"])
                        is not None
                    ):
                        where_c_input_shape = get_shape_from_value_info_proto(
                            self.get_value_info_by_name[
                                info_dict["Where"]["C"]
                            ],
                            False,
                        )
                        if where_c_input_shape[1] == info_dict["num_heads"]:
                            slice_node, slice_init = self._op_factory.get_slice_op(
                                info_dict["Where"]["C"],
                                head_num,
                                head_num + 1,
                                1,
                            )
                            self.model.graph.initializer.extend(
                                slice_init
                            )
                            self.model.graph.node.extend(
                                [slice_node]
                            )
                            inp1 = slice_node.output[0]

                if info_dict["Where"].get("X") is not None:
                    inp2 = info_dict["Where"]["X"]
                if info_dict["Where"].get("Y") is not None:
                    inp3 = info_dict["Where"]["Y"]

                if (
                    info_dict["Where"]["C"]
                    == info_dict["nodes_list"][-1].output[0]
                ):
                    inp1 = sha_nodes_list[-1]
                elif (
                    info_dict["Where"]["X"]
                    == info_dict["nodes_list"][-1].output[0]
                ):
                    inp2 = sha_nodes_list[-1]
                elif (
                    info_dict["Where"]["Y"]
                    == info_dict["nodes_list"][-1].output[0]
                ):
                    inp3 = sha_nodes_list[-1]

                where_node = self._op_factory.get_where_op(inp1, inp2, inp3)
                sha_nodes_list.append(where_node)
                self.model.graph.node.extend([where_node])

            elif self.pattern[j] == "Softmax":
                softmax_node = self._op_factory.get_softmax_op(
                    sha_nodes_list[-1],
                    axis = -1,
                    propose_op_name = propose_sha_name+"_Softmax"
                )
                sha_nodes_list.append(softmax_node)
                self.model.graph.node.extend([softmax_node])
                sha_base_attn_node_list.qk_softmax.append(softmax_node)

            elif self.pattern[j] == "Cast":
                assert info_dict["nodes_list"][j].op_type == "Cast"
                cast_to_int = info_dict["nodes_list"][j].attribute[0].i
                cast_node = self._op_factory.get_cast_op(sha_nodes_list[-1], cast_to_int)
                sha_nodes_list.append(cast_node)
                self.model.graph.node.extend([cast_node])

            elif self.pattern[j] == "Transpose":
                # Ignore transpose in mha pattern, since the transpose are used to align
                # shape between MHA attn_score and attn mask from model input.
                pass

            else:
                log_error(f"MHA2SHA-split: Got op_type without handler: {self.pattern[j]}. Exiting.")

        qkv_matmul = self._op_factory.get_matmul_op(softmax_node, value_inp, propose_sha_name+"_qkv_MatMul")
        sha_base_attn_node_list.qkv_matmul.append(qkv_matmul)

        return qkv_matmul

    def create_sha(self,
                   info_dict,
                   ns,
                   head_num,
                   head_dim,
                   query_matmul_inp,
                   key_matmul_inp,
                   value_matmul_inp,
                   sha_base_attn_node_list,
                   **extenstion_kwargs):
        """
        Creates sha for each head.
        :param info_dict: mha info dict
        :param ns: number of start node (attention layer num)
        :param head_num: head number in heads
        :param head_dim: vector dim for each head
        :param query_matmul_inp: input to query linear
        :param key_matmul_inp: input to key linear
        :param value_matmul_inp: input to value linear
        :param sha_base_attn_node_list: sha base attn node
        :return qkv_matmul: qkv_matmul node
        :return _key_inp: _key_inp for llama to return cached keys
        :return _value_inp: _value_inp for llama to return cached keys
        :return past_value_inp: llama kv$ model value model input
        """
        propose_sha_name, _, propose_sha_key_name, _ = sha_node_name_basis(ns, head_num)
        if self.lora_model:
            raise NotImplementedError("LORA is not supported with linear models")

        # Step 1 Create sha matmuls
        # TODO Add support for qkv projection linear with bias (MatMul -> Add)
        query_inp, key_inp, value_inp = self.create_sha_qkv_matmuls(info_dict,
                                                                    ns,
                                                                    head_num,
                                                                    head_dim,
                                                                    sha_base_attn_node_list,
                                                                    query_matmul_inp,
                                                                    key_matmul_inp,
                                                                    value_matmul_inp)

        # Step 2 Inject ROPE (optional)
        if self.handle_rope_ops:
            if self.use_position_embedding:
                # Prepared llama sin and cos are model input.
                cos_node = info_dict["rope_cos_model_input"]
                sin_node = info_dict["rope_sin_model_input"]
                query_inp = self._rope_extension.create_llama_rope_node(query_inp, cos_node, sin_node, head_dim,
                                                                        BranchType.Q, info_dict)
                key_inp = self._rope_extension.create_llama_rope_node(key_inp, cos_node, sin_node, head_dim,
                                                                      BranchType.K, info_dict)
            else:
                # Unprepared model uses unsqueeze node.
                cos_node = info_dict["rope_cos_unsqueeze"]
                sin_node = info_dict["rope_sin_unsqueeze"]
                query_inp, key_inp = self._rope_extension.get_llama_rope_node_without_positional_embedding(query_inp, key_inp, cos_node, sin_node, head_dim)

        # Step 3 Tranpose K
        # expecting key_matmul_inp_shape = [1, 1, self.seq_len, head_dim]
        key_transpose_perm = [0, 1, 3, 2] # [1, 1, seq_len, head_dim] -> [1, 1, head_dim, seq_len]
        key_transpose = self._op_factory.get_transpose_op(key_inp, key_transpose_perm, propose_sha_key_name+"_Transpose")
        self.model.graph.node.append(key_transpose)

        # Step 4 Concate pask key value input
        _key_inp = key_transpose # [1, 1, emd_dim, seq_len]
        _value_inp = value_inp   # [1, 1, seq_len, emd_dim]

        past_key_inp = None
        past_value_inp = None
        # Concate past key input and past value output
        if self.handle_past_key_value and self.kv_cache:
            past_key_inp = info_dict["key"]["past_input_name"]
            past_value_inp = info_dict["value"]["past_input_name"]
            key_transpose, value_inp = self._past_kv_extension.concat_past_key_value_input(
                past_key_inp, past_value_inp, key_transpose, value_inp, head_num, info_dict["head_dim"]
            )

        if not self.return_new_key_value_only:
            # Update _key_inp with concat key and value
            _key_inp = key_transpose # [1, 1, emd_dim, past+new seq_len]
            _value_inp = value_inp # [1, 1, seq_len, past+new emd_dim]


        if self._scale_on_q_branch:
            query_inp = self.add_scale_on_q_branch(info_dict, query_inp, propose_sha_name, sha_base_attn_node_list)

        # Step 5 QK matmul
        qk_matmul_node = self._op_factory.get_matmul_op(query_inp, key_transpose, propose_sha_name+"_qk_MatMul")
        self.model.graph.node.append(qk_matmul_node)

        sha_base_attn_node_list.qk_matmul.append(qk_matmul_node)

        sha_nodes_list = []
        sha_nodes_list.append(qk_matmul_node)

        # Step 5 Create sha pattern
        # Create attention patterns from qk_matmul to qkv_matmul
        qkv_matmul = self.create_sha_pattern(info_dict, ns, head_num, value_inp, sha_nodes_list, sha_base_attn_node_list)


        self.model.graph.node.extend([qkv_matmul])

        return qkv_matmul, _key_inp, _value_inp, past_value_inp

    def create_sha_func_handler(self, query_lora_inp, key_lora_inp, value_lora_inp):
        extension_kwargs = {"query_lora_inp": query_lora_inp,
                            "key_lora_inp": key_lora_inp,
                            "value_lora_inp": value_lora_inp}

        # Extension other than mha_conv should be handled in create_sha_conv_with_rope, create_sha_conv or create_sha
        if self.mha_conv and self.handle_rope_ops and self.nchw_aligned:
            create_sha_func = self._mha_conv_extension.create_sha_conv_with_rope
        elif self.mha_conv and not self.handle_rope_ops:
            create_sha_func = self._mha_conv_extension.create_sha_conv
        else:
            create_sha_func = self.create_sha

        return create_sha_func, extension_kwargs

    def optimize(self) -> ModelProto:
        """
        Fucntion responsible for:
        1. Modifying the graph by spliting MHA
            1.1 Get Q, K, V linear weight info from MHA
            1.2 For each head:
                1.2.1 Create Q, K, V lienar nodes for SHA.
                1.2.2 Create pattern from qk_matmul to qkv_matmul
        2. Cleaning the model.
        3. Topologically sorting the model.

        :return self.model: ModelProto - modified onnx model.
        """
        if self.handle_past_key_value:
            self._seen_past_key_values = set()
        self._ar_builder.update_initial_ar_values()

        start_end_node_names = zip(self.pattern_start_node_names, self.pattern_end_node_names)

        for ns, (start_node_name, end_node_name) in enumerate(start_end_node_names):
            # 1.1 Get Q, K, V linear weight info from MHA
            log_info(f"\n\nStart node: {ns}, {start_node_name} -> {end_node_name}")
            info_dict = self.get_attention_subgraph_info(start_node_name, end_node_name)

            # Get q k v proj node input
            qkv_input = self.get_qkv_input(info_dict)
            info_dict['qkv_input'] = qkv_input

            # MatMul weight: matmul_init have shape [I, O].
            # Conv model has already reshaped weight before log to info_dict[q/k/v]["matmul_init"]
            if not self.gqa_model:
                block_size = [qkv_input_node["matmul_init"].shape[1] // info_dict["num_heads"] for \
                                qkv_input_node in [info_dict['query'], info_dict['key'], info_dict['value']] ]
            else:
                block_size = [info_dict['query']["matmul_init"].shape[1] // info_dict["num_heads"],
                              info_dict['key']["matmul_init"].shape[1] // self._gqa_extension.kv_head_num,
                              info_dict['value']["matmul_init"].shape[1] // self._gqa_extension.kv_head_num]

            assert block_size[0] == block_size[1] == block_size[2], f"head_dim should be the same for q, k, and v projection, but got {block_size[0]}, {block_size[1]}, {block_size[2]}"
            head_dim = block_size[0]
            info_dict["head_dim"] = head_dim
            log_info(f"head_dim: {head_dim} num_heads: {info_dict['num_heads']}")

            """ Update mha info to encoding mapping dict """
            encoding_mapping_dict = get_encoding_mapping_dict(info_dict, head_dim)

            # Start creating SHA
            # Align sha input to 4D input
            if not self.nchw_aligned:
                # Aligned to N1LD.
                query_matmul_inp, key_matmul_inp, value_matmul_inp = self.align_attn_matmul_input_4d(info_dict)
            else:
                query_matmul_inp = info_dict["qkv_input"][0].output[0]
                key_matmul_inp = info_dict["qkv_input"][1].output[0]
                value_matmul_inp = info_dict["qkv_input"][2].output[0]

            if self.lora_model:
                query_lora_inp, key_lora_inp, value_lora_inp = self._lora_extension.get_qkv_lora_input(info_dict)
            else:
                query_lora_inp, key_lora_inp, value_lora_inp = None, None, None

            # Collects qkv matmal later to be concat
            list_of_last_matmul = []

            key_inp_list = [] # used for llama's past_key and past_value output.
            value_inp_list = [] # used for llama's past_key and past_value output.

            # Save sha activation and param name for encoding mapping
            sha_base_attn_node_list = BaseAttnNode(
                qk_matmul = [],
                qk_scale = [],
                qk_softmax = [],
                qk_mask_add = [],
                qk_alibi_add = [],
                qkv_matmul = [],
                qkv_head_concat = [],
                q_matmul = [],
                k_matmul = [],
                v_matmul = [],
            )

            # Reset extension sha name dict lists
            self._rope_extension.reset_sha_encoding_name_list()
            self._past_kv_extension.reset_sha_encoding_name_list()
            self._lora_extension.reset_sha_encoding_name_list()
            self._rmsnorm_extension.reset_sha_encoding_name_list()

            if self.lora_model:
                self.lora_version = get_lora_version(info_dict, self.get_node_by_output_name, self.mha_model_input_names)

                if self.lora_alpha_from_input:
                    log_assert(self.lora_version != LoraVersion.V1,
                               f"lora {LoraVersion.V1} doesn't support --lora-alpha-from-input option")

            query_lora_split = None
            # GQA models will pre-build KV proj nodes before creating sha_pattern and cache them.
            if self.gqa_model:
                self._gqa_extension.reset_and_pre_build_kv_proj_nodes(
                    info_dict,
                    ns,
                    head_dim,
                    key_matmul_inp,
                    value_matmul_inp,
                    sha_base_attn_node_list,
                    key_lora_inp,
                    value_lora_inp)

                if self.lora_model:
                    query_lora_split = self.lora_b_query_split(info_dict, info_dict["num_heads"], head_dim)

            # Split and create attention for each head
            for head_num in range(info_dict["num_heads"]):
                # Create q, k, v matmul for sha heads
                create_sha_func, extension_kwargs = self.create_sha_func_handler(
                    query_lora_inp, key_lora_inp, value_lora_inp
                )
                extension_kwargs.update({"query_lora_split": query_lora_split})
                qkv_matmul, key_inp, value_inp, past_value_inp = create_sha_func(
                    info_dict,
                    ns,
                    head_num,
                    head_dim,
                    query_matmul_inp,
                    key_matmul_inp,
                    value_matmul_inp,
                    sha_base_attn_node_list,
                    **extension_kwargs,
                )
                key_inp_list.append(key_inp)  # key_inp   [1, 1, emd_dim, seq_len]
                value_inp_list.append(value_inp)  # value_inp [1, 1, seq_len, emd_dim]
                list_of_last_matmul.append(qkv_matmul)

            # GQA model uses key and values from cache for model output.
            if self.gqa_model:
                if self.kv_cache and not self.return_new_key_value_only:
                        key_inp_list = self._gqa_extension.key_groups_list # [B, 1, D, L_new+past]
                        value_inp_list = self._gqa_extension.value_groups_list # [B, 1, L_new+past, D]
                else:
                    assert self.nchw_aligned, "GQA only support NCHW aligned model"
                    key_inp_list = self._gqa_extension.key_groups_list_to_return # [B, 1, D, L_new]
                    value_inp_list = self._gqa_extension.value_groups_list_to_return # [B, D, 1, L_new]

            if self.anchor_network:
                assert self.return_new_key_value_only and self.llm_model, f"Detected Anchor Network in model. While Anchor Network supports \
llm model with return_new_key_value_only, but got llm_model: {self.llm_model} and  return_new_key_value_only: {self.return_new_key_value_only}"
                self._anchor_extension.attach_anchor_network(info_dict, key_inp_list)

            # Add past key past value to model output
            if self.handle_past_key_value:
                past_key_name = info_dict["key"]["past_output_name"]
                past_value_name = info_dict["value"]["past_output_name"]
                if self.kv_cache:
                    past_value_index = self.mha_model_input_names_index_dict[
                        past_value_inp
                    ]
                    if self._ar_builder.buildable:
                        past_seq_len_input = self._ar_builder.new_past_seq_len
                    else:
                        past_seq_len_input = (
                            self.model.graph.input[past_value_index]
                            .type.tensor_type.shape.dim[2]
                            .dim_value
                        )

                else:
                    past_seq_len_input = 0

                self._past_kv_extension.add_past_key_value_for_llama_sha(
                    past_key_name,
                    past_value_name,
                    info_dict["num_heads"] if not self.gqa_model else self._gqa_extension.kv_head_num,
                    head_dim,
                    key_inp_list,
                    value_inp_list,
                    past_seq_len_input
                    )

            concat_node = self._op_factory.get_concat_op(list_of_last_matmul, -1)
            self.model.graph.node.append(concat_node)
            sha_base_attn_node_list.qkv_head_concat.append(concat_node)
            self.qkv_head_concat_node_list.append(concat_node)

            # concat output shape: [B, 1, L, D]/[B, 1, H*W, C]
            # Align [B, 1, L, D]/[B, 1, H*W, C] with mha qkv matmul output shape [B, head, H*W, head_dim]
            # [B, 1, H*W, C] -> Reshape -> [B, H*W, head, head_dim]
            reshape_concat_node, reshape_concat_init = self._op_factory.get_reshape_op(concat_node, [1, -1, info_dict["num_heads"], head_dim])
            # [B, H*W, head, head_dim] -> Transpose -> [B,head, H*W, head_dim]
            transpose_concat_node = self._op_factory.get_transpose_op(reshape_concat_node, [0, 2, 1, 3])
            self.model.graph.node.extend([transpose_concat_node, reshape_concat_node])
            self.model.graph.initializer.extend(reshape_concat_init)

            # replace all usage of original final_matmul_output by new concat_node_output
            final_Matmul_output_node_list = info_dict["final_Matmul_output_node_list"]

            origin_final_matmul_output = final_Matmul_output_node_list[0].input[0]
            for node in final_Matmul_output_node_list:
                for arg_i, node_inarg in enumerate(node.input):
                    if node_inarg == origin_final_matmul_output:
                        node.input[arg_i] = transpose_concat_node.output[0]

            # Remove q,k,v linear MatMul and weight.
            # For some reason it solves ONNX failed to use external data bug.
            for _info_dict_key in ["query", "key", "value"]:
                linear_node = info_dict[_info_dict_key]['matmul_node']
                self.model.graph.node.remove(linear_node)
                self.model.graph.initializer.remove(
                    self.get_initializer_by_name[linear_node.input[1]]
                    )

            # update sha info to encoding_mapping_dict
            update_base_attn_sha_encoding_name_to_base_attn_encoding_mapping_dict(encoding_mapping_dict.base_attn, sha_base_attn_node_list)
            if self.handle_rope_ops and self._rope_extension.map_rope_encoding:
                update_rope_sha_encoding_name_to_rope_encoding_mapping_dict(
                    encoding_mapping_dict.rope,
                    self._rope_extension.q_sha_rope_node,
                    self._rope_extension.k_sha_rope_node,
                )

            if self.handle_past_key_value:
                update_past_key_value_sha_encoding_name_to_encoding_mapping_dict(
                    encoding_mapping_dict.past_key_value,
                    self._past_kv_extension.past_key_value_concat,
                )

            if self.lora_model:
                update_lora_sha_encoding_name_to_lora_encoding_mapping_dict(
                    encoding_mapping_dict.lora,
                    self._lora_extension.q_sha_lora_node,
                    self._lora_extension.k_sha_lora_node,
                    self._lora_extension.v_sha_lora_node,
                )

            if self.handle_internal_rmsnorm:
                update_rmsnorm_sha_encoding_name_to_rmsnorm_encoding_mapping_dict(
                    encoding_mapping_dict.rmsnorm,
                    self._rmsnorm_extension.q_rmsnorm_node,
                    self._rmsnorm_extension.k_rmsnorm_node,
                )

            if self.anchor_network:
                create_anchor_network_encoding_mapping_dict(encoding_mapping_dict, self._anchor_extension.mha_node)
                update_anchor_sha_encoding_name_to_anchor_encoding_mapping_dict(
                    encoding_mapping_dict.anchor_network,
                    self._anchor_extension.sha_node
                )

            self.mha_sha_encoding_mapping_dict[start_node_name] = encoding_mapping_dict


        # LLama return "logits" should be 3 dim. Add a reshape to keep dim consist.
        if (not self.replace_linear_with_conv) and "logits" in self.mha_model_output_names:
            to_logit_node = self.get_node_by_output_name["logits"]
            new_to_logit_output_tensor_name, self.tensor_name_set = create_tensor_name(
                "to_last_node_tensor", self.tensor_name_set
            )
            to_logit_node.output[0] = new_to_logit_output_tensor_name
            reshape_to_3d_node, reshape_to_3d_node_init = self._op_factory.get_reshape_op(
                new_to_logit_output_tensor_name, [1, self.seq_len, -1], "logits"
            )
            self.model.graph.initializer.extend(reshape_to_3d_node_init)
            self.model.graph.node.append(reshape_to_3d_node)

        self._ar_builder.update_reshapes()

        if self.replace_linear_with_conv:
            self._update_all_mapping_dicts()  # optimizer doesn't know about any new MatMul nodes.
            self._linear_to_conv.replace()
        else:
            if not self.mha_conv:
                log_warning("Skipping replacemnet of Linears to Convs")

        self.model = topological_sort(clean_model(self.model))
        self._update_all_mapping_dicts()
        if get_model_size(self.model) > 2:
            log_warning("Model is over 2GB and cannot run `native_checker`")
        else:
            status = native_checker(self.model)
            if not status:
                log_error("Model-Checker failed after MHA2SHA Optimization.")

        return self.model

    def mha_to_sha_encoding_mapping(self, mha_encodings, sha_export_path: Path):
        """
        converts a MHA model based encodings to equivalent SHA model encodings
        :param mha_encodings: prepared MHA encodings
        :return: SHA encodings filepath.
        """
        sha_encodings = self._encoding_mapper.map_encodings(mha_encodings, sha_export_path)

        return sha_encodings
