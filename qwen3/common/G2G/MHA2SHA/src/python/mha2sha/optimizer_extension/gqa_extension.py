# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from dataclasses import dataclass
import numpy as np
from typing import Any, Dict, Tuple
from onnx import numpy_helper, helper
from onnx.onnx_pb import NodeProto, TensorProto
from mha2sha.utils.utils import BranchType

from mha2sha.utils.logger import log_debug, log_error, log_info, log_warning
from mha2sha.utils.onnx import (NodeNotFoundError, get_next_node_up_based_on_cond,
                                get_next_node_down_based_on_cond,
                                get_node_input_constant_op_value,
                                get_initializer_value)

from mha2sha.utils.utils import sha_node_name_basis, BranchType

mha2sha_hf_model_optimizer = Any  # Causes circular import

class GqaExtension:
    """ Extenstion helpers for mha2sha_optimzer to bridge Morpheus pipeline code base and v1.0.0 release.  """
    def __init__(self, mha2sha_optim: mha2sha_hf_model_optimizer) -> None:
        """Initalizes an instance based on the MHA2SHA optimizer provided.

        Args:
            mha2sha_optim:
                MHA2SHAOptimizer instance holding the model loader and model info.
        """
        self.mha2sha_optim = mha2sha_optim

    def get_kv_group_head_num(self,
                              k_proj_end_node,
                              key_matmul_node,
                              qk_matmul_node):
        """
        Get head_num for kv branch by search
        the transpose(Potential) -> reshape pattern.

        :param k_proj_end_node: last node of k projection
                    - for k-projection with lora: this is lora_add node
                    - for k-projection without lora: this is key_matmul_node
        :param key_matmul_node: key matmul
        :param qk_matmul_node: qk_matmul_node
        :return number_of_heads: kv branch head num
        """
        # Use the first reshape node after k_proj_end_node (k proj or k lora add) before qk_matmul
        reshape_node = get_next_node_down_based_on_cond(
                                    k_proj_end_node,
                                    self.mha2sha_optim.get_node_by_input_name,
                                    node_found_cond=lambda n: n.op_type == "Reshape",
                                    node_end_search_cond=lambda n: n == qk_matmul_node
                                )
        matmul_input_shape = get_node_input_constant_op_value(reshape_node,
                                                              self.mha2sha_optim.get_node_by_output_name,
                                                              self.mha2sha_optim.get_initializer_by_name)

        if self.mha2sha_optim.mha_conv and self.mha2sha_optim.nchw_aligned:
            inter_transpose = None
            try:
                inter_transpose = get_next_node_down_based_on_cond(
                    key_matmul_node,
                    self.mha2sha_optim.get_node_by_input_name,
                    node_found_cond=lambda n: n.op_type == "Transpose",
                    node_end_search_cond=lambda n: n == reshape_node
                )
            except NodeNotFoundError:
                ... # inter_transpose is None by default and handled below

            number_of_heads = self.mha2sha_optim.get_head_num_for_conv_model(
                matmul_input_shape, inter_transpose
            )

        else:
            # MHA linear will reshape to [B, h*w, head_num, head_dim]
            number_of_heads = matmul_input_shape[-2]

        return number_of_heads

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
            if self.mha2sha_optim.mha_conv:
                query_proj, key_proj, value_proj = self.mha2sha_optim._mha_conv_extension.get_qkv_conv(qk_matmul_node, qkv_matmul_node)
                query_initializer = self.mha2sha_optim._mha_conv_extension.get_conv_weight_in_OI(query_proj)
                key_initializer = self.mha2sha_optim._mha_conv_extension.get_conv_weight_in_OI(key_proj)
                value_initializer = self.mha2sha_optim._mha_conv_extension.get_conv_weight_in_OI(value_proj)

            else:
                query_proj, key_proj, value_proj = self.mha2sha_optim._mha_conv_extension.get_qkv_matmuls(qk_matmul_node, qkv_matmul_node)
                query_initializer = numpy_helper.to_array(
                    self.get_initializer_by_name[query_proj.input[1]]
                )
                key_initializer = numpy_helper.to_array(
                    self.get_initializer_by_name[key_proj.input[1]]
                )
                value_initializer = numpy_helper.to_array(
                    self.get_initializer_by_name[value_proj.input[1]]
                )

            dquery = {"matmul_node": query_proj,
                      "matmul_init": query_initializer}

            dkey = {"matmul_node": key_proj,
                    "matmul_init": key_initializer}

            dvalue = {"matmul_node": value_proj,
                      "matmul_init": value_initializer}

            self.kv_head_num = self.get_kv_group_head_num(key_proj, qk_matmul_node)

            return dquery, dkey, dvalue

    def _reset_cached_keys_value_nodes(self):
        """ GQA model cache sha key and value Matmul nodes. """
        self.key_groups_list = []
        self.value_groups_list = []

    def reset_and_pre_build_kv_proj_nodes(
            self,
            info_dict,
            ns,
            head_dim,
            key_matmul_inp,
            value_matmul_inp,
            sha_base_attn_node_list,
            key_lora_inp,
            value_lora_inp
        ):
        """
        Reset self.key_groups_list and self.value_groups_list. Then create all k, v proj nodes before
        creating SHA. Add rope node if needed.
        :key_groups_list: each key has shape [B, 1, seq_len, head_dim], [B, 1, head_dim, seq_len] if handle_rope_ops
        :value_groups_list: each value has shape [B, head_dim, 1, seq_len], [B, 1, seq_len, head_dim] if handle_rope_ops
        """
        assert self.mha2sha_optim.mha_conv, "support only mha-conv model"

        self._reset_cached_keys_value_nodes()

        key_lora_split = value_lora_split = None

        if self.mha2sha_optim.lora_model:
            if "mha_lora_node" in info_dict["key"] and "mha_lora_node" in info_dict["value"]:
                init_name = "lora_b_init"
                lora_key_node = info_dict["key"]["mha_lora_node"]
                lora_value_node = info_dict["value"]["mha_lora_node"]

                if init_name not in info_dict["key"] and lora_key_node.lora_b.op_type == "MatMul" \
                        and init_name not in info_dict["value"] and lora_value_node.lora_b.op_type == "MatMul":
                    key_lora_split = self.mha2sha_optim._lora_extension.split_lora_b_adaptor_output(info_dict["key"],
                                                                                                    self.kv_head_num,
                                                                                                    head_dim,
                                                                                                    )
                    value_lora_split = self.mha2sha_optim._lora_extension.split_lora_b_adaptor_output(info_dict["value"],
                                                                                                      self.kv_head_num,
                                                                                                      head_dim,
                                                                                                      )

        for head_num in range(self.kv_head_num):
            # Create weight names
            if "matmul_init" in info_dict["key"].keys() and (conv_weight_init := info_dict["key"]["matmul_init"]) is not None:
                key_conv = self.mha2sha_optim._mha_conv_extension.create_single_conv(
                                conv_weight_init,
                                ns,
                                head_num,
                                head_dim,
                                key_matmul_inp,
                                suffix=None,
                                branch_type=BranchType.K,
                                bias_init=info_dict["key"].get("matmul_init_bias", None)
                            )
            else:
                raise ValueError("key matmul weight is None")

            if "matmul_init" in info_dict["value"].keys() and (conv_weight_init := info_dict["value"]["matmul_init"]) is not None:
                value_conv = self.mha2sha_optim._mha_conv_extension.create_single_conv(
                                conv_weight_init,
                                ns,
                                head_num,
                                head_dim,
                                value_matmul_inp,
                                suffix=None,
                                branch_type=BranchType.V,
                                bias_init=info_dict["value"].get("matmul_init_bias", None)
                            )
            else:
                raise ValueError("value matmul weight is None")

            sha_base_attn_node_list.k_matmul.append(key_conv)
            sha_base_attn_node_list.v_matmul.append(value_conv)

            if self.mha2sha_optim.lora_model:
                key_conv = self.mha2sha_optim._lora_extension.attach_single_lora_adaptor(
                                                    info_dict["key"],
                                                    ns,
                                                    head_num,
                                                    head_dim,
                                                    key_conv,
                                                    key_lora_inp,
                                                    BranchType.K,
                                                    key_lora_split
                                                )
                value_conv = self.mha2sha_optim._lora_extension.attach_single_lora_adaptor(
                                                    info_dict["value"],
                                                    ns,
                                                    head_num,
                                                    head_dim,
                                                    value_conv,
                                                    value_lora_inp,
                                                    BranchType.V,
                                                    value_lora_split
                                                )

            if self.mha2sha_optim.handle_rope_ops:
                cos_node = info_dict["rope_cos_model_input"]
                sin_node = info_dict["rope_sin_model_input"]
                if self.mha2sha_optim.handle_internal_rmsnorm:
                    key_conv = self.mha2sha_optim._rmsnorm_extension.create_sha_rmsnorm(
                        key_conv, BranchType.K, self.kv_head_num, head_dim
                    )
                key_conv = self.mha2sha_optim._rope_extension.create_llama_rope_node(key_conv, cos_node, sin_node, head_dim,
                                                                                     BranchType.K, info_dict)

            self.key_groups_list.append(key_conv)
            self.value_groups_list.append(value_conv)

        if self.mha2sha_optim.handle_past_key_value and self.mha2sha_optim.mha_conv:
            # Pre-transpose K, V for LLM-conv model and concate past_key and value to new key and value
            self.post_process_kv_for_rope(info_dict, ns)

    def post_process_kv_for_rope(self, info_dict, ns):
        """
        Post process k input and q input for rope, reshape k to [B, 1, D, L] and reshape v to [B, 1, L, D].
        create self.key_groups_list_to_return and self.value_groups_list_to_return: list of k, v for model
        output.

        :param self.key_groups_list: [B, 1, L, D]
        :param self.value_groups_list: [B, D, 1, L]

        :update self.key_groups_list: [B, 1, D, L]
        :update self.value_groups_list: [B, 1, L, D]
        :update self.key_groups_list_to_return: past_key model output [B, 1, D, L]
        :update self.value_groups_list_to_return: past_value model output [B, D, 1, L]
        """
        key_groups_list_temp = []
        value_groups_list_temp = []
        # add_past_key_value_for_llama_sha expects v in [B, 1, L, D] or [B, D, 1, L] is NCHW aligned
        self.key_groups_list_to_return = []
        self.value_groups_list_to_return = self.value_groups_list

        for kv_head_num, (key_node, value_node) in enumerate(zip(self.key_groups_list, self.value_groups_list)):
            _, _, propose_sha_key_name, propose_sha_value_name = sha_node_name_basis(ns, kv_head_num)

            # Transpose K: [B, 1, L, D] -> [B, 1, D, L]
            key_transpose = self.mha2sha_optim._op_factory.get_transpose_op(key_node, [0, 1, 3, 2], propose_sha_key_name+"_Transpose")
            self.mha2sha_optim.model.graph.node.append(key_transpose)
            self.key_groups_list_to_return.append(key_transpose) # return post transpose pre concat keys

            # Transpose V: [B, D, 1, L] -> [B, 1, L, D]
            value_transpose = self.mha2sha_optim._op_factory.get_transpose_op(value_node, [0, 2, 3, 1], propose_sha_value_name+"_Transpose")
            self.mha2sha_optim.model.graph.node.append(value_transpose)

            past_key_inp = None
            past_value_inp = None
            # Concate past key input and past value output
            if self.mha2sha_optim.kv_cache:
                past_key_inp = info_dict["key"]["past_input_name"]
                past_value_inp = info_dict["value"]["past_input_name"]
                # concat_past_key_value_input expects key_inp: [B, 1, D, L]; value_inp = [B, 1, L, D]
                key_transpose, value_transpose = self.mha2sha_optim._past_kv_extension.concat_past_key_value_input(
                    past_key_inp, past_value_inp, key_transpose, value_transpose, kv_head_num, info_dict["head_dim"]
                    )
            self.past_value_inp = past_value_inp

            key_groups_list_temp.append(key_transpose)
            value_groups_list_temp.append(value_transpose)

            self.key_groups_list = key_groups_list_temp
            self.value_groups_list = value_groups_list_temp

    def create_sha_convs_with_gqa_rope(self,
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
