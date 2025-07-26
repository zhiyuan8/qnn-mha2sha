# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from typing import Any, Dict, Tuple
from onnx import numpy_helper
from onnx.onnx_pb import NodeProto

from mha2sha.utils.logger import log_debug, log_error, log_info, log_warning
from mha2sha.utils.op_factory import create_tensor_name
from mha2sha.utils.onnx import get_next_node_up_based_on_cond
from mha2sha.utils.utils import sha_node_name_basis, BranchType

mha2sha_hf_model_optimizer = Any  # Causes circular import

class MhaConvExtension:
    """ Extenstion helpers for mha2sha_optimzer to bridge Morpheus pipeline code base and v1.0.0 release.  """
    def __init__(self, mha2sha_optim: mha2sha_hf_model_optimizer) -> None:
        """Initalizes an instance based on the MHA2SHA optimizer provided.

        Args:
            mha2sha_optim:
                MHA2SHAOptimizer instance holding the model loader and model info.
        """
        self.mha2sha_optim = mha2sha_optim

    def get_qkv_conv(self, qk_matmul_node, qkv_matmul_node):
        """
        Q = Query, K = Key, V = Value

        Function to search up the the branches of the QK MatMul to find the QK's origin Conv's and search the QKV
        Conv to find V's origin Conv.
            +----------+      +----------+
            |  Q Conv  |      |  K Conv  |
            +----------+      +----------+
                  |                 |
                  |                 |
                   \               /
                 input 0       input 1
                      +-----------+
                      | QK MatMul |
                      +-----------+
                            |               +----------+
                            |               |  V Conv  |
                        intput 0            +----------+
                      +------------+              |
                      | QKV MatMul | -- input 1 - |
                      +------------+

        :param NodeProto qk_matmul_node: The MatMul node where the Q and K branches meet.
        :param NodeProto qkv_matmul_node: The MatMul node where the Q, K, and V branches meet.
        :return List[NodeProto]: The Q, K, and V origin MatMuls.
        """
        q_conv = get_next_node_up_based_on_cond(
                self.mha2sha_optim.get_node_by_output_name[qk_matmul_node.input[0]],
                    self.mha2sha_optim.get_node_by_output_name,
                    node_found_cond=lambda n: n.op_type == "Conv"
                )
        k_conv = get_next_node_up_based_on_cond(
                    self.mha2sha_optim.get_node_by_output_name[qk_matmul_node.input[1]],
                    self.mha2sha_optim.get_node_by_output_name,
                    node_found_cond=lambda n: n.op_type == "Conv"
                )
        if self.mha2sha_optim.handle_r3_matrix:
            # in this case, the q_conv, k_conv found above are actually convs of R3
            q_r3_matrix = q_conv
            k_r3_matrix = k_conv
            # so we need to find previous convs of these R3 convs
            q_conv = get_next_node_up_based_on_cond(
                        self.mha2sha_optim.get_node_by_output_name[q_r3_matrix.input[0]],
                        self.mha2sha_optim.get_node_by_output_name,
                        node_found_cond=lambda n: n.op_type == "Conv"
                    )
            k_conv = get_next_node_up_based_on_cond(
                        self.mha2sha_optim.get_node_by_output_name[k_r3_matrix.input[0]],
                        self.mha2sha_optim.get_node_by_output_name,
                        node_found_cond=lambda n: n.op_type == "Conv"
                    )
        v_conv = get_next_node_up_based_on_cond(
            self.mha2sha_optim.get_node_by_output_name[qkv_matmul_node.input[1]],
            self.mha2sha_optim.get_node_by_output_name,
            node_found_cond=lambda n: n.op_type == "Conv"
        )
        return [q_conv, k_conv, v_conv]

    def get_conv_weight_in_OI(self, conv_node):
        conv_weight_init = numpy_helper.to_array(
                self.mha2sha_optim.get_initializer_by_name[conv_node.input[1]]
            )
        conv_weight_init = conv_weight_init.reshape(conv_weight_init.shape[:2]).T
        return conv_weight_init

    def get_conv_bias_in_OI(self, conv_node):
        if len(conv_node.input) <= 2:
            return None
        conv_bias_init = numpy_helper.to_array(
                self.mha2sha_optim.get_initializer_by_name[conv_node.input[2]]
            )
        return conv_bias_init

    def get_dqkv_for_qkv_info(self, conv_node):
        assert conv_node.op_type == "Conv", f"expect node op_type == Conv, but got {conv_node.op_type}"
        conv_initializer = self.get_conv_weight_in_OI(conv_node)
        conv_bias_initializer = self.get_conv_bias_in_OI(conv_node)
        return {"matmul_node": conv_node,
                "matmul_init": conv_initializer,
                "matmul_init_bias": conv_bias_initializer}

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
            # Still call them matmuls although they are actually convs
            query_conv, key_conv, value_conv = self.get_qkv_conv(qk_matmul_node, qkv_matmul_node)

            if not (query_conv and key_conv and value_conv):
                log_debug("Cannot find QKV Matmuls.")
            else:
                log_debug(
                    f"QKV MatMuls : {query_conv.name}, {key_conv.name}, {value_conv.name}"
                )
            dquery = self.get_dqkv_for_qkv_info(query_conv)
            dkey = self.get_dqkv_for_qkv_info(key_conv)
            dvalue = self.get_dqkv_for_qkv_info(value_conv)

            return dquery, dkey, dvalue

    def create_single_conv(self,
                           qkv_init,
                           ns,
                           head_num,
                           head_dim,
                           qkv_matmul_input,
                           suffix=None,
                           branch_type=None,
                           bias_init=None
                           ):

        sha_qkv_name = sha_node_name_basis(ns, head_num)[branch_type.value]
        sha_weight_name = sha_qkv_name+".weight" if suffix is None else sha_qkv_name+f"_{suffix}.weight"
        sha_bias_name = sha_qkv_name+".bias" if suffix is None else sha_qkv_name+f"_{suffix}.bias"

        # Create weight names
        conv_init_name, self.mha2sha_optim.tensor_name_set = create_tensor_name(
            sha_weight_name, self.mha2sha_optim.tensor_name_set
        )
        assert sha_weight_name == conv_init_name, f"proposed sha_weight_name and created init name are differnet,\
            got {sha_weight_name} and {conv_init_name}"
        bias_init_name, self.mha2sha_optim.tensor_name_set = create_tensor_name(
            sha_bias_name, self.mha2sha_optim.tensor_name_set
        )

        # Create q, k, v conv weight
        # info_dict["query"]["matmul_init"] has shape [I, O]
        # Conv weight has shape [head_dim, I, kH, kW]
        conv_init = numpy_helper.from_array(
            qkv_init[
                :, head_num * head_dim : (head_num + 1) * head_dim
                ].T[..., None, None],
            conv_init_name,
        )

        # create bias if exists
        if bias_init is not None:
            bias_init = numpy_helper.from_array(
                                bias_init[head_num * head_dim : (head_num + 1) * head_dim],
                                bias_init_name,
                        )
        else:
            bias_init_name = None

        # Create q, k, v matmul op
        conv_node = self.mha2sha_optim._op_factory.get_conv_op(
                        input_node=qkv_matmul_input,
                        weight_tensor_name=conv_init_name,
                        bias_tensor_name=bias_init_name,
                        kernel_shape=[1, 1],
                        padding=[0, 0, 0, 0],
                        strides=[1, 1],
                        propose_op_name=sha_qkv_name+"_Conv" if suffix is None else sha_qkv_name+f"_{suffix}_Conv",
                        output_tensor_name=None
                    )
        self.mha2sha_optim.model.graph.initializer.append(conv_init)
        if bias_init is not None:
            self.mha2sha_optim.model.graph.initializer.append(bias_init)
        self.mha2sha_optim.model.graph.node.append(conv_node)

        return conv_node

    def create_sha_qkv_convs(self,
                             info_dict,
                             ns,
                             head_num,
                             head_dim,
                             sha_base_attn_node_list,
                             query_matmul_inp,
                             key_matmul_inp,
                             value_matmul_inp,
                             init_name="matmul_init",
                             suffix=None,
                             ):
        """
        Create MatMuls with initializer sliced by head.
        :return query_inp: query linear node
        :return key_inp: key linear node
        :return value_inp: value linear node
        """
        _, propose_sha_query_name, propose_sha_key_name, propose_sha_value_name = sha_node_name_basis(ns, head_num)

        # Create weight names
        query_conv = None
        key_conv = None
        value_conv = None
        if init_name in info_dict["query"].keys() and (conv_weight_init := info_dict["query"][init_name]) is not None:
            query_conv = self.create_single_conv(
                            conv_weight_init,
                            ns,
                            head_num,
                            head_dim,
                            query_matmul_inp,
                            suffix=suffix,
                            branch_type=BranchType.Q,
                            bias_init=info_dict["query"].get(init_name+"_bias", None)
                        )

        if init_name in info_dict["key"].keys() and (conv_weight_init := info_dict["key"][init_name]) is not None:
            key_conv = self.create_single_conv(
                            conv_weight_init,
                            ns,
                            head_num,
                            head_dim,
                            key_matmul_inp,
                            suffix=suffix,
                            branch_type=BranchType.K,
                            bias_init=info_dict["key"].get(init_name+"_bias", None)
                        )

        if init_name in info_dict["value"].keys() and (conv_weight_init := info_dict["value"][init_name]) is not None:
            value_conv = self.create_single_conv(
                            conv_weight_init,
                            ns,
                            head_num,
                            head_dim,
                            value_matmul_inp,
                            suffix=suffix,
                            branch_type=BranchType.V,
                            bias_init=info_dict["value"].get(init_name+"_bias", None)
                        )

        if suffix is None:
            sha_base_attn_node_list.q_matmul.append(query_conv)
            sha_base_attn_node_list.k_matmul.append(key_conv)
            sha_base_attn_node_list.v_matmul.append(value_conv)

        if suffix is None and (query_conv or key_conv or value_conv) is None:
            raise ValueError(f"Can not find Q, K, or V got: {query_conv, key_conv, value_conv}")

        query_inp = query_conv
        key_inp = key_conv
        value_inp = value_conv

        return query_inp, key_inp, value_inp

    def create_sha_conv_with_rope(self,
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
        :param sha_base_attn_node_list: sha encoding name dict
        :return qkv_matmul: qkv_matmul node
        :return _key_inp: _key_inp for llama to return cached keys
        :return _value_inp: _value_inp for llama to return cached keys
        :return past_value_inp: llama kv$ model value model input
        """
        propose_sha_name, _, _, _ = sha_node_name_basis(ns, head_num)

        propose_sha_name, _, propose_sha_key_name, propose_sha_value_name = sha_node_name_basis(ns, head_num)

        # Step 0: Transpose from NHWC to NCHW
        if not self.mha2sha_optim.nchw_aligned:
            # if query_matmul_inp, key_matmul_inp, value_matmul_inp are the same node, use shared transpose
            if query_matmul_inp == key_matmul_inp == value_matmul_inp:
                shared_NCHW_transpose_inp =  self.mha2sha_optim._op_factory.get_transpose_op(query_matmul_inp, [0, 3, 1, 2]) # [N, H, W, C] to [N, C, H, W]
                self.mha2sha_optim.model.graph.node.append(shared_NCHW_transpose_inp)
                query_matmul_inp = shared_NCHW_transpose_inp
                key_matmul_inp = shared_NCHW_transpose_inp
                value_matmul_inp = shared_NCHW_transpose_inp
            else:
                # Transpose qkv input from [B, 1, L, D]/[B, H, W, C] to [B, D, 1, L]/[B, C, H, W]
                query_matmul_inp =  self.mha2sha_optim._op_factory.get_transpose_op(query_matmul_inp, [0, 3, 1, 2]) # [N, H, W, C] to [N, C, H, W]
                key_matmul_inp =  self.mha2sha_optim._op_factory.get_transpose_op(key_matmul_inp, [0, 3, 1, 2]) # [N, H, W, C] to [N, C, H, W]
                value_matmul_inp =  self.mha2sha_optim._op_factory.get_transpose_op(value_matmul_inp, [0, 3, 1, 2]) # [N, H, W, C] to [N, C, H, W]
                self.mha2sha_optim.model.graph.node.extend([query_matmul_inp, key_matmul_inp, value_matmul_inp])

        # Step 1 Create sha Convs nodes with conv weight initializers.
        # query_inp, key_inp, value_inp: [B, D, 1, L] -> Conv -> [B, D, 1, L]
        if self.mha2sha_optim.gqa_model:
            # KV_out at gqa_extension.reset_and_pre_build_kv_proj_nodes.post_process_kv_for_rope.concat_past_key_value_input
            if "matmul_init" in info_dict["query"].keys() and (conv_weight_init := info_dict["query"]["matmul_init"]) is not None:
                query_inp = self.create_single_conv(
                                conv_weight_init,
                                ns,
                                head_num,
                                head_dim,
                                query_matmul_inp,
                                suffix=None,
                                branch_type=BranchType.Q,
                                bias_init=info_dict["query"].get("matmul_init_bias", None)
                            )
                sha_base_attn_node_list.q_matmul.append(query_inp)
            else:
                raise ValueError("query matmul weight is None")

            if self.mha2sha_optim.handle_internal_rmsnorm:
                query_inp = self.mha2sha_optim._rmsnorm_extension.create_sha_rmsnorm(
                    query_inp, BranchType.Q, head_num, head_dim
                )

            # Step 2: Add lora adaptor (Optional)
            if self.mha2sha_optim.lora_model:
                query_inp = self.mha2sha_optim._lora_extension.attach_single_lora_adaptor(
                                                    info_dict["query"],
                                                    ns,
                                                    head_num,
                                                    head_dim,
                                                    query_inp,
                                                    extenstion_kwargs["query_lora_inp"],
                                                    BranchType.Q,
                                                    extenstion_kwargs["query_lora_split"],
                                                )


            # Step 3 Insert ROPE
            # Prepared llama sin and cos are model input.
            cos_node = info_dict["rope_cos_model_input"]
            sin_node = info_dict["rope_sin_model_input"]
            # create_llama_rope_node will check if nchw_aligned then slice on c dim -> transpoe to NHWC accordingly.
            query_inp = self.mha2sha_optim._rope_extension.create_llama_rope_node(query_inp, cos_node, sin_node, head_dim,
                                                                                  BranchType.Q, info_dict)

            # GQA model uses key inp and value inp from gqa extension cache.
            assert len(self.mha2sha_optim._gqa_extension.key_groups_list) > 0
            assert len(self.mha2sha_optim._gqa_extension.value_groups_list) > 0

            # e.g. q: 32 head, k, v has 8 heads, then: group_num = 4
            # [q0, q1, q2, q3] -> k0, v0; [q4, q5, q6, q7] -> k1, v1
            # gqa_extension.key_groups_list: [B, 1, D, L]
            # gqa_extension.value_groups_list: [B, 1, L, D]
            head_num_per_group = info_dict["num_heads"]//self.mha2sha_optim._gqa_extension.kv_head_num
            key_transpose = self.mha2sha_optim._gqa_extension.key_groups_list[int(head_num//head_num_per_group)]
            value_inp = self.mha2sha_optim._gqa_extension.value_groups_list[int(head_num//head_num_per_group)]

            # GQA model uses gqa_extension.key_groups_list_to_return and gqa_extension.value_groups_list_to_return
            # as model's past key and past value output
            _key_inp, _value_inp = None, None
            past_value_inp = self.mha2sha_optim._gqa_extension.past_value_inp

        else:
            query_inp, key_inp, value_inp = self.create_sha_qkv_convs(
                                                info_dict,
                                                ns,
                                                head_num,
                                                head_dim,
                                                sha_base_attn_node_list,
                                                query_matmul_inp,
                                                key_matmul_inp,
                                                value_matmul_inp
                                            )

            # Step 2: Add lora adaptor (Optional)
            if self.mha2sha_optim.lora_model and not self.mha2sha_optim.gqa_model:
                query_inp, key_inp, value_inp = self.mha2sha_optim._lora_extension.attach_lora_adaptor(
                                                    info_dict,
                                                    ns,
                                                    head_num,
                                                    head_dim,
                                                    sha_base_attn_node_list,
                                                    query_inp,
                                                    key_inp,
                                                    value_inp,
                                                    extenstion_kwargs["query_lora_inp"],
                                                    extenstion_kwargs["key_lora_inp"],
                                                    extenstion_kwargs["value_lora_inp"]
                                                )

            # Step 3 Insert ROPE
            # Prepared llama sin and cos are model input.
            cos_node = info_dict["rope_cos_model_input"]
            sin_node = info_dict["rope_sin_model_input"]
            # create_llama_rope_node will check if nchw_aligned then slice on c dim -> transpoe to NHWC accordingly.
            query_inp = self.mha2sha_optim._rope_extension.create_llama_rope_node(query_inp, cos_node, sin_node, head_dim,
                                                                                  BranchType.Q, info_dict) # [B, 1, L, D]
            key_inp = self.mha2sha_optim._rope_extension.create_llama_rope_node(key_inp, cos_node, sin_node, head_dim,
                                                                                BranchType.K, info_dict) # [B, 1, L, D]
            # value_inp: [B, D, 1, L]

            # Step 4 Transpose K
            # [B, 1, L, D] -> [B, 1, D, L]
            key_transpose = self.mha2sha_optim._op_factory.get_transpose_op(key_inp, [0, 1, 3, 2], propose_sha_key_name+"_Transpose")
            self.mha2sha_optim.model.graph.node.append(key_transpose)

            # Step 5 Concate pask key value input
            # transpose value: [B, D, 1, L] -> [B, 1, L, D]
            # Capture new key value to return
            _key_inp = key_transpose # [B, 1, D, L]
            _value_inp = value_inp   # [B, D, 1, L] -> goes to concate -> transpose

            # [B, D, 1, L] -> [B, 1, L, D]
            value_transpose = self.mha2sha_optim._op_factory.get_transpose_op(value_inp, [0, 2, 3, 1], propose_sha_value_name+"_Transpose")
            self.mha2sha_optim.model.graph.node.append(value_transpose)
            value_inp = value_transpose # [B, 1, L, D]

            past_key_inp = None
            past_value_inp = None
            # Concate past key input and past value output
            if self.mha2sha_optim.handle_past_key_value and self.mha2sha_optim.kv_cache:
                past_key_inp = info_dict["key"]["past_input_name"]
                past_value_inp = info_dict["value"]["past_input_name"]
                # key_transpose: [B, 1, D, L]; value_inp = [B, 1, L, D]
                key_transpose, value_inp = (
                    self.mha2sha_optim._past_kv_extension.concat_past_key_value_input(
                        past_key_inp, past_value_inp, key_transpose, value_inp, head_num, info_dict["head_dim"]
                    )
                )

            if not self.mha2sha_optim.return_new_key_value_only:
                # Update _key_inp with concat key and value
                _key_inp = key_transpose # [1, 1, emd_dim, past+new seq_len]
                _value_inp = value_inp # [1, 1, seq_len, past+new emd_dim]

        # Step 6. QK matmul
        # query_inp: [B, 1, L, D]
        # key_transpose: [B, 1, D, L]
        # value_inp: [B, 1, L, D]
        qk_matmul_node = self.mha2sha_optim._op_factory.get_matmul_op(query_inp, key_transpose, propose_sha_name+"_qk_MatMul")
        self.mha2sha_optim.model.graph.node.append(qk_matmul_node)
        sha_base_attn_node_list.qk_matmul.append(qk_matmul_node)

        sha_nodes_list = []
        sha_nodes_list.append(qk_matmul_node)

        # Step 7. Create sha pattern
        qkv_matmul = self.mha2sha_optim.create_sha_pattern(info_dict, ns, head_num, value_inp, sha_nodes_list, sha_base_attn_node_list)
        self.mha2sha_optim.model.graph.node.extend([qkv_matmul])

        return qkv_matmul, _key_inp, _value_inp, past_value_inp

    def create_sha_conv(self,
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
        Create efficient sha-conv.
        :param info_dict: mha info dict
        :param ns: number of start node (attention layer num)
        :param head_num: head number in heads
        :param head_dim: vector dim for each head
        :param query_matmul_inp: input to query linear
        :param key_matmul_inp: input to key linear
        :param value_matmul_inp: input to value linear
        :param sha_encoding_name_dict: sha encoding name dict

        mha-conv does not support handle-rope-ops. Return None on _key_inp, _value_inp, past_value_inp
        :return qkv_matmul: MatMul Op in SHA for (QK, V), should have output shape (B, H, W, C)
        :return _key_inp: None
        :return _value_inp: None
        :return past_value_inp: None
        """
        propose_sha_name, _, _, _ = sha_node_name_basis(ns, head_num)

        # Step 0: Transpose from NHWC to NCHW
        if not self.mha2sha_optim.nchw_aligned:
            # Transpose qkv input from [B, 1, L, D]/[B, H, W, C] to [B, D, 1, L]/[B, C, H, W]
            query_matmul_inp =  self.mha2sha_optim._op_factory.get_transpose_op(query_matmul_inp, [0, 3, 1, 2]) # [N, H, W, C] to [N, C, H, W]
            key_matmul_inp =  self.mha2sha_optim._op_factory.get_transpose_op(key_matmul_inp, [0, 3, 1, 2]) # [N, H, W, C] to [N, C, H, W]
            value_matmul_inp =  self.mha2sha_optim._op_factory.get_transpose_op(value_matmul_inp, [0, 3, 1, 2]) # [N, H, W, C] to [N, C, H, W]
            self.mha2sha_optim.model.graph.node.extend([query_matmul_inp, key_matmul_inp, value_matmul_inp])

        # Step 1 Create sha Convs nodes with conv weight initializers.
        query_inp, key_inp, value_inp = self.create_sha_qkv_convs(info_dict,
                                                                  ns,
                                                                  head_num,
                                                                  head_dim,
                                                                  sha_base_attn_node_list,
                                                                  query_matmul_inp,
                                                                  key_matmul_inp,
                                                                  value_matmul_inp)
        # Step 2 Add LORA adaptor
        if self.mha2sha_optim.lora_model:
            query_inp, key_inp, value_inp = self.mha2sha_optim._lora_extension.attach_lora_adaptor(
                                                info_dict,
                                                ns,
                                                head_num,
                                                head_dim,
                                                sha_base_attn_node_list,
                                                query_inp,
                                                key_inp,
                                                value_inp,
                                                extenstion_kwargs["query_lora_inp"],
                                                extenstion_kwargs["key_lora_inp"],
                                                extenstion_kwargs["value_lora_inp"]
                                            )

        # step 4 Add scale on q branch if needed (Optional for LVM models)
        if self.mha2sha_optim._scale_on_q_branch:
            query_inp = self.mha2sha_optim.add_scale_on_q_branch(info_dict, query_inp, propose_sha_name, sha_base_attn_node_list)

        # Step 5 Create efficient Conv Pattern:
        """
        MHA-Conv is expected not to have scale after query conv.
        Expecting have scale folded into query and key weights.
        input has shape (B, C, H, W)

        query_inp = conv(query_inp) # -> (B, head_dim, H, W)
        key_inp = conv(key_inp) # -> (B, head_dim, H, W)
        value_inp = conv(value_inp) # -> (B, head_dim, H, W)

        query = query_inp.permute(0, 2, 3, 1) # -> (B, H, W, head_dim)
        key_T = key_inp.reshape(batch_size, 1, head_dim, -1) # -> (B, 1, head_dim, H*W)
        value = value.permute(0, 2, 3, 1).reshape(batch_size, 1, -1, head_dim) # -> (B, H, W, head_dim) -> (B, 1, H*W, head_dim)

        _attn = MatMul(_query, _key_T) # -> (B, H, W, H*W)
        _attn = Softmax(_attn, dim=-1) # -> (B, H, W, H*W)
        _out = MatMul(_attn, value) # -> (B, H, W, head_dim)
        """
        # Step 6 Concate pask key value input
        query_transpose_node = self.mha2sha_optim._op_factory.get_transpose_op(query_inp, [0, 2, 3, 1]) # (B, head_dim, H, W) -> (B, H, W, head_dim)
        self.mha2sha_optim.model.graph.node.append(query_transpose_node)

        key_reshape_node, key_reshape_init = self.mha2sha_optim._op_factory.get_reshape_op(key_inp, [1, 1, head_dim, -1]) # (B, head_dim, H, W) -> (B, 1, head_dim, H*W)
        self.mha2sha_optim.model.graph.node.append(key_reshape_node)
        self.mha2sha_optim.model.graph.initializer.extend(key_reshape_init)  # is a list so extending

        _key_inp = key_reshape_node
        _value_inp = value_inp

        value_transpose_node = self.mha2sha_optim._op_factory.get_transpose_op(value_inp, [0, 2, 3, 1]) # (B, head_dim, H, W) -> (B, H, W, head_dim)
        value_reshape_node, value_reshape_init = self.mha2sha_optim._op_factory.get_reshape_op(value_transpose_node, [1, 1, -1, head_dim]) # (B, H, W, head_dim) -> (B, 1, H*W, head_dim)
        value_inp = value_reshape_node
        self.mha2sha_optim.model.graph.node.extend([value_transpose_node, value_reshape_node])
        self.mha2sha_optim.model.graph.initializer.extend(value_reshape_init)  # is a list so extending

        past_key_inp = None
        past_value_inp = None
        # Concate past key input and past value output
        if self.mha2sha_optim.handle_past_key_value and self.mha2sha_optim.kv_cache:
            past_key_inp = info_dict["key"]["past_input_name"]
            past_value_inp = info_dict["value"]["past_input_name"]
            # key_transpose: [B, 1, D, L]; value_inp = [B, 1, L, D]
            key_reshape_node, value_inp = (
                self.mha2sha_optim._past_kv_extension.concat_past_key_value_input(
                    past_key_inp, past_value_inp, key_reshape_node, value_inp, head_num, head_dim
                )
            )

        if not self.mha2sha_optim.return_new_key_value_only:
            # Update _key_inp with concat key and value
            _key_inp = key_reshape_node # [1, 1, emd_dim, past+new seq_len]
            _value_inp = value_inp # [1, 1, seq_len, past+new emd_dim]


        qk_matmul_node = self.mha2sha_optim._op_factory.get_matmul_op(query_transpose_node, key_reshape_node, propose_sha_name+"_qk_MatMul") # (B, H, W, H*W)
        sha_base_attn_node_list.qk_matmul.append(qk_matmul_node)
        self.mha2sha_optim.model.graph.node.append(qk_matmul_node)

        # Step 7 Create sha pattern
        sha_nodes_list = [qk_matmul_node]
        qkv_matmul = self.mha2sha_optim.create_sha_pattern(info_dict,
                                                           ns,
                                                           head_num,
                                                           value_inp,
                                                           sha_nodes_list,
                                                           sha_base_attn_node_list)

        self.mha2sha_optim.model.graph.node.append(qkv_matmul)

        return qkv_matmul, _key_inp, _value_inp, past_value_inp
