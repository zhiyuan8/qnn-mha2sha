# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from typing import Union, List, Tuple
from dataclasses import dataclass
from onnx import numpy_helper
from onnx.onnx_pb import ModelProto, NodeProto

from mha2sha.utils.onnx import (
    get_initializer_mappings,
    get_node_by_input_name,
    get_node_by_output_name,
    get_node_mappings,
    get_next_node_up_based_on_cond
)

def get_node_type(node, get_initializer_by_name):
    if node.op_type not in ("MatMul", "Gemm"):
        return node.op_type

    if node.op_type == "Gemm" or node.input[1] in get_initializer_by_name.keys():
        return "Linear"
    return "MatMul"

@dataclass
class MhaInfo:
    first_module_name: str
    qkv_proj: List[NodeProto]
    attn_proj: NodeProto
    qk_matmul: NodeProto
    qkv_matmul: NodeProto

class MhaMapper:
    """
    A mapper that collects potential mha patterns. Also provides utils to verify and export found
    mha_potentials to a MhaInfo.
    """
    def __init__(self, model, mha_conv):
        self.is_use = False # is this mapper being used
        self.mha_potential = None # list of tuple(nodule_name, module) for a potential mha
        self.expecting_module_type_idx = 0 # next core op type
        self.linear1_counter = 0 # counter for linear1
        self.linear1_module = [] # 1 or 3 linear(s) module to project hidden_states to q, k, v
        self.linear2_module = [] # linear module to project attention to hidden_state at the end of mha
        self.MatMul_nodes = [] # MatMul
        self.mha_conv = mha_conv

        # model laoder helpers
        self.model = model
        self.get_initializer_by_name = get_initializer_mappings(self.model)
        self.get_node_by_input_name = get_node_by_input_name(self.model)
        self.get_node_by_output_name = get_node_by_output_name(self.model)
        self.get_node_by_node_name = get_node_mappings(self.model)

    def reset(self):
        """ Reset mapper to init status. """
        self.is_use = False
        self.mha_potential = None
        self.expecting_module_type_idx = 0
        self.linear1_counter = 0
        self.linear1_module = []
        self.linear2_module = []
        self.MatMul_nodes = []

    def _search_qkv_linears_upsteam_qk_MatMul_qkv_MatMul(self):
        targe_node_op_type = "Conv" if self.mha_conv else "Linear"

        qk_matmul_node = self.MatMul_nodes[0]
        qkv_matmul_node = self.MatMul_nodes[1]

        q_matmul = get_next_node_up_based_on_cond(
            self.get_node_by_output_name[qk_matmul_node.input[0]],
            self.get_node_by_output_name,
            node_found_cond=lambda n: get_node_type(n, self.get_initializer_by_name) == targe_node_op_type
            )
        k_matmul = get_next_node_up_based_on_cond(
            self.get_node_by_output_name[qk_matmul_node.input[1]],
            self.get_node_by_output_name,
            node_found_cond=lambda n: get_node_type(n, self.get_initializer_by_name) == targe_node_op_type
            )
        v_matmul = get_next_node_up_based_on_cond(
            self.get_node_by_output_name[qkv_matmul_node.input[1]],
            self.get_node_by_output_name,
            node_found_cond=lambda n: get_node_type(n, self.get_initializer_by_name) == targe_node_op_type
        )
        return q_matmul, k_matmul, v_matmul


    def _verify_mha_and_get_qkv_attn(self)-> Union[Tuple[List[NodeProto], List[NodeProto]], bool]:
        """
        Verify a potential mha by abserving qkv and attn's input output features.
        Verification includes:
        (1) mha should have 1 or 3 linear1.
        (2) There are reshape/view after linear1 and before matmul. (SHA will not require reshape)
        (3) Check input and output dim for linear1. If one linear1, then the output must be 3x of embed
            , otherwise it mush use 3 linear1s for q, k, and v.
        :return Tuple(qkv_proj, attn_proj) if verified, else return False
        """
        # Mha should have 1 or 3 linear1.
        if self.linear1_counter not in (1, 3):
            return False

        qkv = self.linear1_module
        attn = self.linear2_module

        if self.mha_conv:
            # conv weight shape: [O, I, kH, kW]
            attn_weight = numpy_helper.to_array(self.get_initializer_by_name[attn[0].input[1]])
            embed_dim = attn_weight.shape[1]

            if len(qkv) == 1:
                qkv_weight = numpy_helper.to_array(self.get_initializer_by_name[qkv[0].input[1]])

                # if one linear, and qkv have 3*embed_dim out_feature.
                if qkv_weight.shape[1] == 3*embed_dim:
                    return (qkv, attn)

                # if one linear, and qkv have embed_dim out_feature.
                # Search for k, v conv, it might be a cross attention.
                # if searched q is the q found in onnx, then update qkv list with found q, k, v,
                # and go to len(qkv) == 3 verification step.
                elif qkv_weight.shape[1] == embed_dim:
                    q_matmul, k_matmul, v_matmul = self._search_qkv_linears_upsteam_qk_MatMul_qkv_MatMul()
                    if q_matmul == qkv[0]:
                        qkv = [q_matmul, k_matmul, v_matmul]

            # if 3 linears, then each qkv will need same out_feature as embed_dim
            if len(qkv) == 3:
                q_weight = numpy_helper.to_array(self.get_initializer_by_name[qkv[0].input[1]])
                k_weight = numpy_helper.to_array(self.get_initializer_by_name[qkv[1].input[1]])
                v_weight = numpy_helper.to_array(self.get_initializer_by_name[qkv[2].input[1]])

                if q_weight.shape[0] == k_weight.shape[0] ==  v_weight.shape[0] == embed_dim:
                    return (qkv, attn)

        else:
            # Linear weight shape: [I, O]
            attn_weight = numpy_helper.to_array(self.get_initializer_by_name[attn[0].input[1]])
            embed_dim = attn_weight.shape[0]

            # if one linear, then qkv need 3*embed_dim out_feature.
            if len(qkv) == 1:
                qkv_weight = numpy_helper.to_array(self.get_initializer_by_name[qkv[0].input[1]])
                if qkv_weight.shape[0] == 3*embed_dim:
                    return (qkv, attn)
                # if one linear, and qkv have embed_dim out_feature.
                # Search for k, v conv, it might be a cross attention.
                # if searched q is the q found in onnx, then update qkv list with found q, k, v,
                # and go to len(qkv) == 3 verification step.
                elif qkv_weight.shape[1] == embed_dim:
                    q_matmul, k_matmul, v_matmul = self._search_qkv_linears_upsteam_qk_MatMul_qkv_MatMul()
                    if q_matmul == qkv[0]:
                        qkv = [q_matmul, k_matmul, v_matmul]

            # if 3 linears, then each qkv will need same out_feature as embed_dim
            if len(qkv) == 3:
                q_weight = numpy_helper.to_array(self.get_initializer_by_name[qkv[0].input[1]])
                k_weight = numpy_helper.to_array(self.get_initializer_by_name[qkv[1].input[1]])
                v_weight = numpy_helper.to_array(self.get_initializer_by_name[qkv[2].input[1]])

                if q_weight.shape[1] == embed_dim and k_weight.shape[1] == embed_dim and v_weight.shape[1] == embed_dim:
                    return (qkv, attn)

        return False

    def verify_and_export_mha_info(self)-> Union[MhaInfo, bool]:
        """
        Verify mha_potential and return a MhaInfo if sucessful varified, else return False.
        :return mha_info or False if not match a mha.
        """
        result = self._verify_mha_and_get_qkv_attn()

        if result:
            mha_info = MhaInfo(first_module_name = self.mha_potential[0].name,
                               qkv_proj = result[0],
                               attn_proj = result[1],
                               qk_matmul = self.MatMul_nodes[0],
                               qkv_matmul = self.MatMul_nodes[1])
            return mha_info

        return result

    def export_mha_info(self)-> Union[MhaInfo, bool]:
        """
        Return mha_info without verifying
        :return mha_info.
        """

        mha_info = MhaInfo(first_module_name = self.mha_potential[0].name,
                            qkv_proj = None,
                            attn_proj = None,
                            qk_matmul = self.MatMul_nodes[0],
                            qkv_matmul = self.MatMul_nodes[1])
        return mha_info


class AttentionFinder:
    def __init__(self, model: ModelProto, mha_conv: bool):
        self.mha_conv = mha_conv
        self.model = model
        self.get_initializer_by_name = get_initializer_mappings(self.model)
        self.get_node_by_input_name = get_node_by_input_name(self.model)
        self.get_node_by_output_name = get_node_by_output_name(self.model)
        self.get_node_by_node_name = get_node_mappings(self.model)

    def get_attn_info(self):
        """
        Locate potential mha pattern in topologically sorted onnx graph.
        Search for:
                                  [Linear ... Matmal ... Softmax ... Matmul ... Linear]
        or  [Linear ... Linear ... Linear ... Matmal ... Softmax ... Matmul ... Linear]
        Verify the pattern captured is a mha and export to MhaInfo.
        :param mha_model: prepared mha model
        :return: list of MhaInfo for found and verified mha(s).
        """
        get_initializer_by_name = get_initializer_mappings(self.model)

        mha_info_list = []

        # Prepare 3 MhaMapper to detect 1~3 linear1 cases.
        mha_pattern_maps = [MhaMapper(self.model, self.mha_conv),
                            MhaMapper(self.model, self.mha_conv),
                            MhaMapper(self.model, self.mha_conv)]

        qkv_matmul_type = "Linear" if not self.mha_conv else "Conv"
        if self.mha_conv:
            module_type_seq_list = [qkv_matmul_type, "MatMul", "Softmax", "MatMul", qkv_matmul_type]
            module_type_set = (qkv_matmul_type, "MatMul", "Softmax") # Set for module_type_seq_list
            hard_stop_module_type_set = ("LayerNorm", "InstanceNormalization", "ReduceMean") # Hard stop for collecting potential mha pattern

        else:
            module_type_seq_list = [qkv_matmul_type, "MatMul", "Softmax", "MatMul", qkv_matmul_type]
            module_type_set = (qkv_matmul_type, "MatMul", "Softmax") # Set for module_type_seq_list
            hard_stop_module_type_set = ("Conv", "LayerNorm", "InstanceNormalization", "ReduceMean") # Hard stop for collecting potential mha pattern


        # pylint: disable=too-many-nested-blocks
        for node in self.model.graph.node:
            module_type = get_node_type(node, get_initializer_by_name)

            for i, _mapper in enumerate(mha_pattern_maps):
                # Ignore idle mapper but use the first mapper no matter what.
                if _mapper.is_use or i == 0:
                    if module_type == module_type_seq_list[_mapper.expecting_module_type_idx]:
                        # Init mapper if this is the first
                        if _mapper.mha_potential is None:
                            _mapper.is_use = True
                            _mapper.mha_potential = []
                            _mapper.linear1_counter += 1
                            _mapper.linear1_module.append(node)
                        _mapper.expecting_module_type_idx += 1

                        if module_type == "MatMul":
                            _mapper.MatMul_nodes.append(node)

                        # All 5 main modules are found.
                        if _mapper.expecting_module_type_idx == 5:
                            # Add the last module (linear2) to list, verify, and reset.
                            _mapper.mha_potential.append(node)
                            _mapper.linear2_module.append(node)
                            result = _mapper.verify_and_export_mha_info()
                            if result:
                                mha_info_list.append(result)

                            # reset mapper
                            _mapper.reset()

                    # expecting not linear but got linear.
                    elif module_type == qkv_matmul_type:
                        if _mapper.linear1_counter == 3 or _mapper.expecting_module_type_idx != 1:
                            # Got 4th consective linear1 or got linear when not expecting matmul1
                            # Reset mapper and init new one with this linear (start over from this linear)
                            _mapper.reset()
                            _mapper.is_use = True
                            _mapper.mha_potential = [] # modules will be add in the end of the loop
                            _mapper.expecting_module_type_idx = 1
                            _mapper.linear1_counter = 1
                            _mapper.linear1_module = [node]
                        else:
                            # Init next mha_mapper, if all mapper are in use, do nothing.
                            # use %3 to avoid index overflow.
                            mha_pattern_maps[(i+1)%3].is_use = True

                            # Increase counter and add (name, module) to linear1 list.
                            _mapper.linear1_counter += 1
                            _mapper.linear1_module.append(node)

                    # if module not coming in module_type_list sequence or meet a hard stop module, reset the mapper.
                    elif module_type in module_type_set or module_type in hard_stop_module_type_set:
                        # reset mapper
                        _mapper.reset()

                    if _mapper.mha_potential is not None:
                        _mapper.mha_potential.append(node)
        return mha_info_list

class QuickAttentionFinder:
    """
    Works same as Attention Finder but search only the qk_MatMul ... Softmax ... qkv_MatMul pattern.
    Also return mha without sanity check. (check all linear layer I/O dim).
    Dedigned for LORA models that have more than 3 Linears in attention.
    """
    def __init__(self, model: ModelProto, mha_conv: bool):
        self.mha_conv = mha_conv
        self.model = model
        self.get_initializer_by_name = get_initializer_mappings(self.model)
        self.get_node_by_input_name = get_node_by_input_name(self.model)
        self.get_node_by_output_name = get_node_by_output_name(self.model)
        self.get_node_by_node_name = get_node_mappings(self.model)

    def get_attn_info(self):
        """
        Locate potential mha pattern in topologically sorted onnx graph.
        Search for:
                       [Matmal ... Softmax ... Matmul]
        or  [Linear ... Matmal ... Softmax ... Matmul]
        Verify the pattern captured is a mha and export to MhaInfo.
        :param mha_model: prepared mha model
        :return: list of MhaInfo for found and verified mha(s).
        """
        get_initializer_by_name = get_initializer_mappings(self.model)
        def get_node_type(node):
            if node.op_type not in ("MatMul", "Gemm"):
                return node.op_type

            if node.op_type == "Gemm" or node.input[1] in get_initializer_by_name.keys():
                return "Linear"
            return "MatMul"

        mha_info_list = []

        # Prepare 3 MhaMapper to detect 1~3 linear1 cases.
        mha_mapper = MhaMapper(self.model, self.mha_conv)

        if self.mha_conv:
            module_type_seq_list = ["MatMul", "Softmax", "MatMul"]
            module_type_set = ("MatMul", "Softmax") # Set for module_type_seq_list
            hard_stop_module_type_set = ("LayerNorm", "InstanceNormalization", "ReduceMean") # Hard stop for collecting potential mha pattern

        else:
            module_type_seq_list = ["MatMul", "Softmax", "MatMul"]
            module_type_set = ("MatMul", "Softmax") # Set for module_type_seq_list
            hard_stop_module_type_set = ("Conv", "LayerNorm", "InstanceNormalization", "ReduceMean") # Hard stop for collecting potential mha pattern


        # pylint: disable=too-many-nested-blocks
        for node in self.model.graph.node:
            module_type = get_node_type(node)

            if module_type == module_type_seq_list[mha_mapper.expecting_module_type_idx]:
                # Init mapper if this is the first
                if mha_mapper.mha_potential is None:
                    mha_mapper.is_use = True
                    mha_mapper.mha_potential = []
                mha_mapper.expecting_module_type_idx += 1

                if module_type == "MatMul":
                    mha_mapper.MatMul_nodes.append(node)

                # MatMul ... Softmax ... MatMul are found.
                if mha_mapper.expecting_module_type_idx == 3:
                    # Add the last module to list, verify, and reset.
                    mha_mapper.mha_potential.append(node)
                    result = mha_mapper.export_mha_info()
                    if result:
                        mha_info_list.append(result)

                    # reset mapper
                    mha_mapper.reset()

            # if module not coming in module_type_list sequence or meet a hard stop module, reset the mapper.
            elif module_type in module_type_set or module_type in hard_stop_module_type_set:
                # reset mapper
                mha_mapper.reset()

            if mha_mapper.mha_potential is not None:
                mha_mapper.mha_potential.append(node)

        return mha_info_list

def auto_attention_finder(model: ModelProto, mha_conv: bool = False, use_quick_auto_finder: bool = False):
    """
    Get all the mha in a topologically sorted model.
    :return pattern: List of Pattern for mha2sha optimzer.
    :return pattern_start_node_names: List of qk_matmuls.
    :return pattern_end_node_names: List of qkv_matmuls.
    """
    pattern, pattern_start_node_names, pattern_end_node_names = None, None, None
    if not use_quick_auto_finder:
        finder = AttentionFinder(model, mha_conv)
    else:
        finder = QuickAttentionFinder(model, mha_conv)
    mha_info_list = finder.get_attn_info()

    if len(mha_info_list) > 0:
        pattern = []
        start_node = mha_info_list[0].qk_matmul
        end_node = mha_info_list[0].qkv_matmul
        while start_node != end_node:
            pattern.append(start_node.op_type)
            start_node = finder.get_node_by_input_name[start_node.output[0]][0]
        pattern.append(end_node.op_type)

        pattern_start_node_names = [ _info.qk_matmul.name for _info in mha_info_list]
        pattern_end_node_names = [ _info.qkv_matmul.name for _info in mha_info_list]

        assert pattern_start_node_names, "No starting nodes found. Cannot apply MHA to SHA optimization."
        assert pattern_end_node_names, "No ending nodes found. Cannot apply MHA to SHA optimization."
        assert len(pattern_start_node_names) == len(pattern_end_node_names), \
            (f"Amount of start and end nodes should be equal.\n"
             "Got {len(pattern_start_node_names)=} and {len(pattern_end_node_names)=}")
    return pattern, pattern_start_node_names, pattern_end_node_names
