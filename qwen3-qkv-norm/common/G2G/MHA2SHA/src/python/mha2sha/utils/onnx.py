# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from dataclasses import dataclass
import inspect
import os
from pathlib import Path
import sys
import tempfile
from typing import IO, Dict, Iterable, List, Set, Text, Tuple, Union, Optional, Callable

import numpy as np
import onnx
from onnx import helper
from onnx.onnx_pb import (
    AttributeProto,
    GraphProto,
    NodeProto,
    ModelProto,
    TensorProto,
    ValueInfoProto,
)
from onnx.external_data_helper import (
    load_external_data_for_model,
    set_external_data,
)

import onnxruntime

from mha2sha.utils.logger import log_debug, log_info, log_warning
from mha2sha.utils.op_factory import create_node_name

ONNX_EXTERNAL_DATA_THRESHOLD = 1024


def run_model_on_ort(
    onnx_path: str,
    inputs: Dict[str, np.ndarray],
    output_names: List[str],
) -> Tuple[List[str], List[np.ndarray]]:
    """
    Function responsible to run the model on ONNXRT.

    :param onnx_path: str
    :param inputs: Dict[str, np.ndarray]
    :param output_names: List[str]

    :return input_names: list - list of input names.
    :return ort_outputs: list - list of onnx outputs.
    """
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_names = [x.name for x in ort_session.get_inputs()]
    ort_outputs = ort_session.run(output_names, inputs)
    return input_names, ort_outputs


def get_output_node(
    node_name: str,
    graph_output_node_names: List[str],
    nodeNameToNodeDict: dict,
    inputNameToNodeDict: dict,
) -> List:
    """
    Function to get the output node names of the specified node name.

    :param node_name (str) : name of the node.
    :param nodeNameToNodeDict (dict) : input_node to node mapping.
    :param inputNameToNodeDict (dict) : input_name to node mapping.
    :param graph_output_node_names (dict) : list of output nodes in the graph.

    :return List : output names of the nodes specified.
    """
    output_tensor_names = nodeNameToNodeDict[node_name].output
    output_nodes = []
    for op_tensor_name in output_tensor_names:
        if op_tensor_name in graph_output_node_names:
            continue
        op_nodes = inputNameToNodeDict[op_tensor_name]
        output_nodes.extend(op_nodes)
    return output_nodes


# ONNX utils ported over from QAIRT
def assign_names_to_empty_nodes(model: ModelProto) -> ModelProto:
    """
    Function to add new node names for nodes whose name property is "".

    :param ModelProto model: Onnx model reference.
    """
    _node_name_suffix_mapping = {}
    for node in get_nodes(model):
        if node.name == "":
            new_node_name, _node_name_suffix_mapping = create_node_name(
                model.graph, node.op_type, _node_name_suffix_mapping
            )
            node.name = new_node_name
    return model

def check_duplicates_in_attribute(graph: GraphProto, attribute_name: str) -> bool:
    """
    Function to check for duplicates in the given attribute_name.

    :param GraphProto model: Graph reference from model.
    :param str attribute_name: Attribute name of the model. e.g. node, input or output.
    :return bool: boolean status indicating graph is valid or not.
    """
    node_check_status = True
    property_name_set = set()
    output_tensor_name_set = set()

    if not hasattr(graph, attribute_name):
        return False

    list_of_attributes = getattr(graph, attribute_name)

    if not isinstance(list_of_attributes, Iterable):
        return False

    for property in list_of_attributes:
        if property.name != "" and property.name not in property_name_set:
            property_name_set.add(property.name)
        else:
            node_check_status = node_check_status and False
            if property.name == "":
                log_debug(
                    "Graph checker: No {} name found for.".format(attribute_name)
                )
            else:
                log_debug(
                    f"Graph checker: {attribute_name} '{property.name}' is duplicate."
                )

        # Check for intermediate tensors. Each node's output tensor shall have
        # unique name. Two different nodes shall not have same output tensor name.
        if hasattr(property, "output"):
            for node_output in property.output:
                if node_output not in output_tensor_name_set:
                    output_tensor_name_set.add(node_output)
                else:
                    node_check_status = False
                    log_debug(
                        f"Graph checker: Tensor '{node_output}' is the output of two different nodes."
                    )
                    return node_check_status

    return node_check_status


def get_graphs(model: ModelProto) -> List[GraphProto]:
    """
    Function to return all the graph present in model.

    :param ModelProto model: Onnx Model graph proto.
    :return List[GraphProto]: Onnx graphs
    """
    all_graphs = []
    graph_queue = [model.graph]
    while graph_queue:
        graph = graph_queue.pop(0)
        all_graphs.append(graph)
        for node in graph.node:
            for attr in node.attribute:
                if attr.type == AttributeProto.AttributeType.GRAPH:
                    assert isinstance(attr.g, GraphProto)
                    graph_queue.append(attr.g)
                if attr.type == AttributeProto.AttributeType.GRAPHS:
                    for g in attr.graphs:
                        assert isinstance(g, GraphProto)
                        graph_queue.append(g)
    return all_graphs


def get_initializer_mappings(model: ModelProto) -> Dict:
    """
    Initializer name to initializer mapping

    :param onnx.ModelProto model: model (onnx.ModelProto): onnx model.
    :return Dict: initializer name to initializer mapping
    """
    return {n.name: n for n in model.graph.initializer}


def get_inputs(model) -> List[ValueInfoProto]:
    """
    Get the graph inputs tensors except initializers.

    :param model: path to the loaded onnx model proto
    :return List: List of graph inputs
    """
    initializer_names = [x.name for x in model.graph.initializer]
    return [ipt for ipt in model.graph.input if ipt.name not in initializer_names]


def get_model_size(model: ModelProto) -> float:
    """
    Provides the size of the model in GB.

    :param model (ModelProto): Onnx model proto instance.
    :return size_gb (float): size of the model in GB
    """
    NUM_GB_BYTES = 1024**3
    size_bytes = model.ByteSize()
    size_gb = size_bytes / NUM_GB_BYTES
    return size_gb


def get_nodes(model: ModelProto) -> List[NodeProto]:
    """
    Function to return the nodes information.

    :param ModelProto model: Onnx Model graph proto.
    :return List[NodeProto]: Underlying onnx nodes.
    """
    nodes_all = []
    for graph in get_graphs(model):
        for node in graph.node:
            nodes_all.append(node)
    return nodes_all


def get_node_by_input_name(model: onnx.ModelProto) -> Dict:
    """
    Input name to node mappings.
    :param  loader (onnx.ModelProto): onnx.ModelProto model instance.
    :returns: Dict: Input name to node mappings.
    """
    get_node_by_input = {}
    for n in model.graph.node:
        for n_ip in n.input:
            if n_ip not in get_node_by_input:
                get_node_by_input[n_ip] = [n]
            else:
                get_node_by_input[n_ip].append(n)
    return get_node_by_input


def get_node_by_output_name(model: onnx.ModelProto) -> Dict:
    """
    Output name to node mappings.
    :param model (onnx.ModelProto): OnnxModel model instance.
    :return : Dict: Output name to node mappings.
    """
    get_node_by_output = {}
    for n in model.graph.node:
        for n_ip in n.output:
            if n_ip not in get_node_by_output:
                get_node_by_output[n_ip] = n
    return get_node_by_output


def get_node_mappings(model: onnx.ModelProto) -> Dict:
    """
    Node name to nodes mapping.

    :param onnx.ModelProto model:  onnx model.
    :return Dict: Node name to nodes mapping
    """

    return {n.name: n for n in model.graph.node}


def get_outputs(model) -> List[onnx.ValueInfoProto]:
    """
    Get the graph outputs tensors.

    :param model: path to the loaded onnx model proto
    :return List: List of graph iutputs
    """
    initializer_names = [x.name for x in model.graph.initializer]
    return [ipt for ipt in model.graph.output if ipt.name not in initializer_names]


def get_parent_at_any_level(
    output_name: str, get_node_by_output_name: Dict[str, NodeProto], level: int = 1,
) -> List[NodeProto]:
    """
    Function to get the parent of the specified node at given level.
    level 1 - immediate parent
    level 2 - parent of parent and so on.

    :param str output_name: Output name of the node whose parent needs to be
        identified.
    :param Dict[str, NodeProto] get_node_by_output_name: Dict to get the
        node by its output name.
    :param int level: Parent level to be identified.
    :raises Exception: _description_
    :return List[NodeProto]: List of parent nodes.
    """
    output_node = get_node_by_output_name[output_name]

    # 'get_parent' is defined at the end of this file, it was already ported over.
    parent_nodes = get_parent(output_node, get_node_by_output_name)

    if level == 1:
        return parent_nodes
    elif level > 1:
        for i in range(level - 1):
            iterating_nodes = parent_nodes
            final_nodes = []
            for nodes in iterating_nodes:
                candidate_nodes = get_parent(nodes, get_node_by_output_name)
                for node in candidate_nodes:
                    final_nodes.append(node)
            parent_nodes = final_nodes
        return parent_nodes
    else:
        log_error(f"Can't find the parent at given level: {level}")
        return None


def get_shape_from_value_info_proto(
    val_info: ValueInfoProto,
    use_onnx_def_symbols: bool = False,
) -> List[Union[str, int]]:
    """
    Function to get the shape from value info proto.

    :param val_info: onnx.ValueInfoProto
    :param use_onnx_def_symbols: If the shapes need to contain onnx defined
        symbol then pass True. For False onnx defined symbols will be replaced
        by -1. Defaults to False.
    :return List[Union[str, int]]: Tensor shape of given value info proto.
    """
    tensor_shape = []
    tensor_type = val_info.type.tensor_type

    if not tensor_type.HasField("shape"):
        log_warning(f"Shape not found for tensor: {val_info.name}")
        return tensor_shape

    # iterate through dimensions of the shape:
    for d in tensor_type.shape.dim:
        # the dimension may have a definite (integer) value or a symbolic identifier or neither:
        if d.HasField("dim_value"):
            tensor_shape.append(d.dim_value)
        elif d.HasField("dim_param"):
            # unknown dimension with symbolic name
            if use_onnx_def_symbols:
                tensor_shape.append(d.dim_param)
            else:
                tensor_shape.append(-1)
        else:
            tensor_shape.append("?")
    return tensor_shape


def get_value_info_proto_mapping(model: onnx.ModelProto) -> Dict:
    """
    Value name to value mapping.
    :param model (onnx.ModelProto): onnx model.
    :returns:Dict: value name to value mapping
    """
    return {v.name: v for v in model.graph.value_info}


def graph_checker(model: ModelProto) -> bool:
    """
    Function to check the validity of graph. e.g Node names are present or not,
    Is there any duplicate node names, node's input tensor names or node's
    output tensor names etc. present or not.
    :param ModelProto model: Onnx model
    :return bool: boolean status indicating graph is valid or not.
    """
    node_check_status = True
    for graph in get_graphs(model):
        node_check_status = node_check_status and check_duplicates_in_attribute(
            graph, "node"
        )
        node_check_status = node_check_status and check_duplicates_in_attribute(
            graph, "input"
        )
        node_check_status = node_check_status and check_duplicates_in_attribute(
            graph, "output"
        )
    return node_check_status


class ONNXPatternMatcher:
    """
    ONNXPatternOptimizer class contains generic logic for identifying and removing a pattern in onnx graph.
    It has following:

    :param model = Instance of FrameworkModelLoader from the previous stages.
    :return FrameworkModelLoader instance with updated/cleaned model.
    """

    def __init__(self, model: ModelProto):
        """
        Initialization.
        """
        self.model = model
        self.__parse_model()  # Initialized mapping dicts for easy graph traversal.

    def __parse_model(self):
        """
        Store the graph information:
        1. Input nodes
        2. Output nodes
        3. Map of name to node
        4. Input: GraphProto
        """
        self.graph_output_node_names = [
            n.name for n in self.model.graph.output
        ]
        cnt_n = [n for n in get_nodes(self.model) if n.name == ""]
        if len(cnt_n) > 1:
            log_warning(
                "More than one nodes with same name : '' found. Pattern Matcher requires unique node names for all node."
            )
            self.model = assign_names_to_empty_nodes(self.model)
            for n in cnt_n:
                node_name = n.name
                node_input = [n_ip for n_ip in n.input]
                node_output = [n_op for n_op in n.output]
                node_type = n.op_type
                log_debug(
                    f"Node Name: {node_name}, Node Type: {node_type}, Node input: {node_input}, node_output: {node_output}"
                )
        self.get_node_by_node_name = get_node_mappings(self.model)
        self.get_node_by_input_name = get_node_by_input_name(self.model)

    def postprocess_pattern_matcher(
        self,
        pattern: List,
    ) -> Tuple[List, list]:
        """
        Finds a pattern in the graph.

        :param Specific Pattern(list of nodes representing a pattern)

        :return Start nodes of the mentioned pattern if matched.
        :return End nodes of the mentioned pattern if matched.
        """
        pattern_start_nodes_list = []
        pattern_end_nodes_list = []
        matched_pattern = []
        for n in self.model.graph.node:
            if n.op_type == pattern[0]:
                children_depth = 1
                all_op_nodes = get_output_node(
                    n.name,
                    self.graph_output_node_names,
                    self.get_node_by_node_name,
                    self.get_node_by_input_name,
                )
                # If the node "n" is last node with no outputs in graph outputs,
                # all_op_nodes would be empty list. This will be fine as that will
                # create stack an empty list.
                stack = [[op, children_depth] for op in all_op_nodes]
                while len(stack) != 0:
                    _node, _node_level = stack.pop()
                    if len(pattern) <= _node_level:
                        # This means we have checked all the elements in the
                        # pattern. Now we should not traverse further into
                        # children nodes.
                        continue
                    if _node.op_type == pattern[_node_level]:
                        if _node_level + 1 == len(pattern):
                            # This means we reached at the end of the pattern
                            # Pattern fully matched.
                            matched_pattern.append([n, _node])
                        all_node_op_nodes = get_output_node(
                            _node.name,
                            self.graph_output_node_names,
                            self.get_node_by_node_name,
                            self.get_node_by_input_name,
                        )
                        stack.extend(
                            [[op, _node_level + 1] for op in all_node_op_nodes]
                        )
        for start_node, end_node in matched_pattern:
            # Only add if it is not added. So that we can have single instances
            # of same node if found multiple times.
            if start_node not in pattern_start_nodes_list:
                pattern_start_nodes_list.append(start_node)
            if end_node not in pattern_end_nodes_list:
                pattern_end_nodes_list.append(end_node)
        return pattern_start_nodes_list, pattern_end_nodes_list


def get_pattern_start_end_nodes(
    model: ModelProto, attention_patterns: Dict
) -> Tuple[List, List, List]:
    """
    Function responsible for identifying the attention pattern
    and return the "identified pattern", "start nodes of the
    pattern", and the split level where MHA split should begin.

    :param model: ModelProto
    :param attention_patterns: Dict

    :return pattern - list of nodes which matched
    :return pattern_start_node_names - starting nodes of the pattern
    :return pattern_end_node_names - ending nodes of the pattern
    """
    opo = ONNXPatternMatcher(model)
    pattern_index = 0
    sn, en = None, None
    for pt in range(len(attention_patterns)):
        pattern = attention_patterns[pt]["pattern"]
        sn, en = opo.postprocess_pattern_matcher(pattern)
        if (len(sn) or len(en)) > 0:
            pattern_index = pt
            break

    assert sn, "No starting nodes found. Cannot apply MHA to SHA optimization."
    assert en, "No ending nodes found. Cannot apply MHA to SHA optimization."
    assert len(sn) == len(en), \
        f"Amount of start and end nodes should be equal.\nGot sn={len(sn)} and en={len(en)}"

    pattern_start_node_names = [node.name for node in sn]
    pattern_end_node_names = [node.name for node in en]

    pattern = attention_patterns[pattern_index]["pattern"]
    return pattern, pattern_start_node_names, pattern_end_node_names


# Function is inside of onnx.utils in the Unified SDK in get_parent_at_any_level.
def get_parent(node, get_node_by_output_name):
    im_parent_nodes = []
    for inp in node.input:
        if inp in get_node_by_output_name:
            it_node = get_node_by_output_name[inp]
            im_parent_nodes.append(it_node)
    return im_parent_nodes

def get_children(node, get_node_by_input_name):
    im_children_nodes = []
    for out in node.output:
        if out in get_node_by_input_name:
            it_node = get_node_by_input_name[out]
            im_children_nodes.extend(it_node)
    return im_children_nodes # Return only the first?


def get_least_commom_ancestor_with_verified_pathway(node_1, node_2, optimizer,
                                                    pathway_nodes_verifier=None,
                                                    search_input_0_path_only=False):
    """
    return LCA (least common ancestor) of node_1 and node_2.
    LCA can be node or model input
    if those conditions are verified:
        - every nodes between LCA and node_1 should pass the user given
          pathway_nodes_verifier (LCA and node_1 are not included)
        - every nodes between LCA and node_2 should pass the user given
          pathway_nodes_verifier (LCA and node_1 are not included)

    if pathway_nodes_verifier is not given, then LCA will be simply returned.

    if search_input_0_path_only is True, LCA will not search path from input 1.
    Originally designed for LoRA v3 to ignore branches from other adaptors.

    if LCA dosen't exist or those conditions are not satisfied, None will be
    returned

    """
    node_1_ancestors_name = set()
    node_2_ancestors_name = set()
    if node_1.name == node_2.name:
        return node_1

    node_1_input = [node_1.input[0]] if search_input_0_path_only else node_1.input
    node_2_input = [node_2.input[0]] if search_input_0_path_only else node_2.input

    lca = {}
    if pathway_nodes_verifier is None:
        pathway_nodes_verifier = lambda x: True
    while not lca:
        if len(node_1_input) == 0 and len(node_2_input) == 0:
            return None
        node_1_parents_input = []
        node_1_parents_name = set()
        for _input in node_1_input:
            if _input in optimizer.get_node_by_output_name.keys():
                node_1_input_node = optimizer.get_node_by_output_name[_input]
                if pathway_nodes_verifier(node_1_input_node):
                    if search_input_0_path_only:
                        node_1_parents_input.append(node_1_input_node.input[0])
                    else:
                        node_1_parents_input.extend(node_1_input_node.input)

                node_1_parents_name.update({node_1_input_node.name})

            elif _input in optimizer.mha_model_input_names:
                node_1_parents_name.update({_input})

        node_1_ancestors_name.update(node_1_parents_name, {node_1.name})

        node_2_parents_input = []
        node_2_parents_name = set()
        for _input in node_2_input:
            if _input in optimizer.get_node_by_output_name.keys():
                node_2_input_nodes = optimizer.get_node_by_output_name[_input]
                if pathway_nodes_verifier(node_2_input_nodes):
                    if search_input_0_path_only:
                        node_2_parents_input.append(node_2_input_nodes.input[0])
                    else:
                        node_2_parents_input.extend(node_2_input_nodes.input)
                node_2_parents_name.update({node_2_input_nodes.name})

            elif _input in optimizer.mha_model_input_names:
                node_2_parents_name.update({_input})

        node_2_ancestors_name.update(node_2_parents_name, {node_2.name})

        lca = node_1_ancestors_name.intersection(node_2_ancestors_name)

        node_1_input = node_1_parents_input
        node_2_input = node_2_parents_input

    if (lca_name := list(lca)[0]) in optimizer.get_node_by_node_name:
        return optimizer.get_node_by_node_name[list(lca)[0]]
    return lca_name


# Custom Error for easier try expect flow
class NodeNotFoundError(Exception): ...


def _get_next_node_base_on_cond_helper(
    start_node: NodeProto,
    get_node_by_input_or_output_name: dict,
    node_found_cond: Callable[[NodeProto], bool],
    node_end_search_cond: Optional[Callable[[NodeProto], bool]] = None,
    search_down: bool = False,
    search_input_0_path_only: bool = False
) -> Optional[NodeProto]:
    """
    Helper function for searching to find a node. Supports search up or down.

    Args:
        start_node:
            Starting node for searching.
        get_node_by_input_or_output_name:
            Dictionary for quickly finding nodes based on input or output names. Depends on search up / down.
        node_found_cond:
            Condition function for finding the node.
        node_end_search_cond:
            Condition function for finishing the search. Think of this as when to turn around in DFS.
        search_down:
            Bool for whether to search down or up in the graph for a node.
        search_input_0_path_only:
            Bool to allow searching up to search only on input_0 path.

    Raises:
        NodeNotFoundError:
            When none of the nodes in the search space satisfy the node_found_cond.

    Returns:
        The node satisfying the node_found_cond.
    """

    stack = [start_node]
    visited: Set[str] = set()

    if search_input_0_path_only and search_down:
        raise ValueError(f"search_input_0_path_only = {search_input_0_path_only} only support when searching up.")

    get_next_nodes = get_children if search_down else get_parent
    if search_input_0_path_only:
        get_next_nodes = lambda node, get_node_by_output_name: [get_node_by_output_name[node.input[0]]] if node.input[0] in get_node_by_output_name else []

    while stack:
        node = stack.pop()
        visited.add(node.name)

        if node_found_cond(node):
            return node

        if node_end_search_cond and node_end_search_cond(node):
            log_debug(f"Node: '{node.name}' met end search criteria.")
            continue

        # Stacked in  input[1],  input[0] sequence, so that pop in  input[0]  input[1] sequence when searching upstream.
        # Stacked in output[1], output[0] sequence, so that pop in output[0] output[1] sequence when searching downstream.
        next_nodes = reversed(list(
            filter(None, get_next_nodes(node, get_node_by_input_or_output_name))
        ))

        if not next_nodes:
            log_debug(
                f"No {'children' if search_down else 'parent'} nodes found for node: '{node.name}'"
            )
            continue

        for next_node in next_nodes:
            if next_node.name not in visited:
                stack.append(next_node)


    fail_msg = (
        f"Unable to find node {'down' if search_down else 'up'}stream.\n"
        f"{start_node.name=}\n"
        f"{inspect.getsource(node_found_cond)}\n"
    )
    if node_end_search_cond:
        fail_msg += f"{inspect.getsource(node_end_search_cond)}"

    raise NodeNotFoundError(fail_msg)


def get_next_node_down_based_on_cond(start_node, get_node_by_input_name, *, node_found_cond, node_end_search_cond=None):
    """
    Wrapper around _get_next_node_base_on_cond_helper for searching down, see _get_next_node_base_on_cond_helper for more
    info.
    """

    return _get_next_node_base_on_cond_helper(
        start_node,
        get_node_by_input_name,
        node_found_cond,
        node_end_search_cond,
        search_down=True,
    )


def get_next_node_up_based_on_cond(start_node, get_node_by_output_name, *, node_found_cond, node_end_search_cond=None, search_input_0_path_only=False):
    """
    Wrapper around _get_next_node_base_on_cond_helper for searching up, see _get_next_node_base_on_cond_helper for more
    info.
    """

    return _get_next_node_base_on_cond_helper(
        start_node,
        get_node_by_output_name,
        node_found_cond,
        node_end_search_cond,
        search_down=False,
        search_input_0_path_only=search_input_0_path_only
    )


def get_constant_node_value(const_node):
    tensor = const_node.attribute[0].t
    np_dtype = helper.tensor_dtype_to_np_dtype(tensor.data_type)
    const_tensor_value = np.frombuffer(tensor.raw_data, dtype=np_dtype).reshape(tensor.dims)
    return const_tensor_value


def get_initializer_value(init_node):
    np_dtype = helper.tensor_dtype_to_np_dtype(init_node.data_type)
    const_tensor_value = np.frombuffer(init_node.raw_data, dtype=np_dtype).reshape(init_node.dims) # NDArray[Any]
    return const_tensor_value

def get_initializer_value_from_raw_data(value, dtype, dims):
    """
    Create a numpy array with specified value, dtype, and dimensions.
    This mimics what get_initializer_value would return.
    
    :param value: The scalar value to fill the array with
    :param dtype: numpy dtype (e.g. np.int32, np.float32)
    :param dims: List of dimensions for the array shape
    :return: numpy array with the specified value, dtype, and shape
    """
    # Create numpy array directly with the value
    const_tensor_value = np.full(dims, value, dtype=dtype)
    return const_tensor_value


def get_mul_value(mul_node, get_initializer_by_name, get_node_by_output_name):
    """ Return mul value as numpy tensor. """
    if mul_node.input[1] in get_initializer_by_name:
        lora_alpha_init = get_initializer_by_name[mul_node.input[1]]
        lora_alpha_value = get_initializer_value(lora_alpha_init)

    elif mul_node.input[1] in get_node_by_output_name:
        lora_alpha_node = get_node_by_output_name[mul_node.input[1]]
        if lora_alpha_node.op_type == "Identity":
            lora_alpha_init = get_initializer_by_name[lora_alpha_node.input[0]]
            lora_alpha_value = get_initializer_value(lora_alpha_init)[0]

        elif lora_alpha_node.op_type == "Constant":
            lora_alpha_value = get_constant_node_value(lora_alpha_node)

        else:
            raise ValueError(f"expect mul_node: {mul_node} input[0] is Identity or Constant")
    else:
        raise ValueError(f"expect mul_node: {mul_node} [0] is Initializer or Node")

    return lora_alpha_value


@dataclass
class SliceInfo:
    starts: Optional[int]
    ends: Optional[int]
    axes: Optional[int]
    steps: Optional[int]


def get_slice_info(slice_node, get_node_by_output_name, get_initializer_by_name) -> SliceInfo:
    """ Get start, end, axis, step from slice node. """
    slice_val = []

    for idx in range(1, 5):
        if idx >= len(slice_node.input):
            continue
        if slice_node.input[idx] in get_initializer_by_name:
            init = get_initializer_by_name[slice_node.input[idx]]
            init_value = get_initializer_value(init)[0]

        elif slice_node.input[idx] in get_node_by_output_name:
            node = get_node_by_output_name[slice_node.input[idx]]
            if node.op_type == "Identity":
                init = get_initializer_by_name[node.input[0]]
                init_value = get_initializer_value(init)[0]

            elif node.op_type == "Constant":
                init_value = get_constant_node_value(node)

            else:
                raise ValueError(f"expect slice_node: {slice_node} input[{idx}] is Identity or Constant")
        else:
            raise ValueError(f"expect slice_node: {slice_node} [0] is Initializer or Node")
        slice_val.append(init_value)

    slice_info = SliceInfo(*slice_val)
    return slice_info


def get_node_input_constant_op_value(node, get_node_by_output_name, get_initializer_by_name):
    """
    Get a node's Constant op_type input's value. E.g. Get shape from Reshape node, get Add value
    from Add node.
    """
    if not isinstance(node, NodeProto):
        node = get_node_by_output_name[node]

    # node.input[0] in get_node_by_output_name.keys() prevents node input is a model input
    if (
            node.input[0] in get_node_by_output_name.keys()
            and get_node_by_output_name[node.input[0]].op_type == "Constant"
    ):
        const_tensor_value = get_constant_node_value(
            get_node_by_output_name[node.input[0]]
        )

    elif node.input[0] in get_initializer_by_name:
        const_tensor_value = get_initializer_value(
            get_initializer_by_name[node.input[0]]
        )
    elif (
            node.input[1] in get_node_by_output_name.keys()
            and get_node_by_output_name[node.input[1]].op_type == "Constant"
    ):
        const_tensor_value = get_constant_node_value(
            get_node_by_output_name[node.input[1]]
        )
    elif node.input[1] in get_initializer_by_name:
        const_tensor_value = get_initializer_value(
            get_initializer_by_name[node.input[1]]
        )
    else:
        raise ValueError(f"Expecting one of the node's input type to be Constant or an Intializer, \
                         but got {get_node_by_output_name[node.input[0]].op_type} and \
                            {get_node_by_output_name[node.input[1]].op_type}")

    return const_tensor_value


def load_model(model_or_path: str) -> Tuple[ModelProto, str]:
    log_info(f"Loading the onnx model from: {model_or_path}")

    model_or_path = "".join(model_or_path.split())
    model = onnx.load(model_or_path)
    model = assign_names_to_empty_nodes(model)
    return model, model_or_path


# idea from onnx internals with some massaging for Path obj.
def _get_file_path(f: IO[bytes] | str | os.PathLike) -> Path:
    if isinstance(f, Path):
        return f
    if isinstance(f, (str, os.PathLike)):
        return Path(os.path.abspath(f))
    if hasattr(f, "name"):
        assert f is not None
        return Path(os.path.abspath(f.name))
    raise ValueError(f"Unable to create file for: '{f}'")


def save_model(model: ModelProto,
               f: IO[bytes] | str | os.PathLike | Path,
               save_as_external_data = False) -> None:
    """Saves an ONNX model.

    Saves an ONNX model normally, unless the protobuf size is greater than 2GB. Otherwise,
    it will save the model using `save_as_external_data`. If `save_as_external_data` is used,
    the `output_filename` will be used as a directory name to save the artifacts.
    Args:
        model:
            The model to save.
        f:
            The filename to save the model as.
    """
    f = _get_file_path(f)
    kwargs = {}
    if get_model_size(model) >= 2 or save_as_external_data :
        if get_model_size(model) >= 2:
            log_info("Model is over 2GB. Saving model with `save_as_external_data`")
        else:
            log_info("Saving model with `save_as_external_data`")

        dirname = f.parent
        location = f"{f.stem}.data"
        kwargs["save_as_external_data"] = True
        kwargs["all_tensors_to_one_file"] = False
        kwargs["location"] = location
        kwargs["convert_attribute"] = True

        if (dirname / location).exists():
            (dirname / location).unlink()

    onnx.save(model, f.as_posix(), **kwargs)

# TODO: Ugly and should be clean/merged with save_model
def qairt_save_model(model: ModelProto, path: Text):
    """
    Save the onnx model to disk.

    :param ModelProto model: Onnx model proto instance.
    :param Text path: Path at which the model is to be saved.
    """
    make_dirs(path)
    model_name = "qnn_onnx_converter_model.onnx"
    if os.path.isfile(path):
        model_name = os.path.splitext(os.path.basename(path))[0]

    weight_file_name = model_name.split(".")[0] + ".data"

    # onnx.save doesn't support saving the model > 2GB ,
    if model.ByteSize() > onnx.checker.MAXIMUM_PROTOBUF:  # GB
        model_dir = os.path.abspath(os.path.dirname(path))
        convert_model_to_external_data(model, file_name=weight_file_name)
        onnx.save(model, path)
        load_external_data_for_model(model, model_dir)
    else:
        save_model(model, path)


def make_dirs(filename: Text) -> Text:
    """
    create directory recursively.

    :param Text filename: Filename to create directory.
    :return Text: filename.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    return filename


def convert_model_to_external_data(model: ModelProto, file_name: str) -> None:
    """
    Convert the tensors in the given model by updating their data location and
    data offset parameters. Note: This API will not convert data to external data
    but it will populate external data fields in each tensors. Actual conversion
    to external data will happen via onnx.save API.

    :param onnx.ModelProto model: Onnx model proto instance.
    :param str file_name: Path of the onnx external data file.
    """
    for tensor in get_all_tensors(model):
        if (
            tensor.HasField("raw_data")
            and sys.getsizeof(tensor.raw_data) >= ONNX_EXTERNAL_DATA_THRESHOLD
        ):
            set_external_data(tensor, file_name)


def get_all_tensors(model: ModelProto) -> List[TensorProto]:
    """
    Get the list of all the tensors e.g. Initializer and constant attribute tensors
    from the onnx model.

    :param onnx.ModelProto model: Onnx model proto instance.
    :return List[onnx.TensorProto]: List of all the tensors in the model.
    """
    tensors = []
    for graph in get_graphs(model):
        tensors.extend(graph.initializer)

        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("t"):
                    tensors.append(attribute.t)
                tensors.extend(attribute.tensors)
    return tensors

def remove_external_data_from_model(model: ModelProto) -> ModelProto:
    """
    Remove external data fields from Model Proto object.

    :param onnx.ModelProto model: Onnx model reference.
    :return onnx.ModelProto: Updated onnx model.
    """
    for tensor in get_all_tensors(model):
        if uses_external_data(tensor):
            tensor.data_location = TensorProto.DEFAULT
            del tensor.external_data[:]

    return model


def native_checker(model: ModelProto) -> bool:
    """Returns the result of onnx model checker as well as evaluate the model.

    Args:
        model:
            Model to check.

    Returns:
        Boolean indicating the success/failure of the Native Onnx checker
    """
    success = True
    # Calling graph checker for sanity checking about the graph's node names,
    # initializer names etc.
    graph_check_status = graph_checker(model)
    if not graph_check_status:
        log_warning("Duplicate naming found in the graph.")

    try:
        if model.ByteSize() < onnx.checker.MAXIMUM_PROTOBUF:
            onnx.checker.check_model(model)
        else:
            # large models try to convert through a temporary file
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_model_path = os.path.join(tmpdirname, "model.onnx")
                qairt_save_model(model, temp_model_path)
                onnx.checker.check_model(temp_model_path)
    except Exception as e:
        log_warning("The model is invalid: %s" % e)
        return False

    return success
