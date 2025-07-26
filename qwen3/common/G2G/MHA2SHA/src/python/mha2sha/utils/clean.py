# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from typing import List, Union

from mha2sha.utils.onnx import (
    get_graphs,
    get_initializer_mappings,
    get_node_by_input_name,
    get_node_by_output_name,
    get_nodes,
)
from onnx.onnx_pb import GraphProto, ModelProto, NodeProto, TensorProto


def remove_zero_dim_init_input_from_concat(model: ModelProto) -> ModelProto:
    """
    This function is removing initializer's zero Sized tensor from the concat node

    :param ModelProto model: Input ONNX Model.
    :return ModelProto: Modified ONNX Model.
    """
    return remove_zero_dim_input_from_node(model, "Concat")


def remove_zero_dim_input_from_node(model: ModelProto, op_type: str) -> ModelProto:
    """
    This function is removing initializer's zero Sized tensor from the Input node

    :param ModelProto model: Input ONNX Model.
    :param str op_type: ONNX operation type.
    :return ModelProto: Modified ONNX Model.
    """
    # Aggregating zero dims tensors from initializer.
    filtered_nodes = get_nodes_by_op_type(model, op_type)
    zero_dim_init_dict = {
        init_name: init
        for init_name, init in get_initializer_mappings(model).items()
        if init.dims == [0]
    }

    if not zero_dim_init_dict:
        return model

    node_idxs_to_check = set()
    initializers_to_remove = set()
    # Removing node inputs which are having zero dimensions
    # and part of initializer also.
    for node_idx, node in enumerate(filtered_nodes):
        for input in node.input:
            if input not in zero_dim_init_dict:
                continue
            init = zero_dim_init_dict[input]
            initializers_to_remove.add(init.name)

            graph = get_graph_by_node(model, node)
            node_graph_idx = list(graph.node).index(node)
            graph.node[node_graph_idx].input.remove(input)

            node_idxs_to_check.add(node_idx)

    # Removing Initializer which are having zero dimensions.
    for name in initializers_to_remove:
        init = zero_dim_init_dict[name]
        graph = get_graph_by_initializer(model, init)
        graph.initializer.remove(init)

    # Removing nodes which are having zero inputs
    for idx in node_idxs_to_check:
        node = filtered_nodes[idx]
        if len(node.input) == 0:
            graph = get_graph_by_node(model, node)
            graph.node.remove(node)
    return model


def get_nodes_by_op_type(model: ModelProto, op_type: str) -> List[NodeProto]:
    """
    Get all the nodes from underlying onnx model based on op type.

    :param ModelProto model: Onnx Model graph proto.
    :param str op_type: optype
    :return List[NodeProto]: Underlying onnx nodes.
    """
    nodes = []
    for node in get_nodes(model):
        if node.op_type == op_type:
            nodes.append(node)
    return nodes


def clean_model(model: ModelProto) -> ModelProto:
    """
    Cleans up the model by removing unused nodes and dangling inputs/outputs
    :param model: Onnx ModelProto
    :return ModelProto: Cleaned ModelProto
    """
    # Removing empty input tensors from the Concat Node
    model = remove_zero_dim_init_input_from_concat(model)

    # Traverse back from output node till the input node
    # Remove unused nodes
    # Then remove unused input and outputs
    graph = model.graph

    # Creating two separate dict for traversing the graph easily
    node_by_output_name = {}
    node_by_input_name = {}
    for n in graph.node:
        # Only single node can be referenced by any output node name
        for n_op in n.output:
            node_by_output_name[n_op] = n
        # More than one node can be referenced by any input node name
        # That's why using list of nodes
        for n_ip in n.input:
            if n_ip not in node_by_input_name.keys():
                node_by_input_name[n_ip] = [n]
            else:
                node_by_input_name[n_ip].append(n)

    # Traverse the graph from the graph output nodes.
    visited_nodes = set()
    stack = []

    for g_op in graph.output:
        if g_op.name in node_by_output_name.keys():
            node = node_by_output_name[g_op.name]
            stack.append(node)

    while len(stack) != 0:
        node = stack.pop()
        if node.name in visited_nodes:
            continue
        visited_nodes.add(node.name)
        for ip_name in node.input:
            if ip_name in node_by_output_name.keys():
                parent_node = node_by_output_name[ip_name]
                if parent_node.name not in visited_nodes:
                    stack.append(parent_node)

    # Till now visited_nodes is populated with nodes connected with graph outputs
    remove_nodes = [n for n in graph.node if n.name not in visited_nodes]

    # Remove nodes from the graph as well as from two dictionaries.
    for r_n in remove_nodes:
        graph.node.remove(r_n)

        # Delete its entries from node_by_input_name, node_by_output_name
        for n_op in r_n.output:
            node_by_output_name.pop(n_op)

        for n_ip in r_n.input:
            if len(node_by_input_name[n_ip]) == 1:
                node_by_input_name.pop(n_ip)
            else:
                node_by_input_name[n_ip].remove(r_n)
                # Remove node "r_n" from list if the input is used in multiple nodes
                # We only want to remove that "r_n" node and want to keep other nodes which uses
                # the same input.

    # Remove unused initializers from the graph.
    unused_initializers = [
        init for init in graph.initializer if init.name not in node_by_input_name
    ]

    for init in unused_initializers:
        # Remove the initializers which are not connected to any node.
        graph.initializer.remove(init)

    # Remove unused value_info from the graph.
    unused_value_infos = [
        val_info
        for val_info in graph.value_info
        if val_info.name not in node_by_input_name
    ]

    for val_info in unused_value_infos:
        # Remove the value_info which are not connected to any node.
        graph.value_info.remove(val_info)

    # Clean any dangling input or output from model's main graph.
    model = remove_unused_inputs_outputs(model)
    return model


def get_graph_by_initializer(
    model: ModelProto, init: TensorProto
) -> Union[None, GraphProto]:
    """
    Get graph which contains given initializer.

    :param ModelProto model: Onnx Model graph proto.
    :param TensorProto node: Initializer instance.
    :return Union[None, GraphProto]: Underlying onnx graph if found else None.
    """
    for graph in get_graphs(model):
        if init in graph.initializer:
            return graph
    return None


def get_graph_by_node(model: ModelProto, node: NodeProto) -> Union[None, GraphProto]:
    """
    Get graph which contains given node.

    :param ModelProto model: Onnx Model graph proto.
    :param NodeProto node: onnx node
    :return Union[None, GraphProto]: Underlying onnx graph if found else None.
    """
    for graph in get_graphs(model):
        if node in graph.node:
            return graph
    return None


def remove_unused_inputs_outputs(model: ModelProto) -> ModelProto:
    """
    Remove the unused/dangling input and output nodes from graph's inputs
    and outputs.

    :param model: Loaded Onnx Model Proto instance
    :return model: Cleaned model with dangling inputs and outputs removed.
    """
    main_graph = model.graph

    node_by_input_name = get_node_by_input_name(model)
    node_by_output_name = get_node_by_output_name(model)

    def is_dangling_graph_output(graph_output_name: str) -> bool:
        """
        Function to check if the graph output is dangling output (e.g. not an
        output of any nodes.)

        :param str graph_output_name: Name of the graph output tensor to check.
        :return bool: True if the given output is a dangling output, else False.
        """
        if graph_output_name not in node_by_output_name.keys():
            return True
        else:
            return False

    def is_dangling_graph_input(graph_input_name: str) -> bool:
        """
        Function to check if the graph input is dangling input (e.g. not an
        input of any nodes.)

        :param str graph_input_name: Name of the graph input tensor to check.
        :return bool: True if the given input is a dangling input, else False.
        """
        if graph_input_name not in node_by_input_name.keys():
            return True
        else:
            return False

    # Identify and remove any dangling graph inputs and graph outputs
    # Note: We shall not remove dangling input/output of subgraph as they may
    #       have connection to parent graph.
    remove_graph_outputs = [
        g_op for g_op in main_graph.output if is_dangling_graph_output(g_op.name)
    ]
    remove_graph_inputs = [
        g_op for g_op in main_graph.input if is_dangling_graph_input(g_op.name)
    ]

    for r_g_op in remove_graph_outputs:
        main_graph.output.remove(r_g_op)

    for r_g_ip in remove_graph_inputs:
        main_graph.input.remove(r_g_ip)

    return model


def topological_sort(model: ModelProto) -> ModelProto:
    """
    Function to get the topologically sorted graphs.

    :return Self instance of the class.
    """
    for graph in get_graphs(model):
        _graph_topological_sort(graph)
    return model


def _graph_topological_sort(graph: GraphProto):
    """
    Function to get the topologically sorted graph from Graph proto

    :param GraphProto graph: GraphProto of the model.
    :return ONNXModelUtils: Self instance of the class.
    """
    deps_count = [0] * len(graph.node)  # dependency count of each node
    deps_to_nodes = {}  # input to node indices
    sorted_nodes = []  # initialize sorted_nodes

    for node_idx, node in enumerate(graph.node):
        # CANNOT use len(node.input) directly because input can be optional
        deps_count[node_idx] = sum(1 for _ in node.input if _)
        if deps_count[node_idx] == 0:  # Constant doesn't depend on any inputs
            sorted_nodes.append(graph.node[node_idx])
            continue
        for input_name in node.input:
            if input_name not in deps_to_nodes:
                deps_to_nodes[input_name] = [node_idx]
            else:
                deps_to_nodes[input_name].append(node_idx)

    # Note: this logic only applies to top level graph since a sub graph could use intializer from parent graph
    initializer_names = [init.name for init in graph.initializer]
    graph_input_names = [input.name for input in graph.input]
    input_names = initializer_names + graph_input_names

    intermediate_output_tensor_names = set()
    for n in graph.node:
        for n_op in n.output:
            if n_op not in intermediate_output_tensor_names:
                intermediate_output_tensor_names.add(n_op)

    for n in graph.node:
        for n_ip in n.input:
            if n_ip == "":
                # Skip the blank named input as that input is an attribute
                # of node but its value is not present. Assuming this as an
                # optional attribute of the node.
                continue
            if (
                n_ip not in initializer_names
                and n_ip not in graph_input_names
                and n_ip not in intermediate_output_tensor_names
            ):
                # If a node's input is not output of any node in the present graph then
                # add that name in input_names with an assumption that the name is part of
                # parent graph's some node's output or the node has no dependency on input
                # and it is a constant node.
                input_names.append(n_ip)

    input_names.sort()
    prev_input_name = None
    for input_name in input_names:
        if prev_input_name == input_name:
            continue
        prev_input_name = input_name
        if input_name in deps_to_nodes:
            for node_idx in deps_to_nodes[input_name]:
                deps_count[node_idx] = deps_count[node_idx] - 1
                if deps_count[node_idx] == 0:
                    sorted_nodes.append(graph.node[node_idx])

    start = 0
    end = len(sorted_nodes)
    while start < end:
        for output in sorted_nodes[start].output:
            if output in deps_to_nodes:
                for node_idx in deps_to_nodes[output]:
                    deps_count[node_idx] = deps_count[node_idx] - 1
                    if deps_count[node_idx] == 0:
                        sorted_nodes.append(graph.node[node_idx])
                        end = end + 1
        start = start + 1

    assert end == len(graph.node), "Graph is not a DAG"
    graph.ClearField("node")
    graph.node.extend(sorted_nodes)
