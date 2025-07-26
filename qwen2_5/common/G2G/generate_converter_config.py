# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import argparse
import copy

import onnx
import yaml
from yaml import SafeDumper


# Get dict of input output tensor names and rank
def get_graph_input_output(onnx_file) -> dict:
    graph_inputs = dict()
    graph_outputs = dict()
    g = onnx.shape_inference.infer_shapes(onnx.load(onnx_file, load_external_data=False))

    for input_node in g.graph.input:
        input_rank = 0
        tensor_type = input_node.type.tensor_type
        if (tensor_type.HasField("shape")):
            input_rank = len(tensor_type.shape.dim)
        graph_inputs[input_node.name] = input_rank

    for output_node in g.graph.output:
        output_rank = 0
        tensor_type = output_node.type.tensor_type
        if (tensor_type.HasField("shape")):
            output_rank = len(tensor_type.shape.dim)
        graph_outputs[output_node.name] = output_rank

    return graph_inputs, graph_outputs

# Generate converter io config needed for qairt
def generate_converter_config(model_name, onnx_file, output_dir):
    # get dict of input and output to graph and their rank
    graph_inputs, graph_outputs = get_graph_input_output(onnx_file)

    SafeDumper.add_representer(type(None), lambda dumper, value: dumper.represent_scalar(u'tag:yaml.org,2002:null', ''))
    converter_io_config_yaml = {'Input Tensor Configuration': [],
                                 'Output Tensor Configuration': []}

    # Define single input dict for config
    input_dict = {
        'Name': None,
        'Src Model Parameters': {
            'DataType': None,
            'Layout': None,
        },
        'Desired Model Parameters': {
            'DataType': None,
            'Layout': None,
            'Shape': None,
            'Color Conversion': None,
            'QuantParams': {
              'Scale': None,
              'Offset': None,
            }
        }
    }
    # Define single output dict for config
    output_dict = {
        'Name': None,
        'Src Model Parameters': {
            'DataType': None,
            'Layout': None,
        },
        'Desired Model Parameters': {
            'DataType': None,
            'Layout': None,
            'QuantParams': {
              'Scale': None,
              'Offset': None,
            }
        }
    }

    for yaml_key, graph_io_names, io_dict in [('Input Tensor Configuration', graph_inputs, input_dict),
                                             ('Output Tensor Configuration', graph_outputs, output_dict)]:
        for io_name in graph_io_names:
            dict_content = copy.deepcopy(io_dict)
            dict_content['Name'] = io_name

            # Check dimensions of output
            dims = graph_io_names[io_name]
            if dims == 4:
                dict_content['Src Model Parameters']['Layout'] = 'NCHW'
                dict_content['Desired Model Parameters']['Layout'] = 'NHWC'
            elif dims == 2:
                dict_content['Src Model Parameters']['Layout'] = 'NF'
                dict_content['Desired Model Parameters']['Layout'] = 'NF'

            # Add single output config to converter config
            converter_io_config_yaml[yaml_key].append(dict_content)

    # write config
    with open(f'{output_dir}/{model_name}_converter_io_config.yaml', 'w') as file:
        yaml.safe_dump(converter_io_config_yaml, file, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate in/out config for qairt-converter.")
    parser.add_argument("-m", "--model_name", help="Name of model config will be generated for.", required=True)
    parser.add_argument("-i", "--onnx_file", help="Path to onnx file.", required=True)
    parser.add_argument("-o", "--output_dir", help="Path where to dump config.", default="./host_linux/exports")

    args = parser.parse_args()
    model_name = args.model_name
    onnx_file = args.onnx_file
    output_dir = args.output_dir

    generate_converter_config(model_name, onnx_file, output_dir)
