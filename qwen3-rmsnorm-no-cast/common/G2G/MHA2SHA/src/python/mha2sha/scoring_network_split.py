# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import argparse
import json
from typing import Tuple, List, Dict, OrderedDict
import numpy as np
import onnxruntime
from onnx import mapping, helper
from onnx.onnx_pb import (
    AttributeProto,
    GraphProto,
    NodeProto,
    ModelProto,
    TensorProto,
    ValueInfoProto,
)
from mha2sha.defs.tensor_info import get_input_info, get_output_info
from mha2sha.utils.utils import update_all_mapping_dicts
from mha2sha.utils import onnx as ou
from mha2sha.utils.clean import clean_model, topological_sort
from mha2sha.utils.op_factory import OpFactory

def generate_random_test_data(input_info_dict: Dict) -> Dict[str, np.ndarray]:
    """Generate the test inputs based on given shape and data type (Regular).

    Args:
        input_info_dict:
            A dict with mapping from input name to another dict having info regarding input shape and input dtype.

    Returns:
    A dict with mapping from input name to test data of the input in np.array format.
    """

    final_inputs = OrderedDict()
    for input_name, tensor in input_info_dict.items():
        input_shape = tensor.shape
        input_dtype = tensor.dtype
        final_inputs[input_name] = np.random.rand(*input_shape).astype(input_dtype)
    return final_inputs

class ScoringNetworkSpliter:
    def __init__(
            self,
            model: ModelProto,
            encodings: Dict,
            use_conv: False):

        self.model = model
        self.use_conv = use_conv
        self.encodings = encodings
        self.concat_output_head_on_batch_dim = True

        update_all_mapping_dicts(self)
        self._op_factory = OpFactory(
            self.tensor_name_set,
            self.model,
            self.node_name_mapping_dict,
            self.mha_model_input_names_index_dict,
            None
        )

        self.matmul_node_list = [node for node in self.model.graph.node if node.op_type in "MatMul"]
        self.anchor_tensor_list = []
        self.key_tensor_list = []
        self.score_tensor_list = []

        for matmul_node in self.matmul_node_list:
            self.anchor_tensor_list.append(matmul_node.input[0])
            self.key_tensor_list.append(matmul_node.input[1])
            self.score_tensor_list.append(matmul_node.output[0])

        # use 1st key node to get model info: concat_head_on_batch_dim, head_num, head_dum, and seq_len
        key_tensor_idx = self.mha_model_input_names_index_dict[self.key_tensor_list[0]]
        key_node = self.model.graph.input[key_tensor_idx]
        self.concat_head_on_batch_dim = True if key_node.type.tensor_type.shape.dim[0].dim_value != 1 else False

        if self.concat_head_on_batch_dim:
            self.head_num = key_node.type.tensor_type.shape.dim[0].dim_value
        else:
            self.head_num = key_node.type.tensor_type.shape.dim[1].dim_value
        self.head_dim = key_node.type.tensor_type.shape.dim[2].dim_value
        self.seq_len  = key_node.type.tensor_type.shape.dim[3].dim_value

        # Get output score shape, delete output tensor and make a new one.
        output_dim_len = len(self.model.graph.output[0].type.tensor_type.shape.dim)

        # Update output tensor shape when concat_output_head_on_batch_dim
        if (not self.concat_head_on_batch_dim and self.concat_output_head_on_batch_dim):
            for i in range(len(self.model.graph.input)):
                self.model.graph.input[i].type.tensor_type.shape.dim[0].dim_value = self.head_num
                self.model.graph.input[i].type.tensor_type.shape.dim[1].dim_value = 1
            self.concat_head_on_batch_dim = True

        # Update output tensor shape when concat_output_head_on_batch_dim
        if self.concat_output_head_on_batch_dim:
            for i in range(len(self.model.graph.output)):
                self.model.graph.output[i].type.tensor_type.shape.dim[0].dim_value = self.head_num
                self.model.graph.output[i].type.tensor_type.shape.dim[1].dim_value = 1

        self.score_output_shape = [self.model.graph.output[0].type.tensor_type.shape.dim[i].dim_value for i in range(output_dim_len)]

    def split_batch_matmul(self):
        if self.use_conv:
            model, encodings_to_map = self._split_batch_matmul_to_conv()
        else:
            model, encodings_to_map = self._split_batch_matmul()

        # Remove old matmul modes
        matmul_node_index = []
        for i, node in enumerate(model.graph.node):
            if node.name in [node.name for node in self.matmul_node_list]:
                matmul_node_index.append(i)

        for del_idx in reversed(matmul_node_index):
            del model.graph.node[del_idx]

        model = topological_sort(clean_model(self.model))

        split_encodings = {k : self.encodings[k] for k in self.encodings.keys() if k != 'activation_encodings'}

        new_activation_encodings = {}

        for key, value in encodings_to_map.items():
            curr_encodings = self.encodings['activation_encodings'][key]
            for item in value:
                new_activation_encodings[item] = curr_encodings

        split_encodings['activation_encodings'] = self.encodings['activation_encodings'] | new_activation_encodings

        return model, split_encodings

    def _split_batch_matmul_to_conv(self):
        """
        anchor: [head, 1,        1, head_dim] / [1, head,        1, head_dim]
        key   : [head, 1, head_dim,  seq_len] / [1, head, head_dim,  seq_len]
        score : [head, 1,        1,  seq_len] / [1, head,        1,  seq_len]

        anchor[1, 1, HW, Cin]   @  key[1, 1, Cin, Cout] = score [1, 1, HW, Cout]
        anchor[1, Cin, H, W]  conv key[Cin, Cout, 1, 1] = score [1, 1, HW, Cout]
        """
        slice_dim = 0 if self.concat_head_on_batch_dim else 1

        for anchor_name, keys_name, score_name in zip(self.anchor_tensor_list, self.key_tensor_list, self.score_tensor_list):
            # [1, head, 1, head_dim]/[head, 1, 1, head_dim] -> [head, head_dim, 1, 1]
            anchor_node, _anchor_reshape_init = self._op_factory.get_reshape_op(
                anchor_name, [self.head_num, self.head_dim, 1, 1]
            )
            self.model.graph.node.extend([anchor_node])
            self.model.graph.initializer.extend(_anchor_reshape_init)

            conv_node_list = []
            for head in range(self.head_num):
                _anchor_node, _anchor_slice_init_list = self._op_factory.get_slice_op(
                    anchor_node, start=head, end=head + 1, axis=0
                ) # anchor_node [head_num, head_dim, 1, 1] -> [1, head_dim, 1, 1] (N, Cin, H, W)

                _keys_slice_node, _keys_slice_init_list = self._op_factory.get_slice_op(
                    keys_name, start=head, end=head + 1, axis=slice_dim
                ) # keys_name [head, 1, head_dim, seq_len] / [1, head, head_dim, seq_len] -> [1, 1, head_dim, seq_len]

                _keys_node = self._op_factory.get_transpose_op(
                    _keys_slice_node, [3, 2, 0, 1]
                ) # [1, 1, head_dim, seq_len] -> [seq_len, head_dim, 1, 1] (Cout, Cin, 1, 1)

                _conv_node = self._op_factory.get_conv_op(
                    input_node=_anchor_node, # [1, head_dim, 1, 1] (N, Cin, H, W)
                    weight_tensor_name=_keys_node, # [seq_len, head_dim, 1, 1] (Cout, Cin, 1, 1)
                    bias_tensor_name=None,
                    kernel_shape=[1, 1],
                    padding=[0, 0, 0, 0],
                    strides=[1, 1],
                    propose_op_name="ScoringConv",
                    output_tensor_name=None
                )
                self.model.graph.node.extend([_anchor_node, _keys_slice_node, _keys_node, _conv_node])
                self.model.graph.initializer.extend(_anchor_slice_init_list+_keys_slice_init_list)
                conv_node_list.append(_conv_node)

            concat_node = self._op_factory.get_concat_op(
                conv_node_list, 0
            )

            if self.concat_head_on_batch_dim or self.concat_output_head_on_batch_dim:
                concat_reshape_node, concat_shape_init = self._op_factory.get_reshape_op(
                    concat_node, [self.head_num, 1, 1, self.seq_len]
                ) # [head, seq_len, 1, 1] -> [head, 1, 1, seq_len]
            else:
                concat_reshape_node, concat_shape_init = self._op_factory.get_reshape_op(
                    concat_node, [1, self.head_num, 1, self.seq_len]
                ) # [head, seq_len, 1, 1] -> [1, head, 1, seq_len]

            concat_reshape_node.output[0] = score_name
            self.model.graph.node.extend([concat_node, concat_reshape_node])
            self.model.graph.initializer.extend(concat_shape_init)

        return self.model, {}

    def _split_batch_matmul(self):
        """
        anchor: [head, 1,        1, head_dim] / [1, head,        1, head_dim]
        key   : [head, 1, head_dim,  seq_len] / [1, head, head_dim,  seq_len]
        score : [head, 1,        1,  seq_len] / [1, head,        1,  seq_len]

        anchor[1, 1, HW, Cin]  @  key[1, 1, Cin, Cout] = score [1, 1, HW, Cout]
        """
        slice_dim = 0 if self.concat_head_on_batch_dim else 1
        encoding_mapping = {}

        for anchor_name, keys_name, score_name in zip(self.anchor_tensor_list, self.key_tensor_list, self.score_tensor_list):
            score_list = []

            # Get output score shape, delete output tensor and make a new one.
            del self.model.graph.output[0]
            score_tensor = helper.make_tensor_value_info(
                score_name, TensorProto.FLOAT, self.score_output_shape
            )
            self.model.graph.output.append(score_tensor)

            # anchor: [head, 1,        1, head_dim] / [1, head,        1, head_dim]
            # key   : [head, 1, head_dim,  seq_len] / [1, head, head_dim,  seq_len]
            matmul_node_list = []
            for head in range(self.head_num):
                # anchor_node [head_num, head_dim]
                _anchor_slice_node, _anchor_slice_init_list = self._op_factory.get_slice_op(
                    anchor_name, start=head, end=head + 1, axis=slice_dim
                )

                _keys_slice_node, _keys_slice_init_list = self._op_factory.get_slice_op(
                    keys_name, start=head, end=head + 1, axis=slice_dim
                ) # [1, head_num, head_dim, seq_len] -> [1, 1, head_dim, seq_len]

                _matmul_node = self._op_factory.get_matmul_op(
                                input_node_1 = _anchor_slice_node,
                                input_node_2 = _keys_slice_node,
                            )
                self.model.graph.node.extend([_anchor_slice_node, _keys_slice_node, _matmul_node])
                self.model.graph.initializer.extend(_anchor_slice_init_list+_keys_slice_init_list)
                matmul_node_list.append(_matmul_node)

                score_list.append(_matmul_node.output[0])

            # [1, 1, 1, seq_len] -> [head_num, 1, 1, seq_len] or [1, head_num, 1, seq_len]
            concat_dim = 0 if self.concat_output_head_on_batch_dim else slice_dim
            concat_node = self._op_factory.get_concat_op(
                matmul_node_list, concat_dim
            )
            concat_node.output[0] = score_name
            self.model.graph.node.append(concat_node)

            encoding_mapping[score_name] = score_list

        return self.model, encoding_mapping

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-path",
    default=".",
    help="ONNX model path",
    required=True,
)
parser.add_argument(
    "--encoding-path",
    default=".",
    help="ONNX model encodings path",
    required=True,
)
parser.add_argument(
    "--output-path",
    default=".",
    help="Output ONNX model path",
    required=True,
)
parser.add_argument(
    "--use-conv",
    action='store_true',
    help="Use Conv instead of MatMul in split graph",
)

if __name__ == "__main__":
    args = parser.parse_args()
    model_path = args.model_path
    encoding_path = args.encoding_path
    onnx_output_filename = args.output_path
    use_conv = args.use_conv

    model, model_path = ou.load_model(model_path)

    inp_specs = get_input_info(model) # Dict[str, TensorInfo]
    out_specs = get_output_info(model) # Dict[str, TensorInfo]

    np_inputs = generate_random_test_data(inp_specs) # Dict[str, TensorInfo]
    output_names = []
    for key in out_specs.keys():
        output_names.append(key)

    _, golden_outputs = ou.run_model_on_ort(
                    model_path, np_inputs, output_names
                )

    encodings = {}
    with open(encoding_path, "r") as f:
        encodings = json.load(f)

    spliter = ScoringNetworkSpliter(model, encodings, use_conv)

    split_model, split_encodings = spliter.split_batch_matmul()
    ou.save_model(split_model, onnx_output_filename + ".onnx")
    print(f"Model saved at {onnx_output_filename}.onnx")

    with open(onnx_output_filename + ".encodings", 'w') as f:
        json.dump(split_encodings, f, indent=4)
    print(f"Encodings saved at {onnx_output_filename}.encodings")

    if spliter.concat_head_on_batch_dim:
        updated_np_inputs = {}
        for key, value in np_inputs.items():
            if value.shape[0] == 1:
                value = value.transpose((1, 0, 2, 3))
            updated_np_inputs[key] = value
        np_inputs = updated_np_inputs

    _, converted_model_outputs = ou.run_model_on_ort(
        onnx_output_filename + ".onnx", np_inputs, output_names
    )

    for i, (golden_output, converted_output) in enumerate(zip(golden_outputs, converted_model_outputs)):
        if golden_output.shape[0] != converted_output.shape[0]:
            golden_output = golden_output.transpose((1, 0, 2, 3))
        print(f"{output_names[i]=} MAD = {str(np.abs(golden_output - converted_output).max())}")