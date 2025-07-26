#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright 2024 Qualcomm Technologies, Inc. All rights reserved.
#  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
#
#  The party receiving this software directly from QTI (the "Recipient")
#  may use this software as reasonably necessary solely for the purposes
#  set forth in the agreement between the Recipient and QTI (the
#  "Agreement"). The software may be used in source code form solely by
#  the Recipient's employees (if any) authorized by the Agreement. Unless
#  expressly authorized in the Agreement, the Recipient may not sublicense,
#  assign, transfer or otherwise provide the source code to any third
#  party. Qualcomm Technologies, Inc. retains all ownership rights in and
#  to the software
#
#  This notice supersedes any other QTI notices contained within the software
#  except copyright notices indicating different years of publication for
#  different portions of the software. This notice does not supersede the
#  application of any third party copyright notice to that third party's
#  code.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
import torch
from aimet_torch.v2.nn import QuantizationMixin
from aimet_torch.pro.custom_modules_for_qnn import _unary_forward

from aimet_torch.pro.ir_graph_op_handler import OpHandler, extract_tensor_encoding, get_op_name, _create_variable_name

### RMSNorm op definition update
class RmsNorm(torch.nn.Module):
    """Custom module for RmsNorm"""
    def __init__(self, input_shape: list, axes: list, epsilon: float):
        super().__init__()
        self.epsilon = epsilon
        self.axes = axes
        normalized_shape = tuple(input_shape[i] for i in axes)
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        # self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RmsNorm
        """
        #input_dtype = x.dtype
        #x = x.to(dtype=torch.float32, copy=True)
        squared_mean = torch.mean(pow(x, 2), dim=self.axes, keepdim=True)
        rms = torch.sqrt(squared_mean + self.epsilon)
        #res = (torch.div(x, rms) * self.weight + self.bias).to(dtype=input_dtype)
        res = (x / rms) * self.weight
        return res

@QuantizationMixin.implements(RmsNorm)
class QuantizedRmsNorm(QuantizationMixin, RmsNorm):
    """ Quantized class definition for IndexSelect """
    forward = _unary_forward

FakeQuantizedRmsNorm = QuantizedRmsNorm


### RMSNormOpHandler update
class RmsNormOphandler(OpHandler):
    """
    Op Handler for RmsNorm
    """
    def __init__(self, op, op_handler_state):
        super().__init__(op, op_handler_state, num_inputs=1)

    def generate_create_code(self):
        op_name = get_op_name(self._op_name)
        epsilon = self._op.attrs_dict['epsilon']
        axes = list(self._op.attrs_dict['axes'])
        inputs = self._op.inputs()
        input_shape = inputs[0].dims()
        op_str = f'\t\tself.{op_name} = custom_modules_for_qnn.RmsNorm({input_shape}, {axes}, {epsilon})'
        self._op_handler_state.model_def_mgr.add_leaf_module_create_code(op_str)
        self._op_handler_state.state_dict[op_name + '.weight'] = torch.from_numpy(inputs[1].get_data())
        self._op_handler_state.prepared_param_name_map[op_name + '.weight'] = (self._op.get_input_names[1], None)
        # self._op_handler_state.state_dict[op_name + '.bias'] = torch.from_numpy(inputs[2].get_data())
        # self._op_handler_state.prepared_param_name_map[op_name + '.bias'] = (self._op.get_input_names[2], None)

        if not self._op_handler_state.ignore_encodings:
            weight_enc = extract_tensor_encoding(inputs[1])
            if weight_enc is not None:
                self._op_handler_state.encodings['param_encodings'][op_name + '.weight'] = weight_enc
            # bias_enc = extract_tensor_encoding(inputs[2])
            # if bias_enc is not None:
            #     self._op_handler_state.encodings['param_encodings'][op_name + '.bias'] = bias_enc

    def generate_execute_code(self):
        """
        Generate code for executing the op as a Pytorch module in file self._op_handler_state.f.
        """
        # TODO: This custom generate_execute_code is a temporary workaround to avoid a converter failure.
        #  Currently converter fails if Cast Ops are added within custom RmsNorm implementation due to naming conflicts
        #  in the exported onnx model. This can be removed once we support Torch 2.4 and / or when converter
        #  supports RmsNorm pattern with Cast Ops
        string_input = self._op_handler_state.op_to_tensors[self.get_input_op_names()[0]]
        string_input = self.transposed_input_names.get(string_input, string_input)

        output_op_name = self.get_ops_output_names()[0]
        self._op_handler_state.op_to_tensors[output_op_name] = _create_variable_name(output_op_name)
        string_output = self._op_handler_state.op_to_tensors[output_op_name]

        execute_str = f'\t\t{string_output} = self.{get_op_name(self._op_name)}({string_input})\n'

        self._op_handler_state.model_def_mgr.add_execute_code(execute_str, self._op, [string_input], [string_output])
