# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

"""Replaces PyTorch equivalent Linear operations into Conv 1x1's.

For HTP support, PyTorch equivalent Linear operations must be changed into Conv's with 1x1 kernels. Furthermore,
these operations need Transposes before and after the Conv in order to have the dimension correct after replacing
the Linear operation.

ONNX has different "variations" of Linear operations. Firstly, there are MatMul's that have two inputs, where the
second input is the weights of the layer. These can be found by seeing if the second input is an initializer. Regular
MatMuls, will just have two inputs from outputs of other operations. The diagram below, shows the MatMul's this module
are replacing.

                                    Input
                                      |
                                      v
                                +----------+
                                |  MatMul  |<-- Initializer (weights)
                                +----------+
                                      |
                                      v
                                    Output


The other variation, are Gemm operations. These are easier to find as the op type is the search criteria. However,
Gemm operations have attributes which can change how the input shape is used, scalar multipliers, etc. These attributes
are listed below for easy reminders of what they do.

Gemm Attributes
---------------

transA:
    Whether the A input into this op is transposed or not. (0 -> False, 1 -> True)
transB:
    Whether the B input into this op is transposed or not. (0 -> False, 1 -> True)
alpha:
    Scalar multiplier of the product of A*B. (1.0 is default)
beta:
    Scalar multiplier of the input C. (1.0 is default)

.. note::
    Only alpha and beta with their default values of 1 are supported.


Basic usage
-----------

>>> linear_to_conv = LinearToConv(mha2sha)  # Where mha2sha is of type MHA2SHAOptimizer
>>> linear_to_conv.replace()

"""

from typing import Any, Dict, Optional, Tuple, Union
from itertools import chain

import numpy as np

from onnx import numpy_helper
from onnx.onnx_pb import NodeProto, TensorProto

from tqdm import tqdm

from mha2sha.utils.logger import log_info

mha2sha_optimizer = Any  # Causes circular import
LinearTensorInfo = Dict[str, Union[str, np.ndarray, int]]
LinearGemmInfo = Dict[str, float]
LinearInfo = Dict[str, Union[LinearTensorInfo, str, NodeProto, LinearGemmInfo]]

class LinearToConv:
    """Replaces PyTorch equivalent Linear operations into Conv 1x1's.


    Provides an abstracted way to replace PyTorch Linear Operations with Conv 1x1's for HTP support.
    Specifically, replaces MatMuls with initializers as second inputs and Gemm operations. See full
    module docs for futher information.

    Attributes:

    """

    def __init__(self, mha2sha_optim: mha2sha_optimizer) -> None:
        """Initalizes an instance based on the MHA2SHA optimizer provided.

        Args:
            mha2sha_optim:
                MHA2SHAOptimizer instance holding the model loader and model info.
        """

        self.mha2sha_optim = mha2sha_optim

    def replace(self) -> None:
        """Replace Linear operations with Conv 1x1's

        Replace all Linear operations with Conv 1x1's in the ModelProto provided when creating an instance of this class.
        """

        log_info("Replacing Linears with Conv's...")

        # If a MatMul has one of it's inputs in the initializers, then we have Linear style operatiions.
        is_linear_matmul = lambda node: node.op_type == "MatMul" and any([
            inp in self.mha2sha_optim.get_initializer_by_name for inp in node.input
        ])
        is_gemm = lambda node: node.op_type == "Gemm"

        linear_nodes = [
            node for node in self.mha2sha_optim.model.graph.node
            if is_linear_matmul(node) or is_gemm(node)
        ]

        for linear_node in tqdm(linear_nodes):
            # Check if model output in the current Linear's output.
            has_model_output_in_linear_output = any([
                output in self.mha2sha_optim.mha_model_output_names for output in linear_node.output
            ])

            # Collect information on the Linear node.
            linear_info = self._get_linear_info(linear_node, is_gemm(linear_node))

            # We don't need Linear anymore, so we can remove it.
            self.mha2sha_optim.model.graph.node.remove(linear_node)

            # TODO infer input_dim
            last_transpose = self._get_htp_conv_from_linear(
                linear_info=linear_info,
                is_gemm=is_gemm(linear_node),
                last_transpose_output=linear_info["output_name"] if has_model_output_in_linear_output else None
            )

            # If the current Linear does not have any output as model output, then we need to update all the output
            # nodes of the Linear to now consume from the Transpose -> Conv -> Transpose.
            if not has_model_output_in_linear_output:
                self._map_transpose_outputs_to_linears(linear_info, last_transpose)


    def _get_linear_info(self, linear_node: NodeProto, is_gemm: bool) -> LinearInfo:
        """Collects the Linear nodes attributes.

        Collects the Linear node's attributes necessary for converting. Information such as the name of the node, inputs,
        outputs, and the weight are provided. If the Linear node is a Gemm, more information is added about the Gemm
        properites such as if the input is tranposed or not.

        Args:
            linear_node:
                The node to obtain information about.
            is_gemm:
                Flag to add more information if the `linear_node` is a Gemm.

        Returns:
            A Dictionary with information about the Linear node.
        """

        def get_weight_or_bias_info(get_weight: bool = True) -> LinearTensorInfo:
            """Helper function for pulling out information of the weight or bias.
            """
            WEIGHT, BIAS = 1, 2
            linear_weight_or_bias_name = linear_node.input[WEIGHT if get_weight else BIAS]
            return {
                "name": linear_weight_or_bias_name,
                "numpy":  numpy_helper.to_array(self.mha2sha_optim.get_initializer_by_name[linear_weight_or_bias_name]),
                "idx_in_initializer": self.mha2sha_optim.get_initializer_idx_by_name[linear_weight_or_bias_name]
            }

        # We have to pull out all the info and then remove the Linear node. Otherwise, we will have a non DAG error.
        linear_info = {
            "node_name": linear_node.name,
            "non_weight_input_name": linear_node.input[0],
            "weight": get_weight_or_bias_info(),
            "output_name" : linear_node.output[0],
            "input_node" : self.mha2sha_optim.get_node_by_output_name[linear_node.input[0]],
            "output": linear_node.output
        }

        if is_gemm:
            linear_info.update({
                "bias": get_weight_or_bias_info(get_weight=False),
                "attributes": {
                    attr.name: attr.f for attr in linear_node.attribute # .f is the float value of the attribute
                }
            })

            assert linear_info["attributes"]["alpha"] == 1, f"Currently, only Gemms with alpha equal to 1 are supported"
            assert linear_info["attributes"]["beta"] == 1, f"Currently, only Gemms with beta equal to 1 are supported"

        return linear_info

    def _map_transpose_outputs_to_linears(
        self,
        linear_info: LinearInfo,
        last_transpose: NodeProto
    ) -> None:
        """Maps the Transpose's output to the Linear's.

        Mapping and replacing the Linear's outputs to node with the new Transpose's output.

        Args:
            linear_info:
                Dictionary containing information of the Linear node.
            last_transpose:
                The Transpose to map to the Linear nodes outputs.
        """

        # All output nodes as a flattened list.
        output_nodes_of_linear = list(chain.from_iterable(
            map(lambda o: self.mha2sha_optim.get_node_by_input_name[o], linear_info["output"])
        ))

        # Update output nodes of the last Transpose to go to the MatMul's outputs.
        for output_node in output_nodes_of_linear:
            index_of_linear_output = [
                idx for idx, inp in enumerate(output_node.input) if inp == linear_info["output_name"]
            ][0]
            output_node.input[index_of_linear_output] = last_transpose.output[0]

    def _get_htp_conv_from_linear(
        self,
        linear_info: LinearInfo,
        is_gemm: bool,
        input_dim: int = 4,
        last_transpose_output: Optional[str] = None,
    ) -> NodeProto:
        """Gets the `super` node that is a Linear equivalent as a Conv.

        Gives a super node that is better suited for HTP backend specifics. We want the names of the Conv to match
        the original Linear node to have the encodings match up.

        Steps:
        1. Unsqueeze last dim (batch_size, seq_len, emb_dim) -> (batch_size, seq_len, emb_dim, 1)
        2. Premute to BCHW: (batch_size, seq_len, emb_dim, 1) -> (batch_size, emb_dim, seq_len, 1)
        3. Conv2d  (batch_size, emb_dim, seq_len, 1) -> (batch_size, emb_dim, seq_len, 1)
        4. Permute back to B, Seq, Emb (batch_size, emb_dim, seq_len, 1) -> (batch_size, 1, seq_len, emb_dim)

        Args:
            linear_info:
                A dictionary contianing information about the Linear node being replaced. Information can contain
                weight info, names of tensors, etc.
            is_gemm:
                Flag for is the Linear node is a Gemm operation in ONNX.
            input_dim:
                Input dimension of the Linear node.
            last_transpose_output:
                Optional output tensor name of the last Transpose opertation. This is for when the
                MatMul's output is an output of the Graph.

        Returns:
            The last Transpose of the super node of Transpose -> Conv -> Transpose.
        """

        init_list = []
        node_list = []
        linear_weight_should_transpose = not is_gemm or (is_gemm and "transA" in linear_info["attributes"])

        # I, O -> O, I, KH, KW
        # Step 1: Handle Linear weight and bias.
        conv_weight_init, conv_bias_init = self._get_conv_weight_bias_init(
            linear_info, is_gemm, linear_weight_should_transpose
        )

        # Step 2: Handle Input and Reshape nodes.
        transpose_to_BCHW = self._handle_input_and_reshape_into_conv(
            linear_info, is_gemm, linear_weight_should_transpose
        )
        node_list.append(transpose_to_BCHW)

        # Step 3: Create Conv Equivalent.
        conv_as_linear_node = self.mha2sha_optim._op_factory.get_conv_op(
            input_node=transpose_to_BCHW,
            weight_tensor_name=conv_weight_init.name,
            bias_tensor_name=None if not conv_bias_init else conv_bias_init.name,
            kernel_shape=[1, 1],
            padding=[0, 0, 0, 0],
            strides=[1, 1],
            propose_op_name=linear_info["node_name"],
            output_tensor_name=linear_info["output_name"] if linear_info["output_name"] != last_transpose_output else None
        )

        node_list.append(conv_as_linear_node)

        # Step 4: Create Transpose for out of Conv.
        transpose_to_BSE, squeeze_output = self._handle_last_transpose_and_squeeze(
            last_transpose_output, conv_as_linear_node, input_dim
        )
        node_list.append(transpose_to_BSE)

        # Step 5: Squeeze output if output dim = 3: [1, 1, seq_len, vector_out_dim] to [1, seq_len, vector_out_dim]
        if squeeze_output:
            squeeze_output_dim, squeeze_output_dim_list = self.mha2sha_optim._op_factory.get_squeeze_op(
                transpose_to_BSE, 1, last_transpose_output
            )
            init_list.extend(squeeze_output_dim_list)
            node_list.append(squeeze_output_dim)

        self.mha2sha_optim.model.graph.initializer.extend(init_list)
        self.mha2sha_optim.model.graph.node.extend(node_list)

        return node_list[-1]

    def _get_conv_weight_bias_init(
        self,
        linear_info: LinearInfo,
        is_gemm: bool,
        linear_weight_should_transpose: bool
    ) -> Tuple[TensorProto, Optional[TensorProto]]:
        """Get the Conv's weight and bias TensorProtos.

        Takes the Linear's weight and bias - if available - and updates them to be the Conv equivalent. To make things
        more efficient, we update the initializer that holds the Linear weight/bias with the updated one.

        Args:
            linear_info:
                Dictionary containing information of the Linear node.
            is_gemm:
                Flag to include the Conv's bias.
            linear_weight_should_transpose:
                Flag to tell if the weight should be transposed for the Conv node.

        Returns:
            The Conv's weight TensorProto, and either None or the bias TensorProto if it is a Gemm node.
        """

        # NOTE: DO NO CHANGE THE LOGIC HERE FOR TRANSPOSE!!
        # For some reason, ONNX will fail if the code is cleaned up.
        if linear_weight_should_transpose:
            conv_weight_init = numpy_helper.from_array(
                linear_info["weight"]["numpy"].T[..., None, None],
                name=linear_info["weight"]["name"]
            )
        else:
            conv_weight_init = numpy_helper.from_array(
                linear_info["weight"]["numpy"][..., None, None],
                name=linear_info["weight"]["name"]
            )

        weight_bias_to_idx = [
            (conv_weight_init, linear_info["weight"]["idx_in_initializer"]),
        ]

        conv_bias_init = None
        if is_gemm:
            conv_bias_init = numpy_helper.from_array(
                linear_info["bias"]["numpy"],
                name=linear_info["bias"]["name"]
            )

            weight_bias_to_idx.append((conv_bias_init, linear_info["bias"]["idx_in_initializer"]))

        # We are replacing the Linear's weight initializer to now hold the unsqueezed version for the Conv and
        # the updated bias.
        for w_b_init, idx in weight_bias_to_idx:
            self.mha2sha_optim.model.graph.initializer[idx].CopyFrom(w_b_init)

        return conv_weight_init, conv_bias_init

    def _handle_input_and_reshape_into_conv(
        self,
        linear_info: LinearInfo,
        is_gemm: bool,
        linear_weight_should_transpose: bool
    ) -> NodeProto:
        """Handles the input into the Transpose -> Conv -> Transpose.

        Uses the input node to either find the correct input node if it is a Reshape, or use the one provided and
        construct a Reshape followed by the first Transpose in the pattern.

        Args:
            linear_info:
                Dictionary containing information of the Linear node.
            is_gemm:
                Flag to include the Conv's bias.
            linear_weight_should_transpose:
                Flag to tell if the weight should be transposed for the Conv node, to correctly identify the input dim.

        Returns:
           The first Transpose in the Transpose -> Conv -> Transpose pattern.
        """

        # Handle Input node.
        input_node = linear_info["input_node"]
        if input_node.op_type == "Reshape":
            if input_node.input[0] in self.mha2sha_optim.mha_model_input_names_index_dict.keys():
                input_node = input_node.input[0]
            else:
                input_node = self.mha2sha_optim.get_node_by_output_name[input_node.input[0]]

        # Handle Reshape based on if Gemm.
        vector_inp_dim = linear_info["weight"]["numpy"].shape[0 if linear_weight_should_transpose else 1]
        if is_gemm and linear_info["attributes"].get("transB", None) == 1:
            # The input shape is [vector_inp_dim, seq_len]
            reshape_node, reshape_init = self.mha2sha_optim._op_factory.get_reshape_op(input_node, [1, 1, vector_inp_dim, -1])
            # [1, 1, vector_inp_dim, seq_len] -> [1, vector_inp_dim, seq_len, 1], B = 1
            transpose_to_BCHW = self.mha2sha_optim._op_factory.get_transpose_op(reshape_node, [0, 2, 3, 1])
        else:
            reshape_node, reshape_init = self.mha2sha_optim._op_factory.get_reshape_op(input_node, [1, 1, -1, vector_inp_dim])
            # [1, 1, seq_len, vector_inp_dim] -> [1, vector_inp_dim, seq_len, 1], B = 1
            transpose_to_BCHW = self.mha2sha_optim._op_factory.get_transpose_op(reshape_node, [0, 3, 2, 1])

        self.mha2sha_optim.model.graph.initializer.extend(reshape_init)
        self.mha2sha_optim.model.graph.node.append(reshape_node)

        return transpose_to_BCHW

    def _handle_last_transpose_and_squeeze(
        self,
        last_transpose_output: Optional[str],
        conv_as_linear_node: NodeProto,
        input_dim: int
    ) -> Tuple[NodeProto, bool]:
        """Handles the input into the Transpose -> Conv -> Transpose.

        Uses the input node to either find the correct input node if it is a Reshape, or use the one provided and
        construct a Reshape followed by the first Transpose in the pattern.

        Args:
            last_transpose_output:
                An optional output that may have been provided for creating this pattern. Only used when the Transpose
                being returned is an output to the model.
            conv_as_linear_node:
                Conv node equivalent to its Linear counterpart.
            input_dim:
                Input dimension of the original Linear.

        Returns:
           The first Transpose in the Transpose -> Conv -> Transpose pattern.
        """

        # [1, vector_out_dim, seq_len, 1] -> [1, 1, seq_len, vector_out_dim]
        # Check output shape when last_transpose_output is not None. i.e. output is a model output
        # If output dim=3, add a squeeze to transpose
        squeeze_output = False
        if last_transpose_output is not None:
            for model_output in self.mha2sha_optim.model.graph.output:
                if model_output.name == last_transpose_output:
                    output_dim = len([d.dim_value for d in model_output.type.tensor_type.shape.dim])
                    assert output_dim in (3, 4), f"expect model output dim = 3 or 4, but got {output_dim}"
                    if output_dim == 3:
                        squeeze_output = True
                    break

        return self.mha2sha_optim._op_factory.get_transpose_op(
            conv_as_linear_node,
            [0, 3, 2, 1], # BSE: Batch, Seq_len, Emd_dim
            output=last_transpose_output if (input_dim != 3 and not squeeze_output) else None  # Only special output if model output and Transpose is last
        ), squeeze_output

