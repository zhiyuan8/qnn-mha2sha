# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from collections import OrderedDict
from typing import Dict, List, Text, Tuple, Union

from onnx import mapping
from onnx.onnx_pb import ModelProto, ValueInfoProto

from mha2sha.defs.axis_format import AxisFormat
from mha2sha.utils.logger import log_debug
from mha2sha.utils.onnx import get_inputs, get_outputs, get_shape_from_value_info_proto


class TensorInfo:
    def __init__(self, name: str, shape: List, dtype: Text, layout) -> None:
        """
        Store the properties of the tensor.

        :param str name: Name of the tensor
        :param List shape: shape of the tensor
        :param str type: type of the tensor.
        :param layout: layout of the tensor.
        """
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.layout = layout

    def __check_shapes(self, shape1: List, shape2: List) -> bool:
        """
        Check different shapes values of given tensors,
        including the dynamic tensor shapes.

        :param List shape1: Shape of tensor-1
        :param List shape2: Shape of tensor-2
        :return bool: True if the shapes are matching else False.
        """
        if len(shape1) != len(shape2):
            return False
        if shape1 == shape2:
            return True

        for s1, s2 in zip(shape1, shape2):
            if (isinstance(s1, int) and isinstance(s2, int)) and (s1 != s2):
                if s1 == -1 and s2 > 0:
                    return True
                if s1 > 0 and s2 == -1:
                    return True
                return False
        return True

    def __eq__(self, other) -> bool:
        """
        Compare the current TensorInfo object with the user provided object.
        :param TensorInfo other: TensorInfo object to compare with.
        :return bool: True if both are same else False.
        """
        if (
            (self.name == other.name)
            and (self.__check_shapes(self.shape, other.shape))
            and (self.dtype == other.dtype)
            and (self.layout == other.layout)
        ):
            return True
        return False


def elem_type_to_name(elem_type: int) -> str:
    """
    Function to convert the elem_type to its
    equivalent name based on onnx proto

    :param type(int): elem_type int
    :return type(str): elem_type name
    """
    type_map = mapping.TENSOR_TYPE_TO_NP_TYPE
    if elem_type in type_map.keys():
        return type_map[elem_type].name
    else:
        return "undefined"


def determine_layout(tensor_shape: Union[Tuple[int], List[int]]) -> Text:
    """
    Gets the tensor shape layout of a given tensor.

    :param :tensor_shape (Tuple[int]):
        The shape, including batch dimension.
    :returns: Text: The determined data layout.
    """
    layout = AxisFormat
    if (
        not isinstance(tensor_shape, Tuple) and
        not isinstance(tensor_shape, List)
    ):
        log_debug(
            "Tensor Layout cant be obtained from "
            f"tensor shape: {tensor_shape}"
        )
        return layout.NONTRIVIAL

    # The ratio is closer to ~1, the closer a and b are.
    def minmax_ratio(a, b):
        return abs(max(a, b) / min(a, b))

    # Assume all shapes includes unchanged batch dimension at index 0, so we
    # need to check shape[1:].
    if len(tensor_shape) == 4:
        unknown_cnt = sum(
            [1 if not isinstance(s, int) else 0 for s in tensor_shape[1:]]
        )
        if unknown_cnt >= 1:
            log_debug(
                "One or more unknown shape value present "
                f"in tensor shape: {tensor_shape}."
            )
            if 3 in tensor_shape and unknown_cnt == 2:
                # This means the shape is [unk, 3, unk, unk] or
                # [unk, unk, unk, 3] unknown_cnt == 2 means two
                # axes other than batch dim are dynamic. If we
                # found 3 channels then we can determine layout based
                # the index at which 3 channels occur.
                idx = tensor_shape.index(3)
                if idx == 1:
                    return layout.NCS
                elif idx == 3:
                    return layout.NSC
            else:
                return layout.NONTRIVIAL
        # Typically, H and W are quite close,
        # so if minmax_ratio(0, 1) > minmax_ratio(1, 2), then we assume CHW.
        if minmax_ratio(tensor_shape[1], tensor_shape[2]) > minmax_ratio(
            tensor_shape[2], tensor_shape[3]
        ):
            return layout.NCS
        return layout.NSC
    elif len(tensor_shape) == 5:
        unknown_cnt = sum(
            [1 if not isinstance(s, int) else 0 for s in tensor_shape[1:]]
        )
        if unknown_cnt >= 1:
            log_debug(
                "One or more unknown shape value "
                f"present in tensor shape: {tensor_shape}."
            )
            return layout.NONTRIVIAL
        # For yolo models. Also need to check this for
        # spatio temporal models e.g. 3DCNN models
        if 1 in tensor_shape[1:]:
            # For YoloX with No anchors as it is anchor less detector.
            anchor_idx = list.index(list(tensor_shape[1:]), 1) + 1
        elif 3 in tensor_shape[1:]:
            # For Yolos with 3 anchors
            anchor_idx = list.index(list(tensor_shape[1:]), 3) + 1
        elif 5 in tensor_shape[1:]:
            # For YoloV2 with 5 anchors
            anchor_idx = list.index(list(tensor_shape[1:]), 5) + 1
        elif 9 in tensor_shape[1:]:
            # For YoloV2 with 9 anchors
            anchor_idx = list.index(list(tensor_shape[1:]), 9) + 1
        else:
            return layout.NONTRIVIAL

        if anchor_idx == 1:
            return layout.NDHWC
        elif anchor_idx == 2:
            return layout.NCDHW
        else:
            return layout.NONTRIVIAL

    elif len(tensor_shape) == 3:
        return layout.NFC

    elif len(tensor_shape) == 2:
        return layout.NC
    else:
        log_debug(
            "Cannot determine layout for tensor "
            f"with shape: {tensor_shape}."
        )
        return layout.NONTRIVIAL


def get_tensor_info_from_value_info_proto(
    val_info: ValueInfoProto, use_onnx_def_symbols: bool = False
) -> TensorInfo:
    """
    Converts tensor's value info into TensorInfo object which posses
    information about name, shape, dtype and layout.

    :param onnx.ValueInfoProto val_info: Tensor information in terms
        of ValueInfoProto.
    :param bool use_onnx_def_symbols: If the shapes need to contain onnx
        defined symbol then pass True. For False onnx defined symbols
        will be replaced by -1. Defaults to False.
    :return TensorInfo: Information about tensor in terms of TensorInfo which
        contains its name, shape, dtype and layout.
    """
    tensor_name = val_info.name
    tensor_shape = get_shape_from_value_info_proto(
        val_info, use_onnx_def_symbols
    )
    tensor_dtype = elem_type_to_name(val_info.type.tensor_type.elem_type)
    tensor_layout = determine_layout(tensor_shape)

    return TensorInfo(
        tensor_name,
        tensor_shape,
        tensor_dtype,
        tensor_layout
    )


def get_input_info(
    model: ModelProto, use_onnx_def_symbols: bool = False
) -> Dict[str, TensorInfo]:
    """
    Get the graph input info as TensorInfo object.

    :param ModelProto model: Onnx ModelProto instance.
    :param bool use_onnx_def_symbols: If the shapes need to contain onnx
        defined symbol then pass True. For False onnx defined symbols
        will be replaced by -1. Defaults to False.
    :return Dict[str, TensorInfo]: Mapping of graph's input name
        to its TensorInfo.
    """
    model_inputs = get_inputs(model)
    input_specs = OrderedDict()
    for model_input in model_inputs:
        input_specs[model_input.name] = get_tensor_info_from_value_info_proto(
            model_input, use_onnx_def_symbols
        )
    return input_specs

def get_output_info(
    model: ModelProto, use_onnx_def_symbols: bool = False
) -> Dict[str, TensorInfo]:
    """
    Get the graph output info as TensorInfo object.

    :param ModelProto model: Onnx ModelProto instance.
    :param bool use_onnx_def_symbols: If the shapes need to contain onnx defined
        symbol then pass True. For False onnx defined symbols will be replaced
        by -1. Defaults to False.
    :return Dict[str, TensorInfo]: Mapping of graph's output name to its TensorInfo.
    """
    model_outputs = get_outputs(model)
    output_specs = OrderedDict()
    for model_output in model_outputs:
        output_specs[model_output.name] = get_tensor_info_from_value_info_proto(
            model_output, use_onnx_def_symbols
        )
    return output_specs
