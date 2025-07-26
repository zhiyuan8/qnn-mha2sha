# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

r"""Builds/Updates an ONNX Model to an update AR value.

This class handles the ability to update AR values throughout the model. For example,
if a user starts with an AR of 64 and a context length of 1024, then the past key/value
sequence lengths become 960. If we update to AR 8, then the context length stays the same
but the past key/value sequence length now becomes 1016.


.. note::
    Each method inside of this builder class will only run if 'buildable' is set. This means,
    that for each public API call into this class will first be validated it can run with
    the decorator '_run_if_buildable'.


Basic usage
-----------

>>> ar_builder = ArBuilder(
...     mha2sha, # Where mha2sha is of type MHA2SHAOptimizer
...     ar_value=8,
... )

>>> ar_builder.update_initial_ar_values()
"""

from functools import wraps
import itertools
from typing import Any, Callable, Optional, Union

from onnx import helper, numpy_helper

from onnx.onnx_pb import TensorProto, ValueInfoProto
from mha2sha.utils.logger import log_debug

mha2sha_optimizer = Any  # Causes circular import


def _check_buildable(func: Callable):
    r"""Runs wrapped function if buildable is True

    Args:
        func:
            Function to call to check.

    Returns:
        Results of the function call.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if func.__name__ == "__init__" or (
            hasattr(self, "buildable") and self.buildable
        ):
            return func(self, *args, **kwargs)

    return wrapper


def _run_if_buildable(cls):
    r"""Wrapper to check buildable for all methods.

    Args:
        cls:
            Class to check the methods of.
    """
    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value):
            setattr(cls, attr_name, _check_buildable(attr_value))
    return cls


@_run_if_buildable
class ArBuilder:
    r"""Builder for updating the AR.

    Attributes:
       mha2sha_optim:
           MHA2SHA Optimizer instance.
       ar_value:
           New AR value.
       original_seq_len:
           Original sequence length of the model.
       past_value_ar_value:
           Original past value/key sequence length.
       new_past_seq_len:
           New sequence length.
       input_names_updated_from_ar:
           Set of input names updated by the builder.
       reshapes_not_to_update_for_ar:
           Set of Reshape Op's NOT to update by the builder.
    """

    def __init__(
        self, mha2sha_optim: mha2sha_optimizer, ar_value: Optional[int]
    ) -> None:
        self._buildable = ar_value is not None and mha2sha_optim.handle_past_key_value
        if not self._buildable:
            log_debug("No AR value provided, empty initing ArBuilder - NOT buildable")
        else:
            self._mha2sha_optim = mha2sha_optim
            self._ar_value = ar_value
            self._original_seq_len = self._mha2sha_optim.seq_len
            log_debug(
                "Updating MHA2SHAOptimizer 'seq_len' to new AR value in AR Builder"
            )
            # Have to update for AR Build and later splitting
            self._mha2sha_optim.seq_len = self._ar_value
            self._past_value_seq_len = None
            self._new_past_seq_len = None
            self._input_names_updated_from_ar = set()
            self.reshapes_not_to_update_for_ar = set()

    # Public API's
    def update_initial_ar_values(self) -> None:
        r"""Updates initals AR values - Inputs, Value Proto's, and Outputs.

        Initial updates to the model that are needed. Specifically, all inputs - expcept
        past key/value, value protos, and the outputs.

        Raises:
            ValueError:
                If the the original sequence length is unable to be found.
        """

        for vi_or_tp in itertools.chain(
            *[
                getattr(self._mha2sha_optim.model.graph, attr)
                for attr in ("input", "value_info", "output")
            ]
        ):
            # We update past key/value later on during split
            if "past" not in vi_or_tp.name:
                orig_shape = [
                    dim.dim_value for dim in vi_or_tp.type.tensor_type.shape.dim
                ]

                # Inputs -> typically [B, L]
                # Outptus -> typically [B, L, D]
                self._update_input_seq_len(
                    vi_or_tp,
                    seq_len_idx=-2 if len(orig_shape) >= 3 else -1,
                    new_value=self._ar_value,
                )

    def update_past_key_value_inputs(
        self,
        past_key_tp: TensorProto,
        past_value_tp: TensorProto,
        key_input_is_transposed: bool,
    ) -> None:
        r"""Update the past key and value inputs sequence lengths.

        Updates the past key and value inputs sequence lengths to the newly reflected
        on based on context length - new AR value.

        Args:
            past_key_tp:
                Past key TensorProto.
            past_value_tp:
                Past value TensorProto.
            key_input_is_transposed:
                Flag for if the key input is transposed.
        """

        if not self._past_value_seq_len:
            self._set_past_value_seq_len_from_name(past_value_tp.name)
        if not self._new_past_seq_len:
            self._compute_and_set_new_past_seq_len()

        past_inp_to_seq_len_idx = (
            (past_key_tp, -2 if key_input_is_transposed else -1),
            (past_value_tp, -2),
        )

        for past_tp, seq_len_idx in past_inp_to_seq_len_idx:
            if past_tp.name not in self._input_names_updated_from_ar:
                self._update_input_seq_len(
                    past_tp, seq_len_idx=seq_len_idx, new_value=self.new_past_seq_len
                )

    def update_reshapes(self) -> None:
        r"""Updates all Reshapes not in Attention Modules."""

        reshape_nodes = [
            node
            for node in self._mha2sha_optim.model.graph.node
            if node.op_type == "Reshape"
            and node.name not in self.reshapes_not_to_update_for_ar
        ]

        for reshape_node in reshape_nodes:
            # 1 is the shape constant input
            # 0 is the only attribute available which is 'value'
            const_shape_tensor_proto = (
                self._mha2sha_optim.get_node_by_output_name[reshape_node.input[1]]
                .attribute[0]
                .t
            )
            new_shape = list(numpy_helper.to_array(const_shape_tensor_proto))
            try:
                idx_to_change = new_shape.index(self._original_seq_len)
            except ValueError:
                log_debug(
                    f"Reshape node: {reshape_node.name} not updated, "
                    "no original sequence length found"
                )
                continue

            new_shape[idx_to_change] = self._ar_value

            const_shape_tensor_proto.CopyFrom(
                helper.make_tensor(
                    name=const_shape_tensor_proto.name,
                    data_type=const_shape_tensor_proto.data_type,
                    dims=[len(new_shape)],
                    vals=new_shape,
                )
            )

    # Public Read-Only Properties
    @property
    def buildable(self) -> bool:
        r"""Getter - flag if the Builder is buildable"""
        return self._buildable

    @buildable.setter
    def buildable(self, _: Any):
        r"""Setter - raises read-only error"""
        self._raise_read_only_error("buildable")

    @property
    def new_past_seq_len(self) -> int:
        r"""Getter - new sequence length"""
        return self._new_past_seq_len

    @new_past_seq_len.setter
    def new_past_seq_len(self, _: Any):
        r"""Setter - raises read-only error"""
        self._raise_read_only_error("new_past_seq_len")

    # Private
    def _raise_read_only_error(self, attr_name: str) -> None:
        r"""Common function to raise error for read-only properties.

        Args:
            attr_name:
                Attribute name to log to user.

        Raises:
            AttributeError:
                For the read-only property.
        """
        raise AttributeError(f"Cannot modify read-only property '{attr_name}'")

    def _set_past_value_seq_len_from_name(self, past_value_name: str) -> None:
        r"""Gets the past value sequence length.

        Sets the original past value sequence length value for calculating the
        context length.
        """

        past_value_tp = self._mha2sha_optim.model.graph.input[
            self._mha2sha_optim.mha_model_input_names_index_dict[past_value_name]
        ]

        self._past_value_seq_len = past_value_tp.type.tensor_type.shape.dim[
            -2
        ].dim_value

        log_debug(
            f"Set past value sequence length to {self._past_value_seq_len} "
            f"based on past value '{past_value_name}'"
        )

    def _compute_and_set_new_past_seq_len(self) -> None:
        r"""Computes the new sequence length.

        Computes the new sequence length of based on the AR value passed in.

        Returns:
            Newly computed sequence length.

        Raises:
            ValueError:
                If the provided AR value is not supported with the models context length.
        """

        context_len = self._original_seq_len + self._past_value_seq_len
        if self._ar_value > context_len - 1:
            raise ValueError(
                f"New AR value '{self._ar_value}' is not supported with the "
                f"context length '{context_len}' of this model. The supported AR values "
                f"for this model are AR >= 1 and AR <= {context_len - 1}"
            )

        self._new_past_seq_len = context_len - self._ar_value

        log_debug(
            f"New past sequence length set to {self._new_past_seq_len} "
            f"from context length {context_len}"
        )

    def _update_input_seq_len(
        self,
        vi_or_tp: Union[ValueInfoProto, TensorProto],
        seq_len_idx: int,
        new_value: int,
    ) -> None:
        r"""Updates the given input names sequence length.


        Args:
            vi_or_tp:
                Value Info or Tensor Proto to update.
            seq_len_idx:
                Index of the sequence length in the shape.
            new_value:
                New value to set for the sequence length.

        """
        log_debug(f"Updating input: '{vi_or_tp.name}' to new sequence length value")

        vi_or_tp.type.tensor_type.shape.dim[seq_len_idx].dim_value = new_value
        self._input_names_updated_from_ar.add(vi_or_tp.name)

        log_debug(f"Successfully updated input '{vi_or_tp.name}' to new AR.")
