# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from enum import Enum, auto
from typing import Optional, Dict, List, Union
from dataclasses import dataclass
import copy

from mha2sha.utils.logger import log_assert, log_debug, log_error, log_info, log_warning

class AimetEncodingVersion(str, Enum):
    V0_6_1 = "0.6.1"
    V1 = "1.0.0"

class AimetEncodingV1Types(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    PER_TENSOR = auto()
    PER_CHANNEL = auto()
    PER_BLOCK = auto()
    LPBQ = auto()
    VECTOR = auto()

_aimet_v1_type_to_multiples = {
    encoding_type: (
        ["scale", "offset", "per_block_int_scale"]
        if encoding_type == AimetEncodingV1Types.LPBQ
        else ["scale", "offset"]
    )
    for encoding_type in list(AimetEncodingV1Types)[1:]  # ignore PER_TENSOR
}

@dataclass
class NodeMappingDict:
    """
    Store information for single node for encoding mapping.
    mha_mapping_name_list: list mha tensor keys for mapping_name_dict: [mha_key_0, mha_key_1, ...]
    sha_mapping_name_list: list sha tensor keys for mapping_name_dict: [sha_key_0, sha_key_1, ...]
    mapping_name_dict: {
        mha_key_0: [mha tensor name],
        sha_key_0: [mha tensor name(s)],
        mha_key_1: [mha tensor name],
        sha_key_1: [mha tensor name(s)],
        ...
    }
    """
    mha_mapping_name_list: List[str]
    sha_mapping_name_list: List[str]
    mapping_name_dict: Dict[str, Optional[Union[str, List[str]]]]

    def __str__(self):
        # useful for printing and debuging
        def format_mapping_name_dict(mapping_name_dict):
            _str = ""
            for k, v in mapping_name_dict.items():
                _str += " "*8+f"{k}: {v}\n"
            return _str

        return f"NodeMappingDict(\n" \
               f"    mha_mapping_name_list: {self.mha_mapping_name_list},\n" \
               f"    sha_mapping_name_list: {self.sha_mapping_name_list},\n" \
               f"    mapping_name_dict: \n{format_mapping_name_dict(self.mapping_name_dict)},\n" \
               f")"

    def __repr__(self):
        return str(self)

def create_activation_node_mapping_dict(input_1_name=None, input_2_name=None, output_name=None):
    """
    Create activation encoding for NodeMappingDict.
    input_1_name: first input tensor name
    input_2_name: second input tensor name
    output_name: output tensor name
    """
    mha_mapping_name_list = []
    sha_mapping_name_list = []
    mapping_name_dict = {}
    if input_1_name is not None:
        mha_mapping_name_list.append("mha_input_1_activation_name")
        sha_mapping_name_list.append("sha_input_1_activation_name")
        mapping_name_dict["mha_input_1_activation_name"] = input_1_name
        mapping_name_dict["sha_input_1_activation_name"] = None

    if input_2_name is not None:
        mha_mapping_name_list.append("mha_input_2_activation_name")
        sha_mapping_name_list.append("sha_input_2_activation_name")
        mapping_name_dict["mha_input_2_activation_name"] = input_2_name
        mapping_name_dict["sha_input_2_activation_name"] = None

    if output_name is not None:
        mha_mapping_name_list.append("mha_output_activation_name")
        sha_mapping_name_list.append("sha_output_activation_name")
        mapping_name_dict["mha_output_activation_name"] = output_name
        mapping_name_dict["sha_output_activation_name"] = None

    return NodeMappingDict(
                mha_mapping_name_list = mha_mapping_name_list,
                sha_mapping_name_list = sha_mapping_name_list,
                mapping_name_dict = mapping_name_dict
            )

def update_sha_tensor_to_node_mapping_dict(node_mapping_dict, sha_node_list):
    """ Update info from nodes in sha_node_list to node_mapping_dict """
    if "sha_input_1_activation_name" in node_mapping_dict.sha_mapping_name_list:
        node_mapping_dict.mapping_name_dict["sha_input_1_activation_name"] = [node.input[0] for node in sha_node_list]

    if "sha_input_2_activation_name" in node_mapping_dict.sha_mapping_name_list:
        node_mapping_dict.mapping_name_dict["sha_input_2_activation_name"] = [node.input[1] for node in sha_node_list]

    if "sha_output_activation_name" in node_mapping_dict.sha_mapping_name_list:
        node_mapping_dict.mapping_name_dict["sha_output_activation_name"] = [node.output[0] for node in sha_node_list]

    if "sha_param_name" in node_mapping_dict.sha_mapping_name_list:
        node_mapping_dict.mapping_name_dict["sha_param_name"] = [node.input[1] for node in sha_node_list]

def update_multi_input_concat_sha_tensor_to_node_mapping_dict(node_mapping_dict, sha_node):
    """
    Update info from nodes in sha_node_list to node_mapping_dict.
    While keep all the concat inputs at "sha_input_1_activation_name"
    sha_node: a concat node with #head of inputs
    """

    if "sha_input_1_activation_name" in node_mapping_dict.sha_mapping_name_list:
        node_mapping_dict.mapping_name_dict["sha_input_1_activation_name"] = [_input for _input in sha_node.input]

    if "sha_output_activation_name" in node_mapping_dict.sha_mapping_name_list:
        node_mapping_dict.mapping_name_dict["sha_output_activation_name"] = [sha_node.output[0]]

def handle_aimet_v1_activ_encodings(encoding, tensor_name):
    log_assert(
        AimetEncodingV1Types(encoding["enc_type"]) == AimetEncodingV1Types.PER_TENSOR,
        f"Mapping of activation encodings is only supported for PER_TENSOR. Got: {encoding['enc_type']}"
    )
    # only dictify v1 encoding to make a deep copy.
    encoding = dict(encoding)
    # update name internal to encoding
    encoding["name"] = tensor_name

    return encoding

def update_encodings_to_v1_format(encodings, encodings_info) -> dict:
    """
    Updates encodings to AIMET v1 format
    Args:
        encodings: activation and param encodings.
        encodings_info: Other info for encoding files.
    """
    log_debug("Updating encodings from internal v0.6.1 format to v1.0.0")
    for attr in ["param", "activation"]:
        encodings[f"{attr}_encodings"] = list(encodings[f"{attr}_encodings"].values())

    return encodings | encodings_info

def copy_mha_activation_encodings_to_sha(mha_activation_encoding,
                                         sha_activation_encoding,
                                         start_node_name,
                                         qkv_node_name,
                                         head_num,
                                         node_mapping_dict,
                                         _mha_to_sha_encodings_names_to_dump,
                                         model_input_output_list,
                                         handle_potential_concat,
                                         encoding_version):

    mha_mapping_name_list = node_mapping_dict.mha_mapping_name_list
    sha_mapping_name_list = node_mapping_dict.sha_mapping_name_list
    for _mha_acitvation_name, _sha_acitvation_name in zip(mha_mapping_name_list, sha_mapping_name_list):
        mha_encoding_key = node_mapping_dict.mapping_name_dict[_mha_acitvation_name]
        sha_encoding_key_list = node_mapping_dict.mapping_name_dict[_sha_acitvation_name]

        # Allow one mha tensor map to multiple sha tenser for activation encoding.
        # E.g. mha qkv matmul tensor can map to sha qkv matmul and head concat in/out
        if mha_encoding_key not in _mha_to_sha_encodings_names_to_dump["activation_encodings"]:
            _mha_to_sha_encodings_names_to_dump["activation_encodings"][mha_encoding_key] = sha_encoding_key_list
        else:
            for _sha_enc_key in sha_encoding_key_list:
                if _sha_enc_key not in _mha_to_sha_encodings_names_to_dump["activation_encodings"][mha_encoding_key]:
                    _mha_to_sha_encodings_names_to_dump["activation_encodings"][mha_encoding_key].extend(sha_encoding_key_list)

        if mha_encoding_key in mha_activation_encoding.keys():
            encoding = mha_activation_encoding[mha_encoding_key]
            if handle_potential_concat and _sha_acitvation_name == "sha_output_activation_name":
                assert len(sha_encoding_key_list) == 1
            else:
                assert head_num == len(sha_encoding_key_list), f"sha encoding for mha node: {qkv_node_name}, {mha_encoding_key}, sha activation key num: {len(sha_encoding_key_list)} is different to head_num: {head_num} "

            for _, _sha_key in enumerate(sha_encoding_key_list):
                if encoding_version == AimetEncodingVersion.V1:
                    encoding = handle_aimet_v1_activ_encodings(encoding, _sha_key)

                sha_activation_encoding[_sha_key] = encoding

            try:
                # Don't delete mha_encoding_key from sha encoding when sha and mha has same tensor name.
                # Don't delete mha_encoding_key from sha encoding when it is a model input.
                if not any([ _sha_key == mha_encoding_key for _sha_key in sha_encoding_key_list]) and (mha_encoding_key not in model_input_output_list):
                    del sha_activation_encoding[mha_encoding_key]
            except:
                # mha_encoding_key is both input and output encoding that is need to be recorded.
                # This cause deplicate delete activation encoding in sha_encodings (which is deep copy of mha)
                assert mha_encoding_key in [node_mapping_dict.mapping_name_dict[_mha_acitvation_name] for _mha_acitvation_name in mha_mapping_name_list]
        else:
            log_debug(f"mha activation encoding for attention within {start_node_name}, node: {qkv_node_name} key: {mha_encoding_key} not exist in mha encoding file")


def get_encoding_slice(i, slice_len):
    return slice(i*slice_len, (i+1)*slice_len)

def handle_aimet_v1_param_encodings(encoding, _sha_key, i, head_num):
    v1_encoding = dict(encoding)  # copy of encoding to not have reference
    v1_encoding["name"] = _sha_key  # name needs to be the same as sha name
    for key in _aimet_v1_type_to_multiples[AimetEncodingV1Types(encoding["enc_type"])]:
        # TD;LR, we have to be dynamic with slicing depending on the
        # key in the encoding.
        # -------------------------
        # 'slice_len' is different per mutli, - per_block_int_scale is longer than other
        # values like scale and offset. For example, if we have head
        # num as 32 and we have 4096 d_model, then scale and offset
        # will have 4096 / 32 -> 128. However, per_block_int_scale
        # wil have 262144 / 32 -> 8192.
        slice_len = len(encoding[key]) // head_num
        v1_encoding[key] = encoding[key][get_encoding_slice(i, slice_len)]
    return v1_encoding

def copy_mha_param_encodings_to_sha(mha_param_encoding,
                                    sha_param_encoding,
                                    _start_node,
                                    qkv_node_name,
                                    head_num,
                                    node_mapping_dict,
                                    _mha_to_sha_encodings_names_to_dump,
                                    encoding_version):
    """ Copy mha param encodings to sha encodings """


    mha_mapping_name_list = node_mapping_dict.mha_mapping_name_list
    sha_mapping_name_list = node_mapping_dict.sha_mapping_name_list

    for _mha_param_name, _sha_param_name in zip(mha_mapping_name_list, sha_mapping_name_list):
        mha_encoding_key = node_mapping_dict.mapping_name_dict[_mha_param_name]
        sha_encoding_key_list = node_mapping_dict.mapping_name_dict[_sha_param_name]
        _mha_to_sha_encodings_names_to_dump["param_encodings"][mha_encoding_key] = sha_encoding_key_list
        if mha_encoding_key in mha_param_encoding.keys():
            encoding = mha_param_encoding[mha_encoding_key]
            is_v1_aimet_encodings = encoding_version == AimetEncodingVersion.V1

            # PTQ support
            is_ptq_encoding = (
                AimetEncodingV1Types(encoding["enc_type"]) == AimetEncodingV1Types.PER_TENSOR
                if is_v1_aimet_encodings
                else len(encoding) == 1
            )

            if is_ptq_encoding:
                log_warning(f"Got PTQ param encoding on {mha_encoding_key}, applying same param encoding to all these sha weights: {sha_encoding_key_list}.")

            assert head_num == len(sha_encoding_key_list), f"sha param key num: {len(sha_encoding_key_list)} is different to head_num {head_num} "
            for i, _sha_key in enumerate(sha_encoding_key_list):
                if is_v1_aimet_encodings:
                    if is_ptq_encoding:
                        sha_param_encoding[_sha_key] = handle_aimet_v1_activ_encodings(encoding, _sha_key)
                    else:
                        sha_param_encoding[_sha_key] = handle_aimet_v1_param_encodings(encoding, _sha_key, i, head_num)
                else:  # aimet v0.6.1 encodings
                    slice_len = len(encoding) // head_num
                    sha_param_encoding[_sha_key] = encoding[get_encoding_slice(i, slice_len)] if not is_ptq_encoding else encoding

            # Don't delete mha_encoding_key from sha encoding when sha and mha has same tensor name.
            if not any([ _sha == mha_encoding_key for _sha in sha_encoding_key_list]):
                del sha_param_encoding[mha_encoding_key]
        else:
            log_warning(f"mha parameter encoding for attention within {_start_node}, node: {qkv_node_name} key: {mha_encoding_key} not exist in mha encoding file")


def create_tensor_name_to_encodings(encodings):
    """Creates a two dict between tensor names and activation/param encodings"""
    return {
        f"{attr}_encodings": {
            _encoding["name"]: _encoding
            for _encoding in encodings[f"{attr}_encodings"]
        }
        for attr in ["activation", "param"]
    }


