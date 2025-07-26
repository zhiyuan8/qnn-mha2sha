# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import copy
import json
from onnx import numpy_helper
import os
from safetensors import safe_open
from safetensors.numpy import load_file, save_file
import yaml

from mha2sha.utils.logger import (
    log_info,
    log_debug,
    log_warning,
    log_assert
)

from mha2sha.utils.encoding_mapper_utils import (
    AimetEncodingVersion,
    AimetEncodingV1Types,
    _aimet_v1_type_to_multiples,
    handle_aimet_v1_param_encodings,
    handle_aimet_v1_activ_encodings,
    update_encodings_to_v1_format,
    create_tensor_name_to_encodings,
)

LORA_SHA_SAFETENSOR_PATH = "sha_weights.safetensor"

def save_init_to_safetensor(tensor_name_to_init_dict, sha_save_path):
    safe_tensor_dict = {}
    for name, init in tensor_name_to_init_dict.items():
        tensor = numpy_helper.to_array(init).copy()
        safe_tensor_dict[name] = tensor

    if os.path.exists(sha_save_path/LORA_SHA_SAFETENSOR_PATH):
        os.remove(sha_save_path/LORA_SHA_SAFETENSOR_PATH)
    log_info(f"save lora weights to safetensors at {sha_save_path/LORA_SHA_SAFETENSOR_PATH}")
    save_file(safe_tensor_dict, sha_save_path/LORA_SHA_SAFETENSOR_PATH)

    tensors = {}
    with safe_open(sha_save_path/LORA_SHA_SAFETENSOR_PATH, framework="numpy", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    assert tensors.keys() == safe_tensor_dict.keys()

def parse_lora_adaptor_list(lora_adaptor_list_path):
    """ Parse lora_adaptor_list_path.yaml  """
    with open(lora_adaptor_list_path, 'r') as file:
        lora_adaptor_list = yaml.safe_load(file)['lora_adaptor_list']
        lora_adaptor_list = [LoraAdaptor(lora_adaptor_dict) for lora_adaptor_dict in lora_adaptor_list]

    return lora_adaptor_list

class LoraAdaptor:
    """
    Args:
        mha_to_sha_encodings_names: dumped mha_to_sha_encodings_names
        mha_conv: mha_conv model
        base_sha_lora_keys: lora safetensor keys()
        base_sha_encoding: sha encodings from base model
    """
    mha_to_sha_encodings_names = None
    mha_conv = False
    base_sha_lora_keys = None
    base_sha_encoding = None

    def __init__(self, lora_adaptor_dict) -> None:
        self.name = lora_adaptor_dict['name']
        self.safetensor_path = lora_adaptor_dict['safetensor_path']
        self.encodings_path = lora_adaptor_dict['encodings_path']
        self._safetensor = None
        self._encodings = None
        assert os.path.exists(self.safetensor_path), f"Lora adaptor safetensor path {self.safetensor_path} not exist."
        assert os.path.exists(self.encodings_path), f"Lora adaptor encoding path {self.encodings_path} not exist."

    @property
    def safetensor(self):
        if self._safetensor is None:
            self._safetensor = load_file(self.safetensor_path)
        return self._safetensor

    @property
    def encodings(self):
        if self._encodings is None:
            with open(self.encodings_path, "r") as f:
                self._encodings = json.load(f)
        return self._encodings

    def _check_adaptor_weights(self, lora_adaptor_sha_tensor_dict):
        """ Check lora_adaptor_sha_tensor_dict's key are the same as base_sha_lora_keys's tensor.  """
        lora_adaptor_keys = lora_adaptor_sha_tensor_dict.keys()

        if sha_adaptor_diff := (LoraAdaptor.base_sha_lora_keys - lora_adaptor_keys):
            log_warning(f"{list(sha_adaptor_diff)}: exits in sha_weight.safetensor but not found in {self.name}_{LORA_SHA_SAFETENSOR_PATH}.")

    def _check_adaptor_encodings(self, lora_adaptor_sha_encodings, base_sha_encoding):
        """ check base model sha tensor names and """
        for activ_param_encoding in ("activation_encodings", "param_encodings"):
            assert (
                base_sha_encoding[activ_param_encoding].keys()
                == lora_adaptor_sha_encodings[activ_param_encoding].keys()
            ), f"{activ_param_encoding} base model sha and lora adpator sha encoding keys not match.\n\
sha - lora_adaptor_sha:\n    {base_sha_encoding[activ_param_encoding].keys() - lora_adaptor_sha_encodings[activ_param_encoding].keys()}\n\
lora_adaptor_sha - sha:\n    {lora_adaptor_sha_encodings[activ_param_encoding].keys() - base_sha_encoding[activ_param_encoding].keys()}"

            for tensor in base_sha_encoding[activ_param_encoding].keys():
                base_model_sha_channel = len(base_sha_encoding[activ_param_encoding][tensor])
                adaptor_sha_channel = len(lora_adaptor_sha_encodings[activ_param_encoding][tensor])
                if base_model_sha_channel != base_model_sha_channel:
                    if activ_param_encoding == "param_encodings" and (adaptor_sha_channel == 1 or base_model_sha_channel==1):
                        # log warning because it can be base model is in PCQ but lora adaptor in PTQ.
                        log_warning(f"[{activ_param_encoding}][{tensor}]: base model sha and lora adpator sha encoding len not match, got {base_model_sha_channel} and {adaptor_sha_channel}")
                    else:
                        log_assert(f"[{activ_param_encoding}][{tensor}]: base model sha and lora adpator sha encoding len not match, got {base_model_sha_channel} and {adaptor_sha_channel}")

    def map_encoding_and_slice_safetensor(self):
        """ Return sliced sha tensor dict and new encodings.  """
        assert LoraAdaptor.mha_to_sha_encodings_names is not None, "LoraAdaptor.mha_to_sha_encodings_names is None"
        is_v1_aimet_encodings = (AimetEncodingVersion(self.encodings["version"]) == AimetEncodingVersion.V1)
        log_info(f"Converting lora adaptor: {self.name}. Safetensor path: {self.safetensor_path}\n, encoding path: {self.encodings_path}\, encoding version: {AimetEncodingVersion(self.encodings['version'])}")

        # Handle weight.safetensor
        lora_adaptor_sha_tensor_dict = {}
        excluded_lora_adaptor_names = []
        for lora_adaptor_weight_name, lora_adaptor_tensor in self.safetensor.items():
            # Slice and map mha weight to sha lora weright in mha_to_sha_encodings_names
            if lora_adaptor_weight_name in LoraAdaptor.mha_to_sha_encodings_names["param_encodings"]:
                sha_weight_name_list = LoraAdaptor.mha_to_sha_encodings_names["param_encodings"][lora_adaptor_weight_name]
                head_num = len(sha_weight_name_list)
                head_dim = lora_adaptor_tensor.shape[0]//head_num

                # Slice by sha heads
                for head, sha_weight_names in enumerate(sha_weight_name_list):
                    if LoraAdaptor.mha_conv:
                        lora_adaptor_sha_tensor_dict[sha_weight_names] = lora_adaptor_tensor[head_dim*head: head_dim*(head+1), ...] # [O, I, Kh, Kw]
                    else:
                        lora_adaptor_sha_tensor_dict[sha_weight_names] = lora_adaptor_tensor[:, head_dim*head: head_dim*(head+1)] # [I, O]
            # Handle mlp weights as is
            elif lora_adaptor_weight_name in LoraAdaptor.base_sha_lora_keys:
                lora_adaptor_sha_tensor_dict[lora_adaptor_weight_name] = lora_adaptor_tensor
            else:
                excluded_lora_adaptor_names.append(lora_adaptor_weight_name)

        if excluded_lora_adaptor_names:
            log_warning(f"{excluded_lora_adaptor_names}:weights from {self.safetensor_path} not found in {LORA_SHA_SAFETENSOR_PATH}")

        # Slice encodings and map to sha tensors
        not_updated_encodings_list = []
        mha_encodings = self.encodings
        base_sha_encoding = LoraAdaptor.base_sha_encoding

        if is_v1_aimet_encodings:
            _key_values_not_mapped = {
                key: mha_encodings[key]
                for key in mha_encodings.keys()
                if key not in ["param_encodings", "activation_encodings"]
            }
            mha_encodings = create_tensor_name_to_encodings(mha_encodings)
            # use tensor_name_to_encodings for v1 encodings
            base_sha_encoding = create_tensor_name_to_encodings(self.base_sha_encoding)

        sha_encodings = copy.deepcopy(mha_encodings)

        # mha_to_sha_encodings_names
        for activ_param_encoding in ("activation_encodings", "param_encodings"):
            for lora_adaptor_mha_name, lora_adaptor_mha_encoding in mha_encodings[activ_param_encoding].items():

                # Update tensor encodings in mha
                if lora_adaptor_mha_name in LoraAdaptor.mha_to_sha_encodings_names[activ_param_encoding]:
                    sha_encoding_name_list = LoraAdaptor.mha_to_sha_encodings_names[activ_param_encoding][lora_adaptor_mha_name]
                    head_num = len(sha_encoding_name_list)
                    head_dim = len(lora_adaptor_mha_encoding)//head_num
                    ptq_or_activation = (
                            AimetEncodingV1Types(lora_adaptor_mha_encoding["enc_type"]) == AimetEncodingV1Types.PER_TENSOR
                            if is_v1_aimet_encodings
                            else len(lora_adaptor_mha_encoding) == 1
                        )

                    if lora_adaptor_mha_name not in base_sha_encoding[activ_param_encoding]:
                        del sha_encodings[activ_param_encoding][lora_adaptor_mha_name]

                    for head, sha_name in enumerate(sha_encoding_name_list):
                        if is_v1_aimet_encodings:
                            if activ_param_encoding == "param_encodings":
                                if ptq_or_activation:
                                    sliced_mha_encoding = handle_aimet_v1_activ_encodings(lora_adaptor_mha_encoding, sha_name)
                                else:
                                    sliced_mha_encoding = handle_aimet_v1_param_encodings(lora_adaptor_mha_encoding, sha_name, head, head_num)
                            else:
                                sliced_mha_encoding = handle_aimet_v1_activ_encodings(lora_adaptor_mha_encoding, sha_name)

                        else:
                            sliced_mha_encoding = lora_adaptor_mha_encoding[head*head_dim: (head+1)*head_dim] if not ptq_or_activation else lora_adaptor_mha_encoding

                        sha_encodings[activ_param_encoding][sha_name] = sliced_mha_encoding

                # Update mlp tensor encodings from adaptor to copied base model sha
                elif lora_adaptor_mha_name in LoraAdaptor.base_sha_lora_keys:
                    log_debug(f"Update MLP encodings: {lora_adaptor_mha_name}")
                    sha_encodings[activ_param_encoding][lora_adaptor_mha_name] = lora_adaptor_mha_encoding
                else:
                    not_updated_encodings_list.append(lora_adaptor_mha_name)

        log_debug(f"{not_updated_encodings_list}: Encodings from {self.encodings_path} are not lora-related encodings. Not update.")

        # remove extra encodings from lora adaptor sha
        for activ_param_encoding in ("activation_encodings", "param_encodings"):
            base_sha_enc_keys = set(base_sha_encoding[activ_param_encoding].keys())
            lora_sha_enc_keys = set(sha_encodings[activ_param_encoding].keys())
            lora_minus_base_sha_sha_enc = lora_sha_enc_keys - base_sha_enc_keys
            log_debug(f"Remove lora adaptor extra encodings: {list(lora_minus_base_sha_sha_enc)}")
            for lora_adaptor_sha_activ_key in lora_minus_base_sha_sha_enc:
                del sha_encodings[activ_param_encoding][lora_adaptor_sha_activ_key]

        # Check weight and encodings
        self._check_adaptor_weights(lora_adaptor_sha_tensor_dict)
        self._check_adaptor_encodings(sha_encodings, base_sha_encoding)

        if AimetEncodingVersion(self.encodings["version"]) == AimetEncodingVersion.V1:
            sha_encodings = update_encodings_to_v1_format(sha_encodings, _key_values_not_mapped)

        return lora_adaptor_sha_tensor_dict, sha_encodings
