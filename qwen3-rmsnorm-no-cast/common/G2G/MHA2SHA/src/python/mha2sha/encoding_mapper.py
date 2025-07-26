# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import copy
import json
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass


from mha2sha.utils.base_attn_encoding_mapper import (
    create_base_attn_mapping_dict,
    ACTIVATION_ENCODING_KEYS,
    PARAM_ENCODING_KEYS,
)
from mha2sha.optimizer_extension.lora_extension import (
    create_lora_encoding_mapping_dict,
    LORA_ACTIVATION_ENCODING_KEYS,
    LORA_V1_ACTIVATION_ENCODING_KEYS,
    LORA_PARAM_ENCODING_KEYS,
    LoraVersion,
)
from mha2sha.optimizer_extension.rope_extension import (
    create_rope_encoding_mapping_dict,
    ROPE_ACTIVATION_ENCODING_KEYS,
)
from mha2sha.optimizer_extension.past_key_value_extension import (
    create_past_key_value_encoding_mapping_dict,
    PAST_KEY_VALUE_ACTIVATION_ENCODING_KEYS,
)
from mha2sha.optimizer_extension.rmsnorm_extension import (
    create_rmsnorm_encoding_mapping_dict,
    RMSNORM_ACTIVATION_ENCODING_KEYS,
    RMSNORM_PARAM_ENCODING_KEYS,
)
from mha2sha.optimizer_extension.anchor_network_extension import (
    create_anchor_network_encoding_mapping_dict,
    ANCHOR_ACTIVATION_ENCODING_KEYS,
)

from mha2sha.utils.encoding_mapper_utils import (
    NodeMappingDict,
    copy_mha_activation_encodings_to_sha,
    copy_mha_param_encodings_to_sha,
    AimetEncodingVersion,
    create_tensor_name_to_encodings,
    update_encodings_to_v1_format,
    handle_aimet_v1_activ_encodings,
)
from mha2sha.utils.utils import ExtensionTypes, BranchType
from mha2sha.utils.logger import log_debug, log_info, log_warning

MHA_TO_SHA_NAME_MAPPING_PATH = "mha_to_sha_encodings_names.json"


@dataclass
class EncodingMappingDict:
    """ Necessary information for mha to sha encoding mapping for each attention. """
    head_num: Optional[int] = None
    head_dim: Optional[int] = None
    base_attn: Optional[Dict[str, NodeMappingDict]] = None
    rope: Optional[Dict[str, NodeMappingDict]] = None
    past_key_value: Optional[Dict[str, NodeMappingDict]] = None
    lora: Optional[Dict[str, NodeMappingDict]] = None
    rmsnorm: Optional[Dict[str, NodeMappingDict]] = None
    anchor_network: Optional[Dict[str, NodeMappingDict]] = None

def get_encoding_mapping_dict(info_dict, head_dim):
    """Get mha to sha encoding mapping dict for each attention layer."""

    # Provide default values or handling logic
    default_value = None  # Or other appropriate default values

    # Safely get values from the dictionary, using default_value if the key does not exist
    num_heads = info_dict.get("num_heads", default_value)
    mha_base_attn_node = info_dict.get("mha_base_attn_node", default_value)
    q_rope_mha_node = info_dict.get("q_rope_mha_node", default_value)
    k_rope_mha_node = info_dict.get("k_rope_mha_node", default_value)
    past_key_value_concat = info_dict.get("past_key_value_concat", default_value)
    q_lora_mha_node = info_dict.get("q_lora_mha_node", default_value)
    k_lora_mha_node = info_dict.get("k_lora_mha_node", default_value)
    v_lora_mha_node = info_dict.get("v_lora_mha_node", default_value)
    query_mha_rmsnorm_node = info_dict.get("query", {}).get('mha_rmsnorm_node', default_value)
    key_mha_rmsnorm_node = info_dict.get("key", {}).get('mha_rmsnorm_node', default_value)
    mha_anchor_node = None # anchor network mha node info collected when attaching sha to model.

    return EncodingMappingDict(
        head_num=num_heads,
        head_dim=head_dim,
        base_attn=create_base_attn_mapping_dict(mha_base_attn_node),
        rope=create_rope_encoding_mapping_dict(
            [q_rope_mha_node, k_rope_mha_node]
        ),
        past_key_value=create_past_key_value_encoding_mapping_dict(
            past_key_value_concat,
            info_dict=info_dict
        ),
        lora=create_lora_encoding_mapping_dict(
            [
                q_lora_mha_node,
                k_lora_mha_node,
                v_lora_mha_node,
            ]
        ),
        rmsnorm=create_rmsnorm_encoding_mapping_dict(
            [
                query_mha_rmsnorm_node,
                key_mha_rmsnorm_node,
            ]
        ),
        anchor_network=None,
    )

class MHA2SHAEncodingMapper:
    """ Encoding mapper object to map mha to sha encodings.  """
    def __init__(self, mha2sha_optim) -> None:
        """Initalizes an instance based on the MHA2SHA optimizer provided.

        Args:
            mha2sha_optim:
                MHA2SHAOptimizer instance holding the model loader and model info.
        """
        self.mha2sha_optim = mha2sha_optim
        self.model_input_output_names = mha2sha_optim.mha_model_input_names + mha2sha_optim.mha_model_output_names

    def map_encodings(self, mha_encodings, sha_export_dir: Path):
        self.mha_encodings = mha_encodings
        self._encoding_version = AimetEncodingVersion(self.mha_encodings["version"])
        log_info(f"Found AIMET encoding version: {self._encoding_version}")
        if self._encoding_version == AimetEncodingVersion.V1:
            log_debug("Overriding 'self.mha_encodings' for AIMET v1.0.0 encodings")
            self._key_values_not_mapped = {
                key: mha_encodings[key]
                for key in mha_encodings.keys()
                if key not in ["param_encodings", "activation_encodings"]
            }
            mha_encodings = create_tensor_name_to_encodings(self.mha_encodings)
            self.mha_encodings = mha_encodings

        sha_encodings = copy.deepcopy(mha_encodings)
        self._mha_to_sha_encodings_names = {"activation_encodings": {},
                                            "param_encodings": {}}
        self._multi_input_node_names = [
            "past_key_concat_out",
            "past_value_concat_out",
            "qkv_head_concat"
        ]

        # Update activation and param encodings
        self._map_activation_encodings(mha_encodings,
                                       sha_encodings,
                                       ExtensionTypes.BASE_ATTN,
                                       ACTIVATION_ENCODING_KEYS)

        self._map_param_encodings(mha_encodings,
                                  sha_encodings,
                                  ExtensionTypes.BASE_ATTN,
                                  PARAM_ENCODING_KEYS)
        # Update internal rmsnorm
        if self.mha2sha_optim.handle_internal_rmsnorm:
            self._map_activation_encodings(mha_encodings,
                                           sha_encodings,
                                           ExtensionTypes.RMSNORM,
                                           RMSNORM_ACTIVATION_ENCODING_KEYS)

            self._map_param_encodings(mha_encodings,
                                  sha_encodings,
                                  ExtensionTypes.RMSNORM,
                                  RMSNORM_PARAM_ENCODING_KEYS)

        # Update rope activation encodings
        if self.mha2sha_optim.handle_rope_ops and self.mha2sha_optim._rope_extension.map_rope_encoding:
            self._map_activation_encodings(mha_encodings,
                                           sha_encodings,
                                           ExtensionTypes.ROPE,
                                           ROPE_ACTIVATION_ENCODING_KEYS)

        # Update past key value activation encodings
        if self.mha2sha_optim.handle_past_key_value:
            self._map_activation_encodings(
                mha_encodings,
                sha_encodings,
                ExtensionTypes.PAST_KEY_VALUE,
                PAST_KEY_VALUE_ACTIVATION_ENCODING_KEYS
            )
        # Update lora activation encodings and param encodings
        if self.mha2sha_optim.lora_model:
            if self.mha2sha_optim.lora_version == LoraVersion.V1:
                self._map_activation_encodings(mha_encodings,
                                               sha_encodings,
                                               ExtensionTypes.LORA,
                                               LORA_V1_ACTIVATION_ENCODING_KEYS)
            else:
                self._map_activation_encodings(mha_encodings,
                                               sha_encodings,
                                               ExtensionTypes.LORA,
                                               LORA_ACTIVATION_ENCODING_KEYS)
                self._map_param_encodings(mha_encodings,
                                          sha_encodings,
                                          ExtensionTypes.LORA,
                                          LORA_PARAM_ENCODING_KEYS)

        # Update lora activation encodings and param encodings
        if self.mha2sha_optim.anchor_network:
            self._map_activation_encodings(mha_encodings,
                                           sha_encodings,
                                           ExtensionTypes.ANCHOR_NETWORK,
                                           ANCHOR_ACTIVATION_ENCODING_KEYS)

        # Update mapping dict and run sanity check
        self.mha2sha_optim._update_all_mapping_dicts()
        self._check_sha_encoding(sha_encodings)

        if self._encoding_version == AimetEncodingVersion.V1:
            sha_encodings = update_encodings_to_v1_format(sha_encodings, self._key_values_not_mapped)
            mha_encodings = update_encodings_to_v1_format(mha_encodings, self._key_values_not_mapped)

        self._save_mha_to_sha_encodings_names(sha_export_dir)

        return sha_encodings

    def _map_activation_encodings(self,
                                 mha_encodings,
                                 sha_encodings,
                                 extension_type,
                                 activation_encoding_key):
        """ Copy mha extension activation encoding to sha activation encodings to all heads. """
        mha_activation_encoding = mha_encodings["activation_encodings"]
        sha_activation_encoding = sha_encodings["activation_encodings"]
        extension_branch_prefix = [extension_type+"_"+qkv for qkv in [b_type.name.lower() for b_type in BranchType]]

        # Iterate mha param encodings and update sha encodings
        for start_node_name, encoding_mapping_dict in self.mha2sha_optim.mha_sha_encoding_mapping_dict.items():
            ext_node_mapping_dict = getattr(encoding_mapping_dict, extension_type)

            match extension_type:
                case ExtensionTypes.BASE_ATTN | ExtensionTypes.PAST_KEY_VALUE | ExtensionTypes.ANCHOR_NETWORK:
                    is_past_key_ext = extension_type == ExtensionTypes.PAST_KEY_VALUE
                    for ext_node_name in activation_encoding_key:
                        # Some past key/value models might only have
                        # past key/value out
                        if is_past_key_ext and ext_node_name not in ext_node_mapping_dict:
                            continue

                        if (
                            # For past key/value only
                            self.mha2sha_optim.gqa_model and is_past_key_ext
                        ) or (
                            # For all others
                            self.mha2sha_optim.gqa_model
                            and ext_node_name[0] in ("k", "v")
                            and extension_type == ExtensionTypes.BASE_ATTN
                        ) or (
                            self.mha2sha_optim.gqa_model
                            and self.mha2sha_optim.anchor_network
                            and extension_type == ExtensionTypes.ANCHOR_NETWORK
                        ):
                            head_num = self.mha2sha_optim._gqa_extension.kv_head_num
                        else:
                            head_num = encoding_mapping_dict.head_num

                        node_mapping_dict = ext_node_mapping_dict[ext_node_name]

                        if node_mapping_dict:
                            copy_mha_activation_encodings_to_sha(
                                mha_activation_encoding,
                                sha_activation_encoding,
                                start_node_name,
                                ext_node_name,
                                head_num,
                                node_mapping_dict,
                                self._mha_to_sha_encodings_names,
                                self.model_input_output_names,
                                ext_node_name in self._multi_input_node_names,
                                self._encoding_version
                            )

                case ExtensionTypes.LORA:
                    # Iterate Q, K, V branch for extension optimizer
                    for branch_prefix in extension_branch_prefix:
                        head_num = self.mha2sha_optim._gqa_extension.kv_head_num if \
                                   self.mha2sha_optim.gqa_model and branch_prefix[-1] in ("k", "v") else \
                                   encoding_mapping_dict.head_num


                        for ext_node_name in activation_encoding_key:
                            for lora_adaptor_set_num in self.mha2sha_optim.lora_adaptor_set_info_dict.keys():
                                dict_name = "_".join([branch_prefix, ext_node_name, str(lora_adaptor_set_num)])

                                if dict_name in ext_node_mapping_dict:
                                    node_mapping_dict = ext_node_mapping_dict[dict_name]
                                else:
                                    node_mapping_dict = None

                                if node_mapping_dict:
                                    copy_mha_activation_encodings_to_sha(
                                        mha_activation_encoding,
                                        sha_activation_encoding,
                                        start_node_name,
                                        ext_node_name,
                                        head_num,
                                        node_mapping_dict,
                                        self._mha_to_sha_encodings_names,
                                        self.model_input_output_names,
                                        ext_node_name in self._multi_input_node_names,
                                        self._encoding_version
                                    )

                case _:
                    # Iterate Q, K, V branch for extension optimizer
                    for branch_prefix in extension_branch_prefix:
                        head_num = self.mha2sha_optim._gqa_extension.kv_head_num if \
                                   self.mha2sha_optim.gqa_model and branch_prefix[-1] in ("k", "v") else \
                                   encoding_mapping_dict.head_num

                        for ext_node_name in activation_encoding_key:
                            if branch_prefix+"_"+ext_node_name in ext_node_mapping_dict:
                                node_mapping_dict = ext_node_mapping_dict[branch_prefix+"_"+ext_node_name]
                            else:
                                node_mapping_dict = None

                            if node_mapping_dict:
                                copy_mha_activation_encodings_to_sha(
                                    mha_activation_encoding,
                                    sha_activation_encoding,
                                    start_node_name,
                                    ext_node_name,
                                    head_num,
                                    node_mapping_dict,
                                    self._mha_to_sha_encodings_names,
                                    self.model_input_output_names,
                                    ext_node_name in self._multi_input_node_names,
                                    self._encoding_version
                                )

    def _map_param_encodings(self,
                             mha_encodings,
                             sha_encodings,
                             extension_type,
                             param_encoding_keys):
        """
        Slice mha linear PCQ param encoding for each head and copy to sha linear PCQ encodings.
        :param mha_encodings: mha encodings.
        :param sha_encoding: sha encodings to be dump.
        """
        mha_param_encoding = mha_encodings["param_encodings"]
        sha_param_encoding = sha_encodings["param_encodings"]
        extension_branch_prefix = [extension_type+"_"+qkv for qkv in [b_type.name.lower() for b_type in BranchType]]

        # Iterate mha param encodings and update sha encodings
        for _start_node, attn_encoding_mapping_dict in self.mha2sha_optim.mha_sha_encoding_mapping_dict.items():
            head_dim = attn_encoding_mapping_dict.head_dim
            encoding_mapping_dict = getattr(attn_encoding_mapping_dict, extension_type)

            if extension_type == ExtensionTypes.BASE_ATTN:
                # Iterate each node in encoding_mapping_dict
                for qkv_node_name in param_encoding_keys:
                    head_num = self.mha2sha_optim._gqa_extension.kv_head_num if \
                               self.mha2sha_optim.gqa_model and qkv_node_name[0] in ("k", "v") else \
                               attn_encoding_mapping_dict.head_num

                    node_mapping_dict = encoding_mapping_dict[qkv_node_name]
                    copy_mha_param_encodings_to_sha(mha_param_encoding,
                                                    sha_param_encoding,
                                                    _start_node,
                                                    qkv_node_name,
                                                    head_num,
                                                    node_mapping_dict,
                                                    self._mha_to_sha_encodings_names,
                                                    self._encoding_version)
            elif extension_type == ExtensionTypes.LORA:
                for branch_prefix in extension_branch_prefix:
                    head_num = self.mha2sha_optim._gqa_extension.kv_head_num if \
                               self.mha2sha_optim.gqa_model and branch_prefix[-1] in ("k", "v") else \
                               attn_encoding_mapping_dict.head_num

                    for qkv_node_name in param_encoding_keys:
                        for lora_adaptor_set_num in self.mha2sha_optim.lora_adaptor_set_info_dict.keys():
                            dict_name = "_".join([branch_prefix, qkv_node_name, str(lora_adaptor_set_num)])

                            if dict_name in encoding_mapping_dict:
                                node_mapping_dict = encoding_mapping_dict[dict_name]
                            else:
                                node_mapping_dict = None

                            if node_mapping_dict:
                                copy_mha_param_encodings_to_sha(mha_param_encoding,
                                                                sha_param_encoding,
                                                                _start_node,
                                                                qkv_node_name,
                                                                head_num,
                                                                node_mapping_dict,
                                                                self._mha_to_sha_encodings_names,
                                                                self._encoding_version)

            else:
                for branch_prefix in extension_branch_prefix:
                    head_num = self.mha2sha_optim._gqa_extension.kv_head_num if \
                               self.mha2sha_optim.gqa_model and branch_prefix[-1] in ("k", "v") else \
                               attn_encoding_mapping_dict.head_num

                    # Check if lora_{qkv}_lora_b in encoding_mapping_dict for lora exist on the branch or not
                    for qkv_node_name in param_encoding_keys:
                        if branch_prefix+"_"+qkv_node_name in encoding_mapping_dict:
                            node_mapping_dict = encoding_mapping_dict[branch_prefix+"_"+qkv_node_name]
                        else:
                            node_mapping_dict = None

                        if node_mapping_dict:
                            copy_mha_param_encodings_to_sha(mha_param_encoding,
                                                            sha_param_encoding,
                                                            _start_node,
                                                            qkv_node_name,
                                                            head_num,
                                                            node_mapping_dict,
                                                            self._mha_to_sha_encodings_names,
                                                            self._encoding_version)

    def _check_sha_encoding(self, sha_encodings):
        # Check all activation encodings are exist in sha model.
        activation_encodings = sha_encodings["activation_encodings"]

        missing_key = []
        for key, _ in activation_encodings.items():
            is_input_acitvation = key in self.mha2sha_optim.get_node_by_input_name.keys()
            is_output_acitvation = key in self.mha2sha_optim.get_node_by_output_name.keys()

            if not (is_input_acitvation or is_output_acitvation):
                # This is normal to have missing activation encodings, MHA2SHA convertor doesn't
                # capture all the activation encodings in mha. It only create maps for matmul inputs
                # and output, scale output, and softmax output.
                if key not in self.mha_encodings["activation_encodings"].keys():
                    log_warning(f"activation encoding: {key} is not input or output to any node in MHA model and not in SHA model.")
                else:
                    log_debug(f"activation encoding: {key} is not input or output to any node in SHA model. Removing from sha_encoding.")
                missing_key.append(key)
        for _key in missing_key:
            del activation_encodings[_key]

        # Check param encodings exists in sha
        missing_key = []
        param_encodings = sha_encodings["param_encodings"]
        for key, _ in param_encodings.items():
            is_initializer = key in self.mha2sha_optim.get_initializer_by_name.keys()
            if not is_initializer:
                log_warning(f"In converted sha param encoding: {key} is missing initializer in sha model. Removing from sha_encoding.")
                missing_key.append(key)
        for _key in missing_key:
            del param_encodings[_key]

        # Check weight/bias for Linear/Gemm/Conv in sha model has encoding
        # 1. Weights has no encoding in sha and mha is ok.
        # 2. Weights has no encoding in sha but mha has encoding is NOT ok.
        self._mha_to_sha_encodings_names["param_encodings"]
        _sha_to_mha_param_encoding_names = { _sha_name: _mha_name for _mha_name, _sha_name_list in self._mha_to_sha_encodings_names["param_encodings"].items() for _sha_name in _sha_name_list}
        for init_name in self.mha2sha_optim.get_initializer_by_name.keys():
            node = self.mha2sha_optim.get_node_by_input_name[init_name][0]
            if node.op_type in ("MatMul", "Gemm", "Conv"):
                if init_name not in param_encodings.keys():
                    # As part of SHA and the corresponding MHA weight has encoding. That's not ok.
                    if init_name in _sha_to_mha_param_encoding_names.keys() and _sha_to_mha_param_encoding_names[init_name] in self.mha_encodings.keys():
                        log_warning(f"weight/bias initializer {init_name} missing param encoding in sha but mha encoding exists!")

    def _save_mha_to_sha_encodings_names(self, sha_export_path: Path):
        with open((sha_export_path / MHA_TO_SHA_NAME_MAPPING_PATH).as_posix(), 'w') as f:
            json.dump(self._mha_to_sha_encodings_names, f, indent=4)
