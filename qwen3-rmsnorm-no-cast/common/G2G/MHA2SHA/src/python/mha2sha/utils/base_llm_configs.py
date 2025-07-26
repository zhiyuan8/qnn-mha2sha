# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

_llama2_flags = {
    "create_input_lists": False,
    "disable_auto_attn_finder": False,
    "gqa_model": False,
    "handle_rope_ops": True,
    "handle_past_key_value": True,
    "llm_model": True,
    "mha_conv": True,
    "nchw_aligned": False,
    "no_verification": False,
    "prepared_model": True,
    "strict": True,
}

_all_off_flags = {arg_name: False for arg_name in _llama2_flags.keys()}
_lora_flags = {
    "lora_model": True,
    "nchw_aligned": True,
    "mha_conv": True,
}

base_llms_to_flags = {
    "llama2": _llama2_flags,
    "llama2_lora_v2": _llama2_flags | _lora_flags | {"lora_alpha_from_input": True},
    "llama3": _llama2_flags | {"gqa_model": True},
    "llama3_lora_v1": _llama2_flags | {"gqa_model": True} | _lora_flags | {"lora_alpha_from_input": False},
    "llama3_lora_v3": _llama2_flags | {"gqa_model": True} | _lora_flags | {"lora_alpha_from_input": True},
    "sd_2.1": _all_off_flags | {"nchw_aligned": True, "mha_conv": True},
    "sd_2.1_lora_v2": _all_off_flags | _lora_flags | {"lora_alpha_from_input": True},
    "sd_2.1_lora_v3": _all_off_flags | _lora_flags | {"lora_alpha_from_input": True},
    "gpt2": _llama2_flags | {"handle_rope_ops": False}
}
