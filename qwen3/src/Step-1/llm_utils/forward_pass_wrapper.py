#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
"""  utility method to adapt original model, prepared model and model forward pass invocation """
import inspect

import contextlib

import json
import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from aimet_torch.utils import get_device

from packaging import version
from importlib.metadata import version as impLib_version


## NEXA: A simple recursion to convert any combination of tuple and list
#        into a flat tuple with sequential elements
def flatten_tensors(tup):
    """
    Recursively flattens a nested tuple or list of tensors into a generator of tensors.

    Args:
        tup (tuple or list or tensor): Nested structure of tensors.

    Yields:
        torch.Tensor: Flattened tensors.
    """
    """
    Flattens arbitrarily nested tuples/lists of tensors into a flat generator of tensors.
    Useful for handling model outputs or inputs that may be deeply nested.
    """
    if not isinstance(tup, (tuple, list)):
        yield tup
        return
    for x in tup:
        yield from flatten_tensors(x) 


## NEXA: Just zeros key value cache. They are all zeros, note that we specify the past_size here.
def get_padded_kv_values(past_size, num_layers, hidden_size, num_attention_heads, 
                         batch_size=1, num_kv_heads=32,
                         transposed_key_cache=True, device='cuda', dtype=torch.float32):
    """
    Creates zero-padded key and value cache tensors for transformer models.

    Args:
        past_size (int): Length of the past sequence to pad.
        num_layers (int): Number of transformer layers.
        hidden_size (int): Model hidden size.
        num_attention_heads (int): Number of attention heads.
        batch_size (int): Batch size.
        num_kv_heads (int): Number of key-value heads.
        transposed_key_cache (bool): Whether to use transposed key cache.
        device (str): Device to allocate tensors.
        dtype (torch.dtype): Data type of tensors.

    Returns:
        tuple: Tuple of (key, value) pairs for each layer.
    """
    """
    Returns a tuple of zero-initialized key and value tensors for each transformer layer,
    shaped according to the model's configuration. Used to initialize or pad the kv-cache.
    """
    def _cache(shape):
        return torch.zeros(shape).to(device=device, dtype=dtype)

    head_dim = num_kv_heads
    value = (batch_size, head_dim, past_size, hidden_size // num_attention_heads)
    key = (value[0], value[1], value[3], value[2]) if transposed_key_cache else tuple(value)
    past_key_values = tuple((_cache(key), _cache(value)) for _ in range(num_layers))
    return past_key_values



## NEXA: No special handling here, we would need to write our own hanlding code
#        THis is just used to write the code from the position_ids to get the right
#        cos and sin.
class RopeEmbedding:
    """
    Computes rotary positional embeddings (RoPE) for transformer models.

    Attributes:
        cos (torch.Tensor): Precomputed cosine embeddings.
        sin (torch.Tensor): Precomputed sine embeddings.
    """
    """
    Helper class to precompute and provide rotary positional embeddings (RoPE) for transformer models.
    Used to efficiently retrieve cos/sin tables for given position ids.
    """
    def __init__(self, device, head_dim=128, max_length=2048, config=None):
        """
        Initializes RopeEmbedding and precomputes cos/sin tables.

        Args:
            device (str or torch.device): Device for tensors.
            head_dim (int): Dimension per attention head.
            max_length (int): Maximum sequence length.
            config (object): Model config with RoPE parameters.
        """
        self.cos, self.sin = self.precompute(device, head_dim, max_length, config)

    def precompute(self, device, head_dim, max_length, config):
        """
        Precomputes RoPE cos/sin tables using LlamaRotaryEmbedding.

        Args:
            device (str or torch.device): Device for tensors.
            head_dim (int): Dimension per attention head.
            max_length (int): Maximum sequence length.
            config (object): Model config.

        Returns:
            tuple: (cos, sin) tensors for RoPE.
        """
        """
        Uses HuggingFace's LlamaRotaryEmbedding to generate cos/sin tables for RoPE.
        Handles compatibility with different transformers versions.
        """
        def _support_llama3_rope():
            import transformers
            return tuple([int(i) for i in transformers.__version__.split(".")]) >= (4,43,2)
            #return version.parse(impLib_version('transformers')) >= version.parse('4.43.2')

        head_dim = config.head_dim if hasattr(config, 'head_dim') else config.hidden_size // config.num_attention_heads
        kwargs = {
            'max_position_embeddings': config.max_position_embeddings,
            'base': config.rope_theta,
            'device': device,
        }
        if _support_llama3_rope():
            kwargs['config'] = config

        if not hasattr(config, 'rope_scaling'):
            setattr(config, 'rope_scaling', None)

        rope = LlamaRotaryEmbedding(dim=head_dim, **kwargs)
        dummy_x = torch.Tensor([1.0]).to(device)
        position_ids = torch.arange(max_length).view(1, -1).to(device)
        if hasattr(rope, '_original_forward'):
            embeddings = rope._original_forward(dummy_x, position_ids)
        else:
            embeddings = rope.forward(dummy_x, position_ids)

        # for adapted llama
        emb_size = embeddings[0].size(-1) // 2
        embeddings = [emb[:, :, :emb_size] for emb in embeddings]
        embeddings = [emb.unsqueeze(0) for emb in embeddings]
        return embeddings
    
    
    ## NEXA: Note the returned tensor shape as 
    #        [batch_size, 1, sequence_length, head_sim//2], as 4D tensors.
    def get_embedding(self, position_ids, dtype=torch.float32):
        """
        Retrieves RoPE cos/sin embeddings for given position ids.

        Args:
            position_ids (torch.Tensor): [batch_size, sequence_length] position indices.
            dtype (torch.dtype): Data type for output.

        Returns:
            tuple: (cos, sin) tensors for the positions.
        """
        """
        Returns the cos/sin RoPE embeddings for the provided position indices.
        """
        cos = self.cos[0,0,:,:]  # [seq_len, dim]
        sin = self.sin[0,0,:,:]  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1).to(dtype=dtype)
        sin = sin[position_ids].unsqueeze(1).to(dtype=dtype)
        return cos, sin


## NEXA: We note the use of the mask_neg here, which is -50, OK for the precision, and this is OK for the INT8 quantization!
#        We use the -50 since half of INT8 is 64, and abs(-50) < 64.
def prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length, mask_neg=-50.0):
    """
    Prepares a combined decoder attention mask (causal + padding) for transformer models.

    Args:
        attention_mask (torch.Tensor): Padding mask [batch, seq_len].
        input_shape (torch.Size): Shape of input [batch, seq_len].
        inputs_embeds (torch.Tensor): Dummy input for dtype/device.
        past_key_values_length (int): Length of past key/values.
        mask_neg (float): Value to use for masked positions.

    Returns:
        torch.Tensor: Combined attention mask [batch, 1, tgt_seq_len, src_seq_len].
    """
    """
    Returns a mask combining causal and padding masks for decoder self-attention.
    Used to ensure correct masking for both padding and autoregressive constraints.
    """
    # Copied from transformers.models.bart.modeling_bart._make_causal_mask
    def _make_causal_mask(
            input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0,
            mask_neg: float = -50.0
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape[0], input_ids_shape[1]
        # mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
        mask = torch.full((tgt_len, tgt_len), torch.tensor(mask_neg, device=device), device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    # Copied from transformers.models.bart.modeling_bart._expand_mask
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, mask_neg: float = -50.0, tgt_len: int = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        # return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), mask_neg)


    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
            mask_neg=mask_neg,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]

        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[1], mask_neg=mask_neg).to(
            inputs_embeds.device
        )

        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask

def get_position_embeddings_from_position_ids(position_ids, head_dim, max_length, device, dtype, config):
    """
    Utility to get RoPE embeddings for given position ids.

    Args:
        position_ids (torch.Tensor): Position indices.
        head_dim (int): Head dimension.
        max_length (int): Max sequence length.
        device (str or torch.device): Device.
        dtype (torch.dtype): Data type.
        config (object): Model config.

    Returns:
        tuple: (cos, sin) tensors.
    """
    """
    Wrapper to get RoPE cos/sin embeddings for a batch of position ids.
    """
    return RopeEmbedding(device=device, head_dim=head_dim, max_length=max_length, config=config).get_embedding(position_ids, dtype=dtype)

def prepare_combined_attention_mask(attention_mask, input_shape, past_key_values_length, device, mask_neg=-50.0,
                                     dtype=torch.float32):
    """
    Prepares a combined attention mask for models that require both causal and padding masks.

    Args:
        attention_mask (torch.Tensor): Padding mask.
        input_shape (torch.Size): Shape of input.
        past_key_values_length (int): Length of past key/values.
        device (str or torch.device): Device.
        mask_neg (float): Value for masked positions.
        dtype (torch.dtype): Data type.

    Returns:
        torch.Tensor: Combined attention mask.
    """
    """
    Returns a combined attention mask (causal + padding) for use in transformer models.
    """
    dummy_embedding = torch.tensor((1.0,)).to(torch.float32).to(device)
    new_mask = prepare_decoder_attention_mask(attention_mask, input_shape, dummy_embedding, past_key_values_length, mask_neg)
    # NEXA: We clamp the mask_neg from -100 to -50
    return new_mask.clamp_min(mask_neg).to(dtype)


class LLMForwardPassManager:
    """
    Manages the forward pass for LLMs, including input preparation, kv-cache handling, and output post-processing.

    Attributes:
        tokenizer: Tokenizer object.
        model: Model object.
        config: Model config.
        device: Device for computation.
        num_heads: Number of attention heads.
        num_kv_heads: Number of key-value heads.
        num_layers: Number of transformer layers.
        embed_dim: Embedding dimension.
        rope_theta: RoPE base theta.
        max_tokens: Maximum sequence length.
        num_tokens: Number of tokens per forward pass.
        use_position_embedding_input: Whether to use position embedding input.
        use_combined_mask_input: Whether to use combined mask input.
        transposed_key_cache: Whether key cache is transposed.
        mask_neg: Value for masked positions.
        use_input_embeddings: Whether to use input embeddings.
        return_new_key_value_only: Whether to return only new key/values.
        separate_tuple_input_output: Whether to use tuple input/output for kv-cache.
        record_test_vectors: Whether to record test vectors.
        dummy_kvcache_generator: Dummy kv-cache generator.
        input_id_to_embedding_converter: Function to convert input ids to embeddings.
    """
    """
    Main utility class to manage all aspects of LLM forward pass:
    - Prepares and pads inputs, attention masks, and kv-caches
    - Handles input validation and shape checking
    - Supports both input_ids and input_embeddings
    - Handles position embedding and combined mask logic
    - Post-processes outputs to extract logits and update kv-cache
    - Can be used as a callable for a full forward pass
    """
    def __init__(self, cfg, model, tokenizer, separate_tuple_input_output, num_tokens):
        """
        Initializes the forward pass manager with model, tokenizer, and config.
        """
        self.tokenizer = tokenizer
        self.model = model
        self.config = cfg

        self.device = get_device(model)

        self.num_heads = getattr(cfg, 'num_attention_heads', 1)
        self.num_kv_heads = getattr(cfg, 'num_key_value_heads')
        self.num_layers = getattr(cfg, 'num_hidden_layers', 32)
        self.embed_dim = getattr(cfg, 'hidden_size', 1024)
        self.rope_theta = getattr(cfg, "rope_theta", 10000.0)
        self.max_tokens = tokenizer.model_max_length
        
        ## NEXA: num_tokens is the ARN length, which represents the autoregressive number
        self.num_tokens = num_tokens
        self.use_position_embedding_input = getattr(cfg, 'use_position_embedding_input', False)
        self.use_combined_mask_input = getattr(cfg, 'use_combined_mask_input', False)
        self.transposed_key_cache = getattr(cfg, 'transposed_key_cache', False)
        self.mask_neg = getattr(cfg, 'mask_neg', -50)
        self.use_input_embeddings = getattr(cfg, 'use_input_embeddings', False)
        self.return_new_key_value_only = getattr(cfg, 'return_new_key_value_only', False)
        self.separate_tuple_input_output = separate_tuple_input_output
        
        ## TODO: What is the test_vectors again? 
        self.record_test_vectors = False  # users of this block wil enable/disable this as necessary with provided functions
        self.dummy_kvcache_generator = None  # DummyKvcacheGenerator(cfg)
        self.input_id_to_embedding_converter = None


    @property
    def dtype(self):
        """
        Returns the data type of the model parameters.

        Returns:
            torch.dtype: Data type.
        """
        """
        Returns the dtype of the model's parameters (e.g., float32, float16).
        """
        return next(self.parameters()).dtype

    def replace_model(self, new_model):
        """
        Replaces the managed model with a new one and moves it to the correct device.

        Args:
            new_model: The new model to use.
        """
        """
        Replace the internal model reference and move it to the current device.
        """
        self.model = new_model
        self.model.to(self.device)


    ## NEXA: This is a special siganture, which is used to handle the with,
    #        we use the yield to control what happens inside the with block and
    #        outside the with block.
    @contextlib.contextmanager
    def place_on_device(self, device):
        """
        Context manager to temporarily move the model to a different device.

        Args:
            device (str or torch.device): Target device.
        """
        """
        Context manager to temporarily move the model to a different device for a block of code.
        """
        original_device = self.device
        try:
            self.to(device)
            yield
        finally:
            self.to(original_device)

    def to(self, device=torch.device):
        """
        Moves the model to the specified device.

        Args:
            device (str or torch.device): Target device.
        """
        """
        Moves the model and updates the internal device reference.
        """
        self.device = torch.device(device)
        self.model.to(self.device)

    def parameters(self):
        """
        Returns the model parameters.

        Returns:
            Iterator: Model parameters.
        """
        """
        Returns an iterator over the model's parameters.
        """
        return self.model.parameters()

    def _tokenize_text(self, text, max_length):
        """
        Tokenizes input text using the registered tokenizer.

        Args:
            text (str): Input text.
            max_length (int): Maximum length.

        Returns:
            Encoded tensor from tokenizer.
        """
        """
        Tokenizes a string using the registered tokenizer, with truncation and no special tokens.
        """
        if self.tokenizer == None:
            print(
                "No tokenizer was registered with forward pass manager. Attempt to forward text inputs has failed.")
            assert False

        encoded_tensor = self.tokenizer(text, add_special_tokens=False, max_length=max_length, truncation=True)
        return encoded_tensor

    def _update_kv_cache(self, prev_key_value, new_key_value, max_cache_size, is_concatenated=False):
        """
        Updates the key-value cache by concatenating and shifting as needed.

        Args:
            prev_key_value: Previous kv-cache.
            new_key_value: New kv-cache.
            max_cache_size (int): Maximum allowed cache size.
            is_concatenated (bool): If new_key_value is already concatenated.

        Returns:
            Updated kv-cache.
        """
        """
        Concatenates and/or shifts the kv-cache to maintain the correct length.
        Handles both transposed and non-transposed cache layouts.
        """
        # past_key_value: [num_layers][2][key_value], where key_value can be a tensor or tuple of heads
        def _concat(a, b, dim):
            if isinstance(a, tuple):
                assert len(a) == len(b), 'Unexpected key/value pair'
                return tuple(_concat(ai, bi, dim) for ai, bi in zip(a, b))
            return torch.cat((a, b), dim=dim)

        def _do_concat(a, b, key_dim, value_dim):
            return tuple((_concat(ak, bk, key_dim), _concat(av, bv, value_dim)) for (ak, av), (bk, bv) in zip(a, b))
        
        
        ## NEXA: This is doing the shift like the sliding window
        def _shift(a, dim, shift_size):
            if isinstance(a, tuple):
                return tuple(_shift(ai, dim) for ai in a)
            assert dim in (2, 3), 'Unexpected shift axis'
            return a[:, :, shift_size:, :] if dim == 2 else a[:, :, :, shift_size:]

        def _do_shift(a, key_dim, value_dim, shift_size):
            return tuple((_shift(k, key_dim, shift_size), _shift(v, value_dim, shift_size)) for k, v in a)

        value_dim = 2
        key_dim = 3 if self.transposed_key_cache else 2

        if prev_key_value is None or is_concatenated:
            # some models concat new key values and old key values internally
            # `is_concatenated` indicates whether new_key_value is already concatenated
            next_key_value = new_key_value
        elif new_key_value is None:
            # when dummy_kv + None
            next_key_value = prev_key_value
        else:
            # if concat is NOT done, then concat
            next_key_value = _do_concat(prev_key_value, new_key_value, key_dim, value_dim)

        shift_size = next_key_value[0][1].shape[-2] - max_cache_size
        if shift_size > 0:
            next_key_value = _do_shift(next_key_value, key_dim, value_dim, shift_size)

        return next_key_value

    def validate_inputs(self, input_text=None, input_ids=None, input_embeddings=None, past_key_values=None):
        """
        Validates that only one input type is provided and kv-cache is valid.

        Args:
            input_text (str): Input text.
            input_ids (torch.Tensor): Input ids.
            input_embeddings (torch.Tensor): Input embeddings.
            past_key_values: Past kv-cache.

        Returns:
            bool: True if valid, False otherwise.
        """
        """
        Checks that only one of input_text, input_ids, or input_embeddings is provided,
        and that kv-cache and embedding converter are valid.
        """
        # make sure only one of input_text, input_ids, input_embeddings is passed in
        input_count = 0
        for input in (input_text, input_ids, input_embeddings):
            if input is not None:
                input_count = input_count + 1
        if input_count != 1:
            print("Incorrect number of arguments: one of (input_text, input_ids, input_embeddings) expected.")
            return False

        # make sure that input embedding function has been selected if input embeddings are to be used
        if self.use_input_embeddings and self.input_id_to_embedding_converter is None and input_embeddings is None:
            print(
                "use_input_embeddings is set to true, but no input_embeddings were provided, and input_id_to_embedding_converter is None.")
            return False

        if past_key_values is not None and past_key_values[0][1].shape[-2] > self.max_tokens - self.num_tokens:
            print(
                "Provided past_key_values are too long. past_key_values length cannot exceed max_tokens - num_tokens.")
            return False

        return True

    def validate_input_lengths(self, input_length, mask_length, attn_length):
        """
        Validates input, mask, and attention lengths.

        Args:
            input_length (int): Input sequence length.
            mask_length (int): Mask length.
            attn_length (int): Attention length.

        Returns:
            bool: True if valid, False otherwise.
        """
        """
        Checks that input, mask, and attention lengths are consistent and valid for the model.
        """
        if 1 > input_length or input_length > self.num_tokens:
            print(
                f"Incorrect sequence length provided: input_length({input_length}) must be less than or equal to num_tokens ({self.num_tokens}).")
            return False

        if attn_length < mask_length or mask_length < input_length:
            print(
                f"Incorrect attention length provided: mask_length({mask_length}) must be greater than or equal to input_length({input_length}) and less than or equal to the sum({attn_length}) of input_length and kv_length.")
            return False

        return True

    def validate_processed_inputs(self, input=None, attention_mask=None, past_key_values=None):
        """
        Validates processed input, attention mask, and kv-cache shapes.

        Args:
            input (torch.Tensor): Input tensor.
            attention_mask (torch.Tensor): Attention mask.
            past_key_values: Past kv-cache.

        Returns:
            bool: True if valid, False otherwise.
        """
        """
        Checks that processed input, attention mask, and kv-cache have the correct shapes.
        """
        # if input make sure that only correct length sequence is provided
        if input.shape[1] != self.num_tokens:
            print(
                f"Incorrect prcessing for sequence length: dim 1({input.shape[1]}) of input must be of length num_tokens in KV cache mode.")
            return False

        if attention_mask.shape[1] != self.max_tokens:
            print(
                f"Incorrect prcessing for attention length: dim 1({attention_mask.shape[1]}) of input must be of length max_tokens.")
            return False

        if past_key_values is not None and past_key_values[0][1].shape[-2] != self.max_tokens - self.num_tokens:
            print(
                f"Incorrect  prcessing for past_kv length: dim 1({past_key_values[0][1].shape[-2]}) of input must be of length max_tokens - num_tokens.")
            return False

        return True

    def get_position_embeddings_from_position_ids(self, position_ids):
        """
        Gets RoPE embeddings for given position ids.

        Args:
            position_ids (torch.Tensor): Position indices.

        Returns:
            tuple: (cos, sin) tensors or position_ids.
        """
        """
        Returns RoPE cos/sin embeddings for the given position ids using the model config.
        """
        return get_position_embeddings_from_position_ids(position_ids,
                                                          head_dim=self.embed_dim // self.num_heads,
                                                          max_length=self.max_tokens,
                                                          device=self.device,
                                                          dtype=self.dtype,
                                                          config=self.config)

    def prepare_combined_attention_mask(self, attention_mask, input_shape, past_kv_length):
        """
        Prepares a combined attention mask for the model.

        Args:
            attention_mask (torch.Tensor): Attention mask.
            input_shape (torch.Size): Input shape.
            past_kv_length (int): Past kv-cache length.

        Returns:
            torch.Tensor: Combined attention mask.
        """
        """
        Returns a combined (causal + padding) attention mask for the model.
        """
        return prepare_combined_attention_mask(attention_mask, input_shape=input_shape,
                                                past_key_values_length=past_kv_length, device=self.device,
                                                mask_neg=self.mask_neg, dtype=self.dtype)

    def prepare_inputs(self, input_text=None, input_ids=None, input_embeddings=None, attention_mask=None,
                       past_key_values=None, **kwargs):
        """
        Prepares model inputs, including padding, kv-cache, and attention masks.

        Args:
            input_text (str): Input text.
            input_ids (torch.Tensor): Input ids.
            input_embeddings (torch.Tensor): Input embeddings.
            attention_mask (torch.Tensor): Attention mask.
            past_key_values: Past kv-cache.
            **kwargs: Additional arguments.

        Returns:
            tuple: (inputs dict, kv-cache info bundle)
        """
        """
        Prepares and pads all model inputs, attention masks, and kv-caches for a forward pass.
        Handles both input_ids and input_embeddings, and supports tuple or dict kv-cache formats.
        """
        assert self.validate_inputs(input_text, input_ids, input_embeddings, past_key_values)

        kvcache_info_bundle = {}  # dict to hold values needed for KV cache post-processing

        if input_text is not None:
            max_length = self.num_tokens
            encoded = self._tokenize_text(input_text, max_length=max_length)
            input_ids = encoded.input_ids
            attention_mask = encoded.attention_mask

        if self.use_input_embeddings:
            if input_embeddings is None:
                input_embeddings = self.input_id_to_embedding_converter(input_ids).to(dtype=self.dtype)
            input = input_embeddings
            # if we cast this input to long, all floats become zero in the input which we do not want
            input = torch.tensor(input.clone().detach(), dtype=self.dtype, device=self.device)
        else:
            input = input_ids
            input = torch.tensor(input.clone().detach(), dtype=torch.long, device=self.device)
        batch_size = input.shape[0]
        input_length = input.shape[1]

        kvcache_info_bundle["input_length"] = input_length

        # get kv_length from past values because values are not transposed.
        kv_length = past_key_values[0][1].shape[-2] if past_key_values is not None else 0
        attn_length = min(input_length + kv_length, self.max_tokens)

        # Checking attention_mask first, otherwise we will create attention_mask from input_extensions.
        # input_extensions will be empty tensors and so as attention_mask.
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, input_length + kv_length), dtype=torch.long, device=self.device)

        # cast type and move device
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.to(dtype=torch.long, device=self.device)
        else:
            # if attention_mask is not a tensor, get tensor
            attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=self.device)
        mask_length = attention_mask.shape[1]

        assert self.validate_input_lengths(input_length, mask_length, attn_length)

        # Pad inputs
        if input_length < self.num_tokens:
            shape = (batch_size, self.num_tokens - input_length)
            # expand shape if input is input_embeddings
            if self.use_input_embeddings:
                shape += (input.shape[-1],)
            input_extensions = torch.full(
                shape,
                fill_value=self.tokenizer.eos_token_id,
                dtype=input.dtype,
                device=self.device
            )
            input = torch.cat((input_extensions, input), dim=1)

        # Pad attention_mask
        attention_mask_extension_for_padded_kvcache = torch.zeros((batch_size, attn_length - mask_length),
                                                                  dtype=torch.long, device=self.device)
        attn_mask_extensions_for_padded_input = torch.zeros((batch_size, self.num_tokens - input_length), \
                                                            dtype=torch.long, device=self.device)
        attention_mask = torch.cat((
            attention_mask_extension_for_padded_kvcache,
            attention_mask[:, :-input_length],
            attn_mask_extensions_for_padded_input,
            attention_mask[:, -input_length:]
        ), dim=1
        )

        desired_kv_length = self.max_tokens - self.num_tokens
        kv_padding_length = max(desired_kv_length - kv_length, 0)
        kvcache_info_bundle['kv_padding_length'] = kv_padding_length

        past_key_values_extension = get_padded_kv_values(past_size=kv_padding_length,
                                                         num_layers=self.num_layers,
                                                         hidden_size=self.embed_dim,
                                                         num_attention_heads=self.num_heads,
                                                         num_kv_heads=self.num_kv_heads,
                                                         transposed_key_cache=self.transposed_key_cache,
                                                         device=self.device,
                                                         dtype=self.dtype)
        past_key_values = self._update_kv_cache(past_key_values_extension, past_key_values, desired_kv_length)

        attention_mask_extension = torch.zeros((batch_size, kv_padding_length), dtype=torch.long,
                                               device=self.device)
        attention_mask = torch.cat((attention_mask_extension, attention_mask), dim=1)

        assert self.validate_processed_inputs(input, attention_mask, past_key_values)

        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids = position_ids.clip(0, self.max_tokens - 1)
        position_ids = position_ids[..., -self.num_tokens:]

        if self.use_position_embedding_input:
            position_ids = self.get_position_embeddings_from_position_ids(position_ids)

        if self.use_combined_mask_input:
            past_kv_length = self.max_tokens - self.num_tokens
            attention_mask = self.prepare_combined_attention_mask(attention_mask, input.shape, past_kv_length)

        inputs = {
            'attention_mask': attention_mask,
        }

        if self.separate_tuple_input_output and self.config.use_position_embedding_input:
            inputs['position_ids_cos'] = position_ids[0]
            inputs['position_ids_sin'] = position_ids[1]
        else:
            inputs['position_ids'] = position_ids

        if self.use_input_embeddings:
            inputs['inputs_embeds'] = input
        else:
            inputs['input_ids'] = input

        if self.separate_tuple_input_output:
            if "input_names" in kwargs:
                input_names = kwargs['input_names']
            else:
                signature = inspect.signature(self.model.forward)
                input_names = tuple(signature.parameters.keys())
            flattened_key_values = flatten_tensors(past_key_values)
            # input_ids, attention_mask, position_ids_cos, position_ids_sin, (past_key_values)
            # this order is different when we use the input_embeddings -> attention_mask, position_ids_cos, position_ids_sin, (past_key_values), inputs_embeds
            if not self.use_input_embeddings:
                offset = 4 if self.config.use_position_embedding_input else 3
                for key, value in zip(input_names[offset:], flattened_key_values):
                    inputs[key] = value
            else:
                for key, value in zip(input_names[3:-1], flattened_key_values):
                    inputs[key] = value
        else:
            inputs['past_key_values'] = past_key_values
        return inputs, kvcache_info_bundle


    ## NEXA: This is to get what we really need for the next step, training, or inference etc.
    def prepare_outputs(self, outputs, prepared_inputs, kvcache_info_bundle):
        """
        Processes model outputs to extract logits and update kv-cache.

        Args:
            outputs (tuple): Model outputs.
            prepared_inputs (dict): Prepared inputs.
            kvcache_info_bundle (dict): KV-cache info.

        Returns:
            dict: {'lm_logits': logits, 'past_key_values': updated kv-cache}
        """
        """
        Extracts logits and updates kv-cache from model outputs, handling both tuple and dict formats.
        """
        lm_logits = outputs[0]
        lm_logits = lm_logits[:, -kvcache_info_bundle["input_length"]:, :]

        def _get_past_kv_from_outputs(outputs):
            if self.separate_tuple_input_output:
                return tuple((outputs[(2 * i) + 1], outputs[(2 * i) + 2]) for i in range(self.num_layers))
            else:
                return outputs[-1]

        def _get_past_kv_from_prepared_inputs(prepared_inputs):
            if self.separate_tuple_input_output:
                return tuple((prepared_inputs[f"past_key_{i}_in"], prepared_inputs[f"past_value_{i}_in"]) for i in range(self.num_layers))
            else:
                return prepared_inputs['past_key_values'] if 'past_key_values' in prepared_inputs else None

        new_past_key_values = _get_past_kv_from_outputs(outputs)
        new_past_key_values = self._update_kv_cache(
            None,
            new_past_key_values,
            kvcache_info_bundle["input_length"]
        )
        old_past_key_values = _get_past_kv_from_prepared_inputs(prepared_inputs)

        current_kv_length_with_padding_removed = self.max_tokens - self.num_tokens - kvcache_info_bundle[
            'kv_padding_length'] + kvcache_info_bundle['input_length']  # number of non-dummy inputs

        past_key_values = self._update_kv_cache(
            old_past_key_values,
            new_past_key_values,
            current_kv_length_with_padding_removed
        )

        return {'lm_logits': lm_logits, 'past_key_values': past_key_values}

    def __call__(self, *args, **kwargs):
        """
        Runs the full forward pass: prepares inputs, runs model, processes outputs.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict: Model outputs.
        """
        """
        Prepares inputs, runs the model, and post-processes outputs in a single call.
        """
        prepared_inputs, kvcache_info_bundle = self.prepare_inputs(*args, **kwargs)
        outputs = self.model(**prepared_inputs)
        prepared_outputs = self.prepare_outputs(outputs, prepared_inputs, kvcache_info_bundle)
        return prepared_outputs



## NEXA: This is to simulate the real inference process, where we use fixed input length and the fixed KV cache total
#        length.
def slice_inputs_and_run_successive_kvcache_inference(fpm, input_ids=None, input_embeds=None, **kwargs):
    """
    Slices long input sequences and runs inference in chunks, updating kv-cache successively.

    Args:
        fpm (LLMForwardPassManager): Forward pass manager.
        input_ids (torch.Tensor): Input ids.
        input_embeds (torch.Tensor): Input embeddings.
        **kwargs: Additional arguments (e.g., attention_mask).

    Returns:
        dict: Concatenated outputs from all chunks.
    """
    """
    For long sequences, slices input into chunks of fpm.num_tokens, runs inference chunk by chunk,
    and accumulates logits and kv-cache. Useful for models with limited context window.
    """
    if input_ids is not None:
        input_length = input_ids.shape[1]
    else:
        input_length = input_embeds.shape[1]

    outputs = {}

    attention_mask = kwargs.pop('attention_mask', None)

    for idx in range(0, input_length, fpm.num_tokens)[::-1]:
        idx = input_length - idx

        if attention_mask is not None:
            cache_offset = attention_mask.shape[1] - input_length
            kwargs["attention_mask"] = attention_mask[:, max(0, cache_offset + idx - fpm.max_tokens):cache_offset + idx]

        if input_ids is not None:
            cur_outputs = fpm(input_ids=input_ids[:, max(0, idx - fpm.num_tokens):idx], **kwargs)
        elif input_embeds is not None:
            cur_outputs = fpm(input_ids=None, input_embeddings=input_embeds[:, max(0, idx - fpm.num_tokens):idx, :],
                              **kwargs)
        else:
            print("No input_ids or inputs_embeds provided to inference generator!")
            assert False

        # get valid outputs
        bsz, length, dim = cur_outputs['lm_logits'].shape

        outputs['lm_logits'] = torch.cat(
            (outputs.get('lm_logits', torch.zeros((bsz, 0, dim), device=fpm.device)), cur_outputs['lm_logits']),
            dim=1)
        kwargs['past_key_values'] = outputs['past_key_values'] = cur_outputs['past_key_values']

    return outputs

