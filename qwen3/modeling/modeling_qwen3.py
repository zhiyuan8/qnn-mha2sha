from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from configuration_qwen3 import Qwen3Config
from transformers.utils import logging
from transformers.cache_utils import Cache, DynamicCache
import math

logger = logging.get_logger(__name__)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    Applies rotary positional embeddings to query and key tensors.

    Args:
        q (torch.Tensor): Query tensor of shape (..., seq_len, head_dim)
        k (torch.Tensor): Key tensor of shape (..., seq_len, head_dim)
        cos (torch.Tensor): Cosine embedding tensor, usually shape (..., seq_len, head_dim // 2)
        sin (torch.Tensor): Sine embedding tensor, usually shape (..., seq_len, head_dim // 2)
        unsqueeze_dim (int): Dimension along which to unsqueeze cos/sin for broadcasting

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors, same shape as input q and k

    Line-by-line explanation:
        1. cos = cos.unsqueeze(unsqueeze_dim)
           # Add a singleton dimension to cos for broadcasting with q/k.
        2. sin = sin.unsqueeze(unsqueeze_dim)
           # Add a singleton dimension to sin for broadcasting with q/k.
        3. cos = torch.cat([cos, cos], dim=-1)
           # Duplicate cos along the last dimension to match the full head_dim.
           # This is needed if cos was originally computed for head_dim // 2.
        4. sin = torch.cat([sin, sin], dim=-1)
           # Duplicate sin along the last dimension to match the full head_dim.
        5. q_embed = (q * cos) + (rotate_half(q) * sin)
           # Apply rotary embedding to q.
        6. k_embed = (k * cos) + (rotate_half(k) * sin)
           # Apply rotary embedding to k.
        7. return q_embed, k_embed
           # Return the rotated q and k.

    - Some implementations (like Llama, some HuggingFace models) generate cos/sin for only half the head dimension (head_dim // 2), because RoPE is mathematically defined for pairs of features.
    - Other implementations (including some older or custom code) generate cos/sin for the full head_dim directly, interleaving the values as needed.
    In this implementation, we use the full head_dim directly, interleaving the values as needed.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _apply_rope_single(x, rope_vals: Tuple[torch.Tensor, torch.Tensor]):
    """
    Based on FacebookResearch's llama, provided by Carl
    """
    rope_real = rope_vals[0]  # shape should be 1, 1, seqlen, head_dim/2
    rope_im = rope_vals[1]  # shape should be 1, 1, seqlen, head_dim/2

    # TODO: Why HF uses different coordinates from the paper
    x_real = x[:, :, :, : x.shape[-1] // 2]  # extract first half elements
    x_im = x[:, :, :, x.shape[-1] // 2 :]  # extract second half elements

    x_prod_real = x_real * rope_real - x_im * rope_im
    x_prod_im = x_real * rope_im + x_im * rope_real

    # TODO: HF need to uses different interleaving
    x = torch.cat((x_prod_real, x_prod_im), dim=3).view(*x.shape)
    return x


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        k_cache=None,
        v_cache=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = eager_attention_forward
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Qwen3PreTrainedModel(PreTrainedModel):
    config_class = Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen3RMSNorm):
            module.weight.data.fill_(1.0)


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # emb = torch.cat((freqs, freqs), dim=-1)
            emb = freqs
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen3ForCausalLM(Qwen3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


## =========================================================================
##  QNN-based model definition
## =========================================================================
def qnn_apply_rope(input, cos, sin):
    # input: (bsz, num_heads, q_len, head_dim)
    # cos, sin: (bsz, 1, seq_len, head_dim)
    output = (input * cos) + (rotate_half(input) * sin)
    return output


class QNNQwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.qnn_initialized = False

    def qnn_init(self):
        self.gate_proj_conv = nn.Conv2d(
            self.config.hidden_size,
            self.config.intermediate_size,
            1,
            bias=self.gate_proj.bias is not None,
        )
        self.up_proj_conv = nn.Conv2d(
            self.config.hidden_size,
            self.config.intermediate_size,
            1,
            bias=self.up_proj.bias is not None,
        )
        self.down_proj_conv = nn.Conv2d(
            self.config.intermediate_size,
            self.config.hidden_size,
            1,
            bias=self.down_proj.bias is not None,
        )
        self.gate_proj_conv.weight.data.copy_(self.gate_proj.weight[:, :, None, None]).to(self.gate_proj.weight.device)
        self.up_proj_conv.weight.data.copy_(self.up_proj.weight[:, :, None, None]).to(self.up_proj.weight.device)
        self.down_proj_conv.weight.data.copy_(self.down_proj.weight[:, :, None, None]).to(self.down_proj.weight.device)
        if self.gate_proj.bias is not None:
            self.gate_proj_conv.bias.data.copy_(self.gate_proj.bias).to(self.gate_proj.weight.device)
        if self.up_proj.bias is not None:
            self.up_proj_conv.bias.data.copy_(self.up_proj.bias).to(self.up_proj.weight.device)
        if self.down_proj.bias is not None:
            self.down_proj_conv.bias.data.copy_(self.down_proj.bias).to(self.down_proj.weight.device)
        self.is_conv = True
        del self.gate_proj, self.up_proj, self.down_proj
        self.qnn_initialized = True

    def qnn_forward(self, x):
        bsz, _, _ = x.size()
        x = torch.reshape(x, (bsz, -1, 1, self.hidden_size))
        x = x.transpose(1, 3)  # Transpose right before and after Conv
        x = self.down_proj_conv(self.act_fn(self.gate_proj_conv(x)) * self.up_proj_conv(x))
        x = x.transpose(1, 3)
        x = torch.reshape(x, (bsz, -1, self.hidden_size))
        return x

    def forward(self, x):
        if not self.qnn_initialized:
            raise ValueError("QNN-based MLP is not initialized")
        return self.qnn_forward(x)


class QNNQwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        ## QNN adapatation
        self.qnn_initialized = False

    def qnn_init(self):
        self.q_proj_conv = nn.Conv2d(
            self.hidden_size,
            self.config.num_attention_heads * self.head_dim,
            1,
            bias=self.config.attention_bias,
        )
        self.k_proj_conv = nn.Conv2d(
            self.hidden_size,
            self.config.num_key_value_heads * self.head_dim,
            1,
            bias=self.config.attention_bias,
        )
        self.v_proj_conv = nn.Conv2d(
            self.hidden_size,
            self.config.num_key_value_heads * self.head_dim,
            1,
            bias=self.config.attention_bias,
        )
        self.o_proj_conv = nn.Conv2d(
            self.config.num_attention_heads * self.head_dim,
            self.config.hidden_size,
            1,
            bias=self.config.attention_bias,
        )
        self.q_proj_conv.weight.data.copy_(self.q_proj.weight[:, :, None, None])
        self.k_proj_conv.weight.data.copy_(self.k_proj.weight[:, :, None, None])
        self.v_proj_conv.weight.data.copy_(self.v_proj.weight[:, :, None, None])
        self.o_proj_conv.weight.data.copy_(self.o_proj.weight[:, :, None, None])

        del self.q_proj, self.k_proj, self.v_proj, self.o_proj
        self.qnn_initialized = True

    def qnn_forward(
        self,
        hidden_states,  # (1, seq_len, hidden_size), not 4D
        cos,  # (1, 1, seq_len, head_dim)
        sin,  # (1, 1, seq_len, head_dim)
        attention_mask,  # (1, 1, seq_len, dst_len), with -50 for masked tokens
        k_cache,  # (1, 1, head_dim, KV_LEN)
        v_cache,  # (1, 1, KV_LEN, head_dim)
    ):
        bsz, q_len, _ = hidden_states.shape
        hidden_states = torch.reshape(hidden_states, (bsz, q_len, 1, self.config.hidden_size)).transpose(1, 3)
        query_states = self.q_proj_conv(hidden_states)
        key_states = self.k_proj_conv(hidden_states)
        value_states = self.v_proj_conv(hidden_states)
        query_states = query_states.reshape(bsz, self.config.num_attention_heads, self.head_dim, q_len).transpose(
            2, 3
        )  # (bsz, num_heads, q_len, head_dim)
        new_key_states = key_states.reshape(bsz, self.config.num_key_value_heads, self.head_dim, q_len).transpose(
            2, 3
        )  # (bsz, num_kv_heads, q_len, head_dim)
        new_value_states = value_states.reshape(bsz, self.config.num_key_value_heads, self.head_dim, q_len).transpose(
            2, 3
        )  # (bsz, num_kv_heads, q_len, head_dim)

        # apply norm
        ## HACK: No use the norm here to test if the MHA2SHA works then
        query_states = self.q_norm(query_states)
        new_key_states = self.k_norm(new_key_states)

        query_states = _apply_rope_single(query_states, (cos, sin))
        new_key_states = _apply_rope_single(new_key_states, (cos, sin))

        ## transpose ahead to save time
        new_key_states = new_key_states.transpose(2, 3)  # (bsz, num_kv_heads, head_dim, q_len)

        key_states = torch.cat([k_cache, new_key_states], dim=-1)
        value_states = torch.cat([v_cache, new_value_states], dim=-2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)  # (bsz, num_attn_heads, q_len, head_dim)
        value_states = repeat_kv(value_states, self.num_key_value_groups)  # (bsz, num_attn_heads, q_len, head_dim)

        attn_weights = torch.matmul(query_states, key_states) / math.sqrt(self.head_dim)

        ## NEXA: attention mask is prepared as a 4D tensor
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, 1, -1)
        attn_output = attn_output.transpose(1, 3)
        attn_output = self.o_proj_conv(attn_output)
        attn_output = attn_output.transpose(1, 3)
        attn_output = attn_output.reshape(bsz, q_len, -1)  # (bsz, q_len, hidden_size)
        return attn_output, attn_weights, new_key_states, new_value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        k_cache=None,
        v_cache=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if not self.qnn_initialized:
            raise ValueError("QNN-based attention is not initialized")
        return self.qnn_forward(
            hidden_states,
            cos,
            sin,
            attention_mask,
            k_cache,
            v_cache,
            **kwargs,
        )


class QNNQwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.self_attn = QNNQwen3Attention(config, layer_idx)

        # Add the MLP part, the same as before
        self.mlp = QNNQwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def qnn_init(self):
        self.self_attn.qnn_init()
        self.mlp.qnn_init()

    def forward(
        self,
        hidden_states,  # (1, seq_len, hidden_size), not 4D
        cos,  # (1, 1, seq_len, head_dim)
        sin,  # (1, 1, seq_len, head_dim)
        attention_mask,  # (1, 1, seq_len, dst_len), with -50 for masked tokens
        k_cache,  # (1, 1, head_dim, KV_LEN)
        v_cache,  # (1, 1, KV_LEN, head_dim)
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # self attention
        attn_output, _, new_key_states, new_value_states = self.self_attn.forward(
            hidden_states, cos, sin, attention_mask, k_cache, v_cache
        )

        hidden_states = residual + attn_output

        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, new_key_states, new_value_states


class QNNQwen3(Qwen3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # add a self.model so that I can load weights from original HF model
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.model.layers = nn.ModuleList(
            [QNNQwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.model.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.vocab_size = config.vocab_size

        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        # Initialize weights and apply final processing
        self.post_init()

    def qnn_init(self):
        for layer in self.model.layers:
            layer.qnn_init()

        self.model.lm_head_conv = nn.Conv2d(
            self.hidden_size,
            self.vocab_size,
            1,
            bias=False,
        )
        self.model.lm_head_conv.weight.data.copy_(self.model.embed_tokens.weight[:, :, None, None]).to(
            self.model.embed_tokens.weight.device
        )

    def forward(
        self,
        input_ids,  # [1, seq_len]
        cos,  # shape: [1, 1, L, head_dim]
        sin,  # shape: [1, 1, L, head_dim]
        attention_mask,  # shape: [1, 1, tgt_len, src_len]
        all_layers_kv_cache,  # num_layers * 2 for qwen3 1.7B, it is 28*2=56
    ):
        input_embeds = self.model.embed_tokens(input_ids)

        _, seq_len, hidden_size = input_embeds.shape
        hidden_states = input_embeds
        updated_kv_cache = []
        for layer_idx, layer in enumerate(self.model.layers):
            curr_k_cache = all_layers_kv_cache[layer_idx * 2]
            curr_v_cache = all_layers_kv_cache[layer_idx * 2 + 1]
            hidden_states, new_k_cache, new_v_cache = layer.forward(
                hidden_states,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
                k_cache=curr_k_cache,
                v_cache=curr_v_cache,
            )
            updated_kv_cache.append(new_k_cache)
            updated_kv_cache.append(new_v_cache)

        # now last_hidden is of shape (1, seq_len, hidden_size)
        last_hidden = self.model.norm(hidden_states)
        last_hidden = last_hidden.reshape(1, seq_len, 1, self.hidden_size)
        last_hidden = last_hidden.transpose(1, 3)
        logits = self.model.lm_head_conv(last_hidden)  # (1, vocab_size, 1, seq_len)
        logits = logits.transpose(1, 3)

        ## I do this to make it exactly the same as the llama model example
        logits = logits.reshape(1, seq_len, -1)
        result = (
            logits,
            *updated_kv_cache,
        )
        return result


from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import copy

class QNNLLMUtils:

    def __init__(
        self,
        seq_len,
        kv_len,
        device,
        config: Qwen3Config,
    ):
        self.seq_len = seq_len
        self.total_len = kv_len + seq_len
        self.kv_len = kv_len
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_key_value_heads
        self.num_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        self.device = device
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.eos_token_id = 151645

    def get_kv_cache(self):
        all_layers_kv_cache = []
        for _ in range(self.num_layers):
            all_layers_kv_cache.append(torch.zeros(1, self.num_kv_heads, self.head_dim, self.kv_len))
            all_layers_kv_cache.append(torch.zeros(1, self.num_kv_heads, self.kv_len, self.head_dim))
        all_layers_kv_cache = [kv_cache.to(self.device) for kv_cache in all_layers_kv_cache]
        return all_layers_kv_cache

    def get_position_ids(self, n_past, curr_len):
        return torch.arange(n_past, n_past + curr_len).to(self.device)

    def get_cos_sin(self, x, position_ids):
        # position_ids: (seq_len)
        position_ids = position_ids.unsqueeze(0)
        cos, sin = self.rotary_emb(x, position_ids)
        cos, sin = cos.to(self.device), sin.to(self.device)
        return cos.unsqueeze(0), sin.unsqueeze(0)

    ## HACK: This is some special part. For fixed KV length, we concat the KV with previous value,
    # so we have a unique attention mask layout.
    def get_attention_mask(self, n_past, curr_len):
        attention_mask = torch.full((self.seq_len, self.total_len), -50.0)
        for i in range(curr_len):
            for j in range(n_past):
                attention_mask[i, j] = 0

        for i in range(curr_len):
            for j in range(self.kv_len, self.kv_len + curr_len):
                if i >= j - self.kv_len:
                    attention_mask[i, j] = 0
        attention_mask = torch.clamp(attention_mask, min=-50.0)
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        attention_mask = attention_mask.to(self.device)
        return attention_mask

    def update_kv_cache(self, kv_cache_lst, new_kv_lst, n_past, curr_len):
        num_layers = len(kv_cache_lst) // 2  # 28 for qwen3 1.7B
        for layer_idx in range(num_layers):
            # update k_cache
            k_cache = kv_cache_lst[layer_idx * 2]  # size (:, :, head_dim, KV_LEN)
            new_k_cache = new_kv_lst[layer_idx * 2]  # size (:, :, head_dim, curr_len)
            k_cache[:, :, :, n_past : n_past + curr_len] = new_k_cache[:, :, :, :curr_len]

            # update v_cache
            v_cache = kv_cache_lst[layer_idx * 2 + 1]  # size (:, :, KV_LEN, head_dim)
            new_v_cache = new_kv_lst[layer_idx * 2 + 1]  # size (:, :, curr_len, head_dim)
            v_cache[:, :, n_past : n_past + curr_len, :] = new_v_cache[:, :, :curr_len, :]

            # update k_cache_lst and v_cache_lst
            kv_cache_lst[layer_idx * 2] = k_cache
            kv_cache_lst[layer_idx * 2 + 1] = v_cache
        return kv_cache_lst

    def prepare_inputs(self, input_ids):
        curr_len = input_ids.shape[1]
        n_past = 0
        input_ids = torch.cat(
            [input_ids, torch.full((1, self.seq_len - curr_len), self.eos_token_id, device=input_ids.device)],
            dim=-1,
        )
        input_ids = input_ids.to(self.device)
        attention_mask = self.get_attention_mask(n_past, curr_len)
        position_ids = self.get_position_ids(n_past, self.seq_len)
        cos, sin = self.get_cos_sin(attention_mask, position_ids)
        all_layer_kv_cache = self.get_kv_cache()
        return input_ids, attention_mask, cos, sin, all_layer_kv_cache

    def ppl_eval(self, data_loader, model, num_batches=0):
        raise NotImplementedError("PPL evaluation is not implemented for Qwen3")
        # We note that data_loader can only help to get the input_ids
        # all other inputs are initialized
        # if num_batches == 0:
        #     num_batches = len(data_loader)

        # loss = 0

        # ## prepare all inputs except input_ids
        # input_ids = next(iter(data_loader))["input_ids"]
        # input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        # curr_len = input_ids.shape[1]
        # input_ids, attention_mask, cos, sin, all_layer_kv_cache = self.prepare_inputs(input_ids, get_other_inputs=False)
        # attention_mask = attention_mask.to(model.dtype)
        # cos = cos.to(model.dtype)
        # sin = sin.to(model.dtype)
        # all_layer_kv_cache = [kv_cache.to(model.dtype) for kv_cache in all_layer_kv_cache]

        # for batch_id, batch in enumerate(tqdm(data_loader, total=num_batches, desc="Evaluating PPL")):
        #     if batch_id >= num_batches:
        #         break
        #     input_ids = batch["input_ids"]
        #     input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        #     curr_len = input_ids.shape[1]
        #     input_ids = torch.cat(
        #         [input_ids, torch.full((1, self.seq_len - curr_len), self.eos_token_id, device=input_ids.device)],
        #         dim=-1,
        #     )
        #     with torch.no_grad():
        #         output = model(input_ids, cos, sin, attention_mask, *all_layer_kv_cache)
        #     lm_logits = output[0].squeeze(0).permute(1, 2, 0)  # (bsz, seq_len, vocab_size)
        #     shift_logits = lm_logits[:, :-1, :][:, :curr_len, :].contiguous().to(dtype=torch.float32)  # (bsz, seq_len-1, vocab_size)
        #     shift_labels = input_ids[:, 1:][:, :curr_len].contiguous().to(shift_logits.device)  # (bsz, seq_len-1)
        #     loss_fct = CrossEntropyLoss()
        #     loss += loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # loss = loss / num_batches
        # ppl = loss.exp()
        # return ppl
