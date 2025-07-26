import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers import LlamaForCausalLM
from transformers import cache_utils
from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import (
    repeat_kv,
    Cache,
    DynamicCache,
    LlamaAttention,
    LlamaConfig,
    apply_rotary_pos_emb,
)


## NEXA: This is essentially to apply the cos, sin to the input tensor x. 
#        Usually, x is the output after the q_proj and k_proj. We then use
#        cos, sin to apply on the x. x is of shape [1, 1, seq_len, head_dim]
#        and cos, sin are of shape [1, 1, seq_len, head_dim/2]. These are doing
#        rotate half and then concate, rope_vals are the cos, sin.
def _apply_rope_single(x, rope_vals: Tuple[torch.Tensor, torch.Tensor]):
    '''
    Based on FacebookResearch's llama, provided by Carlƒ
    '''
    rope_real = rope_vals[0] # shape should be 1, 1, seqlen, head_dim/2
    rope_im = rope_vals[1] # shape should be 1, 1, seqlen, head_dim/2

    x_real = x[:,:,:,:x.shape[-1]//2] # extract first half elements
    x_im = x[:,:,:,x.shape[-1]//2:] # extract second half elements

    x_prod_real = x_real*rope_real - x_im * rope_im
    x_prod_im = x_real*rope_im + x_im*rope_real

    x = torch.cat((x_prod_real,x_prod_im),dim=3).view(*x.shape)
    return x


## NEXA: We can click into the original model implementation, this function is used to override the 
#        the original forward function. 
def bypass_RotaryEmbedding(self, x, position_ids, *args, **kwargs):
    use_position_embedding_input = self.config.use_position_embedding_input if hasattr(self.config, 'use_position_embedding_input') else False
    if use_position_embedding_input:
        return position_ids
    
    ## NEXA: _original_ prefix is echoed in the update_attr function.
    return self._original_forward(x, position_ids, *args, **kwargs)

class QcAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        # NEXA: The two configs are newly defined in the models. 
        return_new_key_value_only = self.config.return_new_key_value_only if hasattr(self.config, 'return_new_key_value_only') else False
        transposed_key_cache = self.config.transposed_key_cache if hasattr(self.config, 'transposed_key_cache') else False

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            if isinstance(position_ids, (tuple, list)):
                
                ## NEXA: Echo the above comment, _apply_rope_single is just a function to put
                #        positional embedding on the tensors after q and k projection.
                position_embeddings  = position_ids
                cos, sin = position_embeddings
                query_states = _apply_rope_single(query_states, position_embeddings)
                key_states = _apply_rope_single(key_states, position_embeddings)
            else:
                position_embeddings = self.rotary_emb(value_states, position_ids)
                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, position_embeddings, sin)

        ## NEXA: Till here, we have key_states and value_states both as shape
        #        [bsz, num_heads, seq_len, head_dim], this is True in the current setup
        if transposed_key_cache: 
            
            ## NEXA: For QNN, after tranpose, key_states becomes [bsz, num_heads, head_dim, seq_len]
            #        This is helpful for better QNN performance
            key_states = key_states.transpose(2, 3) 

        if past_key_value is not None:
            
            ## NEXA: This modeling code is using a customized DynamicCache class,
            #        it is not the same as the original HF implementation.
            assert isinstance(past_key_value, DynamicCache)
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, 
                        "return_new_key_value_only": return_new_key_value_only,
                        "transposed_key_cache": transposed_key_cache,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if transposed_key_cache:
            attn_weights = torch.matmul(query_states, key_states) / math.sqrt(self.head_dim)
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        ## NEXA: We now consider the 2D attention mask. 
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        ## NEXA: Originally, there is dropout, but this can be safely removed.
        #        According to o3, dropout is definitely not useful for PTQ or QAT.
        # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
            
        ## NEXA: This is important, since we now return the past_key_value.
        return attn_output, attn_weights, past_key_value


    def prepare_conv(self):
        if not hasattr(self, 'forward_no_conv'):
            self.q_proj_conv = nn.Conv2d(self.hidden_size, self.num_heads * self.head_dim, 1, bias=False)
            self.k_proj_conv = nn.Conv2d(self.hidden_size, self.num_key_value_heads * self.head_dim, 1, bias=False)
            self.v_proj_conv = nn.Conv2d(self.hidden_size, self.num_key_value_heads * self.head_dim, 1, bias=False)
            self.o_proj_conv = nn.Conv2d(self.num_heads * self.head_dim, self.hidden_size, 1, bias=False)

            self.forward_no_conv = self.forward
            self.forward = self.forward_conv

            self.q_proj_conv.weight.data.copy_(self.q_proj.weight[:, :, None, None])
            self.k_proj_conv.weight.data.copy_(self.k_proj.weight[:, :, None, None])
            self.v_proj_conv.weight.data.copy_(self.v_proj.weight[:, :, None, None])
            self.o_proj_conv.weight.data.copy_(self.o_proj.weight[:, :, None, None])

            del self.q_proj
            del self.k_proj
            del self.v_proj
            del self.o_proj

    def forward_conv(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        #QC
        return_new_key_value_only = self.config.return_new_key_value_only if hasattr(self.config, 'return_new_key_value_only') else False
        transposed_key_cache = self.config.transposed_key_cache if hasattr(self.config, 'transposed_key_cache') else False

        bsz, q_len, _ = hidden_states.size()

        hidden_states = torch.reshape(hidden_states, (bsz, q_len, 1, self.hidden_size)).transpose(1, 3)

        query_states = self.q_proj_conv(hidden_states)
        key_states = self.k_proj_conv(hidden_states)
        value_states = self.v_proj_conv(hidden_states)

        query_states = query_states.reshape(bsz, self.num_heads, self.head_dim, q_len).transpose(2, 3)
        key_states = key_states.reshape(bsz, self.num_key_value_heads, self.head_dim, q_len).transpose(2, 3)
        value_states = value_states.reshape(bsz, self.num_key_value_heads, self.head_dim, q_len).transpose(2, 3)

        if position_embeddings is None:
            if isinstance(position_ids, (tuple, list)): # QC
                position_embeddings  = position_ids
            else:
                position_embeddings = self.rotary_emb(value_states, position_ids)
        cos, sin = position_embeddings
        query_states = _apply_rope_single(query_states, position_embeddings)
        key_states = _apply_rope_single(key_states, position_embeddings)

        if transposed_key_cache:
            key_states = key_states.transpose(2, 3)

        if past_key_value is not None:
            assert isinstance(past_key_value, DynamicCache)
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, 
                        "return_new_key_value_only": return_new_key_value_only,
                        "transposed_key_cache": transposed_key_cache,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if transposed_key_cache:
            attn_weights = torch.matmul(query_states, key_states) / math.sqrt(self.head_dim)
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, 1, -1)
        attn_output = attn_output.transpose(1, 3)
        attn_output = self.o_proj_conv(attn_output)
        attn_output = attn_output.transpose(1, 3)
        attn_output = attn_output.reshape(bsz, q_len, -1)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def bypass_update_causal_mask(self, attention_mask, *args, **kwargs):
    use_combined_mask_input = self.config.use_combined_mask_input if hasattr(self.config, 'use_combined_mask_input') else False
    if use_combined_mask_input:
        # attention_mask is Causal mask and given as model input
        return attention_mask
    if hasattr(self, '_original_prepare_decoder_attention_mask'):
        return self._prepare_decoder_attention_mask_original(attention_mask, *args, **kwargs)
    return self._original__update_causal_mask(attention_mask, *args, **kwargs)


def LlamaMLP_prepare_conv(self):
    if not hasattr(self, 'forward_linear'):
        self.gate_proj_conv = nn.Conv2d(self.hidden_size, self.intermediate_size, 1, bias=False)
        self.down_proj_conv = nn.Conv2d(self.intermediate_size, self.hidden_size, 1, bias=False)
        self.up_proj_conv = nn.Conv2d(self.hidden_size, self.intermediate_size, 1, bias=False)
        self.forward_linear = self.forward
        self.forward = self.forward_conv

        self.gate_proj_conv.weight.data.copy_(self.gate_proj.weight[:, :, None, None])
        self.down_proj_conv.weight.data.copy_(self.down_proj.weight[:, :, None, None])
        self.up_proj_conv.weight.data.copy_(self.up_proj.weight[:, :, None, None])

        del self.gate_proj
        del self.down_proj
        del self.up_proj

def LlamaMLP_forward_conv(self, x):
    bsz, _, _ = x.size()
    x = torch.reshape(x, (bsz, -1, 1, self.hidden_size))
    x = x.transpose(1,3) # Transpose right before and after Conv
    x = self.down_proj_conv(self.act_fn(self.gate_proj_conv(x)) * self.up_proj_conv(x))
    x = x.transpose(1,3)
    x = torch.reshape(x, (bsz, -1, self.hidden_size))
    return x



def LlamaForCausalLM_prepare_conv(self):
    if not hasattr(self, 'lm_head_conv'):

        def lm_head_conv_forward(x):
            bsz, _, _ = x.size()
            x = torch.reshape(x, (bsz, -1, 1, self.config.hidden_size))
            x = x.transpose(1,3) # Transpose right before and after Conv
            x = self.lm_head_conv(x)
            x = x.transpose(1,3)
            x = torch.reshape(x, (bsz, -1, self.config.vocab_size))
            return x

        self.lm_head_conv = nn.Conv2d(self.config.hidden_size, self.config.vocab_size, 1, bias=False)
        self.lm_head_conv.weight.data.copy_(self.lm_head.weight[:, :, None, None])

        del self.lm_head
        self.lm_head = lm_head_conv_forward



## NEXA: This is used to edit the cache_utils class, and our model is using the DynamicCache class inside
#        the cache_utils class.
def DynamicCache_update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    cache_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Update the number of seen tokens
    # Both now as [bsz, num_heads, seq_len, head_dim]
    if layer_idx == 0:
        self._seen_tokens += value_states.shape[-2]

    # NEXA: self.key_cache is a list, and each list is responsible for a layer.
    if len(self.key_cache) <= layer_idx:
        self.key_cache.append(key_states)
        self.value_cache.append(value_states)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    else:
        return_new_key_value_only = cache_kwargs.get('return_new_key_value_only', False)
        transposed_key_cache = cache_kwargs.get('transposed_key_cache', False)
        key_cat_dim = -1 if transposed_key_cache else -2
        
        
        ## NEXA: We still need the concat but we only return the current key / value states
        key_cache = torch.cat([self.key_cache[layer_idx], key_states], dim=key_cat_dim)
        value_cache = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        if return_new_key_value_only:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = key_cache
            self.value_cache[layer_idx] = key_cache
        return key_cache, value_cache


def DynamicCache_get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
    """Returns the sequence length of the cached states. A layer index can be optionally passed."""
    # TODO: deprecate this function in favor of `cache_position`
    if len(self.value_cache) <= layer_idx:
        return 0
    return self.value_cache[layer_idx].shape[-2]



## NEXA: This update_attr class, especially designed to use _original_attr_name
#        to avoid the nested calling.
def update_attr(cls, attr_name, new_attr):
    attr_backup_name = f'_original_{attr_name}'
    if hasattr(cls, attr_name):
        if not hasattr(cls, attr_backup_name):
            setattr(cls, attr_backup_name, getattr(cls, attr_name))
            setattr(cls, attr_name, new_attr)
        return True
    return False