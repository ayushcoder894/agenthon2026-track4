# Vendored unchanged from nvidia/llama-nemotron-embed-1b-v2 @ 113abe4acafa848e77ead9c0623205e511932348 (Apache-2.0, see below).
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0.
"""
Bidirectional Llama model for embedding tasks.

This module provides a modified LlamaModel that uses bidirectional (non-causal)
attention, suitable for generating embeddings where each token should attend
to all other tokens in the sequence.

Supports transformers version 4.44 and above with a unified forward() implementation.

Version compatibility notes:
    - transformers 4.47: Setting _attn_implementation in __init__ had no effect due to
      attention initialization order
    - transformers 4.48+: Attention refactor (transformers#35235) activated the
      _attn_implementation setting, which defaulted to "eager" instead of "sdpa"
    - transformers < 4.53: LlamaModel has _update_causal_mask method that can be overridden
    - transformers 4.53+: _update_causal_mask removed; masking moved to masking_utils module,
      necessitating a full forward() override for custom attention masks
    - transformers < 4.54: Decoder layer returns tuple, uses past_key_value (singular)
    - transformers 4.54-4.55: Decoder layer returns tensor, uses past_key_value (singular)
    - transformers 4.56+: Decoder layer returns tensor, uses past_key_values (plural),
      DynamicCache accepts config parameter
    - transformers 5.0+: Has native create_bidirectional_mask in masking_utils
"""

import inspect

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaModel
from transformers.utils import logging

logger = logging.get_logger(__name__)

# Check if native create_bidirectional_mask exists (transformers >= 5.0)
try:
    from transformers.masking_utils import create_bidirectional_mask

    _HAS_NATIVE_BIDIRECTIONAL_MASK = True
except ImportError:
    from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

    _HAS_NATIVE_BIDIRECTIONAL_MASK = False

# Detect API differences via introspection
_decoder_forward_params = inspect.signature(LlamaDecoderLayer.forward).parameters
_dynamic_cache_init_params = inspect.signature(DynamicCache.__init__).parameters

# past_key_value (singular) in < 4.56, past_key_values (plural) in >= 4.56
_USE_PLURAL_CACHE_PARAM = "past_key_values" in _decoder_forward_params
# DynamicCache accepts config parameter in >= 4.56
_DYNAMIC_CACHE_ACCEPTS_CONFIG = "config" in _dynamic_cache_init_params


class LlamaBidirectionalConfig(LlamaConfig):
    """Configuration for LlamaBidirectionalModel with pooling and temperature settings."""

    model_type = "llama_bidirec"

    def __init__(
        self, pooling: str = "avg", temperature: float = 1.0, **kwargs
    ) -> None:
        """
        Initialize bidirectional Llama configuration.

        Args:
            pooling: Pooling strategy for embeddings ("avg", "cls", "last", etc.)
            temperature: Temperature scaling for embeddings
            **kwargs: Additional arguments passed to LlamaConfig
        """
        self.pooling = pooling
        self.temperature = temperature
        super().__init__(**kwargs)


class LlamaBidirectionalModel(LlamaModel):
    """
    LlamaModel modified to use bidirectional (non-causal) attention.

    In standard Llama, each token can only attend to previous tokens (causal attention).
    This model removes that restriction, allowing each token to attend to all tokens
    in the sequence, which is useful for embedding tasks.

    The key modifications are:
        1. Setting is_causal=False on all attention layers
        2. Using a bidirectional attention mask instead of causal mask
    """

    config_class = LlamaBidirectionalConfig

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        for layer in self.layers:
            layer.self_attn.is_causal = False

    def _create_bidirectional_mask(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """
        Create bidirectional attention mask.

        Args:
            input_embeds: Input embeddings tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional 2D attention mask of shape (batch_size, seq_len)
                where 1 indicates tokens to attend to and 0 indicates masked tokens

        Returns:
            4D attention mask suitable for the attention implementation, or None
            if no masking is needed
        """
        if attention_mask is None:
            return None

        if _HAS_NATIVE_BIDIRECTIONAL_MASK:
            return create_bidirectional_mask(
                self.config,
                input_embeds,
                attention_mask=attention_mask,
            )

        # Fallback for transformers < 5.0 without create_bidirectional_mask

        # Flash attention handles 2D masks internally; only pass mask if there
        # are actually masked tokens (zeros), otherwise return None for efficiency
        if getattr(self.config, "_attn_implementation", None) == "flash_attention_2":
            has_masked_tokens = (attention_mask == 0).any()
            return attention_mask if has_masked_tokens else None

        return _prepare_4d_attention_mask(attention_mask, input_embeds.dtype)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """
        Forward pass with bidirectional attention.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            position_ids: Position IDs for rotary embeddings
            past_key_values: Cached key/value states for incremental decoding
            inputs_embeds: Pre-computed input embeddings (alternative to input_ids)
            cache_position: Position indices for cache updates
            use_cache: Whether to return cached key/value states
            **kwargs: Additional arguments passed to decoder layers

        Returns:
            BaseModelOutputWithPast containing last_hidden_state and past_key_values
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Initialize cache if needed
        if use_cache and past_key_values is None:
            if _DYNAMIC_CACHE_ACCEPTS_CONFIG:
                past_key_values = DynamicCache(config=self.config)
            else:
                past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        bidirectional_mask = self._create_bidirectional_mask(
            inputs_embeds, attention_mask
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Build decoder layer kwargs with correct cache parameter name
        # (past_key_value in < 4.56, past_key_values in >= 4.56)
        layer_kwargs = {
            "attention_mask": bidirectional_mask,
            "position_ids": position_ids,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
        }
        if _USE_PLURAL_CACHE_PARAM:
            layer_kwargs["past_key_values"] = past_key_values
        else:
            layer_kwargs["past_key_value"] = past_key_values

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_outputs = decoder_layer(hidden_states, **layer_kwargs)

            # Decoder returns tuple in < 4.54, tensor in >= 4.54
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
