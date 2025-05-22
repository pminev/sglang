# Copyright 2025 Plamen Minev
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/llama.py#L1
"""Inference-only LLaMA model compatible with HuggingFace weights."""

import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
import torchaudio
import torch.nn as nn
import numpy as np

from sglang.srt.configs import DiaConfig

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
# from sglang.srt.layers.activation import SiluAndMul
# from sglang.srt.layers.layernorm import RMSNorm
# from sglang.srt.layers.linear import (
#     MergedColumnParallelLinear,
#     QKVParallelLinear,
#     RowParallelLinear,
# )
# from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
# from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
# from sglang.srt.layers.radix_attention import RadixAttention
# from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    # ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    kv_cache_scales_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.utils import add_prefix, make_layers
from sglang.utils import get_exception_traceback

import torch.nn.functional as F

from dataclasses import dataclass
import typing as tp

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 44100

def build_delay_indices(B: int, T: int, C: int, delay_pattern: tp.List[int]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute (t_idx_BxTxC, indices_BTCx3) so that out[t, c] = in[t - delay[c], c].
    Negative t_idx => BOS; t_idx >= T => PAD.
    """
    delay_arr = torch.tensor(delay_pattern, dtype=torch.int32)

    t_idx_BxT = torch.broadcast_to(
        torch.arange(T, dtype=torch.int32)[None, :],
        [B, T],
    )
    t_idx_BxTx1 = t_idx_BxT[..., None]
    t_idx_BxTxC = t_idx_BxTx1 - delay_arr.view(1, 1, C)

    b_idx_BxTxC = torch.broadcast_to(
        torch.arange(B, dtype=torch.int32).view(B, 1, 1),
        [B, T, C],
    )
    c_idx_BxTxC = torch.broadcast_to(
        torch.arange(C, dtype=torch.int32).view(1, 1, C),
        [B, T, C],
    )

    # We must clamp time indices to [0..T-1] so gather_nd equivalent won't fail
    t_clamped_BxTxC = torch.clamp(t_idx_BxTxC, 0, T - 1)

    indices_BTCx3 = torch.stack(
        [
            b_idx_BxTxC.reshape(-1),
            t_clamped_BxTxC.reshape(-1),
            c_idx_BxTxC.reshape(-1),
        ],
        dim=1,
    ).long()  # Ensure indices are long type for indexing

    return t_idx_BxTxC, indices_BTCx3


def apply_audio_delay(
    audio_BxTxC: torch.Tensor,
    pad_value: int,
    bos_value: int,
    precomp: tp.Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    Applies the delay pattern to batched audio tokens using precomputed indices,
    inserting BOS where t_idx < 0 and PAD where t_idx >= T.

    Args:
        audio_BxTxC: [B, T, C] int16 audio tokens (or int32/float)
        pad_value: the padding token
        bos_value: the BOS token
        precomp:  (t_idx_BxTxC, indices_BTCx3) from build_delay_indices

    Returns:
        result_BxTxC: [B, T, C] delayed audio tokens
    """
    device = audio_BxTxC.device  # Get device from input tensor
    t_idx_BxTxC, indices_BTCx3 = precomp
    t_idx_BxTxC = t_idx_BxTxC.to(device)  # Move precomputed indices to device
    indices_BTCx3 = indices_BTCx3.to(device)

    # Equivalent of tf.gather_nd using advanced indexing
    # Ensure indices are long type if not already (build_delay_indices should handle this)
    gathered_flat = audio_BxTxC[indices_BTCx3[:, 0], indices_BTCx3[:, 1], indices_BTCx3[:, 2]]
    gathered_BxTxC = gathered_flat.view(audio_BxTxC.shape)

    # Create masks on the correct device
    mask_bos = t_idx_BxTxC < 0  # => place bos_value
    mask_pad = t_idx_BxTxC >= audio_BxTxC.shape[1]  # => place pad_value

    # Create scalar tensors on the correct device
    bos_tensor = torch.tensor(bos_value, dtype=audio_BxTxC.dtype, device=device)
    pad_tensor = torch.tensor(pad_value, dtype=audio_BxTxC.dtype, device=device)

    # If mask_bos, BOS; else if mask_pad, PAD; else original gather
    # All tensors should now be on the same device
    result_BxTxC = torch.where(mask_bos, bos_tensor, torch.where(mask_pad, pad_tensor, gathered_BxTxC))

    return result_BxTxC


def build_revert_indices(B: int, T: int, C: int, delay_pattern: tp.List[int]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute indices for the revert operation using PyTorch.

    Returns:
        A tuple (t_idx_BxTxC, indices_BTCx3) where:
            - t_idx_BxTxC is a tensor of shape [B, T, C] computed as time indices plus the delay.
            - indices_BTCx3 is a tensor of shape [B*T*C, 3] used for gathering, computed from:
                batch indices, clamped time indices, and channel indices.
    """
    # Use default device unless specified otherwise; assumes inputs might define device later
    device = None  # Or determine dynamically if needed, e.g., from a model parameter

    delay_arr = torch.tensor(delay_pattern, dtype=torch.int32, device=device)

    t_idx_BT1 = torch.broadcast_to(torch.arange(T, device=device).unsqueeze(0), [B, T])
    t_idx_BT1 = t_idx_BT1.unsqueeze(-1)

    t_idx_BxTxC = torch.minimum(
        t_idx_BT1 + delay_arr.view(1, 1, C),
        torch.tensor(T - 1, device=device),
    )
    b_idx_BxTxC = torch.broadcast_to(torch.arange(B, device=device).view(B, 1, 1), [B, T, C])
    c_idx_BxTxC = torch.broadcast_to(torch.arange(C, device=device).view(1, 1, C), [B, T, C])

    indices_BTCx3 = torch.stack(
        [
            b_idx_BxTxC.reshape(-1),
            t_idx_BxTxC.reshape(-1),
            c_idx_BxTxC.reshape(-1),
        ],
        axis=1,
    ).long()  # Ensure indices are long type

    return t_idx_BxTxC, indices_BTCx3


def revert_audio_delay(
    audio_BxTxC: torch.Tensor,
    pad_value: int,
    precomp: tp.Tuple[torch.Tensor, torch.Tensor],
    T: int,
) -> torch.Tensor:
    """
    Reverts a delay pattern from batched audio tokens using precomputed indices (PyTorch version).

    Args:
        audio_BxTxC: Input delayed audio tensor
        pad_value: Padding value for out-of-bounds indices
        precomp: Precomputed revert indices tuple containing:
            - t_idx_BxTxC: Time offset indices tensor
            - indices_BTCx3: Gather indices tensor for original audio
        T: Original sequence length before padding

    Returns:
        Reverted audio tensor with same shape as input
    """
    t_idx_BxTxC, indices_BTCx3 = precomp
    device = audio_BxTxC.device  # Get device from input tensor

    # Move precomputed indices to the same device as audio_BxTxC if they aren't already
    t_idx_BxTxC = t_idx_BxTxC.to(device)
    indices_BTCx3 = indices_BTCx3.to(device)

    # Using PyTorch advanced indexing (equivalent to tf.gather_nd or np equivalent)
    gathered_flat = audio_BxTxC[indices_BTCx3[:, 0], indices_BTCx3[:, 1], indices_BTCx3[:, 2]]
    gathered_BxTxC = gathered_flat.view(audio_BxTxC.size())  # Use .size() for robust reshaping

    # Create pad_tensor on the correct device
    pad_tensor = torch.tensor(pad_value, dtype=audio_BxTxC.dtype, device=device)
    # Create T tensor on the correct device for comparison
    T_tensor = torch.tensor(T, device=device)

    result_BxTxC = torch.where(t_idx_BxTxC >= T_tensor, pad_tensor, gathered_BxTxC)  # Changed np.where to torch.where

    return result_BxTxC


@torch.no_grad()
@torch.inference_mode()
def decode(
    model,
    audio_codes,
):
    """
    Decodes the given frames into an output audio waveform
    """
    if len(audio_codes) != 1:
        raise ValueError(f"Expected one frame, got {len(audio_codes)}")

    try:
        audio_values = model.quantizer.from_codes(audio_codes)
        audio_values = model.decode(audio_values[0])

        return audio_values
    except Exception as e:
        print(f"Error in decode method: {str(e)}")
        raise


def create_attn_mask(
    q_padding_mask_1d: torch.Tensor,
    k_padding_mask_1d: torch.Tensor,
    device: torch.device,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Creates the attention mask (self or cross) mimicking JAX segment ID logic.
    """
    B1, Tq = q_padding_mask_1d.shape
    B2, Tk = k_padding_mask_1d.shape
    assert B1 == B2, "Query and key batch dimensions must match"

    p_mask_q = q_padding_mask_1d.unsqueeze(2)  # Shape [B, Tq, 1]
    p_mask_k = k_padding_mask_1d.unsqueeze(1)  # Shape [B, 1, Tk]

    # Condition A: Non-padding query attends to non-padding key
    non_pad_attends_non_pad = p_mask_q & p_mask_k  # Shape [B, Tq, Tk]

    # Condition B: Padding query attends to padding key
    pad_attends_pad = (~p_mask_q) & (~p_mask_k)  # Shape [B, Tq, Tk]

    # Combine: True if padding status is compatible (both non-pad OR both pad)
    mask = non_pad_attends_non_pad | pad_attends_pad  # Shape [B, Tq, Tk]

    if is_causal:
        assert Tq == Tk, "Causal mask requires query and key sequence lengths to be equal"
        causal_mask_2d = torch.tril(torch.ones((Tq, Tk), dtype=torch.bool, device=device))  # Shape [Tq, Tk]
        causal_mask = mask & causal_mask_2d  # Shape [B, Tq, Tk]
        return causal_mask.unsqueeze(1)  # Shape [B, 1, Tq, Tk]
    else:
        return mask.unsqueeze(1)  # Shape [B, 1, Tq, Tk]


@dataclass
class EncoderInferenceState:
    """Parameters specifically for encoder inference."""

    max_seq_len: int
    device: torch.device
    positions: torch.Tensor
    padding_mask: torch.Tensor
    attn_mask: torch.Tensor

    @classmethod
    def new(cls, config: DiaConfig, cond_src: torch.Tensor) -> "EncoderInferenceState":
        """Creates EtorchrInferenceParams from DiaConfig and a device."""
        device = cond_src.device

        positions = torch.arange(config.data.text_length, device=device).to(torch.long).unsqueeze(0).expand(2, -1)
        padding_mask = (cond_src != config.data.text_pad_value).to(device).expand(2, -1)
        attn_mask = create_attn_mask(padding_mask, padding_mask, device, is_causal=False)

        return cls(
            max_seq_len=config.data.text_length,
            device=device,
            positions=positions,
            padding_mask=padding_mask,
            attn_mask=attn_mask,
        )


class KVCache:
    def __init__(
        self,
        num_heads: int,
        max_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
    ):
        self.k = torch.zeros((2, num_heads, max_len, head_dim), dtype=dtype, device=device) if k is None else k
        self.v = torch.zeros((2, num_heads, max_len, head_dim), dtype=dtype, device=device) if v is None else v
        self.current_idx = torch.tensor(0)

    @classmethod
    def from_kv(cls, k: torch.Tensor, v: torch.Tensor) -> "KVCache":
        return cls(
            num_heads=k.shape[1],
            max_len=k.shape[2],
            head_dim=k.shape[3],
            dtype=k.dtype,
            device=k.device,
            k=k,
            v=v,
        )

    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.k[:, :, self.current_idx : self.current_idx + 1, :] = k
        self.v[:, :, self.current_idx : self.current_idx + 1, :] = v
        self.current_idx += 1
        return self.k[:, :, : self.current_idx, :], self.v[:, :, : self.current_idx, :]

    def prefill(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prefill_len = k.shape[2]
        self.k[:, :, :prefill_len, :] = k
        self.v[:, :, :prefill_len, :] = v
        self.current_idx = prefill_len - 1


@dataclass
class DecoderInferenceState:
    """Parameters specifically for decoder inference."""

    device: torch.device
    dtype: torch.dtype
    enc_out: torch.Tensor
    enc_positions: torch.Tensor
    dec_positions: torch.Tensor
    dec_cross_attn_mask: torch.Tensor
    self_attn_cache: list[KVCache]
    cross_attn_cache: list[KVCache]

    @classmethod
    def new(
        cls,
        config: DiaConfig,
        enc_state: EncoderInferenceState,
        enc_out: torch.Tensor,
        dec_cross_attn_cache: list[KVCache],
        compute_dtype: torch.dtype,
    ) -> "DecoderInferenceState":
        """Creates DecoderInferenceParams from DiaConfig and a device."""
        device = enc_out.device
        max_audio_len = config.data.audio_length

        dec_positions = torch.full((2, 1), fill_value=0, dtype=torch.long, device=device)
        tgt_padding_mask = torch.ones((2, 1), dtype=torch.bool, device=device)
        dec_cross_attn_mask = create_attn_mask(tgt_padding_mask, enc_state.padding_mask, device, is_causal=False)

        self_attn_cache = [
            KVCache(
                config.model.decoder.kv_heads,
                max_audio_len,
                config.model.decoder.gqa_head_dim,
                compute_dtype,
                device,
            )
            for _ in range(config.model.decoder.n_layer)
        ]

        return cls(
            device=device,
            dtype=compute_dtype,
            enc_out=enc_out,
            enc_positions=enc_state.positions,
            dec_positions=dec_positions,
            dec_cross_attn_mask=dec_cross_attn_mask,
            self_attn_cache=self_attn_cache,
            cross_attn_cache=dec_cross_attn_cache,
        )

    def prepare_step(self, step_from: int, step_to: int | None = None) -> None:
        if step_to is None:
            step_to = step_from + 1
        self.dec_positions = torch.arange(step_from, step_to, device=self.device).unsqueeze(0).expand(2, -1)


@dataclass
class DecoderOutput:
    generated_tokens: torch.Tensor
    prefill_step: int

    @classmethod
    def new(cls, config: DiaConfig, device: torch.device) -> "DecoderOutput":
        max_audio_len = config.data.audio_length
        return cls(
            generated_tokens=torch.full(
                (max_audio_len, config.data.channels),
                fill_value=-1,
                dtype=torch.int,
                device=device,
            ),
            prefill_step=0,
        )

    def get_tokens_at(self, step_from: int, step_to: int | None = None) -> torch.Tensor:
        if step_to is None:
            step_to = step_from + 1
        return self.generated_tokens[step_from:step_to, :]

    def update_one(self, dec_out: torch.Tensor, step: int, apply_mask: bool = False):
        if apply_mask:
            mask = self.generated_tokens[step : step + 1, :] == -1
            self.generated_tokens[step : step + 1, :] = torch.where(
                mask, dec_out, self.generated_tokens[step : step + 1, :]
            )
        else:
            self.generated_tokens[step : step + 1, :] = dec_out

    def prefill(self, dec_out: torch.Tensor, prefill_step: int):
        length = dec_out.shape[0]
        self.generated_tokens[0:length, :] = dec_out
        self.prefill_step = prefill_step


def _normalize_axes(axes: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


class DenseGeneral(nn.Module):
    """
    PyTorch equivalent of flax.linen.DenseGeneral with shapes defined at init.

    Stores weights (`kernel`) in the same layout as Jax and uses torch.tensordot
    for the generalized matrix multiplication. Weight/bias shapes are calculated
    and parameters created during initialization based on config.
    `load_weights` validates shapes and copies data.

    Attributes:
        axis (Tuple[int, ...]): Input axis or axes to contract.
        in_shapes (Tuple[int, ...]): Sizes of the input dimensions specified by `axis`.
        out_features (Tuple[int, ...]): Shape of the output features (non-contracted dims).
        use_bias (bool): Whether to add a bias term.
        weight (nn.Parameter): The kernel parameter.
        bias (Optional[nn.Parameter]): The bias parameter (if use_bias=True).
    """

    def __init__(
        self,
        in_shapes: tuple[int, ...],
        out_features: tuple[int, ...],
        axis: tuple[int, ...] = (-1,),
        weight_dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.in_shapes = in_shapes
        self.out_features = out_features
        self.axis = axis
        self.kernel_shape = self.in_shapes + self.out_features

        factory_kwargs = {"device": device, "dtype": weight_dtype}
        self.weight = nn.Parameter(torch.empty(self.kernel_shape, **factory_kwargs))
        self.register_parameter("bias", None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        norm_axis = _normalize_axes(self.axis, inputs.ndim)
        kernel_contract_axes = tuple(range(len(norm_axis)))

        output = torch.tensordot(
            inputs.to(self.weight.dtype),
            self.weight,
            dims=(norm_axis, kernel_contract_axes),
        ).to(inputs.dtype)
        return output


class MlpBlock(nn.Module):
    """MLP block using DenseGeneral."""

    def __init__(self, embed_dim: int, intermediate_dim: int, compute_dtype: torch.dtype):
        super().__init__()
        self.dtype = compute_dtype

        self.wi_fused = DenseGeneral(
            in_shapes=(embed_dim,),
            out_features=(2, intermediate_dim),
            axis=(-1,),
            weight_dtype=compute_dtype,
        )

        self.wo = DenseGeneral(
            in_shapes=(intermediate_dim,),
            out_features=(embed_dim,),
            axis=(-1,),
            weight_dtype=compute_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        fused_x = self.wi_fused(x)

        gate = fused_x[..., 0, :]
        up = fused_x[..., 1, :]

        hidden = torch.mul(F.silu(gate), up).to(self.dtype)

        output = self.wo(hidden)
        return output


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation in PyTorch."""

    def __init__(
        self,
        embedding_dims: int,
        min_timescale: int = 1,
        max_timescale: int = 10000,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if embedding_dims % 2 != 0:
            raise ValueError("Embedding dim must be even for RoPE.")
        self.embedding_dims = embedding_dims
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.dtype = dtype

        half_embedding_dim = embedding_dims // 2
        fraction = (2.0 * torch.arange(0, half_embedding_dim)).to(torch.float32) / embedding_dims
        self.register_buffer(
            "timescale",
            self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction,
            persistent=False,
        )

    def extra_repr(self) -> str:
        s = f"{self.timescale.shape}"
        return s

    def forward(self, inputs: torch.Tensor, position: torch.Tensor):
        """Applies RoPE."""
        position = position.unsqueeze(-1).unsqueeze(-1)
        timescale = self.timescale.to(inputs.device)
        sinusoid_inp = position / timescale
        sin = torch.sin(sinusoid_inp).to(inputs.dtype)
        cos = torch.cos(sinusoid_inp).to(inputs.dtype)
        first_half, second_half = torch.chunk(inputs, 2, dim=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        return torch.cat((first_part, second_part), dim=-1)


class Attention(nn.Module):
    """Attention using DenseGeneral."""

    def __init__(
        self,
        config: DiaConfig,
        q_embed_dim: int,
        kv_embed_dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        compute_dtype: torch.dtype,
        is_cross_attn: bool = False,
        out_embed_dim: int | None = None,
    ):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.is_cross_attn = is_cross_attn
        self.output_dim = out_embed_dim if out_embed_dim is not None else q_embed_dim
        self.projected_query_dim = num_query_heads * head_dim
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(f"num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
        self.num_gqa_groups = num_query_heads // num_kv_heads

        # --- Projection Layers using DenseGeneral ---
        self.q_proj = DenseGeneral(
            in_shapes=(q_embed_dim,),
            out_features=(num_query_heads, head_dim),
            axis=(-1,),
            weight_dtype=compute_dtype,
        )
        self.k_proj = DenseGeneral(
            in_shapes=(kv_embed_dim,),
            out_features=(num_kv_heads, head_dim),
            axis=(-1,),
            weight_dtype=compute_dtype,
        )
        self.v_proj = DenseGeneral(
            in_shapes=(kv_embed_dim,),
            out_features=(num_kv_heads, head_dim),
            axis=(-1,),
            weight_dtype=compute_dtype,
        )
        self.o_proj = DenseGeneral(
            in_shapes=(num_query_heads, head_dim),
            out_features=(self.output_dim,),
            axis=(-2, -1),
            weight_dtype=compute_dtype,
        )

        # --- Rotary Embedding ---
        self.rotary_emb = RotaryEmbedding(
            embedding_dims=self.head_dim,
            min_timescale=config.model.rope_min_timescale,
            max_timescale=config.model.rope_max_timescale,
            dtype=compute_dtype,
        )

    def forward(
        self,
        Xq: torch.Tensor,  # (B, T, D) T = 1 in AR generation
        Xkv: torch.Tensor,  # (B, S, E) S = 1 in AR generation
        q_positions: torch.Tensor,  # (B, T)
        kv_positions: torch.Tensor | None = None,  # (B, S)
        attn_mask: torch.Tensor | None = None,  # None in Decoder Self Attention, Valid mask in Others
        cache: KVCache | None = None,  # None in Encoder, KVCache in Decoder
        prefill: bool = False,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Performs attention calculation with optional KV caching.

        Args:
            Xq: Query tensor (B, T, D). T=1 during single-step decoding.
            Xkv: Key/Value source tensor (B, S, E). S=1 during single-step decoding for self-attn.
            q_positions: Positions for queries (B, T).
            kv_positions: Positions for keys/values (B, S). If None, uses q_positions.
            attn_mask: Attention mask.
            cache: KVCache.
            prefill: If True, use prefill mode.

        Returns:
            A tuple containing:
            - output: The attention output tensor (B, T, output_dim).
            - present_kv: The K/V state to be cached for the next step ((B, N, S_new, H), (B, N, S_new, H)). For self-attn, S_new = S_past + S. For cross-attn, S_new = S_kv.
        """
        if kv_positions is None:
            kv_positions = q_positions
        original_dtype = Xq.dtype

        Xq_BxTxNxH = self.q_proj(Xq)
        Xq_BxTxNxH = self.rotary_emb(Xq_BxTxNxH, position=q_positions)
        Xq_BxNxTxH = Xq_BxTxNxH.transpose(1, 2)

        attn_k: torch.Tensor | None = None
        attn_v: torch.Tensor | None = None

        if self.is_cross_attn:
            attn_k, attn_v = cache.k, cache.v
        else:
            Xk_BxSxKxH = self.k_proj(Xkv)  # (B, S, K, H)
            Xv_BxSxKxH = self.v_proj(Xkv)  # (B, S, K, H)
            Xk_BxSxKxH = self.rotary_emb(Xk_BxSxKxH, position=kv_positions)  # (B, S, K, H)

            Xk_BxKxSxH = Xk_BxSxKxH.transpose(1, 2)  # (B, K, S, H)
            Xv_BxKxSxH = Xv_BxSxKxH.transpose(1, 2)  # (B, K, S, H)

            if cache is None:
                attn_k = Xk_BxKxSxH
                attn_v = Xv_BxKxSxH
            else:
                if prefill:
                    attn_k, attn_v = Xk_BxKxSxH, Xv_BxKxSxH
                    cache.prefill(attn_k, attn_v)
                else:
                    attn_k, attn_v = cache.update(Xk_BxKxSxH, Xv_BxKxSxH)

        attn_output = F.scaled_dot_product_attention(
            Xq_BxNxTxH,
            attn_k,
            attn_v,
            attn_mask=attn_mask,
            scale=1.0,
            enable_gqa=self.num_gqa_groups > 1,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, T, N, H)
        output = self.o_proj(attn_output)

        return output.to(original_dtype)


class EncoderLayer(nn.Module):
    """Transformer Encoder Layer using DenseGeneral."""

    def __init__(self, config: DiaConfig, compute_dtype: torch.dtype):
        super().__init__()
        self.config = config
        model_config = config.model
        enc_config = config.model.encoder
        embed_dim = enc_config.n_embd

        self.pre_sa_norm = nn.RMSNorm(
            embed_dim,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float16,
        )
        self.self_attention = Attention(
            config,
            q_embed_dim=embed_dim,
            kv_embed_dim=embed_dim,
            num_query_heads=enc_config.n_head,
            num_kv_heads=enc_config.n_head,
            head_dim=enc_config.head_dim,
            compute_dtype=compute_dtype,
            is_cross_attn=False,
            out_embed_dim=embed_dim,
        )
        self.post_sa_norm = nn.RMSNorm(
            embed_dim,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float16,
        )
        self.mlp = MlpBlock(
            embed_dim=embed_dim,
            intermediate_dim=enc_config.n_hidden,
            compute_dtype=compute_dtype)

    def forward(
        self,
        x: torch.Tensor,
        state: EncoderInferenceState,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.pre_sa_norm(x)
        sa_out = self.self_attention(
            Xq=x_norm,
            Xkv=x_norm,
            q_positions=state.positions,
            kv_positions=state.positions,
            attn_mask=state.attn_mask,
        )
        x = residual + sa_out

        residual = x
        x_norm = self.post_sa_norm(x)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out

        return x


class Encoder(nn.Module):
    """Transformer Encoder Stack using DenseGeneral."""

    def __init__(self, config: DiaConfig, compute_dtype: torch.dtype):
        super().__init__()
        self.config = config
        model_config = config.model
        enc_config = config.model.encoder

        # self.embedding = nn.Embedding(
        #     model_config.src_vocab_size,
        #     enc_config.n_embd,
        #     dtype=compute_dtype,
        # )
        self.embedding = VocabParallelEmbedding(
            model_config.src_vocab_size,
            enc_config.n_embd
        )
        # self.layers = nn.ModuleList([EncoderLayer(config, compute_dtype) for _ in range(enc_config.n_layer)])
        self.layers = make_layers(
            enc_config.n_layer,
            lambda idx, prefix: EncoderLayer(config, compute_dtype),
            prefix="encoder.layers"
        )
        self.norm = nn.RMSNorm(
            enc_config.n_embd,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float16,
        )

    def forward(
        self,
        x_ids: torch.Tensor,
        state: EncoderInferenceState,
    ) -> torch.Tensor:
        x = self.embedding(x_ids)

        for layer in self.layers:
            x = layer(x, state)

        x = self.norm(x)
        return x


class DecoderLayer(nn.Module):
    """Transformer Decoder Layer using DenseGeneral."""

    def __init__(self, config: DiaConfig, compute_dtype: torch.dtype):
        super().__init__()
        self.config = config
        model_config = config.model
        dec_config = config.model.decoder
        enc_config = config.model.encoder
        dec_embed_dim = dec_config.n_embd
        enc_embed_dim = enc_config.n_embd

        # Norms
        self.pre_sa_norm = nn.RMSNorm(
            dec_embed_dim,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float16,
        )
        self.pre_ca_norm = nn.RMSNorm(
            dec_embed_dim,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float16,
        )
        self.pre_mlp_norm = nn.RMSNorm(
            dec_embed_dim,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float16,
        )

        # Self-Attention (GQA) with Causal Masking
        self.self_attention = Attention(
            config,
            q_embed_dim=dec_embed_dim,
            kv_embed_dim=dec_embed_dim,
            num_query_heads=dec_config.gqa_query_heads,
            num_kv_heads=dec_config.kv_heads,
            head_dim=dec_config.gqa_head_dim,
            compute_dtype=compute_dtype,
            is_cross_attn=False,
            out_embed_dim=dec_embed_dim,
        )
        # Cross-Attention (MHA)
        self.cross_attention = Attention(
            config=config,
            q_embed_dim=dec_embed_dim,
            kv_embed_dim=enc_embed_dim,  # Note kv_embed_dim
            num_query_heads=dec_config.cross_query_heads,
            num_kv_heads=dec_config.cross_query_heads,
            head_dim=dec_config.cross_head_dim,
            compute_dtype=compute_dtype,
            is_cross_attn=True,
            out_embed_dim=dec_embed_dim,
        )
        # MLP
        self.mlp = MlpBlock(
            embed_dim=dec_embed_dim,
            intermediate_dim=dec_config.n_hidden,
            compute_dtype=compute_dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: DecoderInferenceState,
        self_attn_cache: KVCache | None = None,
        cross_attn_cache: KVCache | None = None,
        prefill: bool = False,
    ) -> torch.Tensor:
        residual = x

        x_norm = self.pre_sa_norm(x)

        sa_out = self.self_attention(
            Xq=x_norm,  # (2, 1, D)
            Xkv=x_norm,  # (2, 1, D)
            q_positions=state.dec_positions,  # (2, 1)
            kv_positions=state.dec_positions,  # (2, 1)
            attn_mask=None,
            cache=self_attn_cache,
            prefill=prefill,
            is_causal=prefill,
        )

        x = residual + sa_out

        residual = x
        x_norm = self.pre_ca_norm(x)
        ca_out = self.cross_attention(
            Xq=x_norm,
            Xkv=state.enc_out,
            q_positions=state.dec_positions,
            kv_positions=state.enc_positions,
            attn_mask=state.dec_cross_attn_mask,
            cache=cross_attn_cache,
        )
        x = residual + ca_out

        residual = x
        x_norm = self.pre_mlp_norm(x)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out

        return x


class Decoder(nn.Module):
    """Transformer Decoder Stack using DenseGeneral."""

    def __init__(self, config: DiaConfig, quant_config: QuantizationConfig, compute_dtype: torch.dtype):
        super().__init__()
        self.config = config
        model_config = config.model
        dec_config = config.model.decoder
        data_config = config.data
        self.num_channels = data_config.channels
        self.num_layers = dec_config.n_layer

        # self.embeddings = nn.ModuleList(
        #     [
        #         nn.Embedding(model_config.tgt_vocab_size, dec_config.n_embd, dtype=compute_dtype)
        #         for _ in range(self.num_channels)
        #     ]
        # )
        self.embeddings = make_layers(
            self.num_channels,
            lambda idx, prefix: VocabParallelEmbedding(
                model_config.tgt_vocab_size,
                dec_config.n_embd,
            ),
            prefix="decoder.embeddings",
        )

        # self.layers = nn.ModuleList(
        #     [DecoderLayer(config=config, compute_dtype=compute_dtype) for _ in range(self.num_layers)]
        # )
        self.layers = make_layers(
            self.num_layers,
            lambda idx, prefix: DecoderLayer(config=config, compute_dtype=compute_dtype),
            prefix="decoder.layers"
        )

        self.norm = nn.RMSNorm(
            dec_config.n_embd,
            eps=model_config.normalization_layer_epsilon,
        )

        self.logits_dense = DenseGeneral(
            in_shapes=(dec_config.n_embd,),
            out_features=(self.num_channels, model_config.tgt_vocab_size),
            axis=(-1,),
            weight_dtype=compute_dtype,
        )

    def precompute_cross_attn_cache(
        self,
        enc_out: torch.Tensor,  # (B, S, E)
        enc_positions: torch.Tensor,  # (B, S)
    ) -> list[KVCache]:
        """
        Computes the Key and Value tensors for cross-attention for each layer from the encoder output.
        """
        per_layer_kv_cache: list[KVCache] = []

        for layer in self.layers:
            cross_attn_module = layer.cross_attention
            k_proj = cross_attn_module.k_proj(enc_out)
            v_proj = cross_attn_module.v_proj(enc_out)

            k_proj = cross_attn_module.rotary_emb(k_proj, position=enc_positions)
            k = k_proj.transpose(1, 2)
            v = v_proj.transpose(1, 2)

            per_layer_kv_cache.append(KVCache.from_kv(k, v))

        return per_layer_kv_cache

    def decode_step(
        self,
        tgt_ids_Bx1xC: torch.Tensor,  # [B, 1, C]
        state: DecoderInferenceState,
    ) -> torch.Tensor:
        """
        Performs a single decoding step, managing KV caches layer by layer.

        Returns:
            A tuple containing:
            - logits_Bx1xCV: The final output logits for the current step (B, 1, C*V), cast to float32.
        """

        x = None
        for i in range(self.num_channels):
            channel_tokens = tgt_ids_Bx1xC[..., i]
            channel_embed = self.embeddings[i](channel_tokens)
            x = channel_embed if x is None else x + channel_embed

        for i, layer in enumerate(self.layers):
            self_cache = state.self_attn_cache[i]
            cross_cache = state.cross_attn_cache[i]
            x = layer(
                x,  # (2, 1, D)
                state,
                self_attn_cache=self_cache,
                cross_attn_cache=cross_cache,
            )

        x = self.norm(x)
        logits_Bx1xCxV = self.logits_dense(x)

        return logits_Bx1xCxV.to(torch.float16)

    def forward(self, tgt_ids_BxTxC: torch.Tensor, state: DecoderInferenceState) -> torch.Tensor:
        """
        Forward pass for the Decoder stack, managing KV caches.

        Args:
            tgt_ids_BxTxC: Target token IDs (B, T, C).
            encoder_out: Output from the encoder (B, S, E).
            tgt_positions: Positions for target sequence (B, T).
            src_positions: Positions for source sequence (B, S).
            self_attn_mask: Mask for self-attention.
            cross_attn_mask: Mask for cross-attention.
            past_key_values: List containing the self-attention KV cache for each layer
                             from the previous decoding step. `len(past_key_values)` should
                             equal `num_layers`.
            precomputed_cross_attn_kv: A single tuple containing the pre-computed K/V cache
                                      derived from `encoder_out`. This is passed identically
                                      to all layers.

        Returns:
            A tuple containing:
            - logits: The final output logits (B, T, C * V), cast to float32.
            - present_key_values: A list containing the updated self-attention KV cache
                                 for each layer for the *current* decoding step.
        """
        _, _, num_channels_in = tgt_ids_BxTxC.shape
        assert num_channels_in == self.num_channels, "Input channels mismatch"

        # Embeddings
        x = None
        for i in range(self.num_channels):
            channel_tokens = tgt_ids_BxTxC[..., i]
            channel_embed = self.embeddings[i](channel_tokens)
            x = channel_embed if x is None else x + channel_embed

        for i, layer in enumerate(self.layers):
            self_cache = state.self_attn_cache[i]
            cross_cache = state.cross_attn_cache[i]
            x = layer(x, state, self_attn_cache=self_cache, cross_attn_cache=cross_cache, prefill=True)

        # Final Norm
        x = self.norm(x)
        logits_BxTxCxV = self.logits_dense(x)

        return logits_BxTxCxV.to(torch.float32)

class DiaModel(nn.Module):
    """PyTorch Dia Model using DenseGeneral."""
    default_instance = None

    def __init__(self, config: DiaConfig, compute_dtype: torch.dtype, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config, compute_dtype)
        self.decoder = Decoder(config, quant_config, compute_dtype)
        self.device = torch.device("cuda")

        self.compute_dtype = compute_dtype

        state, out = self._prepare_generation("[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face.", None, False)
        self.dec_state = state
        self.dec_output = out
        self.dec_step = self.dec_output.prefill_step - 1


    @classmethod
    def init_default(cls, *args, **kwargs):
        if cls.default_instance is None:
            print("Initializing DiaModel default instance")
            cls.default_instance = cls(*args, **kwargs)
        return cls.default_instance

    @classmethod
    def get_default_instance(cls) -> "DiaModel":
        # while cls.default_instance is None:
        #     time.sleep(0.01)  # Wait 10ms before checking again

        if cls.default_instance is None:
            raise ValueError("DiaModel has not been initialized yet.")

        return cls.default_instance

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        audio_eos_value = self.config.data.audio_eos_value
        audio_pad_value = self.config.data.audio_pad_value
        delay_pattern = self.config.data.delay_pattern
        max_tokens = self.config.data.audio_length
        max_delay_pattern = max(delay_pattern)
        bos_countdown = max_delay_pattern

        # self.dec_state.prepare_step(self.dec_step)

        tokens_Bx1xC = self.dec_output.get_tokens_at(self.dec_step).unsqueeze(0).expand(2, -1, -1)
        pred_C = self._decoder_step(
                tokens_Bx1xC,
                self.dec_state,
                3.0, #cfg_scale,
                1.3,#temperature,
                0.95, #top_p,
                35, #cfg_filter_top_k,
            )

        # This is part of after sampling generation
        # I don't know where it should be moved
        # TODO: Check where in llama.py is checking for eos
        # if (not eos_detected and pred_C[0] == audio_eos_value) or dec_step == max_tokens - max_delay_pattern - 1:
        #     eos_detected = True
        #     eos_countdown = max_delay_pattern

        # if eos_countdown > 0:
        #     step_after_eos = max_delay_pattern - eos_countdown
        #     for i, d in enumerate(delay_pattern):
        #         if step_after_eos == d:
        #             pred_C[i] = audio_eos_value
        #         elif step_after_eos > d:
        #             pred_C[i] = audio_pad_value
        #     eos_countdown -= 1

        bos_countdown = max(0, bos_countdown - 1)
        self.dec_output.update_one(pred_C, self.dec_step + 1, bos_countdown > 0)

        # TODO: Check where in llama.py is checking for eos
        # This is commented on puprose
        # if eos_countdown == 0:
        #     break

        self.dec_step += 1

        return pred_C

    def _decoder_step(
        self,
        tokens_Bx1xC: torch.Tensor,
        dec_state: DecoderInferenceState,
        cfg_scale: float,
        temperature: float,
        top_p: float,
        cfg_filter_top_k: int,
    ) -> torch.Tensor:
        audio_eos_value = self.config.data.audio_eos_value
        logits_Bx1xCxV = self.decoder.decode_step(tokens_Bx1xC, dec_state)

        logits_last_BxCxV = logits_Bx1xCxV[:, -1, :, :]
        uncond_logits_CxV = logits_last_BxCxV[0, :, :]
        cond_logits_CxV = logits_last_BxCxV[1, :, :]

        logits_CxV = cond_logits_CxV + cfg_scale * (cond_logits_CxV - uncond_logits_CxV)
        logits_CxV[:, audio_eos_value + 1 :] = -torch.inf
        logits_CxV[1:, audio_eos_value:] = -torch.inf

        # TODO: Check where llama.py is making the sampling and move this there
        pred_C = _sample_next_token(
            logits_CxV.float(),
            temperature=temperature,
            top_p=top_p,
            cfg_filter_top_k=cfg_filter_top_k,
        )
        return pred_C

    def _prepare_generation(self, text: str, audio_prompt: str | torch.Tensor | None, verbose: bool):
        enc_input_cond = self._prepare_text_input(text)
        enc_input_uncond = torch.zeros_like(enc_input_cond)
        enc_input = torch.cat([enc_input_uncond, enc_input_cond], dim=0)

        if isinstance(audio_prompt, str):
            audio_prompt = self.load_audio(audio_prompt)
        prefill, prefill_step = self._prepare_audio_prompt(audio_prompt)

        if verbose:
            print("generate: data loaded")

        enc_state = EncoderInferenceState.new(self.config, enc_input_cond)
        encoder_out = self.encoder(enc_input, enc_state)

        dec_cross_attn_cache = self.decoder.precompute_cross_attn_cache(encoder_out, enc_state.positions)
        dec_state = DecoderInferenceState.new(
            self.config, enc_state, encoder_out, dec_cross_attn_cache, self.compute_dtype
        )
        dec_output = DecoderOutput.new(self.config, self.device)
        dec_output.prefill(prefill, prefill_step)

        dec_step = prefill_step - 1
        if dec_step > 0:
            dec_state.prepare_step(0, dec_step)
            tokens_BxTxC = dec_output.get_tokens_at(0, dec_step).unsqueeze(0).expand(2, -1, -1)
            self.decoder.forward(tokens_BxTxC, dec_state)

        return dec_state, dec_output

    def load_audio(self, audio_path: str) -> torch.Tensor:
        audio, sr = torchaudio.load(audio_path, channels_first=True)  # C, T
        if sr != DEFAULT_SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, DEFAULT_SAMPLE_RATE)
        audio = audio.to(self.device).unsqueeze(0)  # 1, C, T
        audio_data = self.dac_model.preprocess(audio, DEFAULT_SAMPLE_RATE)
        _, encoded_frame, _, _, _ = self.dac_model.encode(audio_data)  # 1, C, T
        return encoded_frame.squeeze(0).transpose(0, 1)

    # TODO: Probably to be removed, maybe should be as tokenizer
    def _prepare_text_input(self, text: str) -> torch.Tensor:
        """Encodes text prompt, pads, and creates attention mask and positions."""
        text_pad_value = self.config.data.text_pad_value
        max_len = self.config.data.text_length

        byte_text = text.encode("utf-8")
        replaced_bytes = byte_text.replace(b"[S1]", b"\x01").replace(b"[S2]", b"\x02")
        text_tokens = list(replaced_bytes)

        current_len = len(text_tokens)
        padding_needed = max_len - current_len
        if padding_needed <= 0:
            text_tokens = text_tokens[:max_len]
            padded_text_np = np.array(text_tokens, dtype=np.uint8)
        else:
            padded_text_np = np.pad(
                text_tokens,
                (0, padding_needed),
                mode="constant",
                constant_values=text_pad_value,
            ).astype(np.uint8)

        src_tokens = torch.from_numpy(padded_text_np).to(torch.long).to(self.device).unsqueeze(0)  # [1, S]
        return src_tokens

    def _prepare_audio_prompt(self, audio_prompt: torch.Tensor | None) -> tuple[torch.Tensor, int]:
        num_channels = self.config.data.channels
        audio_bos_value = self.config.data.audio_bos_value
        audio_pad_value = self.config.data.audio_pad_value
        delay_pattern = self.config.data.delay_pattern
        max_delay_pattern = max(delay_pattern)

        prefill = torch.full(
            (1, num_channels),
            fill_value=audio_bos_value,
            dtype=torch.int,
            device=self.device,
        )

        prefill_step = 1

        if audio_prompt is not None:
            prefill_step += audio_prompt.shape[0]
            prefill = torch.cat([prefill, audio_prompt], dim=0)

        delay_pad_tensor = torch.full(
            (max_delay_pattern, num_channels), fill_value=-1, dtype=torch.int, device=self.device
        )
        prefill = torch.cat([prefill, delay_pad_tensor], dim=0)

        delay_precomp = build_delay_indices(
            B=1,
            T=prefill.shape[0],
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        prefill = apply_audio_delay(
            audio_BxTxC=prefill.unsqueeze(0),
            pad_value=audio_pad_value,
            bos_value=audio_bos_value,
            precomp=delay_precomp,
        ).squeeze(0)

        return prefill, prefill_step

def _sample_next_token(
    logits_BCxV: torch.Tensor,
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int | None = None,
) -> torch.Tensor:
    if temperature == 0.0:
        return torch.argmax(logits_BCxV, dim=-1)

    logits_BCxV = logits_BCxV / temperature
    if cfg_filter_top_k is not None:
        _, top_k_indices_BCxV = torch.topk(logits_BCxV, k=cfg_filter_top_k, dim=-1)
        mask = torch.ones_like(logits_BCxV, dtype=torch.bool)
        mask.scatter_(dim=-1, index=top_k_indices_BCxV, value=False)
        logits_BCxV = logits_BCxV.masked_fill(mask, -torch.inf)

    if top_p < 1.0:
        probs_BCxV = torch.softmax(logits_BCxV, dim=-1)
        sorted_probs_BCxV, sorted_indices_BCxV = torch.sort(probs_BCxV, dim=-1, descending=True)
        cumulative_probs_BCxV = torch.cumsum(sorted_probs_BCxV, dim=-1)

        sorted_indices_to_remove_BCxV = cumulative_probs_BCxV > top_p
        sorted_indices_to_remove_BCxV[..., 1:] = sorted_indices_to_remove_BCxV[..., :-1].clone()
        sorted_indices_to_remove_BCxV[..., 0] = 0

        indices_to_remove_BCxV = torch.zeros_like(sorted_indices_to_remove_BCxV)
        indices_to_remove_BCxV.scatter_(dim=-1, index=sorted_indices_BCxV, src=sorted_indices_to_remove_BCxV)
        logits_BCxV = logits_BCxV.masked_fill(indices_to_remove_BCxV, -torch.inf)

    final_probs_BCxV = torch.softmax(logits_BCxV, dim=-1)

    sampled_indices_BC = torch.multinomial(final_probs_BCxV, num_samples=1)
    sampled_indices_C = sampled_indices_BC.squeeze(-1)
    return sampled_indices_C

class DiaTTS(nn.Module):
    def __init__(
        self,
        config: DiaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = self._init_model(config, quant_config)

    def _init_model(
        self,
        config: DiaConfig,
        quant_config: Optional[QuantizationConfig] = None
    ):
        # TODO: Add option for other dtypes - float16, bfloat16
        return DiaModel.init_default(config, torch.float16, quant_config)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            name = "model." + name
            if name in params_dict.keys():
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
            else:
                logger.warning(f"Parameter {name} not found in params_dict")

    def prepare_generation(self):
        self.model.dec_state.prepare_step(self.model.dec_step)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        print("forward")
        return self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

EntryClass = [DiaTTS]