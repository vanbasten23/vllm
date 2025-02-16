from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch_xla.experimental.custom_kernel  # Required to register custom ops.

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata, AttentionType)
from vllm.attention.backends.utils import CommonAttentionState


class PallasAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "PALLAS_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> Type["PallasAttentionBackendImpl"]:
        return PallasAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["PallasMetadata"]:
        return PallasMetadata

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (num_kv_heads, num_blocks, block_size, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the TPU backend.")

    @torch.compile(backend="openxla")
    @staticmethod
    def copy_blocks(
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        src_to_dists: Tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        assert False, "I assume this PallasAttentionBackend.copy_blocks function should not be used. But I could be wrong."  # TODO(xw32): remove
        src_indices, dst_indices = src_to_dists
        for k_cache, v_cache in kv_caches:
            torch.ops.xla.dynamo_set_buffer_donor_(k_cache, True)
            k_cache[:, dst_indices] = k_cache[:, src_indices]
            torch.ops.xla.dynamo_set_buffer_donor_(v_cache, True)
            v_cache[:, dst_indices] = v_cache[:, src_indices]


# Why remove the base class AttentionMetadata. Because the base requires
# to set num_prefills, num_prefill_tokens, and num_decode_tokens.
@dataclass
class PallasMetadata():

    # old begins
    # Currently, input sequences can only contain all prefills
    # or all decoding.
    # block_tables: Optional[torch.Tensor] = None
    # context_lens: Optional[torch.Tensor] = None
    # effective_query_lens: Optional[torch.Tensor] = None

    # @property
    # def prefill_metadata(self) -> Optional["PallasMetadata"]:
    #     if self.num_prefills == 0:
    #         return None

    #     assert self.num_decode_tokens == 0
    #     return self

    # @property
    # def decode_metadata(self) -> Optional["PallasMetadata"]:
    #     if self.num_decode_tokens == 0:
    #         return None

    #     assert self.num_prefills == 0
    #     assert self.num_prefill_tokens == 0
    #     assert self.block_tables is not None
    #     assert self.context_lens is not None
    #     return self
    # old ends

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    # Used in the PallasAttentionBackendImpl
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    context_lens: torch.Tensor
    query_start_loc: torch.Tensor
    num_seqs: int


class PallasAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "Paged attention Pallas kernel does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if head_size % 128 != 0:
            raise NotImplementedError("Head size must be a multiple of 128.")
        if alibi_slopes is not None:
            raise NotImplementedError("Alibi slopes is not supported.")
        if sliding_window is not None:
            raise NotImplementedError("Sliding window is not supported.")
        if kv_cache_dtype != "auto":
            raise NotImplementedError("FP8 KV cache dtype is not supported.")
        if blocksparse_params is not None:
            raise NotImplementedError("Blocksparse is not supported.")
        if logits_soft_cap is not None:
            raise NotImplementedError(
                "Attention logits soft-capping is not supported.")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PallasAttentionBackendImpl")

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: PallasMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with Pallas attention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        print(f'xw32 PallasAttentionBackendImpl.forward  begins {query.shape=}, {key.shape=}')

        if attn_metadata is None:
            if output is None:
                output = torch.ones_like(query)
            return output

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        num_tokens, hidden_size = query.shape
        query = query.view(num_tokens, self.num_heads, self.head_size)
        key = key.view(num_tokens, self.num_kv_heads, self.head_size)
        value = value.view(num_tokens, self.num_kv_heads, self.head_size)

        if kv_cache[0].numel() > 0:
            print('xw32 write to kv cache')
            slot_mapping = attn_metadata.slot_mapping
            key_cache, value_cache = kv_cache
            write_to_kv_cache(key, value, key_cache, value_cache, slot_mapping)

        query = query * self.scale
        output = torch.ops.xla.ragged_paged_attention(
            query,
            key_cache,
            value_cache,
            attn_metadata.context_lens,
            attn_metadata.block_tables,
            attn_metadata.query_start_loc,
            attn_metadata.num_seqs,
            num_kv_pages_per_block=16,
            num_queries_per_block=128,
            use_kernel=True,
        )

        return output


def write_to_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """ Write the key and values to the KV cache.

    Args:
        key: shape = [num_tokens, num_kv_heads, head_size]
        value: shape = [num_tokens, num_kv_heads, head_size]
        k_cache = [num_blocks, block_size, num_kv_heads, head_size]
        v_cache = [num_blocks, block_size, num_kv_heads, head_size]

    """
    torch.ops.xla.dynamo_set_buffer_donor_(key_cache, True)
    torch.ops.xla.dynamo_set_buffer_donor_(value_cache, True)

    key = key.flatten(0, 1)
    value = value.flatten(0, 1)
    key_cache = key_cache.flatten(0, 2)
    value_cache = value_cache.flatten(0, 2)
    key_cache.index_copy_(0, slot_mapping, key)
    value_cache.index_copy_(0, slot_mapping, value)
