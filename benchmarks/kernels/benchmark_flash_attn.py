import random
import time
from typing import List, Optional, Tuple

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, FlexibleArgumentParser,
                        create_kv_caches_with_random)
from vllm.vllm_flash_attn import (flash_attn_varlen_func,
                                  flash_attn_with_kvcache)

NUM_BLOCKS = 1024
PARTITION_SIZE = 512


@torch.inference_mode()
def main(args) -> None:
    seq_lens = [(1, 1328), (5, 18), (129, 463)]
    num_heads = (8, 2)
    head_size = 128
    block_size = 16
    sliding_window = None
    dtype = torch.bfloat16
    soft_cap = None
    num_blocks = 2048

    torch.set_default_device("cuda")
    current_platform.seed_everything(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = ((sliding_window - 1, 0) if sliding_window is not None else
                   (-1, -1))
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens),
                        num_query_heads,
                        head_size,
                        dtype=dtype)
    key_cache = torch.randn(num_blocks,
                            block_size,
                            num_kv_heads,
                            head_size,
                            dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens,
                                 dtype=torch.int32).cumsum(dim=0,
                                                           dtype=torch.int32)
    cu_kv_lens = torch.tensor([0] + kv_lens,
                              dtype=torch.int32).cumsum(dim=0,
                                                        dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 num_blocks,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)

    def run_cuda_benchmark(num_iters: int, profile: bool = False) -> float:
        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.perf_counter()

        for _ in range(num_iters):
            flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=cu_query_lens,
                cu_seqlens_k=cu_kv_lens,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_kv_len,
                softmax_scale=scale,
                causal=True,
                window_size=window_size,
                block_table=block_tables,
                softcap=soft_cap if soft_cap is not None else 0,
            )
        torch.cuda.synchronize()

        end_time = time.perf_counter()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        return (end_time - start_time) / num_iters
    
    # Warmup.
    print("Warming up...")
    run_benchmark = run_cuda_benchmark
    run_benchmark(num_iters=3, profile=False)

    # Benchmark.
    print("Run benchmark...")
    if args.profile:
        latency = run_benchmark(num_iters=1, profile=True)
    else:
        latency = run_benchmark(num_iters=10, profile=False)
    print(f"Kernel running time: {latency * 1000000:.3f} us")


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description="Benchmark the flash attention kernel.")
    parser.add_argument("--kernel",
                        type=str,
                        choices=["single-query-flash-attn", "multi-queries-flash-attn"],
                        default="multi-queries-flash-attn")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    # print(args)

    main(args)
