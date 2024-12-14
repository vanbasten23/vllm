# Benchmark flash attention kernel using the same input as in the real prefix caching benchmark script.
# To run, do
# (myvllmenv) xiowei@a100-8:~/github/myforks/vllm/benchmarks$ python kernels/benchmark_flash_attn.py --kernel "multi-queries-flash-attn"
# To get the profile, do
# (myvllmenv) xiowei@a100-8:~/github/myforks/vllm/benchmarks$ rm my_profile.nsys-rep
# (myvllmenv) xiowei@a100-8:~/github/myforks/vllm/benchmarks$ nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi  --cudabacktrace=true -x true -o ./my_profile python kernels/benchmark_flash_attn.py --kernel "multi-queries-flash-attn" --profile 
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

# When running benchmark_prefix_caching, here is the input size:
# num_tokens=1342, num_q_heads=8, head_size=256, 
# num_kv_heads=1, total_num_pages=231746, page_size=16 
# query_start_loc=tensor([0,1248,1249, 1250,...,1342]), max_query_len=1249
# query_start_loc.shape=torch.Size([95])
# seq_start_loc=tensor([0,1248,2496,...,117406]),max_seq_len=1249
# seq_start_loc.shape=torch.Size([95])
# softmax_scale=0.0625, window_size=(-1, -1), alibi_slopes=None
# logits_soft_cap=0, block_tables.shape=torch.Size([94, 78])
# block_tables=tensor([[ 0,  1,  2,  ..., 75, 76, 77],
#        [ 0,  1,  2,  ..., 75, 76, 77],
#        [ 0,  1,  2,  ..., 75, 76, 77],
#        ...,
#        [ 0,  1,  2,  ..., 75, 76, 77],
#        [ 0,  1,  2,  ..., 75, 76, 77],
#        [ 0,  1,  2,  ..., 75, 76, 77]]
@torch.inference_mode()
def main(args) -> None:
    num_seqs = 94
    
    seq_lens = [(1248, 1248)]
    for _ in range(num_seqs-1):
        seq_lens.append((1, 1248))

    num_heads = (8, 1)
    head_size = 256
    block_size = 16
    sliding_window = None
    dtype = torch.bfloat16
    soft_cap = None
    num_blocks = 231746

    torch.set_default_device("cuda")
    current_platform.seed_everything(0)
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

    assert args.kernel == "single-query-flash-attn" or args.kernel == "multi-queries-flash-attn", f"Invalid argument {args.kernel}"
    if args.kernel == "multi-queries-flash-attn":
        query = torch.randn(sum(query_lens),
                          num_query_heads,
                          head_size,
                          dtype=dtype)
    elif args.kernel == "single-query-flash-attn":
        query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)
        
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
    print(f'{query.shape=}, {key_cache.shape=},{value_cache.shape=},{cu_query_lens.shape=}, {cu_kv_lens.shape=}, {max_query_len=}, {max_kv_len=}, {block_tables=}')

    def run_cuda_benchmark(num_iters: int, profile: bool = False) -> float:
        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.perf_counter()

        for _ in range(num_iters):
            if args.kernel == "multi-queries-flash-attn":
                torch.cuda.nvtx.range_push("line82 flash_attn_varlen_func")
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
                torch.cuda.nvtx.range_pop()
            elif args.kernel == "single-query-flash-attn":
                kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)
                flash_attn_with_kvcache(
                    q=query.unsqueeze(1),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    softmax_scale=scale,
                    causal=True,
                    block_table=block_tables,
                    cache_seqlens=kv_lens_tensor,
                    softcap=soft_cap if soft_cap is not None else 0,
                    window_size=window_size,
                ).squeeze(1)
        torch.cuda.synchronize()

        end_time = time.perf_counter()
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
        return (end_time - start_time) / num_iters
   
    # Warmup.
    print("Warming up...")
    run_benchmark = run_cuda_benchmark
    run_benchmark(num_iters=3, profile=False)

    # Benchmark.
    print("Run benchmark...")
    if args.profile:
        latency = run_benchmark(num_iters=3, profile=True)
    else:
        latency = run_benchmark(num_iters=100, profile=False)
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
