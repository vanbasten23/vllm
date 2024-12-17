"""
(myvllmenv) xiowei@t1v-n-f468dc5f-w-0:/mnt/disks/persist/myforks/vllm/benchmarks$ XLA_HLO_DEBUG=1  VLLM_TPU_PROFILE_DURATION_MS=2000 VLLM_TPU_PROFILE_DELAY_MS=1000 python3 profile_tpu.py     --model google/gemma-2b     --input-len 1     --output-len 128     --batch-size 32     --enforce-eager     --profile-result-dir /mnt/disks/persist/torchprofiles     --max-model-len 2048 2>&1 | tee ~/out.txt
## VLLM TPU Profiling

This guide explains how to profile the TPU performance on VLLM for specific shapes. Note: an actual running server is a mix of both prefill of many shapes and decode of many shapes.

>> In all examples below, we run several warmups before (so `--enforce-eager` is okay)
### Generate Decode Trace

```bash
export XLA_HLO_DEBUG=1
export MODEL=meta-llama/Llama-3.1-70B-Instruct
export VLLM_TPU_PROFILE_DURATION_MS=2000
export VLLM_TPU_PROFILE_DELAY_MS=1000

rm -rf ~/.cache/vllm/xla_cache
python3 profile_tpu.py \
    --model $MODEL \
    --input-len 1 \
    --output-len 128 \
    --batch-size 32 \
    --enforce-eager \
    --profile-result-dir profiles \
    --max-model-len 2048 --tensor-parallel-size 8
```

### Generate Prefill Trace

```bash
export XLA_HLO_DEBUG=1
export MODEL=meta-llama/Llama-3.1-8B-Instruct
export VLLM_TPU_PROFILE_DURATION_MS=3000
export VLLM_TPU_PROFILE_DELAY_MS=0

rm -rf ~/.cache/vllm/xla_cache
python3 profile_tpu.py \
    --model $MODEL \
    --input-len 1024 \
    --output-len 1 \
    --batch-size 1 \
    --enforce-eager \
    --profile-result-dir profiles \
    --max-model-len 2048 --tensor-parallel-size 8
```
"""
import argparse
import dataclasses
import time
import os
from typing import List, Optional

import numpy as np
import torch_xla.debug.profiler as xp
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.utils import FlexibleArgumentParser

DURATION_MS = int(os.getenv("VLLM_TPU_PROFILE_DURATION_MS", 3000))
DELAY_MS = int(os.getenv("VLLM_TPU_PROFILE_DELAY_MS", 0))

def main(args: argparse.Namespace):
    print(args)

    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**dataclasses.asdict(engine_args))
    server = xp.start_server(9012)

    sampling_params = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)
    dummy_prompt_token_ids = np.random.randint(10000,
                                               size=(args.batch_size,
                                                     args.input_len))
    dummy_prompts: List[PromptType] = [{
        "prompt_token_ids": batch
    } for batch in dummy_prompt_token_ids.tolist()]

    def run_to_completion(profile_dir: Optional[str] = None):
        if profile_dir:
            xp.trace_detached("localhost:9012",
                              profile_dir,
                              delay_ms=DELAY_MS,
                              duration_ms=DURATION_MS)
            if DELAY_MS == 0:
                time.sleep(1.0)
            for _ in range(5):
                llm.generate(dummy_prompts,
                            sampling_params=sampling_params,
                            use_tqdm=False)
        else:
            start_time = time.perf_counter()
            llm.generate(dummy_prompts,
                         sampling_params=sampling_params,
                         use_tqdm=False)
            end_time = time.perf_counter()
            latency = end_time - start_time
            return latency

    print("Warming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        run_to_completion(profile_dir=None)

    profile_dir = args.profile_result_dir
    print(f"Profiling (results will be saved to '{profile_dir}')...")
    run_to_completion(profile_dir=profile_dir)
    return

if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-iters-warmup',
                        type=int,
                        default=3,
                        help='Number of iterations to run for warmup.')
    parser.add_argument(
        '--profile-result-dir',
        type=str,
        default="profiles",
        help=('path to save the pytorch profiler output. Can be visualized '
              'with ui.perfetto.dev or Tensorboard.'))

    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
