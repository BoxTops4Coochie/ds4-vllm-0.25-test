# DeepSeek-V4-Flash on stock vLLM v0.25.0  
## SM120 / RTX PRO 6000 Blackwell — FlashInfer 0.6.14 workaround and 450 W benchmark

**Fix for:**

```text
TypeError: trtllm_batch_decode_sparse_mla_dsv4()
got an unexpected keyword argument 'swa_topk_lens'
```

This repository documents running the official `deepseek-ai/DeepSeek-V4-Flash` checkpoint on the stock `vllm/vllm-openai:v0.25.0` image using two RTX PRO 6000 Blackwell GPUs.

No vLLM source patches or custom fork are used. The container upgrades FlashInfer from the image's pinned `0.6.13` release to `0.6.14` at startup.

> This is the older stock-vLLM baseline. It intentionally does not include the later DeepSeek-V4 vLLM PRs, FlashInfer CUTLASS MoE patch, or DSpark work.

---

## Hardware

| Component | Configuration |
|---|---|
| OS | Ubuntu 26.04 LTS |
| Kernel | `7.0.0-27-generic` |
| CPU | AMD Ryzen Threadripper PRO 7965WX, 24 cores |
| System RAM | Approximately 256 GiB |
| Serving GPUs | 2× NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| Excluded GPU | NVIDIA GeForce RTX 5090, host GPU 0 |
| Serving GPU IDs | Host GPUs 1 and 2 |
| Interconnect | PCIe, no NVLink |
| Tensor parallelism | TP=2 |
| Benchmark power limit | 450 W per RTX PRO 6000 |

---

## Background

vLLM v0.25.0 includes the major pieces needed to run DeepSeek-V4-Flash on SM120 Blackwell:

- DeepGEMM support for SM120
- DeepSeek-V4-specific sparse-MLA and MXFP8 work
- DeepSeek-V4 tool and reasoning parsers
- Native MTP speculative decoding support

The official checkpoint uses native FP8 weights and contains 46 safetensors shards totaling approximately 148.66 GiB. With tensor parallelism across two GPUs, the final model footprint was approximately 75.64 GiB per GPU.

The stock vLLM v0.25.0 image resolves the model's experts to FP4 and uses:

```text
DEEPGEMM_MXFP4
```

for the MoE backend. Attention uses the default:

```text
DEEPSEEK_SPARSE_SWA
```

path.

---

## The stock-image bug

The vLLM v0.25.0 image pins:

```text
flashinfer-python==0.6.13
```

However, the DeepSeek-V4 sparse-MLA code calls the FlashInfer decode kernel with the newer `swa_topk_lens` argument. That argument is available in FlashInfer 0.6.14 but not in 0.6.13.

The mismatch causes inference startup to fail with:

```text
TypeError: trtllm_batch_decode_sparse_mla_dsv4()
got an unexpected keyword argument 'swa_topk_lens'
```

---

## Workaround

Upgrade `flashinfer-python` to 0.6.14 inside the container before launching vLLM.

At the time of this test, a matching `flashinfer-cubin==0.6.14` package was unavailable. The startup script therefore removes the stale 0.6.13 cubin and JIT-cache packages so FlashInfer can download or compile matching kernels at runtime.

```bash
if ! pip install --no-cache-dir \
  "flashinfer-python==0.6.14" \
  "flashinfer-cubin==0.6.14"; then

  echo ">>> flashinfer-cubin 0.6.14 not available; removing stale cubin packages"

  pip install --no-cache-dir "flashinfer-python==0.6.14"
  pip uninstall -y flashinfer-cubin flashinfer-jit-cache || true
fi
```

The expected pip dependency warning is intentional:

```text
vllm 0.25.0 requires flashinfer-python==0.6.13,
but flashinfer-python 0.6.14 is installed
```

Do not bypass FlashInfer's version check while retaining the old 0.6.13 binaries. The Python package and kernel binaries must be version-compatible.

---

## Docker Compose

```yaml
services:
  ds4-flash:
    # Stock vLLM v0.25.0 on SM120 + FlashInfer 0.6.14.
    image: vllm/vllm-openai:v0.25.0
    container_name: ds4-flash-stock
    restart: "no"

    network_mode: host
    ipc: host
    shm_size: "64g"

    gpus:
      # Host GPU 0 is an RTX 5090 and is excluded.
      # The RTX PRO 6000 cards are host GPUs 1 and 2.
      - driver: nvidia
        device_ids: ["1", "2"]
        capabilities: [gpu]

    runtime: nvidia
    init: true

    ulimits:
      memlock: -1
      nofile: 1048576
      stack: 67108864

    environment:
      - CUDA_DEVICE_ORDER=PCI_BUS_ID
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - NCCL_IB_DISABLE=1
      - NCCL_P2P_LEVEL=SYS
      - SAFETENSORS_FAST_GPU=1

      # Pin the Hugging Face cache because XDG_CACHE_HOME is changed below.
      - HF_HOME=/root/.cache/huggingface
      - HF_HUB_OFFLINE=1

      # Persistent runtime and compiled-kernel caches.
      - XDG_CACHE_HOME=/cache
      - VLLM_CACHE_ROOT=/cache/vllm
      - TRITON_CACHE_DIR=/cache/triton
      - TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor
      - FLASHINFER_WORKSPACE_BASE=/cache/flashinfer

    volumes:
      - /m2-2/vllm/root:/root/.cache/huggingface
      - /m2-2/vllm/dsv4flash-stock-cache:/cache

    entrypoint: ["/bin/bash", "-c"]

    command:
      - |
        set -euo pipefail

        # Plan A: install a matched 0.6.14 Python/cubin pair.
        # Plan B: if the cubin package is unavailable, remove stale
        # 0.6.13 cubin/JIT-cache packages and use runtime download/JIT.
        if ! pip install --no-cache-dir \
          "flashinfer-python==0.6.14" \
          "flashinfer-cubin==0.6.14"; then

          echo ">>> flashinfer-cubin 0.6.14 not available; removing stale cubin packages"

          pip install --no-cache-dir "flashinfer-python==0.6.14"
          pip uninstall -y flashinfer-cubin flashinfer-jit-cache || true
        fi

        python3 -c \
          "import flashinfer; print('>>> flashinfer version:', flashinfer.__version__)"

        exec vllm serve deepseek-ai/DeepSeek-V4-Flash \
          --served-model-name d4f \
          --api-key CHANGE_ME \
          --trust-remote-code \
          --host 0.0.0.0 \
          --port 15004 \
          --tensor-parallel-size 2 \
          --disable-custom-all-reduce \
          --kv-cache-dtype fp8 \
          --gpu-memory-utilization 0.95 \
          --max-model-len 1048576 \
          --max-num-seqs 64 \
          --max-num-batched-tokens 4096 \
          --enable-prefix-caching \
          --enable-chunked-prefill \
          --tokenizer-mode deepseek_v4 \
          --tool-call-parser deepseek_v4 \
          --reasoning-parser deepseek_v4 \
          --enable-auto-tool-choice \
          --speculative-config \
            '{"method":"mtp","num_speculative_tokens":2}' \
          --async-scheduling
```

Change the API key, volume paths, port, and GPU IDs for your host.

---

## Confirmed startup configuration

```text
vLLM version:                     0.25.0
FlashInfer version:               0.6.14
Model:                            deepseek-ai/DeepSeek-V4-Flash
Target architecture:              DeepseekV4ForCausalLM
Draft architecture:               DeepSeekV4MTPModel
Model quantization:               deepseek_v4_fp8
Expert dtype:                     fp4
MoE backend:                      DEEPGEMM_MXFP4
KV-cache dtype:                   fp8
Tensor parallel size:             2
Pipeline parallel size:           1
MTP speculative tokens:           2
Max model length:                 1,048,576
Max sequences:                    64
Max batched tokens:               4,096
NCCL version:                     2.28.9
All-reduce backend:               PYNCCL
```

### Model loading

```text
Checkpoint shards:                46
Checkpoint size:                  148.66 GiB
Target weight load time:          16.47 seconds
MTP draft load time:              1.24 seconds
Total model memory per GPU:       75.64 GiB
Total model-load duration:        18.56 seconds
```

The MTP drafter loaded successfully and shared the target model's:

- embedding weights
- LM head
- top-k indices buffer

### KV cache and CUDA graphs

```text
Available KV-cache memory:        10.29 GiB
GPU KV-cache capacity:            1,993,100 tokens
Maximum 1M-context concurrency:   1.90x
Estimated CUDA graph memory:      1.58 GiB
Actual CUDA graph pool:           1.73 GiB
CUDA graph capture duration:      32 seconds
Full engine initialization:       133.36 seconds
```

The server then started successfully on:

```text
http://0.0.0.0:15004
```

---

## Warmup validation

A 512-token chat-completion request completed successfully twice:

```bash
curl -s http://127.0.0.1:15004/v1/chat/completions \
  -H "Authorization: Bearer CHANGE_ME" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "d4f",
    "messages": [{
      "role": "user",
      "content": "Explain CPU cache coherence"
    }],
    "max_tokens": 512,
    "temperature": 0
  }'
```

Both responses returned:

```text
completion_tokens: 512
finish_reason: length
system_fingerprint: vllm-0.25.0-tp2-54609fcf
```

---

## Benchmark methodology

The serving benchmark used the following matrix:

```text
Concurrency:
1, 2, 4, 8, 16, 32, 64

Prompt context:
0, 8k, 16k, 32k, 64k, 128k, 256k
```

Values below are aggregate output throughput in tokens per second.

Missing cells indicate that the selected prompt-depth and concurrency combination exceeded the practical KV-cache test matrix at the configured 1,048,576-token maximum model length.

All GPUs were configured for a 450 W power limit.

---

## 450 W benchmark results

| Prompt depth \ Concurrency | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 182.51 | 271.99 | 296.14 | 575.25 | 828.93 | 1288.15 | 1944.42 |
| 8k | 186.75 | 275.28 | 316.51 | 507.52 | 743.74 | 1063.34 | 1640.74 |
| 16k | 188.54 | 257.80 | 280.07 | 521.11 | 673.96 | 1099.87 | 1593.25 |
| 32k | 183.20 | 242.14 | 274.72 | 497.77 | 676.83 | 1055.78 | — |
| 64k | 186.14 | 239.71 | 284.75 | 495.33 | 701.67 | — | — |
| 128k | 158.76 | 239.85 | 271.38 | 460.93 | — | — | — |
| 256k | 169.48 | 246.76 | 256.44 | — | — | — | — |

### Observations

- Single-request throughput remains relatively flat through 64k context:
  - 0 context: **182.51 tok/s**
  - 64k context: **186.14 tok/s**
- At the deepest tested context:
  - 256k, concurrency 1: **169.48 tok/s**
- Highest aggregate throughput:
  - 0 context, concurrency 64: **1944.42 tok/s**
- Long-context aggregate throughput remains strong:
  - 64k, concurrency 16: **701.67 tok/s**
  - 128k, concurrency 8: **460.93 tok/s**
  - 256k, concurrency 4: **256.44 tok/s**

---

## GPU telemetry during the benchmark

A live snapshot was captured while the benchmark was running:

| GPU | Power limit | Power draw | Temperature | SM clock | VRAM used | Utilization |
|---|---:|---:|---:|---:|---:|---:|
| RTX PRO 6000 — GPU 1 | 450 W | 449.97 W | 76°C | 2332 MHz | 95,424 MiB | 100% |
| RTX PRO 6000 — GPU 2 | 450 W | 449.99 W | 91°C | 1867 MHz | 95,454 MiB | 100% |

The second GPU was considerably hotter and clocked lower during this snapshot. This may have limited aggregate throughput and should be considered when comparing results across different chassis or cooling configurations.

The excluded RTX 5090 remained idle at approximately 10 W.

---

## Metrics note

A Prometheus metrics snapshot was captured while the benchmark was in progress, but the counters were cumulative from container startup and may include warmup or earlier requests.

For that reason, cumulative speculative-acceptance and prefix-cache percentages are **not presented as benchmark-specific results** in this README.

The snapshot did confirm at capture time:

```text
MTP speculative decoding active
No preemptions
No completed-request errors
16 requests running
0 requests waiting
```

---

## Relevant startup warnings

The following warnings were expected or non-fatal:

```text
vLLM 0.25.0 pins flashinfer-python 0.6.13,
while this workaround intentionally installs 0.6.14.
```

```text
num_speculative_tokens > 1 may reduce MTP acceptance rate.
```

```text
max_num_scheduled_tokens=4096 may be suboptimal for the configured
speculative token count and max_num_seqs.
```

```text
VLLM_USE_BREAKABLE_CUDAGRAPH disabled the torch.compile/Inductor pipeline.
```

```text
SymmMemCommunicator is unavailable on compute capability 12.0;
PYNCCL was selected for tensor-parallel all-reduce.
```

None prevented successful startup or benchmark completion.

---

## Why this baseline matters

This run measures the stock vLLM v0.25.0 execution path with only the FlashInfer dependency workaround:

```text
Attention:  DEEPSEEK_SPARSE_SWA
MoE:        DEEPGEMM_MXFP4
Spec decode: MTP-2
```

It does not include later changes such as:

- DeepSeek-V4 MTP compression/RoPE fixes
- FlashInfer CUTLASS MXFP4/MXFP8 MoE support
- hybrid KV-cache reporting fixes
- DSpark speculative decoding
- additional SM120 topk=256 sparse-MLA kernel support

This makes it useful as a clean baseline for evaluating those later patches.

---

## Expiry

This workaround is temporary.

Once a stock vLLM image ships with a compatible FlashInfer version, remove the startup-time pip installation and stale-package cleanup. The remainder of the Compose configuration can stay substantially the same.
