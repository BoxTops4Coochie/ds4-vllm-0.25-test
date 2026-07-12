
# DeepSeek-V4-Flash Patched vLLM Benchmark

Comprehensive benchmark report for running **DeepSeek-V4-Flash** on two NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs using custom patched vLLM images featuring:

- MTP-2 speculative decoding
- DSpark speculative decoding
- FlashInfer CUTLASS MoE support
- DeepSeek-V4 upstream fixes
- Blackwell (SM120) optimizations

---

# Hardware

## Host

- Ubuntu 26.04 LTS
- AMD Threadripper Pro 7965WX
- 24 CPU cores
- ~256 GB RAM

## GPUs

- 2× NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- Tensor Parallel = 2
- PCIe Gen5 x16
- No NVLink
- 450 W power limit per GPU

| GPU | Device |
|---|---|
| GPU0 | RTX 5090 (unused) |
| GPU1 | RTX PRO 6000 |
| GPU2 | RTX PRO 6000 |

---

# Purpose

This document records the software revisions, runtime configuration, patches, startup behavior, benchmark methodology, and measured performance for DeepSeek‑V4‑Flash using custom patched vLLM images.

---

# Benchmark Summary

| Configuration | Best Throughput (tok/s) | Best Scenario |
|---|---:|---|
| **Patched MTP-2** | **2362.2** | Context 0, Concurrency 64 |
| **Patched DSpark** | **1926.71** | Context 8192, Concurrency 64 |
| **Stock vLLM 0.25.0 + FlashInfer 0.6.14** | **1944.42** | Context 0, Concurrency 64 |

Overall ranking:

1. Patched MTP-2
2. Patched DSpark
3. Stock vLLM

---

# Upstream Changes

## vLLM

- PR #48303 — DeepSeek MXFP4 FlashInfer CUTLASS MoE support
- PR #48304 — DeepSeek‑V4 MTP compression / RoPE fix
- PR #48317 — Hybrid KV-cache capacity reporting fix

## FlashInfer

- SM120 sparse MLA Top-K=256 support

## DSpark

- FlashInfer Issue #3828 — DeepSeek‑V4 DSpark compatibility (`draft_id_to_target_id` fix)

---

# MTP‑2 Image

Image

```text
vllm-ds4:v0.25.0-patched
```

Digest

```text
sha256:74fa0eec47faa81a8fca3d9d372494ffeb16a60942cb20088a04db1504bb17f0
```

Included:

- PR #48303
- PR #48304
- PR #48317
- FlashInfer 0.6.14
- NVRTC development headers

---

# MTP‑2 Runtime

Model: `deepseek-ai/DeepSeek-V4-Flash`

Served name: `d4f`

Port: `15004`

Speculative decoding:

```text
method: mtp
num_speculative_tokens: 2
```

Key runtime flags:

```text
--tensor-parallel-size 2
--kv-cache-dtype fp8
--gpu-memory-utilization 0.95
--max-model-len 1048576
--max-num-seqs 64
--max-num-batched-tokens 4096
--enable-prefix-caching
--enable-chunked-prefill
--enable-auto-tool-choice
--tool-call-parser deepseek_v4
--reasoning-parser deepseek_v4
--async-scheduling
--moe-backend flashinfer_cutlass
```

---

# DSpark Image

Image

```text
vllm-ds4:v0.25.0-dspark-topk256
```

Digest

```text
sha256:e6176d085417967353c7e998ec2cc3b46b7edd87221e1aef44150a1076ecc9c1
```

---

# Additional DSpark Changes

In addition to the upstream vLLM pull requests, this image includes the DeepSeek‑V4 compatibility fix documented in **FlashInfer Issue #3828**.

Additional changes:

- FlashInfer SM120 sparse MLA Top‑K=256 support
- DSpark speculative decoding backend
- DeepSeek‑V4 compatibility fix (`draft_id_to_target_id`)
- Reused FlashInfer autotune cache
- Rebuilt against the patched vLLM runtime

Runtime:

```text
method: dspark
num_speculative_tokens: 5
draft_sample_method: greedy
```

Stability adjustment:

- Initial `--gpu-memory-utilization=0.95`
- Reduced to `0.93` after CUDA OOM during extended requests

---

# Benchmark Methodology

Tool: `llm-bench`

Contexts tested: 0, 8192, 16384, 32768, 65536, 131072, 262144

Concurrency: 1, 2, 4, 8, 16, 32, 64

---

# MTP‑2 Throughput (tok/s)

| Context | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
|0|189.9|297.0|349.4|670.0|994.3|1542.1|2362.2|
|8192|194.3|300.8|332.5|607.3|899.9|1303.1|1947.6|
|16384|167.0|304.4|328.8|583.5|884.9|1345.3|1891.3|
|32768|182.7|273.3|322.1|598.6|833.1|1256.4|—|
|65536|184.1|263.5|321.4|569.7|805.8|—|—|
|131072|195.5|256.4|306.3|550.6|—|—|—|
|262144|182.5|261.5|308.0|—|—|—|—|

# MTP‑2 Speculative Metrics

| Metric | Value |
|---|---:|
| Speculative rounds | 422,517 |
| Draft tokens | 845,034 |
| Accepted tokens | 439,151 |
| Generated tokens | 862,227 |
| Acceptance | 51.97% |
| Accepted / round | 1.039 |

## Prefix Cache

| Metric | Value |
|---|---:|
| Queries | 20,848,155 |
| Hits | 19,462,656 |
| Hit Rate | 93.35% |

---

# DSpark Throughput

## Run 1

| Context | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
|0|175.00|252.09|365.14|492.50|729.24|1124.33|1726.52|
|8192|218.07|354.69|456.42|530.06|849.19|1195.57|1919.39|
|16384|201.30|269.24|428.51|508.37|809.93|1265.02|—|
|32768|225.50|260.04|446.88|566.16|709.07|1295.52|—|
|65536|194.69|320.31|393.15|516.97|753.60|—|—|
|131072|177.51|259.43|367.11|528.09|—|—|—|
|262144|194.54|282.76|351.96|—|—|—|—|

## Run 2

| Context | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
|0|177.01|258.48|344.16|503.96|768.82|1129.40|1711.55|
|8192|223.85|294.46|427.49|538.11|883.81|1348.99|1926.71|
|16384|242.70|255.37|387.16|534.97|787.38|1221.22|—|
|32768|219.64|254.33|398.33|480.90|845.08|1142.76|—|
|65536|211.21|301.53|398.25|487.44|803.95|—|—|
|131072|211.15|276.31|371.59|522.83|—|—|—|
|262144|168.19|313.36|373.76|—|—|—|—|

# DSpark Speculative Metrics

Unlike the earlier validation run, the full benchmark logs expose continuous
speculative decoding metrics throughout benchmark execution.

| Metric | Typical Value / Range |
|---|---:|
| Mean acceptance length | 2.2–5.5 tokens |
| Typical draft acceptance | 40–60% |
| Peak draft acceptance | 89.2% |
| Peak mean acceptance length | 5.46 tokens |

Representative steady-state samples:

| Mean Acceptance Length | Avg Draft Acceptance | Accepted Throughput | Draft Throughput |
|---:|---:|---:|---:|
| 3.84 | 56.7% | 422.6 tok/s | 745.4 tok/s |
| 3.72 | 54.4% | 382.9 tok/s | 704.0 tok/s |
| 3.93 | 58.5% | 165.6 tok/s | 283.0 tok/s |
| 4.17 | 63.5% | 179.0 tok/s | 282.0 tok/s |
| 4.33 | 66.6% | 175.4 tok/s | 263.5 tok/s |
| 5.30 | 86.1% | 240.1 tok/s | 279.0 tok/s |
| 5.46 | 89.2% | 63.8 tok/s | 71.5 tok/s |

Per-position acceptance decreases with deeper speculative positions, which is
expected for DSpark. These values were collected from the full benchmark logs
rather than a short validation request.

---

# Stock vLLM 0.25.0 + FlashInfer 0.6.14

| Context | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
|0|182.51|271.99|296.14|575.25|828.93|1288.15|1944.42|
|8192|186.75|275.28|316.51|507.52|743.74|1063.34|1640.74|
|16384|188.54|257.80|280.07|521.11|673.96|1099.87|1593.25|
|32768|183.20|242.14|274.72|497.77|676.83|1055.78|—|
|65536|186.14|239.71|284.75|495.33|701.67|—|—|
|131072|158.76|239.85|271.38|460.93|—|—|—|
|262144|169.48|246.76|256.44|—|—|—|—|

---

# Startup Characteristics

- Model size: 148.66 GiB
- Weight load: 16.47 s
- MTP load: 1.24 s
- CUDA graph capture: 32 s
- Engine initialization: 133.36 s
- KV cache: 1,993,100 tokens
- Max theoretical concurrency at 1M context: 1.90×

Runtime features confirmed:

- Async scheduling
- Prefix caching
- Chunked prefill
- FP8 KV cache
- DeepGEMM FP4 experts
- FlashInfer sparse MLA
- CUDA Graphs
- TileLang kernels

---

# GPU Snapshot

GPU1

- 435.58 W
- 61 °C
- 2737 MHz
- 96,658 MiB
- 99% utilization

GPU2

- 437.43 W
- 66 °C
- 2700 MHz
- 96,688 MiB
- 99% utilization

---

# Stability

## MTP‑2

- No CUDA OOM
- No NCCL failures
- No runtime crashes

## DSpark

- Stable after reducing GPU memory utilization from **0.95 → 0.93**

---

# Future Work

- Include exact `llm-bench` command
- Publish raw benchmark JSON
- Include complete benchmark logs
- Capture full DSpark speculative acceptance statistics
- Add MTP‑2 vs DSpark comparison charts
- Add power efficiency analysis
