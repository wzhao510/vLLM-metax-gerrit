# vLLM-MetaX v0.30.0 User Guide

[中文](/release/0.30.0/zh.md){ .md-button }

> This document is the user guide shipped with vLLM-MetaX v0.30.0, for users and delivery teams deploying large models on MetaX GPUs (MACA).
> Since v0.24.0 every release carries its own guide, all of them collected under `docs/release/`.
>
> Repository: <https://github.com/MetaX-MACA/vLLM-metax>

vLLM-MetaX is the hardware backend plugin for vLLM, following vLLM's hardware-pluggable RFCs (#11162, #19161). This document is the **quick start plus release notes** for the MetaX platform; for general vLLM usage, building from source and accuracy debugging, see the links in "Related documentation" at the end.

## Version compatibility matrix (important)

vLLM-MetaX, mcoplib (the kernel component), the MACA driver and torch are **tightly version-coupled**. Use the matching combination; mixing non-matching versions may cause build failures, accuracy problems or runtime crashes, and is not supported.

| plugin version | maca version | mcoplib version | docker image url |
| :---: | :---: | :---: | :---: |
| v0.30.0 | maca3.9.0.x | 0.4.14 | [vllm-metax:0.30.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.30.0-torch2.10) |

> The complete version mapping and image downloads are in the [Releases table](/getting_started/quickstart.md#releases). Image URL template: `https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.30.0-torch2.10`

Dependency baseline of this release (`requirements/`):

| Component | Version |
| :---: | :---: |
| torch | `2.10.0+metax3.9.0.8` |
| mcoplib | `0.4.14+g5780a78.maca3.9.0.31.torch2.10` |
| deep_ep | `1.0.0+metax3.9.0.11torch2.10` |
| deep_gemm | `0.4.0+maca3.9.0.2` |
| apache-tvm-ffi | `0.1.11+maca` |
| tilelang | `0.1.12+maca` |
| transformers | >= 5.10.4 (see `requirements/common.txt`) |

> The published image tag (of the form `vllm-metax:0.30.0-maca.ai3.9.0.xx-torch2.10-py312-ubuntu22.04-amd64`) follows the actual release page.

## 1. Using vLLM-MetaX on the MetaX platform

You can build your own image or pull the prebuilt one. This document uses the prebuilt image.

### 1.1. Pull the image

```bash
docker pull cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.30.0-maca.ai3.9.0.31-torch2.10-py312-ubuntu22.04-amd64
# Tag rules (see the release page for the authoritative values):
#   0.30.0            -> plugin version
#   maca.ai3.9.0.31   -> MACA driver version
#   torch2.10         -> torch version
#   py312-ubuntu22.04 -> base environment
```

### 1.2. Start the container

```bash
# MetaX GPUs must be passed through with --device=/dev/dri --device=/dev/mxcd
# The service needs its port exposed; --network=host keeps it simple
docker run -it --net=host --uts=host --ipc=host --privileged=true --group-add video \
    --shm-size '100gb' --ulimit memlock=-1 \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    --device=/dev/dri --device=/dev/mxcd \
    -v /root/workspace:/external \
    -v /model_weights:/model_weights \
    --name vllm_metax_test \
    cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.30.0-maca.ai3.9.0.31-torch2.10-py312-ubuntu22.04-amd64
```

> More information: <https://developer.metax-tech.com>
> Building from source, environment variables (MACA_PATH / CUCC_PATH, ...) and build caveats are documented in the [MACA installation guide](/getting_started/installation/maca.md).

## 2. Environment checks

### 2.1. Check that MetaX GPUs are available

Thanks to the CUDA compatibility layer, MetaX devices are checked exactly like NVIDIA GPUs:

```python
import torch
print(torch.cuda.is_available())
# True
```

### 2.2. Check the P2P topology between GPUs

```bash
mx-smi topo -m
# output
=================== MetaX System Management Interface Log ===================
...
Attached GPUs                                     : 8
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
GPU0    X       MX      MX      MX      SYS     SYS     SYS     SYS
...
Legend:
X    = Self
MX   = Connection traversing MetaXLink
SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes
...
```

### 2.3. Check GPU status

```bash
mx-smi
# output
=================== MetaX System Management Interface Log ===================
Attached GPUs                                     : 8
+-------------------------+-----------------+---------------------+----------------------+
| Board Name   | GPU Persist-M | Bus-id              | GPU-Util      sGPU-M |
| Pwr:Usage/Cap| Temp   Perf  | Memory-Usage        | GPU-State            |
|-------------------------+-----------------+---------------------+----------------------|
| 0  MetaX C500 | 0  Off | 0000:0e:00.0 | 0%  Disabled |
| 57W / 350W   | 35C  P0  | 826/65536 MiB | Available            |
...
```

## 3. Running a model

The image already contains matching vLLM-MetaX and mcoplib builds, so the OpenAI-compatible server can be started directly.

### 3.1. Start the vLLM server

```bash
# Qwen3-8B is used as an example; adjust the path to your weights
vllm serve /model_weights/Qwen/Qwen3-8B \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 --port 8000
```

Key lines from a successful startup:

```text
INFO 03-09 15:43:01 [api_server.py] Starting vLLM API server on http://0.0.0.0:8000
INFO 03-09 15:43:05 [core.py] Initializing a V1 LLM engine (v0.30.0) with config: ...
INFO 03-09 15:43:12 [maca_platform.py] MACA backend initialized, driver maca3.9.0.31
INFO 03-09 15:43:35 [gpu_model_runner.py] Model loading took 23.41 GiB and 22.65 seconds
INFO 03-09 15:43:36 [api_server.py] Application startup complete.
```

### 3.2. Send a request

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/model_weights/Qwen/Qwen3-8B",
        "prompt": "MetaX is a",
        "max_tokens": 32
    }'
```

Example response:

```json
{
  "id": "cmpl-xxxxxxxx",
  "object": "text_completion",
  "model": "/model_weights/Qwen/Qwen3-8B",
  "choices": [
    { "text": " company focused on high-performance AI compute, whose MACA software stack is CUDA compatible.", "index": 0, "finish_reason": "length" }
  ],
  "usage": { "prompt_tokens": 4, "completion_tokens": 32, "total_tokens": 36 }
}
```

> For the full set of serving flags, parallel strategies (tensor/pipeline/expert parallel) and quantized loading, see the [vLLM documentation](https://docs.vllm.ai/). Section 3.1 only demonstrates the minimal working example.

### 3.3. Other versions / building from source

Building another version or a customisation requires MACA-compatible packages. The build depends on torch 2.10 (metax3.9.0.8); pass `--no-deps` so the pre-installed PyTorch is not replaced:

```bash
git clone -b mx/v0.30.0-dev https://github.com/MetaX-MACA/vLLM-metax.git
cd vLLM-metax
pip install -r requirements/maca.txt -r requirements/maca_private.txt
pip install -e . --no-deps
```

After changing the environment, check the PyTorch version and availability:

```bash
pip list | grep torch
# torch                    2.10.0+metax3.9.0.8
```

```python
import torch
torch.cuda.is_available()  # True
```

> The full build flow (building vLLM itself with an empty device, what `USE_PRECOMPILED_KERNEL` means, why production must use mcoplib, ...) is in the [MACA installation guide](/getting_started/installation/maca.md).
> The Metax components pinned in `requirements/maca_private.txt` carry build-specific local version numbers; `python tools/sync_maca_private.py` (`--check` to report, `--write` to rewrite in place) aligns them with the current build environment.

## 4. Highlights of v0.30.0

- **Upstream vLLM v0.30.0 sync**
    - Aligned to the vLLM v0.30.0 baseline (upstream `v0.30.0` = `ced6857afa0e`, released 2026-09-22) and synced `patch` / `models` / `registry` / attention backends / quant configs / modular fused moe / platform
- **Upstream v0.30.0 features (from the official release notes, with MetaX status)**
    - **New models**: DeepSeek-V4.1-Flash (whole KV in MXFP8, Mega-mHC, Engram prefetch), DeepSeek-V4-Flash-Vision-Exp, GLM-5.3-Flash (with EPLB), K2-Horizon, Cohere Compass, Bailing V3 VL (with MTP), Nanbeige4.2 (Transformers backend) and a DeepSeek-V4 CPU backend
        - MetaX: **DeepSeek-V4.1 is supported** (new in this release, see section 5); GLM-5.x / GLM-4.6 keep their existing V3.2-style MTP path; the remaining upstream-new architectures are not adapted or tested on MetaX (no weights on the shared storage) and are listed in section 5 as pending verification
    - **Qwen3.8-Flash-Next performance**: separate prefill and decode QSA indexer kernels, fused PLE kernels, FP8 indexer cache, sparse GQA skipping padded indices, fused PLE residual and QSA output gate, UVA PLE offload and Engram TP (`--engram-config`), ...
        - MetaX: **Qwen3.8-Flash-Next (Qwen4Exp) is newly supported in this release**, implemented in this repository (`vllm_metax/models/qwen4_exp/`: QSA sparse attention and indexer, short-conv, PLE layer, hyper-connection, MTP and their tilelang / triton kernels). The CUDA-specific switches of the upstream implementation (Engram UVA offload, SM100 LL-GEMM, ...) do not apply to MACA
    - **Model Runner V2**: dual-batch overlap in eager mode and with FULL CUDA graphs, MTP / EAGLE3 / DFlash / DSpark speculation under pipeline parallelism, adaptive verification from an online acceptance estimator, GC frozen during graph capture (capture 12s -> 2s on H200), GPU-side compaction of `--return-sampling-mask`
        - MetaX: synced with the baseline; `patch/enhancement/dbo.py` and `patch/performance/speculative_decode_perf.py` were re-reviewed against the new implementation
    - **Fast Start**: a persistent per-GPU weight-cache daemon with `--load-format ipc_cache` (covering FP4 checkpoints and multi-node TP)
        - MetaX: `[fused_moe] upstream fast start changes` is synced with the baseline; its benefit and stability on MACA are unverified
    - **Watermarking / HiSparse**: Gumbel-max watermarked generation (compatible with speculative decoding, with a detection endpoint), host-resident KV tier for sparse-MLA decode (`HiSparseConnector`)
        - MetaX: the CUDA implementations of watermarking and HiSparse are not adapted; sampling now uses the mcoplib Gumbel kernel (below), the watermarking path is unverified
    - **Kimi K3 performance**: AttnRes on by default, KDA mixed-batch gather/scatter removed, grouped FP8 MLA cache insertion, DSV3 low-latency GEMM, FlashInfer KDA kernels, ...
        - MetaX: the K3-specific paths are not adapted; the KDA plugin patches (`bugfix/triton_support/kda.py`, `chunk_delta_h.py`) are kept for the MACA limits
    - **Large-scale serving**: PCP + DCP on sparse-MLA models (#56157), PCP with single-module MTP and replicated DSpark, Elastic EP reusing CUDA graphs, DeepEP v2 async finalize, Mooncake / NIXL encoder-cache sharing, ...
        - MetaX: this release adds DCP + PCP support for `flashmla_sparse` BF16 together with an upstream hotfix; Elastic EP and encoder-cache sharing depend on CUDA-side transports and are unverified
    - **Quantization**: targeted online quantization via `quantization_config.targets`, online quantization of partially pre-quantized checkpoints, W4A16 DSA with the `nvfp4_fp8_ds_mla` KV cache, NVFP4 across backends, AutoRound for every bit width, DeepSelect top-k for the DSA sparse indexer
        - MetaX: this release adds MXFP4 / INT8 support for DeepSeek-V4.1, an out-of-tree QuantFP8 registry and the MACA `indexer_topk_backend` (DeepSelect); the CUDA-only NVFP4 / AutoRound paths do not apply
    - **Upstream breaking changes**: scale-out endpoints are opt-in via `--enable-scale-out`; GPTQ activation reordering (`g_idx`) removed; env vars deprecated in 0.29 (`VLLM_PREFIX_CACHE_RETENTION_INTERVAL`, `VLLM_MM_HASHER_ALGORITHM`, ...) removed; the `all` Mamba cache mode deprecated; `python -m vllm.entrypoints.grpc_server` replaced by `vllm serve --grpc`; YaRN aligned with Transformers so vendor YaRN aliases no longer re-scale `max_model_len`
        - MetaX: services using those entry points, env vars or YaRN aliases must be adjusted before upgrading (section 6)
- **MetaX changes in this release (relative to the v0.29.0 branch)**
    - **DeepSeek-V4.1**: `[deepseek_v41][1/N] support deepseek_v41`, `[deepseek_v41][2/N] update mxfp4 and int8 support` (model, attention, quantization configs, DSpark/MTP); `[dsv4][quant_config] unify dsv4/dsv41 quant_config`, `[dsv4][fp8] add QuantFP8 registry for oot`, `[dsv4] process wo_a fp8 shape for bmm deepgemm`
    - **Qwen3.8-Flash-Next**: `[Model] Support Qwen3.8-Flash-Next` (registrations + the full `qwen4_exp` operator tree + startup hooks)
    - **Attention / MLA**: `[attention] update attention impl for v0.30.0`, `[attention] support flashmla_sparse bf16 dcp+pcp`, `[BugFix] flashmla_sparse dcp + pcp upstream hotfix`, `fix GQA dcp kernel trap`
    - **FusedMoE / mctlass**: `[fused_moe] upstream fast start changes`, `[fused_moe] remove wNa16 cuda path since it has migrated into mctlass`, `[oracle] remove invalid moe_backend`, `[ModularKernel] remove redundant code`
    - **TileLang / MHC**: `[Tilelang][mhc] update tilelang kernels`, `[mhc] add mhc kernel tests`, `[tilelang] strict compute_num_split to specific bounds`
    - **Distributed / communication**: `[Distribute] add strict rules for AgRs dispatch`, `[Mccl] add dummy mccl API for compatibility`
    - **Model fixes**: `[models][step3p5] layer_types out of indexes if MTP`, `[models][step3p5_mtp] MTP weight loading error`, `[model][deepseek_v4]` / `[models][deepseek_v2] update to v0.30.0`, `[model][dsv2] update ds mtp`, `[issue]: Fix Dsv32 mtp model indexer forward path patch missing`
    - **Patches ported from v0.29.0** (carried by this release): the dsv4 top-k indexer block-table out-of-bounds guard, the per-channel scale fix for transposed fused Qwen3-VL RoutedExperts checkpoints, the MoE-LoRA shared-memory cap, AutoAWQ MoE `match_mode`, SWIGLUSTEP activation support (including the fused FP8 block quant path), the msprobe worker-selection log line, and `tools/sync_maca_private.py` with `maca_private.txt` aligned to the build container
    - **Performance and defaults**: `[performance] Apply optimized gumbel_sample_kernel` (mcoplib Gumbel sampling, with a unit test); MACA disables `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` by default (section 6); new `VLLM_METAX_USE_QUANT_MAIN_PROJ` switch for `main_proj` quantization
    - **Additional model support**: CosyVoice2 speech-token LM, the `DeepseekV32MTPModel` registration (GLM-5.2 and friends), JoyAI-LLM-Flash Chinese encoding and MTP fixes, MiniMax-M3-VL video-processor fix
    - **Patch infrastructure**: `[patch] reviewed v0.30.0 patches and AUDIT.md` (full patch review plus the audit table), `[Patch] fix patch not take effect correctly`; templates and README updated (`@patch` must be the outermost decorator, alias synchronization rules)
    - **Build**: `[chores] prepare for v0.30.0 update` (common / test dependency refresh), `[build] support non-standard maca version`
- **Tools / engineering**
    - New dummy models and smoke-test tooling (`[tools]` / `[skills]`) to exercise the loading path without weights on disk
    - Patch layout rules: only the `bugfix` / `enhancement` / `performance` top-level directories are imported automatically; patches must use the `@patch` decorator and document purpose, added version and removal condition, and the keep/remove decision is recorded in the neighbouring `AUDIT.md`

## 5. New and updated models

The following models are new or materially updated in v0.30.0 (some are not yet backfilled into `docs/models/supported_models.md`; this guide is authoritative):

### Text / reasoning models

- **Qwen3.8-Flash-Next (Qwen4Exp)**: **newly supported** in this release. Registers `Qwen4ExpForCausalLM` / `Qwen4ExpForConditionalGeneration` / `Qwen4ExpMTP`; the MetaX implementation provides its own QSA (sparse attention and indexer), short-conv (PLE convolution), PLE layer, hyper-connection, MTP and the corresponding tilelang / triton kernels. MTP with speculative decoding and CUDA graph capture run through this repository's implementation and still require the checks in section 8
- **DeepSeek-V4.1 / DeepSeek-V4**: **DeepSeek-V4.1 is newly supported** in this release (model, attention, MXFP4 / INT8 quantization, DSpark / MTP), and the FP8 quantization config of dsv4 / dsv41 is unified (`registry/quant_config/deepseek_fp8.py`, registration name `deepseek_v4_fp8`); note that tp must divide the checkpoint's `o_groups` (Flash=8, Pro=16)
- **DeepSeek-V4 / V4.1 DSpark**: the draft-model state and MTP are updated for 0.30 (`models/deepseek_v4/metax/dspark.py`, `models/deepseek_v41/metax/*`); `main_proj` quantization is controlled by `VLLM_METAX_USE_QUANT_MAIN_PROJ` (default 1, i.e. quantized)
- **GLM-5.x / GLM-4.6**: the `DeepseekV32MTPModel` -> `vllm_metax.models.deepseek_mtp:DeepSeekMTP` registration is kept so that V3.2-style MTP drafts (`deepseek_v32` / `glm_moe_dsa`) use the MetaX sparse indexer instead of the upstream implementation, which always quantizes the indexer Q to FP8
- **MiniMax-M3 / M3-VL**: weight loading, index-score kernels and sparse-attention patches re-reviewed for 0.30; a new video-processor fix (Transformers >= 5.19 pops the required `do_convert_rgb` argument) repairs the `profile_run` failure during engine startup
- **Step-3.5-Flash**: `layer_types` out-of-range and MTP weight-loading fixes
- **JD JoyAI-LLM-Flash**: Chinese encoding fix (restores the `tokenizer.json` backend under Transformers v5) and MTP fix (draft config rewrite)
- **CosyVoice2**: **newly supported** speech-token language model (`vllm_metax/models/cosyvoice2.py`, reusing the Qwen2 trunk with a biased speech head)

### Multimodal models

- **Qwen3-VL-MoE (transposed fused W8A8 and similar checkpoints)**: the per-channel scale fix is carried by this release (`bugfix/routed_experts_fused_weight_fix.py`)
- **MiniMax-M3-VL**: see above (video-processor fix)
- **Qwen3.8-Flash-Next (VL)**: `Qwen4ExpForConditionalGeneration` is registered; the vision quantization ops (`metax/vision_quantization.py`) ship with the model
- The vision architectures added upstream in v0.30 (DeepSeek-V4-Flash-Vision-Exp, Bailing V3 VL) are unverified on MetaX

### Quantized model support

- FP8 (including the unified `deepseek_v4_fp8` registration), INT8 W8A8, W4A8, MXFP4 / MXFP8, AWQ / GPTQ / compressed-tensors, on top of the existing implementations; this release adds the out-of-tree QuantFP8 registry and the DeepSeek-V4.1 MXFP4 / INT8 combination
- `w8a8_int8` only runs on the EP (deep_ep) path

### Upstream v0.30.0 new / migrated models (pending MetaX verification)

The following architectures are new or migrated upstream and enter this repository with the baseline, but are **not yet adapted or tested on MetaX** (and have no weights on the shared storage). They are listed here so they can be covered later:

- **DeepSeek-V4-Flash-Vision-Exp** (vision variant of DeepSeek-V4)
- **GLM-5.3-Flash** (with EPLB)
- **K2-Horizon** (with reasoning / tool parsers)
- **Cohere Compass**
- **Bailing V3 VL** (Ling 3.0 Flash VL, with MTP)
- **Nanbeige4.2** (Transformers backend)
- **FunASR** (speech)
- **DeepSeek-V4 CPU backend** (AVX512/AMX sparse MLA / indexer / mHC / compressor): not applicable on MACA

> The full list of tested models and feature status (LoRA, PP, ...) is in [Supported models](/models/supported_models.md).
> To add a model outside that list, see the [model registration guide](/developing/model/registration.md).

## 6. Migrating from v0.29

1. **Switch image / environment**: docker users move to the v0.30.0 image; source users check out `mx/v0.30.0-dev` again and reinstall the dependencies as in 3.3.
2. **torch / MACA stay on the 3.9 baseline**: v0.30.0 still uses torch 2.10 (metax3.9.0.8), but mcoplib moves from `0.4.13+g78bea36.maca3.9.0.29.torch2.10` to `0.4.14+g5780a78.maca3.9.0.31.torch2.10`, and `lmcache` / `nixl` / `mooncake-transfer-engine` move to `maca3.9.0.0`; locally built csrc must match the new mcoplib kernel symbols or the runtime reports missing symbols.
3. **CUDA graph memory profiling is disabled by default**: on MACA, `platform.py` uses `maybe_override_vllm_env` to set `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` to 0, so the KV cache budget no longer subtracts the measured CUDA graph pool (matching v0.28, whereas v0.29 kept the upstream default of 1). An explicit environment value wins; if you need a larger KV cache, raise `--gpu-memory-utilization` first.
4. **Unified DeepSeek-V4 / V4.1 quantization config**: `deepseek_v4_fp8` is now served by `registry/quant_config/deepseek_fp8.py::MacaDeepseekV41FP8Config` (v0.29 used `MacaDeepseekV4FP8Config` in `fp8.py`) and is shared by dsv4 and dsv41 checkpoints; deployments that only name the quantization method need no change.
5. **New environment variable**: `VLLM_METAX_USE_QUANT_MAIN_PROJ` (default 1) controls whether the DSpark `main_proj` uses quantized weights; set it to 0 to keep full precision.
6. **Hunyuan V1 / HunyuanOCR on the Transformers backend** (since v0.29): on MACA, CUDA graph capture fails with `cudaErrorStreamCaptureImplicit` because HF's dynamic RoPE (`rope_scaling.type=dynamic`) performs an implicit sync; add `--enforce-eager`. This was measured on v0.29 and needs a re-check on 0.30.
7. **LoRA MoE small-batch fast path** (measured on v0.29): with LoRA enabled on MoE models such as `Qwen3-30B-A3B`, upstream's `_fused_moe_lora_small_batch_kernel` traps with `Xnack Error/ATU Fault(0x8)` during capture on MACA. Workaround: set `--max-lora-rank` above 64 (for example 128) to skip the fast path; this release also carries the shared-memory cap for the one-shot path (section 8).
8. **Upstream breaking changes**:
    - Ten architectures remain removed (since v0.29): Arctic, Fairseq2Llama, GritLM, MPT, Falcon-RW, StableLM Epoch, Chameleon, Cheers, HyperCLOVAX-SEED-Vision, FireRedLID; services relying on them must fall back to an older plugin or migrate;
    - scale-out endpoints are opt-in via `--enable-scale-out` (no longer controlled by `VLLM_ENABLE_SCALE_OUT_ENDPOINTS`);
    - GPTQ activation reordering (`g_idx`) is removed; env vars deprecated in 0.29 (`VLLM_PREFIX_CACHE_RETENTION_INTERVAL`, `VLLM_MM_HASHER_ALGORITHM`, ...) are removed; the `all` Mamba cache mode is deprecated; `python -m vllm.entrypoints.grpc_server` is replaced by `vllm serve --grpc`;
    - YaRN is aligned with Transformers: vendor YaRN aliases no longer re-scale `max_model_len`, so checkpoints using custom rope_scaling aliases need their context length re-checked.
9. **Dependency layout**: the Metax components pinned in `requirements/maca_private.txt` carry build-specific local version numbers; `tools/sync_maca_private.py` (`--check` / `--write`) aligns them with the build container.
10. **New defaults**: `max_num_batched_tokens` defaults to 16384, Mamba models enable prefix caching by default (since v0.28), FlashInfer all-reduce is on by default for TP CUDA groups (`VLLM_ALLREDUCE_USE_FLASHINFER=0` opts out), and prefix-cache `NONE_HASH` is deterministic. Align these before comparing performance against v0.29 and earlier.
11. **Recommended regressions after upgrading**: Qwen3.8-Flash-Next (first MACA support), DeepSeek-V4 / V4.1 (including MTP / DSpark and the new FP8 registration), GLM-5.x MTP, MiniMax-M3-VL, JoyAI-LLM-Flash (encoding / MTP) and CosyVoice2, plus the eagle / MTP proposal accuracy.

## 7. Differences between Metax and NVIDIA CUDA

Metax matches NVIDIA on most interfaces, but some software behaviours and environment variables differ.

### 7.1. MACA_MPS_MODE (sharing a GPU between processes)

By default MACA does not let several processes share one GPU: if the GPU is already in use, a new process cannot start.

To enable MPS-like (Multi-Process Service) behaviour, set `MACA_MPS_MODE=1`:

```bash
export MACA_MPS_MODE=1
vllm serve /model_weights/Qwen/Qwen3-8B --tensor-parallel-size 1
```

### 7.2. Multi-node deployment

For vLLM distributed inference (tensor/pipeline/expert parallel) across nodes, set the following environment variables so that inter-node communication works:

- `MCCL_SOCKET_IFNAME`: network interface used by MCCL
- `GLOO_SOCKET_IFNAME`: network interface used by GLOO
- `MCCL_IB_HCA`: InfiniBand devices to use

Use `ifconfig` and `mx-smi topo -n` to identify the NICs and IB devices (see the topology examples in the verl / Megatron documentation). Recommended settings:

```bash
export MCCL_SOCKET_IFNAME=ens20f0np0
export GLOO_SOCKET_IFNAME=ens20f0np0
export MCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
```

> ⚠️ Adjust these for your own machine using the `ifconfig` and `mx-smi topo -n` output; do not copy the commands blindly.

## 8. Known issues and workarounds

The table covers the problems measured on MACA for this release and long-standing limits. Rows marked "measured on v0.29" come from the previous release and still need a re-check on v0.30.0; rows marked "pending" are new capabilities of this release that have not yet run with real weights.

| Issue | Impact | Workaround / status |
| ------ | ------ | ------------ |
| Qwen3.8-Flash-Next (Qwen4Exp) | Pending: newly supported in this release, but the full path (QSA sparse attention, PLE / short-conv, MTP, CUDA graph capture) has not yet run on MACA with real weights | Cover it first once weights are available; upstream changed AttentionImpl variant discovery (`find_attention_impl_variant`), so check that path first if startup or forward breaks |
| The four patches ported in this release (dsv4 top-k guard, MoE-LoRA shared-memory cap, mcoplib Gumbel sampling, Qwen3-VL scale normalization) | Pending: each was adapted to the v0.30.0 upstream implementation but has not had a runtime smoke test on v0.30.0 | Before release, run DeepSeek-V4 / V4.1 decode after a request-geometry change, a Qwen3-VL-MoE W8A8 load, and `Qwen3-MoE + --enable-lora` once each |
| Upstream v0.30.0 new / migrated models (DeepSeek-V4-Flash-Vision-Exp, GLM-5.3-Flash, K2-Horizon, Cohere Compass, Bailing V3 VL, Nanbeige4.2, FunASR) | Unverified on MetaX, no weights on the shared storage | Registered as placeholders in the model list; to be covered once weights are available |
| Upstream features Watermarking (Gumbel-max), HiSparse and `--load-format ipc_cache` | CUDA-side implementations, not adapted or verified on MACA | Ask the PDE/AI team before relying on them; sampling itself now runs on the mcoplib Gumbel kernel |
| Hunyuan V1 / HunyuanOCR on the Transformers modeling backend (measured on v0.29) | During CUDA graph capture HF's `dynamic_frequency_update` syncs implicitly and raises `cudaErrorStreamCaptureImplicit: operation would make the legacy stream depend on a capturing blocking stream` (measured: Hunyuan-7B-Instruct, tp1) | Add `--enforce-eager` for those models; a proper fix needs a MACA-side patch or bypassing the dynamic frequency update during capture |
| LoRA MoE small-batch fast path (`_fused_moe_lora_small_batch_kernel`, measured on v0.29) | The upstream Triton kernel traps with `Xnack Error/ATU Fault(0x8)` during capture on MACA (measured: Qwen3-30B-A3B-lora, tp2, `--max-lora-rank 64`) | Use `--max-lora-rank 128` (> 64) to skip the fast path; the one-shot path's shared-memory cap is handled by this release's patch |
| MLA + MTP models such as DeepSeek-R1-W48A8-PCPT_NAIVE (measured on v0.29) | `flash_fwd_splitkv_combine_kernel` traps with an ATU fault during the warmup decode step (`CUDA error: an illegal memory access was encountered`) | v0.29-line regression (the same model passes with 0.28.0+gd14be0); to be re-checked on 0.30.0, do not add it to the test list before then |
| Per-channel scales of transposed fused RoutedExperts checkpoints (PR #50137 follow-up) | Transposed fused models such as Qwen3-VL-MoE W8A8 fail to load | Upstream only handles the fused weights, so `bugfix/routed_experts_fused_weight_fix.py` is still required (shipped in this release) |
| `minimax_qk_norm` all-reduce fusion disabled by default | The fused op is off by default and MiniMax-M3 performance is a little lower | Enable it through the `rms_norm_tp` configuration and re-check accuracy if needed |
| Ten architectures removed upstream | Services still using Arctic / MPT / Falcon-RW / StableLM Epoch / Chameleon / Cheers / HyperCLOVAX-SEED-Vision / FireRedLID / GritLM / Fairseq2-Llama cannot start | Fall back to an older plugin version or migrate to a supported architecture |
| MACA flash attention is FA2 only (`is_fa_version_supported` returns 2) | Upstream code paths that require FA3, such as FP8 KV cache, have no MetaX implementation | The startup check reports them as unsupported; FP8 KV cache is unavailable for now |
| DeepSeek-V4 requires tp to divide the checkpoint's `o_groups` (Flash=8, Pro=16) | Starting Flash with tp16 crashes in `profile_run` with `ZeroDivisionError: n_local_heads // n_local_groups` | Use tp8 x pp2 or tp8 x dp2 for a 16-card two-node deployment; V4-Pro (o_groups=16) is fine with tp16 |
| Locally built csrc versus the prebuilt mcoplib | Local builds must match the kernel symbols of mcoplib `0.4.14+g5780a78.maca3.9.0.31` | Production should use the prebuilt mcoplib combination |
| tilelang / tvm-ffi / xgrammar are tightly coupled | Mixing versions can break kernel loading or grammars | Install exactly the combination in `requirements/` (tilelang `0.1.12+maca`, apache-tvm-ffi `0.1.11+maca`) |
| Quantized paths such as deep_ep / mctlass / W4A8 | Accuracy still needs a regression | Use msprobe for accuracy comparisons ([guide](/developing/msprobe/msprobe_guide.md)) |
| Some tokenizers under Transformers v5 (Moonlight / Kimi / JoyAI, ...) | The old import path (`bytes_to_unicode`) is gone, and JoyAI can lose the backend stored in `tokenizer.json` | This release ships `bugfix/bytes_to_unicode.py` and `bugfix/tokenizer.py` (including the JoyAI backend restoration); pin the Transformers version if problems persist |
| `max_num_batched_tokens` defaults to 16384 and Mamba models enable prefix caching by default | Throughput and latency baselines from v0.27 and earlier are not directly comparable | Fix the flags explicitly when comparing |
| v0.16.0 was skipped upstream | The version does not exist | Do not use v0.16.x |

## 9. Related documentation

- **Release mapping table**: [Releases](/getting_started/quickstart.md#releases)
- **Supported models**: [Supported Models](/models/supported_models.md)
- **Installing / building from source**: [MACA installation guide](/getting_started/installation/maca.md)
- **Accuracy debugging (msprobe)**: [msprobe guide](/developing/msprobe/msprobe_guide.md)
- **Patch management (monkey / git patches)**: [Patches](/developing/patches/README.md)
- **Adding custom / new models**: [Model registration guide](/developing/model/registration.md)
- **Plugin architecture and contributing**: [Contributing / plugin system](/developing/README.md)
- mcoplib repository: <https://github.com/MetaX-MACA/mcoplib>
- Plugin home / source: <https://github.com/MetaX-MACA/vLLM-metax>
- Developer community / image downloads: <https://developer.metax-tech.com>
- vLLM documentation: <https://docs.vllm.ai/>

---
Release date: TODO (the actual release date applies) | Maintained by: PDE/AI vLLM-MetaX team
