# vLLM-MetaX v0.30.0 使用指导手册

[English](/release/0.30.0/en.md){ .md-button }

> 本文是 vLLM-MetaX v0.30.0 版本随版发布的用户指导手册，面向在 MetaX GPU（MACA）上部署大模型的用户与交付团队。
> 从 v0.24.0 起，每个发布版本都会附一份对应的指导手册，统一记录在 `docs/release/` 下。
>
> 仓库地址：<https://github.com/MetaX-MACA/vLLM-metax>

vLLM-MetaX 是 vLLM 的硬件后端插件（plugin），遵循 vLLM 的硬件可插拔 RFC（#11162、#19161），以插件形式接入上游 vLLM。本文是 MetaX 平台上的**快速上手与版本说明**，通用 vLLM 用法、从源码构建、精度调测等深入内容请参考文末「相关文档」中的对应链接。

## 版本兼容矩阵（重要）

vLLM-MetaX、mcoplib（内核组件）、MACA 驱动、torch 四者**版本强耦合**，请严格使用匹配组合，混用非对应版本可能导致编译失败、精度异常或运行期崩溃，且不保证可支持。

| plugin version | maca version | mcoplib version | docker image url |
| :---: | :---: | :---: | :---: |
| v0.30.0 | maca3.9.0.x | 0.4.14 | [vllm-metax:0.30.0](https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.30.0-torch2.10) |

> 完整版本映射与镜像下载见 [Releases 总表](/getting_started/quickstart.md#releases)。镜像 URL 模板：`https://developer.metax-tech.com/softnova/docker?package_name=vllm-metax:0.30.0-torch2.10`

本版本的依赖基线（`requirements/`）：

| 组件 | 版本 |
| :---: | :---: |
| torch | `2.10.0+metax3.9.0.8` |
| mcoplib | `0.4.14+g5780a78.maca3.9.0.31.torch2.10` |
| deep_ep | `1.0.0+metax3.9.0.11torch2.10` |
| deep_gemm | `0.4.0+maca3.9.0.2` |
| apache-tvm-ffi | `0.1.11+maca` |
| tilelang | `0.1.12+maca` |
| transformers | >= 5.10.4（见 `requirements/common.txt`） |

> 发布镜像 tag（形如 `vllm-metax:0.30.0-maca.ai3.9.0.xx-torch2.10-py312-ubuntu22.04-amd64`）以实际发布页为准。

## 1. 在 Metax 平台上使用 vLLM-MetaX

你可以选择构建自己的镜像，也可以直接拉取已有的预构建镜像。本文以拉取预构建镜像为例。

### 1.1. 拉取镜像

```bash
docker pull cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.30.0-maca.ai3.9.0.31-torch2.10-py312-ubuntu22.04-amd64
# 示例 tag 规则（以实际发布页为准）：
#   0.30.0            -> 插件版本
#   maca.ai3.9.0.31   -> MACA 驱动版本
#   torch2.10         -> torch 版本
#   py312-ubuntu22.04 -> 基础环境
```

### 1.2. 启动容器

```bash
# 必须通过 --device 挂载 Metax GPU 设备：--device=/dev/dri --device=/dev/mxcd
# 推理服务需要暴露端口，这里用 --network=host 方便直接访问
docker run -it --net=host --uts=host --ipc=host --privileged=true --group-add video \
    --shm-size '100gb' --ulimit memlock=-1 \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    --device=/dev/dri --device=/dev/mxcd \
    -v /root/workspace:/external \
    -v /model_weights:/model_weights \
    --name vllm_metax_test \
    cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.30.0-maca.ai3.9.0.31-torch2.10-py312-ubuntu22.04-amd64
```

> 更多信息请参考官方文档：<https://developer.metax-tech.com>
> 从源码构建、环境变量（MACA_PATH / CUCC_PATH 等）与构建注意事项，详见 [MACA 安装指南](/getting_started/installation/maca.md)。

## 2. 环境检查

### 2.1. 检查 Metax GPU 是否可用

得益于与 CUDA 的兼容性，可以像使用 NVIDIA GPU 一样检查 Metax 设备是否可用：

```python
import torch
print(torch.cuda.is_available())
# True
```

### 2.2. 检查 GPU 之间的 P2P 连接拓扑

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

### 2.3. 查看 GPU 状态

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

## 3. 运行示例

镜像内已包含匹配版本的 vLLM-MetaX 与 mcoplib，可直接启动 OpenAI 兼容推理服务。

### 3.1. 启动 vLLM 推理服务

```bash
# 以 Qwen3-8B 为例，请根据实际模型路径修改
vllm serve /model_weights/Qwen/Qwen3-8B \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 --port 8000
```

服务启动成功的关键日志（节选）：

```text
INFO 03-09 15:43:01 [api_server.py] Starting vLLM API server on http://0.0.0.0:8000
INFO 03-09 15:43:05 [core.py] Initializing a V1 LLM engine (v0.30.0) with config: ...
INFO 03-09 15:43:12 [maca_platform.py] MACA backend initialized, driver maca3.9.0.31
INFO 03-09 15:43:35 [gpu_model_runner.py] Model loading took 23.41 GiB and 22.65 seconds
INFO 03-09 15:43:36 [api_server.py] Application startup complete.
```

### 3.2. 发送推理请求

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/model_weights/Qwen/Qwen3-8B",
        "prompt": "MetaX 是一家",
        "max_tokens": 32
    }'
```

返回示例：

```json
{
  "id": "cmpl-xxxxxxxx",
  "object": "text_completion",
  "model": "/model_weights/Qwen/Qwen3-8B",
  "choices": [
    { "text": "专注于高性能 AI 计算的芯片公司，其 MACA 软件栈兼容 CUDA 生态。", "index": 0, "finish_reason": "length" }
  ],
  "usage": { "prompt_tokens": 4, "completion_tokens": 32, "total_tokens": 36 }
}
```

> 更多启动参数、并行策略（tensor/pipeline/expert parallel）、量化加载等通用用法，参考 [vLLM 官方文档](https://docs.vllm.ai/)。本手册 3.1 仅演示最小可运行示例。

### 3.3. 使用其他版本 / 从源码构建

如需构建其他版本或定制，需使用与 MACA 兼容的软件包。编译依赖 torch 2.10（metax3.9.0.8），建议使用 `--no-deps` 避免覆盖环境内已有 PyTorch：

```bash
git clone -b mx/v0.30.0-dev https://github.com/MetaX-MACA/vLLM-metax.git
cd vLLM-metax
pip install -r requirements/maca.txt -r requirements/maca_private.txt
pip install -e . --no-deps
```

环境变更后请检查 PyTorch 版本及其可用性：

```bash
pip list | grep torch
# torch                    2.10.0+metax3.9.0.8
```

```python
import torch
torch.cuda.is_available()  # True
```

> 完整构建流程（含 vllm 本体用 empty device 构建、USE_PRECOMPILED_KERNEL 含义、生产必须用 mcoplib 等注意事项）见 [MACA 安装指南](/getting_started/installation/maca.md)。
> `requirements/maca_private.txt` 里的 Metax 组件带构建特定的本地版本号，可用 `python tools/sync_maca_private.py`（`--check` 只检查、`--write` 就地改写）按当前构建环境对齐。

## 4. v0.30.0 本版本重点新特性

- **同步上游 vLLM v0.30.0**
    - 对齐 vLLM v0.30.0 基线（上游 `v0.30.0` = `ced6857afa0e`，2026-09-22 发布），并同步 `patch` / `models` / `registry` / attention backends / quant configs / modular fused moe / platform 等模块
- **上游 v0.30.0 官方新特性（摘自官方 release notes，附 MetaX 侧状态）**
    - **新模型**：DeepSeek-V4.1-Flash（MXFP8 全量 KV、Mega-mHC、Engram 预取）、DeepSeek-V4-Flash-Vision-Exp、GLM-5.3-Flash（含 EPLB）、K2-Horizon、Cohere Compass、Bailing V3 VL（含 MTP）、Nanbeige4.2（transformers backend），以及 DeepSeek-V4 的 CPU 后端
        - MetaX：**DeepSeek-V4.1 已支持**（本版本新增，见第 5 节）；GLM-5.x / GLM-4.6 走既有 V3.2 风格 MTP 路径；其余上游新增架构未适配/未跑测（本地共享存储暂无权重），登记在第 5 节待验证
    - **Qwen3.8-Flash-Next 性能**：prefill / decode 分离的 QSA indexer kernel、融合 PLE kernel、FP8 indexer cache、稀疏 GQA 跳过 padding index、PLE residual 与 QSA output gate 融合、UVA PLE offload 与 Engram TP（`--engram-config`）等
        - MetaX：**本版本新增支持 Qwen3.8-Flash-Next（Qwen4Exp）**，模型与算子由本仓库独立实现（`vllm_metax/models/qwen4_exp/`：QSA / short-conv / PLE / hyper-connection / MTP 及对应 tilelang・triton 算子）；上游的 CUDA 专属开关（Engram UVA offload、SM100 LL-GEMM 等）不适用 MACA
    - **Model Runner V2**：eager 与 FULL CUDA graph 下的双批次重叠、PP 下的 MTP / EAGLE3 / DFlash / DSpark 投机、按在线接受率估计的自适应验证、capture 期间冻结 GC（H200 上 capture 12s → 2s）、GPU 侧压缩 `--return-sampling-mask`
        - MetaX：随基线同步；`patch/enhancement/dbo.py`、`patch/performance/speculative_decode_perf.py` 已按新实现复核
    - **Fast Start**：常驻权重缓存 daemon + `--load-format ipc_cache`（覆盖 FP4 checkpoint 与多机 TP）
        - MetaX：`[fused_moe] upstream fast start changes` 随基线同步，MACA 上的收益与稳定性待验证
    - **Watermarking / HiSparse**：Gumbel-max 水印（含投机解码兼容与检测端点）、稀疏 MLA decode 的 host 侧 KV 分层（`HiSparseConnector`）
        - MetaX：水印与 HiSparse 的 CUDA 实现未适配；采样侧本版本改用 mcoplib 的 Gumbel kernel（见下），水印能力待验证
    - **Kimi K3 性能**：AttnRes 默认开启、KDA 混合 batch 去 gather/scatter、分组 FP8 MLA cache 插入、DSV3 低延迟 GEMM、FlashInfer KDA kernel 等
        - MetaX：K3 专用路径未适配；KDA 相关插件补丁（`bugfix/triton_support/kda.py`、`chunk_delta_h.py`）仍按 MACA 限制保留
    - **大规模部署**：稀疏 MLA 的 PCP + DCP（#56157）、PCP 下单模块 MTP 与复制 DSpark、Elastic EP 复用 CUDA graph、DeepEP v2 异步 finalize、Mooncake / NIXL encoder cache 等
        - MetaX：本版本补上 `flashmla_sparse` bf16 的 DCP + PCP 支持并叠加上游 hotfix；Elastic EP / encoder-cache 共享等依赖 CUDA 侧传输栈的能力未验证
    - **量化**：`quantization_config.targets` 定向在线量化、部分预量化 checkpoint 的在线量化、W4A16 DSA + `nvfp4_fp8_ds_mla` KV cache、NVFP4 多后端、AutoRound 全位宽、DSA 稀疏 indexer 的 DeepSelect top-k
        - MetaX：本版本新增 DeepSeek-V4.1 的 MXFP4 / INT8 支持、OOT QuantFP8 registry、MACA 侧 `indexer_topk_backend`（DeepSelect）；CUDA 专属 NVFP4 / AutoRound 路径不适用
    - **上游破坏性变更**：scale-out 端点改为 `--enable-scale-out` opt-in；GPTQ 的 `g_idx`（激活重排）移除；0.29 起弃用的 `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`、`VLLM_MM_HASHER_ALGORITHM` 等环境变量移除；Mamba cache `all` 模式弃用；`python -m vllm.entrypoints.grpc_server` 改为 `vllm serve --grpc`；YaRN 与 transformers 对齐（厂商别名不再二次缩放 `max_model_len`）
        - MetaX：使用上述入口 / 环境变量 / YaRN 别名的服务升级前需先改造（第 6 节）
- **Metax 侧本版本改动（相对 v0.29.0 分支）**
    - **DeepSeek-V4.1**：`[deepseek_v41][1/N] support deepseek_v41`、`[deepseek_v41][2/N] update mxfp4 and int8 support`（模型、attention、量化配置、DSpark/MTP 全套）；`[dsv4][quant_config] unify dsv4/dsv41 quant_config`、`[dsv4][fp8] add QuantFP8 registry for oot`、`[dsv4] process wo_a fp8 shape for bmm deepgemm`
    - **Qwen3.8-Flash-Next**：`[Model] Support Qwen3.8-Flash-Next`（模型注册 + `qwen4_exp` 全量算子 + 启动期 hook）
    - **Attention / MLA**：`[attention] update attention impl for v0.30.0`、`[attention] support flashmla_sparse bf16 dcp+pcp`、`[BugFix] flashmla_sparse dcp + pcp upstream hotfix`、`fix GQA dcp kernel trap`
    - **FusedMoE / mctlass**：`[fused_moe] upstream fast start changes`、`[fused_moe] remove wNa16 cuda path since it has migrated into mctlass`、`[oracle] remove invalid moe_backend`、`[ModularKernel] remove redundant code`
    - **TileLang / MHC**：`[Tilelang][mhc] update tilelang kernels`、`[mhc] add mhc kernel tests`、`[tilelang] strict compute_num_split to specific bounds`
    - **分布式 / 通信**：`[Distribute] add strict rules for AgRs dispatch`、`[Mccl] add dummy mccl API for compatibility`
    - **模型侧修复**：`[models][step3p5] layer_types out of indexes if MTP`、`[models][step3p5_mtp] MTP weight loading error`、`[model][deepseek_v4] / [models][deepseek_v2] update to v0.30.0`、`[model][dsv2] update ds mtp`、`[issue]: Fix Dsv32 mtp model indexer forward path patch missing`
    - **从 v0.29.0 移植的补丁**（本版本随版带上）：dsv4 topk indexer blocktable 越界守卫、Qwen3-VL RoutedExperts 转置 fused checkpoint 的 per-channel scale 修复、MoE-LoRA 共享内存上限、AutoAWQ MoE `match_mode`、SWIGLUSTEP 激活（融合 FP8 block quant）、msprobe worker 选择日志、`tools/sync_maca_private.py` 与 `maca_private.txt` 按构建容器对齐
    - **性能与默认值**：`[performance] Apply optimized gumbel_sample_kernel`（mcoplib Gumbel 采样，含单测）；Maca 平台默认关闭 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`（第 6 节）；新增 `VLLM_METAX_USE_QUANT_MAIN_PROJ` 控制 `main_proj` 量化
    - **模型支持补充**：CosyVoice2 语音 token LM、`DeepseekV32MTPModel` 注册（GLM-5.2 等）、JoyAI-LLM-Flash 中文编码与 MTP 修复、MiniMax-M3-VL video processor 修复
    - **Patch 体系**：`[patch] reviewed v0.30.0 patches and AUDIT.md`（全量补丁复核 + 审计表）、`[Patch] fix patch not take effect correctly`；补丁模板与 README 补充（`@patch` 必须是最外层装饰器、别名同步规则）
    - **构建**：`[chores] prepare for v0.30.0 update`（common / test 依赖刷新）、`[build] support non-standard maca version`
- **工具 / 工程**
    - 新增 dummy 模型与 smoke test 工具（`[tools]` / `[skills]` 系列），便于没有权重时验证加载链路
    - patch 目录规范：只有 `bugfix` / `enhancement` / `performance` 三个顶层目录会被主动 import，补丁须使用 `@patch` 装饰器并写明用途 / 添加版本 / 移除条件；补丁取舍记录在同目录 `AUDIT.md`

## 5. 新增与更新的模型

以下模型为 v0.30.0 新增或显著更新（部分尚未回填 `docs/models/supported_models.md`，以本手册为准）：

### 文本 / 推理模型

- **Qwen3.8-Flash-Next（Qwen4Exp）**：本版本**新增支持**。注册 `Qwen4ExpForCausalLM` / `Qwen4ExpForConditionalGeneration` / `Qwen4ExpMTP`；MetaX 侧独立实现 QSA（稀疏注意力与 indexer）、short-conv（PLE 卷积）、PLE 层、hyper-connection、MTP 及对应 tilelang / triton 算子。MTP 与投机解码、CUDA graph capture 走本仓库实现，需按第 8 节实测确认
- **DeepSeek-V4.1 / DeepSeek-V4**：本版本**新增 DeepSeek-V4.1 支持**（模型、attention、MXFP4 / INT8 量化、DSpark / MTP），并统一 dsv4 / dsv41 的 FP8 量化配置（`registry/quant_config/deepseek_fp8.py`，注册名 `deepseek_v4_fp8`）；注意 tp 必须整除 checkpoint 的 `o_groups`（Flash=8、Pro=16）
- **DeepSeek-V4 / V4.1 DSpark**：草稿模型状态、MTP 随 0.30 同步（`models/deepseek_v4/metax/dspark.py`、`models/deepseek_v41/metax/*`）；`main_proj` 的量化由 `VLLM_METAX_USE_QUANT_MAIN_PROJ` 控制（默认 1，即量化）
- **GLM-5.x / GLM-4.6**：保留 `DeepseekV32MTPModel` → `vllm_metax.models.deepseek_mtp:DeepSeekMTP` 的注册，使 V3.2 风格 MTP 草稿（`deepseek_v32` / `glm_moe_dsa`）使用 MetaX 稀疏 indexer，而不是上游始终以 FP8 量化 indexer Q 的实现
- **MiniMax-M3 / M3-VL**：权重加载、index-score kernel、sparse attention 补丁随 0.30 复核；新增 video processor 修复（transformers >= 5.19 去掉 `do_convert_rgb` 必填参数），修复引擎启动期 `profile_run` 失败
- **Step-3.5-Flash**：MTP 的 `layer_types` 越界与权重加载问题修复
- **京东 JoyAI-LLM-Flash**：中文编码（transformers v5 下还原 `tokenizer.json` backend）与 MTP（草稿配置改写）修复
- **CosyVoice2**：本版本**新增支持**的语音 token 语言模型（`vllm_metax/models/cosyvoice2.py`，复用 Qwen2 主干 + 带 bias 的语音 head）

### 多模态模型

- **Qwen3-VL-MoE（W8A8 等转置 fused checkpoint）**：per-channel scale 修复随版带上（`bugfix/routed_experts_fused_weight_fix.py`）
- **MiniMax-M3-VL**：见上（video processor 修复）
- **Qwen3.8-Flash-Next（VL）**：`Qwen4ExpForConditionalGeneration` 已注册；视觉量化算子（`metax/vision_quantization.py`）随模型一起移植
- 上游 v0.30 新增的 DeepSeek-V4-Flash-Vision-Exp、Bailing V3 VL 等视觉架构未在 MetaX 上验证

### 量化模型支持

- FP8（含 `deepseek_v4_fp8` 统一注册）、INT8 W8A8、W4A8、MXFP4 / MXFP8、AWQ / GPTQ / compressed-tensors 沿既有实现；本版本新增 OOT QuantFP8 registry 与 DeepSeek-V4.1 的 MXFP4 / INT8 组合
- `w8a8_int8` 仅走 EP（deep_ep）路径

### 上游 v0.30.0 新增 / 迁移模型（MetaX 侧待验证）

下列架构为上游 v0.30.0 新增或迁移，随基线进入本仓库代码，但 **MetaX 侧尚未适配/跑测**（本地共享存储也暂无权重），先登记在此便于后续补测：

- **DeepSeek-V4-Flash-Vision-Exp**（DeepSeek-V4 视觉版）
- **GLM-5.3-Flash**（含 EPLB）
- **K2-Horizon**（含 reasoning / tool parser）
- **Cohere Compass**
- **Bailing V3 VL**（Ling 3.0 Flash VL，含 MTP）
- **Nanbeige4.2**（transformers backend）
- **FunASR**（语音）
- **DeepSeek-V4 CPU 后端**（AVX512/AMX 稀疏 MLA / indexer / mHC / compressor）：MACA 上不适用

> 全部已测试模型与特性状态（LoRA、PP 等）见 [支持模型列表](/models/supported_models.md)。
> 若需接入列表之外的模型或自定义模型，参考 [模型注册指南](/developing/model/registration.md)。

## 6. 从 v0.29 升级迁移

1. **更换镜像 / 环境**：docker 用户直接切到 v0.30.0 镜像；源码用户重新 checkout `mx/v0.30.0-dev` 并按 3.3 重装依赖。
2. **torch / MACA 保持 3.9 基线**：v0.30.0 仍使用 torch 2.10（metax3.9.0.8），但 mcoplib 由 `0.4.13+g78bea36.maca3.9.0.29.torch2.10` 升到 `0.4.14+g5780a78.maca3.9.0.31.torch2.10`，`lmcache` / `nixl` / `mooncake-transfer-engine` 同步到 `maca3.9.0.0`；本地编译的 csrc 需与新 mcoplib 的内核符号匹配，否则运行期报缺符号。
3. **CUDA graph 显存 profiling 默认关闭**：Maca 平台在 `platform.py` 里通过 `maybe_override_vllm_env` 把 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 置为 0，KV cache 预算不再扣减实测的 CUDA graph pool（与 v0.28 行为一致，v0.29 曾沿用上游默认值 1）。显式设置该环境变量时以显式值为准；需要更大 KV cache 时优先抬高 `--gpu-memory-utilization`。
4. **DeepSeek-V4 / V4.1 量化配置统一**：`deepseek_v4_fp8` 现在由 `registry/quant_config/deepseek_fp8.py::MacaDeepseekV41FP8Config` 承担（v0.29 为 `fp8.py` 中的 `MacaDeepseekV4FP8Config`），dsv4 / dsv41 checkpoint 共用；只按量化名称配置的服务无需改动。
5. **新增环境变量**：`VLLM_METAX_USE_QUANT_MAIN_PROJ`（默认 1）控制 DSpark `main_proj` 是否使用量化权重，需要全精度时可设为 0。
6. **Hunyuan V1 / HunyuanOCR 走 transformers backend**（v0.29 起）：在 MACA 上 CUDA graph 捕获期会因 HF dynamic RoPE（`rope_scaling.type=dynamic`）做隐式同步而报 `cudaErrorStreamCaptureImplicit`，需加 `--enforce-eager`；该结论来自 v0.29 实测，0.30 上待复核。
7. **LoRA MoE 小批量快路径**（v0.29 起实测）：`Qwen3-30B-A3B` 等 MoE 模型挂 LoRA 时，上游的 `_fused_moe_lora_small_batch_kernel` 在 MACA 上 capture 阶段会报 ATU 越界（`Xnack Error/ATU Fault(0x8)`）。规避：把 `--max-lora-rank` 设到 64 以上（例如 128）以跳过该快路径；本版本另带了 one-shot 路径的共享内存上限补丁（见第 8 节）。
8. **上游破坏性变更**：
    - 移除 10 个废弃架构（v0.29 起）：Arctic、Fairseq2Llama、GritLM、MPT、Falcon-RW、StableLM Epoch、Chameleon、Cheers、HyperCLOVAX-SEED-Vision、FireRedLID，服务侧若仍在用需回退版本或迁移；
    - scale-out 端点改为 `--enable-scale-out` opt-in（不再由 `VLLM_ENABLE_SCALE_OUT_ENDPOINTS` 控制）；
    - GPTQ 的 `g_idx`（激活重排）移除；0.29 起弃用的 `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`、`VLLM_MM_HASHER_ALGORITHM` 等环境变量移除；Mamba cache `all` 模式弃用；`python -m vllm.entrypoints.grpc_server` 改用 `vllm serve --grpc`；
    - YaRN 与 transformers 对齐，厂商 YaRN 别名不再二次缩放 `max_model_len`，使用自定义 rope_scaling 别名的 checkpoint 需重新核对上下文长度。
9. **依赖结构调整**：`requirements/maca_private.txt` 的 Metax 组件版本带构建特定的本地版本号，已提供 `tools/sync_maca_private.py` 用于按构建容器对齐（`--check` / `--write`）。
10. **默认值变化**：`max_num_batched_tokens` 默认 16384、Mamba 类默认开启 prefix caching（v0.28 起）、TP CUDA 组默认启用 FlashInfer all-reduce（`VLLM_ALLREDUCE_USE_FLASHINFER=0` 关闭）、prefix-cache `NONE_HASH` 默认确定性。与 v0.29 及更早版本直接对比性能时需先对齐这些基线。
11. **升级后建议回归**：Qwen3.8-Flash-Next（首次在 MACA 支持）、DeepSeek-V4 / V4.1（含 MTP / DSpark 与新的 FP8 注册）、GLM-5.x MTP、MiniMax-M3-VL、JoyAI-LLM-Flash（中文编码 / MTP）、CosyVoice2，以及 eagle / MTP 的提议精度。

## 7. Metax 与 NVIDIA CUDA 的差异

Metax 在大部分接口上与 NVIDIA 对齐，但在某些软件行为和环境变量上存在差异。

### 7.1. MACA_MPS_MODE（多进程共享 GPU）

默认情况下，MACA 不允许多个进程共享同一块 GPU。如果 GPU 已被占用，则无法启动新进程。

如需启用类似 MPS（Multi-Process Service）的功能，需设置：`MACA_MPS_MODE=1`

```bash
export MACA_MPS_MODE=1
vllm serve /model_weights/Qwen/Qwen3-8B --tensor-parallel-size 1
```

### 7.2. 多节点部署

vLLM 分布式（tensor/pipeline/expert parallel）跨节点时，建议设置以下环境变量以确保节点间通信正常：

- `MCCL_SOCKET_IFNAME`：用于 MCCL 通信的网络接口
- `GLOO_SOCKET_IFNAME`：用于 GLOO 通信的网络接口
- `MCCL_IB_HCA`：指定使用的 InfiniBand 设备

可通过 `ifconfig` 和 `mx-smi topo -n` 确定所用网卡和 IB 设备（参考 verl / Megatron 文档中的拓扑示例）。推荐设置：

```bash
export MCCL_SOCKET_IFNAME=ens20f0np0
export GLOO_SOCKET_IFNAME=ens20f0np0
export MCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
```

> ⚠️ 请根据自己机器的 `ifconfig` 和 `mx-smi topo -n` 输出进行相应调整，切勿盲目复制上述命令。

## 8. 已知问题与规避

下表包含本版本在 MACA 上实测到的问题与长期限制。标注「v0.29 实测」的条目来自上一版本，需在 v0.30.0 上复核；标注「待实测」的条目是本版本新增能力，尚未在真机权重上跑通。

| 问题 | 影响 | 规避 / 状态 |
| ------ | ------ | ------------ |
| Qwen3.8-Flash-Next（Qwen4Exp） | 待实测：本版本新增支持，尚未在 MACA 上用真实权重跑通完整链路（含 QSA 稀疏注意力、PLE / short-conv、MTP 与 CUDA graph capture） | 权重到位后优先补测；上游改过 AttentionImpl 变体发现逻辑（`find_attention_impl_variant`），若启动或前向异常先查该路径 |
| 本版本随版移植的四个补丁（dsv4 topk 越界守卫、MoE-LoRA 共享内存上限、mcoplib Gumbel 采样、Qwen3-VL scale 归一化） | 待实测：均已按 v0.30.0 上游实现适配，但未在 v0.30.0 上做运行期冒烟测试 | 上线前用 DeepSeek-V4 / V4.1（请求几何变化后的 decode）、Qwen3-VL-MoE W8A8、`Qwen3-MoE + --enable-lora` 各跑一次 |
| 上游 v0.30.0 新增 / 迁移模型（DeepSeek-V4-Flash-Vision-Exp、GLM-5.3-Flash、K2-Horizon、Cohere Compass、Bailing V3 VL、Nanbeige4.2、FunASR） | MetaX 侧未验证，共享存储暂无权重 | 已在模型清单登记占位，权重到位后补测 |
| 上游新特性 Watermarking（Gumbel-max 水印）/ HiSparse / `--load-format ipc_cache` | CUDA 侧实现，MACA 未适配或未验证 | 需要这些能力时先与 PDE/AI 团队确认；采样侧本版本已切到 mcoplib Gumbel kernel |
| Hunyuan V1 / HunyuanOCR 走 transformers modeling backend（v0.29 实测） | CUDA graph 捕获阶段 `dynamic_frequency_update` 做隐式同步，报 `cudaErrorStreamCaptureImplicit: operation would make the legacy stream depend on a capturing blocking stream`（实测 Hunyuan-7B-Instruct，tp1） | 该模型加 `--enforce-eager`；根治需 MACA 侧补丁或在 capture 期绕过 dynamic frequency update |
| LoRA MoE 小批量快路径（`_fused_moe_lora_small_batch_kernel`，v0.29 实测） | 上游 Triton 内核在 MACA 上 capture 期 `Xnack Error/ATU Fault(0x8)`（实测 Qwen3-30B-A3B-lora，tp2，`--max-lora-rank 64`） | 暂用 `--max-lora-rank 128`（>64）绕过该快路径；one-shot 路径的共享内存上限已由本版本补丁处理 |
| DeepSeek-R1-W48A8-PCPT_NAIVE 等 MLA + MTP 模型（v0.29 实测） | warmup decode 阶段 `flash_fwd_splitkv_combine_kernel` 触发 ATU 越界（`CUDA error: an illegal memory access was encountered`） | v0.29 线回归（0.28.0+gd14be0 上该模型正常）；0.30.0 上待复核，未复核前不建议放入测试清单 |
| RoutedExperts 转置 fused checkpoint 的 per-channel scale（PR #50137 follow-up） | Qwen3-VL-MoE W8A8 等转置 fused 模型加载失败 | 上游仅处理融合权重，仍需 `bugfix/routed_experts_fused_weight_fix.py`（本版本已带） |
| `minimax_qk_norm` all-reduce fusion 默认关闭 | 该融合算子默认不启用，MiniMax-M3 相关性能略低 | 如需开启请修改 `rms_norm_tp` 相关配置并回归精度 |
| 上游移除 10 个废弃架构 | 旧服务若使用 Arctic / MPT / Falcon-RW / StableLM Epoch / Chameleon / Cheers / HyperCLOVAX-SEED-Vision / FireRedLID / GritLM / Fairseq2-Llama 将无法启动 | 回退插件版本或迁移到受支持架构 |
| MACA flash attention 只有 FA2（`is_fa_version_supported` 仅返回 2） | 上游依赖 FA3 的 FP8 KV cache 路径在 MetaX 无实现 | 启动期检查直接判为 unsupported，FP8 KV cache 暂不可用 |
| DeepSeek-V4 要求 tp 整除 checkpoint 的 `o_groups`（Flash=8、Pro=16） | Flash 用 tp16 启动即在 profile_run 阶段崩溃：`ZeroDivisionError: n_local_heads // n_local_groups` | 两机 16 卡改用 tp8×pp2 或 tp8×dp2；V4-Pro（o_groups=16）tp16 合法 |
| 本地编译 csrc 与预编译 mcoplib 差异 | 本地编译需与 mcoplib `0.4.14+g5780a78.maca3.9.0.31` 的内核符号匹配 | 生产环境建议使用预编译 mcoplib 组合 |
| tilelang / tvm-ffi / xgrammar 强匹配 | 混用组合可能导致内核加载或 grammar 异常 | 严格按 `requirements/` 中的组合安装（tilelang `0.1.12+maca`、apache-tvm-ffi `0.1.11+maca`） |
| deep_ep / mctlass / W4A8 等量化路径 | 需回归验证精度 | 精度异常时用 msprobe 抓取对比（[指南](/developing/msprobe/msprobe_guide.md)） |
| transformers v5 下部分 tokenizer（Moonlight / Kimi / JoyAI 等） | 旧 import 路径（`bytes_to_unicode`）被移除；JoyAI 还可能丢失 `tokenizer.json` 中的后端 | 本版本已带 `bugfix/bytes_to_unicode.py` 与 `bugfix/tokenizer.py`（含 JoyAI backend 还原）；仍异常时锁定 transformers 版本 |
| `max_num_batched_tokens` 默认 16384、Mamba 默认开启 prefix caching | 与 v0.27 及更早的吞吐/时延基线不可直接对比 | 对比测试时显式固定参数 |
| v0.16.0 被官方跳过 | 无此版本 | 请勿使用 v0.16.x |

## 9. 相关文档

- **版本映射总表**：[Releases](/getting_started/quickstart.md#releases)
- **支持模型列表**：[Supported Models](/models/supported_models.md)
- **从源码安装 / 构建**：[MACA 安装指南](/getting_started/installation/maca.md)
- **精度调测（msprobe）**：[msprobe 精度调测指南](/developing/msprobe/msprobe_guide.md)
- **Patch 管理（monkey / git patches）**：[Patches](/developing/patches/README.md)
- **接入自定义 / 新模型**：[模型注册指南](/developing/model/registration.md)
- **插件架构与贡献**：[Contributing / 插件系统](/developing/README.md)
- mcoplib 仓库：<https://github.com/MetaX-MACA/mcoplib>
- 插件主页 / 源码：<https://github.com/MetaX-MACA/vLLM-metax>
- 开发者社区 / 镜像下载：<https://developer.metax-tech.com>
- vLLM 官方文档：<https://docs.vllm.ai/>

---
发布日期：TODO（以实际发布为准）｜维护：PDE/AI vLLM-MetaX 团队
