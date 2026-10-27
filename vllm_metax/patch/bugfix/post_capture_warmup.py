# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Run scheduler-driven MRV2 warmup after CUDA graph capture by default.
#     This avoids the GLM4.7 MTP memory fault reproduced on MetaX when warmup
#     only runs before capture. Preserve the pre-capture workspace preparation.
#
# Affected versions: vLLM 0.30.0 (ced6857afa0e), verified 2026-10-27.
#
# Remove at: Upstream provides equivalent post-capture initialization, or the
#     MetaX fault is fixed and GLM4.7 MTP passes without the extra warmup.
# -----------------------------------------------------------------------------

import torch

from vllm.config import CUDAGraphMode
from vllm.config.compilation import CompilationMode
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.warmup.kernel_warmup import kernel_warmup
from vllm.tracing import instrument
from vllm.utils.gc_utils import freeze_gc_heap, maybe_attach_gc_debug_callback
from vllm.utils.gpu_sync_debug import enable_gpu_sync_check
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_random_seed, set_torch_threads_for_runtime
from vllm.v1.worker.gpu.warmup import warmup_kernels
from vllm.v1.worker.startup_plan import maybe_save_startup_plan
from vllm.v1.worker.worker_base import CompilationTimes

from vllm_metax.patch.utils import patch

logger = init_logger(__name__)


@patch("vllm.v1.worker.gpu_worker", "Worker.compile_or_warm_up_model")
@instrument(span_name="Warmup (GPU)")
def compile_or_warm_up_model(self) -> CompilationTimes:
    warmup_sizes: list[int] = []

    if self.vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE:
        # warm up sizes that are not in cudagraph capture sizes,
        # but users still want to compile for better performance,
        # e.g. for the max-num-batched token size in chunked prefill.
        compile_sizes = self.vllm_config.compilation_config.compile_sizes
        warmup_sizes = compile_sizes.copy() if compile_sizes is not None else []  # type: ignore[assignment]
        cg_capture_sizes: list[int] = []

        if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            cg_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
            cg_capture_sizes = [] if cg_sizes is None else cg_sizes
            warmup_sizes = [x for x in warmup_sizes if x not in cg_capture_sizes]

        compile_ranges = self.vllm_config.compilation_config.get_compile_ranges()
        # For each compile_range, if none of the batch sizes
        # in warmup_sizes or cudagraph_capture_sizes are in the range,
        # add the end of the range to ensure compilation/warmup.
        all_sizes = set(cg_capture_sizes)
        all_sizes.update([x for x in warmup_sizes if isinstance(x, int)])
        for compile_range in compile_ranges:
            if not any(x in compile_range for x in all_sizes):
                warmup_sizes.append(compile_range.end)

    # We skip EPLB here since we don't want to record dummy metrics
    for size in sorted(warmup_sizes, reverse=True):
        logger.info("Compile and warming up model for size %d", size)
        self.model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)
    self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)

    # Warmup and tune the kernels used during model execution before
    # cuda graph capture.
    kernel_warmup(self)

    if self.use_v2_model_runner:
        # A workspace resize after capture frees what the graphs point at.
        warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)

    cuda_graph_memory_bytes = 0
    if not self.model_config.enforce_eager:
        with self._get_cudagraph_capture_context():
            cuda_graph_memory_bytes = self.model_runner.capture_model()

    # /-------------------- MetaX Modification --------------------\
    if self.use_v2_model_runner and not self.model_config.enforce_eager:
        logger.info("Post-capture warmup: synchronizing captured work.")
        torch.accelerator.synchronize()
        logger.info("Post-capture warmup: starting scheduler-driven replay.")
        warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)
        torch.accelerator.synchronize()
        logger.info("Post-capture warmup: completed and synchronized.")
    # \-------------------- MetaX Modification --------------------/

    # Compare actual vs estimated CUDA graph memory (if we did profiling)
    if (
        hasattr(self, "cudagraph_memory_estimate")
        and self.cudagraph_memory_estimate > 0
    ):
        GiB = lambda b: round(b / GiB_bytes, 2)
        diff = abs(cuda_graph_memory_bytes - self.cudagraph_memory_estimate)
        logger.info(
            "CUDA graph pool memory: %s GiB (actual), %s GiB (estimated), "
            "difference: %s GiB (%.1f%%).",
            GiB(cuda_graph_memory_bytes),
            GiB(self.cudagraph_memory_estimate),
            GiB(diff),
            100 * diff / max(cuda_graph_memory_bytes, 1),
        )

    if self.cache_config.kv_cache_memory_bytes is None and hasattr(
        self, "peak_activation_memory"
    ):
        # Suggests optimal kv cache memory size if we rely on
        # memory_profiling to guess the kv cache memory size which
        # provides peak_activation_memory and a few other memory
        # consumption. `memory_profiling` does not consider
        # CUDAGraph memory size and may not utilize all gpu memory.
        # Users may want fine-grained control to specify kv cache
        # memory size.

        # empirically observed that the memory profiling may
        # slightly underestimate the memory consumption.
        # So leave a small buffer (=150MiB) to avoid OOM.
        redundancy_buffer_memory = 150 * (1 << 20)

        non_kv_cache_memory = (
            self.total_consumed + self.peak_activation_memory + cuda_graph_memory_bytes
        )
        kv_cache_memory_bytes_to_gpu_limit = (
            self.init_snapshot.free_memory
            - non_kv_cache_memory
            - redundancy_buffer_memory
        )
        kv_cache_memory_bytes_to_requested_limit = (
            int(self.requested_memory) - non_kv_cache_memory - redundancy_buffer_memory
        )

        msg = (
            f"Free memory on device "
            f"({format_gib(self.init_snapshot.free_memory)}/"
            f"{format_gib(self.init_snapshot.total_memory)} GiB) on startup. "
            f"Desired GPU memory utilization is "
            f"({self.cache_config.gpu_memory_utilization}, "
            f"{format_gib(self.requested_memory)} GiB). "
            f"Actual usage is {format_gib(self.total_consumed)} "
            f"GiB for consumed memory (weights + non-torch), "
            f"{format_gib(self.peak_activation_memory)} GiB "
            f"for peak activation, and {format_gib(cuda_graph_memory_bytes)} "
            f"GiB for CUDAGraph memory. Replace gpu_memory_utilization "
            f"config with `--kv-cache-memory="
            f"{kv_cache_memory_bytes_to_requested_limit}` "
            f"({format_gib(kv_cache_memory_bytes_to_requested_limit)} GiB) to fit "
            f"into requested memory, or `--kv-cache-memory="
            f"{kv_cache_memory_bytes_to_gpu_limit}` "
            f"({format_gib(kv_cache_memory_bytes_to_gpu_limit)} GiB) to fully "
            f"utilize gpu memory. Current kv cache memory in use is "
            f"{format_gib(self.available_kv_cache_memory_bytes)} GiB."
        )

        logger.info(msg)

        maybe_save_startup_plan(self, kv_cache_memory_bytes_to_requested_limit)

    if not self.use_v2_model_runner and get_pp_group().is_last_rank:
        # V1: Warm up sampler and preallocate memory buffer for logits and other
        # sampling related tensors of max possible shape to avoid memory
        # fragmentation issue.
        # NOTE: This is called after `capture_model` on purpose to prevent
        # memory buffers from being cleared by `torch.accelerator.empty_cache`.
        max_num_reqs = min(
            self.scheduler_config.max_num_seqs,
            self.scheduler_config.max_num_batched_tokens,
        )

        # We skip EPLB here since we don't want to record dummy metrics
        hidden_states, last_hidden_states = self.model_runner._dummy_run(
            num_tokens=max_num_reqs,
            skip_eplb=True,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )
        if self.model_runner.is_pooling_model:
            self.model_runner._dummy_pooler_run(hidden_states)
        else:
            self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)

    # Reset the seed to ensure that the random state is not affected by
    # the model initialization and profiling.
    set_random_seed(self.model_config.seed)

    # Eagerly trigger inductor's once-per-process lazy inits during
    # warmup (rather than on a later compile cache-miss at runtime).
    c_config = self.compilation_config
    if c_config.mode != CompilationMode.NONE and c_config.backend == "inductor":
        from vllm.compilation.compiler_interface import (
            trigger_inductor_lazy_init,
        )

        trigger_inductor_lazy_init(self.device)

    # All warmup is done — start monitoring for unexpected JIT
    # compilations that would cause latency spikes during inference.
    from vllm.utils.jit_monitor import activate as activate_jit_monitor

    activate_jit_monitor(
        mode=self.observability_config.jit_monitor_mode,
        verbose=self.observability_config.jit_monitor_verbose,
    )

    # Freeze the worker heap so the GC won't scan static objects
    # (model weights, KV caches, CUDA graphs) during inference.
    freeze_gc_heap()
    maybe_attach_gc_debug_callback()

    # Warmup / first-compile is done — activate the `VLLM_GPU_SYNC_CHECK`
    # gate so subsequent `execute_model` / `sample_tokens` calls enforce it.
    enable_gpu_sync_check()

    # Startup is done; steady-state serving gets no benefit from torch
    # intra-op parallelism.
    set_torch_threads_for_runtime()

    return CompilationTimes(
        language_model=self.compilation_config.compilation_time,
        encoder=self.compilation_config.encoder_compilation_time,
    )
