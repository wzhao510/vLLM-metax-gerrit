# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""CPU tests for default post-capture warmup and patch registration."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


PACKAGE = Path(__file__).resolve().parents[3] / "vllm_metax"


@pytest.fixture
def runtime(monkeypatch):
    events = []
    state = SimpleNamespace(captured=False, capturing=False, fail_post=False)

    def stub(name, **attributes):
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            prefix = ".".join(parts[:i])
            if prefix not in created:
                module = ModuleType(prefix)
                module.__path__ = []
                monkeypatch.setitem(sys.modules, prefix, module)
                created.add(prefix)
                if i > 1:
                    setattr(sys.modules[".".join(parts[: i - 1])], parts[i - 1], module)
        module = sys.modules[name]
        for key, value in attributes.items():
            setattr(module, key, value)
        return module

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    class Worker:
        def compile_or_warm_up_model(self):
            raise AssertionError("The runtime patch was not installed")

    original = Worker.compile_or_warm_up_model

    class CaptureContext:
        def __enter__(self):
            state.capturing = True
            events.append("capture_enter")

        def __exit__(self, *args):
            state.capturing = False
            events.append("capture_exit")

    def capture():
        assert state.capturing
        state.captured = True
        events.append("capture")
        return 0

    def warmup(*args):
        assert not state.capturing
        events.append("post" if state.captured else "pre")
        if state.captured and state.fail_post:
            raise RuntimeError("post-capture failure")

    created = set()
    stub(
        "torch",
        accelerator=SimpleNamespace(
            synchronize=lambda: events.append("sync"),
        ),
    )
    stub("vllm.config", CUDAGraphMode=SimpleNamespace(NONE=0))
    stub(
        "vllm.config.compilation",
        CompilationMode=SimpleNamespace(
            NONE=0,
            VLLM_COMPILE=3,
        ),
    )
    stub("vllm.distributed", get_pp_group=lambda: SimpleNamespace(is_last_rank=False))
    stub("vllm.logger", init_logger=lambda name: SimpleNamespace(info=lambda *a: None))
    stub(
        "vllm.model_executor.warmup.kernel_warmup",
        kernel_warmup=lambda worker: events.append("kernel"),
    )
    stub("vllm.tracing", instrument=lambda **kwargs: lambda function: function)
    stub(
        "vllm.utils.gc_utils",
        freeze_gc_heap=lambda: events.append("freeze"),
        maybe_attach_gc_debug_callback=lambda: None,
    )
    stub(
        "vllm.utils.gpu_sync_debug",
        enable_gpu_sync_check=lambda: events.append("ready"),
    )
    stub("vllm.utils.mem_constants", GiB_bytes=1 << 30)
    stub("vllm.utils.mem_utils", format_gib=str)
    stub(
        "vllm.utils.torch_utils",
        set_random_seed=lambda seed: events.append("seed"),
        set_torch_threads_for_runtime=lambda: None,
    )
    stub("vllm.utils.jit_monitor", activate=lambda **kwargs: events.append("monitor"))
    stub("vllm.v1.worker.gpu.warmup", warmup_kernels=warmup)
    stub("vllm.v1.worker.startup_plan", maybe_save_startup_plan=lambda *args: None)
    stub("vllm.v1.worker.worker_base", CompilationTimes=SimpleNamespace)
    stub("vllm.v1.worker.gpu_worker", Worker=Worker)
    stub("vllm_metax.patch")
    load("vllm_metax.patch.utils", PACKAGE / "patch/utils.py")
    module = load(
        "vllm_metax.patch.bugfix.post_capture_warmup",
        PACKAGE / "patch/bugfix/post_capture_warmup.py",
    )

    worker = Worker()
    worker.compilation_config = SimpleNamespace(
        mode=0,
        backend="eager",
        compilation_time=1,
        encoder_compilation_time=2,
    )
    worker.vllm_config = SimpleNamespace(compilation_config=worker.compilation_config)
    worker.model_runner = SimpleNamespace(
        lora_config=None,
        maybe_remove_all_loras=lambda config: None,
        capture_model=capture,
    )
    worker.use_v2_model_runner = True
    worker.model_config = SimpleNamespace(enforce_eager=False, seed=0)
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=1)
    worker.observability_config = SimpleNamespace(
        jit_monitor_mode="warn",
        jit_monitor_verbose=False,
    )
    worker._get_cudagraph_capture_context = CaptureContext
    worker.execute_model = None
    worker.sample_tokens = None
    return SimpleNamespace(
        worker=worker,
        events=events,
        state=state,
        module=module,
        original=original,
    )


@pytest.mark.parametrize("legacy_flag", [None, "0", "1"])
def test_default_post_capture_warmup(runtime, monkeypatch, legacy_flag):
    if legacy_flag is None:
        monkeypatch.delenv("VLLM_DEBUG_POST_CAPTURE_WARMUP", raising=False)
    else:
        monkeypatch.setenv("VLLM_DEBUG_POST_CAPTURE_WARMUP", legacy_flag)
    result = runtime.worker.compile_or_warm_up_model()
    assert result.language_model == 1
    assert result.encoder == 2
    assert runtime.events == [
        "kernel",
        "pre",
        "capture_enter",
        "capture",
        "capture_exit",
        "sync",
        "post",
        "sync",
        "seed",
        "monitor",
        "freeze",
        "ready",
    ]


@pytest.mark.parametrize("v2,eager", [(False, False), (True, True)])
def test_other_execution_paths_skip_post_warmup(runtime, v2, eager):
    runtime.worker.use_v2_model_runner = v2
    runtime.worker.model_config.enforce_eager = eager
    runtime.worker.compile_or_warm_up_model()
    assert "post" not in runtime.events
    assert "sync" not in runtime.events
    assert ("pre" in runtime.events) is v2
    assert ("capture" in runtime.events) is not eager
    assert runtime.events[-1] == "ready"


def test_failure_prevents_startup_completion(runtime):
    runtime.state.fail_post = True
    with pytest.raises(RuntimeError, match="post-capture failure"):
        runtime.worker.compile_or_warm_up_model()
    assert runtime.events[-1] == "post"
    assert "monitor" not in runtime.events
    assert "ready" not in runtime.events


def test_patch_installed_with_original_metadata(runtime):
    function = runtime.worker.compile_or_warm_up_model.__func__
    assert function is runtime.module.compile_or_warm_up_model
    assert (
        function.__vllm_metax_patch__[
            "vllm.v1.worker.gpu_worker.Worker.compile_or_warm_up_model"
        ]
        is runtime.original
    )
