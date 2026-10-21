# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# -----------------------------------------------------------------------------
# Note: Preserve MiniMax-M3 block-FP8 inverse scales when fused or expert
#       parameters expose weight_scale_inv. Keep upstream's weight_scale rename
#       for MXFP8 destinations, rather than suppressing it for every checkpoint.
#       Validate hybrid coverage once at the outermost text/VL load boundary,
#       accumulating interleaved backbone groups without resetting defaults.
#
# Affected versions: vLLM 0.30.0 (ced6857afa0e), reviewed 2026-10-12.
#
# Remove at: Upstream weight loading preserves weight_scale_inv for block-FP8 parameters
#     on MetaX.
# -----------------------------------------------------------------------------

from collections.abc import Iterable
from contextvars import ContextVar
from types import SimpleNamespace

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import (
    is_pp_missing_parameter,
)

from vllm_metax.patch import patch

logger = init_logger(__name__)

_hybrid_session: ContextVar["_HybridLoadSession | None"] = ContextVar(
    "minimax_m3_hybrid_loading", default=None
)
_MODEL_MODULE = "vllm.models.minimax_m3.nvidia.model"


def _is_kv_runtime_scale(name):
    return ".attn." in name and name.rsplit(".", 1)[-1] in {
        "q_scale", "k_scale", "v_scale", "q_zero_point", "k_zero_point",
        "v_zero_point",
    }


def _fused_shards(module_name, module):
    leaf = module_name.rsplit(".", 1)[-1]
    if leaf == "qkv_proj":
        shards = {"q", "k", "v"}
        if hasattr(module, "total_num_index_heads"):
            shards.update(("index_q", "index_k"))
        return shards
    if leaf == "gate_up_proj":
        return {0, 1}
    return None


class _HybridLoadSession:
    def __init__(self, root):
        self.module_names = {id(module): name for name, module in root.named_modules()}
        self.params = dict(root.named_parameters(remove_duplicate=False))
        self.backbones = {}
        self.fused = {}
        self.restore = []

    def watch_fused_loads(self, root):
        # Vision returns only fused parameter names, even for partial Q/K/V.
        # Observe successful writes, not postprocess hooks or source key names.
        modules = dict(root.named_modules())
        for name, param in self.params.items():
            module_name = name.rpartition(".")[0]
            shards = _fused_shards(module_name, modules.get(module_name))
            if shards is None or id(param) in self.fused:
                continue
            seen = set()
            self.fused[id(param)] = (name, shards, seen)
            had_loader = hasattr(param, "weight_loader")
            original = getattr(param, "weight_loader", default_weight_loader)
            self.restore.append((param, had_loader, original))
            param.weight_loader = self._watch_loader(original, shards, seen)

    @staticmethod
    def _watch_loader(original, expected, seen):
        def record(param, tensor, *args, **kwargs):
            result = original(param, tensor, *args, **kwargs)
            if result is not False:
                shard = args[0] if args else kwargs.get(
                    "loaded_shard_id", kwargs.get(
                        "shard_id", getattr(tensor, "shard_id", None)
                    )
                )
                if shard is None:
                    seen.update(expected)
                else:
                    seen.update(shard if isinstance(shard, tuple) else (shard,))
            return result
        return record

    def finish(self, loaded):
        loaded = set(loaded)
        exempt_ids = set()
        for model, state in self.backbones.items():
            prefix = self.module_names[id(model)]
            loaded.update(
                f"{prefix}.{name}" if prefix else name
                for name in _finish_hybrid_weights(state)
            )
            exempt_ids.update(
                id(state.params[name]) for name, slots in state.expected.items()
                if not slots  # An EP rank can own zero expert rows.
            )
        loaded_ids = {
            id(param) for name, param in self.params.items() if name in loaded
        }
        covered_ids = loaded_ids | exempt_ids
        missing = [
            name for name, param in self.params.items()
            if id(param) not in covered_ids and not _is_kv_runtime_scale(name)
        ]
        missing_shards = [
            f"{name}: {sorted(expected - seen, key=repr)}"
            for name, expected, seen in self.fused.values() if expected - seen
        ]
        if missing or missing_shards:
            raise ValueError(
                "MiniMax-M3 hybrid top-level weight loading incomplete; "
                f"missing parameters ({len(missing)}): {missing[:20]}; "
                f"missing fused shards ({len(missing_shards)}): {missing_shards[:20]}"
            )
        return loaded

    def close(self):
        for param, had_loader, original in reversed(self.restore):
            if had_loader:
                param.weight_loader = original
            else:
                del param.weight_loader


def _load_top_level_weights(model, weights, original):
    session = _hybrid_session.get()
    if session is not None and id(model) in session.module_names:
        # ConditionalGeneration -> CausalLM, possibly many interleaved calls.
        return original(model, weights)
    if not _has_hybrid_quant_config(model):
        return original(model, weights)
    session = _HybridLoadSession(model)
    token = _hybrid_session.set(session)
    try:
        session.watch_fused_loads(model)
        loaded = original(model, weights)
        return session.finish(loaded)
    finally:
        session.close()
        _hybrid_session.reset(token)


@patch(_MODEL_MODULE, "MiniMaxM3SparseForCausalLM.load_weights")
def load_causal_weights(self, weights):
    original = load_causal_weights.__vllm_metax_patch__[
        f"{_MODEL_MODULE}.MiniMaxM3SparseForCausalLM.load_weights"
    ]
    return _load_top_level_weights(self, weights, original)


@patch(_MODEL_MODULE, "MiniMaxM3SparseForConditionalGeneration.load_weights")
def load_conditional_weights(self, weights):
    original = load_conditional_weights.__vllm_metax_patch__[
        f"{_MODEL_MODULE}.MiniMaxM3SparseForConditionalGeneration.load_weights"
    ]
    return _load_top_level_weights(self, weights, original)


def _is_hybrid_quant_config(config) -> bool:
    return (
        getattr(config, "moe_quant_algo", None) == "NVFP4"
        and getattr(config, "moe_quant_format", None) == "nvfp4-pack-quantized"
    )


def _has_hybrid_quant_config(model) -> bool:
    # MiniMaxM3Model does not retain vllm_config.quant_config; RoutedExperts does.
    return any(
        _is_hybrid_quant_config(getattr(module, "quant_config", None))
        or _is_hybrid_quant_config(
            getattr(getattr(module, "quant_method", None), "quant_config", None)
        )
        for module in model.modules()
    )


def _new_hybrid_state(model, expert_mapping):
    """Initialize local coverage and defaults once per complete loading stream."""
    params = dict(model.named_parameters())
    modules = dict(model.named_modules())
    # A slot is (physical expert id, logical projection shard). TP slicing stays
    # in the upstream loader; every TP rank still needs each checkpoint shard.
    expected: dict[str, set[tuple[int | None, int | str | None]]] = {}
    expert_modules = {}
    input_defaults: dict[str, float] = {}
    expert_ids = sorted({entry[2] for entry in expert_mapping})
    for name, module in modules.items():
        # MoERunner delegates expert-id mapping, but only RoutedExperts owns
        # the parameters exposed by model.named_parameters().
        if not isinstance(module, RoutedExperts):
            continue
        expert_modules[name] = module
        for projection in ("w13", "w2"):
            for suffix in (
                "weight_packed", "weight_scale", "weight_global_scale",
                "input_global_scale",
            ):
                target = f"{name}.{projection}_{suffix}"
                if target not in params:
                    raise ValueError(f"MiniMax-M3 hybrid target is missing: {target}")

    for name, param in params.items():
        module_name, _, leaf = name.rpartition(".")
        module = modules.get(module_name)
        if module_name in expert_modules:
            local_ids = [
                expert_id for expert_id in expert_ids
                if module._map_global_expert_id_to_local_expert_id(expert_id) != -1
            ]
            local_rows = {
                module._map_global_expert_id_to_local_expert_id(expert_id)
                for expert_id in local_ids
            }
            if local_rows != set(range(param.shape[0])):
                raise ValueError(
                    f"MiniMax-M3 hybrid expert mapping does not cover {name}"
                )
            if leaf.startswith("w13_"):
                shards = ("w1", "w3")
            elif leaf.startswith("w2_"):
                shards = ("w2",)
            else:
                raise ValueError(f"Unsupported MiniMax-M3 hybrid expert target: {name}")
            expected[name] = {(eid, shard) for eid in local_ids for shard in shards}
            if leaf.endswith("_input_global_scale"):
                # Match the demo's legacy minfer defaults. Initialize before the
                # stream so explicit checkpoint scales win in any key order.
                value = 1.0 if leaf.startswith("w13_") else 16.0
                param.data.fill_(value)
                input_defaults[name] = value
        elif _is_kv_runtime_scale(name):
            # KV-cache quantization has initialized runtime defaults, independent
            # of checkpoint weight quantization. Supplied keys must still map.
            continue
        elif (shards := _fused_shards(module_name, module)) is not None:
            expected[name] = {(None, shard) for shard in shards}
        else:
            expected[name] = {(None, None)}

    return SimpleNamespace(
        params=params, expected=expected, expert_modules=expert_modules,
        input_defaults=input_defaults, seen={name: set() for name in expected},
        loaded=set(),
    )


def _load_hybrid_weights(model, weights, stacked_mapping, expert_mapping) -> set[str]:
    session = _hybrid_session.get()
    if session is not None and id(model) not in session.module_names:
        session = None
    state = session.backbones.get(model) if session is not None else None
    if state is None:
        state = _new_hybrid_state(model, expert_mapping)
        if session is not None:
            session.backbones[model] = state
    params, seen, loaded = state.params, state.seen, state.loaded
    expert_modules, expected = state.expert_modules, state.expected
    unmapped: set[str] = set()
    rejected: set[str] = set()
    for source_name, tensor in weights:
        if source_name.startswith("mtp.") or ".mtp." in source_name:
            continue
        if is_pp_missing_parameter(source_name, model):
            continue
        name = source_name
        if "weight_scale_inv" in name:
            candidates = [name]
            candidates.extend(
                name.replace(src, dst)
                for dst, src, _ in stacked_mapping if src in name
            )
            candidates.extend(
                name.replace(src, dst)
                for dst, src, _, _ in expert_mapping if src in name
            )
            if not any(candidate in params for candidate in candidates):
                name = name.replace("weight_scale_inv", "weight_scale")

        if name.endswith((
            ".kv_scale", ".q_scale", ".k_scale", ".v_scale",
            ".q_zero_point", ".k_zero_point", ".v_zero_point",
        )):
            remapped = maybe_remap_kv_scale_name(name, params)
            if remapped is None:
                unmapped.add(source_name)
                continue
            name = remapped

        matched_expert = False
        for dst, src, expert_id, shard in expert_mapping:
            if src not in name:
                continue
            matched_expert = True
            target = name.replace(src, dst)
            if is_pp_missing_parameter(target, model):
                continue
            if target not in params:
                unmapped.add(source_name)
                continue
            param = params[target]
            success = param.weight_loader(
                param, tensor, target, shard_id=shard, expert_id=expert_id,
                return_success=True,
            )
            owner = expert_modules.get(target.rpartition(".")[0])
            if owner is None:
                raise ValueError(f"Missing MiniMax-M3 hybrid expert owner: {target}")
            is_local = owner._map_global_expert_id_to_local_expert_id(expert_id) != -1
            if success is not True:
                if is_local or success is not False:
                    rejected.add(f"{source_name} -> {target} [{expert_id}, {shard}]")
                continue
            loaded.add(target)
            if target in seen:
                seen[target].add((expert_id, shard))
        if matched_expert:
            continue

        matched_stacked = False
        for dst, src, shard in stacked_mapping:
            if src not in name:
                continue
            matched_stacked = True
            target = name.replace(src, dst)
            if is_pp_missing_parameter(target, model):
                break
            if target not in params:
                unmapped.add(source_name)
                break
            param = params[target]
            param.weight_loader(param, tensor, shard)
            loaded.add(target)
            if target in seen:
                seen[target].add((None, shard))
            break
        if matched_stacked:
            continue

        target = maybe_remap_kv_scale_name(name, params)
        if target is None or target not in params:
            unmapped.add(source_name)
            continue
        if is_pp_missing_parameter(target, model):
            continue
        if target.rpartition(".")[0] in expert_modules:
            # A raw fused expert tensor cannot use the per-expert loader without
            # explicit expert/shard metadata. Do not accidentally mark it loaded.
            unmapped.add(source_name)
            continue
        param = params[target]
        getattr(param, "weight_loader", default_weight_loader)(param, tensor)
        loaded.add(target)
        if target in seen:
            seen[target].update(expected[target])

    if unmapped or rejected:
        raise ValueError(
            "MiniMax-M3 hybrid weight loading incomplete; "
            f"unmapped checkpoint keys ({len(unmapped)}): {sorted(unmapped)[:20]}; "
            f"rejected local loads ({len(rejected)}): {sorted(rejected)[:20]}"
        )
    if session is not None:
        return set(loaded)
    return _finish_hybrid_weights(state)


def _finish_hybrid_weights(state):
    expected, seen, loaded = state.expected, state.seen, state.loaded
    input_defaults = state.input_defaults
    defaulted = {
        name: expected[name] - seen[name] for name in input_defaults
        if expected[name] - seen[name]
    }
    if defaulted:
        examples = [
            f"{name}={input_defaults[name]} ({len(slots)} expert/shard slots)"
            for name, slots in sorted(defaulted.items())[:20]
        ]
        logger.warning(
            "MiniMax-M3 hybrid uses demo input_global_scale defaults for %d "
            "targets / %d local slots: %s",
            len(defaulted), sum(map(len, defaulted.values())), examples,
        )
        # These targets are initialized, not checkpoint-covered. Keep `seen`
        # unchanged so the warning never hides partial checkpoint coverage.
        loaded.update(defaulted)
    missing = {
        name: slots - seen[name] for name, slots in expected.items()
        if name not in input_defaults and slots - seen[name]
    }
    if missing:
        examples = [
            f"{name}: {sorted(slots, key=repr)[:8]}"
            for name, slots in sorted(missing.items())[:20]
        ]
        raise ValueError(
            "MiniMax-M3 hybrid weight loading incomplete; "
            f"missing targets/shards ({len(missing)}): {examples}"
        )
    return loaded


@patch("vllm.models.minimax_m3.nvidia.model", "MiniMaxM3Model.load_weights")
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    # q/k/v_proj -> fused qkv_proj; gate_proj/up_proj -> fused gate_up_proj
    # (dense MLP and shared expert). On sparse layers the indexer
    # index_q/index_k_proj fold into the same fused qkv_proj
    # (MinimaxM3QKVParallelLinearWithIndexer); these entries simply never match on
    # dense layers, whose checkpoints have no index_*_proj weights. Leading
    # dots keep `q_proj`/`k_proj` from matching `index_q_proj`/`index_k_proj`
    # (preceded by `_`, not `.`).
    stacked_params_mapping: list[tuple[str, str, int | str]] = [
        # (param_name, shard_name, shard_id)
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".qkv_proj", ".index_q_proj", "index_q"),
        (".qkv_proj", ".index_k_proj", "index_k"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]
    # (param_name, weight_name, expert_id, shard_id)
    expert_params_mapping = self.get_expert_mapping()
    if _has_hybrid_quant_config(self):
        return _load_hybrid_weights(
            self, weights, stacked_params_mapping, expert_params_mapping
        )
    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()
    for name, loaded_weight in weights:
        # The MTP module is not modeled yet.
        if "mtp." in name:
            continue
        # The checkpoint stores block scales as ``weight_scale_inv``; the
        # ModelOpt MXFP8 layers expose them as ``weight_scale``.
        # /-------------------- MetaX Modification --------------------\
        # Preserve inverse-scale names only when the destination actually exposes
        # them, including fused projections and experts. MXFP8 parameters still
        # need upstream's rename: q_proj.weight_scale_inv becomes
        # qkv_proj.weight_scale, while block-FP8 keeps qkv_proj.weight_scale_inv.
        if "weight_scale_inv" in name:
            inverse_scale_names = [name]
            inverse_scale_names.extend(
                name.replace(weight_name, param_name)
                for param_name, weight_name, _ in stacked_params_mapping
                if weight_name in name
            )
            inverse_scale_names.extend(
                name.replace(weight_name, param_name)
                for param_name, weight_name, _, _ in expert_params_mapping
                if weight_name in name
            )
            if not any(candidate in params_dict for candidate in inverse_scale_names):
                name = name.replace("weight_scale_inv", "weight_scale")
        # \-------------------- MetaX Modification --------------------/
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            # Routed experts (w1/w2/w3) are handled below; don't let the
            # stacked mapping rewrite them.
            if ("block_sparse_moe.experts." in name) and name not in params_dict:
                continue
            name = name.replace(weight_name, param_name)
            if name.endswith(".bias") and name not in params_dict:
                continue
            if is_pp_missing_parameter(name, self):
                continue
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            for (
                param_name,
                weight_name,
                expert_id,
                expert_shard_id,
            ) in expert_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(
                    param,
                    loaded_weight,
                    name,
                    shard_id=expert_shard_id,
                    expert_id=expert_id,
                )
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                remapped = maybe_remap_kv_scale_name(name, params_dict)
                if remapped is None:
                    continue
                name = remapped
                if is_pp_missing_parameter(name, self):
                    continue
                # Modules not modeled yet (e.g. attention) are skipped until
                # they are ported.
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
        loaded_params.add(name)
    return loaded_params
