# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# -----------------------------------------------------------------------------
# Note: Normalize the per-channel weight scales of transposed fused MoE
#       checkpoints before they are loaded (patches
#       ``RoutedExperts.load_weights``).
#
#       Qwen3-VL-MoE int8 W8A8 checkpoints (Qwen3-VL-235B-A22B-Instruct_W8A8,
#       Qwen3-VL-30B-A3B-Instruct_W8A8, ...) store the fused experts transposed
#       and store the per-channel weight scales in that same layout, so the
#       quantized (singleton) axis ends up in the middle:
#
#           mlp.experts.gate_up_proj        [E, hidden, 2*intermediate]
#           mlp.experts.gate_up_proj_scale  [E, 1, 2*intermediate]
#           mlp.experts.down_proj           [E, intermediate, hidden]
#           mlp.experts.down_proj_scale     [E, 1, hidden]
#
#       RoutedExperts.load_weights only transposes the fused weights, because
#       `uses_weight_layout` excludes scale tensors unless they are BLOCK
#       scales. The per-channel scales therefore reach the weight loaders with
#       the channel axis in the middle and loading fails:
#
#         - w2: RuntimeError: output with shape [1, 1] doesn't match the
#               broadcast shape [1, 4096] (routed_experts.py
#               _load_per_channel_weight_scale)
#         - w1/w3: chunk(2, dim=1) cannot separate the two channel shards
#               along the singleton axis, so loading or shard lookup fails.
#
#       vLLM allocates per-channel MoE scales as [E, N, 1], so moving the
#       singleton axis back to the last position fixes both paths. Resolve the
#       destination parameter before chunking, and normalize only CHANNEL
#       weight scales in transposed fused checkpoints. A native BLOCK scale
#       [E, 1, K_blocks] has a different meaning and must remain unchanged.
#
# Purpose: port the v0.25.0/v0.26.0 fix of routed_experts_fused_weight_fix.py to
#          v0.29.0 (from v0.28.0 / vllm_metax f2414b9a), where upstream
#          `RoutedExperts._orient_fused_weight` still only transposes the fused
#          weights and leaves the per-channel scales in the transposed layout.
#
# Affected versions: Originally vLLM 0.29.1.dev0 (98dff2a81d); reviewed against
#     vLLM 9eba576dde on 2026-10-19.
#
# Remove at: Remove once upstream normalizes the per-channel scales of
#            transposed fused checkpoints (follow-up of vLLM PR #50137), or
#            once the affected checkpoints are re-quantized with native
#            [E, N, 1] scale layouts.
# -----------------------------------------------------------------------------
from collections.abc import Iterable

import torch
from vllm.model_executor.layers.fused_moe.routed_experts import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
)

from vllm_metax.patch.utils import patch

_original_load_weights = RoutedExperts.load_weights


def _normalize_channel_scales(
    layer: RoutedExperts, weights: Iterable[tuple[str, torch.Tensor]]
) -> Iterable[tuple[str, torch.Tensor]]:
    expert_mapping = layer.get_expert_mapping(include_fused=True)
    for name, tensor in weights:
        if tensor.ndim == 3 and tensor.shape[-2] == 1 and tensor.shape[-1] != 1:
            qual_name = f"{layer.layer_name}.{name}"
            # Match the upstream destination resolution. w1/w3 share the same
            # parameter, so inspecting the first matching entry is sufficient.
            for param_name, weight_name, _, _ in expert_mapping:
                if weight_name not in qual_name:
                    continue
                destination = qual_name.replace(weight_name, param_name)
                param = getattr(
                    layer, destination.removeprefix(f"{layer.layer_name}."), None
                )
                if (
                    "scale" in destination
                    and getattr(param, "quant_method", None)
                    == FusedMoeWeightScaleSupported.CHANNEL.value
                ):
                    tensor = tensor.transpose(-1, -2)
                break
        yield name, tensor


@patch(
    "vllm.model_executor.layers.fused_moe.routed_experts", "RoutedExperts.load_weights"
)
def load_weights(
    self: RoutedExperts, weights: Iterable[tuple[str, torch.Tensor]]
) -> Iterable[str]:
    """Normalize fused CHANNEL scales, then inherit upstream weight loading."""
    if self.is_fused_checkpoint_transposed:
        weights = _normalize_channel_scales(self, weights)
    return _original_load_weights(self, weights)
