# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: MiniMax-M3 calls FlashInfer Gemma norm and fused all-reduce directly.
#     Use graph-capturable Torch math and the platform's standard TP collective
#     on MetaX, preserving the fused norm's in-place residual contract.
#
# Affected versions: vLLM 0.30.0 (ced6857afa0e).
#
# Remove at: M3 uses platform-dispatched Gemma norm and all-reduce helpers.
# -----------------------------------------------------------------------------

"""FlashInfer-free MiniMax-M3 normalization and deferred TP reduction."""

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
    fused_allreduce_gemma_rms_norm as _original_allreduce_norm,
)
from vllm.models.minimax_m3.nvidia.model import MiniMAXGemmaRMSNorm

from vllm_metax.patch.utils import patch


@patch("vllm.models.minimax_m3.nvidia.model", "MiniMAXGemmaRMSNorm.forward")
def forward(
    self,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    # Keep the unrounded FP32 sum for the variance, like the fused Gemma kernel.
    value = x.float()
    if residual is not None:
        value = value + residual.float()
    variance = value.pow(2).mean(dim=-1, keepdim=True)
    output = value * torch.rsqrt(variance + self.variance_epsilon)
    output = (output * (self.weight.float() + 1.0)).to(x.dtype)
    if residual is None:
        return output
    residual.copy_(value)
    x.copy_(output)
    return x, residual


@patch("vllm.model_executor.layers.fused_allreduce_gemma_rms_norm")
def fused_allreduce_gemma_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm,
) -> tuple[torch.Tensor, torch.Tensor]:
    # The patch also repoints loaded aliases; leave non-M3 callers unchanged.
    if not isinstance(norm, MiniMAXGemmaRMSNorm):
        return _original_allreduce_norm(hidden_states, residual, norm)
    if get_tensor_model_parallel_world_size() > 1:
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)
    return norm(hidden_states, residual)
