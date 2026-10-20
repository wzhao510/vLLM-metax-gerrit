# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ----------------------------------------
# NOTE(metax): Register the MHC ops for maca

import torch
from vllm.model_executor.layers.mhc import (
    MHCPreOp as _vllm_MHCPreOp,
    MHCPreDelayedOp as _vllm_MHCPreDelayedOp,
    MHCPostOp as _vllm_MHCPostOp,
    MHCFusedPostPreOp as _vllm_MHCFusedPostPreOp,
    HCHeadOp as _vllm_HCHeadOp,
)

import vllm_metax.model_executor.kernels.mhc.tilelang  # noqa: F401

# --8<-- [start:mhc_pre]
@_vllm_MHCPreOp.register_oot
class MHCPreOp(_vllm_MHCPreOp):
    """MHC pre block.

    Computes mix logits from RMS-normalized HC residual streams, then
    returns post_mix, comb_mix, and
    layer_input = sum_i pre_mix_i * residual_i.
    """

    # --8<-- [end:mhc_pre]
    def forward_oot(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.mx_mhc_pre_tilelang(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            norm_weight,
            norm_eps,
        )


# --8<-- [start:mhc_pre_delayed]
@_vllm_MHCPreDelayedOp.register_oot
class MHCPreDelayedOp(_vllm_MHCPreDelayedOp):
    """MHC pre block using the pre-mix carried from the previous sublayer.

    Same gates as :class:`MHCPreOp`, but the stream collapse applies the
    caller's ``pre_mix`` and this sublayer's pre-mix is returned for the next
    sublayer seam.

    Passing ``sublayer_out`` / ``post_layer_mix`` / ``comb_res_mix`` also
    applies the preceding post block, which lets AITER fold it into the pre
    projection and saves a launch. Returns residual, post_mix, comb_mix,
    layer_input, next_pre_mix; ``residual`` is the caller's own tensor when
    no post is requested.
    """

    # --8<-- [end:mhc_pre_delayed]
    def __init__(self) -> None:
        super(_vllm_MHCPreDelayedOp, self).__init__()
        # Built here, not lazily: CustomOp resolves dispatch against the
        # current vLLM config, which is only set during model construction.
        self._post = MHCPostOp()

    def _maybe_post(
        self,
        residual: torch.Tensor,
        sublayer_out: torch.Tensor | None,
        post_layer_mix: torch.Tensor | None,
        comb_res_mix: torch.Tensor | None,
    ) -> torch.Tensor:
        if sublayer_out is None:
            return residual
        assert post_layer_mix is not None and comb_res_mix is not None
        return self._post(sublayer_out, residual, post_layer_mix, comb_res_mix)

    def forward_oot(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        pre_mix: torch.Tensor | None = None,
        x: torch.Tensor | None = None,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 1e-6,
        sublayer_out: torch.Tensor | None = None,
        post_layer_mix: torch.Tensor | None = None,
        comb_res_mix: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = self._maybe_post(
            residual, sublayer_out, post_layer_mix, comb_res_mix
        )
        return residual, *torch.ops.vllm.mx_mhc_pre_delayed_tilelang(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            pre_mix,
            x,
            norm_weight,
            norm_eps,
        )


# --8<-- [start:mhc_post]
@_vllm_MHCPostOp.register_oot
class MHCPostOp(_vllm_MHCPostOp):
    """MHC post block.

    Combines the layer output with the HC residual streams:
    out_j = post_layer_mix_j * x + sum_i comb_res_mix_ij * residual_i.
    """

    # --8<-- [end:mhc_post]

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.vllm.mx_mhc_post_tilelang(
            x, residual, post_layer_mix, comb_res_mix
        )


# --8<-- [start:hc_head]
@_vllm_HCHeadOp.register_oot
class HCHeadOp(_vllm_HCHeadOp):
    """HC head reduction for DeepSeek V4.

    Computes gates from the RMS-normalized flattened HC residual and
    returns out = sum_i gate_i * residual_i, collapsing hc_mult streams
    to one.
    """

    # --8<-- [end:hc_head]
    def __init__(self) -> None:
        super().__init__()

    def forward_oot(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_norm_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        hc_mult, hidden_size = hidden_states.shape[-2:]
        outer_shape = hidden_states.shape[:-2]
        hs_flat = hidden_states.view(-1, hc_mult, hidden_size)
        out = torch.ops.vllm.mx_hc_head_fused_kernel_tilelang(
            hs_flat,
            hc_fn,
            hc_scale,
            hc_base,
            rms_norm_eps,
            hc_eps,
        )
        return out.view(*outer_shape, hidden_size)


# --8<-- [start:mhc_fused_post_pre]
@_vllm_MHCFusedPostPreOp.register_oot
class MHCFusedPostPreOp(_vllm_MHCFusedPostPreOp):
    """Fused MHC post block followed by the next MHC pre block.

    Equivalent to applying MHCPostOp and then MHCPreOp to the updated
    residual streams, returning residual_cur, post_mix_cur, comb_mix_cur,
    and layer_input_cur.
    """

    # --8<-- [end:mhc_fused_post_pre]

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.mx_mhc_fused_post_pre_tilelang(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            tile_n,
            norm_weight,
            norm_eps,
        )


def hc_expand(x: torch.Tensor, n: int) -> torch.Tensor:
    """[s, hidden_size] -> [s, n * hidden_size] by replication."""
    return x.unsqueeze(1).expand(-1, n, -1).contiguous()


def hc_contract(x: torch.Tensor, n: int) -> torch.Tensor:
    """[s, n * hidden_size] -> [s, hidden_size] by averaging."""
    return x.mean(dim=1)
