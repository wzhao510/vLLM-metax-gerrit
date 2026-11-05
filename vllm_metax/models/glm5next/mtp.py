# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

"""MetaX GLM-5.3 MTP with QuaRot online rotation support."""

import os

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.glm5next.nvidia.model import Glm5NextMLAAttention
from vllm.models.glm5next.nvidia.mtp import (
    Glm5NextMTP,
    Glm5NextMultiTokenPredictor,
    Glm5NextMultiTokenPredictorLayer,
    SharedHead,
)
from vllm.models.glm5next.nvidia.ops.fused_eh_norm import fused_eh_norm
from vllm.platforms import current_platform

from .model import MacaGlm5NextDecoderLayer

from vllm_metax.model_executor.layers.online_rotation import (
    build_rotation,
    get_online_rotation_module_config,
)


def _fused_eh_norm_unfused(
    positions: torch.Tensor,
    inputs_embeds: torch.Tensor,
    previous_hidden_states: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Pure-PyTorch fallback for the nvidia ``fused_eh_norm`` op.

    The upstream op is a fused CUDA kernel with no MetaX backend; if it
    produces garbage on MACA this bit-exact equivalent (zero pos-0 embeds +
    enorm(embeds) + hnorm(prev) + cat -> [N, 2H]) can be enabled via
    ``VLLM_METAX_MTP_UNFUSED_EH_NORM=1``.
    """
    # Zero position-0 embeddings (the draft's first slot has no real embed).
    mask = (positions != 0).to(inputs_embeds.dtype).unsqueeze(-1)
    embeds = inputs_embeds * mask
    # enorm(embeds) -- RMSNorm
    embeds_norm = embeds * torch.rsqrt(
        embeds.pow(2).mean(dim=-1, keepdim=True) + eps
    ) * enorm_weight
    # hnorm(prev) -- RMSNorm
    prev_norm = previous_hidden_states * torch.rsqrt(
        previous_hidden_states.pow(2).mean(dim=-1, keepdim=True) + eps
    ) * hnorm_weight
    return torch.cat([embeds_norm, prev_norm], dim=-1)


class MacaGlm5NextMultiTokenPredictorLayer(Glm5NextMultiTokenPredictorLayer):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        # Do not delegate to the upstream layer: it builds the MTP block with
        # the nvidia decoder, which hard-codes ``quant_config=None`` for MLA
        # projections. That drops the W8A8 ``kv_a_proj_with_mqa.weight_scale``
        # this checkpoint stores for layer 45, so draft weight loading fails.
        # Building the block with ``MacaGlm5NextDecoderLayer`` reuses our MLA
        # that upgrades itself to compressed-tensors W8A8.
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )

        topk_tokens = config.index_topk
        assert topk_tokens is not None
        kpool = config.index_kpool
        assert kpool is not None
        buffer_width = topk_tokens + (kpool - 1 if kpool > 1 else 0)
        sparse_topk_block_n = 128
        buffer_width = (
            (buffer_width + sparse_topk_block_n - 1) // sparse_topk_block_n
        ) * sparse_topk_block_n
        topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            buffer_width,
            dtype=torch.int32,
            device=current_platform.device_type,
        )
        self.shared_head = SharedHead(
            config=config, prefix=prefix, quant_config=quant_config
        )

        layer_idx = int(prefix.rsplit(".", 1)[-1])
        self.mtp_block = MacaGlm5NextDecoderLayer(
            vllm_config=vllm_config,
            config=config,
            layer_idx=layer_idx,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
            is_mtp_layer=True,
        )

        rotation_module_name = maybe_prefix(
            prefix,
            "shared_head.prenorm_hadamard_rotation",
        )
        rotation_config = get_online_rotation_module_config(
            self.config,
            rotation_module_name,
        )
        self.use_hadamard_rotation = rotation_config is not None
        if rotation_config is not None:
            self.shared_head.prenorm_hadamard_rotation = build_rotation(
                rotation_config
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        # Fused: zero pos-0 embeds + enorm(embeds) + hnorm(prev) + cat -> [N, 2H].
        if os.getenv("VLLM_METAX_MTP_UNFUSED_EH_NORM"):
            eh_input = _fused_eh_norm_unfused(
                positions,
                inputs_embeds,
                previous_hidden_states,
                self.enorm.weight,
                self.hnorm.weight,
                self.enorm.variance_epsilon,
            )
        else:
            eh_input = fused_eh_norm(
                positions,
                inputs_embeds,
                previous_hidden_states,
                self.enorm.weight,
                self.hnorm.weight,
                self.enorm.variance_epsilon,
            )
        hidden_states = self.eh_proj(eh_input)
        _dbg = os.getenv("VLLM_METAX_HADAMARD_DEBUG")
        if _dbg:
            def _stats(tag, t):
                tf = t.float()
                print(f"[MTP-FWD] {tag:18s} mean={tf.mean().item():.4e} "
                      f"std={tf.std().item():.4e} "
                      f"finite={bool(torch.isfinite(tf).all().item())}",
                      flush=True)
            _stats("inputs_embeds", inputs_embeds)
            _stats("prev_hidden", previous_hidden_states)
            _stats("eh_input", eh_input)
            _stats("post_eh_proj", hidden_states)
        # Fuse the residual add and final RMSNorm. Glm5NextMoE already performs
        # its all-reduce, so no collective is needed here. The post-norm result
        # feeds both draft logits and the next recycled hidden state.
        hidden_states, residual, _, _ = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None
        )
        if _dbg:
            _stats("mtp_hidden_out", hidden_states)
            _stats("mtp_residual", residual)
        if self.use_hadamard_rotation:
            # Rotate the summed residual stream back before shared_head norm.
            _x = hidden_states + residual
            if _dbg:
                print(f"[ROTATION-DEBUG-MTP] before mean={_x.float().mean().item():.6f} "
                      f"std={_x.float().std().item():.6f}", flush=True)
            hidden_states = self.shared_head.prenorm_hadamard_rotation(_x)
            if _dbg:
                print(f"[ROTATION-DEBUG-MTP] after  mean={hidden_states.float().mean().item():.6f} "
                      f"std={hidden_states.float().std().item():.6f}", flush=True)
            hidden_states = self.shared_head.norm(hidden_states)
        else:
            hidden_states, _ = self.shared_head.norm(
                hidden_states, residual=residual
            )
        return hidden_states, hidden_states


class MacaGlm5NextMultiTokenPredictor(Glm5NextMultiTokenPredictor):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Glm5NextMultiTokenPredictor, self).__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): MacaGlm5NextMultiTokenPredictorLayer(
                    vllm_config, f"{prefix}.layers.{idx}"
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        # Plain list for the per-propose lookup: ModuleDict[str(...)] builds a
        # string and hashes it on every draft step.
        self._mtp_layers = list(self.layers.values())
        self._mtp_mla_attns = []
        for layer in self._mtp_layers:
            self_attn = layer.mtp_block.self_attn
            assert isinstance(self_attn, Glm5NextMLAAttention)
            self._mtp_mla_attns.append(self_attn.mla_attn)
        self.logits_processor = LogitsProcessor(config.vocab_size)


class MacaGlm5NextMTP(Glm5NextMTP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Glm5NextMTP, self).__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = MacaGlm5NextMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.set_moe_parameters()

    def load_weights(self, weights):
        # The GLM-5.3 checkpoint stores the MTP language-model head at the
        # top-level ``lm_head.weight`` key (shared with the base model), not
        # under the spec layer.  The inherited Glm5NextMTP.load_weights only
        # accepts spec-layer keys (it skips anything get_spec_layer_idx...
        # returns None for), so without this remap shared_head.head stays at
        # its torch.empty initial value and draft logits are garbage.
        spec_layer = self.config.num_hidden_layers
        head_target = f"model.layers.{spec_layer}.shared_head.head.weight"

        def _remap_head(iterable):
            for name, w in iterable:
                if name == "lm_head.weight":
                    yield head_target, w
                else:
                    yield name, w

        loaded = super().load_weights(_remap_head(weights))
        self._probe_mtp_weights(loaded)
        return loaded

    def _probe_mtp_weights(self, loaded_params: set) -> None:
        """Post-load diagnostic (VLLM_METAX_MTP_WEIGHT_PROBE).

        Prints which layer-45 params actually loaded vs. which stay at
        torch.empty. Non-sharded params (layernorms, eh_proj) are replicated
        across TP ranks, so unloaded ones produce identical inf output on
        every rank -- the symptom we are chasing.
        """
        if not os.getenv("VLLM_METAX_MTP_WEIGHT_PROBE"):
            return

        for layer_idx, layer in self.model.layers.items():
            print(f"[MTP-WEIGHT] === layer {layer_idx} ===", flush=True)
            probe_targets = [
                ("eh_proj", layer.eh_proj.weight),
                ("enorm", layer.enorm.weight),
                ("hnorm", layer.hnorm.weight),
                ("mtp_block.input_layernorm",
                 layer.mtp_block.input_layernorm.weight),
                ("mtp_block.post_attention_layernorm",
                 layer.mtp_block.post_attention_layernorm.weight),
                ("shared_head.norm", layer.shared_head.norm.weight),
                ("shared_head.head", layer.shared_head.head.weight),
                ("mtp_block.self_attn.fused_qkv_a_proj",
                 layer.mtp_block.self_attn.fused_qkv_a_proj.weight),
                ("mtp_block.self_attn.kv_b_proj",
                 layer.mtp_block.self_attn.kv_b_proj.weight),
            ]
            # W4A8 MoE: spot-check the first routed expert's gate weight.
            # moe.experts may be a MoERunner (not len()-able), so guard it.
            try:
                moe = layer.mtp_block.mlp
                if hasattr(moe, "experts"):
                    experts = moe.experts
                    # experts is nn.ModuleList in FusedMoE, MoERunner otherwise.
                    if isinstance(experts, torch.nn.ModuleList) and len(experts) > 0:
                        exp0 = experts[0]
                        for pname in ("gate_proj", "up_proj", "down_proj"):
                            if hasattr(exp0, pname):
                                probe_targets.append(
                                    (f"mtp_block.mlp.exp0.{pname}",
                                     getattr(exp0, pname).weight)
                                )
            except Exception as exc:
                print(f"[MTP-WEIGHT] moe probe skipped: {exc}", flush=True)
            moe_gate = getattr(getattr(layer.mtp_block, "mlp", None), "gate", None)
            if moe_gate is not None:
                probe_targets.append(("mtp_block.mlp.gate", moe_gate.weight))
            indexer = getattr(layer.mtp_block.self_attn, "indexer", None)
            if indexer is not None:
                probe_targets.append(
                    ("mtp_block.self_attn.indexer.wq_b", indexer.wq_b.weight)
                )
            for tag, w in probe_targets:
                wf = w.float()
                finite = bool(torch.isfinite(wf).all().item())
                absmax = wf.abs().max().item()
                status = "OK" if (finite and absmax < 1e3) else "GARBAGE"
                print(
                    f"[MTP-WEIGHT] {tag:35s} dtype={str(w.dtype):12s} "
                    f"mean={wf.mean().item():.4e} std={wf.std().item():.4e} "
                    f"absmax={absmax:.4e} finite={finite} -> {status}",
                    flush=True,
                )
            # Which layer params were NOT loaded?
            all_names = [n for n, _ in layer.named_parameters()]
            loaded_layer = [
                n for n in all_names
                if any(n in lp or lp.endswith(n) for lp in loaded_params)
            ]
            missing = [n for n in all_names if n not in loaded_layer]
            print(
                f"[MTP-WEIGHT] loaded {len(loaded_layer)}/{len(all_names)} "
                f"params; missing={missing[:20]}",
                flush=True,
            )


__all__ = [
    "MacaGlm5NextMTP",
    "MacaGlm5NextMultiTokenPredictor",
    "MacaGlm5NextMultiTokenPredictorLayer",
]
