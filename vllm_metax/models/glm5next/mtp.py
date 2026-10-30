# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

"""MetaX GLM-5.3 MTP with QuaRot online rotation support."""

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
        eh_input = fused_eh_norm(
            positions,
            inputs_embeds,
            previous_hidden_states,
            self.enorm.weight,
            self.hnorm.weight,
            self.enorm.variance_epsilon,
        )
        hidden_states = self.eh_proj(eh_input)
        # Fuse the residual add and final RMSNorm. Glm5NextMoE already performs
        # its all-reduce, so no collective is needed here. The post-norm result
        # feeds both draft logits and the next recycled hidden state.
        hidden_states, residual, _, _ = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None
        )
        if self.use_hadamard_rotation:
            # Rotate the summed residual stream back before shared_head norm.
            import os
            _x = hidden_states + residual
            if os.getenv("VLLM_METAX_HADAMARD_DEBUG"):
                print(f"[ROTATION-DEBUG-MTP] before mean={_x.float().mean().item():.6f} "
                      f"std={_x.float().std().item():.6f}", flush=True)
            hidden_states = self.shared_head.prenorm_hadamard_rotation(_x)
            if os.getenv("VLLM_METAX_HADAMARD_DEBUG"):
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


__all__ = [
    "MacaGlm5NextMTP",
    "MacaGlm5NextMultiTokenPredictor",
    "MacaGlm5NextMultiTokenPredictorLayer",
]
