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
            hidden_states = self.shared_head.prenorm_hadamard_rotation(
                hidden_states + residual
            )
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


# GLM-5.3 checkpoints serialize the text tower as
# ``model.language_model.layers.N.xxx`` (HF layout), while vLLM mounts the base
# tower at ``language_model.model.`` and the MTP draft at ``model.``.
# ``compression_config.ignore`` keeps the checkpoint spelling and vLLM matches
# it by exact string equality (``should_ignore_layer``), so the draft's
# ``model.layers.45.self_attn.kv_b_proj`` never matches and falls back to the
# default W8A8 scheme: the bf16 checkpoint weight is truncated into an int8
# buffer and, since the checkpoint carries no ``weight_scale`` for it, dequant
# reads a torch.empty scale (~1e29) -> draft logits become noise.
# Mirror every ignored text-tower module under all three prefixes so each model
# tree matches its own runtime spelling.
_TEXT_TOWER_PREFIXES = (
    "model.language_model.",
    "language_model.model.",
    "model.",
)


def _ignore_prefix_aliases(entry: str) -> list[str]:
    for prefix in _TEXT_TOWER_PREFIXES:
        if entry.startswith(prefix):
            rest = entry[len(prefix) :]
            if not rest.startswith(("layers.", "embed_tokens.")):
                return [entry]
            return [f"{p}{rest}" for p in _TEXT_TOWER_PREFIXES]
    return [entry]


def _expand_quant_ignore_aliases(quant_config) -> None:
    """Rewrite ``ignore`` in place to cover every runtime prefix spelling."""
    ignore = getattr(quant_config, "ignore", None)
    if not ignore:
        return
    ignore = [str(entry) for entry in ignore]
    additions: list[str] = []
    for entry in ignore:
        for alias in _ignore_prefix_aliases(entry):
            if alias not in ignore and alias not in additions:
                additions.append(alias)
    if additions:
        quant_config.ignore = ignore + additions


class MacaGlm5NextMTP(Glm5NextMTP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Glm5NextMTP, self).__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        # Must run before the predictor is built: LinearBase resolves its
        # quant method (and thus the ignore check) during construction.
        _expand_quant_ignore_aliases(self.quant_config)
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

        return super().load_weights(_remap_head(weights))


__all__ = [
    "MacaGlm5NextMTP",
    "MacaGlm5NextMultiTokenPredictor",
    "MacaGlm5NextMultiTokenPredictorLayer",
]
