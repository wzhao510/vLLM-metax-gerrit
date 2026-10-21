# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM 36ae8803291c, minimax_m3/common/dspark.py.
"""MiniMax DSpark on vLLM 0.30's portable DFlash attention/cache API."""
from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3Attention, DFlashQwen3DecoderLayer, DFlashQwen3ForCausalLM,
    _get_dflash_fc_input_size, _resolve_layer_attention,
)
from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead, DSparkMarkovHead, Qwen3DSparkForCausalLM, Qwen3DSparkModel,
)
from vllm.model_executor.models.utils import WeightsMapper, get_draft_quant_config, maybe_prefix

CHECKPOINT_PREFIX = "language_model.model.dspark."


def rename_checkpoint_weights(weights: Iterable):
    for name, tensor in weights:
        name = name.removeprefix(CHECKPOINT_PREFIX).replace(".decoder_layer.", ".")
        if name.startswith("final_norm."):
            name = "norm." + name.removeprefix("final_norm.")
        yield name, tensor


def map_draft_quant_config(quant_config):
    if quant_config is None:
        return None
    quant_config.apply_vllm_mapper(WeightsMapper(
        orig_to_new_prefix={CHECKPOINT_PREFIX: "model."},
        orig_to_new_substr={".decoder_layer.": "."}))
    # compressed-tensors intentionally leaves regex targets unchanged.
    # BF16 auxiliary heads move from checkpoint dspark.* to runtime model.*.
    if hasattr(quant_config, "ignore"):
        quant_config.ignore = [
            name.replace(r"dspark\.", r"model\.")
            if name.startswith("re:") else name
            for name in quant_config.ignore
        ]
    return quant_config


class MiniMaxM3DSparkMLP(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        if config.hidden_act != "swigluoai" or not config.dspark_use_dense_ffn:
            raise ValueError("MiniMax DSpark requires dense swigluoai FFNs")
        intermediate = config.dspark_ffn_hidden_size
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [intermediate] * 2, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(
            intermediate, config.hidden_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.down_proj")
        self.act_fn = SiluAndMulWithClamp(
            swiglu_limit=config.swiglu_limit, alpha=config.swiglu_alpha, beta=config.swiglu_beta)

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(x))
        return x


class MiniMaxM3DSparkDecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(self, vllm_config, *, config, layer_idx, cache_config=None,
                 quant_config=None, prefix=""):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        sliding_window, causal = _resolve_layer_attention(config, layer_idx)
        self.self_attn = DFlashQwen3Attention(
            hidden_size=config.hidden_size, num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters={"rope_theta": config.rope_theta,
                             "partial_rotary_factor": config.partial_rotary_factor},
            max_position=config.max_position_embeddings, head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps, sliding_window=sliding_window,
            causal=causal, cache_config=cache_config, quant_config=quant_config,
            prefix=f"{prefix}.self_attn")
        self.self_attn.q_norm = GemmaRMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.self_attn.k_norm = GemmaRMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.mlp = MiniMaxM3DSparkMLP(config, quant_config, f"{prefix}.mlp")
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


@support_torch_compile
class MiniMaxM3DSparkModel(Qwen3DSparkModel):
    decoder_layer_cls = MiniMaxM3DSparkDecoderLayer

    def __init__(self, *, vllm_config: VllmConfig, start_layer_id=0, prefix=""):
        # 0.30's base constructor cannot map quantization exclusions before
        # constructing layers. Keep this initialization local to MiniMax.
        nn.Module.__init__(self)
        self.config = config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = config.vocab_size
        self.quant_config = map_draft_quant_config(get_draft_quant_config(vllm_config))
        self.use_aux_hidden_state = True
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=maybe_prefix(prefix, "embed_tokens"))
        self.mask_token_id = config.dflash_config["mask_token_id"]
        self.mask_embedding = nn.Parameter(torch.zeros(config.hidden_size), requires_grad=False)
        self.has_separate_mask_embedding = False
        self.layers = nn.ModuleList([
            self.decoder_layer_cls(
                vllm_config, config=config, layer_idx=i,
                cache_config=vllm_config.cache_config, quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, f"layers.{start_layer_id + i}"))
            for i in range(config.num_hidden_layers)])
        self.fc = ReplicatedLinear(
            _get_dflash_fc_input_size(vllm_config), config.hidden_size,
            bias=False, params_dtype=vllm_config.model_config.dtype,
            quant_config=self.quant_config, prefix=maybe_prefix(prefix, "fc"), return_bias=False)
        self.hidden_norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        draft_vocab = getattr(config, "draft_vocab_size", None) or config.vocab_size
        self.markov_head = DSparkMarkovHead(
            config.vocab_size, draft_vocab, config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"), quant_config=self.quant_config)
        self.confidence_head = None
        if getattr(config, "enable_confidence_head", False):
            with_markov = getattr(config, "confidence_head_with_markov", False)
            self.confidence_head = DSparkConfidenceHead(
                config.hidden_size + (config.markov_rank if with_markov else 0),
                prefix=maybe_prefix(prefix, "confidence_head"), bias=True, with_markov=with_markov)

    def _build_context_kv_buffers(self, layers_attn, has_bias):
        # Never concatenate serialized FP8/MXFP8 weights into an F.linear.
        # The registered quant method owns the post-load layout and scales.
        self._context_attentions = layers_attn
        self._k_norm_weights = torch.stack([a.k_norm.weight.data for a in layers_attn])

    def _project_context_kv(self, context_states, num_ctx, num_layers, num_kv_heads, head_dim):
        normalized = self.hidden_norm(context_states)
        kv = []
        for attention in self._context_attentions:
            qkv, _ = attention.qkv_proj(normalized)
            kv.append(qkv[:, attention.q_size:].reshape(num_ctx, 2, num_kv_heads, head_dim))
        all_kv = torch.stack(kv, dim=1).permute(2, 1, 0, 3, 4).contiguous()
        return all_kv[0], all_kv[1]

    def _normalize_context_k(self, all_k):
        return torch.stack([layer.self_attn.k_norm(k)
                            for layer, k in zip(self.layers, all_k, strict=True)])


class MiniMaxM3DSparkForCausalLM(Qwen3DSparkForCausalLM):
    model_cls = MiniMaxM3DSparkModel

    def __init__(self, *, vllm_config, prefix=""):
        # Qwen3 DSpark 0.30 hardcodes its model class; DFlash already dispatches
        # through model_cls and creates the same shared embedding/head contract.
        DFlashQwen3ForCausalLM.__init__(self, vllm_config=vllm_config, prefix=prefix)
        self.target_vocab_size = vllm_config.model_config.get_vocab_size()

    def load_weights(self, weights):
        # Only MXFP8 uses weight_scale. CT block-FP8 needs weight_scale_inv.
        def mapped():
            for name, tensor in rename_checkpoint_weights(weights):
                if self.draft_model_config.quantization in ("mxfp8", "modelopt_mxfp8"):
                    name = name.replace(".weight_scale_inv", ".weight_scale")
                yield name, tensor
        return super().load_weights(mapped())
