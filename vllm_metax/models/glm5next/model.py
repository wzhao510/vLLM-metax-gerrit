from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm.model_executor.layers.logits_processor import LogitsProcessor

from vllm.model_executor.layers.mhc import (
    MHCFusedPostPreOp,
    MHCPostOp,
    MHCPreOp,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.glm4_1v import (
    Glm4vDummyInputsBuilder,
)

from vllm.model_executor.models.utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig

from vllm.models.glm5next.nvidia.kda import Glm5NextLinearAttention
from vllm.models.glm5next.nvidia.multimodal import (
    Glm5NextMultiModalProcessor,
    Glm5NextProcessingInfo,
)
from vllm.models.glm5next.nvidia.model import (
    _try_load_fp8_attn_proj,
    _try_load_fp8_indexer_wk,
    get_spec_layer_idx_from_weight_name,
    Glm5NextMoE,
    Glm5NextMLP,
    Glm5NextDecoderLayer,
    Glm5NextModel,
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
)

from .attention import MacaGlm5NextMLAAttention

class MacaGlm5NextDecoderLayer(Glm5NextDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextConfig,
        layer_idx: int,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        is_mtp_layer: bool = False,
        **kwargs,
    ) -> None:
        super(Glm5NextDecoderLayer, self).__init__()

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_moe = config.is_moe
        self.num_hidden_layers = config.num_hidden_layers
        self.rms_norm_eps = config.rms_norm_eps
        self.num_experts = config.n_routed_experts
        self.is_mtp_layer = is_mtp_layer
        self.mhc = config.mhc
        is_kda_layer = not is_mtp_layer and config.is_kda_layer(layer_idx)
        self.layer_kind = "kda" if is_kda_layer else "mla"
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if is_kda_layer:
            self.self_attn = Glm5NextLinearAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            # MLA layers require the latent head dims, which are guaranteed set
            # on MLA configs; narrow away the `int | None`.
            assert config.v_head_dim is not None
            assert config.kv_lora_rank is not None
            self.self_attn = MacaGlm5NextMLAAttention(
                vllm_config=vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                max_position_embeddings=config.max_position_embeddings,
                cache_config=cache_config,
                #/-------------------MetaX-------------------\#
                #[TODO] test with quant_config=vllm_config.quant_config, and remove .attention.MacaGlm5NextMLAAttention
                quant_config=None,  # MLA projections are BF16 in checkpoint
                #\-------------------MetaX-------------------/#
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
                skip_rope=config.mla_nope,
            )

        # MTP layers sit past the base model's hidden layers (layer_idx >=
        # num_hidden_layers), so they're outside mlp_layer_types; default them
        # to the last base layer's MLP type (sparse/MoE for these checkpoints).
        mlp_layer_types = config.mlp_layer_types
        mlp_type = (
            mlp_layer_types[layer_idx]
            if layer_idx < len(mlp_layer_types)
            else (mlp_layer_types[-1] if mlp_layer_types else "sparse")
        )
        if self.is_moe and self.num_experts is not None and mlp_type == "sparse":
            self.mlp = Glm5NextMoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Glm5NextMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                swiglu_limit=config.swiglu_limit,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Cached for the hot forward path (isinstance per layer per step).
        self._mlp_is_moe = isinstance(self.mlp, Glm5NextMoE)
        # In SP, the attention output projection leaves a partial sum; the
        # decoder-layer reduce_scatter after attention completes it (DSv4 pattern).
        # MTP layers use the non-mHC path which has no sp_reduce_scatter, so
        # their o_proj must still reduce normally.
        if self.is_sequence_parallel and not is_mtp_layer:
            self.self_attn.o_proj.reduce_results = False
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if self.mhc and not is_mtp_layer:
            # mhc config
            self.mhc_num_residual_streams = config.mhc_num_residual_streams
            self.mhc_tau = config.mhc_tau
            self.hc_eps = config.hc_eps
            self.mhc_sinkhorn_iterations = config.mhc_sinkhorn_iterations
            self.mhc_post_mult_value = config.mhc_post_mult_value

            n = config.mhc_num_residual_streams
            d_model = n * self.hidden_size
            mix_hc = (2 + n) * n

            self.n = n

            # attn hc
            self.hc_attn_fn = nn.Parameter(
                torch.empty(mix_hc, d_model, dtype=torch.float32)
            )
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

            # ffn hc
            self.hc_ffn_fn = nn.Parameter(
                torch.empty(mix_hc, d_model, dtype=torch.float32)
            )
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

            self.mhc_pre_op = MHCPreOp()
            self.mhc_post_op = MHCPostOp()
            self.mhc_fused_post_pre_op = MHCFusedPostPreOp()

            if vllm_config.kernel_config.enable_jit_warmup:
                #/-------------------MetaX Modification-------------------\#
                from vllm_metax.model_executor.kernels.mhc.tilelang_kernels import (
                    _HC_PRENORM_GEMM_TILELANG_KERNEL,
                    _MHC_FUSED_TILELANG_KERNEL,
                    _MHC_POST_TILELANG_KERNEL,
                    _MHC_PRE_BIG_FUSE_TILELANG_KERNEL,
                )
                #\-------------------MetaX Modification-------------------/#
                from vllm.utils.deep_gemm import is_deep_gemm_supported

                include_pre_gemm_splits = is_deep_gemm_supported()
                _MHC_PRE_BIG_FUSE_TILELANG_KERNEL.register_warmup(
                    vllm_config,
                    hidden_size=self.hidden_size,
                    hc_mult=self.n,
                    use_norm_weight=True,
                    include_pre_gemm_splits=include_pre_gemm_splits,
                    include_broadcast_splits=False,
                    rms_eps=self.rms_norm_eps,
                    hc_pre_eps=self.hc_eps,
                    hc_sinkhorn_eps=self.hc_eps,
                    hc_post_mult_value=self.mhc_post_mult_value,
                    sinkhorn_repeat=self.mhc_sinkhorn_iterations,
                    norm_eps=(
                        self.input_layernorm.variance_epsilon,
                        self.post_attention_layernorm.variance_epsilon,
                    ),
                )
                if not include_pre_gemm_splits:
                    _HC_PRENORM_GEMM_TILELANG_KERNEL.register_warmup(
                        vllm_config,
                        hidden_size=self.hidden_size,
                        hc_mult=self.n,
                        n_out=self.n * (2 + self.n),
                    )
                _MHC_POST_TILELANG_KERNEL.register_warmup(
                    hidden_size=self.hidden_size,
                    hc_mult=self.n,
                )
                _MHC_FUSED_TILELANG_KERNEL.register_warmup(
                    vllm_config,
                    hidden_size=self.hidden_size,
                    hc_mult=self.n,
                )





#################################################
# Following:
#   Only substitution for Maca version of classes 
#################################################



class MacaGlm5NextModel(Glm5NextModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Glm5NextModel, self).__init__()

        config = vllm_config.model_config.hf_config
        self.config = config

        self.vocab_size = config.vocab_size
        self.device = current_platform.device_type

        self.is_v32 = config.index_topk is not None
        if self.is_v32:
            topk_tokens = config.index_topk
            assert topk_tokens is not None
            # Reserve room for the incomplete pool tail.
            kpool = config.index_kpool
            assert kpool is not None
            buffer_width = topk_tokens + (kpool - 1 if kpool > 1 else 0)
            # Sparse MLA tiles top-k in 128 columns; padded slots remain masked.
            sparse_topk_block_n = 128
            buffer_width = (
                (buffer_width + sparse_topk_block_n - 1) // sparse_topk_block_n
            ) * sparse_topk_block_n
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                buffer_width,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            # Full-MLA config (no kpool sparse indexer): no topk buffer.
            topk_indices_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return MacaGlm5NextDecoderLayer( # Metax Modification
                vllm_config=vllm_config,
                config=config,
                layer_idx=layer_idx,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        # The active slice is fixed after construction; cache it so forward
        # doesn't rebuild the slice (a fresh list) every step.
        self._active_layers = self.layers[self.start_layer : self.end_layer]

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.is_sequence_parallel = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )

        world_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0, (
            "num_attention_heads must be divisible by world_size"
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            # MLA: fuse q_a_proj and kv_a_proj_with_mqa
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            # Indexer: fuse wk and weights_proj
            (".wk_weights_proj", ".wk", 0),
            (".wk_weights_proj", ".weights_proj", 1),
            # KDA: merge q, k, v, b, f_a, g_a projections into one GEMM
            (".in_proj_qkvbfg_a", ".q_proj", 0),
            (".in_proj_qkvbfg_a", ".k_proj", 1),
            (".in_proj_qkvbfg_a", ".v_proj", 2),
            (".in_proj_qkvbfg_a", ".b_proj", 3),
            (".in_proj_qkvbfg_a", ".f_a_proj", 4),
            (".in_proj_qkvbfg_a", ".g_a_proj", 5),
        ]
        if self.config.is_moe:
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)
            # EPLB: the mapping enumerates physical experts, so it must cover
            # the redundant replicas or their slots are never loaded.
            num_redundant_experts = next(
                (
                    layer.mlp.n_redundant_experts
                    for layer in self.layers
                    if isinstance(layer, MacaGlm5NextDecoderLayer) # Metax Modification
                    and isinstance(layer.mlp, Glm5NextMoE)
                ),
                0,
            )
            expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.n_routed_experts,
                num_redundant_experts=num_redundant_experts,
            )
        else:
            expert_params_mapping = []
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # GLM-5.3-Flash NoPE checkpoints omit the RoPE rows from
        # ``kv_a_proj_with_mqa``; pad them with zeros for the model shape.
        kv_a_pad_size = 0
        if self.config.mla_nope and self.config.qk_rope_head_dim > 0:
            kv_a_pad_size = self.config.qk_rope_head_dim

        _pending_wk_fp8: dict = {}

        for args in weights:
            name, loaded_weight = args[:2]
            kwargs: dict = args[2] if len(args) > 2 else {}
            if "rotary_emb.inv_freq" in name:
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue

            # Handle FP8 indexer WK: dequantize to BF16 for fusion with
            # weights_proj into wk_weights_proj.
            if _try_load_fp8_indexer_wk(
                name,
                loaded_weight,
                _pending_wk_fp8,
                params_dict,
                loaded_params,
            ):
                continue

            # FP8 checkpoint: dequantize BF16-kept MLA projections
            # (q_a_proj / kv_a_proj_with_mqa / o_proj) to BF16.
            if _try_load_fp8_attn_proj(
                name,
                loaded_weight,
                _pending_wk_fp8,
                params_dict,
                loaded_params,
                kv_a_pad_size,
            ):
                continue

            # Pad kv_a_proj_with_mqa for NoPE models
            if kv_a_pad_size > 0 and ".kv_a_proj_with_mqa." in name:
                pad = torch.zeros(
                    kv_a_pad_size,
                    *loaded_weight.shape[1:],
                    dtype=loaded_weight.dtype,
                    device=loaded_weight.device,
                )
                loaded_weight = torch.cat([loaded_weight, pad], dim=0)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                # QKV fusion: skip if fused module doesn't exist in model
                if param_name == ".fused_qkv_a_proj" and name_mapped not in params_dict:
                    continue
                name = name_mapped
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    expert_shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    # A checkpoint expert may map to several physical replicas
                    # under EPLB; keep `name` intact and try the next entry
                    # when this physical expert is not local to the rank.
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = param.weight_loader
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        expert_id=expert_id,
                        shard_id=expert_shard_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias")
                        and name not in params_dict
                        and not self.config.is_linear_attn
                    ):  # noqa: E501
                        continue
                    # Remapping the name of FP8 kv-scale.
                    remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped_name is None:
                        continue
                    name = remapped_name
                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, **kwargs)
            loaded_params.add(name)
        return loaded_params


class MacaGlm5NextForCausalLM(Glm5NextForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Glm5NextForCausalLM, self).__init__()
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config = self.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        self.model = MacaGlm5NextModel( # Metax Modification
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, scale=self.config.logit_scale
        )


@MULTIMODAL_REGISTRY.register_processor(
    Glm5NextMultiModalProcessor,
    info=Glm5NextProcessingInfo,
    dummy_inputs=Glm4vDummyInputsBuilder,
)
class MacaGlm5NextForConditionalGeneration(Glm5NextForConditionalGeneration):
    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        from .model import MacaGlm5NextForCausalLM # Metax Modification
        return MacaGlm5NextForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        from .model import MacaGlm5NextForCausalLM # Metax Modification
        return MacaGlm5NextForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        from .model import MacaGlm5NextForCausalLM # Metax Modification
        return MacaGlm5NextForCausalLM.get_mamba_state_copy_func()

    def set_moe_parameters(self) -> None:
        self.moe_mlp_layers = [
            layer.mlp
            for layer in self.language_model.model.layers
            if isinstance(layer, MacaGlm5NextDecoderLayer) # Metax Modification
            and isinstance(layer.mlp, Glm5NextMoE)
        ]
        self.moe_layers = [moe.experts for moe in self.moe_mlp_layers]
        self.num_moe_layers = len(self.moe_layers)
        if not self.num_moe_layers:
            return
        example_moe = self.moe_mlp_layers[0]
        self.num_expert_groups = self.config.text_config.n_group
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_shared_experts = example_moe.n_shared_experts
        self.num_redundant_experts = example_moe.n_redundant_experts