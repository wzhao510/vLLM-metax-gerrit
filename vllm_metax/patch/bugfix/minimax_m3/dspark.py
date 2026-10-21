# SPDX-License-Identifier: Apache-2.0
"""MiniMax DSpark configuration backport for vLLM 0.30.0.

Remove when upstream resolves the dspark/ checkpoint and recognizes its
architecture. No installed vLLM source files are modified.
"""
import ast
import inspect
import os
import textwrap

from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_reduce

from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models import ModelRegistry


from vllm_metax.patch.utils import patch

_ARCH = "DSparkMiniMaxDraftModel"
_original_post_init = SpeculativeConfig.__post_init__


def _with_minimax_allowlist(original):
    # 0.30 has no architecture extension point before its DeepSeek fallback.
    # Add exactly one exclusion to that guard, retaining all other validation.
    # Fail closed if the pinned upstream function no longer has this shape.
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    function.decorator_list = []
    matches = []
    for node in ast.walk(function):
        if (isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And)
                and any(isinstance(value, ast.Compare)
                        and isinstance(value.left, ast.Constant)
                        and value.left.value == "K3DSparkModel"
                        and len(value.ops) == 1
                        and isinstance(value.ops[0], ast.NotIn)
                        for value in node.values)):
            matches.append(node)
    if len(matches) != 1:
        raise RuntimeError("MiniMax DSpark backport requires vLLM 0.30's unique DeepSeek fallback guard")
    matches[0].values.append(ast.parse(
        f"{_ARCH!r} not in self.draft_model_config.architectures", mode="eval").body)
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, inspect.getfile(original), "exec"), original.__globals__, namespace)
    return namespace[function.name]


_post_init_with_minimax = _with_minimax_allowlist(_original_post_init)


@patch("vllm.config.speculative", "SpeculativeConfig.__post_init__")
def __post_init__(self):
    target = self.target_model_config
    if (self.method == "dspark" and self.model is None and target is not None
            and target.hf_config.model_type in ("minimax_m3_vl", "minimax_m3_text")):
        # This backport targets local checkpoints. A remote draft can still
        # be supplied explicitly through speculative_config.model.
        self.model = os.path.join(target.model, "dspark")
        if not os.path.isfile(os.path.join(self.model, "config.json")):
            raise ValueError(
                f"MiniMax DSpark checkpoint not found in {self.model}; "
                "set speculative_config.model to a standalone DSpark checkpoint")
        # Deliberately do not inherit target quantization: the standalone draft
        # can use compressed-tensors while the target uses block-FP8.
    return _post_init_with_minimax(self)


@patch("vllm.models.minimax_m3.nvidia.model", "MiniMaxM3Model._maybe_add_hidden_state")
def _maybe_add_hidden_state(self, aux_hidden_states, layer_idx, hidden_states, residual):
    if layer_idx in self.aux_hidden_state_layers:
        # The old target captures before the next input norm completes the
        # previous FFN's deferred reduction. Reduce a copy, not the live stream:
        # the next input norm must still perform its original all-reduce.
        if (layer_idx > self.start_layer
                and self.layers[layer_idx - 1].ffn_all_reduce_deferred
                and get_tensor_model_parallel_world_size() > 1):
            hidden_states = tensor_model_parallel_all_reduce(hidden_states.clone())
        value = hidden_states + residual if residual is not None else hidden_states.clone()
        aux_hidden_states.append(value)
    return aux_hidden_states


ModelRegistry.register_model(
    _ARCH, "vllm_metax.models.minimax_m3_dspark:MiniMaxM3DSparkForCausalLM")
