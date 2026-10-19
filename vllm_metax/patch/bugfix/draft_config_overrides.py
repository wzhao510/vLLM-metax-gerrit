# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Compose dictionary HF overrides with upstream draft-model rewrites. Upstream
#     already composes callable overrides but deliberately leaves dictionaries target-
#     only; MetaX checkpoints need dictionary inheritance.
#     Recognize JoyAI draft configs while preserving joyai_dense_mtp so the
#     decoder selects its dense MTP layer after the model-type rewrite.
#
# Affected versions: vLLM 0.30.0 (ced6857afa0e), reviewed 2026-10-12.
#     JoyAI draft identity reviewed against vLLM 9eba576dde on 2026-10-19.
#
# Remove at: Upstream supports inheriting dictionary HF overrides when constructing
#     draft model configs and preserves JoyAI's dense MTP identity.
# -----------------------------------------------------------------------------

"""Propagate dictionary HF overrides to MetaX draft models.

Affected: vLLM 0.30.0 (ced6857afa0e). Upstream composes callable overrides,
but deliberately leaves dictionaries target-only. MetaX checkpoints still need
the dictionary overrides. Remove when upstream supports opting into inheritance.
"""

import copy
from collections.abc import Mapping
from typing import Any

from transformers import PretrainedConfig
from vllm.config.speculative import SpeculativeConfig

from vllm_metax.patch.utils import patch

_original_compose = SpeculativeConfig.compose_draft_hf_overrides
_original_hf_config_override = SpeculativeConfig.hf_config_override


def _update_nested_hf_config(target, updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        nested = (
            target.get(key) if isinstance(target, dict) else getattr(target, key, None)
        )
        if isinstance(value, dict) and isinstance(nested, (dict, PretrainedConfig)):
            _update_nested_hf_config(nested, value)
        elif isinstance(target, dict):
            target[key] = copy.deepcopy(value)
        else:
            setattr(target, key, copy.deepcopy(value))


@patch("vllm.config.speculative", "SpeculativeConfig.hf_config_override")
def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
    """Recognize JoyAI's dense MTP while deriving the draft configuration."""
    if hf_config.model_type == "joyai_llm_flash":
        # The MACA decoder needs this identity after the upstream type rewrite
        # to distinguish JoyAI's dense MTP layer from its routed backbone.
        hf_config.joyai_dense_mtp = True
        hf_config.model_type = "deepseek_mtp"
    return _original_hf_config_override(hf_config)


class _DraftHfOverrides:
    """Picklable dictionary overrides followed by upstream's draft rewrite."""

    def __init__(self, target_hf_overrides: Mapping[str, Any]) -> None:
        self.target_hf_overrides = copy.deepcopy(dict(target_hf_overrides))

    def __call__(self, hf_config: PretrainedConfig) -> PretrainedConfig:
        for key, value in self.target_hf_overrides.items():
            nested = getattr(hf_config, key, None)
            if isinstance(nested, PretrainedConfig) and isinstance(value, dict):
                _update_nested_hf_config(nested, value)
            else:
                setattr(hf_config, key, copy.deepcopy(value))
        return SpeculativeConfig.hf_config_override(hf_config)


@patch("vllm.config.speculative", "SpeculativeConfig.compose_draft_hf_overrides")
def compose_draft_hf_overrides(target_hf_overrides):
    # /-------------------- MetaX Modification --------------------\
    if isinstance(target_hf_overrides, Mapping) and target_hf_overrides:
        return _DraftHfOverrides(target_hf_overrides)
    # \-------------------- MetaX Modification --------------------/
    return _original_compose(target_hf_overrides)
