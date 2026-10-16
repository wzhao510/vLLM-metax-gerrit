# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: This patch is to relax platform check at EngramConfig.verify_model_config,
#       since MACA is CUDA-like and compatible with the Engram code. Without it,
#       a non-empty --engram-config is rejected on MACA and the DeepSeek-V4.1
#       n-gram tables cannot be configured.
#
# Purpose: accept Engram configuration for CUDA-like out-of-tree platforms.
# Affected versions: vLLM 0.30.0, reviewed 2026-10-16.
#
# Remove at: Upstream Engram config accepts CUDA-like out-of-tree platforms.
# -----------------------------------------------------------------------------

from typing import TYPE_CHECKING

from vllm.config.engram import _NGRAM_LAYER_FIELDS

from vllm_metax.patch.utils import patch

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig


@patch("vllm.config.engram", "EngramConfig.verify_model_config")
def verify_model_config(self, model_config: "ModelConfig | None") -> None:
    """Reject Engram configuration for models without n-gram embeddings."""
    from vllm.platforms import current_platform

    field = (
        _NGRAM_LAYER_FIELDS.get(model_config.architecture)
        if model_config is not None
        else None
    )
    if (
        model_config is None
        or field is None
        # /-------------------- MetaX Modification --------------------\
        # MACA is CUDA-like and compatible with the following Engram code.
        or not current_platform.is_cuda_alike()
        # \-------------------- MetaX Modification --------------------/
        or not getattr(model_config.hf_text_config, field, None)
    ):
        raise ValueError(
            "EngramConfig requires a model with supported Engram "
            "embeddings, non-empty n-gram layer ids, and CUDA."
        )
