# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Load DeepSeek-V4 pipeline-parallel compatibility at plugin initialization. This
#     initializer applies the patches by importing the child modules.
#
# Affected versions: vLLM 0.30.0 (ced6857afa0e), reviewed 2026-10-12.
#
# Remove at: All DeepSeek-V4 patches imported here have been removed or migrated to
#     native extension points.
# -----------------------------------------------------------------------------

from . import parallel_state  # noqa: F401
from . import topk_global_indices_oob  # noqa: F401
