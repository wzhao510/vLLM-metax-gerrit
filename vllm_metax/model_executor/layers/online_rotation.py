# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

"""Config-driven online rotation modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


from vllm_metax.model_executor.kernels.online_rotate import (
    hadamard,
    make_hadamard_gemm_matrix,
    resolve_hadamard_dim,
    select_gemm_order,
)


_TEXT_TOWER_PREFIXES = (
    "model.language_model.model.",
    "model.language_model.",
    "language_model.model.",
    "model.",
)


def _rotation_module_aliases(module_name: str) -> list[str]:
    """Return equivalent module paths for multimodal GLM checkpoints.

    GLM-5.3 checkpoints serialize the text tower under ``model.language_model.``
    while vLLM mounts it at ``language_model.model.``; the MTP head is mounted
    at ``model.``. The config keeps the checkpoint naming, so accept both.
    """
    aliases = [module_name]
    if module_name.startswith("language_model.model."):
        rest = module_name[len("language_model.model.") :]
        aliases.extend((f"model.language_model.{rest}", f"model.{rest}"))
    elif module_name.startswith("model.language_model."):
        rest = module_name[len("model.language_model.") :]
        aliases.extend((f"language_model.model.{rest}", f"model.{rest}"))
    elif module_name.startswith("model."):
        rest = module_name[len("model.") :]
        aliases.append(f"model.language_model.{rest}")
    return aliases


def _canonical_rotation_module_name(module_name: str) -> str:
    """Strip the text-tower wrapper prefix so all namings compare equal."""
    for tower_prefix in _TEXT_TOWER_PREFIXES:
        if module_name.startswith(tower_prefix):
            return module_name[len(tower_prefix) :]
    return module_name


def get_online_rotation_module_config(
    config: Any,
    module_name: str,
) -> Mapping[str, Any] | None:
    """Return the online-rotation spec mounted at ``module_name``."""
    rotation_config = getattr(config, "online_rotation", None)
    if rotation_config is None:
        return None
    if not isinstance(rotation_config, Mapping):
        raise TypeError("online_rotation must be a mapping")

    modules = rotation_config.get("modules")
    if modules is None:
        raise ValueError("online_rotation.modules is required")
    if not isinstance(modules, Mapping):
        raise TypeError("online_rotation.modules must be a mapping")

    module_config = None
    matched_name = module_name
    for candidate in _rotation_module_aliases(module_name):
        module_config = modules.get(candidate)
        if module_config is not None:
            matched_name = candidate
            break
    if module_config is None:
        canonical_name = _canonical_rotation_module_name(module_name)
        for candidate, candidate_config in modules.items():
            if (
                _canonical_rotation_module_name(candidate) == canonical_name
            ):
                module_config = candidate_config
                matched_name = candidate
                break
    if module_config is None:
        return None
    if not isinstance(module_config, Mapping):
        raise TypeError(
            f"online_rotation.modules[{matched_name!r}] must be a mapping"
        )
    return module_config


def build_rotation(config: Mapping[str, Any]) -> nn.Module:
    """Build an online rotation from one module-local rotation spec."""
    rotation_type = config.get("type")
    if rotation_type == "composite_hadamard":
        return CompositeHadamardRotation.from_rotation_config(config)
    raise ValueError(f"unsupported online rotation type: {rotation_type!r}")


class CompositeHadamardRotation(nn.Module):
    r"""CompositeHadamardRotation implements a normalized structured transform
    of the form \(D(H_K \otimes H_P)/\sqrt{N}\), where \(N=K\cdot P\) and \(P\)
    is a power of two. When \(K=1\), it reduces to the standard power-of-two
    Hadamard transform.
    """

    def __init__(
        self,
        hidden_size: int,
        sign: torch.Tensor | None = None,
        hk: torch.Tensor | None = None,
        *,
        k: int | None = None,
        p: int | None = None,
        inverse: bool = False,
    ) -> None:
        super().__init__()
        k, power_of_two_part = resolve_hadamard_dim(hidden_size, k=k, p=p)

        gemm_order = select_gemm_order(power_of_two_part)
        if (sign is None) != (hk is None):
            raise ValueError("sign and hk must either both be provided or both be None")

        if sign is None:
            # These parameters are populated by the checkpoint loader. Keeping
            # them as Parameters makes standard vLLM weight loading work without
            # a model-specific loader in the common case.
            sign = torch.empty(hidden_size, dtype=torch.int8)
            hk = torch.empty((k, k), dtype=torch.float32)
        else:
            if sign.shape != (hidden_size,) or sign.dtype != torch.int8:
                raise ValueError("sign must be a [hidden_size] int8 tensor")
            if hk is None or hk.shape != (k, k):
                raise ValueError("hk must be a [K, K] tensor")

        sign_param = sign.detach().to(dtype=torch.int8).contiguous()
        # The public constructor accepts the exporter-facing orientation. The
        # kernel keeps the transposed orientation used by the original module.
        hk_param = hk.detach().T.to(dtype=torch.float32).contiguous()
        self.register_parameter(
            "sign", nn.Parameter(sign_param, requires_grad=False)
        )
        self.register_parameter("hk", nn.Parameter(hk_param, requires_grad=False))
        self.register_buffer(
            "h_gemm",
            make_hadamard_gemm_matrix(
                gemm_order,
                device=sign_param.device,
                dtype=torch.bfloat16,
            ),
            persistent=False,
        )

        self.hidden_size = hidden_size
        self.k = k
        self.p = power_of_two_part
        self.inverse = inverse

    def forward(
        self,
        x: torch.Tensor,
        inverse: bool | None = None,
    ) -> torch.Tensor:
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected hidden_size={self.hidden_size}, got {x.shape[-1]}"
            )
        if inverse is None:
            inverse = self.inverse
        sign = self.sign.to(device=x.device)
        hk = self.hk.to(device=x.device, dtype=torch.float32)
        h_gemm = self.h_gemm.to(device=x.device, dtype=x.dtype)
        return hadamard(
            x,
            sign,
            hk,
            h_gemm,
            inverse=inverse,
            k=self.k,
            p=self.p,
        )

    @classmethod
    def from_rotation_config(
        cls,
        config: Mapping[str, Any],
        sign: torch.Tensor | None = None,
        hk: torch.Tensor | None = None,
    ) -> "CompositeHadamardRotation":
        return cls(
            hidden_size=int(config["hidden_size"]),
            sign=sign,
            hk=None if hk is None else hk.T,
            k=int(config["K"]),
            p=None if config.get("P") is None else int(config["P"]),
            inverse=bool(config.get("inverse", True)),
        )

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, K={self.k}, P={self.p}, "
            f"inverse={self.inverse}"
        )


__all__ = [
    "CompositeHadamardRotation",
    "build_rotation",
    "get_online_rotation_module_config",
]
