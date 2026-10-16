# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm_metax.patch.performance import gumbel_sample as metax_gumbel


@pytest.mark.parametrize(
    ("active_indices", "expected_all_greedy"),
    [([1, 3], True), ([1, 2], False), ([0, 2], False)],
)
def test_all_greedy_uses_active_request_slots(
    active_indices: list[int], expected_all_greedy: bool
) -> None:
    temperatures = np.array([0.6, 0.0, 0.6, 0.0], dtype=np.float32)
    states = SimpleNamespace(
        temperature=SimpleNamespace(np=temperatures, gpu=torch.from_numpy(temperatures)),
        seeds=SimpleNamespace(gpu=torch.zeros(4, dtype=torch.int64)),
        get_top_k_top_p=Mock(return_value=(None, None)),
    )
    logits = torch.zeros(len(active_indices), 8)
    sampled = torch.zeros(len(active_indices), dtype=torch.int64)
    sampler = SimpleNamespace(
        apply_sampling_params=Mock(return_value=logits),
        sampling_states=states,
        use_flashinfer=False,
        use_fp64_gumbel=False,
    )

    with (
        patch.object(metax_gumbel, "apply_top_k_top_p", side_effect=lambda x, *_: x),
        patch.object(metax_gumbel, "mcop_gumbel_sample", return_value=sampled) as op,
    ):
        result, processed = Sampler.sample(
            sampler,
            logits,
            torch.tensor(active_indices),
            torch.tensor(active_indices),
            np.array(active_indices),
            torch.zeros(len(active_indices), dtype=torch.int64),
            torch.zeros(len(active_indices), dtype=torch.int64),
            torch.zeros(len(active_indices), dtype=torch.int64),
        )

    assert result is sampled
    assert processed is logits
    assert op.call_args.kwargs["all_greedy"] is expected_all_greedy
    assert op.call_args.kwargs["is_drafting"] is False
