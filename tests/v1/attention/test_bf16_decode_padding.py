# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for sparse BF16 decode metadata."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.plugins import load_general_plugins

load_general_plugins()

from vllm_metax.v1.attention.backends.mla import flashmla_sparse as mla


class BF16DecodePaddingTest(unittest.TestCase):
    def build(self, query_lens, decode_tokens, num_decodes=None, num_prefills=0):
        if num_decodes is None:
            num_decodes = len(query_lens)
        starts = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(0)
        seq_lens = torch.tensor(
            [100 + i if length else 0 for i, length in enumerate(query_lens)],
            dtype=torch.int32,
        )
        common = SimpleNamespace(
            query_start_loc_cpu=starts,
            seq_lens=seq_lens,
            num_actual_tokens=decode_tokens + sum(query_lens[num_decodes:]),
        )
        metadata = SimpleNamespace(
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=decode_tokens,
        )
        capacity = max(1, decode_tokens)
        builder = SimpleNamespace(
            device=torch.device("cpu"),
            dummy_block_table=torch.zeros(capacity, 1, dtype=torch.int32),
            max_model_len_tensor=torch.full((capacity,), 4096, dtype=torch.int32),
        )
        with patch.object(mla, "get_mla_metadata", return_value=(None, None)):
            build = (
                mla.FlashMLASparseBF16MetadataBuilder
                ._build_bf16_separate_prefill_decode
            )
            return build(builder, common, metadata)

    def check_decode(self, output, expected_seq_lens, query_len, tokens):
        self.assertEqual(output.num_decode_tokens, tokens)
        self.assertEqual(output.num_decodes * query_len, tokens)
        self.assertEqual(output.decode.decode_query_len, query_len)
        self.assertEqual(output.decode.seq_lens.tolist(), expected_seq_lens)
        self.assertEqual(output.decode.seq_lens.dtype, torch.int32)
        kernel = output.decode.kernel_metadata
        self.assertEqual(kernel.dummy_block_table.shape[0], output.num_decodes)
        self.assertEqual(kernel.cache_lens.shape[0], output.num_decodes)
        queries = torch.arange(tokens).reshape(output.num_decodes, query_len)
        self.assertEqual(queries.flatten().tolist(), list(range(tokens)))

    def test_single_token_with_padding(self):
        output = self.build([1] * 5 + [0] * 3, 8)
        self.check_decode(output, [100, 101, 102, 103, 104, 0, 0, 0], 1, 8)

    def test_single_token_without_padding(self):
        self.check_decode(self.build([1] * 5, 5), [100, 101, 102, 103, 104], 1, 5)

    def test_uniform_mtp_with_padding(self):
        self.check_decode(self.build([3, 3, 3, 0], 12), [100, 101, 102, 0], 3, 12)

    def test_uniform_mtp_without_padding(self):
        self.check_decode(self.build([3, 3], 6), [100, 101], 3, 6)

    def test_variable_decode_without_padding(self):
        self.check_decode(self.build([1, 2, 1], 4), [100, 101, 101, 102], 1, 4)

    def test_variable_decode_with_padding(self):
        output = self.build([1, 2, 1, 0], 8)
        self.check_decode(output, [100, 101, 101, 102, 0, 0, 0, 0], 1, 8)

    def test_mixed_prefill_decode(self):
        output = self.build([1, 2, 1, 257], 4, num_decodes=3, num_prefills=1)
        self.check_decode(output, [100, 101, 101, 102], 1, 4)
        self.assertEqual(output.num_prefills, 1)
        self.assertEqual(output.num_prefill_tokens, 257)

    def test_pure_prefill(self):
        output = self.build([256, 512], 0, num_decodes=0, num_prefills=2)
        self.assertEqual(output.num_decodes, 0)
        self.assertEqual(output.num_prefill_tokens, 768)
        self.assertIsNone(output.decode)

    def test_all_padding(self):
        self.check_decode(self.build([0, 0], 2), [0, 0], 1, 2)

    def test_invalid_token_count_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "query tokens"):
            self.build([2, 2], 3)

    def test_mixed_batch_padding_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "pure decode"):
            self.build([1, 257], 2, num_decodes=1, num_prefills=1)

    def test_misaligned_uniform_mtp_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "uniform query length"):
            self.build([3, 3, 0], 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
