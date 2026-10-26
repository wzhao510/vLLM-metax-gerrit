# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA Engram DP sharding, shared host storage, and asynchronous prefetch."""

import mmap
import tempfile
import weakref
from contextlib import ExitStack
from typing import Any, ClassVar

import numpy as np
import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_dp_group,
    get_engram_dp_group,
    get_engram_dp_size,
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_gather,
)
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v41.common.engram import (
    DEAD_ID,
    EngramLayout,
    _engram_head_shard_weight_loader,
    _engram_lookup_kernel,
    _engram_select_rows,
)
from vllm.models.deepseek_v41.common.engram import (
    Engram as BaseEngram,
)
from vllm.models.deepseek_v41.common.engram import (
    ParallelEngramEmbedding as BaseParallelEngramEmbedding,
)
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
from vllm_metax.model_executor.layers.online_rotation import (
    build_rotation,
    get_online_rotation_module_config,
)
from vllm_metax.model_executor.kernels.online_rotate import fused_engram_gate
import vllm_metax.envs as mx_envs

logger = init_logger(__name__)


def engram_head_shard_rank() -> int:
    """This rank's slot among the hash-head shards of one engram table.

    TP-major, so the shards a DP gather brings in are contiguous heads and
    the following TP gather completes the head order.
    """
    dp_group = get_engram_dp_group()
    dp_size = dp_group.world_size if dp_group is not None else 1
    dp_rank = dp_group.rank_in_group if dp_group is not None else 0
    return get_tensor_model_parallel_rank() * dp_size + dp_rank


def engram_gathered_num_tokens() -> int:
    """Per-replica token slot for the node-local Engram DP group."""
    dp_metadata = get_forward_context().dp_metadata
    if dp_metadata is None:
        raise RuntimeError("a DP-shared engram table needs DP token metadata")
    group = get_engram_dp_group()
    assert group is not None
    # Engram groups are contiguous slices of the full DP group.
    start = get_dp_group().rank_in_group - group.rank_in_group
    return int(
        dp_metadata.num_tokens_across_dp_cpu[start : start + group.world_size].max()
    )


def gather_engram_hashes(
    hash_ids: torch.Tensor, *, dp_shared_memory: bool = False
) -> torch.Tensor:
    """Collect the n-gram ids of every DP replica sharing one table.

    Replicas are padded to a common token slot, so the gathered shape is
    static under CUDA graph capture (where DP already pads alike).
    """
    dp_group = get_engram_dp_group()
    if dp_group is None or dp_shared_memory:
        return hash_ids
    slot = engram_gathered_num_tokens()
    if hash_ids.shape[0] > slot:
        raise ValueError("Engram token count exceeds the DP token slot")
    if hash_ids.shape[0] < slot:
        pad = hash_ids.new_full(
            (slot - hash_ids.shape[0], *hash_ids.shape[1:]), DEAD_ID
        )
        hash_ids = torch.cat((hash_ids, pad))
    return dp_group.all_gather(hash_ids, dim=0)


class DPSharedEngramStorage:
    """Registered host weights shared by a node-local DP group with one writer.

    The table is one physical allocation holding the quantized rows followed by
    their per-block scales. Both element types are supplied by the embedding
    class, so the FP8 (E4M3 + UE8M0 bytes) and INT8 (INT8 + FP32) layouts share
    this code.
    """

    def __init__(
        self,
        num_rows: int,
        dim: int,
        block_size: int,
        group: GroupCoordinator,
        weight_dtype: torch.dtype = torch.float8_e4m3fn,
        scale_dtype: torch.dtype = torch.uint8,
        scale_param_name: str = "weight_scale_inv",
    ) -> None:
        self.group = group
        self.scale_param_name = scale_param_name
        weight_bytes = num_rows * dim * weight_dtype.itemsize
        scale_bytes = num_rows * (dim // block_size) * scale_dtype.itemsize
        storage = self._allocate(weight_bytes + scale_bytes)
        self.weight = storage[:weight_bytes].view(weight_dtype).view(-1, dim)
        setattr(
            self,
            scale_param_name,
            storage[weight_bytes:].view(scale_dtype).view(-1, dim // block_size),
        )
        self._views: tuple[torch.Tensor, torch.Tensor] | None = None

    def _allocate(self, num_bytes: int) -> torch.Tensor:
        """Map and register one physical allocation across a node-local DP group."""
        from vllm.distributed.device_communicators.shm_broadcast import (
            check_shm_free_space,
        )

        group = self.group
        with ExitStack() as stack:
            path, error = None, None
            if group.rank_in_group == 0:
                try:
                    check_shm_free_space(num_bytes)
                    backing_file = stack.enter_context(
                        tempfile.NamedTemporaryFile(
                            prefix="vllm_engram_", dir="/dev/shm"
                        )
                    )
                    backing_file.truncate(num_bytes)
                    path = backing_file.name
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            path, error = group.broadcast_object((path, error))
            if error is not None:
                raise RuntimeError(
                    "Engram shared-memory creation failed on EDP rank 0: " + error
                )

            mapping = owner = tensor = finalizer = None
            stage = "open"
            try:
                try:
                    with open(path, "r+b") as file:
                        stage = "mmap"
                        mapping = mmap.mmap(
                            file.fileno(), num_bytes, flags=mmap.MAP_SHARED
                        )
                    stage = "cudaHostRegister"
                    owner = np.frombuffer(mapping, dtype=np.uint8)
                    pointer = owner.ctypes.data
                    tensor = torch.from_numpy(owner)
                    result = torch.cuda.cudart().cudaHostRegister(pointer, num_bytes, 0)
                    if result.value != 0:
                        raise RuntimeError(f"cudaHostRegister failed: {result}")
                    finalizer = weakref.finalize(
                        owner, self._unregister, mapping, pointer
                    )
                    finalizer.atexit = False  # type: ignore[misc]
                    # The UVA helper otherwise allocates a private pinned copy.
                    if not tensor.is_pinned():
                        raise RuntimeError(
                            "CUDA did not recognize the shared Engram registration"
                        )
                except Exception as exc:
                    error = f"{stage}: {type(exc).__name__}: {exc}"

                errors: list[str | None] = [None] * group.world_size
                # Also fences peer mappings before the leader unlinks the file.
                torch.distributed.all_gather_object(
                    errors, error, group=group.cpu_group
                )
                failures = "; ".join(
                    f"EDP rank {rank}: {error}"
                    for rank, error in enumerate(errors)
                    if error is not None
                )
                if failures:
                    raise RuntimeError(
                        "Engram shared-memory initialization failed: " + failures
                    )
                assert tensor is not None
                return tensor
            except Exception:
                if finalizer is not None:
                    finalizer()
                tensor = owner = None
                if mapping is not None:
                    mapping.close()
                raise

    @staticmethod
    def _unregister(mapping: mmap.mmap, pointer: int) -> None:
        # Torch storage retains the numpy owner, including through cached UVA views.
        # Keep its mmap alive until CUDA has released the registration.
        result = torch.cuda.cudart().cudaHostUnregister(pointer)
        if result.value != 0:
            logger.warning("Engram cudaHostUnregister failed: %s", result)

    def load_weight(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        if self.group.rank_in_group == 0:
            _engram_head_shard_weight_loader(param, loaded_weight)
        # Read order may differ across ranks. Equal load counts ensure all shared
        # weights are ready after the last weight-loader call returns.
        torch.distributed.barrier(group=self.group.cpu_group)

    def get_views(
        self, weight: torch.Tensor, scales: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (weight.data_ptr(), scales.data_ptr()) != (
            self.weight.data_ptr(),
            getattr(self, self.scale_param_name).data_ptr(),
        ):
            raise RuntimeError("Shared Engram parameter storage must not be replaced")
        if self._views is None:
            self._views = (
                get_accelerator_view_from_cpu_tensor(self.weight),
                get_accelerator_view_from_cpu_tensor(
                    getattr(self, self.scale_param_name)
                ),
            )
        return self._views


@triton.jit(
    do_not_specialize=[
        "vocab_start",
        "vocab_end",
        "num_rows",
        "ids_stride_t",
        "ids_stride_h",
        "GRID",
    ]
)
def _engram_lookup_kernel_int8(
    weight,
    scales,
    ids,
    out,
    vocab_start,
    vocab_end,
    num_rows,
    ids_stride_t,
    ids_stride_h,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GRID,
):
    """Gather INT8 rows, apply their FP32 block scales, write bf16.

    Same shape/grid contract as the FP8 lookup; only the dequantization differs
    (INT8 values times FP32 block scales instead of E4M3 times decoded UE8M0
    exponents). Only this rank's heads are read; padded heads write zeros for
    the all-gather. `weight`/`scales` may address pinned host memory through UVA.
    """
    cols = tl.arange(0, DIM)
    scale_cols = cols // QUANT_BLOCK
    for base in tl.range(tl.program_id(0) * BLOCK_R, num_rows, GRID * BLOCK_R):
        rows = base + tl.arange(0, BLOCK_R)
        valid = rows < num_rows
        head = HEAD_START + rows % LOCAL_HEADS
        token = (rows // LOCAL_HEADS).to(tl.int64)
        index = tl.load(
            ids + token * ids_stride_t + head * ids_stride_h,
            mask=valid & (head < TOTAL_HEADS),
            other=-1,
        ).to(tl.int64)
        owned = valid & (head < TOTAL_HEADS)
        owned &= (index >= vocab_start) & (index < vocab_end)
        local = tl.where(owned, index - vocab_start, 0)
        values = tl.load(
            weight + local[:, None] * DIM + cols[None, :],
            mask=owned[:, None],
            other=0,
        )
        scale = tl.load(
            scales + local[:, None] * (DIM // QUANT_BLOCK) + scale_cols[None, :],
            mask=owned[:, None],
            other=0.0,
        )
        tl.store(
            out + rows[:, None] * DIM + cols[None, :],
            (values.to(tl.float32) * scale).to(tl.bfloat16),
            mask=valid[:, None],
        )


class ParallelEngramEmbedding(BaseParallelEngramEmbedding):
    """Extend TP lookup with DP head sharding or shared, CPU-offloaded TP slices.

    The table layout is described by the class attributes below so the INT8
    variant ([`MacaInt8EngramEmbedding`]) can reuse every part of the lookup,
    the host-shared storage, and the offload plumbing.
    """

    _shared_memory: DPSharedEngramStorage | None = None

    #: Lookup-table rows: FP8 E4M3 for FP8 checkpoints, INT8 for W4A8 ones.
    weight_dtype: ClassVar[torch.dtype] = torch.float8_e4m3fn
    #: Per-block scales: UE8M0 exponent bytes for FP8 (power-of-two scales),
    #: FP32 for INT8 (the checkpoints store them in the model's bf16 dtype;
    #: they are held as FP32 so the lookup dequantizes in one dtype and either
    #: checkpoint would still load, a bf16 to FP32 upcast being exact).
    scale_dtype: ClassVar[torch.dtype] = torch.uint8
    #: Parameter name the checkpoint ships the block scales under. INT8 tables
    #: use the same ``weight_scale`` spelling as every other quantized weight,
    #: FP8 tables keep ``weight_scale_inv`` (mirroring ``.scale``).
    scale_param_name: ClassVar[str] = "weight_scale_inv"
    #: gather-and-dequantize kernel over one rank's rows.
    lookup_kernel: ClassVar[Any] = staticmethod(_engram_lookup_kernel)
    #: Constant fill values for the (huge) dummy tables under ``--load-format
    #: dummy``; avoids randomizing host memory that is never read meaningfully.
    dummy_weight_value: ClassVar[float] = 1.0
    #: The ue8m0 encoding of scale 1.0 is exponent byte 127.
    dummy_scale_value: ClassVar[float] = 127

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        head_sizes: tuple[int, ...],
        block_size: int = 32,
        cpu_offload: bool = False,
        dp_shared_memory: bool = False,
    ) -> None:
        self.cpu_offload = cpu_offload
        self.dp_shared_memory = dp_shared_memory
        self.dp_size = get_engram_dp_size()
        if dp_shared_memory:
            if not cpu_offload:
                raise ValueError("dp_shared_memory requires cpu_offload=True")
            if self.dp_size <= 1:
                raise ValueError(
                    "dp_shared_memory requires a node-local Engram DP "
                    f"group with size > 1; effective Engram DP size is {self.dp_size}. "
                    "Check that the node layout and rank placement allow complete "
                    "DP replicas to be co-located."
                )
            self.dp_size = 1
        if cpu_offload and not is_uva_available():
            raise RuntimeError("Engram CPU offload requires UVA support")
        self._views: tuple[torch.Tensor, torch.Tensor] | None = None
        self._view_src: tuple[int, int] | None = None
        super().__init__(num_embeddings, dim, head_sizes, block_size)
        self._rename_scale_param()
        if cpu_offload:
            # Constant dummy values avoid randomizing huge CPU lookup tables.
            set_weight_attrs(
                self.weight, {"dummy_weight_value": self.dummy_weight_value}
            )
            set_weight_attrs(
                self._scale_param(), {"dummy_weight_value": self.dummy_scale_value}
            )
            logger.info(
                "Engram table offloaded to pinned host memory: %d rows x %d, "
                "%.2f GiB %s",
                self.part_num_embeddings,
                dim,
                self.part_num_embeddings * self._bytes_per_row() / 1024**3,
                "shared across DP replicas" if dp_shared_memory else "per rank",
            )

    def _bytes_per_row(self) -> int:
        """Host bytes one table row occupies: quantized values plus its blocks."""
        values = self.dim * self.weight_dtype.itemsize
        scales = (self.dim // self.block_size) * self.scale_dtype.itemsize
        return values + scales

    def _scale_param(self) -> torch.nn.Parameter:
        """This rank's per-block scale parameter."""
        return getattr(self, self.scale_param_name)

    def _rename_scale_param(self) -> None:
        """Re-register the block scales under the checkpoint's parameter name.

        ``BaseParallelEngramEmbedding`` always registers ``weight_scale_inv``;
        the INT8 tables name it ``weight_scale`` like every other quantized
        weight, so the mapper's ``engram.embed.weight_scale`` rule lands here.
        """
        if self.scale_param_name == "weight_scale_inv":
            return
        scale = self.weight_scale_inv
        del self._parameters["weight_scale_inv"]
        self.__dict__.pop("weight_scale_inv", None)
        setattr(self, self.scale_param_name, scale)

    def _get_shard_info(self) -> tuple[int, int]:
        if self.dp_size == 1:
            return super()._get_shard_info()
        return self.tp_size * self.dp_size, engram_head_shard_rank()

    def _allocate_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dp_shared_memory:
            group = get_engram_dp_group()
            assert group is not None
            storage = DPSharedEngramStorage(
                self.part_num_embeddings,
                self.dim,
                self.block_size,
                group,
                weight_dtype=self.weight_dtype,
                scale_dtype=self.scale_dtype,
                scale_param_name=self.scale_param_name,
            )
            self._shared_memory = storage
            self._weight_loader = storage.load_weight
            return storage.weight, getattr(storage, self.scale_param_name)
        # Model initialization may be inside a CUDA device context, so the
        # offloaded tables are pinned CPU allocations; resident ones follow the
        # current device like any other parameter.
        pin = self.cpu_offload
        device = "cpu" if pin else None
        return (
            torch.empty(
                self.part_num_embeddings,
                self.dim,
                dtype=self.weight_dtype,
                device=device,
                pin_memory=pin,
            ),
            torch.empty(
                self.part_num_embeddings,
                self.dim // self.block_size,
                dtype=self.scale_dtype,
                device=device,
                pin_memory=pin,
            ),
        )

    def _storage(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._shared_memory is not None:
            return self._shared_memory.get_views(self.weight, self._scale_param())
        if not self.cpu_offload:
            return super()._storage()
        src = (self.weight.data_ptr(), self._scale_param().data_ptr())
        if self._view_src != src:
            self._views = (
                get_accelerator_view_from_cpu_tensor(self.weight.data),
                get_accelerator_view_from_cpu_tensor(self._scale_param().data),
            )
            self._view_src = src
        assert self._views is not None
        return self._views

    def lookup(
        self, indices: torch.Tensor, out: torch.Tensor, background: bool = False
    ) -> None:
        """Look up local heads of [T, heads] into [T, local_heads, dim] bf16.

        `background` limits the grid to leave SMs for concurrent work.
        """
        rows = indices.shape[0] * self.part_n_hash_cols
        if not rows:
            return
        weight, scales = self._storage()
        # The table dwarfs TLB reach, so a persistent grid near the SM count
        # beats one program per row; halve it to leave SMs for the main stream.
        tiles = triton.cdiv(rows, 16)
        grid = min(tiles, self._num_sms // 2 if background else self._num_sms)
        self.lookup_kernel[(grid,)](
            weight,
            scales,
            indices,
            out,
            self.vocab_start_idx,
            self.vocab_end_idx,
            rows,
            indices.stride(0),
            indices.stride(1),
            HEAD_START=self.head_start,
            LOCAL_HEADS=self.part_n_hash_cols,
            TOTAL_HEADS=self.n_hash_cols,
            DIM=self.dim,
            QUANT_BLOCK=self.block_size,
            BLOCK_R=16,
            GRID=grid,
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if self.dp_size == 1:
            return super().forward(indices)
        num_tokens = indices.shape[0]
        indices = gather_engram_hashes(indices)
        out = torch.empty(
            (indices.shape[0], self.part_n_hash_cols, self.dim),
            dtype=torch.bfloat16,
            device=indices.device,
        )
        self.lookup(indices, out)
        out = _gather_engram_rows(out, num_tokens)
        if self.tp_size > 1:
            out = tensor_model_parallel_all_gather(out, dim=1)
        return out[:, : self.n_hash_cols]


class MacaInt8EngramEmbedding(ParallelEngramEmbedding):
    """Engram tables for the W4A8 checkpoints: INT8 rows, one scale per 32 columns.

    The W4A8 export ships ``engram.embed.weight`` (INT8) plus
    ``engram.embed.weight_scale``: like every other *group-wise* int scale in
    these checkpoints the blocks are of the model's own dtype, so they arrive as
    bf16 and are loaded into the FP32 table allocated here. The FP8 checkpoints
    keep the E4M3 + UE8M0 block layout of [`ParallelEngramEmbedding`] instead;
    ``VLLM_METAX_SUPPORTS_FP8`` picks between the two.
    """

    weight_dtype: ClassVar[torch.dtype] = torch.int8
    scale_dtype: ClassVar[torch.dtype] = torch.float32
    scale_param_name: ClassVar[str] = "weight_scale"
    lookup_kernel: ClassVar[Any] = staticmethod(_engram_lookup_kernel_int8)
    dummy_weight_value: ClassVar[float] = 1.0
    dummy_scale_value: ClassVar[float] = 1.0


def _gather_engram_rows(staged: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Exchange DP tokens for heads, retaining only this replica's tokens."""
    dp_group = get_engram_dp_group()
    assert dp_group is not None
    slot, remainder = divmod(staged.shape[0], dp_group.world_size)
    assert remainder == 0 and 0 <= num_tokens <= slot
    gathered = dp_group.all_gather(staged, dim=0)
    local_heads, dim = staged.shape[1:]
    rows = staged.new_empty((num_tokens, dp_group.world_size * local_heads, dim))
    _engram_select_rows(
        gathered,
        rows,
        staged.shape[0],
        dp_group.rank_in_group * slot,
        local_heads * dim,
    )
    return rows


class Engram(BaseEngram):
    """NVIDIA Engram with asynchronous offload and node-local DP lookup."""

    _prefetch_stream: torch.cuda.Stream | None = None

    def __init__(
        self, 
        config, 
        quant_config, 
        layout, 
        layer_hash_index, 
        use_sequence_parallel, 
        prefix
    ):

        super().__init__(
            config, quant_config, layout, layer_hash_index, use_sequence_parallel, prefix
        )
        
        layer_id = layout.layer_ids[layer_hash_index]
        rotation_name = f"model.layers.{layer_id}.engram.global_rotation"
        rotation_config = get_online_rotation_module_config(config, rotation_name)
        
        if rotation_config is not None:
            self.global_rotation = build_rotation(rotation_config)
        elif not hasattr(self, "global_rotation"):
            self.global_rotation = None

    def _embedding_cls(self) -> type[ParallelEngramEmbedding]:
        """Block format of the checkpoint's n-gram tables.

        ``VLLM_METAX_SUPPORTS_FP8`` selects the FP8 checkpoints (E4M3 rows with
        UE8M0 block scales); the W4A8 checkpoints ship INT8 rows with one
        block scale per 32 columns, so they take the INT8 lookup.
        """
        if mx_envs.VLLM_METAX_SUPPORTS_FP8:
            return ParallelEngramEmbedding
        return MacaInt8EngramEmbedding

    def _create_embedding(
        self, layout: EngramLayout, layer_hash_index: int
    ) -> ParallelEngramEmbedding:
        engram_config = get_current_vllm_config().engram_config
        assert engram_config is not None
        embedding_cls = self._embedding_cls()
        embedding = embedding_cls(
            layout.num_embeddings[layer_hash_index],
            layout.head_dim,
            tuple(size for order in layout.primes[layer_hash_index] for size in order),
            cpu_offload=engram_config.cpu_offload,
            dp_shared_memory=engram_config.dp_shared_memory,
        )
        logger.info_once(
            "Engram n-gram tables: %s %s rows with one %s scale per %d columns",
            embedding_cls.__name__,
            embedding.weight_dtype,
            embedding.scale_dtype,
            embedding.block_size,
        )
        return embedding

    def _init_staging(self, max_tokens: int, head_dim: int) -> None:
        super()._init_staging(max_tokens * self.embed_tokens.dp_size, head_dim)
        if self.embed_tokens.cpu_offload:
            self._prefetch_stream = torch.cuda.Stream(device=self.staged_rows.device)

    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        """Prefetch local shared rows or the DP group's gathered hash IDs."""
        if self._prefetch_stream is None:
            return super().prepare_embeddings(hash_ids)
        rows = self.staged_rows[: hash_ids.shape[0]]
        assert rows.shape[0] == hash_ids.shape[0], "engram staging buffer too small"
        self._start_prefetch(hash_ids, rows, self._prefetch_stream)

    @eager_break_during_capture
    def _start_prefetch(
        self, hash_ids: torch.Tensor, rows: torch.Tensor, stream: torch.cuda.Stream
    ) -> None:
        # Eager boundaries let the lookup span piecewise graph segments.
        stream.wait_stream(torch.cuda.current_stream())
        # Keep temporary hash storage alive until lookup finishes reading it.
        hash_ids.record_stream(stream)
        with torch.cuda.stream(stream):
            self.embed_tokens.lookup(hash_ids, rows, background=True)

    @eager_break_during_capture
    def _finish_prefetch(self, stream: torch.cuda.Stream) -> None:
        torch.cuda.current_stream().wait_stream(stream)

    def _ready_rows(self, num_tokens: int) -> torch.Tensor:
        if self._prefetch_stream is not None:
            self._finish_prefetch(self._prefetch_stream)
        if self.embed_tokens.dp_size > 1:
            slot = engram_gathered_num_tokens()
            staged = self.staged_rows[: slot * self.embed_tokens.dp_size]
            return _gather_engram_rows(staged, num_tokens)
        return super()._ready_rows(num_tokens)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        hash_ids: torch.Tensor, 
        token_mask: torch.Tensor | None = None, 
        gate_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        

        if getattr(self, "global_rotation", None) is None and gate_input is None:
            return super().forward(hidden_states, hash_ids, token_mask)

        emb = self.embed_tokens(hash_ids)
        wkv_out = self.wkv(emb.flatten(-2))
        kv = wkv_out[0] if isinstance(wkv_out, tuple) else wkv_out
        
        if kv.dtype not in (torch.bfloat16, torch.float32):
            kv = kv.to(torch.bfloat16)
        kv = kv.contiguous()

        if gate_input is not None:
            gate_input = gate_input.contiguous()

        out = fused_engram_gate(
            hidden_states.contiguous(),
            kv,
            self.q_weight,
            self.k_weight,
            self.eps,
            self.clamp_value,
            gate_input=gate_input,
        )
        
        if token_mask is not None:
            out = torch.where(token_mask.view(-1, 1, 1), out, hidden_states)
            
        return out
