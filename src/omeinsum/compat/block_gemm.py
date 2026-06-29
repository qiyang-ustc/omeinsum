from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import torch

from .scheduler import BlockGemmPolicy, BlockGemmStats, estimate_block_gemm_stats


def yastn_transpose_dot_sum(
    Adata: torch.Tensor,
    Bdata: torch.Tensor,
    meta_dot: Sequence,
    Areshape: Sequence,
    Breshape: Sequence,
    Aorder: Sequence[int],
    Border: Sequence[int],
    Dsize: int,
    *,
    policy: BlockGemmPolicy | None = None,
) -> torch.Tensor:
    """Execute YASTN ``transpose_dot_sum`` metadata with PyTorch block GEMMs."""

    policy = policy or BlockGemmPolicy()
    if policy.cpu_enabled:
        raise NotImplementedError("CPU/GPU mixed placement is reserved for a later version.")
    if policy.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if policy.min_group <= 0:
        raise ValueError("min_group must be positive.")
    if policy.executor not in {"stack_bmm", "custom_no_stack"}:
        raise ValueError("block GEMM executor must be 'stack_bmm' or 'custom_no_stack'.")
    if Adata.device != Bdata.device:
        raise ValueError("Adata and Bdata must live on the same device in this implementation.")

    dtype = torch.promote_types(Adata.dtype, Bdata.dtype)
    if dtype != Adata.dtype:
        Adata = Adata.to(dtype=dtype)
    if dtype != Bdata.dtype:
        Bdata = Bdata.to(dtype=dtype)

    if policy.executor == "custom_no_stack":
        return _YastnTransposeDotSumNoStack.apply(
            Adata,
            Bdata,
            tuple(meta_dot),
            tuple(Areshape),
            tuple(Breshape),
            tuple(int(x) for x in Aorder),
            tuple(int(x) for x in Border),
            int(Dsize),
            policy.stats,
        )

    At = tuple(_block_view(Adata, item, Aorder) for item in Areshape)
    Bt = tuple(_block_view(Bdata, item, Border) for item in Breshape)
    Cdata = torch.zeros(int(Dsize), dtype=dtype, device=Adata.device)
    accumulators: list[torch.Tensor | None] = [None] * len(meta_dot)
    groups: dict[tuple[int, int, int], list[tuple[int, int, int]]] = defaultdict(list)
    stats = policy.stats
    _add_stat(stats, "yastn_compat_block_gemm_calls", 1)
    _add_stat(stats, "yastn_compat_block_gemm_a_blocks", len(Areshape))
    _add_stat(stats, "yastn_compat_block_gemm_b_blocks", len(Breshape))
    _add_stat(stats, "yastn_compat_block_gemm_c_blocks", len(meta_dot))

    for out_idx, (_, _, list_tab) in enumerate(meta_dot):
        fanin = len(list_tab)
        _add_stat(stats, "yastn_compat_block_gemm_output_blocks", 1)
        _add_stat(stats, "yastn_compat_block_gemm_output_fanin_tasks", fanin)
        _max_stat(stats, "yastn_compat_block_gemm_max_output_fanin", fanin)
        _record_size_bucket(stats, "yastn_compat_block_gemm_output_fanin", fanin)
        for ta, tb in list_tab:
            ta, tb = int(ta), int(tb)
            a_block = At[ta]
            b_block = Bt[tb]
            if a_block.shape[1] != b_block.shape[0]:
                raise ValueError(f"Block GEMM inner dimensions differ: {a_block.shape[1]} != {b_block.shape[0]}")
            groups[(int(a_block.shape[0]), int(a_block.shape[1]), int(b_block.shape[1]))].append((out_idx, ta, tb))

    _add_stat(stats, "yastn_compat_block_gemm_contributions", sum(len(tasks) for tasks in groups.values()))
    _add_stat(stats, "yastn_compat_block_gemm_unique_shapes", len(groups))
    for tasks in groups.values():
        _add_stat(stats, "yastn_compat_block_gemm_shape_groups", 1)
        _add_stat(stats, "yastn_compat_block_gemm_shape_group_tasks", len(tasks))
        _max_stat(stats, "yastn_compat_block_gemm_max_shape_group_tasks", len(tasks))
        _record_size_bucket(stats, "yastn_compat_block_gemm_shape_group", len(tasks))
        if len(tasks) >= policy.min_group:
            for start in range(0, len(tasks), policy.batch_size):
                chunk = tasks[start : start + policy.batch_size]
                _add_stat(stats, "yastn_compat_block_gemm_batched_chunks", 1)
                _add_stat(stats, "yastn_compat_block_gemm_batched_tasks", len(chunk))
                _record_size_bucket(stats, "yastn_compat_block_gemm_batched_chunk", len(chunk))
                Ablocks = torch.stack([At[ta] for _, ta, _ in chunk], dim=0)
                Bblocks = torch.stack([Bt[tb] for _, _, tb in chunk], dim=0)
                for value, (out_idx, _, _) in zip(torch.bmm(Ablocks, Bblocks), chunk, strict=True):
                    accumulators[out_idx] = value if accumulators[out_idx] is None else accumulators[out_idx] + value
        else:
            _add_stat(stats, "yastn_compat_block_gemm_fallback_groups", 1)
            _add_stat(stats, "yastn_compat_block_gemm_fallback_tasks", len(tasks))
            for out_idx, ta, tb in tasks:
                value = At[ta] @ Bt[tb]
                accumulators[out_idx] = value if accumulators[out_idx] is None else accumulators[out_idx] + value

    for idx, acc in enumerate(accumulators):
        if acc is None:
            continue
        sl, dslc, _ = meta_dot[idx]
        Cdata[slice(*sl)] = acc.reshape(tuple(int(x) for x in dslc)).reshape(-1)

    return Cdata


class _YastnTransposeDotSumNoStack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Adata, Bdata, meta_dot, Areshape, Breshape, Aorder, Border, Dsize, stats):
        ctx.meta_dot = meta_dot
        ctx.Areshape = Areshape
        ctx.Breshape = Breshape
        ctx.Aorder = Aorder
        ctx.Border = Border
        ctx.Dsize = int(Dsize)
        ctx.stats = stats
        ctx.save_for_backward(Adata, Bdata)
        Cdata = torch.zeros(ctx.Dsize, dtype=torch.promote_types(Adata.dtype, Bdata.dtype), device=Adata.device)
        At = tuple(_block_view(Adata, item, Aorder) for item in Areshape)
        Bt = tuple(_block_view(Bdata, item, Border) for item in Breshape)
        _add_stat(stats, "yastn_compat_block_gemm_custom_no_stack_calls", 1)
        _add_stat(stats, "yastn_compat_block_gemm_custom_no_stack_c_blocks", len(meta_dot))
        for out_idx, (sl, dslc, list_tab) in enumerate(meta_dot):
            acc = None
            for ta, tb in list_tab:
                ta, tb = int(ta), int(tb)
                value = At[ta] @ Bt[tb]
                acc = value if acc is None else acc + value
            if acc is None:
                continue
            Cdata[slice(*sl)] = acc.reshape(tuple(int(x) for x in dslc)).reshape(-1)
            _add_stat(stats, "yastn_compat_block_gemm_custom_no_stack_output_blocks", 1)
            _add_stat(stats, "yastn_compat_block_gemm_custom_no_stack_tasks", len(list_tab))
        return Cdata

    @staticmethod
    def backward(ctx, grad_Cdata):
        Adata, Bdata = ctx.saved_tensors
        grad_Adata = torch.zeros_like(Adata) if ctx.needs_input_grad[0] else None
        grad_Bdata = torch.zeros_like(Bdata) if ctx.needs_input_grad[1] else None
        At = tuple(_block_view(Adata, item, ctx.Aorder) for item in ctx.Areshape)
        Bt = tuple(_block_view(Bdata, item, ctx.Border) for item in ctx.Breshape)
        _add_stat(ctx.stats, "yastn_compat_block_gemm_custom_no_stack_backward_calls", 1)
        for sl, dslc, list_tab in ctx.meta_dot:
            if not list_tab:
                continue
            first_a, first_b = (int(x) for x in list_tab[0])
            grad_block = grad_Cdata[slice(*sl)].reshape(tuple(int(x) for x in dslc))
            grad_mat = grad_block.reshape(int(At[first_a].shape[0]), int(Bt[first_b].shape[1]))
            for ta, tb in list_tab:
                ta, tb = int(ta), int(tb)
                a_block = At[ta]
                b_block = Bt[tb]
                if grad_Adata is not None:
                    _add_block_grad_(grad_Adata, ctx.Areshape[ta], ctx.Aorder, grad_mat @ b_block.adjoint())
                if grad_Bdata is not None:
                    _add_block_grad_(grad_Bdata, ctx.Breshape[tb], ctx.Border, a_block.adjoint() @ grad_mat)
                _add_stat(ctx.stats, "yastn_compat_block_gemm_custom_no_stack_backward_tasks", 1)
        return grad_Adata, grad_Bdata, None, None, None, None, None, None, None


def yastn_block_gemm_stats(
    meta_dot: Sequence,
    Areshape: Sequence,
    Breshape: Sequence,
) -> BlockGemmStats:
    return estimate_block_gemm_stats(meta_dot, Areshape, Breshape)


def _block_view(
    data: torch.Tensor,
    reshape: tuple,
    order: Sequence[int],
) -> torch.Tensor:
    sl, dims, left, right = reshape
    block = data[slice(*sl)].view(tuple(int(x) for x in dims))
    return block.permute(tuple(order)).reshape(int(left), int(right))


def _add_block_grad_(
    data_grad: torch.Tensor,
    reshape: tuple,
    order: Sequence[int],
    grad_matrix: torch.Tensor,
) -> None:
    sl, dims, left, right = reshape
    order = tuple(int(x) for x in order)
    dims = tuple(int(x) for x in dims)
    permuted_dims = tuple(dims[axis] for axis in order)
    inverse_order = [0] * len(order)
    for pos, axis in enumerate(order):
        inverse_order[axis] = pos
    grad_block = grad_matrix.reshape(permuted_dims).permute(tuple(inverse_order)).reshape(-1)
    data_grad[slice(*sl)].add_(grad_block)


def _add_stat(stats: dict[str, int] | None, key: str, value: int) -> None:
    if stats is not None:
        stats[key] = int(stats.get(key, 0)) + int(value)


def _max_stat(stats: dict[str, int] | None, key: str, value: int) -> None:
    if stats is not None:
        stats[key] = max(int(stats.get(key, 0)), int(value))


def _record_size_bucket(stats: dict[str, int] | None, prefix: str, count: int) -> None:
    if count <= 1:
        bucket = "size_1"
    elif count <= 4:
        bucket = "size_2_4"
    elif count <= 8:
        bucket = "size_5_8"
    elif count <= 16:
        bucket = "size_9_16"
    else:
        bucket = "size_17_plus"
    _add_stat(stats, f"{prefix}_{bucket}", 1)
