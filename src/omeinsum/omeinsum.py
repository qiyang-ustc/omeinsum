from __future__ import annotations

from typing import Sequence

from collections import defaultdict

import opt_einsum
import torch
import torch.nn as nn

from .chunk_multi import run_chunked_einsum_multi_device
from .chunk_single import run_chunked_einsum
from .omeco_adapter import OMECO_AVAILABLE, OmecoOptimizer, einsum_to_omeco_format


class OMEinsum(nn.Module):
    def __init__(
        self,
        equation: str,
        block_dim: str,
        batch_size: int = 16,
        use_checkpoint: bool = False,
        use_reentrant: bool = False,
        force_block_dim_first: bool = False,
        log_path: bool = True,
        optimize: str | bool | None = "auto",
        path_optimizer: str = "auto",
        omeco_method: str = "treesa",
        omeco_optimize_tc: bool = True,
    ) -> None:
        """
        OMEinsum: Chunked einsum with automatic path optimization.

        Args:
            equation: Einsum equation with explicit output (e.g., "ij,jk->ik")
            block_dim: Dimension to chunk/block along
            batch_size: Chunk size for blocking
            use_checkpoint: Use gradient checkpointing
            use_reentrant: Use reentrant checkpointing
            force_block_dim_first: Force contracting blocked tensor first
            log_path: Print contraction path info
            optimize: opt_einsum optimization strategy (used when path_optimizer="opt_einsum")
            path_optimizer: "auto" (try omeco, fallback to opt_einsum), "omeco", or "opt_einsum"
            omeco_method: omeco method - "greedy" or "treesa"
            omeco_optimize_tc: If True, optimize only time complexity (recommended)
        """
        super().__init__()
        self.equation = equation
        self.block_dim = block_dim
        self.batch_size = batch_size
        self.use_checkpoint = use_checkpoint
        self.use_reentrant = use_reentrant
        self.force_block_dim_first = force_block_dim_first
        self.log_path = log_path
        self.optimize = optimize
        self.omeco_method = omeco_method
        self.omeco_optimize_tc = omeco_optimize_tc

        # Resolve path_optimizer: "auto" tries omeco first, falls back to opt_einsum
        if path_optimizer == "auto":
            if OMECO_AVAILABLE:
                self.path_optimizer = "omeco"
            else:
                self.path_optimizer = "opt_einsum"
        elif path_optimizer == "omeco" and not OMECO_AVAILABLE:
            raise ImportError(
                "omeco is not installed. Install it with: pip install omeco"
            )
        else:
            self.path_optimizer = path_optimizer

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        if "->" not in equation:
            raise ValueError("Only explicit einsum equations with '->' are supported.")
        input_part, output_subscript = equation.split("->")
        self.input_subscripts = [s.strip() for s in input_part.split(",")]
        self.output_subscript = output_subscript.strip()

        found = False
        for tidx, sub in enumerate(self.input_subscripts):
            if block_dim in sub:
                self.tidx = tidx
                self.didx = sub.index(block_dim)
                found = True
                break
        if not found:
            raise ValueError(f"block_dim '{block_dim}' not found in any input subscript.")
        if block_dim not in self.output_subscript:
            raise ValueError(f"block_dim '{block_dim}' must appear in output for slicing.")

        count = sum(sub.count(block_dim) for sub in self.input_subscripts)
        if count != 1:
            raise ValueError(
                f"block_dim '{block_dim}' is contracted (appears in multiple inputs), cannot block."
            )
        self.out_dim = self.output_subscript.index(block_dim)

        self._path = None
        self._path_shapes: tuple[tuple[int, ...], ...] | None = None
        self._path_logged = False
        self._path_optimizer = None

        # Set up path optimizer
        if self.path_optimizer == "omeco":
            self._path_optimizer = OmecoOptimizer(
                method=self.omeco_method,
                optimize_tc=self.omeco_optimize_tc,
            )
        elif self.force_block_dim_first:
            self._path_optimizer = _BlockFirstOptimizer(self.tidx, tail_optimize=self.optimize)

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        return self._forward(*tensors)

    def _validate_inputs(self, tensors: Sequence[torch.Tensor]) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("OMEinsum requires CUDA tensors and devices.")
        if len(tensors) != len(self.input_subscripts):
            raise ValueError(
                f"Expected {len(self.input_subscripts)} input tensors, got {len(tensors)}."
            )
        for t in tensors:
            if not isinstance(t, torch.Tensor):
                raise TypeError("OMEinsum inputs must be torch.Tensor instances.")
            if not t.is_cuda:
                raise RuntimeError("OMEinsum does not support CPU tensors.")

    def _get_path(self, tensors: Sequence[torch.Tensor]):
        shapes = tuple(tuple(t.shape) for t in tensors)
        if self._path is None or self._path_shapes != shapes:
            optimize = self._path_optimizer or self.optimize
            path, info = opt_einsum.contract_path(self.equation, *tensors, optimize=optimize)
            self._path = path
            self._path_shapes = shapes
            if self.log_path and not self._path_logged:
                if self.path_optimizer == "omeco":
                    optimizer_info = f"path_optimizer=omeco method={self.omeco_method} optimize_tc={self.omeco_optimize_tc}"
                else:
                    optimizer_info = f"path_optimizer=opt_einsum optimize={optimize}"
                print(
                    f"[OMEinsum] eq={self.equation} block_dim={self.block_dim} "
                    f"batch_size={self.batch_size} {optimizer_info}",
                    flush=True,
                )
                print(info, flush=True)
                self._path_logged = True
        return self._path

    def _forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(tensors)

        tensor_to_block = tensors[self.tidx]
        path = self._get_path(tensors)

        def einsum_fn(*args: torch.Tensor) -> torch.Tensor:
            return opt_einsum.contract(self.equation, *args, optimize=path)

        if tensor_to_block.size(self.didx) == 0:
            return einsum_fn(*tensors)

        chunks = torch.split(tensor_to_block, self.batch_size, dim=self.didx)

        if torch.cuda.device_count() <= 1:
            return run_chunked_einsum(
                einsum_fn,
                list(tensors),
                self.tidx,
                chunks,
                self.out_dim,
                self.use_checkpoint,
                self.use_reentrant,
            )

        return run_chunked_einsum_multi_device(
            einsum_fn,
            tensors,
            self.tidx,
            chunks,
            self.out_dim,
            self.use_checkpoint,
            self.use_reentrant,
        )


class _BlockFirstOptimizer(opt_einsum.paths.PathOptimizer):
    def __init__(self, tidx: int, tail_optimize: str | opt_einsum.paths.PathOptimizer | None = None) -> None:
        self.tidx = tidx
        self.tail_optimize = tail_optimize or "greedy"

    @staticmethod
    def _estimate_size(indices: frozenset[str], size_dict: dict[str, int]) -> int:
        return opt_einsum.paths.compute_size_by_dict(indices, size_dict)

    def __call__(
        self,
        inputs: list[frozenset[str]],
        output: frozenset[str],
        size_dict: dict[str, int],
        memory_limit: int | None = None,
    ):
        inputs = list(inputs)
        if self.tidx >= len(inputs):
            return opt_einsum.paths.greedy(inputs, output, size_dict, memory_limit)

        output = frozenset(output) | frozenset.intersection(*inputs)

        dim_to_keys: dict[str, set[frozenset[str]]] = defaultdict(set)
        for key in inputs:
            for dim in key - output:
                dim_to_keys[dim].add(key)

        dim_ref_counts = {
            count: {dim for dim, keys in dim_to_keys.items() if len(keys) >= count} - output
            for count in (2, 3)
        }
        footprints = {key: self._estimate_size(key, size_dict) for key in inputs}

        bidx = self.tidx
        k1 = inputs[bidx]
        candidates = [j for j in range(len(inputs)) if j != bidx]
        shared = [j for j in candidates if k1 & inputs[j]]
        if shared:
            candidates = shared

        best_j = None
        best_cost = None
        best_k12 = None
        best_size = None
        for j in candidates:
            k2 = inputs[j]
            either = k1 | k2
            two = k1 & k2
            one = either - two
            k12 = (either & output) | (two & dim_ref_counts[3]) | (one & dim_ref_counts[2])
            size12 = self._estimate_size(k12, size_dict)
            cost = size12 - footprints[k1] - footprints[k2]
            if best_cost is None or cost < best_cost or (cost == best_cost and size12 < best_size):
                best_cost = cost
                best_size = size12
                best_j = j
                best_k12 = k12

        if best_j is None or best_k12 is None:
            return opt_einsum.paths.greedy(inputs, output, size_dict, memory_limit)

        path = [(bidx, best_j)]
        for idx in sorted((bidx, best_j), reverse=True):
            inputs.pop(idx)
        inputs.append(best_k12)

        sub_path = self._tail_path(inputs, output, size_dict, memory_limit)
        path.extend(sub_path)
        return path

    def _tail_path(
        self,
        inputs: list[frozenset[str]],
        output: frozenset[str],
        size_dict: dict[str, int],
        memory_limit: int | None,
    ):
        opt = self.tail_optimize
        if isinstance(opt, opt_einsum.paths.PathOptimizer):
            return opt(inputs, output, size_dict, memory_limit)
        if callable(opt) and not isinstance(opt, str):
            return opt(inputs, output, size_dict, memory_limit)
        if isinstance(opt, str):
            if opt == "optimal":
                return opt_einsum.paths.optimal(inputs, output, size_dict, memory_limit)
            if opt.startswith("branch"):
                parts = opt.split("-", 1)
                nbranch = int(parts[1]) if len(parts) > 1 else 1
                return opt_einsum.paths.branch(
                    inputs,
                    output,
                    size_dict,
                    memory_limit,
                    nbranch=nbranch,
                )
            if opt in ("auto", "greedy"):
                return opt_einsum.paths.greedy(inputs, output, size_dict, memory_limit)
        return opt_einsum.paths.greedy(inputs, output, size_dict, memory_limit)
