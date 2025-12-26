from __future__ import annotations

from typing import Sequence

import opt_einsum
import torch
import torch.nn as nn

from .chunk_multi import run_chunked_einsum_multi_device
from .chunk_single import run_chunked_einsum


class OMEinsum(nn.Module):
    def __init__(
        self,
        equation: str,
        block_dim: str,
        batch_size: int = 16,
        use_checkpoint: bool = False,
        use_reentrant: bool = False,
        optimize: str | bool | None = "auto",
    ) -> None:
        super().__init__()
        self.equation = equation
        self.block_dim = block_dim
        self.batch_size = batch_size
        self.use_checkpoint = use_checkpoint
        self.use_reentrant = use_reentrant
        self.optimize = optimize

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
            path, _ = opt_einsum.contract_path(self.equation, *tensors, optimize=self.optimize)
            self._path = path
            self._path_shapes = shapes
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
