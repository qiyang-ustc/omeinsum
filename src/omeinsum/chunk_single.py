from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch.utils.checkpoint import checkpoint


def run_chunked_einsum(
    einsum_fn: Callable[..., torch.Tensor],
    base_inputs: Sequence[torch.Tensor | None],
    tidx: int,
    chunks: Sequence[torch.Tensor],
    out_dim: int,
    use_checkpoint: bool,
    use_reentrant: bool,
) -> torch.Tensor:
    if not chunks:
        return einsum_fn(*base_inputs)

    def _call(*args: torch.Tensor) -> torch.Tensor:
        return einsum_fn(*args)

    results = []
    for chunk in chunks:
        inputs = list(base_inputs)
        inputs[tidx] = chunk
        if use_checkpoint and torch.is_grad_enabled():
            out = checkpoint(_call, *inputs, use_reentrant=use_reentrant)
        else:
            out = einsum_fn(*inputs)
        results.append(out)
    return torch.cat(results, dim=out_dim)
