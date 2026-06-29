# OMEinsum

PyTorch einsum with chunked execution, optional checkpointing, and multi-GPU dispatch.

## Features

- Chunk a single einsum dimension to reduce peak memory.
- Optional per-chunk checkpointing for lower activation memory.
- Automatically dispatch chunks across multiple CUDA devices when available.
- Uses opt_einsum path caching for repeated calls with the same shapes.

## Requirements

- Python 3.9+
- PyTorch with CUDA support
- opt-einsum

## Installation (GitHub)

This package is intentionally distributed via GitHub only.

```bash
pip install git+https://github.com/qyang/omeinsum
```

Editable install for development:

```bash
pip install -e .[dev]
```

## Quickstart

```python
import torch
from omeinsum import OMEinsum

device = torch.device("cuda:0")

A = torch.randn(8, 4, 6, 5, device=device)
B = torch.randn(6, 3, device=device)
C = torch.randn(5, 2, device=device)
D = torch.randn(8, 7, device=device)

equation = "abcd,ce,df,ag->ebfg"
model = OMEinsum(
    equation,
    block_dim="b",   # chunk along this dimension
    batch_size=2,    # chunk size
    use_checkpoint=False,
    use_reentrant=False,
)

out = model(A, B, C, D)
```

## Notes and limitations

- CUDA is required; CPU tensors are not supported.
- `block_dim` must appear in the output subscript and only once across input subscripts.
- If multiple GPUs are available, chunks are distributed across them automatically.
- Use `use_checkpoint=True` to trade compute for memory when gradients are enabled.
- Experimental research code; APIs may change between commits.

## Optional YASTN compatibility

OMEinsum also includes an experimental YASTN compatibility layer for YASTN's
PyTorch `no_fusion` block-GEMM path. It is isolated under `omeinsum.compat.yastn`
and does not patch YASTN at import time.

```python
import yastn
from yastn.backend import backend_torch
from omeinsum.compat.yastn import enable_yastn_omeinsum

backend = enable_yastn_omeinsum(backend_torch)
config = yastn.make_config(backend=backend, sym="U1", tensordot_policy="no_fusion")
```

The first implementation keeps tensors on their current device and reserves
CPU/GPU mixed placement for a later, explicitly profiled version.

## Examples

- Heisenberg iPEPS demo (D=12,chi=1536 on four H100 GPUs): `tests/heisenberg_omeinsum.py`

## Development

Run tests (CUDA required for most):

```bash
pytest
```
