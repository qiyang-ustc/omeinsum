# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OMEinsum is a PyTorch library for einsum operations with chunked execution, optional gradient checkpointing, and automatic multi-GPU dispatch. It reduces peak memory by chunking along a specified dimension and distributes chunks across available CUDA devices.

## Development Commands

```bash
# Install for development
pip install -e .[dev]

# Run all tests (CUDA required)
pytest

# Run a single test
pytest tests/test_single_gpu.py::test_single_gpu_correctness -v
```

## Architecture

The library is structured around three main components:

- **`omeinsum.py`**: Contains `OMEinsum`, an `nn.Module` that parses einsum equations, validates inputs, and dispatches to single or multi-GPU execution. Also implements `_BlockFirstOptimizer`, a custom `opt_einsum.paths.PathOptimizer` that prioritizes contracting the blocked tensor first.

- **`chunk_single.py`**: `run_chunked_einsum()` processes chunks sequentially on a single device, optionally using `torch.utils.checkpoint` for gradient checkpointing.

- **`chunk_multi.py`**: `run_chunked_einsum_multi_device()` distributes chunks across multiple GPUs, moves tensors to target devices, runs single-device chunking per GPU, then gathers results back.

## Key Constraints

- CUDA is required; CPU tensors are not supported
- `block_dim` must appear exactly once across all input subscripts and must appear in the output subscript
- Explicit einsum notation with `->` is required
- Uses `opt_einsum` for contraction path optimization with caching based on tensor shapes

## HPC Cluster Tasks

When working on HPC/cluster tasks, follow the cluster-compute skill at:
https://github.com/qiyang-ustc/HPC-cluster-skills/blob/main/skills/cluster-compute/SKILL.md
