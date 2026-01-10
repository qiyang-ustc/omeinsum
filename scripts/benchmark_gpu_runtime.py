#!/usr/bin/env python3
"""
GPU Runtime Benchmark: Compare actual execution time of different contraction paths.

This script measures the real runtime performance of tensor contractions
using paths found by opt_einsum vs omeco optimizers.

Usage:
    python scripts/benchmark_gpu_runtime.py [--warmup 3] [--repeat 10]
"""

from __future__ import annotations

import argparse
import time
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import opt_einsum

try:
    import omeco
    OMECO_AVAILABLE = True
except ImportError:
    OMECO_AVAILABLE = False


@dataclass
class BenchmarkResult:
    case_name: str
    optimizer: str
    chi: int
    D: int
    mean_time_ms: float
    std_time_ms: float
    flops: float  # theoretical FLOPs
    tflops: float  # achieved TFLOPS


def einsum_to_omeco_format(equation: str, shapes: tuple) -> tuple:
    """Convert einsum equation to omeco format (integer indices)."""
    input_part, output_part = equation.split("->")
    input_subscripts = [s.strip() for s in input_part.split(",")]
    output_subscript = output_part.strip()

    all_chars = set()
    for sub in input_subscripts:
        all_chars.update(sub)
    all_chars.update(output_subscript)
    char_to_int = {c: i for i, c in enumerate(sorted(all_chars))}

    ixs = [[char_to_int[c] for c in sub] for sub in input_subscripts]
    out = [char_to_int[c] for c in output_subscript]

    sizes = {}
    for subscript, shape in zip(input_subscripts, shapes):
        for idx_char, dim_size in zip(subscript, shape):
            sizes[char_to_int[idx_char]] = dim_size

    return ixs, out, sizes


def get_opt_einsum_path(equation: str, tensors: list[torch.Tensor], method: str = "optimal"):
    """Get contraction path from opt_einsum."""
    shapes = [t.shape for t in tensors]
    arrays = [np.empty(s) for s in shapes]
    path, info = opt_einsum.contract_path(equation, *arrays, optimize=method)
    flops = float(info.opt_cost)  # Note: opt_einsum counts MAC as 2
    return path, flops


def get_omeco_path(equation: str, tensors: list[torch.Tensor], method: str = "treesa"):
    """Get contraction path from omeco."""
    if not OMECO_AVAILABLE:
        raise ImportError("omeco not installed")

    shapes = [tuple(t.shape) for t in tensors]
    ixs, out, sizes = einsum_to_omeco_format(equation, shapes)

    if method == "greedy":
        method_obj = omeco.GreedyMethod()
    else:
        method_obj = omeco.TreeSA()

    tree = omeco.optimize_code(ixs, out, sizes, method_obj)
    comp = omeco.contraction_complexity(tree, ixs, sizes)
    flops = 2 ** comp.tc * 2  # Convert to opt_einsum convention (MAC as 2)

    # Extract path from tree (if available)
    # For now, we'll use opt_einsum to execute with the same path structure
    return None, flops


def benchmark_contraction(
    equation: str,
    tensors: list[torch.Tensor],
    path: list,
    warmup: int = 3,
    repeat: int = 10,
) -> tuple[float, float]:
    """Benchmark a single contraction with given path."""
    # Warmup
    for _ in range(warmup):
        _ = opt_einsum.contract(equation, *tensors, optimize=path)
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = opt_einsum.contract(equation, *tensors, optimize=path)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return np.mean(times), np.std(times)


def heisenberg_tensors(chi: int, d: int, D: int, device: torch.device, dtype: torch.dtype):
    """Create tensors for Heisenberg CTMRG Update-T."""
    return [
        torch.randn(chi, D, D, chi, device=device, dtype=dtype),  # T
        torch.randn(chi, D, D, chi, device=device, dtype=dtype),  # v1
        torch.randn(d, D, D, D, D, device=device, dtype=dtype),   # M
        torch.randn(d, D, D, D, D, device=device, dtype=dtype),   # M.conj()
        torch.randn(chi, D, D, chi, device=device, dtype=dtype),  # v2
    ]


def kitaev_tensors(chi: int, d: int, D: int, device: torch.device, dtype: torch.dtype):
    """Create tensors for Kitaev CTMRG Update-R."""
    return [
        torch.randn(chi, D, D, chi, device=device, dtype=dtype),  # v.conj()
        torch.randn(chi, D, D, chi, device=device, dtype=dtype),  # R
        torch.randn(d, D, D, D, device=device, dtype=dtype),      # M
        torch.randn(d, D, D, D, device=device, dtype=dtype),      # M.conj()
        torch.randn(d, D, D, D, device=device, dtype=dtype),      # M
        torch.randn(d, D, D, D, device=device, dtype=dtype),      # M.conj()
        torch.randn(chi, D, D, chi, device=device, dtype=dtype),  # v
    ]


HEISENBERG_EQ = "ibfj,iaep,xabcd,xefgh,jcgq->pdhq"
KITAEV_EQ = "iABt,ijkl,xjAp,xkBq,yJap,yKbq,labc->tJKc"


def run_benchmark(args):
    """Run the full benchmark suite."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    device = torch.device("cuda:0")
    dtype = torch.float64

    # Print GPU info
    print("=" * 70)
    print("GPU Runtime Benchmark: opt_einsum path optimization")
    print("=" * 70)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"dtype: {dtype}")
    print(f"Warmup: {args.warmup}, Repeat: {args.repeat}")

    results: list[BenchmarkResult] = []

    # Test configurations
    configs = [
        # (D, chi_list)
        (4, [24, 32, 48, 64]),
        (8, [64, 96, 128]),
        (12, [144, 192]),
    ]

    for D, chi_list in configs:
        print(f"\n{'='*70}")
        print(f"D = {D}")
        print(f"{'='*70}")

        for chi in chi_list:
            print(f"\nchi = {chi}")
            print("-" * 50)

            # Heisenberg
            tensors = heisenberg_tensors(chi, 2, D, device, dtype)
            path, flops = get_opt_einsum_path(HEISENBERG_EQ, tensors, "optimal")
            mean_ms, std_ms = benchmark_contraction(
                HEISENBERG_EQ, tensors, path, args.warmup, args.repeat
            )
            tflops = (flops / 1e12) / (mean_ms / 1000)

            print(f"  Heisenberg: {mean_ms:.2f} ± {std_ms:.2f} ms | {tflops:.2f} TFLOPS")

            results.append(BenchmarkResult(
                case_name="Heisenberg",
                optimizer="opt_einsum(optimal)",
                chi=chi, D=D,
                mean_time_ms=mean_ms,
                std_time_ms=std_ms,
                flops=flops,
                tflops=tflops,
            ))

            # Kitaev
            tensors = kitaev_tensors(chi, 2, D, device, dtype)
            path, flops = get_opt_einsum_path(KITAEV_EQ, tensors, "optimal")
            mean_ms, std_ms = benchmark_contraction(
                KITAEV_EQ, tensors, path, args.warmup, args.repeat
            )
            tflops = (flops / 1e12) / (mean_ms / 1000)

            print(f"  Kitaev:     {mean_ms:.2f} ± {std_ms:.2f} ms | {tflops:.2f} TFLOPS")

            results.append(BenchmarkResult(
                case_name="Kitaev",
                optimizer="opt_einsum(optimal)",
                chi=chi, D=D,
                mean_time_ms=mean_ms,
                std_time_ms=std_ms,
                flops=flops,
                tflops=tflops,
            ))

            # Clean up memory
            del tensors
            torch.cuda.empty_cache()

    # Print summary table
    print("\n" + "=" * 70)
    print("Summary Table")
    print("=" * 70)
    print(f"{'Case':<12} {'D':>3} {'chi':>5} {'Time (ms)':>12} {'TFLOPS':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r.case_name:<12} {r.D:>3} {r.chi:>5} {r.mean_time_ms:>8.2f} ± {r.std_time_ms:>4.2f} {r.tflops:>10.2f}")

    # Save results to file
    output_file = "docs/gpu_benchmark_results.txt"
    with open(output_file, "w") as f:
        f.write("GPU Runtime Benchmark Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"CUDA: {torch.version.cuda}\n")
        f.write(f"PyTorch: {torch.__version__}\n")
        f.write(f"dtype: {dtype}\n\n")
        f.write(f"{'Case':<12} {'D':>3} {'chi':>5} {'Time (ms)':>12} {'TFLOPS':>10}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f.write(f"{r.case_name:<12} {r.D:>3} {r.chi:>5} {r.mean_time_ms:>8.2f} ± {r.std_time_ms:>4.2f} {r.tflops:>10.2f}\n")

    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="GPU Runtime Benchmark")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--repeat", type=int, default=10, help="Number of timed iterations")
    args = parser.parse_args()

    run_benchmark(args)


if __name__ == "__main__":
    main()
