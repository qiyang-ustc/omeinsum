#!/usr/bin/env python3
"""
GPU Runtime Benchmark: Compare opt_einsum vs omeco paths on GPU.

Tests Kitaev CTMRG equation with d/D < 0.5 to verify if omeco's
memory-efficient paths are also faster on GPU.
"""

import time
import numpy as np
import torch
import opt_einsum

try:
    import omeco
    OMECO_AVAILABLE = True
except ImportError:
    OMECO_AVAILABLE = False
    print("[LLM] ERROR: omeco not installed")
    exit(1)


KITAEV_EQ = "iABt,ijkl,xjAp,xkBq,yJap,yKbq,labc->tJKc"


def kitaev_tensors(chi: int, d: int, D: int, device, dtype):
    """Create tensors for Kitaev CTMRG."""
    return [
        torch.randn(chi, D, D, chi, device=device, dtype=dtype),
        torch.randn(chi, D, D, chi, device=device, dtype=dtype),
        torch.randn(d, D, D, D, device=device, dtype=dtype),
        torch.randn(d, D, D, D, device=device, dtype=dtype),
        torch.randn(d, D, D, D, device=device, dtype=dtype),
        torch.randn(d, D, D, D, device=device, dtype=dtype),
        torch.randn(chi, D, D, chi, device=device, dtype=dtype),
    ]


def einsum_to_omeco_format(equation: str, shapes: list):
    """Convert einsum to omeco format."""
    input_part, output_part = equation.split("->")
    subscripts = input_part.split(",")

    all_chars = set()
    for sub in subscripts:
        all_chars.update(sub)
    all_chars.update(output_part)
    char_to_int = {c: i for i, c in enumerate(sorted(all_chars))}

    ixs = [[char_to_int[c] for c in sub] for sub in subscripts]
    out = [char_to_int[c] for c in output_part]
    sizes = {}
    for sub, shape in zip(subscripts, shapes):
        for c, s in zip(sub, shape):
            sizes[char_to_int[c]] = s

    return ixs, out, sizes


def omeco_tree_to_path(tree, n_tensors):
    """Convert omeco NestedEinsum tree to opt_einsum path."""
    path = []
    current_indices = list(range(n_tensors))
    next_idx = n_tensors

    def process_node(node_dict):
        nonlocal next_idx, current_indices
        if 'tensor_index' in node_dict:
            return node_dict['tensor_index']
        args = node_dict['args']
        left_idx = process_node(args[0])
        right_idx = process_node(args[1])
        pos_left = current_indices.index(left_idx)
        pos_right = current_indices.index(right_idx)
        if pos_left > pos_right:
            pos_left, pos_right = pos_right, pos_left
        path.append((pos_left, pos_right))
        new_idx = next_idx
        next_idx += 1
        current_indices.remove(left_idx)
        current_indices.remove(right_idx)
        current_indices.append(new_idx)
        return new_idx

    tree_dict = tree.to_dict()
    process_node(tree_dict)
    return path


def benchmark(warmup: int = 3, repeat: int = 10):
    """Run GPU benchmark."""

    if not torch.cuda.is_available():
        print("[LLM] ERROR: CUDA not available")
        return

    device = torch.device("cuda:0")
    dtype = torch.float64

    print("[LLM] GPU Benchmark: opt_einsum(optimal) vs omeco(TreeSA)")
    print(f"[LLM] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[LLM] Kitaev equation: {KITAEV_EQ}")
    print(f"[LLM] d=2, D=6 (d/D = 0.33 < 0.5)")
    print()

    D = 6
    d = 2

    print(f"{'chi':>5} | {'opt_einsum':>12} | {'omeco':>12} | {'ratio':>8} | {'FLOP比':>8}")
    print("-" * 65)

    for chi in [32, 48, 64, 80, 96, 112, 128]:
        try:
            tensors = kitaev_tensors(chi, d, D, device, dtype)
            shapes = [tuple(t.shape) for t in tensors]

            # Get paths using CPU numpy arrays
            cpu_arrays = [np.empty(s) for s in shapes]
            path_opt, info_opt = opt_einsum.contract_path(KITAEV_EQ, *cpu_arrays, optimize="optimal")
            opt_flops = float(info_opt.opt_cost) / 2

            ixs, out, sizes = einsum_to_omeco_format(KITAEV_EQ, shapes)
            tree = omeco.optimize_code(ixs, out, sizes, omeco.TreeSA())
            comp = omeco.contraction_complexity(tree, ixs, sizes)
            omeco_flops = 2 ** comp.tc
            path_omeco = omeco_tree_to_path(tree, 7)

            flop_ratio = omeco_flops / opt_flops

            # Warmup opt_einsum path
            for _ in range(warmup):
                _ = opt_einsum.contract(KITAEV_EQ, *tensors, optimize=path_opt)
                torch.cuda.synchronize()

            # Time opt_einsum path
            times_opt = []
            for _ in range(repeat):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = opt_einsum.contract(KITAEV_EQ, *tensors, optimize=path_opt)
                torch.cuda.synchronize()
                times_opt.append(time.perf_counter() - start)
            mean_opt = np.mean(times_opt) * 1000

            # Warmup omeco path
            for _ in range(warmup):
                _ = opt_einsum.contract(KITAEV_EQ, *tensors, optimize=path_omeco)
                torch.cuda.synchronize()

            # Time omeco path
            times_omeco = []
            for _ in range(repeat):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = opt_einsum.contract(KITAEV_EQ, *tensors, optimize=path_omeco)
                torch.cuda.synchronize()
                times_omeco.append(time.perf_counter() - start)
            mean_omeco = np.mean(times_omeco) * 1000

            runtime_ratio = mean_omeco / mean_opt

            print(f"{chi:>5} | {mean_opt:>10.2f}ms | {mean_omeco:>10.2f}ms | {runtime_ratio:>7.2f}x | {flop_ratio:>7.2f}x")

            # Clean up
            del tensors
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"{chi:>5} | ERROR: {e}")
            torch.cuda.empty_cache()

    print()
    print("[LLM] FLOP比 = omeco_flops / opt_flops")
    print("[LLM] ratio = omeco_time / opt_time")
    print("[LLM] CPU结果显示: FLOP比 ~1.5x 但 runtime ratio ~0.7x (omeco更快)")
    print("[LLM] 如果GPU也是 ratio < 1, 说明内存效率在GPU上也有帮助")


if __name__ == "__main__":
    benchmark()
