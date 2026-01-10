#!/usr/bin/env python3
"""
GPU AD Benchmark: Compare opt_einsum vs omeco paths with automatic differentiation.

Tests the Kitaev CTMRG update_R contraction including backward pass.
This is a stronger test because it measures the full forward + gradient computation.
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
DTYPE = torch.complex128


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


def create_tensors(chi: int, d: int, D: int, device, dtype, requires_grad=False):
    """Create tensors for Kitaev CTMRG update_R."""
    # v.conj(): (chi, D, D, chi)
    # R: (chi, D, D, chi)
    # M: (d, D, D, D) x 4
    # v: (chi, D, D, chi)
    v = torch.randn(chi, D, D, chi, device=device, dtype=dtype)
    R = torch.randn(chi, D, D, chi, device=device, dtype=dtype)
    M = torch.randn(d, D, D, D, device=device, dtype=dtype, requires_grad=requires_grad)

    return [v.conj(), R, M, M.conj(), M, M.conj(), v]


def benchmark_forward_backward(equation, tensors, path, warmup=3, repeat=5):
    """Benchmark forward + backward pass."""
    # Find which tensor requires grad
    grad_idx = None
    for i, t in enumerate(tensors):
        if t.requires_grad:
            grad_idx = i
            break

    # Warmup
    for _ in range(warmup):
        result = opt_einsum.contract(equation, *tensors, optimize=path)
        loss = result.real.sum() + result.imag.sum()
        loss.backward()
        if grad_idx is not None:
            tensors[grad_idx].grad = None
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()

        result = opt_einsum.contract(equation, *tensors, optimize=path)
        loss = result.real.sum() + result.imag.sum()
        loss.backward()

        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

        if grad_idx is not None:
            tensors[grad_idx].grad = None

    return np.mean(times) * 1000, np.std(times) * 1000


def benchmark():
    """Run GPU AD benchmark."""

    if not torch.cuda.is_available():
        print("[LLM] ERROR: CUDA not available")
        return

    device = torch.device("cuda:0")

    print("[LLM] GPU AD Benchmark: opt_einsum(optimal) vs omeco(TreeSA)")
    print(f"[LLM] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[LLM] Equation: {KITAEV_EQ}")
    print(f"[LLM] Testing forward + backward (AD)")
    print()

    # Test configurations: (chi, d, D)
    configs = [
        (16, 2, 4),   # d/D = 0.5 (boundary)
        (20, 2, 4),
        (24, 2, 4),
        (16, 2, 6),   # d/D = 0.33 < 0.5
        (20, 2, 6),
        (24, 2, 6),
        (32, 2, 6),
    ]

    print(f"{'chi':>5} {'d':>3} {'D':>3} {'d/D':>6} | {'opt_einsum':>12} | {'omeco':>12} | {'ratio':>8} | {'FLOP比':>8}")
    print("-" * 80)

    for chi, d, D in configs:
        try:
            # Create tensors with gradient
            tensors_opt = create_tensors(chi, d, D, device, DTYPE, requires_grad=True)
            tensors_omeco = create_tensors(chi, d, D, device, DTYPE, requires_grad=True)
            shapes = [tuple(t.shape) for t in tensors_opt]

            # Get paths using CPU arrays
            cpu_arrays = [np.empty(s) for s in shapes]
            path_opt, info_opt = opt_einsum.contract_path(KITAEV_EQ, *cpu_arrays, optimize="optimal")
            opt_flops = float(info_opt.opt_cost) / 2

            ixs, out, sizes = einsum_to_omeco_format(KITAEV_EQ, shapes)
            tree = omeco.optimize_code(ixs, out, sizes, omeco.TreeSA())
            comp = omeco.contraction_complexity(tree, ixs, sizes)
            omeco_flops = 2 ** comp.tc
            path_omeco = omeco_tree_to_path(tree, 7)

            flop_ratio = omeco_flops / opt_flops

            # Benchmark
            mean_opt, std_opt = benchmark_forward_backward(KITAEV_EQ, tensors_opt, path_opt)
            mean_omeco, std_omeco = benchmark_forward_backward(KITAEV_EQ, tensors_omeco, path_omeco)

            runtime_ratio = mean_omeco / mean_opt
            d_over_D = d / D

            print(f"{chi:>5} {d:>3} {D:>3} {d_over_D:>6.2f} | {mean_opt:>10.2f}ms | {mean_omeco:>10.2f}ms | {runtime_ratio:>7.2f}x | {flop_ratio:>7.2f}x")

            # Clean up
            del tensors_opt, tensors_omeco
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"{chi:>5} {d:>3} {D:>3} | ERROR: {e}")
            torch.cuda.empty_cache()

    print()
    print("[LLM] 说明:")
    print("[LLM] - FLOP比 = omeco_flops / opt_flops")
    print("[LLM] - ratio = omeco_time / opt_time (包含forward+backward)")
    print("[LLM] - d/D >= 0.5: 两者应找到相同路径 (FLOP比 ~1.0)")
    print("[LLM] - d/D < 0.5: omeco有更多FLOP但更小中间张量")


if __name__ == "__main__":
    benchmark()
