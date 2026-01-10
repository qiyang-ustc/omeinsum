#!/usr/bin/env python3
"""
CPU Runtime Benchmark: Verify theoretical FLOP difference with actual runtime.

Compare opt_einsum(optimal) vs omeco(TreeSA) on Kitaev equation.
Keep D=6, chi<=40 to avoid OOM.
"""

import time
import numpy as np
import opt_einsum

try:
    import omeco
    OMECO_AVAILABLE = True
except ImportError:
    OMECO_AVAILABLE = False
    print("ERROR: omeco not installed")
    exit(1)


KITAEV_EQ = "iABt,ijkl,xjAp,xkBq,yJap,yKbq,labc->tJKc"


def kitaev_tensors(chi: int, d: int, D: int):
    """Create tensors for Kitaev CTMRG."""
    return [
        np.random.randn(chi, D, D, chi),
        np.random.randn(chi, D, D, chi),
        np.random.randn(d, D, D, D),
        np.random.randn(d, D, D, D),
        np.random.randn(d, D, D, D),
        np.random.randn(d, D, D, D),
        np.random.randn(chi, D, D, chi),
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


def omeco_tree_to_path(tree, n_tensors: int):
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


def benchmark(warmup: int = 2, repeat: int = 5):
    """Run benchmark comparing opt_einsum vs omeco paths."""

    print("=" * 70)
    print("CPU Runtime Benchmark: opt_einsum(optimal) vs omeco(TreeSA)")
    print("=" * 70)
    print(f"\nKitaev equation: {KITAEV_EQ}")
    print(f"d=2, D=6 (d/D = 0.33 < 0.5, expect omeco to be worse)")
    print(f"Warmup: {warmup}, Repeat: {repeat}")
    print()

    D = 6
    d = 2

    print(f"{'chi':>5} | {'opt_einsum':>12} | {'omeco':>12} | {'ratio':>8} | {'理论比':>8}")
    print("-" * 60)

    for chi in [16, 20, 24, 28, 32, 36, 40]:
        tensors = kitaev_tensors(chi, d, D)
        shapes = [t.shape for t in tensors]

        # Get opt_einsum path
        path_opt, info_opt = opt_einsum.contract_path(KITAEV_EQ, *tensors, optimize="optimal")
        opt_flops = float(info_opt.opt_cost) / 2  # Normalize

        # Get omeco path
        ixs, out, sizes = einsum_to_omeco_format(KITAEV_EQ, shapes)
        tree = omeco.optimize_code(ixs, out, sizes, omeco.TreeSA())
        comp = omeco.contraction_complexity(tree, ixs, sizes)
        omeco_flops = 2 ** comp.tc
        path_omeco = omeco_tree_to_path(tree, len(tensors))

        theoretical_ratio = omeco_flops / opt_flops

        # Warmup opt_einsum path
        for _ in range(warmup):
            _ = opt_einsum.contract(KITAEV_EQ, *tensors, optimize=path_opt)

        # Time opt_einsum path
        times_opt = []
        for _ in range(repeat):
            start = time.perf_counter()
            _ = opt_einsum.contract(KITAEV_EQ, *tensors, optimize=path_opt)
            times_opt.append(time.perf_counter() - start)
        mean_opt = np.mean(times_opt) * 1000  # ms

        # Warmup omeco path
        for _ in range(warmup):
            _ = opt_einsum.contract(KITAEV_EQ, *tensors, optimize=path_omeco)

        # Time omeco path
        times_omeco = []
        for _ in range(repeat):
            start = time.perf_counter()
            _ = opt_einsum.contract(KITAEV_EQ, *tensors, optimize=path_omeco)
            times_omeco.append(time.perf_counter() - start)
        mean_omeco = np.mean(times_omeco) * 1000  # ms

        runtime_ratio = mean_omeco / mean_opt

        print(f"{chi:>5} | {mean_opt:>10.2f}ms | {mean_omeco:>10.2f}ms | {runtime_ratio:>7.2f}x | {theoretical_ratio:>7.2f}x")

    print()
    print("理论比 = omeco_flops / opt_flops (应该 ~1.6x)")
    print("ratio = omeco_time / opt_time (实际运行时间比)")
    print()
    print("如果两个比例接近，说明理论分析正确。")


if __name__ == "__main__":
    benchmark()
