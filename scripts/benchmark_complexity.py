#!/usr/bin/env python3
"""
Benchmark script comparing opt_einsum vs omeco theoretical complexity.

Creates plots of FLOPs vs chi for different D values to verify the ~2x improvement.
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt

import opt_einsum

try:
    import omeco
    OMECO_AVAILABLE = True
except ImportError:
    OMECO_AVAILABLE = False
    print("Warning: omeco not installed. Install with: pip install omeco")


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


def get_opt_einsum_flops(equation: str, shapes: tuple, method: str = "optimal") -> float:
    """Get FLOPs (log10) from opt_einsum.

    Note: opt_einsum counts multiply-add as 2 ops, so we divide by 2
    to normalize with omeco which counts MAC as 1 op.
    """
    arrays = [np.empty(shape) for shape in shapes]
    try:
        _, info = opt_einsum.contract_path(equation, *arrays, optimize=method)
        flops = float(info.opt_cost) / 2.0  # Normalize: opt_einsum counts mul+add as 2
        return math.log10(flops) if flops > 0 else 0
    except Exception as e:
        print(f"opt_einsum error: {e}")
        return float('nan')


def get_omeco_flops(equation: str, shapes: tuple, method: str = "treesa") -> float:
    """Get FLOPs (log10) from omeco."""
    if not OMECO_AVAILABLE:
        return float('nan')

    ixs, out, sizes = einsum_to_omeco_format(equation, shapes)

    try:
        if method == "greedy":
            method_obj = omeco.GreedyMethod()
        else:
            method_obj = omeco.TreeSA()

        tree = omeco.optimize_code(ixs, out, sizes, method_obj)
        complexity = omeco.contraction_complexity(tree, ixs, sizes)
        # tc is log2 of FLOPs
        return complexity.tc * math.log10(2)
    except Exception as e:
        print(f"omeco error: {e}")
        return float('nan')


def heisenberg_shapes(chi: int, d: int, D: int) -> tuple:
    """Get tensor shapes for Heisenberg CTMRG Update-T."""
    # Equation: "ibfj,iaep,xabcd,xefgh,jcgq->pdhq"
    return (
        (chi, D, D, chi),  # T: ibfj
        (chi, D, D, chi),  # v1: iaep
        (d, D, D, D, D),   # M: xabcd
        (d, D, D, D, D),   # M.conj(): xefgh
        (chi, D, D, chi),  # v2: jcgq
    )


def kitaev_shapes(chi: int, d: int, D: int) -> tuple:
    """Get tensor shapes for Kitaev CTMRG Update-R."""
    # Equation: "iABt,ijkl,xjAp,xkBq,yJap,yKbq,labc->tJKc"
    return (
        (chi, D, D, chi),  # v.conj(): iABt
        (chi, D, D, chi),  # R: ijkl
        (d, D, D, D),      # M: xjAp
        (d, D, D, D),      # M.conj(): xkBq
        (d, D, D, D),      # M: yJap
        (d, D, D, D),      # M.conj(): yKbq
        (chi, D, D, chi),  # v: labc
    )


HEISENBERG_EQ = "ibfj,iaep,xabcd,xefgh,jcgq->pdhq"
KITAEV_EQ = "iABt,ijkl,xjAp,xkBq,yJap,yKbq,labc->tJKc"


def benchmark_vs_chi(D_values: list[int], chi_range: range, d: int = 2):
    """Benchmark FLOPs vs chi for different D values."""
    results = {
        "heisenberg": {},
        "kitaev": {},
    }

    for D in D_values:
        print(f"\n=== D = {D} ===")
        results["heisenberg"][D] = {
            "chi": [],
            "opt_einsum_greedy": [],
            "opt_einsum_optimal": [],
            "omeco_greedy": [],
            "omeco_treesa": [],
        }
        results["kitaev"][D] = {
            "chi": [],
            "opt_einsum_greedy": [],
            "opt_einsum_optimal": [],
            "omeco_greedy": [],
            "omeco_treesa": [],
        }

        for chi in chi_range:
            if chi < D * D:
                continue  # chi should be >= D^2 for meaningful CTMRG

            print(f"  chi={chi}", end=" ", flush=True)

            # Heisenberg
            h_shapes = heisenberg_shapes(chi, d, D)
            results["heisenberg"][D]["chi"].append(chi)
            results["heisenberg"][D]["opt_einsum_greedy"].append(
                get_opt_einsum_flops(HEISENBERG_EQ, h_shapes, "greedy")
            )
            results["heisenberg"][D]["opt_einsum_optimal"].append(
                get_opt_einsum_flops(HEISENBERG_EQ, h_shapes, "optimal")
            )
            results["heisenberg"][D]["omeco_greedy"].append(
                get_omeco_flops(HEISENBERG_EQ, h_shapes, "greedy")
            )
            results["heisenberg"][D]["omeco_treesa"].append(
                get_omeco_flops(HEISENBERG_EQ, h_shapes, "treesa")
            )

            # Kitaev
            k_shapes = kitaev_shapes(chi, d, D)
            results["kitaev"][D]["chi"].append(chi)
            results["kitaev"][D]["opt_einsum_greedy"].append(
                get_opt_einsum_flops(KITAEV_EQ, k_shapes, "greedy")
            )
            results["kitaev"][D]["opt_einsum_optimal"].append(
                get_opt_einsum_flops(KITAEV_EQ, k_shapes, "optimal")
            )
            results["kitaev"][D]["omeco_greedy"].append(
                get_omeco_flops(KITAEV_EQ, k_shapes, "greedy")
            )
            results["kitaev"][D]["omeco_treesa"].append(
                get_omeco_flops(KITAEV_EQ, k_shapes, "treesa")
            )

            print("done")

    return results


def plot_results(results: dict, D_values: list[int], output_prefix: str = "benchmark"):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        "opt_einsum_greedy": "C0",
        "opt_einsum_optimal": "C1",
        "omeco_greedy": "C2",
        "omeco_treesa": "C3",
    }
    labels = {
        "opt_einsum_greedy": "opt_einsum (greedy)",
        "opt_einsum_optimal": "opt_einsum (optimal)",
        "omeco_greedy": "omeco (greedy)",
        "omeco_treesa": "omeco (TreeSA)",
    }
    linestyles = {
        "opt_einsum_greedy": "--",
        "opt_einsum_optimal": "-",
        "omeco_greedy": "--",
        "omeco_treesa": "-",
    }

    for row, case in enumerate(["heisenberg", "kitaev"]):
        for col, D in enumerate(D_values):
            ax = axes[row, col]
            data = results[case][D]
            chi = data["chi"]

            for method in ["opt_einsum_greedy", "opt_einsum_optimal", "omeco_greedy", "omeco_treesa"]:
                ax.plot(
                    chi,
                    data[method],
                    color=colors[method],
                    linestyle=linestyles[method],
                    marker="o" if "optimal" in method or "treesa" in method else "s",
                    markersize=4,
                    label=labels[method],
                    linewidth=2,
                )

            ax.set_xlabel("χ (bond dimension)", fontsize=11)
            ax.set_ylabel("FLOPs (log₁₀)", fontsize=11)
            ax.set_title(f"{case.capitalize()} iPEPS, D={D}", fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.suptitle("Theoretical Contraction Complexity: opt_einsum vs omeco", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"docs/{output_prefix}_flops_vs_chi.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: docs/{output_prefix}_flops_vs_chi.png")

    # Also create speedup plot
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    for col, D in enumerate(D_values):
        ax = axes2[col]

        for case, marker in [("heisenberg", "o"), ("kitaev", "s")]:
            data = results[case][D]
            chi = data["chi"]

            # Speedup = 10^(opt_flops - omeco_flops)
            speedup_greedy = [
                10 ** (opt - ome) if not (np.isnan(opt) or np.isnan(ome)) else np.nan
                for opt, ome in zip(data["opt_einsum_greedy"], data["omeco_greedy"])
            ]
            speedup_optimal = [
                10 ** (opt - ome) if not (np.isnan(opt) or np.isnan(ome)) else np.nan
                for opt, ome in zip(data["opt_einsum_optimal"], data["omeco_treesa"])
            ]

            ax.plot(chi, speedup_greedy, marker=marker, linestyle="--",
                    label=f"{case} (greedy vs greedy)", alpha=0.7)
            ax.plot(chi, speedup_optimal, marker=marker, linestyle="-",
                    label=f"{case} (optimal vs TreeSA)", linewidth=2)

        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(y=2.0, color="red", linestyle=":", alpha=0.5, label="2x speedup")
        ax.set_xlabel("χ (bond dimension)", fontsize=11)
        ax.set_ylabel("FLOPs Ratio (opt_einsum / omeco)", fontsize=11)
        ax.set_title(f"D={D}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 5)

    plt.suptitle("FLOPs Speedup: opt_einsum / omeco", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"docs/{output_prefix}_speedup.png", dpi=150, bbox_inches="tight")
    print(f"Saved: docs/{output_prefix}_speedup.png")


def main():
    print("=" * 60)
    print("Benchmark: opt_einsum vs omeco - Theoretical Complexity")
    print("=" * 60)
    print("\nNote: FLOPs normalized (opt_einsum counts MAC as 2, omeco as 1)")

    if not OMECO_AVAILABLE:
        print("\nERROR: omeco not installed. Run: pip install omeco")
        return

    # Benchmark parameters
    D_values = [8, 12]
    chi_range = range(64, 257, 16)  # chi from 64 to 256, step 16
    d = 2

    print(f"\nParameters:")
    print(f"  D values: {D_values}")
    print(f"  chi range: {list(chi_range)}")
    print(f"  d (physical): {d}")

    results = benchmark_vs_chi(D_values, chi_range, d)
    plot_results(results, D_values)

    # Print summary table
    print("\n" + "=" * 60)
    print("Summary: FLOPs Speedup (opt_einsum / omeco)")
    print("=" * 60)

    for D in D_values:
        print(f"\nD = {D}:")
        print(f"{'chi':>6} | {'Heisenberg':^20} | {'Kitaev':^20}")
        print(f"{'':>6} | {'greedy':>9} {'optimal':>9} | {'greedy':>9} {'optimal':>9}")
        print("-" * 55)

        h_data = results["heisenberg"][D]
        k_data = results["kitaev"][D]

        for i, chi in enumerate(h_data["chi"]):
            h_speedup_g = 10 ** (h_data["opt_einsum_greedy"][i] - h_data["omeco_greedy"][i])
            h_speedup_o = 10 ** (h_data["opt_einsum_optimal"][i] - h_data["omeco_treesa"][i])
            k_speedup_g = 10 ** (k_data["opt_einsum_greedy"][i] - k_data["omeco_greedy"][i])
            k_speedup_o = 10 ** (k_data["opt_einsum_optimal"][i] - k_data["omeco_treesa"][i])

            print(f"{chi:>6} | {h_speedup_g:>8.2f}x {h_speedup_o:>8.2f}x | {k_speedup_g:>8.2f}x {k_speedup_o:>8.2f}x")


if __name__ == "__main__":
    main()
