#!/usr/bin/env python3
"""
Theoretical complexity analysis comparing opt_einsum vs omeco path optimizers.

Analyzes two extreme cases from the test suite:
1. Heisenberg iPEPS: "ibfj,iaep,xabcd,xefgh,jcgq->pdhq" (5 tensors)
2. Kitaev iPEPS: "iABt,ijkl,xjAp,xkBq,yJap,yKbq,labc->tJKc" (7 tensors)

Outputs theoretical time complexity (FLOPs) and space complexity (max intermediate size).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import opt_einsum

try:
    import omeco
    OMECO_AVAILABLE = True
except ImportError:
    OMECO_AVAILABLE = False
    print("Warning: omeco not installed. Only opt_einsum results will be shown.")
    print("Install with: pip install omeco")


@dataclass
class ComplexityResult:
    """Results from complexity analysis."""
    case_name: str
    optimizer: str
    equation: str
    path: list
    flops_log10: float
    max_intermediate: int
    num_contractions: int
    path_info: Optional[str] = None


def einsum_to_omeco_format(
    equation: str,
    shapes: tuple[tuple[int, ...], ...],
) -> tuple[list[list[int]], list[int], dict[int, int]]:
    """Convert einsum equation and shapes to omeco (ixs, out, sizes) format.

    omeco uses integer indices, so we map each unique character to an integer.
    """
    input_part, output_part = equation.split("->")
    input_subscripts = [s.strip() for s in input_part.split(",")]
    output_subscript = output_part.strip()

    # Build mapping from character to integer index
    all_chars = set()
    for sub in input_subscripts:
        all_chars.update(sub)
    all_chars.update(output_subscript)
    char_to_int = {c: i for i, c in enumerate(sorted(all_chars))}

    # Convert to integer indices
    ixs = [[char_to_int[c] for c in sub] for sub in input_subscripts]
    out = [char_to_int[c] for c in output_subscript]

    # Build sizes dictionary with integer keys
    sizes: dict[int, int] = {}
    for subscript, shape in zip(input_subscripts, shapes):
        for idx_char, dim_size in zip(subscript, shape):
            idx_int = char_to_int[idx_char]
            if idx_int in sizes:
                assert sizes[idx_int] == dim_size
            else:
                sizes[idx_int] = dim_size

    return ixs, out, sizes


def analyze_opt_einsum(
    case_name: str,
    equation: str,
    shapes: tuple[tuple[int, ...], ...],
    optimize: str = "greedy",
) -> ComplexityResult:
    """Analyze contraction complexity using opt_einsum."""
    # Create dummy arrays for path finding
    arrays = [np.empty(shape) for shape in shapes]

    try:
        path, info = opt_einsum.contract_path(equation, *arrays, optimize=optimize)
    except Exception as e:
        return ComplexityResult(
            case_name=case_name,
            optimizer=f"opt_einsum({optimize})",
            equation=equation,
            path=[],
            flops_log10=-1,
            max_intermediate=-1,
            num_contractions=-1,
            path_info=f"ERROR: {e}",
        )

    # Extract complexity from PathInfo
    flops = info.opt_cost if hasattr(info, 'opt_cost') else 0
    flops_log10 = math.log10(flops) if flops > 0 else 0

    return ComplexityResult(
        case_name=case_name,
        optimizer=f"opt_einsum({optimize})",
        equation=equation,
        path=path,
        flops_log10=round(flops_log10, 2),
        max_intermediate=info.largest_intermediate,
        num_contractions=len(path),
        path_info=str(info),
    )


def analyze_omeco(
    case_name: str,
    equation: str,
    shapes: tuple[tuple[int, ...], ...],
    method: str = "greedy",
) -> ComplexityResult:
    """Analyze contraction complexity using omeco."""
    if not OMECO_AVAILABLE:
        return ComplexityResult(
            case_name=case_name,
            optimizer=f"omeco({method})",
            equation=equation,
            path=[],
            flops_log10=-1,
            max_intermediate=-1,
            num_contractions=-1,
            path_info="ERROR: omeco not installed",
        )

    ixs, out, sizes = einsum_to_omeco_format(equation, shapes)

    try:
        if method == "treesa_slicer":
            # TreeSASlicer requires a two-step process:
            # 1. First optimize with TreeSA
            # 2. Then apply slicing
            tree = omeco.optimize_code(ixs, out, sizes, omeco.TreeSA())
            slicer = omeco.TreeSASlicer.fast()
            sliced = omeco.slice_code(tree, ixs, sizes, slicer)
            complexity = omeco.sliced_complexity(sliced, ixs, sizes)

            tc = complexity.tc if hasattr(complexity, 'tc') else 0
            sc = complexity.sc if hasattr(complexity, 'sc') else 0
            flops_log10 = tc * math.log10(2) if tc > 0 else 0
            max_intermediate = int(2 ** sc) if sc > 0 else 0

            # Get slicing info
            slicing_info = ""
            if hasattr(sliced, 'slicing'):
                slicing_info = f", sliced_indices={sliced.slicing()}"

            return ComplexityResult(
                case_name=case_name,
                optimizer=f"omeco({method})",
                equation=equation,
                path=[],  # Sliced tree doesn't have simple path
                flops_log10=round(flops_log10, 2),
                max_intermediate=max_intermediate,
                num_contractions=len(ixs) - 1,
                path_info=f"tc={tc:.2f} (log2), sc={sc:.2f} (log2){slicing_info}",
            )
        else:
            method_map = {
                "greedy": omeco.GreedyMethod,
                "treesa": omeco.TreeSA,
            }
            method_obj = method_map[method]()

            tree = omeco.optimize_code(ixs, out, sizes, method_obj)
            complexity = omeco.contraction_complexity(tree, ixs, sizes)

            # tc is log2 of time complexity (FLOPs)
            # sc is log2 of space complexity (max intermediate size)
            tc = complexity.tc if hasattr(complexity, 'tc') else 0
            sc = complexity.sc if hasattr(complexity, 'sc') else 0

            # Convert log2 to log10 for FLOPs
            flops_log10 = tc * math.log10(2) if tc > 0 else 0

            # Convert log2 to actual size for space
            max_intermediate = int(2 ** sc) if sc > 0 else 0

            # Try to get path from tree
            path = []
            if hasattr(tree, 'flat_path'):
                path = tree.flat_path()
            elif hasattr(tree, 'get_path'):
                path = tree.get_path()

            return ComplexityResult(
                case_name=case_name,
                optimizer=f"omeco({method})",
                equation=equation,
                path=path,
                flops_log10=round(flops_log10, 2),
                max_intermediate=max_intermediate,
                num_contractions=len(ixs) - 1,  # n-1 contractions for n tensors
                path_info=f"tc={tc:.2f} (log2), sc={sc:.2f} (log2)",
            )

    except Exception as e:
        import traceback
        return ComplexityResult(
            case_name=case_name,
            optimizer=f"omeco({method})",
            equation=equation,
            path=[],
            flops_log10=-1,
            max_intermediate=-1,
            num_contractions=-1,
            path_info=f"ERROR: {e}\n{traceback.format_exc()}",
        )


def heisenberg_case() -> tuple[str, str, tuple[tuple[int, ...], ...]]:
    """
    Heisenberg iPEPS CTMRG Update-T contraction.

    Equation: "ibfj,iaep,xabcd,xefgh,jcgq->pdhq"
    From: tests/heisenberg_omeinsum.py line 82

    Parameters: chi=24, d=2, D=4
    Tensors:
      - T: (chi, D, D, chi) -> ibfj
      - v1: (chi, D, D, chi) -> iaep
      - M: (d, D, D, D, D) -> xabcd
      - M.conj(): (d, D, D, D, D) -> xefgh
      - v2: (chi, D, D, chi) -> jcgq
    """
    chi, d, D = 24, 2, 4
    equation = "ibfj,iaep,xabcd,xefgh,jcgq->pdhq"
    shapes = (
        (chi, D, D, chi),  # T: ibfj
        (chi, D, D, chi),  # v1: iaep
        (d, D, D, D, D),   # M: xabcd
        (d, D, D, D, D),   # M.conj(): xefgh
        (chi, D, D, chi),  # v2: jcgq
    )
    return "Heisenberg", equation, shapes


def kitaev_case() -> tuple[str, str, tuple[tuple[int, ...], ...]]:
    """
    Kitaev iPEPS CTMRG Update-R contraction.

    Equation: "iABt,ijkl,xjAp,xkBq,yJap,yKbq,labc->tJKc"
    From: tests/kitaev_omeinsum.py line 143

    Parameters: chi=20, d=2, D=4
    Tensors:
      - v.conj(): (chi, D, D, chi) -> iABt
      - R: (chi, D, D, chi) -> ijkl
      - M: (d, D, D, D) -> xjAp
      - M.conj(): (d, D, D, D) -> xkBq
      - M: (d, D, D, D) -> yJap
      - M.conj(): (d, D, D, D) -> yKbq
      - v: (chi, D, D, chi) -> labc
    """
    chi, d, D = 20, 2, 4
    equation = "iABt,ijkl,xjAp,xkBq,yJap,yKbq,labc->tJKc"
    shapes = (
        (chi, D, D, chi),  # v.conj(): iABt
        (chi, D, D, chi),  # R: ijkl
        (d, D, D, D),      # M: xjAp
        (d, D, D, D),      # M.conj(): xkBq
        (d, D, D, D),      # M: yJap
        (d, D, D, D),      # M.conj(): yKbq
        (chi, D, D, chi),  # v: labc
    )
    return "Kitaev", equation, shapes


def format_number(n: int) -> str:
    """Format large numbers with commas."""
    if n < 0:
        return "N/A"
    return f"{n:,}"


def generate_report(results: list[ComplexityResult]) -> str:
    """Generate markdown report from results."""
    lines = [
        "# Optimizer Comparison Report: opt_einsum vs omeco",
        "",
        "This report compares contraction path optimizers on extreme tensor network cases.",
        "",
        "## Summary Table",
        "",
        "| Case | Optimizer | FLOPs (log10) | Max Intermediate | Contractions |",
        "|------|-----------|---------------|------------------|--------------|",
    ]

    for r in results:
        flops = f"{r.flops_log10:.2f}" if r.flops_log10 >= 0 else "ERROR"
        intermediate = format_number(r.max_intermediate)
        contractions = str(r.num_contractions) if r.num_contractions >= 0 else "ERROR"
        lines.append(f"| {r.case_name} | {r.optimizer} | {flops} | {intermediate} | {contractions} |")

    lines.extend([
        "",
        "## Test Cases",
        "",
        "### Heisenberg iPEPS (5 tensors)",
        "",
        "**Equation:** `ibfj,iaep,xabcd,xefgh,jcgq->pdhq`",
        "",
        "**Parameters:** chi=24, d=2, D=4",
        "",
        "**Tensor shapes:**",
        "- T: (24, 4, 4, 24) - transfer tensor",
        "- v1: (24, 4, 4, 24) - isometry",
        "- M: (2, 4, 4, 4, 4) - bulk PEPS tensor",
        "- M*: (2, 4, 4, 4, 4) - conjugate bulk tensor",
        "- v2: (24, 4, 4, 24) - isometry",
        "",
        "### Kitaev iPEPS (7 tensors)",
        "",
        "**Equation:** `iABt,ijkl,xjAp,xkBq,yJap,yKbq,labc->tJKc`",
        "",
        "**Parameters:** chi=20, d=2, D=4",
        "",
        "**Tensor shapes:**",
        "- v*: (20, 4, 4, 20) - conjugate isometry",
        "- R: (20, 4, 4, 20) - row transfer tensor",
        "- M: (2, 4, 4, 4) - bulk tensor (appears 4 times with different indices)",
        "- v: (20, 4, 4, 20) - isometry",
        "",
        "## Detailed Results",
        "",
    ])

    # Group results by case
    heisenberg_results = [r for r in results if r.case_name == "Heisenberg"]
    kitaev_results = [r for r in results if r.case_name == "Kitaev"]

    for case_results, case_name in [(heisenberg_results, "Heisenberg"), (kitaev_results, "Kitaev")]:
        lines.append(f"### {case_name} Case Details")
        lines.append("")

        for r in case_results:
            lines.extend([
                f"#### {r.optimizer}",
                "",
                f"- **FLOPs (log10):** {r.flops_log10:.2f}" if r.flops_log10 >= 0 else "- **FLOPs:** ERROR",
                f"- **Max Intermediate Size:** {format_number(r.max_intermediate)} elements",
                f"- **Number of Contractions:** {r.num_contractions}",
                "",
            ])

            if r.path and len(r.path) > 0:
                lines.append(f"**Path:** `{r.path}`")
                lines.append("")

            if r.path_info and "ERROR" in r.path_info:
                lines.extend([
                    "**Error Details:**",
                    "```",
                    r.path_info,
                    "```",
                    "",
                ])

    lines.extend([
        "## Interpretation",
        "",
        "- **FLOPs (log10):** Lower is better. Represents total floating-point operations.",
        "- **Max Intermediate:** Lower is better. Peak memory usage during contraction.",
        "- **TreeSA** uses simulated annealing and typically finds better paths than greedy.",
        "- **TreeSASlicer** can reduce memory by slicing indices at the cost of more FLOPs.",
        "",
        "## Notes",
        "",
        "- opt_einsum's `optimal` method uses dynamic programming (exponential in tensor count).",
        "- omeco is a Rust library providing fast path optimization for tensor networks.",
        "- These are theoretical complexity estimates; actual runtime depends on hardware.",
    ])

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Optimizer Comparison: opt_einsum vs omeco")
    print("=" * 60)
    print()

    results: list[ComplexityResult] = []

    # Define test cases
    cases = [heisenberg_case(), kitaev_case()]

    # opt_einsum methods to test
    opt_einsum_methods = ["greedy", "optimal", "branch-2"]

    # omeco methods to test
    omeco_methods = ["greedy", "treesa", "treesa_slicer"]

    for case_name, equation, shapes in cases:
        print(f"Analyzing {case_name} case...")
        print(f"  Equation: {equation}")
        print(f"  Tensors: {len(shapes)}")
        print()

        # Test opt_einsum methods
        for method in opt_einsum_methods:
            print(f"  Running opt_einsum({method})...", end=" ", flush=True)
            result = analyze_opt_einsum(case_name, equation, shapes, optimize=method)
            results.append(result)
            if result.flops_log10 >= 0:
                print(f"FLOPs=10^{result.flops_log10:.2f}, MaxMem={format_number(result.max_intermediate)}")
            else:
                print("ERROR")

        # Test omeco methods
        if OMECO_AVAILABLE:
            for method in omeco_methods:
                print(f"  Running omeco({method})...", end=" ", flush=True)
                result = analyze_omeco(case_name, equation, shapes, method=method)
                results.append(result)
                if result.flops_log10 >= 0:
                    print(f"FLOPs=10^{result.flops_log10:.2f}, MaxMem={format_number(result.max_intermediate)}")
                else:
                    print("ERROR")
        else:
            print("  Skipping omeco (not installed)")

        print()

    # Generate and save report
    report = generate_report(results)

    report_path = "docs/optimizer_comparison.md"
    try:
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Report saved to: {report_path}")
    except Exception as e:
        print(f"Failed to save report: {e}")
        print("\n--- Report Content ---\n")
        print(report)

    return results


if __name__ == "__main__":
    main()
