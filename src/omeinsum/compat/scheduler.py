from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BlockGemmPolicy:
    """Execution policy for YASTN block GEMM tasks.

    CPU routing is intentionally disabled in the first implementation.  The
    field is kept to make future placement changes explicit instead of hidden
    behind fallback behavior.
    """

    cpu_enabled: bool = False
    batch_size: int = 16
    min_group: int = 2
    executor: str = "stack_bmm"
    stats: dict[str, int] | None = None


@dataclass(frozen=True)
class BlockGemmStats:
    num_a_blocks: int
    num_b_blocks: int
    num_c_blocks: int
    num_contributions: int
    unique_shapes: int
    total_flops: int
    largest_flops: int
    smallest_flops: int


def estimate_block_gemm_stats(
    meta_dot: Sequence,
    areshape: Sequence,
    breshape: Sequence,
) -> BlockGemmStats:
    shapes: set[tuple[int, int, int]] = set()
    total_flops = 0
    largest = 0
    smallest: int | None = None
    contributions = 0

    for _, _, list_tab in meta_dot:
        for ta, tb in list_tab:
            m = int(areshape[ta][2])
            k = int(areshape[ta][3])
            kb = int(breshape[tb][2])
            n = int(breshape[tb][3])
            if k != kb:
                raise ValueError(f"Block GEMM inner dimensions differ: {k} != {kb}")
            flops = 2 * m * k * n
            shapes.add((m, k, n))
            total_flops += flops
            largest = max(largest, flops)
            smallest = flops if smallest is None else min(smallest, flops)
            contributions += 1

    return BlockGemmStats(
        num_a_blocks=len(areshape),
        num_b_blocks=len(breshape),
        num_c_blocks=len(meta_dot),
        num_contributions=contributions,
        unique_shapes=len(shapes),
        total_flops=total_flops,
        largest_flops=largest,
        smallest_flops=0 if smallest is None else smallest,
    )
