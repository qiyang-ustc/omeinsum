# Optimizer Comparison Report: opt_einsum vs omeco

This report compares contraction path optimizers on extreme tensor network cases.

## Important Note on FLOP Counting

**The two libraries use different conventions:**
- **opt_einsum**: Counts multiply-add as 2 operations
- **omeco**: Counts multiply-add as 1 operation (MAC)

All comparisons below are **normalized** (opt_einsum values divided by 2) for fair comparison.

## Summary

After normalization, the comparison shows:

| Case | Result |
|------|--------|
| **Heisenberg (5 tensors)** | Both find same quality paths (1.0x) |
| **Kitaev (7 tensors)** | opt_einsum(optimal) is better (~0.5-0.6x) |

**Key Finding:** For the more complex 7-tensor Kitaev case, opt_einsum's dynamic programming `optimal` method finds better paths than omeco's TreeSA simulated annealing.

## Benchmark Results

### D=8

| chi | Heisenberg (opt/omeco) | Kitaev (opt/omeco) |
|-----|------------------------|-------------------|
| 64  | 1.00x | 0.62x |
| 128 | 1.00x | 0.56x |
| 192 | 1.00x | 0.54x |
| 256 | 1.00x | 0.53x |

### D=12

| chi | Heisenberg (opt/omeco) | Kitaev (opt/omeco) |
|-----|------------------------|-------------------|
| 144 | 1.00x | 0.58x |
| 192 | 1.00x | 0.56x |
| 256 | 1.00x | 0.55x |

**Note:** Ratio < 1.0 means opt_einsum uses fewer FLOPs (better).

## Test Cases

### Heisenberg iPEPS (5 tensors)
- **Equation:** `ibfj,iaep,xabcd,xefgh,jcgq->pdhq`
- **Parameters:** d=2, varying D and chi

### Kitaev iPEPS (7 tensors)
- **Equation:** `iABt,ijkl,xjAp,xkBq,yJap,yKbq,labc->tJKc`
- **Parameters:** d=2, varying D and chi

## Conclusions

### Why the Earlier Report Was Misleading

The initial analysis showed "2x speedup" for omeco, but this was entirely due to:
1. opt_einsum counting multiply-add as 2 FLOPs
2. omeco counting multiply-add as 1 FLOP

After normalizing for this difference, the actual path quality comparison shows:
- For simple cases (5 tensors): equivalent performance
- For complex cases (7 tensors): opt_einsum's DP algorithm is superior

### Recommendations

1. **For 5-tensor contractions** (like Heisenberg CTMRG): Either optimizer works equally well

2. **For 7+ tensor contractions** (like Kitaev CTMRG): opt_einsum's `optimal` method finds better paths

3. **For very large tensor networks** (10+ tensors): opt_einsum's `optimal` becomes exponentially slow; omeco's TreeSA may be more practical

### When omeco May Still Be Useful

- **Speed of optimization**: TreeSA is faster than DP for large networks
- **Slicing support**: TreeSASlicer can reduce memory at cost of more computation
- **Custom scoring**: More flexible optimization objectives

## Plots

See:
- `docs/benchmark_flops_vs_chi.png` - FLOPs vs chi comparison
- `docs/benchmark_speedup.png` - Speedup ratio plot
