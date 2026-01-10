# Optimizer Comparison Report: opt_einsum vs omeco

This report compares contraction path optimizers on extreme tensor network cases.

## Summary Table

| Case | Optimizer | FLOPs (log10) | Max Intermediate | Contractions |
|------|-----------|---------------|------------------|--------------|
| Heisenberg | opt_einsum(greedy) | 7.95 | 147,456 | 4 |
| Heisenberg | opt_einsum(optimal) | 7.52 | 294,912 | 4 |
| Heisenberg | opt_einsum(branch-2) | 7.52 | 294,912 | 4 |
| Heisenberg | omeco(greedy) | 7.65 | 147,455 | 4 |
| Heisenberg | omeco(treesa) | 7.22 | 294,911 | 4 |
| Heisenberg | omeco(treesa_slicer) | 7.22 | 294,911 | 4 |
| Kitaev | opt_einsum(greedy) | 7.79 | 102,400 | 6 |
| Kitaev | opt_einsum(optimal) | 7.09 | 102,400 | 6 |
| Kitaev | opt_einsum(branch-2) | 7.31 | 51,200 | 6 |
| Kitaev | omeco(greedy) | 7.49 | 102,399 | 6 |
| Kitaev | omeco(treesa) | 6.79 | 102,399 | 6 |
| Kitaev | omeco(treesa_slicer) | 6.79 | 102,399 | 6 |

## Test Cases

### Heisenberg iPEPS (5 tensors)

**Equation:** `ibfj,iaep,xabcd,xefgh,jcgq->pdhq`

**Parameters:** chi=24, d=2, D=4

**Tensor shapes:**
- T: (24, 4, 4, 24) - transfer tensor
- v1: (24, 4, 4, 24) - isometry
- M: (2, 4, 4, 4, 4) - bulk PEPS tensor
- M*: (2, 4, 4, 4, 4) - conjugate bulk tensor
- v2: (24, 4, 4, 24) - isometry

### Kitaev iPEPS (7 tensors)

**Equation:** `iABt,ijkl,xjAp,xkBq,yJap,yKbq,labc->tJKc`

**Parameters:** chi=20, d=2, D=4

**Tensor shapes:**
- v*: (20, 4, 4, 20) - conjugate isometry
- R: (20, 4, 4, 20) - row transfer tensor
- M: (2, 4, 4, 4) - bulk tensor (appears 4 times with different indices)
- v: (20, 4, 4, 20) - isometry

## Detailed Results

### Heisenberg Case Details

#### opt_einsum(greedy)

- **FLOPs (log10):** 7.95
- **Max Intermediate Size:** 147,456 elements
- **Number of Contractions:** 4

**Path:** `[(2, 3), (0, 1), (1, 2), (0, 1)]`

#### opt_einsum(optimal)

- **FLOPs (log10):** 7.52
- **Max Intermediate Size:** 294,912 elements
- **Number of Contractions:** 4

**Path:** `[(0, 1), (0, 3), (0, 2), (0, 1)]`

#### opt_einsum(branch-2)

- **FLOPs (log10):** 7.52
- **Max Intermediate Size:** 294,912 elements
- **Number of Contractions:** 4

**Path:** `[(0, 1), (0, 3), (0, 2), (0, 1)]`

#### omeco(greedy)

- **FLOPs (log10):** 7.65
- **Max Intermediate Size:** 147,455 elements
- **Number of Contractions:** 4

#### omeco(treesa)

- **FLOPs (log10):** 7.22
- **Max Intermediate Size:** 294,911 elements
- **Number of Contractions:** 4

#### omeco(treesa_slicer)

- **FLOPs (log10):** 7.22
- **Max Intermediate Size:** 294,911 elements
- **Number of Contractions:** 4

### Kitaev Case Details

#### opt_einsum(greedy)

- **FLOPs (log10):** 7.79
- **Max Intermediate Size:** 102,400 elements
- **Number of Contractions:** 6

**Path:** `[(2, 4), (2, 3), (3, 4), (0, 1), (1, 2), (0, 1)]`

#### opt_einsum(optimal)

- **FLOPs (log10):** 7.09
- **Max Intermediate Size:** 102,400 elements
- **Number of Contractions:** 6

**Path:** `[(0, 1), (0, 5), (0, 4), (0, 3), (0, 2), (0, 1)]`

#### opt_einsum(branch-2)

- **FLOPs (log10):** 7.31
- **Max Intermediate Size:** 51,200 elements
- **Number of Contractions:** 6

**Path:** `[(2, 4), (0, 2), (1, 2), (0, 2), (0, 2), (0, 1)]`

#### omeco(greedy)

- **FLOPs (log10):** 7.49
- **Max Intermediate Size:** 102,399 elements
- **Number of Contractions:** 6

#### omeco(treesa)

- **FLOPs (log10):** 6.79
- **Max Intermediate Size:** 102,399 elements
- **Number of Contractions:** 6

#### omeco(treesa_slicer)

- **FLOPs (log10):** 6.79
- **Max Intermediate Size:** 102,399 elements
- **Number of Contractions:** 6

## Interpretation

- **FLOPs (log10):** Lower is better. Represents total floating-point operations.
- **Max Intermediate:** Lower is better. Peak memory usage during contraction.
- **TreeSA** uses simulated annealing and typically finds better paths than greedy.
- **TreeSASlicer** can reduce memory by slicing indices at the cost of more FLOPs.

## Conclusions

### Key Findings

1. **omeco(treesa) consistently outperforms opt_einsum(optimal):**
   - Heisenberg: 10^7.22 vs 10^7.52 FLOPs (**2x fewer operations**)
   - Kitaev: 10^6.79 vs 10^7.09 FLOPs (**2x fewer operations**)

2. **omeco(greedy) outperforms opt_einsum(greedy):**
   - Heisenberg: 10^7.65 vs 10^7.95 FLOPs (**2x fewer operations**)
   - Kitaev: 10^7.49 vs 10^7.79 FLOPs (**2x fewer operations**)

3. **Memory usage is comparable** between the two libraries for similar algorithms.

### Recommendation

Based on this analysis, **omeco is recommended** as the default path optimizer for the following reasons:

1. **Better path quality:** TreeSA finds paths with ~2x fewer FLOPs than opt_einsum's optimal method
2. **Fast greedy:** Even omeco's greedy method outperforms opt_einsum's greedy
3. **Rust performance:** omeco is implemented in Rust with Python bindings, providing fast optimization
4. **Active development:** omeco is a port of OMEinsumContractionOrders.jl with ongoing improvements

### Migration Path

1. Start by adding omeco as an optional optimizer (current implementation)
2. Run validation tests on production workloads
3. If results are satisfactory, consider making omeco the default
4. Eventually, opt_einsum dependency could be reduced to just the contraction execution

## Notes

- opt_einsum's `optimal` method uses dynamic programming (exponential in tensor count).
- omeco is a Rust library providing fast path optimization for tensor networks.
- These are theoretical complexity estimates; actual runtime depends on hardware.