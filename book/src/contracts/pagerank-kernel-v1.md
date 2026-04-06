# pagerank-kernel-v1

**Version:** 1.0.0

PageRank kernel — power iteration for stationary distribution

## References

- Brin & Page (1998) The Anatomy of a Large-Scale Hypertextual Web Search Engine

## Equations

### pagerank

$$
r = d * M * r + (1-d)/N * 1
$$

**Domain:** $M in R^{N x N} (column-stochastic), d in (0,1), N >= 1$

**Codomain:** $r in R^N, r_i >= 0, sum(r) = 1$

**Invariants:**

- $r_i >= 0 for all i (non-negativity)$
- $sum(r) = 1 (probability distribution)$
- $r is left eigenvector of Google matrix G = d*M + (1-d)/N * 1*1^T$

### power_iteration

$$
r_{t+1} = G * r_t, converges when ||r_{t+1} - r_t|| < eps
$$

**Domain:** $r_0 = 1/N * 1 (uniform), G column-stochastic$

**Codomain:** $r_t -> r* (stationary distribution)$

**Invariants:**

- $sum(r_t) = 1 at every iteration$
- $||r_{t+1} - r*|| <= d * ||r_t - r*|| (linear convergence)$
- $Convergence rate bounded by damping factor d$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Probability distribution | $\|sum(r) - 1.0\| < eps and r_i >= 0 for all i$ |
| 2 | monotonicity | Convergence | $\|\|r_{t+1} - r*\|\| <= \|\|r_t - r*\|\| (contraction)$ |
| 3 | bound | Scores non-negative | $r_i >= 0 for all i at every iteration$ |
| 4 | invariant | Normalization preserved per iteration | $\|sum(r_t) - 1.0\| < eps at every iteration t$ |
| 5 | equivalence | SIMD matches scalar within ULP |  |

## Kernel Phases

1. **build_transition**: Normalize adjacency matrix to column-stochastic M — *Each column sums to 1 (or handled as dangling)*
2. **initialize**: Set r_0 = uniform distribution — *sum(r_0) = 1*
3. **iterate**: r_{t+1} = d * M * r_t + (1-d)/N — *sum(r_{t+1}) = 1*
4. **check_convergence**: Check ||r_{t+1} - r_t||_1 < eps — *L1 norm decreases monotonically*

## SIMD Dispatch

| Kernel | ISA | Target |
|--------|-----|--------|
| pagerank_iterate | avx2 | `pagerank_iterate_avx2` |
| pagerank_iterate | ptx | `pagerank_ptx` |
| pagerank_iterate | scalar | `pagerank_iterate_scalar` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-PR-001 | Probability distribution | sum(r) = 1 and r_i >= 0 after convergence | Normalization not maintained during iteration |
| FALSIFY-PR-002 | Convergence | \|\|r_{t+1} - r_t\|\|_1 decreases each iteration | Transition matrix not properly stochastic |
| FALSIFY-PR-003 | Non-negativity | r_i >= 0 at every iteration | Negative values from floating-point error |
| FALSIFY-PR-004 | SIMD equivalence | \|iterate_avx2(r) - iterate_scalar(r)\| < 8 ULP | SIMD matrix-vector product accumulation differs |
| FALSIFY-PR-005 | Boundary - single node | PageRank of single-node graph is [1.0] | Edge case in graph construction |
| FALSIFY-PR-006 | Uniform graph symmetry | Complete graph gives uniform PageRank r_i = 1/N | Damping factor or normalization incorrect |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-PR-001 | PR-INV-001 | 4 | stub_float |
| KANI-PR-002 | PR-BND-001 | 4 | stub_float |
| KANI-PAGERA-003 | Probability distribution | 8 | exhaustive |
| KANI-PAGERA-004 | Convergence | 8 | exhaustive |
| KANI-PAGERA-005 | Scores non-negative | 8 | exhaustive |
| KANI-PAGERA-006 | Normalization preserved per iteration | 8 | exhaustive |
| KANI-PAGERA-007 | SIMD matches scalar within ULP | 8 | exhaustive |

## QA Gate

**PageRank Contract** (F-PR-001)

Power iteration PageRank quality gate

**Checks:** probability_distribution, convergence, non_negativity, simd_equivalence

**Pass criteria:** All 6 falsification tests pass + Kani harnesses verify

