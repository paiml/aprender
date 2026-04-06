# shannon-entropy-v1

**Version:** 1.0.0

Shannon entropy bounds for model profiling and data analysis

## References

- Shannon (1948) A Mathematical Theory of Communication
- Qwen2.5-Coder Showcase Spec §11.5 — entropy-based profiling

## Equations

### entropy

$$
H(X) = -sum(p_i * log2(p_i)) for i in alphabet
$$

**Domain:** $p_i >= 0, sum(p_i) = 1, 0 * log2(0) := 0$

**Codomain:** $H(X) \in [0, log2(|alphabet|)]$

**Invariants:**

- $H(X) >= 0 for all distributions$
- $H(X) = 0 iff X is deterministic (one p_i = 1)$
- $H(X) = log2(|alphabet|) iff X is uniform$

### uniform_entropy

$$
H_uniform(k) = log2(k)
$$

**Domain:** $k \in \mathbb{Z}, k >= 1$

**Codomain:** $H_uniform \in [0, ∞)$

**Invariants:**

- $Strictly monotonically increasing in k$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | bound | Range bound | $0 <= H(X) <= log2(256) = 8.0 for byte data$ |
| 2 | invariant | Constant input zero entropy | $H([c, c, ..., c]) = 0.0 for any constant byte c$ |
| 3 | monotonicity | Uniform entropy monotonic | $k1 < k2 => H_uniform(k1) < H_uniform(k2)$ |
| 4 | equivalence | SIMD entropy equivalence |  |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-SE-001 | Range bound | Entropy of random byte distribution in [0, 8] | Entropy calculation overflow or sign error |
| FALSIFY-SE-002 | Zero entropy | Constant input yields exactly 0.0 | Edge case in 0*log(0) convention |
| FALSIFY-SE-003 | Monotonicity | More symbols => higher uniform entropy | log2 implementation error |
| FALSIFY-SE-004 | SIMD entropy equivalence | SIMD entropy matches scalar |  compare scalar vs SIMD entropy computation:SIMD entropy calculation diverges |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-SE-001 | SE-BND-001 | 4 | bounded_int |
| KANI-SHANNO-002 | Range bound | 8 | stub_float |
| KANI-SHANNO-003 | Constant input zero entropy | 8 | exhaustive |
| KANI-SHANNO-004 | Uniform entropy monotonic | 8 | exhaustive |
| KANI-SHANNO-005 | SIMD entropy equivalence | 8 | exhaustive |

## QA Gate

**Shannon Entropy Contract** (F-SE-001)

Entropy computation quality gate

**Checks:** range_bound, zero_entropy, monotonicity

**Pass criteria:** All 4 falsification tests pass

