# performance-grading-v1

**Version:** 1.0.0

Performance grading systems for model evaluation

## References

- Qwen2.5-Coder Showcase Spec §11.6 — Ollama parity grade
- Qwen2.5-Coder Showcase Spec §11.7 — performance efficiency grade

## Equations

### concrete_instance

$$
Qwen3-8B Q4K: bw_ceiling = 33GB/s / 4.19GB \approx 7.9 tok/s
$$

**Domain:** $DDR4 desktop, specific model$

**Invariants:**

- $Concrete value within 10\% of theoretical$

### efficiency_grade

$$
eff = actual_tps / roofline_ceiling; grade = classify(eff)
$$

**Domain:** $eff \in [0, 1]$

**Codomain:** $grade \in {F, D, C, B, A}$

**Invariants:**

- $Grade boundaries: F(<10\%), D[10\%,20\%), C[20\%,40\%), B[40\%,50\%), A(>=50\%)$
- $Monotonic: higher efficiency => same or better grade$

### llamacpp_parity

$$
ratio = apr_tps / llamacpp_tps; grade = classify(ratio)
$$

**Domain:** $ratio \in [0, ∞)$

**Codomain:** $grade \in {F, D, C, B, A, A+}$

**Invariants:**

- $Same grade boundaries as ollama_parity$
- $Measured at c=1 (single request) and c=4 (concurrent)$

### ollama_parity

$$
ratio = apr_tps / ollama_tps; grade = classify(ratio)
$$

**Domain:** $ratio \in [0, ∞)$

**Codomain:** $grade \in {F, D, C, B, A, A+}$

**Invariants:**

- $Grade boundaries: F(<0.5), D[0.5,0.75), C[0.75,1.0), B[1.0,1.5), A[1.5,2.0), A+(>=2.0)$
- $Monotonic: higher ratio => same or better grade$
- $Boundaries are exhaustive and non-overlapping$

### vllm_parity

$$
ratio = apr_tps / vllm_tps; grade = classify(ratio)
$$

**Domain:** $ratio \in [0, ∞), vLLM uses AWQ INT4 (not GGUF)$

**Codomain:** $grade \in {F, D, C, B, A, A+}$

**Invariants:**

- $Same grade boundaries as ollama_parity$
- $vLLM is the ceiling for continuous batching (c>=4)$
- $Compare at c=4+ where vLLM's PagedAttention advantage matters$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Ollama grade exhaustive | $For all ratio >= 0, exactly one grade bucket matches$ |
| 2 | monotonicity | Ollama grade monotonic | $r1 > r2 => grade(r1) >= grade(r2)$ |
| 3 | monotonicity | Efficiency grade monotonic | $e1 > e2 => grade(e1) >= grade(e2)$ |
| 4 | bound | Concrete ceiling bound | $DDR4 33 GB/s, 4.19 GB model => ceiling \in [7.0, 9.0]$ |
| 5 | equivalence | SIMD grading equivalence |  |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-PG-001 | Exhaustive grading | No ratio falls between grade boundaries | Gap or overlap in grade boundaries |
| FALSIFY-PG-002 | Monotonicity | Increasing ratio never decreases grade | Grade boundaries not monotonically ordered |
| FALSIFY-PG-003 | Concrete ceiling | Calculation matches expected range | Arithmetic error in ceiling formula |
| FALSIFY-PG-004 | Efficiency grade monotonic | Higher efficiency => same or better grade | Grade function not monotonic |
| FALSIFY-PG-005 | SIMD grading equivalence | SIMD grading matches scalar | SIMD path assigns different grades |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-PG-001 | PG-INV-001 | 4 | bounded_int |
| KANI-PERFOR-002 | Ollama grade exhaustive | 8 | exhaustive |
| KANI-PERFOR-003 | Ollama grade monotonic | 8 | exhaustive |
| KANI-PERFOR-004 | Efficiency grade monotonic | 8 | exhaustive |
| KANI-PERFOR-005 | Concrete ceiling bound | 8 | stub_float |
| KANI-PERFOR-006 | SIMD grading equivalence | 8 | exhaustive |

## QA Gate

**Performance Grading Contract** (F-PG-001)

Grading system quality gate

**Checks:** ollama_exhaustive, ollama_monotonic, efficiency_monotonic, concrete_bound

**Pass criteria:** All 5 falsification tests pass

