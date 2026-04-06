# quantization-ordering-v1

**Version:** 1.0.0

Quantization size ordering and LoRA alpha scaling

## References

- Dettmers et al. (2023) QLoRA: Efficient Finetuning of Quantized LLMs
- GGML quantization format documentation
- Qwen3.5 Fine-Tune Spec Phase 3

## Equations

### alpha_scaling

$$
lora_output = (alpha / rank) * (A @ B @ x)
$$

**Domain:** $alpha \in \mathbb{R}^{+}, rank \in \mathbb{Z}^{+}$

**Invariants:**

- $Scale factor = alpha / rank$
- $Standard: alpha=16, rank=64 => scale=0.25$

### bytes_per_param

$$
Q4K\approx0.5625, Q6K\approx0.8125, Q8_0\approx1.0625, F16=2.0, F32=4.0 bytes/param
$$

**Domain:** $Standard GGML quantization schemes$

**Invariants:**

- $Q4K: 18 bytes per 32-element block (scales + quants)$
- $Q6K: 26 bytes per 32-element block$
- $Q8_0: 34 bytes per 32-element block$

### dropout_expectation

$$
E[mask_i] = 1 - p
$$

**Domain:** $p \in [0, 1), Bernoulli mask$

**Invariants:**

- $Mean of mask converges to 1-p$
- $Inference: p=0 (no dropout)$

### size_ordering

$$
size(Q4K) < size(Q6K) < size(Q8_0) < size(F16) < size(F32)
$$

**Domain:** $Same parameter count, different quantization$

**Invariants:**

- $Strict ordering for any non-zero parameter count$
- $Ratios approximately: 1 : 1.5 : 2 : 4 : 8$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | monotonicity | Size ordering strict | $Q4K < Q6K < Q8_0 < F16 < F32 bytes for same param count$ |
| 2 | invariant | Alpha scaling correctness | $output scaled by exactly alpha/rank$ |
| 3 | invariant | Dropout expectation | $E[mask] = 1 - p within statistical tolerance$ |
| 4 | bound | Concrete Qwen3.5 sizes | $9B params: Q4K~5GB, Q6K~7GB, Q8~9GB, F16~18GB (within 20\%)$ |
| 5 | equivalence | SIMD quantization equivalence |  |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-QO-001 | Size ordering | Bytes strictly increase Q4K < Q6K < Q8 < F16 < F32 | Block size formula wrong |
| FALSIFY-QO-002 | Alpha scaling | alpha/rank matches expected ratio | Scaling formula error |
| FALSIFY-QO-003 | Concrete sizes | 9B param model sizes within 20% of expected | Block size constants wrong |
| FALSIFY-QO-004 | Dropout expectation | Expected output unchanged after dropout scaling | Dropout scaling factor wrong |
| FALSIFY-QO-005 | SIMD quantization equivalence | SIMD quantization matches scalar | SIMD rounding differs from scalar |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-QO-001 | QO-MON-001 | 4 | bounded_int |
| KANI-QUANTI-002 | Size ordering strict | 8 | exhaustive |
| KANI-QUANTI-003 | Alpha scaling correctness | 8 | exhaustive |
| KANI-QUANTI-004 | Dropout expectation | 8 | exhaustive |
| KANI-QUANTI-005 | Concrete Qwen3.5 sizes | 8 | exhaustive |
| KANI-QUANTI-006 | SIMD quantization equivalence | 8 | exhaustive |

## QA Gate

**Quantization Ordering Contract** (F-QO-001)

Quantization size and LoRA scaling quality gate

**Checks:** size_ordering, alpha_scaling, dropout_expectation, concrete_sizes

**Pass criteria:** All 5 falsification tests pass

