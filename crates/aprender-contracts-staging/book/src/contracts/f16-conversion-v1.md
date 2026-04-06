# f16-conversion-v1

**Version:** 1.0.0

IEEE 754 half-precision (F16) to single-precision (F32) conversion invariants

## References

- IEEE 754-2008 — Binary floating-point arithmetic
- Qwen2.5-Coder Showcase Spec §11.5 — F16 passthrough

## Dependency Graph

```mermaid
graph LR
    fp8_interchange_v1["fp8-interchange-v1"] --> f16_conversion_v1["f16-conversion-v1"]
```

## Equations

### f16_to_f32_bias

$$
f32_bits = (sign << 31) | ((exp_f16 + 112) << 23) | (mantissa << 13)
$$

**Domain:** $exp_f16 \in [1, 30] (normal f16), mantissa \in [0, 1023]$

**Codomain:** $valid f32 bit pattern$

**Invariants:**

- `Sign preserved: sign(f32) == sign(f16)`
- $Exponent bias shift: e_f32 = e_f16 + 112 (bias 127 - bias 15)$
- $Mantissa zero-padded: lower 13 bits of f32 mantissa are 0$

### roundtrip

```
f32_to_f16(f16_to_f32(h)) == h
```

**Domain:** $h \in normal f16 values (exp \in [1, 30])$

**Codomain:** $identity mapping$

**Invariants:**

- $Exact roundtrip for all normal f16 values$
- $Subnormals may lose precision (not covered)$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | equivalence | Bias trick correctness | `f16_to_f32 via bit manipulation == f16_to_f32 via arithmetic conversion` |
| 2 | invariant | Roundtrip identity | `f32_to_f16(f16_to_f32(h)) == h for normal f16` |
| 3 | invariant | Sign preservation | `sign(f16_to_f32(h)) == sign(h)` |
| 4 | equivalence | SIMD conversion equivalence |  |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-F16-001 | Bias trick | Bit manipulation matches Rust's f32::from(f16) | Exponent bias constant wrong (should be 112) |
| FALSIFY-F16-002 | Roundtrip | f16→f32→f16 is identity for normal values | Precision loss in f32→f16 truncation |
| FALSIFY-F16-003 | Sign preservation | Negative f16 maps to negative f32 | Sign bit position error |
| FALSIFY-F16-004 | SIMD conversion equivalence | SIMD f16 conversion matches scalar |  compare scalar vs SIMD f16 conversion:SIMD conversion rounding differs |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-F16-001 | F16-INV-001 | 5 | bounded_int |
| KANI-F16_CO-002 | Bias trick correctness | 8 | exhaustive |
| KANI-F16_CO-003 | Roundtrip identity | 8 | exhaustive |
| KANI-F16_CO-004 | Sign preservation | 8 | exhaustive |
| KANI-F16_CO-005 | SIMD conversion equivalence | 8 | exhaustive |

## QA Gate

**F16 Conversion Contract** (F-F16-001)

IEEE 754 half-precision conversion quality gate

**Checks:** bias_trick_correct, roundtrip_identity, sign_preserved

**Pass criteria:** All 4 falsification tests pass

