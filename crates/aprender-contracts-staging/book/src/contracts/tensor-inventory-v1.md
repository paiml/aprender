# tensor-inventory-v1

**Version:** 1.0.0

Tensor inventory algebra and parameter count decomposition

## References

- Qwen3 Performance Parity Spec — tensor counting
- Vaswani et al. (2017) Attention Is All You Need — parameter analysis

## Equations

### architecture_delta

```
delta = L * (per_layer_B - per_layer_A)
```

**Domain:** $Two model configs A, B with same L$

**Codomain:** $delta \in \mathbb{Z} (signed)$

**Invariants:**

- $delta = 0 when architectures identical$
- $delta proportional to L$

### parameter_decomposition

$$
total_params = embed_params + sum(layer_params) + head_params
$$

**Domain:** $All tensor element counts$

**Codomain:** $total_params \in \mathbb{Z}^{+}$

**Invariants:**

- $Sum of parts equals whole$
- $Each component non-negative$

### quantization_bytes

```
bytes = params * block_bytes / elements_per_block
```

**Domain:** `params ∈ ℤ⁺, block_bytes/elements_per_block from quantization scheme`

**Codomain:** $bytes \in \mathbb{Z}^{+}$

**Invariants:**

- $bytes proportional to params$
- $bytes decreases with more aggressive quantization$

### tensor_count

$$
total = base + L * per_layer
$$

**Domain:** $base \in {2, 3} (embed + lm_head ± final_norm), per_layer = 7 + 2*norm + 2*qk_norm + 3*bias$

**Codomain:** $total \in \mathbb{Z}^{+}$

**Invariants:**

- $total > 0 for any valid config$
- $Linear in L (layer count)$

### tied_embeddings

$$
tied=true => tensor_count -= 1, but params unchanged (shared storage)
$$

**Domain:** $Boolean flag$

**Invariants:**

- $Tied reduces tensor count by exactly 1$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Tensor count formula | $total = base + L * per_layer for valid configs$ |
| 2 | invariant | Architecture delta linear | `delta(A,B) = L * (per_layer_B - per_layer_A)` |
| 3 | invariant | Parameter decomposition exact | $sum of component params = total_params$ |
| 4 | invariant | Tied embedding count | $tied => count(untied) - count(tied) = 1$ |
| 5 | monotonicity | Quantization byte ordering | $Q4K < Q6K < Q8 < F16 < F32 bytes for same params$ |
| 6 | equivalence | SIMD inventory equivalence |  |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-TI-001 | Tensor count formula | Formula matches hand-counted tensor inventory | Missing tensor in per_layer formula |
| FALSIFY-TI-002 | Architecture delta | Delta proportional to L | Non-linear component in delta |
| FALSIFY-TI-003 | Quantization ordering | More aggressive quant => fewer bytes | Quant block size formula wrong |
| FALSIFY-TI-004 | Parameter decomposition exact | Sum of per-layer params == total params | Missing tensor in decomposition |
| FALSIFY-TI-005 | Tied embedding count | Tied weights counted exactly once | Tied weights double-counted or missing |
| FALSIFY-TI-006 | SIMD inventory equivalence | SIMD tensor shapes match scalar | SIMD layout differs from scalar |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-TI-001 | TI-INV-001 | 4 | bounded_int |
| KANI-TENSOR-002 | Tensor count formula | 8 | exhaustive |
| KANI-TENSOR-003 | Architecture delta linear | 8 | exhaustive |
| KANI-TENSOR-004 | Parameter decomposition exact | 8 | exhaustive |
| KANI-TENSOR-005 | Tied embedding count | 8 | exhaustive |
| KANI-TENSOR-006 | Quantization byte ordering | 8 | exhaustive |
| KANI-TENSOR-007 | SIMD inventory equivalence | 8 | exhaustive |

## QA Gate

**Tensor Inventory Contract** (F-TI-001)

Tensor counting and parameter decomposition quality gate

**Checks:** tensor_count_formula, architecture_delta, parameter_decomposition, tied_embeddings, quantization_ordering

**Pass criteria:** All 6 falsification tests pass

