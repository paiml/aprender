# fused-qkv-projection-v1

**Version:** 1.0.0

Fused QKV projection — concatenated weight matrix for single-matvec attention projection

## References

- Vaswani et al. (2017) Attention Is All You Need
- Whisper decoder: pre-norm transformer with separate Q/K/V weight matrices

## Dependencies

- [linear-projection-v1.yaml](linear-projection-v1.yaml.md)
- [layernorm-kernel-v1.yaml](layernorm-kernel-v1.yaml.md)

## Dependency Graph

```mermaid
graph LR
    fused_qkv_projection_v1["fused-qkv-projection-v1"] --> linear_projection_v1.yaml["linear-projection-v1.yaml"]
    fused_qkv_projection_v1["fused-qkv-projection-v1"] --> layernorm_kernel_v1.yaml["layernorm-kernel-v1.yaml"]
```

## Equations

### fused_qkv

```
Fused (1 matvec with concatenated weights):
  W_qkv = [W_q; W_k; W_v]   ∈ ℝ^{3·d_model × d_model}
  b_qkv = [b_q; b_k; b_v]   ∈ ℝ^{3·d_model}
  normed = LayerNorm(x)
  qkv = W_qkv @ normed + b_qkv    (d_model → 3·d_model)
  q = qkv[0..d_model]
  k = qkv[d_model..2·d_model]
  v = qkv[2·d_model..3·d_model]

```

**Domain:** $x \in \mathbb{R}^{d_model}$

**Codomain:** $q, k, v \in \mathbb{R}^{d_model} (sliced from \mathbb{R}^{3·d_model})$

**Invariants:**

- $W_qkv rows [0..d) = W_q rows, [d..2d) = W_k rows, [2d..3d) = W_v rows$
- $Contiguous memory layout for prefetch-friendly sequential access$

### separate_qkv

$$
Standard (3 separate matvecs):
  normed = LayerNorm(x)
  q = W_q @ normed + b_q    (d_model \to d_model)
  k = W_k @ normed + b_k    (d_model \to d_model)
  v = W_v @ normed + b_v    (d_model \to d_model)

$$

**Domain:** $x \in \mathbb{R}^{d_model}$

**Codomain:** $q, k, v \in \mathbb{R}^{d_model}$

### shared_q8_qkv

```
Shared Q8_1 activation quantization (PMAT-054A):
  normed = RMSNorm(x)                                      # same input
  q8 = Q8Quantize(normed)                                   # quantize ONCE
  q = DP4A_GEMV(W_q, q8)                                    # reuse q8
  k = DP4A_GEMV(W_k, q8)                                    # reuse q8
  v = DP4A_GEMV(W_v, q8)                                    # reuse q8

vs baseline (3 independent calls):
  q8_q = Q8Quantize(normed); q = DP4A_GEMV(W_q, q8_q)      # quantize 1
  q8_k = Q8Quantize(normed); k = DP4A_GEMV(W_k, q8_k)      # quantize 2
  q8_v = Q8Quantize(normed); v = DP4A_GEMV(W_v, q8_v)      # quantize 3

```

**Domain:** $normed \in \mathbb{R}^{M × d_model}, M \in [2, 8]$

**Codomain:** $q \in \mathbb{R}^{M × q_dim}, k, v \in \mathbb{R}^{M × kv_dim}$

**Invariants:**

- $Output identical to separate path (Q8Quantize is deterministic for same input)$
- $Saves 2 Q8Quantize kernel launches per layer (56 per token at 28 layers)$
- $Q8_1 buffer reused across all 3 GEMV — no redundant allocations$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | equivalence | Fused matches separate QKV | $\|fused_qkv(x) - separate_qkv(x)\| < \varepsilon element-wise$ |
| 2 | invariant | Output dimension correct | $len(qkv) = 3 * d_model$ |
| 3 | invariant | Weight concatenation preserves values | $W_qkv[i*d..(i+1)*d, :] = W_i for i \in {q,k,v}$ |
| 4 | invariant | Bias concatenation preserves values | $b_qkv[i*d..(i+1)*d] = b_i for i \in {q,k,v}$ |
| 5 | invariant | Single matvec call | `Exactly one tiled_matvec_f16_into call for Q+K+V combined` |
| 6 | equivalence | Shared Q8_1 matches separate quantization (PMAT-054A) | `\|shared_q8_qkv(x) - separate_qkv(x)\| = 0 (exact, no FP rounding diff)` |

## Kernel Phases

1. **weight_fusion**: At load time: concatenate W_q, W_k, W_v into contiguous W_qkv — *W_qkv layout matches concatenation spec*
2. **projection**: At inference: single matvec producing 3·d_model output, then slice — *q = qkv[0..d], k = qkv[d..2d], v = qkv[2d..3d]*

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-QKV-001 | Equivalence to separate projections | \|fused_qkv(x) - [w_q@x; w_k@x; w_v@x]\| < 1e-6 element-wise | Weight concatenation order or slicing error |
| FALSIFY-QKV-002 | Weight layout verification | W_qkv[0..d*d] = W_q.flatten(), W_qkv[d*d..2*d*d] = W_k.flatten() | Row-major vs column-major confusion in concatenation |
| FALSIFY-QKV-003 | Bias correctness | b_qkv[0..d] = b_q, b_qkv[d..2d] = b_k, b_qkv[2d..3d] = b_v | Bias vector ordering error |
| FALSIFY-QKV-004 | Whisper-specific dimensions | Correct for d_model ∈ {384, 512, 768, 1024, 1280} | Dimension-specific edge case |
| FALSIFY-QKV-005 | Shared Q8_1 equivalence (PMAT-054A) | batched_qkv_dp4a output == 3× batched_hw_dp4a_q4k_gemv_into output | Q8_1 buffer corruption between GEMV launches or workspace aliasing |
| FALSIFY-QKV-006 | Shared Q8_1 buffer not clobbered between GEMV launches | Q8_1 buffer contents identical before each of the 3 GEMV launches | GEMV kernel writes to Q8 input buffer (read-only violation) |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-QKV-001 | Fused matches separate QKV | 32 | exhaustive |
| KANI-QKV-002 | Shared Q8_1 matches separate quantization (PMAT-054A) | 32 | exhaustive |
| KANI-QKV-003 | Output dimension correct | 32 | exhaustive |
| KANI-FUSED_-004 | Weight concatenation preserves values | 8 | exhaustive |
| KANI-FUSED_-005 | Bias concatenation preserves values | 8 | exhaustive |
| KANI-FUSED_-006 | Single matvec call | 8 | exhaustive |

## QA Gate

**fused-qkv-projection-v1 Contract** (F-FQPV-001)

Quality gate for Fused QKV projection — concatenated weight matrix for single

**Checks:** validation, falsification

