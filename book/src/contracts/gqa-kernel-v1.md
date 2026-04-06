# gqa-kernel-v1

**Version:** 1.0.0

GQA kernel — grouped query attention with KV head broadcasting

## References

- Ainslie et al. (2023) GQA: Training Generalized MQT Models
- Vaswani et al. (2017) Attention Is All You Need

## Dependencies

- [softmax-kernel-v1](softmax-kernel-v1.md)
- [matmul-kernel-v1](matmul-kernel-v1.md)
- [attention-kernel-v1](attention-kernel-v1.md)

## Dependency Graph

```mermaid
graph LR
    gqa_kernel_v1["gqa-kernel-v1"] --> softmax_kernel_v1["softmax-kernel-v1"]
    gqa_kernel_v1["gqa-kernel-v1"] --> matmul_kernel_v1["matmul-kernel-v1"]
    gqa_kernel_v1["gqa-kernel-v1"] --> attention_kernel_v1["attention-kernel-v1"]
```

## Equations

### gqa

$$
GQA(Q, K, V) = softmax(Q_g * K_h^T / \sqrt{d_k}) * V_h
$$

**Domain:** `Q in R^{n x d}, K in R^{s x d}, V in R^{s x d_v}, num_heads > 0, num_kv_heads > 0`

**Codomain:** $GQA(Q, K, V) in R^{n x d}$

**Invariants:**

- $Attention weights sum to 1 per query position (normalization)$
- $Output is convex combination of V rows per head$
- `GQA(kv_heads=num_heads) = standard MHA`
- `num_heads must be divisible by num_kv_heads`

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | subcontract | GQA refines standard MHA — accepts same Q/K/V shapes, produces compatible output | `pre(MHA) → pre(GQA) ∧ post(GQA) → post(MHA) when n_kv_heads = n_heads` |
| 2 | invariant | Attention weight normalization | $\|sum(attn_weights[i, :]) - 1.0\| < eps per query position i$ |
| 3 | equivalence | GQA degenerates to MHA | `GQA(kv_heads=num_heads) == MHA(Q, K, V) within tolerance` |
| 4 | bound | Output is convex combination of V | $min(V) <= output_i <= max(V) per head$ |
| 5 | invariant | KV head broadcasting correctness | `Q heads [g*r..(g+1)*r] share K_g, V_g where r = num_heads/num_kv_heads` |
| 6 | equivalence | SIMD matches scalar within ULP |  |
| 7 | equivalence | GPU PTX matches CPU within cosine >= 0.98 | $cosine(gqa_ptx(Q,K,V), gqa_cpu(Q,K,V)) >= 0.98$ |
| 8 | invariant | GPU head mapping for non-power-of-2 ratios | `kv_head_idx(q) == q * num_kv_heads / num_heads for all q in [0..num_heads)` |

## Kernel Phases

1. **kv_broadcast**: Broadcast KV heads to match query head groups — *Each KV head serves exactly num_heads/num_kv_heads query heads*
2. **qk_matmul**: Compute Q * K^T / sqrt(d_k) per head — *Score matrix is n x s per head*
3. **attention_softmax**: Apply softmax to scores per query position — *Weights sum to 1 per row*
4. **weighted_sum**: Compute weighted sum of V rows — *Output is convex combination of V*

## SIMD Dispatch

| Kernel | ISA | Target |
|--------|-----|--------|
| gqa | avx2 | `gqa_avx2` |
| gqa | ptx | `gqa_ptx` |
| gqa | scalar | `gqa_scalar` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-GQ-001 | Weight normalization | sum(attn_weights[i, :]) = 1.0 for all query positions i | Softmax applied along wrong dimension or missing |
| FALSIFY-GQ-002 | MHA degeneration | \|GQA(kv=h) - MHA(Q,K,V)\| < 1e-6 when kv_heads=num_heads | KV broadcasting logic incorrect when groups=1 |
| FALSIFY-GQ-003 | Convex combination bound | min(V) <= output_i <= max(V) per head | Attention weights not properly normalized |
| FALSIFY-GQ-004 | Head divisibility | num_heads % num_kv_heads == 0 enforced at construction | Missing divisibility check |
| FALSIFY-GQ-005 | SIMD equivalence | \|gqa_avx2(Q,K,V) - gqa_scalar(Q,K,V)\| < 8 ULP | SIMD matmul accumulation order differs |
| FALSIFY-GQ-006 | Boundary - single KV head (MQA) | GQA(kv_heads=1) broadcasts single KV to all query heads | Multi-query degenerate case not handled |
| FALSIFY-GQ-007 | GPU parity - non-power-of-2 head ratio | cosine(gqa_gpu, gqa_cpu) >= 0.98 for heads=28, kv=4 (ratio=7) | GPU PTX kernel computes wrong function for this head configuration |
| FALSIFY-GQ-008 | GPU parity - power-of-2 head ratio | cosine(gqa_gpu, gqa_cpu) >= 0.98 for heads=32, kv=8 (ratio=4) | GPU PTX kernel fails even for power-of-2 ratios |
| FALSIFY-GQ-009 | GPU head mapping correctness | kv_head_idx(q) == q * num_kv_heads / num_heads for all q | PTX integer division produces wrong kv_head_idx on target GPU |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-GQ-001 | GQ-INV-001 | 4 | stub_float |
| KANI-GQ-002 | GQ-EQV-001 | 4 | stub_float |
| KANI-GQ-003 | GQ-BND-001 | 4 | stub_float |
| KANI-GQA_KE-004 | GQA refines standard MHA — accepts same Q/K/V shapes, produces compatible output | 8 | exhaustive |
| KANI-GQA_KE-005 | Attention weight normalization | 8 | exhaustive |
| KANI-GQA_KE-006 | GQA degenerates to MHA | 8 | exhaustive |
| KANI-GQA_KE-007 | Output is convex combination of V | 8 | exhaustive |
| KANI-GQA_KE-008 | KV head broadcasting correctness | 8 | exhaustive |
| KANI-GQA_KE-009 | SIMD matches scalar within ULP | 8 | exhaustive |
| KANI-GQA_KE-010 | GPU PTX matches CPU within cosine >= 0.98 | 8 | exhaustive |
| KANI-GQA_KE-011 | GPU head mapping for non-power-of-2 ratios | 8 | exhaustive |

## QA Gate

**GQA Contract** (F-GQ-001)

Grouped query attention with KV broadcasting quality gate

**Checks:** weight_normalization, mha_equivalence, convex_bound, simd_equivalence

**Pass criteria:** All 9 falsification tests pass + Kani harnesses verify

