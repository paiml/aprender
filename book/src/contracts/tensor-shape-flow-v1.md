# tensor-shape-flow-v1

**Version:** 1.0.0

Pipeline shape flow — tensor shape transformations through transformer layers

## References

- Vaswani et al. (2017) Attention Is All You Need — transformer architecture
- Ainslie et al. (2023) GQA: Training Generalized Multi-Query
- Shazeer (2020) GLU Variants Improve Transformer — SwiGLU FFN

## Dependency Graph

```mermaid
graph LR
    gguf_format_safety_v1["gguf-format-safety-v1"] --> tensor_shape_flow_v1["tensor-shape-flow-v1"]
    safetensors_format_safety_v1["safetensors-format-safety-v1"] --> tensor_shape_flow_v1["tensor-shape-flow-v1"]
```

## Equations

### gqa_grouping

$$
group_size = n_h / n_kv (integer)
$$

**Domain:** `n_h, n_kv ∈ ℤ⁺, n_h % n_kv == 0`

**Invariants:**

- $n_h / n_kv is exact integer$
- $attention output dim = n_h * d_k$

### lm_head

$$
[h] @ [V, h]^T \to [V]
$$

**Domain:** $x \in \mathbb{R}^h, W_lm \in \mathbb{R}^{V×h}$

**Invariants:**

- $Output dimension = vocab_size$

### qkv_projection

```
Q = x @ W_q^T, shape: [h] @ [n_h*d_k, h]^T → [n_h*d_k]
```

**Domain:** `x ∈ ℝ^h, W_q ∈ ℝ^{n_h*d_k × h}`

**Invariants:**

- $Q output dim = n_h * d_k$
- $K output dim = n_kv * d_k$
- $V output dim = n_kv * d_k$

### residual

$$
y = x + sublayer(x)
$$

**Domain:** $x, sublayer(x) \in \mathbb{R}^h$

**Invariants:**

- $Residual connection preserves shape$

### swiglu_shape

$$
gate[d_ff, h] × up[d_ff, h] \to SiLU(gate·x) * (up·x) \to down[h, d_ff] \to [h]
$$

**Domain:** $x \in \mathbb{R}^h, gate \in \mathbb{R}^{d_ff×h}, up \in \mathbb{R}^{d_ff×h}, down \in \mathbb{R}^{h×d_ff}$

**Invariants:**

- $Gate and up project h \to d_ff$
- $Down projects d_ff \to h$
- $Output shape = input shape = [h]$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | QKV shape compatibility | $Q_dim = n_h * d_k, K_dim = n_kv * d_k, V_dim = n_kv * d_k$ |
| 2 | invariant | GQA grouping exact | `n_h % n_kv == 0` |
| 3 | invariant | Residual shape preservation | `shape(x + sublayer(x)) == shape(x)` |
| 4 | invariant | SwiGLU intermediate shape | $gate/up: [h]\to[d_ff], down: [d_ff]\to[h]$ |
| 5 | invariant | LM head output shape | `output_dim == vocab_size` |
| 6 | equivalence | SIMD shape equivalence |  |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-TSF-001 | QKV shape | Q/K/V projection output dims match config | Head dimension mismatch in projection weights |
| FALSIFY-TSF-002 | GQA grouping | n_h / n_kv is always integer for valid configs | GQA allows non-integer group sizes |
| FALSIFY-TSF-003 | Residual | Input and output shapes match at every residual point | Dimension mismatch at residual connection |
| FALSIFY-TSF-004 | SwiGLU shape | FFN gate/up expand, down contracts, preserving h | FFN dimension chain broken |
| FALSIFY-TSF-005 | LM head | Final output dim == vocab_size | LM head weight shape mismatch |
| FALSIFY-TSF-006 | SIMD shape equivalence | SIMD shape propagation matches scalar |  compare scalar vs SIMD shape flow:SIMD path computes different shapes |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-TSF-001 | TSF-INV-001 | 4 | bounded_int |
| KANI-TENSOR-002 | QKV shape compatibility | 8 | exhaustive |
| KANI-TENSOR-003 | GQA grouping exact | 8 | exhaustive |
| KANI-TENSOR-004 | Residual shape preservation | 8 | exhaustive |
| KANI-TENSOR-005 | SwiGLU intermediate shape | 8 | exhaustive |
| KANI-TENSOR-006 | LM head output shape | 8 | exhaustive |
| KANI-TENSOR-007 | SIMD shape equivalence | 8 | exhaustive |

## QA Gate

**Tensor Shape Flow Contract** (F-TSF-001)

Pipeline shape transformation quality gate

**Checks:** qkv_shape, gqa_grouping, residual, swiglu_shape, lm_head

**Pass criteria:** All 6 falsification tests pass

