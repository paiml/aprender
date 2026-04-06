# tied-embeddings-v1

**Version:** 1.0.0

Tied embeddings — reuse embedding weight matrix as language model head projection

## References

- Press & Wolf (2017) Using the Output Embedding to Improve Language Models

## Equations

### tied_lm_head

$$
logits = x @ W_embed^T
$$

**Domain:** $x in R^{seq_len x d_model}, W_embed in R^{vocab_size x d_model}$

**Codomain:** $logits in R^{seq_len x vocab_size}$

**Invariants:**

- $logits.shape = (seq_len, vocab_size)$
- $logits = matmul(x, W_embed^T) — equivalent to explicit separate weight matmul$
- $No additional learnable parameters beyond W_embed$
- $All output elements are finite when inputs are finite$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | bound | Output shape correctness | $logits.shape = (seq_len, vocab_size) for x.shape = (seq_len, d_model)$ |
| 2 | equivalence | Equivalence to separate matmul | `tied_lm_head(x, W_embed) = matmul(x, W_separate^T) when W_separate = W_embed` |
| 3 | invariant | No extra parameters | `param_count(tied_lm_head) = 0 (reuses W_embed, adds no new weights)` |
| 4 | bound | Finite output | $x finite and W_embed finite implies logits finite$ |

## Kernel Phases

1. **transpose_embed**: Transpose embedding matrix W_embed to shape (d_model, vocab_size) — *W_embed_T.shape = (d_model, vocab_size)*
2. **matmul_logits**: Compute logits = x @ W_embed^T via matrix multiplication — *logits.shape = (seq_len, vocab_size)*

## SIMD Dispatch

| Kernel | ISA | Target |
|--------|-----|--------|
| tied_lm_head | avx2 | `tied_lm_head_avx2` |
| tied_lm_head | ptx | `tied_lm_head_ptx` |
| tied_lm_head | scalar | `tied_lm_head_scalar` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-TE-001 | Output shape correctness | logits.shape = (seq_len, vocab_size) for all valid seq_len and vocab_size | Transpose or matmul dimension mismatch |
| FALSIFY-TE-002 | Equivalence to separate matmul | tied_lm_head(x, W) = matmul(x, W_copy^T) bit-for-bit when W_copy = W.clone() | Tied path uses different memory layout causing numerical divergence |
| FALSIFY-TE-003 | No extra parameters | tied_lm_head introduces zero additional learnable parameters | Implementation allocates a separate projection weight |
| FALSIFY-TE-004 | Finite output | All logits elements are finite when x and W_embed are finite | Accumulation overflow in matmul for large d_model |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-TE-001 | TE-SHP-001 | 4 | stub_float |
| KANI-TE-002 | TE-EQV-001 | 4 | stub_float |
| KANI-TIED_E-003 | Output shape correctness | 8 | exhaustive |
| KANI-TIED_E-004 | Equivalence to separate matmul | 8 | exhaustive |
| KANI-TIED_E-005 | No extra parameters | 8 | exhaustive |
| KANI-TIED_E-006 | Finite output | 8 | exhaustive |

## QA Gate

**Tied Embeddings Contract** (F-TE-001)

Weight-tied language model head projection quality gate

**Checks:** output_shape, matmul_equivalence, no_extra_parameters, finite_output

**Pass criteria:** All 4 falsification tests pass + Kani harnesses verify

