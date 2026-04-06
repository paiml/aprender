# absolute-position-v1

**Version:** 1.0.0

Absolute position embeddings — learned additive positional encoding

## References

- Vaswani et al. (2017) Attention Is All You Need

## Equations

### absolute_position_add

$$
output[t] = token_embed[t] + pos_embed[t]
$$

**Domain:** $t in {0, ..., seq_len - 1}, token_embed in R^d, pos_embed in R^d$

**Codomain:** $output in R^d$

**Invariants:**

- $output.shape = token_embed.shape (shape preservation)$
- $pos_embed = 0 implies output = token_embed (additive identity)$
- $t < max_position for all valid positions$
- $output[t] is finite for finite inputs$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Shape preservation | $output.shape = token_embed.shape = (seq_len, d)$ |
| 2 | invariant | Additive identity | $pos_embed[t] = 0 implies output[t] = token_embed[t]$ |
| 3 | bound | Max position bound | $t < max_position for all positions in the input$ |
| 4 | bound | Finite output | `is_finite(token_embed[t]) and is_finite(pos_embed[t]) implies is_finite(output[t])` |

## Kernel Phases

1. **lookup_position_embedding**: Index into pos_embed table with position t — *t < max_position*
2. **elementwise_add**: Compute token_embed[t] + pos_embed[t] elementwise — *output.shape = input.shape*

## SIMD Dispatch

| Kernel | ISA | Target |
|--------|-----|--------|
| absolute_position_add | avx2 | `absolute_position_add_avx2` |
| absolute_position_add | ptx | `absolute_position_add_ptx` |
| absolute_position_add | scalar | `absolute_position_add_scalar` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-AP-001 | Shape preservation | output.shape = token_embed.shape for all inputs | Embedding addition changes tensor shape or broadcasts incorrectly |
| FALSIFY-AP-002 | Additive identity | output = token_embed when pos_embed is all zeros | Zero position embedding not acting as identity under addition |
| FALSIFY-AP-003 | Max position bound | Positions >= max_position are rejected or clamped | Out-of-bounds position index accepted without error |
| FALSIFY-AP-004 | Finite output | output is finite for all finite inputs | Float addition producing NaN or Inf from finite inputs |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-AP-001 | AP-INV-001 | 4 | stub_float |
| KANI-AP-002 | AP-INV-002 | 8 | stub_float |
| KANI-ABSOLU-003 | Shape preservation | 8 | exhaustive |
| KANI-ABSOLU-004 | Additive identity | 8 | exhaustive |
| KANI-ABSOLU-005 | Max position bound | 8 | stub_float |
| KANI-ABSOLU-006 | Finite output | 8 | exhaustive |

## QA Gate

**Absolute Position Contract** (F-AP-001)

Absolute position embeddings quality gate

**Checks:** shape_preservation, additive_identity, position_bound, finite_output

**Pass criteria:** All 4 falsification tests pass + Kani harnesses verify

