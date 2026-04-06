# special-tokens-registry-v1

**Version:** 1.0.0

Canonical EOS/BOS/PAD token IDs per model family

## References

- PROVABLE_CONTRACTS_ANALYSIS.md Gap 1
- realizar/src/gguf/config.rs default_eos_for_architecture()
- realizar/src/gguf/config.rs default_bos_for_architecture()

## Equations

### token_bounds

$$
\forall family F: eos_id(F) < vocab_size(F) ∧ (bos_id(F) = null ∨ bos_id(F) < vocab_size(F))
$$

**Domain:** $F \in { qwen2, qwen3, qwen3_moe, qwen3_5, llama, mistral, gemma, deepseek, phi2, phi3, gpt2 }$

**Codomain:** $bool (all token IDs within vocab bounds)$

**Invariants:**

- `eos_token_id < vocab_size for every family`
- `bos_token_id < vocab_size when not null`
- `pad_token_id < vocab_size when not null`
- $All additional_eos entries < vocab_size$
- $Every architecture key maps to a valid family$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | bound | EOS token ID within vocab for every family | `∀ F: eos_token_id(F) < vocab_size(F)` |
| 2 | bound | BOS token ID within vocab when not null | `∀ F: bos_token_id(F) = null ∨ bos_token_id(F) < vocab_size(F)` |
| 3 | bound | PAD token ID within vocab when not null | `∀ F: pad_token_id(F) = null ∨ pad_token_id(F) < vocab_size(F)` |
| 4 | completeness | Architecture mapping covers all families | $\forall key \in architecture_mapping: mapping(key) \in families$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-STOK-001 | EOS bounds | All families have eos_token_id < vocab_size | EOS ID exceeds vocab — will cause index-out-of-bounds in embedding lookup |
| FALSIFY-STOK-002 | Architecture coverage | Every architecture_mapping value resolves to a defined family | Unknown family in mapping — will cause lookup miss at runtime |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-SPECIA-001 | EOS token ID within vocab for every family | 8 | exhaustive |
| KANI-SPECIA-002 | BOS token ID within vocab when not null | 8 | exhaustive |
| KANI-SPECIA-003 | PAD token ID within vocab when not null | 8 | exhaustive |
| KANI-SPECIA-004 | Architecture mapping covers all families | 8 | exhaustive |

## QA Gate

**special-tokens-registry-v1 Contract** (F-STRV-001)

Quality gate for Canonical EOS/BOS/PAD token IDs per model family

**Checks:** validation, falsification

