# Design by Contract in Aprender

## Overview

Aprender is the hub of the Sovereign AI Stack's Design by Contract (DbC) system.
It owns the YAML contract definitions in `contracts/` that every downstream crate
(trueno, realizar, entrenar, batuta) enforces at their respective layer.

The system implements **Meyer's Design by Contract** (1992) fused with **Popperian
falsificationism**: every contract is expressed as a falsifiable property, and every
falsification test attempts to **disprove** correctness rather than confirm it.

| Mechanism | Definition | Rust Equivalent |
|-----------|-----------|-----------------|
| **Precondition** | Requirements caller must satisfy | Argument validation, `Result::Err` on violation |
| **Postcondition** | Properties ensured on return | Return type guarantees, `debug_assert!` on output |
| **Class Invariant** | Property of all instances | Valid-by-construction structs, newtypes |

## Contract Inventory

Ten YAML contracts live in `contracts/`:

| Contract | Purpose | Enforcement |
|----------|---------|-------------|
| `tensor-layout-v1.yaml` | Row-major layout, transpose rules | `src/format/layout_contract.rs` |
| `special-tokens-registry-v1.yaml` | BOS/EOS/PAD per model family | `src/format/special_tokens_contract_falsify.rs` |
| `model-metadata-bounds-v1.yaml` | Upper bounds on config fields | `realizar::ValidatedModelConfig` |
| `chat-template-semantics-v1.yaml` | Chat template format contracts | `src/text/chat_template.rs` |
| `tokenizer-vocab-v1.yaml` | Tokenizer vocabulary contracts | `src/text/` |
| `classification-finetune-v1.yaml` | Fine-tuning contracts | `entrenar` |
| `kernel-fusion-v1.yaml` | Kernel fusion contracts | `trueno` |
| `layer-parity-v1.yaml` | Layer equivalence verification | Cross-stack |
| `quantized-dot-product-v1.yaml` | Quantized computation contracts | `trueno` |
| `model-families/*.yaml` | 17 architecture-specific contracts | `realizar::ArchConstraints` |

## Four-Phase Roadmap (All Complete)

1. **Phase 1 — Contract Inventory** (638 FALSIFY tests identified)
2. **Phase 2 — Falsification Sweep** (110 findings across 7 repos)
3. **Phase 3 — Fix Critical Findings** (19 findings C-01..C-11, N-01..N-08 resolved)
4. **Phase 4 — Continuous Enforcement** (CI gates, proptest contracts)

## Findings Summary

### Critical (C-01 through C-11)

| ID | Finding | Resolution |
|----|---------|------------|
| C-01 | Inconsistent default architecture ("llama" vs "qwen2") | Explicit arch required |
| C-02 | Inconsistent rope_theta (10K vs 1M) | Architecture-specific defaults |
| C-03 | Qwen2-1.5B dims used as universal defaults | Required fields, no defaults |
| C-04..C-09 | Various tensor/shape/import issues | Contract enforcement at boundary |
| C-10 | vocab_size silently defaulted | `ok_or_else` in entrenar |
| C-11 | Dimensional fields silently defaulted | `ok_or_else` in entrenar |

### Notable (N-01 through N-08)

| ID | Finding | Resolution |
|----|---------|------------|
| N-05 | `estimate_num_heads` returns 0 for unknown | Documented precondition |
| N-06 | Leaderboard sort with missing scores | NEG_INFINITY/INFINITY semantics |
| N-07 | Missing optional fields silent | eprintln warnings added |

## Cross-Repo Enforcement Map

```
contracts/*.yaml           ← Source of truth (this repo)
    │
    ├── trueno/contracts.rs        ← Kernel-level: buffer sizes, GEMV shapes
    ├── realizar/config.rs         ← Load-time: ValidatedModelConfig, ArchConstraints
    ├── entrenar/loader.rs         ← Train-time: required fields (C-10/C-11)
    ├── batuta/bug_hunter/         ← Audit: contract gap analysis
    └── provable-contracts/        ← Formal: YAML→Kani proofs
```

## Key Rust APIs

### Tensor Layout Contract

```rust
use aprender::format::layout_contract::{LayoutContract, TensorContract, block_sizes};

let contract = LayoutContract::new();

// Should this tensor be transposed during GGUF→APR import?
assert!(contract.should_transpose_gguf("output.weight"));

// Validate APR tensor shape
contract.validate_apr_shape("lm_head.weight", &[151936, 896], 151936, 896)?;

// Calculate expected bytes for quantized tensor
let bytes = LayoutContract::calculate_q4k_bytes(4096, 4096);
```

### Block Size Constants

```rust
use aprender::format::layout_contract::block_sizes;

assert_eq!(block_sizes::Q4_K, 144);  // bytes per super-block
assert_eq!(block_sizes::Q6_K, 210);
assert_eq!(block_sizes::QK_K, 256);  // elements per super-block
```

## Running Falsification Tests

```bash
# All falsification tests
cargo test --lib -- falsify

# Layout contract tests
cargo test --lib -- layout_contract

# Special tokens contract
cargo test --lib -- special_tokens_contract
```

## Example: Qwen3.5 Contract-Driven Architecture Support (GH-278)

The `contracts/model-families/qwen3_5.yaml` contract demonstrates how DbC drives
new architecture support end-to-end:

1. **`has_bias=false`** in constraints skips bias tensor loading — no wasted memory
2. **`head_dim=256`** flows through `explicit_head_dim` to RoPE computation
3. **`layer_types`** (from `config.json`) routes each layer to the correct
   attention kernel: softmax or Gated Delta Net
4. **Shape templates** (`q_proj: [num_heads * head_dim, hidden_dim]`) drive
   tensor validation at import time

The contract is protected by 8 falsification tests (`FALSIFY-MF-QWEN35-001..008`)
that attempt to break each invariant. Run with:

```bash
cargo test -- falsify_mf_qwen35
```

See also: `cargo run --example design_by_contract` for a runnable demo.

## References

- **Full specification**: `docs/specifications/enforce-provable-DbC.md`
- **Unified architecture**: `docs/specifications/unified-contract-by-design.md`
- [trueno DbC](../../trueno/docs/design-by-contract.md)
- [realizar DbC](../../realizar/docs/design-by-contract.md)
- [entrenar DbC](../../entrenar/docs/design-by-contract.md)

## Theoretical Foundations

- Meyer, B. (1992). *Applying Design by Contract*. IEEE Computer.
- Popper, K. (1959). *The Logic of Scientific Discovery*. Routledge.
- Shingo, S. (1986). *Zero Quality Control: Source Inspection and the Poka-Yoke System*. Productivity Press.
