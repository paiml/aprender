# Design by Contract in Realizar

**Version**: 1.0.0
**Updated**: 2026-02-25
**Scope**: realizar inference engine
**References**: Meyer (1992) *Object-Oriented Software Construction*, PMAT-235, GH-278, GH-305, GH-330

---

## Overview

Realizar enforces Design by Contract (DbC) at the model configuration boundary.
Every model -- whether loaded from GGUF, APR, or SafeTensors -- must pass through
`ValidatedModelConfig`, a Poka-Yoke newtype that guarantees 11 structural invariants
before any tensor allocation or inference computation begins.

The key insight: **invalid configs caught at construction time cannot cause OOM,
NaN propagation, or silent garbage at inference time.**

---

## ValidatedModelConfig: The Poka-Yoke Gate

`ValidatedModelConfig` wraps `GGUFConfig` with a private inner field. There is no
public constructor that bypasses validation. The only way to obtain one is through
`ValidatedModelConfig::validate(config)` or the format-specific constructors
(`from_gguf`, `from_apr`, `from_safetensors_config`), all of which call `validate()`.

```
GGUFConfig (unvalidated)
    |
    v
ValidatedModelConfig::validate()  <-- 11 invariants checked here
    |
    v
ValidatedModelConfig (guaranteed valid)
    |
    v
Inference engine (safe to allocate tensors, compute attention, etc.)
```

Source: `realizar/src/gguf/config.rs`

---

## 11 Structural Invariants

| ID | Invariant | Rationale |
|----|-----------|-----------|
| SI-001 | `hidden_dim > 0` | Zero hidden_dim means zero-size tensors |
| SI-002 | `num_layers > 0` | Zero layers means no transformer blocks |
| SI-003 | `vocab_size > 0` | Zero vocab means no embeddings |
| SI-004 | `num_heads > 0` | Zero heads means no attention |
| SI-005 | `num_kv_heads > 0` | Zero KV heads means no key-value projection |
| SI-006 | `intermediate_dim > 0` | Zero FFN dim means no feed-forward network |
| SI-007 | `hidden_dim % num_heads == 0` | head_dim must divide evenly (when not explicitly set) |
| SI-008 | `num_heads % num_kv_heads == 0` | GQA ratio must be an integer |
| SI-009 | `head_dim > 0` | Derived dimension must be positive |
| SI-010 | Upper bounds (OOM prevention) | hidden_dim <= 65536, num_layers <= 256, etc. |
| SI-011 | Range checks (NaN prevention) | rope_theta in [1.0, 1e8], eps in [1e-10, 0.01] |

---

## Three Loading Funnels, One Validation Gate

All three model format paths converge at the same `validate()` function:

```
GGUF file   --> GGUFConfig::from_gguf()          --> ValidatedModelConfig::validate()
APR file    --> GGUFConfig::from_apr()            --> ValidatedModelConfig::validate()
SafeTensors --> from_safetensors_config()          --> ValidatedModelConfig::validate()
```

This eliminates the class of bugs where one loading path silently accepts a config
that another path rejects. Every path enforces the same 11 invariants.

---

## Architecture-Specific Defaults (Contract C-02)

When GGUF/APR metadata omits optional fields, realizar falls back to
architecture-specific defaults rather than arbitrary constants:

| Field | Function | Example |
|-------|----------|---------|
| `rope_theta` | `default_rope_theta_for_architecture()` | Qwen2: 1,000,000 vs LLaMA: 10,000 |
| `eos_token_id` | `default_eos_for_architecture()` | Qwen2: 151645 vs LLaMA: 128001 |
| `bos_token_id` | `default_bos_for_architecture()` | Qwen2: 151643 vs LLaMA: 128000 |
| `eps` | `ArchConstraints::default_eps` | Architecture-specific norm epsilon |

Source of truth: `aprender/contracts/special-tokens-registry-v1.yaml`

---

## ArchConstraints: Compile-Time Architecture Behavior

`ArchConstraints::from_architecture()` maps architecture names to their
contract-defined behavior. These are compile-time constants per architecture,
auto-generated from `provable-contracts/contracts/arch-constraints-v1.yaml`
via `build.rs`.

| Field | LLaMA | Qwen2 | GPT-2 |
|-------|-------|-------|-------|
| `norm_type` | RmsNorm | RmsNorm | LayerNorm |
| `activation` | Silu | Silu | Gelu |
| `positional_encoding` | Rope | Rope | Absolute |
| `mlp_type` | SwiGlu | SwiGlu | GeluMlp |
| `weight_layout` | Linear | Linear | Conv1D |
| `has_bias` | false | true | true |
| `tied_embeddings` | false | true | true |

This replaces runtime heuristics (tensor presence checks, string matching)
with contract data. Adding a new architecture requires only a YAML entry.

---

## Metadata Bounds (OOM and NaN Prevention)

Upper-bound checks from `aprender/contracts/model-metadata-bounds-v1.yaml`
prevent corrupted or adversarial configs from causing allocation failures:

| Field | Maximum | Rationale |
|-------|---------|-----------|
| `hidden_dim` | 65,536 | Beyond this, single-layer weight matrices exceed GPU memory |
| `num_layers` | 256 | No known model exceeds 160 layers |
| `num_heads` | 256 | Matches hidden_dim / min_head_dim (256) |
| `vocab_size` | 1,000,000 | Embedding matrix at 1M * 8K = 8GB, maximum practical |
| `intermediate_dim` | 262,144 | 4 * max hidden_dim |
| `context_length` | 2,097,152 | 2M tokens (current frontier) |
| `rope_theta` | 100,000,000 | Beyond 1e8, positional encodings collapse |
| `eps` | [1e-10, 0.01] | Below 1e-10: underflow. Above 0.01: norm distortion |

---

## Running Tests

```bash
# Run all validation tests
cargo test --lib -- validate

# Run the falsification tests that verify Rust bounds match YAML contracts
cargo test --lib -- falsify_bounds
cargo test --lib -- falsify_eos
cargo test --lib -- falsify_bos

# Run ArchConstraints tests
cargo test --lib -- arch_constraints
```

---

## Cross-References

- **aprender**: `contracts/model-metadata-bounds-v1.yaml` (source of truth for bounds)
- **aprender**: `contracts/special-tokens-registry-v1.yaml` (source of truth for BOS/EOS)
- **aprender**: `contracts/model-families/*.yaml` (source of truth for architecture constraints)
- **aprender**: `docs/specifications/unified-contract-by-design.md` (stack-wide DbC spec)
- **trueno**: SIMD kernel contracts in `provable-contracts/contracts/`
- **PMAT-235**: Original Poka-Yoke newtype validation pattern
- **GH-278**: ArchConstraints replacing runtime heuristics with contract data
- **GH-305**: Explicit head_dim for models where `hidden_dim != num_heads * head_dim`
- **GH-330**: EOS token as class invariant (no hardcoded fallbacks)
