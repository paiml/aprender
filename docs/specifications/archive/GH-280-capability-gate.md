---
title: "GH-280: Kernel Capability Gate — Contract-Driven GPU Admission Control"
issue: GH-280
status: Draft
created: 2026-02-17
updated: 2026-02-17
depends_on: [GH-279, GH-277]
---

# GH-280: Kernel Capability Gate

**Goal:** Prevent GPU garbage output by enforcing capability matching between model contracts and inference backends at load time — before any forward pass occurs.

## Five-Whys Root Cause Analysis

### Incident: Qwen3-8B GPU produces garbage (GH-279-3)

| Why | Finding |
|-----|---------|
| **Why 1** | GPU golden output is garbage (`_increaseoyaย่อม Sil`), CPU output is coherent |
| **Why 2** | GPU kernel (`forward_all_layers_gpu_to_logits_graphed`) does not apply QK RMSNorm; CPU path does |
| **Why 3** | The CUDA kernel was written for Qwen2 (no QK norm). Qwen3 added QK norm but the kernel was never updated |
| **Why 4** | No mechanism checks "does this kernel support all operations this model requires?" at load time |
| **Why 5 (root)** | **Contracts enforce structural invariants (tensor shape, dtype, name) but not behavioral invariants (does the kernel apply the correct mathematical operations?).** The contract says "this model HAS QK norm tensors" but nothing says "you MUST USE them." |

### The Gap

```
CURRENTLY ENFORCED (structural):
  ✓ Tensor exists with correct name (model.layers.{n}.self_attn.q_norm.weight)
  ✓ Tensor has correct shape [head_dim]
  ✓ Tensor has correct dtype
  ✓ Metadata keys present (rope_theta, rms_norm_eps, etc.)
  ✓ Final logit parity (parity gate — but runs AFTER garbage is produced)

NOT ENFORCED (behavioral):
  ✗ Kernel applies q_norm before attention score computation
  ✗ Kernel uses correct rope_theta (not default 10000)
  ✗ Kernel respects GQA grouping ratio
  ✗ Kernel supports Conv1D weight layout (GPT-2)
```

The parity gate catches the *symptom* (different outputs) but cannot diagnose the *cause* (which operation was skipped). It runs AFTER model load, meaning the user has already waited for GPU memory allocation, weight upload, and a forward pass before learning the model can't be served.

## Design: Required Operations

Every model architecture requires a set of computational operations. These are derivable from the model family contract:

| Contract Field | Required Operation | Enum Variant |
|---|---|---|
| `constraints.qk_norm: "true"` | Per-head QK RMSNorm | `RequiredOp::QkNorm` |
| `constraints.positional_encoding: rope` | Rotary position embedding | `RequiredOp::RoPE` |
| `constraints.attention_type: gqa` | Grouped-query attention | `RequiredOp::GQA` |
| `constraints.attention_type: mha` | Multi-head attention | `RequiredOp::MHA` |
| `constraints.mlp_type: swiglu` | SwiGLU FFN (gate + up + silu) | `RequiredOp::SwiGLU` |
| `constraints.mlp_type: gelu_mlp` | GELU FFN (up + gelu + down) | `RequiredOp::GeluMlp` |
| `constraints.norm_type: rmsnorm` | RMS normalization | `RequiredOp::RMSNorm` |
| `constraints.norm_type: layernorm` | Layer normalization | `RequiredOp::LayerNorm` |
| `constraints.has_bias: "true"` | Bias addition in attention/FFN | `RequiredOp::BiasAdd` |
| `constraints.positional_encoding: absolute` | Absolute position embedding | `RequiredOp::AbsolutePos` |
| `tensor_template.per_layer.causal_mask` (non-null) | Static causal mask | `RequiredOp::CausalMask` |

### Worked Example: Qwen3-8B

From `contracts/model-families/qwen3.yaml`:

```yaml
constraints:
  attention_type: gqa        # → RequiredOp::GQA
  activation: silu           # → RequiredOp::SwiGLU (via mlp_type)
  norm_type: rmsnorm         # → RequiredOp::RMSNorm
  has_bias: "false"          # no BiasAdd required
  positional_encoding: rope  # → RequiredOp::RoPE
  mlp_type: swiglu           # → RequiredOp::SwiGLU
  qk_norm: "true"            # → RequiredOp::QkNorm
```

Required ops: `{RoPE, GQA, SwiGLU, RMSNorm, QkNorm}`

### Worked Example: GPT-2

From `contracts/model-families/gpt2.yaml`:

```yaml
constraints:
  attention_type: mha        # → RequiredOp::MHA
  activation: gelu           # → RequiredOp::GeluMlp (via mlp_type)
  norm_type: layernorm       # → RequiredOp::LayerNorm
  has_bias: "true"           # → RequiredOp::BiasAdd
  positional_encoding: absolute # → RequiredOp::AbsolutePos
  mlp_type: gelu_mlp         # → RequiredOp::GeluMlp
```

Required ops: `{MHA, GeluMlp, LayerNorm, BiasAdd, AbsolutePos}`

## Design: Backend Capabilities

Each inference backend declares what operations it supports. This is a compile-time constant, not runtime discovery.

### Current trueno GPU kernel capabilities

The kernel `forward_all_layers_gpu_to_logits_graphed` currently supports:

```rust
const GPU_KERNEL_V1_CAPS: &[RequiredOp] = &[
    RequiredOp::RoPE,
    RequiredOp::GQA,
    RequiredOp::MHA,
    RequiredOp::SwiGLU,
    RequiredOp::RMSNorm,
    // NOT: QkNorm, LayerNorm, BiasAdd, GeluMlp, AbsolutePos, CausalMask
];
```

### CPU backend capabilities

The CPU backend (`forward_single_with_cache`) supports all operations:

```rust
const CPU_CAPS: &[RequiredOp] = &[
    RequiredOp::RoPE,
    RequiredOp::GQA,
    RequiredOp::MHA,
    RequiredOp::SwiGLU,
    RequiredOp::GeluMlp,
    RequiredOp::RMSNorm,
    RequiredOp::LayerNorm,
    RequiredOp::BiasAdd,
    RequiredOp::QkNorm,
    RequiredOp::AbsolutePos,
    RequiredOp::CausalMask,
];
```

## Design: Capability Gate

### Location

The gate runs at **model load time** in realizar, before any forward pass:

```
Model file → parse config → derive required_ops from contract
                          → check backend.supported_ops ⊇ required_ops
                          → if gap: REFUSE TO LOAD (not garbage)
                          → if ok: proceed with forward pass
```

### realizar API

```rust
// realizar/src/capability.rs

/// Operations that a model architecture may require.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RequiredOp {
    RoPE,
    GQA,
    MHA,
    SwiGLU,
    GeluMlp,
    RMSNorm,
    LayerNorm,
    BiasAdd,
    QkNorm,
    AbsolutePos,
    CausalMask,
}

/// Derive required operations from architecture constraints.
pub fn required_ops(constraints: &ArchConstraints) -> HashSet<RequiredOp> {
    let mut ops = HashSet::new();

    // Attention type
    match constraints.attention_type {
        AttentionType::Gqa => { ops.insert(RequiredOp::GQA); }
        AttentionType::Mha => { ops.insert(RequiredOp::MHA); }
    }

    // Positional encoding
    match constraints.positional_encoding {
        PositionalEncoding::Rope => { ops.insert(RequiredOp::RoPE); }
        PositionalEncoding::Absolute => { ops.insert(RequiredOp::AbsolutePos); }
        PositionalEncoding::None => {}
    }

    // Normalization
    match constraints.norm_type {
        NormType::RmsNorm => { ops.insert(RequiredOp::RMSNorm); }
        NormType::LayerNorm => { ops.insert(RequiredOp::LayerNorm); }
    }

    // FFN
    match constraints.mlp_type {
        MlpType::SwiGlu => { ops.insert(RequiredOp::SwiGLU); }
        MlpType::GeluMlp => { ops.insert(RequiredOp::GeluMlp); }
    }

    // Optional features
    if constraints.has_bias {
        ops.insert(RequiredOp::BiasAdd);
    }
    if constraints.has_qk_norm {
        ops.insert(RequiredOp::QkNorm);
    }

    ops
}

/// Check if a backend supports all required operations.
/// Returns Ok(()) or Err with the unsupported operations.
pub fn check_capability(
    required: &HashSet<RequiredOp>,
    supported: &HashSet<RequiredOp>,
) -> std::result::Result<(), Vec<RequiredOp>> {
    let missing: Vec<RequiredOp> = required.difference(supported).copied().collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(missing)
    }
}
```

### Integration Point: GPU model load

In `realizar/src/gguf/cuda/mod.rs` (or equivalent), before allocating GPU memory:

```rust
pub fn load_model_gpu(config: &GGUFConfig) -> Result<OwnedQuantizedModelCuda> {
    let required = capability::required_ops(&config.constraints);
    let supported = gpu_kernel_capabilities(); // compile-time constant

    if let Err(missing) = capability::check_capability(&required, &supported) {
        return Err(RealizarError::CapabilityMismatch {
            architecture: config.architecture.clone(),
            missing_ops: missing,
            suggestion: "Use CPU backend, or upgrade GPU kernel to support these operations.",
        });
    }

    // ... proceed with GPU allocation and weight upload
}
```

### Integration Point: `apr qa` (new Gate 0)

Add a **Capability Match** gate as Gate 0 in `apr qa` — runs BEFORE all other gates:

```
Gate 0: Capability Match
  Reads model contract → derives required_ops
  Checks CPU backend: always passes (supports all ops)
  Checks GPU backend: may fail with specific missing ops

Output on failure:
  ✗ FAIL Capability Match
    Model requires: [RoPE, GQA, SwiGLU, RMSNorm, QkNorm]
    GPU supports:   [RoPE, GQA, SwiGLU, RMSNorm]
    Missing:        [QkNorm]
    → GPU cannot serve this model. CPU inference available.
    → To add GPU support: implement QK RMSNorm in trueno kernel (GH-280)
```

This gate runs in <1ms (no forward pass needed) and gives actionable root cause.

### Integration Point: Parity gate improvement

The existing parity gate becomes a **second line of defense**. If capability gate passes but parity still fails, that indicates a kernel *bug* (not a missing feature). The error message changes:

```
Before (current):
  PARITY-GATE FAILED: GPU computes a DIFFERENT function than CPU.
  Cosine similarity: 0.000479 (required: ≥0.99)
  → Unhelpful. User doesn't know WHY.

After (with capability gate):
  PARITY-GATE FAILED: GPU output diverges despite passing capability check.
  Cosine similarity: 0.985 (required: ≥0.99)
  This indicates a KERNEL BUG, not a missing feature.
  Run `apr parity <model> --trace` for layer-by-layer diagnosis.
```

## Implementation Plan

### Phase 1: Core types and derivation (realizar)

| File | Change |
|------|--------|
| `realizar/src/capability.rs` | NEW — `RequiredOp` enum, `required_ops()`, `check_capability()` |
| `realizar/src/gguf/config.rs` | Add `required_ops()` method to `ArchConstraints` |
| `realizar/src/lib.rs` | Export `capability` module |

### Phase 2: GPU admission control (realizar)

| File | Change |
|------|--------|
| `realizar/src/gguf/cuda/mod.rs` | Call `check_capability()` before GPU allocation |
| `realizar/src/gguf/cuda/mod_part_02.rs` | Remove `has_qk_norm` skip hack in parity gate — capability gate handles it |
| `realizar/src/error.rs` | Add `CapabilityMismatch` variant to `RealizarError` |

### Phase 3: `apr qa` capability gate (apr-cli)

| File | Change |
|------|--------|
| `crates/apr-cli/src/commands/qa_part_02.rs` | Add Gate 0 (`capability_match`) before `tensor_contract` |
| `crates/apr-cli/src/commands/qa_capability.rs` | NEW — `run_capability_gate()` implementation |

### Phase 4: Contract extension (aprender)

| File | Change |
|------|--------|
| `src/format/model_family.rs` | Add `qk_norm: bool` to `ModelConstraints` (currently only in realizar's `ArchConstraints`) |
| `contracts/model-families/*.yaml` | Already have `qk_norm` field — no contract changes needed |

## Verification

1. `apr qa qwen3-8b-q4k.gguf` — Gate 0 reports `Missing: [QkNorm]` for GPU, skips GPU gates gracefully
2. `apr qa qwen2-7b-q4k.gguf` — Gate 0 passes for both CPU and GPU (no QK norm needed)
3. `apr qa gpt2.gguf` — Gate 0 reports `Missing: [LayerNorm, BiasAdd, GeluMlp, AbsolutePos]` for GPU
4. After implementing QK norm in kernel: Gate 0 passes, parity gate validates correctness
5. Regression: no existing model that currently works on GPU should be blocked

## Architecture Matrix

| Model Family | Required Ops | GPU Today | GPU After QK Norm |
|---|---|---|---|
| LLaMA | RoPE, GQA, SwiGLU, RMSNorm | PASS | PASS |
| Qwen2 | RoPE, GQA, SwiGLU, RMSNorm, BiasAdd | FAIL (BiasAdd) | FAIL (BiasAdd) |
| Qwen3 | RoPE, GQA, SwiGLU, RMSNorm, QkNorm | FAIL (QkNorm) | PASS |
| GPT-2 | MHA, GeluMlp, LayerNorm, BiasAdd, AbsolutePos | FAIL (4 ops) | FAIL (4 ops) |
| Qwen3.5 | RoPE, GQA, SwiGLU, RMSNorm, QkNorm, GatedDeltaNet | FAIL (2 ops) | FAIL (GatedDeltaNet) |

**Important insight from this matrix:** Qwen2 with `has_bias: true` should also fail the GPU capability gate. Currently it produces subtly wrong output (bias not applied). This is a *silent correctness bug* that the capability gate would surface.

## Success Criteria

1. **Zero garbage**: No model ever produces garbage on GPU. Either capability gate blocks it, or parity gate catches kernel bugs.
2. **Actionable diagnostics**: Every failure message names the specific missing operations and suggests a fix.
3. **<1ms overhead**: Capability check is a set comparison, no forward pass needed.
4. **Extensible**: Adding a new operation (e.g., `GatedDeltaNet` for Qwen3.5) is one enum variant + one contract field.

## References

- [Qwen3 pipeline spec](./qwen3-perf-parity.md) — GH-279-3 GPU garbage incident
- [BUG-GGUF-001 five-whys](./BUG-GGUF-001-five-whys-analysis.md) — prior root cause analysis
- [Tensor layout contract](../../contracts/tensor-layout-v1.yaml) — structural contract (what this extends)
- realizar `ArchConstraints` — `src/gguf/config.rs:92`
- trueno GPU kernel — `forward_all_layers_gpu_to_logits_graphed`
