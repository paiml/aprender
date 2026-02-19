# Unified Contract-by-Design Specification

**Version**: 2.0.0
**Status**: Phase 4 Complete (kernels), Phase 6 Complete (algorithms P0-P3), Step 6.7 AllImplemented enforced. Bindings: 236/239 implemented (3 allowed gaps). 24 `#[contract]` annotations. 21/24 algorithm contracts Bound, ~100 equations, 100 FALSIFY tests. Total: 264 contract tests passing. Remaining unbound: active-learning-v1, gnn-v1, metaheuristics-v1 (no implementation).
**Created**: 2026-02-19
**Updated**: 2026-02-19
**Scope**: trueno, realizar, aprender, entrenar, whisper.apr
**Depends On**: `provable-contracts` (49 kernel YAML + new algorithm YAMLs), `apr-model-qa-playbook` (217+ gates)

> **Core Thesis**: Every kernel, every ML algorithm, every statistical estimator, every
> mathematical function in the Sovereign AI Stack is governed by a provable contract.
> If a contract is missing, **the code does not compile**. There is ONE way to load,
> ONE way to serve, ONE way to import, ONE way to export. Every equation has a YAML
> source of truth, a `#[contract]` annotation, and a falsification test.
> Duplicate paths are deleted, not deprecated.

---

## Table of Contents

1. [Five Whys: Why This Spec Exists](#1-five-whys)
2. [Design Principles](#2-design-principles)
3. [Stack Ownership Matrix](#3-stack-ownership-matrix)
4. [The One Path Rule](#4-the-one-path-rule)
5. [Contract Enforcement Architecture](#5-contract-enforcement-architecture)
6. [Proc Macro Contract System](#6-proc-macro-contract-system)
7. [build.rs Binding Verification](#7-buildrs-binding-verification)
8. [Realizar: Universal Model Runtime](#8-realizar-universal-model-runtime)
9. [Dead Code Elimination Plan](#9-dead-code-elimination-plan)
10. [Binding Completeness Mandate](#10-binding-completeness-mandate)
11. [QA Playbook Integration](#11-qa-playbook-integration)
12. [Migration Path](#12-migration-path)
13. [Verification](#13-verification)
14. [Algorithm & Statistics Contract Architecture](#14-algorithm--statistics-contract-architecture)
15. [Algorithm Contract Inventory](#15-algorithm-contract-inventory)

---

## 1. Five Whys

**Symptom**: GPU inference produces garbage for Qwen3 models.

1. **Why?** QK norm weights silently skipped during forward pass.
2. **Why?** `attn_q_norm_len == 0` — sentinel value means both "not present" and "bug."
3. **Why?** APR loader has no mapping for QK norm tensors.
4. **Why?** No compile-time enforcement that loaders satisfy architecture requirements.
5. **Why?** **Multiple loading paths with no shared contract — each path can independently forget tensors.**

**Root cause**: The stack has N loading paths (GGUF in aprender, GGUF in realizar, SafeTensors in realizar, APR in realizar, SafeTensors CUDA in realizar) with no shared proof that they all produce complete, correct weight sets. Adding a new architecture feature (QK norm) requires updating N paths independently, and forgetting one produces silent garbage.

**Solution**: ONE loading path, ONE contract, compile-time enforcement.

---

## 2. Design Principles

### 2.1 Contract-First (not test-first)

Tests verify behavior. Contracts **prevent** invalid behavior from being expressible.

```
Traditional:  code → tests → "works on my machine"
Contract:     YAML contract → proc macro → compile-time proof → code MUST satisfy
```

### 2.2 Popperian Falsificationism

Every contract must be falsifiable. A contract that cannot be violated is vacuous.
The `apr-model-qa-playbook` provides 217+ falsification gates (F-*) that attempt
to break contracts. Passing means "not yet falsified," never "proven correct."

### 2.3 Toyota Jidoka (Autonomation)

Stop on first defect. A model that fails any P0 gateway (G0-G4) gets MQS = 0.
No partial loading. No "best effort" inference. Load completely or fail completely.

### 2.4 Poka-Yoke (Mistake-Proofing)

Make the wrong thing impossible to express in the type system:
- `ValidatedLayerWeights` cannot be constructed without passing architecture validation
- `ValidatedGpuWeights` cannot be constructed without dimension checks
- `LmHeadWeight` / `LmHeadWeightTransposed` newtypes prevent argument swaps
- Sealed constructors (private inner fields) prevent bypassing validation

### 2.5 Equation Enforcement for All Mathematics

Every mathematical function — not just kernels — must have:
1. A **YAML contract** with the canonical equation, domain, codomain, and invariants
2. A **`#[contract]` annotation** on the implementing function
3. A **falsification test** (FALSIFY-*) that verifies the invariants hold

This applies to:
- **Kernels** (softmax, RMSNorm, RoPE, attention) — already bound
- **ML algorithms** (decision trees, random forests, GBM, SVM, k-NN, naive Bayes)
- **Statistical estimators** (linear/logistic regression, Bayesian inference, GLMs)
- **Metrics** (R², MSE, accuracy, F1, silhouette, NDCG)
- **Loss functions** (BCE, NLL, Huber, cross-entropy)
- **Optimization** (gradient descent, conjugate gradient, L-BFGS, CMA-ES)
- **Decomposition** (PCA, ICA, SVD, eigendecomposition)
- **Time series** (ARIMA, ACF/PACF, Ljung-Box)
- **Graph algorithms** (centrality, shortest path, community detection)
- **Calibration** (Platt scaling, isotonic regression, temperature scaling)
- **Drift detection** (PSI, KL divergence, KS test)

```
Traditional:  implement algorithm → write unit tests → "seems to work"
Contract:     define equation in YAML → annotate function → falsification test → "invariants not yet falsified"
```

The difference is epistemic: tests check specific inputs, contracts enforce
mathematical properties that must hold for ALL valid inputs.

### 2.6 One Way Only

For every operation, there is exactly ONE canonical implementation:

| Operation | Owner | Canonical Path | Alternatives |
|-----------|-------|----------------|--------------|
| Model loading | realizar | `realizar::model_loader::load_from_path()` | **NONE** |
| GGUF parsing | realizar | `realizar::gguf::*` | **NONE** (aprender GGUF reader = dead code) |
| SafeTensors parsing | realizar | `realizar::safetensors::*` | **NONE** |
| APR parsing | realizar | `realizar::apr::*` | aprender read-only for inspection |
| Dequantization | realizar | `realizar::quantize::*` | **NONE** (aprender dequant = dead code) |
| Transpose | trueno | `trueno::blis::transpose` | **NONE** (all delegates via `contract_gate::transpose_f32`) |
| Format conversion | aprender | `aprender::format::converter::*` | **NONE** |
| HTTP serving | realizar | `realizar::api::*` + `realizar::serve::*` | **NONE** |
| Kernel compute | trueno | `trueno::backends::*` | **NONE** |
| Training | entrenar | `entrenar::train::*` | **NONE** |

---

## 3. Stack Ownership Matrix

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Sovereign AI Stack                                │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ apr CLI   │  │ entrenar │  │ whisper  │  │ other    │  CONSUMERS │
│  │ (commands)│  │ (train)  │  │ .apr     │  │ apps     │           │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       │              │              │              │                 │
│       ▼              ▼              ▼              ▼                 │
│  ┌──────────────────────────────────────────────────────┐           │
│  │              realizar (Universal Model Runtime)       │           │
│  │                                                       │           │
│  │  load_from_path() ─── THE ONE LOADING PATH           │           │
│  │    ├── detect_format() (magic bytes)                  │           │
│  │    ├── GGUF  → gguf::MappedGGUFModel                │           │
│  │    ├── APR   → apr::load_apr_model()                 │           │
│  │    └── SafeT → safetensors::MappedSafeTensorsModel   │           │
│  │                                                       │           │
│  │  serve() ─── THE ONE SERVING PATH                    │           │
│  │    ├── /v1/chat/completions (OpenAI compat)          │           │
│  │    ├── /generate (text generation)                    │           │
│  │    └── /predict (classical ML)                        │           │
│  │                                                       │           │
│  │  contract_gate ─── THE ONE VALIDATION PATH           │           │
│  │    ├── validate_model_load_basic()                    │           │
│  │    ├── ValidatedLayerWeights::validate()              │           │
│  │    └── ValidatedGpuWeights::new()                     │           │
│  └──────────────────────┬────────────────────────────────┘           │
│                         │                                            │
│                         ▼                                            │
│  ┌──────────────────────────────────────────────────────┐           │
│  │                 trueno (Compute Substrate)            │           │
│  │                                                       │           │
│  │  backends::cpu    ─── SIMD kernels (AVX2/NEON)       │           │
│  │  backends::gpu    ─── CUDA/wgpu kernels              │           │
│  │  blis::transpose  ─── THE ONE TRANSPOSE              │           │
│  │  tiling::q4k_matvec ─── THE ONE Q4K KERNEL          │           │
│  │  brick::*         ─── Fused kernel composition       │           │
│  └──────────────────────────────────────────────────────┘           │
│                                                                     │
│  ┌──────────────────────────────────────────────────────┐           │
│  │            aprender (Format Conversion ONLY)          │           │
│  │                                                       │
│  │  format::converter::import()  ─── SafeTensors→APR    │           │
│  │  format::converter::export()  ─── APR→GGUF           │           │
│  │  format::converter::convert() ─── quantize/merge     │           │
│  │  format::layout_contract      ─── SOURCE OF TRUTH    │           │
│  │  format::validation           ─── QA gates           │           │
│  │                                                       │           │
│  │  FORBIDDEN: No inference, no serving, no loading     │           │
│  └──────────────────────────────────────────────────────┘           │
│                                                                     │
│  ┌──────────────────────────────────────────────────────┐           │
│  │         provable-contracts (Contract Registry)        │           │
│  │                                                       │           │
│  │  KERNEL CONTRACTS (48 YAMLs, 166 eq, 262 oblig.)    │           │
│  │    197/203 bindings implemented (97.0%)              │           │
│  │                                                       │           │
│  │  ALGORITHM CONTRACTS (new — ~30 YAMLs, ~120 eq)     │           │
│  │    Metrics, loss, estimators, optimization,           │           │
│  │    decomposition, time series, graph, calibration     │           │
│  │                                                       │           │
│  │  Proc macros for compile-time enforcement            │           │
│  │  build.rs binding verification                        │           │
│  │  Kani bounded model checking (81 harnesses)          │           │
│  └──────────────────────────────────────────────────────┘           │
│                                                                     │
│  ┌──────────────────────────────────────────────────────┐           │
│  │       apr-model-qa-playbook (Falsification Engine)    │           │
│  │                                                       │           │
│  │  217+ falsification gates (F-*)                      │           │
│  │  5 P0 gateways (G0-G4)                              │           │
│  │  MQS scoring (0-1000)                                │           │
│  │  49/114 models certified (A+)                        │           │
│  │  1,800,000+ test assertions                          │           │
│  └──────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. The One Path Rule

### 4.1 Model Loading: ONE Path

Every consumer in the stack loads models through `realizar::model_loader::load_from_path()`.

```rust
// THE ONE WAY (all consumers use this)
use realizar::model_loader;
let model = model_loader::load_from_path(&path)?;

// FORBIDDEN — these paths will be deleted:
use aprender::format::gguf::load_gguf_raw;          // DEAD CODE
use aprender::format::gguf::dequant::dequantize_q4_0; // DEAD CODE
```

**Consumers**:
- `apr run` → `realizar::load_from_path()`
- `apr serve` → `realizar::load_from_path()`
- `entrenar::finetune` → `realizar::load_from_path()` (weights for LoRA base)
- `entrenar::distill` → `realizar::load_from_path()` (teacher model)
- `whisper.apr` → `realizar::load_from_path()` (audio model)
- `apr-model-qa-playbook` → `realizar::load_from_path()` (QA testing)

### 4.2 Format Conversion: ONE Path

All format conversion goes through `aprender::format::converter`:

```rust
// THE ONE WAY
use aprender::format::converter::{import, export, convert};

// Import: external format → APR
import::run_import(&opts)?;  // SafeTensors/GGUF → APR

// Export: APR → external format
export::run_export(&opts)?;  // APR → GGUF/SafeTensors

// Convert: APR → APR (quantize, merge, prune)
convert::run_convert(&opts)?;
```

### 4.3 HTTP Serving: ONE Path

All model serving goes through `realizar::api`:

```rust
// THE ONE WAY
use realizar::api::start_server;
start_server(config).await?;

// This serves:
// POST /v1/chat/completions  (OpenAI-compatible)
// POST /generate              (text generation)
// POST /predict               (classical ML)
// GET  /health, /ready, /metrics
```

### 4.4 Kernel Compute: ONE Path

Every mathematical kernel is implemented ONCE in trueno, with a provable contract:

```rust
// THE ONE WAY — every kernel has a contract
#[contract("rmsnorm-kernel-v1", equation = "rmsnorm")]
pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    // Implementation must satisfy all proof obligations from
    // provable-contracts/contracts/rmsnorm-kernel-v1.yaml
}
```

### 4.5 Transpose: ONE Path

All matrix transposition in the stack goes through one function:

```rust
// THE ONE WAY (added in PMAT-285)
trueno::blis::transpose::transpose(rows, cols, data, &mut out)

// All other transpose functions delegate to this via contract_gate::transpose_f32()
// Zero duplicate implementations allowed.
```

---

## 5. Contract Enforcement Architecture

### 5.1 Three Layers of Enforcement

```
Layer 1: YAML Contract (source of truth)
    provable-contracts/contracts/<name>-v1.yaml
    ├── equations (mathematical definitions)
    ├── proof_obligations (what must hold)
    ├── falsification_tests (what must break on bad input)
    └── kani_harnesses (formal verification)

Layer 2: Proc Macro (compile-time)
    #[contract("<name>-v1", equation = "<eq>")]
    ├── Verifies contract YAML exists
    ├── Generates compile-time type checks
    ├── Emits binding metadata for build.rs
    └── Fails compilation if contract missing

Layer 3: build.rs (link-time)
    build.rs reads binding.yaml and verifies:
    ├── Every binding has a corresponding #[contract] attribute
    ├── Every YAML equation has at least one binding
    ├── No unimplemented bindings remain
    └── Cross-crate parity (aprender ↔ realizar ↔ trueno)
```

### 5.2 Contract Lifecycle

```
Paper/Spec → YAML contract → binding.yaml → #[contract] proc macro → impl → test → certify
     │              │               │              │                     │       │        │
     │              │               │              │                     │       │        │
     ▼              ▼               ▼              ▼                     ▼       ▼        ▼
  Research    provable-     provable-        Rust source            Tests   QA gates   MQS
  citation   contracts/    contracts/       (trueno/              (cargo    (apr-qa)  score
             contracts/    contracts/        realizar/              test)
                           aprender/         aprender)
                           binding.yaml
```

---

## 6. Proc Macro Contract System

### 6.1 New Crate: `provable-contracts-macros`

A proc macro crate that reads YAML contracts and generates compile-time checks.

```toml
# provable-contracts/crates/provable-contracts-macros/Cargo.toml
[lib]
proc-macro = true

[dependencies]
syn = { version = "2", features = ["full"] }
quote = "1"
proc-macro2 = "1"
serde_yaml = "0.9"
```

### 6.2 The `#[contract]` Attribute

```rust
use provable_contracts_macros::contract;

/// RMS Normalization
///
/// Equation: output_i = (x_i / sqrt(mean(x^2) + eps)) * weight_i
///
/// Contract: rmsnorm-kernel-v1.yaml, equation "rmsnorm"
/// Proof obligations: PROOF-RMSN-001 through PROOF-RMSN-004
#[contract("rmsnorm-kernel-v1", equation = "rmsnorm")]
pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let hidden_dim = weight.len();
    let sum_sq: f32 = input.iter().map(|&v| v * v).sum();
    let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
    input.iter().zip(weight).map(|(&v, &w)| (v / rms) * w).collect()
}
```

### 6.3 What the Proc Macro Generates

The `#[contract]` attribute expands to:

```rust
// 1. Compile-time contract existence check
const _: () = {
    // build.rs sets this env var after validating the YAML exists
    // If YAML is missing → build.rs fails → this never compiles
    const _CONTRACT_EXISTS: &str = env!("CONTRACT_RMSNORM_KERNEL_V1_RMSNORM");
};

// 2. Binding registration (read by build.rs)
#[cfg_attr(feature = "contract-audit", link_section = ".contract_bindings")]
static _BINDING_RMSNORM: &str = concat!(
    "contract=rmsnorm-kernel-v1",
    ",equation=rmsnorm",
    ",module=", module_path!(),
    ",function=rms_norm",
);

// 3. The actual function (unchanged)
pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    // ... original implementation ...
}

// 4. Debug assertion for proof obligations (debug builds only)
#[cfg(debug_assertions)]
fn _check_rmsnorm_obligations(input: &[f32], weight: &[f32], output: &[f32], eps: f32) {
    // PROOF-RMSN-001: output.len() == input.len()
    debug_assert_eq!(output.len(), input.len());
    // PROOF-RMSN-002: no NaN in output
    debug_assert!(output.iter().all(|v| !v.is_nan()));
    // PROOF-RMSN-003: scale invariance
    // (generated from YAML proof_obligations)
}
```

### 6.4 Contract for Model Loading

```rust
/// Load model from any supported format.
///
/// Contract: inference-pipeline-v1.yaml, equation "model_load"
/// This is THE ONE loading path for the entire stack.
#[contract("inference-pipeline-v1", equation = "model_load")]
pub fn load_from_path(path: &Path) -> Result<LoadedModel> {
    let format = detect_format(path)?;
    match format {
        Format::Gguf => load_gguf(path),
        Format::Apr => load_apr(path),
        Format::SafeTensors => load_safetensors(path),
        // No wildcard — exhaustive match enforced by contract
    }
}
```

### 6.5 Contract for Architecture Completeness

```rust
/// Validate that all architecture-required weights are present.
///
/// Contract: model-config-algebra-v1.yaml, equation "weight_completeness"
/// GH-279: This is the gate that prevents the Qwen3 QK-norm bug class.
#[contract("model-config-algebra-v1", equation = "weight_completeness")]
pub fn validate_architecture_completeness(
    weights: &IndexedLayerWeights,
    arch: &ArchConstraints,
    layer_idx: usize,
) -> Result<ValidatedLayerWeights, WeightValidationError> {
    ValidatedLayerWeights::validate(weights, arch, layer_idx)
}
```

---

## 7. build.rs Binding Verification

### 7.1 Contract: Every Binding Must Be Wired

Each crate's `build.rs` reads `provable-contracts/contracts/aprender/binding.yaml`
and verifies that every binding with status `implemented` has a corresponding
`#[contract]` attribute in the source code.

```rust
// build.rs (simplified)
fn main() {
    let bindings = load_binding_yaml("../provable-contracts/contracts/aprender/binding.yaml");

    for binding in &bindings {
        if binding.status == "not_implemented" {
            // HARD ERROR — all bindings must be implemented (per spec mandate)
            panic!(
                "CONTRACT VIOLATION: binding {}.{} is not_implemented. \
                 All bindings must be implemented per unified-contract-by-design.md §10.",
                binding.contract, binding.equation
            );
        }

        // Set env var that #[contract] attribute checks
        let key = format!(
            "CONTRACT_{}_{}",
            binding.contract.to_uppercase().replace('-', "_"),
            binding.equation.to_uppercase()
        );
        println!("cargo:rustc-env={key}=bound");
    }

    // Verify no orphan #[contract] attributes (contract exists but no binding)
    verify_no_orphans(&bindings);

    println!("cargo:rerun-if-changed=../provable-contracts/contracts/");
}
```

### 7.2 Cross-Crate Binding Parity

The build.rs also verifies cross-crate parity:

```
For each contract equation:
  - If trueno implements it → realizar must use trueno's implementation
  - If realizar implements it → aprender must NOT duplicate it
  - If aprender implements it → it must be in format::converter (conversion only)
```

### 7.3 Architecture Requirements (GH-279)

The `architecture-requirements-v1.yaml` contract generates exhaustive requirements:

```yaml
# provable-contracts/contracts/architecture-requirements-v1.yaml
architectures:
  qwen2:
    required_weights: [attn_norm, ffn_norm, q_proj, k_proj, v_proj, o_proj,
                       gate_proj, up_proj, down_proj]
    required_bias: [q_bias, k_bias, v_bias]
    optional: [qk_norm]

  qwen3:
    required_weights: [attn_norm, ffn_norm, q_proj, k_proj, v_proj, o_proj,
                       gate_proj, up_proj, down_proj]
    required_qk_norm: [attn_q_norm, attn_k_norm]  # THE FIX for GH-279
    optional: []

  llama:
    required_weights: [attn_norm, ffn_norm, q_proj, k_proj, v_proj, o_proj,
                       gate_proj, up_proj, down_proj]
    optional: []

  whisper:
    required_weights: [encoder_attn_norm, decoder_attn_norm,
                       encoder_q_proj, encoder_k_proj, encoder_v_proj,
                       decoder_q_proj, decoder_k_proj, decoder_v_proj,
                       cross_attn_q_proj, cross_attn_k_proj, cross_attn_v_proj,
                       encoder_ffn, decoder_ffn]
    required_bias: [all projections have bias]
    optional: []
```

---

## 8. Realizar: Universal Model Runtime

### 8.1 Why Realizar Owns Everything

Realizar already has:
- The fastest dequantization (Q2K through Q6K, parallel)
- GPU scheduling and CUDA support
- KV cache with PagedAttention
- HTTP serving (axum)
- Memory-mapped zero-copy loading
- Contract gate validation

Aprender should NOT duplicate any of this. Aprender's role is:
1. Format conversion (SafeTensors/GGUF → APR, APR → GGUF)
2. Layout contract enforcement (SOURCE OF TRUTH for tensor shapes)
3. Validation, linting, inspection CLI commands
4. Training-specific code (via entrenar dependency)

### 8.2 Universal Load API

```rust
/// realizar/src/model_loader.rs
///
/// THE ONE LOADING PATH for the entire Sovereign AI Stack.
/// All consumers (apr CLI, entrenar, whisper.apr) go through here.

#[contract("inference-pipeline-v1", equation = "model_load")]
pub fn load_from_path(path: impl AsRef<Path>) -> Result<LoadedModel> {
    let path = path.as_ref();
    let format = detect_format(path)?;

    // Contract gate: validate before loading
    let _proof = contract_gate::validate_pre_load(path, &format)?;

    match format {
        ModelFormat::Gguf => {
            let mapped = gguf::MappedGGUFModel::open(path)?;
            let model = OwnedQuantizedModel::from_mapped(&mapped)?;
            Ok(LoadedModel::Quantized(model))
        }
        ModelFormat::Apr => {
            let model = apr::load_apr_model(path)?;
            Ok(LoadedModel::Apr(model))
        }
        ModelFormat::SafeTensors => {
            let mapped = safetensors::MappedSafeTensorsModel::open(path)?;
            Ok(LoadedModel::SafeTensors(mapped))
        }
        // No wildcard. Adding a new format requires:
        // 1. Add variant to ModelFormat enum
        // 2. Add contract equation to inference-pipeline-v1.yaml
        // 3. Add binding to binding.yaml
        // 4. Implement loader with #[contract] attribute
        // Missing any step = compile error
    }
}
```

### 8.3 Entrenar Integration

```rust
/// entrenar/src/lib.rs
///
/// Training uses realizar for ALL model I/O.
/// Entrenar owns: autograd, optimizers, training loops.
/// Entrenar does NOT own: model loading, serving, format parsing.

use realizar::model_loader;

pub fn load_base_model(path: &Path) -> Result<TrainableModel> {
    // THE ONE WAY — through realizar
    let loaded = model_loader::load_from_path(path)?;
    TrainableModel::from_loaded(loaded)
}
```

### 8.4 Whisper.apr Integration

```rust
/// Whisper audio models load through realizar like everything else.
/// The whisper architecture is defined in contracts/model-families/whisper.yaml
/// and enforced by architecture-requirements-v1.yaml.

use realizar::model_loader;

pub fn load_whisper(path: &Path) -> Result<WhisperModel> {
    let loaded = model_loader::load_from_path(path)?;
    WhisperModel::from_loaded(loaded)
}
```

---

## 9. Code Consolidation Plan

### 9.1 Clarification: Aprender GGUF Reader Is NOT Dead Code

**CORRECTION**: The original spec incorrectly marked aprender's GGUF reader as dead code.
Aprender's GGUF reader is the **conversion pipeline reader** used by `apr import` to convert
GGUF files to APR format. It MUST be kept. The distinction:

- **Inference loading** (realizar): `realizar::model_loader::load_from_path()` — loads for inference
- **Conversion reading** (aprender): `aprender::format::gguf::load_gguf_raw()` — reads for format conversion

| Module | Status | Reason |
|--------|--------|--------|
| `src/format/gguf/reader.rs` + parts | **KEEP** | Conversion pipeline reader for `apr import` |
| `src/format/gguf/dequant.rs` + parts | **KEEP** | Used by reader's F32 extraction path |
| `src/format/gguf/api.rs` | **KEEP** | Public API for conversion pipeline |
| `src/format/gguf/types.rs` | **KEEP** | Type definitions shared with converter |
| `src/serialization/safetensors.rs` | **KEEP** | Used by inspection/validation CLI commands |

### 9.2 Dequant Consolidation (aprender → trueno)

Aprender's dequant functions in `src/format/gguf/dequant.rs` duplicate trueno's implementations.
These should be consolidated to delegate to trueno where signatures match:

| aprender Function | trueno Equivalent | Action |
|-------------------|-------------------|--------|
| `dequantize_q4_k()` | `trueno::quantize::dequantize_q4_k()` | Delegate |
| `dequantize_q6_k()` | `trueno::quantize::dequantize_q6_k()` | Delegate |
| `dequantize_q4_0()` | No direct equivalent | Keep in aprender |
| `dequantize_q8_0()` | No direct equivalent | Keep in aprender |
| `f16_to_f32()` | `trueno::f16::to_f32()` | Delegate |

### 9.3 Realizar Duplicate Helpers (Deferred)

The APR CPU inference helpers (`rms_norm`, `matmul`, `simple_attention`, `apply_rope_norm`)
in `realizar/src/apr/helpers.rs` have **different APIs** from the GPU scheduler equivalents:

| Helper | GPU Equivalent | Difference |
|--------|---------------|------------|
| `rms_norm(x, weight, eps)` | `layer_norm_static(input, weight, bias, dim, eps)` | No bias param |
| `matmul(x, w, seq, in, out)` | `cpu_matmul(a, b, m, k, n)` | Transposed weight convention |
| `simple_attention(...)` | `gqa_multihead_attention(...)` | Multi-seq vs single-pos |

**Status**: API unification deferred. These serve the APR CPU inference path which has
~70 call sites. Unifying requires changing all callers. Track as PMAT-290.

### 9.4 Already Completed (PMAT-285)

Duplicate transpose implementations consolidated to `contract_gate::transpose_f32()`
delegating to `trueno::blis::transpose`. 6 implementations now delegate to one.

### 9.3 Deletion Safety Protocol

For each deletion:
1. Verify no external callers: `pmat query --literal "function_name" --files-with-matches`
2. Verify no re-exports: check `mod.rs` and `lib.rs`
3. Run `cargo build -p aprender && cargo build -p realizar && cargo build -p apr-cli`
4. Run `cargo test -p aprender && cargo test -p realizar && cargo test -p apr-cli`
5. Run `scripts/check_include_files.sh` (include!() safety)
6. Commit with `refactor: delete duplicate <name> (unified-contract-by-design §9)`

---

## 10. Binding Completeness Mandate

### 10.1 All 174 Bindings Must Be Implemented

Per user mandate: **every binding in `provable-contracts/contracts/aprender/binding.yaml` must have status `implemented`**. No `partial` or `not_implemented` allowed.

### 10.2 Current Gaps (2 bindings — SSM only)

**Updated 2026-02-19**: 197/203 bindings are now `implemented` (97.0%).
All Tier 1 (Qwen3/3.5) and Tier 2 (architecture completeness) gaps have been closed.

The only remaining gaps are SSM/Mamba (3 equations × 2 target modules = 6 bindings),
which is not yet implemented in any stack crate:

| Contract | Equation | Target Module | Priority | Status |
|----------|----------|---------------|----------|--------|
| `ssm-kernel-v1` | `ssm_discretize` | `realizar::ssm` | P2 | not_implemented |
| `ssm-kernel-v1` | `ssm_scan` | `realizar::ssm` | P2 | not_implemented |
| `ssm-kernel-v1` | `selective_gate` | `realizar::ssm` | P2 | not_implemented |

These require implementing the Mamba state space model (zero-order hold discretization,
parallel associative scan, input-dependent selection). Tracked as future work.

#### Previously Closed Gaps (all now implemented)

- Gated Delta Net (decay, read, delta, write, output) — implemented in realizar
- Sliding window attention — implemented via proptest
- RoPE extrapolation (NTK, YaRN, linear) — implemented in realizar
- Flash attention — implemented in realizar + provable-contracts
- SiLU standalone — implemented in provable-contracts
- Conv1D depthwise causal — implemented via proptest
- Architecture completeness gate (GH-279) — implemented in aprender

### 10.3 Enforcement Timeline

| Milestone | Deadline | Gate |
|-----------|----------|------|
| All Tier 1 (Qwen3/3.5) | 2026-03-01 | `cargo build` fails if missing |
| All Tier 2 (architecture) | 2026-03-15 | `cargo build` fails if missing |
| All Tier 3 (completeness) | 2026-04-01 | `build.rs` hard error on any `not_implemented` |
| Zero `partial` bindings | 2026-04-15 | `build.rs` hard error on `partial` status |

---

## 11. QA Playbook Integration

### 11.1 Contract ↔ QA Gate Mapping

Every provable-contract maps to one or more QA playbook gates:

| Contract | QA Gate(s) | What It Verifies |
|----------|-----------|------------------|
| `rmsnorm-kernel-v1` | F-TRACE-PAYLOAD-001 | RMSNorm output matches reference |
| `attention-kernel-v1` | F-TRACE-PAYLOAD-002 | Attention scores correct |
| `rope-kernel-v1` | F-TRACE-PAYLOAD-003 | RoPE positional encoding |
| `tensor-layout-v1` | F-LAYOUT-002 | Row-major compliance |
| `model-config-algebra-v1` | G0-DIM-CONFIG_PARSE | Dimensions match config |
| `format-parity-v1` | F-ROSETTA-DIFF-* | Cross-format equivalence |
| `inference-pipeline-v1` | G2-BASIC_INFERENCE | End-to-end inference works |
| `performance-grading-v1` | F-PROFILE-CI-* | Throughput/latency gates |

### 11.2 Certification Requirements

For a model to achieve MQS A+ certification:

1. **All P0 gateways pass** (G0-G4) — Jidoka stop-on-defect
2. **All format contracts satisfied** — tensor-layout-v1, format-parity-v1
3. **All kernel contracts exercised** — trace payload gates verify each kernel
4. **Architecture completeness proven** — model-config-algebra-v1 + architecture-requirements-v1
5. **Cross-backend parity** — layer-parity-v1 (GPU ≈ CPU within tolerance)
6. **Performance within budget** — performance-grading-v1, kernel-launch-budget-v1

### 11.3 The Falsification Chain

```
provable-contracts YAML
    │
    ├── defines proof obligations
    │
    ▼
#[contract] proc macro
    │
    ├── compile-time: equation exists, types match
    │
    ▼
cargo test (unit + property)
    │
    ├── runtime: invariants hold for tested inputs
    │
    ▼
apr-model-qa-playbook (F-* gates)
    │
    ├── falsification: try to BREAK the contract with real models
    │
    ▼
MQS certification
    │
    └── "not yet falsified" across 1,800,000+ assertions
```

---

## 12. Migration Path

### Phase 1: Proc Macro Infrastructure ~~(Week 1)~~ COMPLETE

1. ~~Create `provable-contracts-macros` crate with `#[contract]` attribute~~ Done (PMAT-286)
2. ~~Add `build.rs` to realizar, aprender, trueno, entrenar~~ Done (PMAT-287)
3. ~~Annotate existing kernel functions with `#[contract]`~~ Done — 24 annotations across 15 files
4. ~~Verify `cargo build` still succeeds with annotations~~ Done — 11,626 tests pass

### Phase 2: Realize ALL Loading Through Realizar ~~(Week 2)~~ COMPLETE

1. ~~Make entrenar depend on realizar for model loading~~ Done (PMAT-291)
2. ~~Update apr CLI commands to use `realizar::model_loader` exclusively~~ Done
3. ~~Add `#[deprecated]` to aprender's GGUF reader and dequant functions~~ Deferred — see §9.1
4. ~~Run full test suite + QA playbook~~ Done

### Phase 3: Code Consolidation ~~(Week 3)~~ PARTIALLY COMPLETE

1. ~~Dequant consolidation (aprender → trueno delegation)~~ Done (PMAT-288)
2. ~~Realizar helper API unification~~ Deferred (PMAT-290, different APIs)
3. ~~Aprender GGUF reader~~ KEPT — conversion pipeline reader, not dead code (§9.1)
4. ~~Run `scripts/check_include_files.sh`~~ Done — 562 include!() files tracked

### Phase 4: Binding Completeness ~~(Weeks 4-8)~~ COMPLETE (98.7%)

1. ~~Implement Tier 1 bindings (Gated Delta Net, Conv1D)~~ Done (197/203)
2. ~~Implement Tier 2 bindings (Flash Attention, SiLU, LoRA)~~ Done
3. ~~Implement remaining Tier 3 bindings~~ Done (except 2 SSM gaps)
4. ~~build.rs WarnOnGaps policy active~~ Done — tracks 2 SSM gaps silently via `CONTRACT_GAPS` env var
5. ~~Quality gates clippy clean (`--all-targets -D warnings`)~~ Done — test/example allows scoped
6. ~~42 contract test files with 164 passing falsification tests~~ Done
7. ~~`pmat comply check`: COMPLIANT~~ Done

### Phase 5: build.rs Hard Enforcement (pending SSM implementation)

1. Enable `build.rs` hard error for missing contracts — requires SSM implementation
2. Enable cross-crate parity verification
3. Run full QA playbook certification pass
4. Tag release

### Phase 6: Algorithm & Statistics Contracts — COMPLETE (P0-P3)

1. **P0 contracts** (5 YAMLs): metrics-regression-v1, metrics-classification-v1, metrics-clustering-v1, loss-functions-v1, graph-centrality-v1 — DONE
2. **P0 annotations**: `#[contract]` + comment annotations on ~20 functions — DONE
3. **P0 falsification tests**: 29 FALSIFY-* tests passing — DONE
4. **P1 contracts** (4 YAMLs): calibration-v1, preprocessing-normalization-v1, decision-tree-v1, naive-bayes-v1 — DONE
5. **P1 annotations + tests**: 23 FALSIFY-* tests passing — DONE
6. **P2 contracts** (4 YAMLs): linear-models-v1, svm-v1, pca-v1, ica-v1 — DONE
7. **P2 annotations + tests**: 15 FALSIFY-* tests passing — DONE
8. **P3 contracts** (8 YAMLs): arima-v1, bayesian-v1, glm-v1, drift-detection-v1, metrics-ranking-v1, gbm-v1, optimization-v1, random-forest-v1 — DONE
9. **P3 annotations + tests**: 33 FALSIFY-* tests passing — DONE
10. **Binding registration**: ~70 algorithm equations in binding.yaml — DONE
11. **Hard enforcement**: `build.rs` AllImplemented policy — 236/239 implemented, 3 allowed gaps (SSM, robust_scaler) — DONE

---

## 13. Verification

### 13.1 Compile-Time Verification

```bash
# If any contract is missing, this fails
cargo build --workspace

# If any binding is not_implemented, this fails (after Phase 5)
cargo build -p realizar

# Cross-crate parity check
cargo build -p aprender -p realizar -p trueno
```

### 13.2 Runtime Verification

```bash
# Unit + property tests
cargo test --workspace

# Contract-specific tests
cargo test -p provable-contracts

# Kani bounded model checking (81 harnesses)
cd ~/src/provable-contracts && cargo kani
```

### 13.3 Falsification Verification

```bash
# QA playbook — attempts to falsify all contracts with real models
cd ~/src/apr-model-qa-playbook
cargo run --bin apr-qa -- certify --tier mvp --model Qwen/Qwen2.5-1.5B

# Full qualification (all 114 models)
cargo run --bin apr-qa -- certify --tier full --all
```

### 13.4 End-to-End Smoke Test

```bash
# THE ONE PATH — load → infer → verify
apr run hf://Qwen/Qwen2.5-1.5B-Instruct --prompt "2+2=" --max-tokens 8
# Expected: "4" (not garbage, not empty, not repeated tokens)

# THE ONE PATH — serve
apr serve hf://Qwen/Qwen2.5-1.5B-Instruct --port 8080
curl -X POST localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"2+2="}]}'

# THE ONE PATH — import → export round-trip
apr import hf://Qwen/Qwen2.5-1.5B -o model.apr
apr export model.apr --format gguf -o model.gguf
apr diff model.apr model.gguf  # Should show zero tensor diffs
```

### 13.5 Contract Audit

```bash
# Verify all bindings are wired
cd ~/src/provable-contracts
cargo run --bin pv -- audit --binding

# Verify obligation coverage
cargo run --bin pv -- coverage

# Verify no orphan contracts (YAML exists but no binding)
cargo run --bin pv -- audit --orphans
```

---

## 14. Algorithm & Statistics Contract Architecture

### 14.1 Five Whys: Why Algorithms Need Contracts Too

**Symptom**: Coverage at 87% despite 11,626 tests — pure algorithm modules at 0-35% coverage.

1. **Why?** Algorithm functions (metrics, loss, centrality, calibration) have no exercising tests.
2. **Why?** Tests were written ad-hoc — no systematic obligation to cover all functions.
3. **Why?** No contract system for algorithms — only kernels have `#[contract]` annotations.
4. **Why?** Original UCBD spec was scoped to kernels (inference pipeline bugs).
5. **Why?** **The assumption that "algorithms are just code, not kernels" left them without mathematical enforcement.**

**Root cause**: The contract system distinguishes between "kernels" (softmax, RMSNorm) and "algorithms" (R², Gini impurity, Platt scaling) — but mathematically they're the same: functions with equations, domains, codomains, and invariants. The type system doesn't distinguish "kernel math" from "algorithm math."

**Solution**: Extend the contract system to ALL mathematical functions. Every equation gets a YAML, a `#[contract]`, and a FALSIFY test.

### 14.2 Contract-Algorithm Isomorphism

A kernel contract and an algorithm contract have IDENTICAL structure:

```yaml
# Kernel contract (already exists):
equations:
  softmax:
    formula: "σ(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))"
    invariants:
      - "Σ σ(x)_i = 1.0"
      - "σ(x)_i > 0"

# Algorithm contract (new — identical structure):
equations:
  r_squared:
    formula: "R² = 1 - SS_res/SS_tot = 1 - Σ(y-ŷ)²/Σ(y-ȳ)²"
    invariants:
      - "R² ≤ 1.0 (upper bound)"
      - "R² = 1.0 when ŷ = y (perfect prediction)"
      - "R² = 0.0 when ŷ = ȳ (predict mean)"
```

The `#[contract]` proc macro, build.rs verification, and falsification test framework work identically for both.

### 14.3 Algorithm Contract Categories

| Category | YAML Prefix | Equations | Module(s) |
|----------|-------------|-----------|-----------|
| Regression Metrics | `metrics-regression-v1` | R², MSE, MAE, RMSE | `metrics/mod.rs` |
| Classification Metrics | `metrics-classification-v1` | accuracy, precision, recall, F1, confusion_matrix, ROC-AUC | `metrics/classification.rs` |
| Clustering Metrics | `metrics-clustering-v1` | silhouette_score, inertia, adjusted_rand | `metrics/mod.rs`, `cluster/` |
| Ranking Metrics | `metrics-ranking-v1` | hit_at_k, mrr, ndcg | `metrics/ranking.rs` |
| Loss Functions | `loss-functions-v1` | bce, nll, huber, smooth_l1, l1, mse_loss | `nn/loss.rs` |
| Decision Trees | `decision-tree-v1` | gini_impurity, entropy, information_gain | `tree/` |
| Random Forest | `random-forest-v1` | bootstrap_aggregation, oob_error, feature_importance | `tree/` |
| GBM | `gbm-v1` | gradient_boost, shrinkage, negative_gradient | `ensemble/` |
| Naive Bayes | `naive-bayes-v1` | gaussian_posterior, prior_update, log_likelihood | `classification/` |
| Linear Models | `linear-models-v1` | ols, ridge, lasso, logistic_sigmoid | `regression/`, `classification/` |
| SVM | `svm-v1` | hinge_loss, rbf_kernel, margin | `svm/` |
| PCA | `pca-v1` | eigendecomposition, explained_variance, projection | `preprocessing/` |
| ICA | `ica-v1` | negentropy, fastica_iteration, whitening | `decomposition/ica.rs` |
| Graph Centrality | `graph-centrality-v1` | degree, betweenness, closeness, eigenvector, katz, harmonic | `graph/centrality.rs` |
| Calibration | `calibration-v1` | platt_scaling, isotonic, temperature, ece, brier_score | `calibration.rs` |
| Active Learning | `active-learning-v1` | uncertainty, margin, entropy, qbc | `active_learning.rs` |
| ARIMA | `arima-v1` | ar_forecast, differencing, ma_filter, aic | `time_series/` |
| GLMs | `glm-v1` | poisson_link, gamma_link, binomial_link, irls | `glm/` |
| Bayesian | `bayesian-v1` | conjugate_prior, posterior_update, blr_predict | `bayesian/` |
| Optimization | `optimization-v1` | conjugate_gradient, gradient_descent, newtons_method | `optim/` |
| Drift Detection | `drift-detection-v1` | psi, kl_divergence, ks_test | `metrics/drift.rs` |
| Normalization | `normalization-v1` | l2_normalize, min_max_scale, standardize | `preprocessing/` |
| GNN | `gnn-v1` | gcn_aggregate, message_passing, graph_pool | `gnn/` |
| Metaheuristics | `metaheuristics-v1` | simulated_annealing, genetic, particle_swarm | `metaheuristics/` |

**Total: ~24 new YAML contracts, ~120 equations, ~200 proof obligations.**

### 14.4 Equation Format for Algorithms

Every algorithm equation follows the same YAML schema as kernel equations:

```yaml
# provable-contracts/contracts/metrics-regression-v1.yaml
metadata:
  version: "1.0.0"
  description: "Regression metrics — error measurement for continuous predictions"
  references:
    - "Draper & Smith (1998) Applied Regression Analysis"

equations:
  r_squared:
    formula: "R² = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²"
    domain: "y, ŷ ∈ ℝⁿ, n ≥ 1, Var(y) > 0"
    codomain: "R² ∈ (-∞, 1]"
    invariants:
      - "R² ≤ 1.0 (upper bound from Cauchy-Schwarz)"
      - "R² = 1.0 iff ŷᵢ = yᵢ for all i (perfect fit)"
      - "R² = 0.0 iff ŷᵢ = ȳ for all i (predict mean)"
      - "R² < 0 possible (worse than mean predictor)"

  mse:
    formula: "MSE = (1/n) Σ(yᵢ - ŷᵢ)²"
    domain: "y, ŷ ∈ ℝⁿ, n ≥ 1"
    codomain: "MSE ∈ [0, ∞)"
    invariants:
      - "MSE ≥ 0 (non-negativity from squared terms)"
      - "MSE = 0 iff ŷᵢ = yᵢ for all i"
      - "MSE(y, ŷ) = MSE(ŷ, y) (symmetry)"

  mae:
    formula: "MAE = (1/n) Σ|yᵢ - ŷᵢ|"
    domain: "y, ŷ ∈ ℝⁿ, n ≥ 1"
    codomain: "MAE ∈ [0, ∞)"
    invariants:
      - "MAE ≥ 0 (non-negativity from absolute value)"
      - "MAE = 0 iff ŷᵢ = yᵢ for all i"
      - "MAE ≤ RMSE (Jensen's inequality)"

  rmse:
    formula: "RMSE = √MSE = √((1/n) Σ(yᵢ - ŷᵢ)²)"
    domain: "y, ŷ ∈ ℝⁿ, n ≥ 1"
    codomain: "RMSE ∈ [0, ∞)"
    invariants:
      - "RMSE ≥ 0"
      - "RMSE ≥ MAE (always, from Jensen's)"
      - "RMSE = 0 iff MSE = 0"

proof_obligations:
  - type: bound
    property: "R² upper bound"
    formal: "R² ≤ 1.0 for all y, ŷ with Var(y) > 0"
  - type: bound
    property: "MSE non-negativity"
    formal: "MSE ≥ 0 for all y, ŷ"
  - type: invariant
    property: "MAE-RMSE ordering"
    formal: "MAE(y, ŷ) ≤ RMSE(y, ŷ) for all y, ŷ"
  - type: equivalence
    property: "Perfect prediction"
    formal: "R² = 1 ∧ MSE = 0 ∧ MAE = 0 ∧ RMSE = 0 when ŷ = y"

falsification_tests:
  - id: FALSIFY-RM-001
    rule: "R² upper bound"
    prediction: "R² ≤ 1.0 for random y, ŷ ∈ [-1000, 1000]ⁿ"
    test: "proptest with 10000 random vector pairs, n=2..128"
    if_fails: "Division by zero or sign error in SS computation"
  - id: FALSIFY-RM-002
    rule: "MSE non-negativity"
    prediction: "MSE ≥ 0 for all inputs"
    test: "proptest with extreme values"
    if_fails: "Arithmetic overflow in squared difference"
  - id: FALSIFY-RM-003
    rule: "MAE ≤ RMSE (Jensen's inequality)"
    prediction: "MAE ≤ RMSE for random inputs"
    test: "proptest comparing MAE and RMSE on same inputs"
    if_fails: "Implementation violates Jensen's inequality"
  - id: FALSIFY-RM-004
    rule: "Perfect prediction identity"
    prediction: "When ŷ = y: R²=1, MSE=0, MAE=0, RMSE=0"
    test: "proptest with y = ŷ for random vectors"
    if_fails: "Floating-point error accumulation"
```

### 14.5 Annotating Algorithm Functions

Existing algorithm functions receive `#[contract]` annotations identical to kernels:

```rust
// BEFORE: no contract, no mathematical enforcement
pub fn r_squared(y_pred: &Vector<f32>, y_true: &Vector<f32>) -> f32 { ... }

// AFTER: contract-bound, equation-enforced
#[contract("metrics-regression-v1", equation = "r_squared")]
pub fn r_squared(y_pred: &Vector<f32>, y_true: &Vector<f32>) -> f32 { ... }
```

```rust
// BEFORE: loss function with only cross-entropy contracted
pub fn bce_loss(predictions: &[f32], targets: &[f32]) -> f32 { ... }

// AFTER: all loss functions have contracts
#[contract("loss-functions-v1", equation = "bce")]
pub fn bce_loss(predictions: &[f32], targets: &[f32]) -> f32 { ... }
```

```rust
// BEFORE: graph centrality with only pagerank contracted
pub fn degree_centrality(graph: &Graph) -> Vec<f64> { ... }

// AFTER: all centrality measures have contracts
#[contract("graph-centrality-v1", equation = "degree")]
pub fn degree_centrality(graph: &Graph) -> Vec<f64> { ... }
```

### 14.6 Falsification Tests for Algorithms

Algorithm contract tests follow the SAME pattern as kernel contract tests:

```rust
// tests/contracts/regression_metrics_contract.rs
// CONTRACT: metrics-regression-v1.yaml

proptest! {
    /// Obligation: R² upper bound
    /// Formal: R² ≤ 1.0 for all y, ŷ with Var(y) > 0
    #[test]
    fn prop_r_squared_upper_bound(
        y_true in proptest::collection::vec(-1000.0f32..1000.0, 4..64usize),
        noise in proptest::collection::vec(-100.0f32..100.0, 4..64usize),
    ) {
        let n = y_true.len().min(noise.len());
        let y_true = Vector::from_slice(&y_true[..n]);
        let y_pred_data: Vec<f32> = y_true.as_slice().iter()
            .zip(noise[..n].iter())
            .map(|(y, n)| y + n)
            .collect();
        let y_pred = Vector::from_slice(&y_pred_data);

        // Skip degenerate case: constant y_true
        let var: f32 = y_true.as_slice().iter()
            .map(|y| (y - y_true.mean()).powi(2))
            .sum();
        prop_assume!(var > 1e-6);

        let r2 = r_squared(&y_pred, &y_true);
        prop_assert!(r2 <= 1.0 + 1e-6, "R²={r2} > 1.0 — upper bound violated");
    }

    /// Obligation: MSE ≥ 0 (non-negativity)
    #[test]
    fn prop_mse_non_negative(
        y_true in proptest::collection::vec(-1e6f32..1e6, 2..128usize),
        y_pred in proptest::collection::vec(-1e6f32..1e6, 2..128usize),
    ) {
        let n = y_true.len().min(y_pred.len());
        let yt = Vector::from_slice(&y_true[..n]);
        let yp = Vector::from_slice(&y_pred[..n]);
        let m = mse(&yp, &yt);
        prop_assert!(m >= 0.0, "MSE={m} < 0 — non-negativity violated");
    }

    /// Obligation: MAE ≤ RMSE (Jensen's inequality)
    #[test]
    fn prop_mae_le_rmse(
        y_true in proptest::collection::vec(-1e3f32..1e3, 2..64usize),
        y_pred in proptest::collection::vec(-1e3f32..1e3, 2..64usize),
    ) {
        let n = y_true.len().min(y_pred.len());
        let yt = Vector::from_slice(&y_true[..n]);
        let yp = Vector::from_slice(&y_pred[..n]);
        let m = mae(&yp, &yt);
        let r = rmse(&yp, &yt);
        prop_assert!(m <= r + 1e-6, "MAE={m} > RMSE={r} — Jensen's violated");
    }
}
```

### 14.7 Implementation Priority (by coverage impact)

Priority is determined by: (1) uncovered lines, (2) equation clarity, (3) pure testability.

| Priority | Contract YAML | Uncov Lines | Key Functions |
|----------|--------------|-------------|---------------|
| **P0** | metrics-regression-v1 | ~200 | r_squared, mse, mae, rmse, inertia, silhouette |
| **P0** | metrics-classification-v1 | ~300 | accuracy, precision, recall, f1, confusion_matrix |
| **P0** | loss-functions-v1 | ~200 | bce, nll, huber, smooth_l1, mse_loss |
| **P0** | graph-centrality-v1 | ~150 | degree, betweenness, closeness, eigenvector, katz |
| **P1** | calibration-v1 | ~150 | platt, isotonic, temperature, ece, brier |
| **P1** | active-learning-v1 | ~220 | uncertainty, margin, entropy, qbc |
| **P1** | decision-tree-v1 | ~180 | gini, entropy, information_gain, cart_split |
| **P1** | normalization-v1 | ~150 | rmsnorm, layernorm, groupnorm forward passes |
| **P2** | naive-bayes-v1 | ~120 | gaussian_posterior, prior, log_likelihood |
| **P2** | linear-models-v1 | ~100 | ols, ridge, logistic |
| **P2** | pca-v1 | ~100 | eigendecomposition, explained_variance |
| **P2** | optimization-v1 | ~100 | conjugate_gradient, gradient_descent |
| **P2** | arima-v1 | ~120 | ar_forecast, differencing, ma_filter |
| **P2** | bayesian-v1 | ~100 | conjugate_prior, posterior_update |
| **P3** | svm-v1 | ~80 | hinge_loss, rbf_kernel |
| **P3** | ica-v1 | ~80 | negentropy, fastica |
| **P3** | gbm-v1 | ~80 | gradient_boost, shrinkage |
| **P3** | random-forest-v1 | ~80 | bootstrap, oob_error, feature_importance |
| **P3** | glm-v1 | ~80 | poisson_link, gamma_link |
| **P3** | drift-detection-v1 | ~80 | psi, kl_divergence, ks_test |
| **P3** | metrics-ranking-v1 | ~60 | hit_at_k, mrr, ndcg |
| **P3** | gnn-v1 | ~60 | gcn_aggregate, message_passing |
| **P3** | metaheuristics-v1 | ~100 | simulated_annealing, genetic, pso |

---

## 15. Algorithm Contract Inventory

### 15.1 Phase 6 Scope: ~24 YAMLs, ~120 equations, ~200 proof obligations

This section tracks the algorithm contract implementation progress (Phase 6).

### Tier A1: Core Metrics (3 contracts, ~20 equations)

| Contract | Equations | Key Invariants | Status |
|----------|-----------|----------------|--------|
| metrics-regression-v1 | r_squared, mse, mae, rmse | R²≤1, MSE≥0, MAE≤RMSE | Bound |
| metrics-classification-v1 | accuracy, precision, recall, f1, confusion_matrix, roc_auc | F1=harmonic_mean(P,R), accuracy∈[0,1] | Bound |
| metrics-clustering-v1 | silhouette_score, inertia, adjusted_rand | silhouette∈[-1,1], inertia≥0 | Bound |

### Tier A2: Loss & Activation (2 contracts, ~10 equations)

| Contract | Equations | Key Invariants | Status |
|----------|-----------|----------------|--------|
| loss-functions-v1 | bce, nll, huber, smooth_l1, l1_loss, mse_loss | all≥0, bce=-Σ[y·log(ŷ)+(1-y)·log(1-ŷ)] | Bound |
| normalization-v1 | rmsnorm_forward, layernorm_forward, groupnorm_forward | output_shape=input_shape, unit_variance | Bound |

### Tier A3: Supervised Learning (5 contracts, ~25 equations)

| Contract | Equations | Key Invariants | Status |
|----------|-----------|----------------|--------|
| decision-tree-v1 | gini_impurity, entropy, information_gain, cart_split | gini∈[0,0.5], entropy≥0, IG≥0 | Bound |
| random-forest-v1 | bootstrap_aggregation, oob_error, feature_importance | importance sums to 1, oob∈[0,1] | Bound |
| naive-bayes-v1 | gaussian_posterior, prior_update, log_likelihood | posterior∝prior×likelihood | Bound |
| linear-models-v1 | ols_normal_eq, ridge_penalty, logistic_sigmoid | β=(X^TX)^{-1}X^Ty, σ∈(0,1) | Bound |
| svm-v1 | hinge_loss, rbf_kernel, polynomial_kernel, margin | hinge≥0, kernel positive semi-definite | Bound |

### Tier A4: Unsupervised & Decomposition (3 contracts, ~12 equations)

| Contract | Equations | Key Invariants | Status |
|----------|-----------|----------------|--------|
| pca-v1 | eigendecomposition, explained_variance, projection | Σ explained_var = 1, orthogonal components | Bound |
| ica-v1 | negentropy, fastica_iteration, whitening | independence maximized, E[s]=0 | Bound |
| gbm-v1 | gradient_boost, shrinkage, negative_gradient | loss non-increasing per stage | Bound |

### Tier A5: Graph & Optimization (3 contracts, ~15 equations)

| Contract | Equations | Key Invariants | Status |
|----------|-----------|----------------|--------|
| graph-centrality-v1 | degree, betweenness, closeness, eigenvector, katz, harmonic | all≥0, betweenness∈[0,1] normalized | Bound |
| optimization-v1 | conjugate_gradient, gradient_descent, newtons_method | convergence: f(x_{k+1})≤f(x_k) | Bound |
| calibration-v1 | platt_scaling, isotonic, temperature, ece, brier_score | calibrated P(y|p)≈p, ECE≥0, Brier∈[0,1] | Bound |

### Tier A6: Specialized (5 contracts, ~20 equations)

| Contract | Equations | Key Invariants | Status |
|----------|-----------|----------------|--------|
| active-learning-v1 | uncertainty, margin, entropy, qbc | entropy≥0, margin≥0, QBC variance≥0 | **Not bound** |
| arima-v1 | ar_forecast, differencing, ma_filter, aic | stationarity after d differences | Bound |
| bayesian-v1 | conjugate_prior, posterior_update, blr_predict | posterior∝prior×likelihood, predictive well-calibrated | Bound |
| glm-v1 | poisson_link, gamma_link, binomial_link, irls | link(μ)=Xβ, variance=V(μ) | Bound |
| drift-detection-v1 | psi, kl_divergence, ks_test | KL≥0, PSI≥0, KS∈[0,1] | Bound |

### Tier A7: Auxiliary (3 contracts, ~12 equations)

| Contract | Equations | Key Invariants | Status |
|----------|-----------|----------------|--------|
| metrics-ranking-v1 | hit_at_k, mrr, ndcg | all∈[0,1], NDCG normalized by ideal DCG | Bound |
| gnn-v1 | gcn_aggregate, message_passing, graph_pool | permutation equivariance | **Not bound** |
| metaheuristics-v1 | simulated_annealing, genetic_crossover, pso_update | monotone non-increasing best-so-far | **Not bound** |

### 15.2 Phase 6 Migration Path

| Step | Action | Gate |
|------|--------|------|
| 6.1 | Create P0 contract YAMLs (metrics-regression, metrics-classification, loss-functions, graph-centrality) | YAML review |
| 6.2 | Add `#[contract]` annotations to P0 functions | `cargo build` |
| 6.3 | Write FALSIFY-* contract tests for P0 equations | `cargo test` |
| 6.4 | Register P0 bindings in `binding.yaml` | `build.rs` pass |
| 6.5 | Create P1 contract YAMLs and repeat | Coverage ≥95% |
| 6.6 | Create P2-P3 contract YAMLs and repeat | Full binding coverage |
| 6.7 | Enable `build.rs` hard enforcement for algorithm contracts | `AllImplemented` |

---

## Appendix A: Contract YAML Inventory (48 kernel + ~24 algorithm contracts)

### Tier 1: Core Kernels (7)
| Contract | Equations | Obligations | Status |
|----------|-----------|-------------|--------|
| softmax-kernel-v1 | 3 | 8 | Bound |
| rmsnorm-kernel-v1 | 2 | 6 | Bound |
| rope-kernel-v1 | 3 | 7 | Bound |
| activation-kernel-v1 | 4 | 9 | Bound |
| attention-kernel-v1 | 4 | 12 | Bound |
| matmul-kernel-v1 | 3 | 8 | Bound |
| flash-attention-v1 | 3 | 9 | Bound |

### Tier 2: Compound Kernels (6)
| Contract | Equations | Obligations | Status |
|----------|-----------|-------------|--------|
| swiglu-kernel-v1 | 2 | 5 | Bound |
| gqa-kernel-v1 | 3 | 8 | Bound |
| layernorm-kernel-v1 | 2 | 6 | Bound |
| silu-kernel-v1 | 1 | 3 | Bound |
| cross-entropy-kernel-v1 | 2 | 5 | Bound |
| adamw-kernel-v1 | 4 | 10 | Bound |

### Tier 3: Extended (7)
| Contract | Equations | Obligations | Status |
|----------|-----------|-------------|--------|
| ssm-kernel-v1 | 3 | 7 | **Not bound** |
| conv1d-kernel-v1 | 2 | 5 | Bound |
| batchnorm-kernel-v1 | 2 | 5 | Bound |
| kmeans-kernel-v1 | 3 | 6 | Bound |
| pagerank-kernel-v1 | 2 | 5 | Bound |
| lbfgs-kernel-v1 | 3 | 7 | Bound |
| cma-es-kernel-v1 | 3 | 8 | Bound |

### Model Architecture (21)
All bound. See `binding.yaml` for per-equation status.

### Qwen 3.5 Verification (7)
| Contract | Equations | Obligations | Status |
|----------|-----------|-------------|--------|
| sliding-window-attention-v1 | 2 | 4 | Bound |
| rope-extrapolation-v1 | 3 | 6 | Bound |
| embedding-algebra-v1 | 2 | 4 | Bound |
| inference-pipeline-v1 | 4 | 8 | Bound |
| qwen35-hybrid-forward-v1 | 6 | 7 | Bound |
| attention-scaling-v1 | 2 | 4 | Bound |
| qwen35-e2e-verification-v1 | 6 | 5 | Bound |

### Algorithm Contracts — Tier A1: Core Metrics (3) — P0 COMPLETE
| Contract | Equations | Obligations | Status |
|----------|-----------|-------------|--------|
| metrics-regression-v1 | 4 | 7 | Bound |
| metrics-classification-v1 | 5 | 8 | Bound |
| metrics-clustering-v1 | 3 | 5 | Bound |

### Algorithm Contracts — Tier A2: Loss & Normalization (2) — P0/P1 COMPLETE
| Contract | Equations | Obligations | Status |
|----------|-----------|-------------|--------|
| loss-functions-v1 | 6 | 6 | Bound |
| normalization-v1 | 3 | 6 | Bound |

### Algorithm Contracts — Tier A3: Supervised Learning (5) — P1/P2 COMPLETE
| Contract | Equations | Obligations | Status |
|----------|-----------|-------------|--------|
| decision-tree-v1 | 4 | 8 | Bound |
| random-forest-v1 | 3 | 6 | Bound |
| naive-bayes-v1 | 3 | 6 | Bound |
| linear-models-v1 | 4 | 8 | Bound |
| svm-v1 | 4 | 8 | Bound |

### Algorithm Contracts — Tier A4-A7: Extended (12) — P0-P3 COMPLETE (except active-learning, gnn, metaheuristics)
| Contract | Equations | Obligations | Status |
|----------|-----------|-------------|--------|
| pca-v1 | 3 | 6 | Bound |
| ica-v1 | 3 | 6 | Bound |
| gbm-v1 | 3 | 6 | Bound |
| graph-centrality-v1 | 6 | 8 | Bound |
| optimization-v1 | 3 | 6 | Bound |
| calibration-v1 | 5 | 8 | Bound |
| active-learning-v1 | 4 | 6 | **Not bound** |
| arima-v1 | 4 | 8 | Bound |
| bayesian-v1 | 3 | 6 | Bound |
| glm-v1 | 4 | 8 | Bound |
| drift-detection-v1 | 3 | 6 | Bound |
| metrics-ranking-v1 | 3 | 6 | Bound |

---

## Appendix B: Model Family Coverage (17 families)

| Family | YAML | Contract | QA Cert | Loading Path |
|--------|------|----------|---------|--------------|
| Qwen2 | qwen2.yaml | model-config-algebra-v1 | A+ (49 models) | realizar |
| Qwen3 | qwen3.yaml | model-config-algebra-v1 | A+ (3 models) | realizar |
| Qwen3.5 | qwen3_5.yaml | qwen35-hybrid-forward-v1 | **Pending** | realizar |
| LLaMA | llama.yaml | model-config-algebra-v1 | A+ | realizar |
| Mistral | mistral.yaml | model-config-algebra-v1 | A+ | realizar |
| Gemma | gemma.yaml | model-config-algebra-v1 | A+ | realizar |
| Phi | phi.yaml | model-config-algebra-v1 | A+ | realizar |
| DeepSeek | deepseek.yaml | model-config-algebra-v1 | A+ | realizar |
| GPT-2 | gpt2.yaml | model-config-algebra-v1 | A+ | realizar |
| Falcon-H1 | falcon_h1.yaml | ssm-kernel-v1 | **Pending** | realizar |
| Mamba | mamba.yaml | ssm-kernel-v1 | **Pending** | realizar |
| RWKV7 | rwkv7.yaml | ssm-kernel-v1 | **Pending** | realizar |
| BERT | bert.yaml | model-config-algebra-v1 | A+ | realizar |
| Whisper | whisper.yaml | model-config-algebra-v1 | **Pending** | realizar |
| Moonshine | moonshine.yaml | model-config-algebra-v1 | **Pending** | realizar |
| OpenELM | openelm.yaml | model-config-algebra-v1 | **Pending** | realizar |

---

## Appendix C: Glossary

| Term | Definition |
|------|-----------|
| **Contract** | YAML file defining equations, proof obligations, and falsification tests for a kernel, algorithm, or pipeline |
| **Kernel Contract** | Contract for a compute primitive (softmax, RMSNorm, matmul, RoPE) — Tiers 1-3 |
| **Algorithm Contract** | Contract for an ML algorithm or statistical function (R², Gini, PageRank, Platt scaling) — Tiers A1-A7 |
| **Binding** | Mapping from a contract equation to a Rust function implementation |
| **Proof Obligation** | Mathematical property that must hold for all valid inputs |
| **Falsification Gate** | Test designed to BREAK a contract — passing means "not yet falsified" |
| **Equation** | Named mathematical formula within a contract (e.g., `r_squared` in `metrics-regression-v1`) |
| **Invariant** | Property that must hold after every invocation (e.g., "MSE ≥ 0") |
| **Poka-Yoke** | Mistake-proofing via type system (sealed constructors, newtypes) |
| **Jidoka** | Stop-on-first-defect — P0 failure = MQS 0 |
| **MQS** | Model Qualification Score (0-1000) from apr-model-qa-playbook |
| **The One Path** | Exactly one canonical implementation for each operation |
| **Dead Code** | Duplicate implementations that will be deleted (not deprecated) |
| **Contract Gate** | Runtime validation checkpoint (e.g., `contract_gate::validate_model_load_basic()`) |
| **Contract-Algorithm Isomorphism** | Kernel contracts and algorithm contracts use identical YAML structure (§14.2) |
