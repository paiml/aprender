# Stack Architecture

Version: 1.0
Status: proposed
Date: 2026-04-09

**Version**: 1.0.0
**Status**: Active
**Parent**: [aprender-spec.md](../aprender-spec.md) §2

---

## 1. Overview

The sovereign Rust AI stack comprises six components with strict responsibility
boundaries enforced by provable contracts. Each crate has exactly one job —
violations are compiler errors, not policy. Contracts are written BEFORE code.

```
┌─────────────────────────────────────────────────────┐
│                    apr-cli (bin)                     │
│              User-facing CLI commands               │
├─────────────┬─────────────┬─────────────────────────┤
│  aprender   │  realizar   │       entrenar          │
│  (training) │ (inference) │ (distributed training)  │
├─────────────┴─────────────┴─────────────────────────┤
│                   trueno (compute)                   │
│          SIMD · wgpu · CUDA PTX · cuBLAS · WASM     │
├─────────────────────────────────────────────────────┤
│            provable-contracts (verification)         │
│    YAML specs · Kani BMC · Lean 4 · #[contract]     │
└─────────────────────────────────────────────────────┘

       decy (standalone) ─── C/CUDA → Rust transpiler
```

See [provable-contracts.md](provable-contracts.md) for the contract-first
design methodology that governs all crate boundaries and kernel equivalence.

---

## 2. Crate Responsibilities

### 2.1 provable-contracts — Verification Foundation

The true foundation. Every kernel, backend dispatch, and tensor transformation
starts as a YAML contract with equations, proof obligations, and Kani harnesses.
See [provable-contracts.md](provable-contracts.md).

- 171 YAML contracts with mathematical specifications
- 985 Kani bounded model checking harnesses
- 64 Lean 4 theorems (0 sorry)
- `#[contract]` proc macro for compile-time enforcement
- `pv` CLI (37 subcommands) for contract lifecycle
- Binding registries for 26 downstream repos

### 2.2 trueno — Compute Primitives

The compute layer. Provides `Vector`, `Matrix`, and backend-agnostic ops.

- SIMD-accelerated CPU operations (AVX2, NEON, auto-vectorized)
- wgpu GPU compute (WGSL shaders → Vulkan/Metal/DX12/WebGPU)
- Custom CUDA PTX kernels (fused dequant+GEMV, RMSNorm, RoPE)
- cuBLAS FFI for peak NVIDIA GEMM performance
- WASM target for browser deployment
- Quantized tensor types (Q4K, Q5K, Q6K, Q8K)
- Always use crates.io published version — never git dependency

**Verification**:
```bash
cargo search trueno           # Latest on crates.io
cargo tree | grep trueno      # Currently pinned version
```

### 2.3 aprender — ML Library and Training

The ML algorithm layer. Training only — never inference or serving.

- TOP 10 ML algorithms + advanced modules
- Three-tier API (Estimator traits → Optimizers → Trueno primitives)
- APR binary format read/write
- Neural network building blocks (Linear, Attention, Normalization)
- Text processing (tokenizers, chat templates, stemming)
- Graph algorithms (Dijkstra, PageRank, community detection)
- Bayesian inference, GLMs, time series (ARIMA)

**Forbidden in aprender**: HTTP servers, KV caches, GGUF loading,
model serving, inference loops.

### 2.4 realizar — Inference and Serving

The production inference engine. Uses trueno for compute.

- Model loading (GGUF, SafeTensors, APR — read-only)
- Autoregressive inference with KV cache (PagedAttention)
- OpenAI-compatible API server (axum)
- Chat session management
- Fused dequant+matmul kernels
- FFN gate+up kernel fusion (SwiGLU)
- GPU-resident model serving
- Inference tracing and profiling

**Performance targets**:

| Model | CPU tok/s | GPU tok/s | Memory |
|-------|-----------|-----------|--------|
| 1B Q4K | 100+ | 500+ | 600MB |
| 7B Q4K | 30+ | 150+ | 4GB |
| 13B Q4K | 15+ | 80+ | 8GB |

### 2.5 entrenar — Distributed Training

GPU-accelerated training with cuBLAS backward pass.

- cuBLAS GEMM parity (CPU vs GPU verified)
- Multi-GPU data-parallel training
- Distributed coordinator/worker architecture
- Multi-adapter concurrent training (CUDA MPS)

### 2.6 decy — C/CUDA → Rust Transpiler

Standalone tool. Converts open-source CUDA C++ to safe Rust.

- Parses C/CUDA AST → HIR → ownership analysis → Rust codegen
- Minimizes `unsafe` blocks (<5 per 1000 LOC)
- Enables harvesting battle-tested CUDA kernels (FlashAttention,
  CUTLASS) into native Rust that trueno ships directly
- Eliminates C++ build toolchain dependency

```
CUDA C++ kernel (open source)
        │
    decy transpile
        │
Safe Rust + inline PTX (minimal unsafe)
        │
   trueno backend kernel (pure Rust)
```

---

## 3. Dependency Graph

```
apr-cli ──► aprender ──► trueno ──► provable-contracts
       ├──► realizar ──► trueno
       │            └──► aprender (read-only APR)
       └──► entrenar ──► trueno
                    └──► aprender
```

**Publishing order** (strict — each depends on prior):
1. `provable-contracts`
2. `trueno`
3. `aprender`
4. `realizar`
5. `entrenar`
6. `apr-cli`

Always use `batuta stack release` for coordinated publishing.

```bash
batuta stack release --all --bump minor --publish -y  # Full stack
batuta stack versions                                  # Current vs published
batuta stack drift                                     # Detect version skew
```

---

## 4. Boundary Enforcement

### 4.1 Hard Rules

| Rule | Enforcement |
|------|-------------|
| aprender never does inference | No `generate()` / `forward()` — deleted in GH-247 |
| realizar never trains | No `backward()` / `Optimizer` imports |
| trueno has no ML semantics | Pure math — no "model" or "layer" concepts |
| APR row-major only | `LayoutContract` validated at import boundary |
| No banned crates | Workspace `deny.toml` blocks serde, rayon, tokio, etc. |

### 4.2 The Realizar-First Rule

All inference **must** go through realizar. Aprender's old
`Qwen2Model::generate()` / `forward()` have been deleted (Round 47, GH-247).

```rust
// WRONG — bypasses realizar, 0.3 tok/s
use aprender::models::Qwen2Model;
let output = model.generate(&input_ids, 32, 0.7, 0.9);

// CORRECT — uses realizar, 225+ tok/s
use realizar::Model;
let model = Model::load_safetensors(&path)?;
let output = model.generate(&input_ids, config)?;
```

```bash
# BEST — apr CLI uses realizar automatically
apr run model.safetensors --prompt "What is 2+2?" --max-tokens 32
```

---

## 5. Cross-Project Patterns

### 5.1 Tensor Layout Contract (LAYOUT-001/002)

All projects use row-major layout. GGUF column-major data is transposed once
at the import boundary. Source of truth: `contracts/tensor-layout-v1.yaml`.

```
GGUF (col-major) ──[TRANSPOSE]──► APR (row-major) ──► realizar ──► output
SafeTensors (native) ────────────► APR (row-major) ──► realizar ──► output
```

### 5.2 Provable Contracts

YAML contract files in `contracts/` define equations, invariants, and
falsification tests. Validated at compile time via `#[contract]` proc macro
and at runtime via `debug_assert!()`. See
[provable-contracts.md](provable-contracts.md) for the full methodology.

### 5.3 Observability Crates (First-Class)

| Crate | Role | Integration |
|-------|------|-------------|
| **renacer** | System tracing, BrickTracer, OTLP export | `--trace` on all `apr` commands |
| **probar** (`jugar-probar`) | Visual regression, GUI coverage, E2E testing | `apr probar` command |

These are not optional — they are core to the development workflow.
See [tracing.md](tracing.md) and [testing.md](testing.md).

**renacer** provides: layer tracing (8-step inference state machine),
BrickTracer (automatic anomaly escalation with syscall breakdown),
W3C Trace Context spans, OTLP export to Jaeger/Tempo, renacer-core
zero-dependency primitives (SpanRecord, LazySpan, LamportClock).

**probar** provides: golden activation snapshots for model operations,
pixel-level GUI coverage heatmaps, TUI snapshot testing, YAML playbook
state machine testing with M1–M5 mutation classes, PTX static analysis.

### 5.4 Supporting Ecosystem

| Crate | Role |
|-------|------|
| **alimentar** | Data loading, Arrow RecordBatch, drift detection |
| **pacha** | Artifact registry, lineage tracking, BLAKE3 integrity |
| **certeza** | Test methodology (tiered TDD-X, mutation, property) |
| **repartir** | Distributed compute (CPU/GPU/remote executors) |
| **batuta** | Stack orchestration, RAG oracle, cross-project pipelines |
| **albor** | 350M code completion model (stack dogfood) |
| **simular** | Simulation engine (physics, Monte Carlo, ML) |

### 5.4 Feature Flags

```toml
[features]
inference = ["realizar", "tokio", "axum"]  # Default-enabled in apr-cli
gpu = ["trueno-gpu"]                        # Optional CUDA support
wgpu = ["trueno/wgpu"]                      # Optional wgpu support
```

---

## 6. Debugging Workflow

**Always start with apr tools, not code reading** (GH-202 lesson):

| Step | Tool | Purpose |
|------|------|---------|
| 1 | `apr qa model.apr` | Falsifiable QA gates (catches 80%) |
| 2 | `apr tensors model.apr` | Tensor shapes and statistics |
| 3 | `apr diff model.apr ref.gguf` | Compare against known-good |
| 4 | `apr validate model.apr` | Format/metadata integrity |
| 5 | Read code | Only after tools fail to diagnose |
