# CLAUDE.md

## Project Overview

**Trueno** — Rust library for unified high-performance compute:
CPU SIMD (SSE2/AVX/AVX2/AVX-512/NEON/WASM SIMD128),
NVIDIA CUDA (pure Rust PTX via `trueno-gpu`, no nvcc),
cross-platform GPU (wgpu: Vulkan/Metal/DX12/WebGPU).

**Canonical spec:** `docs/specifications/trueno-spec.md`
(ONE spec, sub-specs in `docs/specifications/sub/`).

**Core Principles**: Write once optimize everywhere, runtime dispatch,
zero unsafe in public API, every optimization must prove ≥10% speedup,
>90% test coverage, **contract-first** (no kernel ships without a YAML
contract in `contracts/`).

## Development Commands

```bash
# Build
cargo build                                    # standard
cargo build --all-features                     # all features
cargo build --target wasm32-unknown-unknown     # WASM

# Test
cargo test --all-features                      # all tests
cargo test property_tests                      # proptest
cargo test backend_equivalence                 # cross-backend
cargo test --test integration_tests            # integration

# Coverage — ONLY use make coverage (never cargo llvm-cov or tarpaulin)
make coverage                                  # MUST be ≥90%
make coverage-check                            # exits with error if <90%

# Lint
cargo clippy --all-features -- -D warnings
cargo fmt -- --check

# Bench
cargo bench --no-fail-fast

# Profile (Renacer v0.5.0+)
make profile                                   # benchmark profiling
make profile-flamegraph                        # flamegraph
make profile-otlp-jaeger                       # OTLP → Jaeger

# Golden trace validation
renacer --assert renacer.toml -- \
  ./target/release/examples/backend_detection

# Quality gates
pmat analyze tdg --min-grade B+
pmat repo-score . --min-score 90
cargo mutants --timeout 120 --minimum-pass-rate 80
```

## Code Search

**Use `pmat query` for code discovery, not grep/rg.**
See global CLAUDE.md for full flag reference.

```bash
pmat query "simd kernel" --limit 10
pmat query "matrix multiply" --min-grade A --exclude-tests
pmat query "gpu compute" --faults --churn -G   # full audit
```

## Backend Story Policy (NEVER VIOLATE)

Every operation MUST work on ALL backends: Scalar, SSE2,
AVX/AVX2, AVX-512, NEON, WASM SIMD128, CUDA, wgpu.
**NO EXCEPTIONS.** If GPU acceleration isn't beneficial,
the GPU method must fall back to CPU.

When adding a new operation:
1. **Write contract FIRST** (`contracts/my-op-v1.yaml`)
2. Add to `VectorBackend` trait in `src/backends/mod.rs`
3. Implement in ALL backend modules
4. Add WGSL shader if GPU-accelerable
5. Add sync + async device methods
6. Add integration test in `tests/backend_story.rs`

Enforcement: `tests/backend_story.rs` + CI.

## Provable-Contracts Integration

**Contract-first design:** `build.rs` reads
`../provable-contracts/contracts/trueno/binding.yaml`
(38/38 bindings implemented), sets `CONTRACT_*` env vars,
enforces **AllImplemented** policy — any `not_implemented`
binding fails the build.

**Escape-proof pipeline:** Equation → Lean proof →
YAML validation → build.rs codegen → `#[contract]` proc
macro → FALSIFY tests.
See `docs/specifications/sub/contracts.md` for details.

```bash
pv lint contracts/                  # 7-gate quality check
pv verify-bindings                  # check binding registry
pv score                            # PVScore metric
```

## Quality Standards

**Every commit**: clippy clean, all tests pass,
≥90% coverage (`make coverage`), rustfmt, PMAT TDG ≥B+.

**Every PR**: Tests for new code (unit, property, backend
equivalence, mutation, benchmark), rustdoc updated,
benchmarks prove ≥10% improvement.

**Test categories**: Unit (empty, NaN, subnormal),
property-based (proptest), backend equivalence
(f32 tolerance < 1e-5), mutation (≥80% kill rate),
benchmarks.

## LAYOUT-002: Row-Major Mandate (Q4K/Q6K)

The Sovereign AI Stack uses **ROW-MAJOR exclusively** for
APR/GGUF data. Column-major kernels exist for internal
BLAS-style ops only. Garbage inference output = wrong kernel
layout. Aprender handles GGUF→APR transpose during import.
See `book/src/advanced/phase15-fused-q4k.md` for fused
dequant+dot spec.

## Safety Rules

- `unsafe` ONLY in backend implementations, never in public API
- Every `unsafe` block needs a safety comment
- SIMD intrinsics wrapped in `#[target_feature]` functions
- Always handle SIMD remainder with scalar fallback
- GPU only for >100K elements (PCIe transfer ~0.5ms)
- f32 comparisons: tolerance < 1e-5

## trueno-gpu (Pure Rust CUDA)

Pure Rust PTX generation — no nvcc, no LLVM. Kernels: GEMM
(naive/tiled/tensor core), Softmax, LayerNorm, Attention
(FlashAttention-style), Q4_K dequantization.

```bash
cargo test -p trueno-gpu --features cuda   # requires GPU
cargo test -p trueno-gpu property_tests    # no hardware
```

## WGPU Inference & Training

**Inference**: `WgslForwardPass` — RMSNorm, GEMV, SiLU, RoPE.
27.6 tok/s on Radeon Pro W5700X.

**Training**: 9 shaders in `src/backends/gpu/shaders/backward.rs`
(6 backward + adamw + nf4_dequant + cross_entropy_forward).
All FALSIFY tests pass. Full training loop on AMD/Intel/Apple.

## Blackwell (trueno#200, trueno#203)

**JIT Bug (#200)**: `cuModuleLoadDataEx` fails on sm_121 during
active GPU work. Forward kernels work after pre-warming;
backward kernels crash. Inference unaffected (cuBLAS/SIMD).

**Fix (#203)**: Dimension-independent kernels (M,K,N as runtime
params, ~15 types vs 50+ variants) + pre-compiled cubin pipeline
(`build.rs → nvcc → include_bytes!()` → zero JIT at runtime).

## Stack Search

```bash
batuta oracle --rag "your question here"
batuta oracle --rag-index               # reindex
```
