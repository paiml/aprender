# Aprender Specification — Single Source of Truth

**Version**: 6.0.0
**Status**: Active
**Created**: 2025-10-01
**Last Updated**: 2026-03-31
**Crates**: `aprender` (lib), `apr-cli` (bin), `apr` (alias)

> This is the **mono spec** for the Aprender project. Each section summarizes
> the component and links to its detailed specification in `components/`. No
> other specification files are authoritative.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Provable Contracts — Design Foundation](#2-provable-contracts--design-foundation)
3. [apr-cli — The Primary Interface](#3-apr-cli--the-primary-interface)
4. [Layer Tracing and Brick Profiling](#4-layer-tracing-and-brick-profiling)
5. [Stack Architecture](#5-stack-architecture)
6. [Compute Backends](#6-compute-backends)
7. [ML Algorithms](#7-ml-algorithms)
8. [APR Binary Format](#8-apr-binary-format)
9. [Model Operations](#9-model-operations)
10. [Training Pipeline](#10-training-pipeline)
11. [Inference and Serving](#11-inference-and-serving)
12. [Analysis and Debugging](#12-analysis-and-debugging)
13. [Testing with Probar](#13-testing-with-probar)
14. [Format Conversion and Tokenization](#14-format-conversion-and-tokenization)
15. [Quality Standards](#15-quality-standards)
16. [References](#16-references)
17. [Component Index](#17-component-index)

---

## 1. Overview

Aprender ("to learn" in Spanish) is a machine learning library in pure Rust.
It provides the TOP 10 ML algorithms plus advanced modules (time series, NLP,
Bayesian, GLM, graph, audio) with 12,974 tests and 96.35% line coverage.

**The `apr` CLI is the primary interface.** Every capability — inference,
training, profiling, format conversion, debugging — is accessed through
`apr` subcommands. The library crate exists to power the CLI, not the reverse.

**Core principles**:

- **Contracts First**: YAML contract → scaffolding → implement → verify.
  Code without a contract cannot compile.
- **CLI First**: Every feature ships as an `apr` command. If it's not in
  the CLI, it doesn't exist for users.
- **Observable by Default**: Layer tracing (`--trace`) and brick profiling
  (`--profile`) are built into every inference and training command. Probar
  validates correctness via visual regression and GUI coverage.
- **Backend Agnostic**: SIMD/wgpu/CUDA/cuBLAS/WASM dispatch is automatic.
  Correctness proven equivalent across all backends via contracts.
- **Sovereign AI**: All operations run locally. `--offline` blocks network.

---

## 2. Provable Contracts — Design Foundation

**Detail**: [components/provable-contracts.md](components/provable-contracts.md)

Every component follows the contracts-first workflow:

```
arXiv Paper → Equations (YAML) → Contract (proof obligations)
  → Scaffolded traits → Implementation → Verification (L1–L5)
```

| Level | Guarantee | Mechanism |
|-------|-----------|-----------|
| L5 | True for ALL inputs | Lean 4 theorem (provable-contracts repo) |
| L4 | True for inputs ≤ N | Kani bounded model checking (12 harnesses) |
| L3 | True for ~10K inputs | `#[contract]` debug_assert (38 functions) |
| L2 | Specific edge cases | Falsification tests + probar (73 contracts) |
| L1 | By construction | Rust type system (all public APIs) |

Delete a contract YAML → compile error. The compiler itself enforces
correctness — `build.rs` → env vars → `#[contract]` proc macro →
`compile_error!()` on missing contract. Zero runtime cost in release.

---

## 3. apr-cli — The Primary Interface

**Detail**: [components/cli.md](components/cli.md)

`apr` is the single entry point for all ML operations. It is not a thin
wrapper — it is the **product**. Features that exist only as library APIs
without CLI exposure are considered incomplete.

### Command Groups (53 commands)

| Group | Commands | Priority |
|-------|----------|----------|
| **Runtime** | run, chat, serve, pull | P0 — daily use |
| **Observe** | trace, profile, cbtop, probar | P0 — always available |
| **Model Ops** | finetune, prune, distill, merge, quantize | P0 |
| **Analysis** | inspect, debug, validate, diff, tensors, lint | P0 |
| **Training** | train, tune, monitor, runs, experiment | P1 |
| **Evaluation** | eval, bench, qa, parity, qualify | P1 |
| **Tools** | export, import, convert, compile, tokenize | P1 |
| **Visualization** | tui, cbtop, ptx-map, ptx, hex, tree, flow | P2 |

### Global Flags

Every command respects: `--json`, `--verbose`, `-q/--quiet`, `--offline`,
`--skip-contract`. The `--trace` and `--profile` flags are on `run`,
`chat`, and `serve` commands only.

### CLI-First Development Rule

```
1. Design the `apr <command>` UX first (flags, output format, exit codes)
2. Write the provable contract for the underlying operation
3. Implement the library function
4. Wire the CLI command to the library
5. Test via `apr probar` (visual regression) + unit tests
```

If a library function has no corresponding `apr` subcommand, file a ticket.

---

## 4. Layer Tracing and Brick Profiling

**Detail**: [components/tracing.md](components/tracing.md)

Every inference and training command supports deep observability via two
complementary systems: **layer tracing** (renacer) and **brick profiling**
(trueno BrickProfiler).

### 4.1 Layer Tracing (renacer)

Inference is a state machine with 8 traced steps:

```
TOKENIZE → EMBED → TRANSFORMER_BLOCK (×N) → LM_HEAD → SAMPLE → DECODE
                                                        ↑
                                              KERNEL_LAUNCH (GPU)
                                              BRICK_PROFILE (compute)
```

Each step records `TensorStats` (min, max, mean, std, NaN/Inf detection).

```bash
# Basic layer trace
apr run model.apr --prompt "hello" --trace

# Verbose with tensor values
apr run model.apr --prompt "hello" --trace-level payload

# Filter specific steps
apr run model.apr --prompt "hello" --trace-steps "embed,attention,sample"

# Export to JSON for analysis
apr run model.apr --prompt "hello" --trace-output trace.json
```

### 4.2 BrickTracer (renacer)

Automatic escalation from measurement to tracing when CV > 15% or
efficiency < 25%. Categorizes time into mmap/futex/ioctl/compute.

### 4.3 Brick Profiling (trueno BrickProfiler)

Per-kernel microsecond-resolution timing for every compute brick:

```bash
# Roofline analysis with per-brick breakdown
apr profile model.apr --granular --format flamegraph

# CI assertion mode
apr profile model.apr --ci --assert-throughput 30 --assert-p99 50ms

# Compare against Ollama baseline
apr profile model.apr --compare-hf --perf-grade

# Focus on specific compute area
apr profile model.apr --focus attention --energy
```

**Contract C-GDP-001**: Profiling and CUDA graph replay are mutually
exclusive — profiling uses eager decode path to instrument each brick
individually.

### 4.4 cbtop — Live ComputeBrick Monitor

`apr cbtop` monitors brick pipeline live with `--brick-score` assertions.

---

## 5. Stack Architecture

**Detail**: [components/stack-architecture.md](components/stack-architecture.md)

| Crate | Role | Depends On |
|-------|------|------------|
| **provable-contracts** | Contract specification + verification | — |
| **trueno** | Compute primitives + BrickProfiler | provable-contracts |
| **aprender** | ML algorithms, training, APR format | trueno |
| **realizar** | Inference, serving, KV cache, layer tracing | trueno, aprender |
| **entrenar** | Distributed training, cuBLAS backward | trueno, aprender |
| **renacer** | System tracing, BrickTracer, OTLP export | — |
| **probar** | Visual regression, GUI coverage, E2E testing | — |
| **decy** | C/CUDA → Rust transpiler | — |

**apr-cli** depends on all of the above and is the **primary deliverable**.

---

## 6. Compute Backends

**Detail**: [components/compute-backends.md](components/compute-backends.md)

Two dispatch layers provide compute across all targets:

**Layer 1 — trueno `Backend` enum (CPU + wgpu):**

| Backend | Targets | ISA |
|---------|---------|-----|
| **AVX2+FMA** | x86_64 | 256-bit SIMD + fused multiply-add |
| **AVX-512** | x86_64 (server) | 512-bit SIMD |
| **SSE2** | x86_64 (baseline) | 128-bit SIMD |
| **NEON** | aarch64 | 128-bit ARM SIMD |
| **WasmSIMD** | Browsers | 128-bit SIMD128 |
| **GPU** | All vendors | wgpu WGSL shaders (39 in trueno) |

**Layer 2 — GPU kernel dispatch (realizar + trueno-gpu):**

| Path | When | Targets |
|------|------|---------|
| **Custom PTX** | M=1 decode (bandwidth-bound) | NVIDIA sm_50+ |
| **cuBLAS** | M>1 prefill (compute-bound, tensor cores) | NVIDIA |
| **cuBLASLt** | FP8 E4M3 GEMM | NVIDIA sm_89+ |
| **wgpu WGSL** | Cross-vendor GPU | AMD, Intel, Apple, NVIDIA |

cuBLAS and PTX coexist within the same inference pass. Backend
correctness validated by equivalence contracts and parity gate.

---

## 7. ML Algorithms

**Detail**: [components/ml-algorithms.md](components/ml-algorithms.md)

TOP 10: LinearRegression, LogisticRegression, DecisionTree, RandomForest,
GBM, NaiveBayes, KNN, SVM, KMeans, PCA. Advanced: ARIMA, NLP, Bayesian,
GLM, ICA, Graph, Neural Networks. Three-tier API: Estimator traits →
Optimizers → Trueno primitives.

---

## 8. APR Binary Format

**Detail**: [components/format.md](components/format.md)

Zero-copy binary format. Header (32B) + JSON metadata + tensor index +
64-byte aligned tensor data + checksum footer. LZ4/ZSTD compression.
Sharding for >2GB. WASM streaming.

---

## 9. Model Operations

**Detail**: [components/merge.md](components/merge.md),
[finetune.md](components/finetune.md),
[distill.md](components/distill.md),
[prune.md](components/prune.md),
[quantize.md](components/quantize.md)

| Operation | `apr` Command | Methods |
|-----------|--------------|---------|
| Merge | `apr merge` | average, weighted, slerp, ties, dare |
| Fine-Tune | `apr finetune` | auto, full, LoRA, QLoRA |
| Distill | `apr distill` | standard, progressive, ensemble |
| Prune | `apr prune` | magnitude, structured, depth, Wanda, SparseGPT |
| Quantize | `apr quantize` | fp16, int8, int4, Q4K |

All model ops support `--trace` for layer tracing and `--profile` for
brick profiling. Results validated by `apr probar` golden snapshots.

---

## 10. Training Pipeline

**Detail**: [components/train.md](components/train.md),
[tune.md](components/tune.md), [data.md](components/data.md),
[checkpoints.md](components/checkpoints.md)

```bash
apr train plan config.yaml          # Validate, estimate resources
apr train apply config.yaml         # Execute with layer tracing
apr train watch --hang-timeout 30m  # Auto-restart + anomaly detection
apr tune --strategy tpe --trials 50 # HPO with BrickTracer
apr data validate dataset.jsonl     # Data quality checks
```

Training commands emit BrickTracer spans for each backward pass. Anomalies
(gradient explosion, NaN loss) trigger automatic escalation to full
syscall tracing via renacer.

---

## 11. Inference and Serving

**Detail**: [components/inference.md](components/inference.md),
[serve.md](components/serve.md)

```bash
# Inference with full tracing
apr run model.apr --prompt "hello" --trace --profile

# Chat with live brick monitoring
apr chat model.apr --trace-level layer

# Serve with OTLP export to Jaeger
apr serve model.apr --otlp-endpoint http://localhost:4317
```

All inference uses realizar. Layer tracing and brick profiling are
available on every inference path. `apr serve` exports W3C Trace Context
spans via OTLP for distributed tracing with Jaeger/Tempo.

---

## 12. Analysis and Debugging

**Detail**: [components/inspection.md](components/inspection.md),
[qa.md](components/qa.md), [profile.md](components/profile.md)

| Command | Purpose | Tracing |
|---------|---------|---------|
| `apr inspect` | Metadata, vocab, structure | — |
| `apr validate` | Integrity, 100-point quality score | — |
| `apr diff` | Two-model comparison | — |
| `apr trace` | Layer-by-layer state machine analysis | Full |
| `apr profile` | Roofline, flamegraph, energy, CI mode | BrickProfiler |
| `apr qa` | 8+ falsifiable gates + golden output | BrickTracer |
| `apr cbtop` | Live ComputeBrick pipeline monitor | Auto-escalation |
| `apr probar` | Visual regression of layer activations | Snapshot |

**Debugging order**: `apr qa` → `apr trace` → `apr profile` → `apr cbtop`
→ `apr diff` → read code. Tools before code (GH-202 lesson).

---

## 13. Testing with Probar

**Detail**: [components/testing.md](components/testing.md)

Probar (`jugar-probar` on crates.io) provides three testing capabilities
used throughout the stack:

### 13.1 Visual Regression Testing

```bash
# Capture golden activation snapshots
apr probar model.apr --golden golden/ --format png

# Compare against golden reference
apr probar model.apr --golden golden/ --layer "attn.*"

# Export JSON for programmatic comparison
apr probar model.apr --golden golden/ --format json
```

Every model operation that changes weights (merge, finetune, prune,
quantize) runs `apr probar` to validate activations against golden
snapshots.

### 13.2 GUI Coverage (TUI/WASM Testing)

```rust
use jugar_probar::prelude::*;

let mut gui = gui_coverage! {
    buttons: ["run", "stop", "trace"],
    screens: ["model_select", "inference", "profile"]
};
assert!(gui.meets(80.0)); // 80% GUI coverage required
```

Used to test `apr tui` and WASM deployments.

### 13.3 Probar in CI/CD

```bash
# Tier 2 gate: probar golden regression
apr probar model.apr --golden tests/golden/ --assert

# Tier 3 gate: full visual regression + brick profiling
apr probar model.apr --golden tests/golden/ --profile --assert
```

Probar golden snapshots are committed to the repo. Any activation
divergence beyond tolerance fails CI. Combined with BrickProfiler to
catch both correctness and performance regressions simultaneously.

---

## 14. Format Conversion and Tokenization

**Detail**: [components/format-conversion.md](components/format-conversion.md),
[compile.md](components/compile.md),
[tokenize.md](components/tokenize.md)

```bash
apr export model.apr --format gguf -o model.gguf
apr import hf://org/model -o model.apr
apr convert model.safetensors --quantize q4k
apr compile model.apr --target aarch64
apr tokenize train corpus.txt --vocab-size 32000
```

All format conversions support `--trace` for tensor layout verification.

---

## 15. Quality Standards

| Metric | Target | Current |
|--------|--------|---------|
| Test count | — | 12,972 |
| Line coverage | ≥95% | 96.35% |
| Mutation score | ≥85% | 85.3% |
| TDG score | ≥95/100 | 95.2/100 |
| Provable contracts | — | 171 validated |
| Kani harnesses | — | 985 passing |
| Lean 4 theorems | — | 64 (0 sorry) |
| Probar golden tests | — | All passing |
| `unwrap()` | Banned | clippy.toml enforced |

**Tiered gates**:

| Tier | Time | Includes |
|------|------|----------|
| tier1 | <1s | fmt, clippy, check |
| tier2 | <5s | tests, probar golden regression |
| tier3 | 1-5min | coverage, brick profiling, full probar |
| tier4 | CI | pmat, contract verification, Kani |

---

## 16. References

1. Ivanov et al. (2021) "Data Movement Is All You Need." MLSys.
2. Chatterjee et al. (2025) "ProofWright: Agentic Formal Verification
   of CUDA." arXiv:2511.12294.
3. Arora et al. (2025) "TensorRight: Automated Verification of Tensor
   Graph Rewrites." arXiv:2511.17838.
4. Mace et al. (2015) "Pivot Tracing: Dynamic Causal Monitoring." SOSP.
5. Sigelman et al. (2010) "Dapper, a Large-Scale Distributed Systems
   Tracing Infrastructure." Google.
6. Curtsinger & Berger (2013) "Stabilizer: Statistically Sound Performance
   Evaluation." ASPLOS.
7. Gond et al. (2026) "LLM-42: Enabling Determinism in LLM Inference."
   arXiv:2601.17768.

---

## 17. Component Index

| Component | File | Status |
|-----------|------|--------|
| Provable Contracts | [components/provable-contracts.md](components/provable-contracts.md) | Active |
| CLI (apr) | [components/cli.md](components/cli.md) | Active |
| Tracing & Profiling | [components/tracing.md](components/tracing.md) | Active |
| Testing (Probar) | [components/testing.md](components/testing.md) | Active |
| Stack Architecture | [components/stack-architecture.md](components/stack-architecture.md) | Active |
| Compute Backends | [components/compute-backends.md](components/compute-backends.md) | Active |
| ML Algorithms | [components/ml-algorithms.md](components/ml-algorithms.md) | Active |
| APR Binary Format | [components/format.md](components/format.md) | Active |
| Merge | [components/merge.md](components/merge.md) | Active |
| Fine-Tune | [components/finetune.md](components/finetune.md) | Active |
| Distill | [components/distill.md](components/distill.md) | Active |
| Prune | [components/prune.md](components/prune.md) | Active |
| Quantize | [components/quantize.md](components/quantize.md) | Active |
| Checkpoints | [components/checkpoints.md](components/checkpoints.md) | Active |
| Train | [components/train.md](components/train.md) | Active |
| Tune (HPO) | [components/tune.md](components/tune.md) | Active |
| Serve | [components/serve.md](components/serve.md) | Active |
| Eval | [components/eval.md](components/eval.md) | Active |
| Data | [components/data.md](components/data.md) | Active |
| Export / Import | [components/format-conversion.md](components/format-conversion.md) | Active |
| Inference | [components/inference.md](components/inference.md) | Active |
| Inspection | [components/inspection.md](components/inspection.md) | Active |
| Compile | [components/compile.md](components/compile.md) | Active |
| QA | [components/qa.md](components/qa.md) | Active |
| Profile | [components/profile.md](components/profile.md) | Active |
| Tokenize | [components/tokenize.md](components/tokenize.md) | Active |
| CI Infrastructure | [components/ci-infrastructure.md](components/ci-infrastructure.md) | Active |
