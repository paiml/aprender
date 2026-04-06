# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL: Contract-First Design

**NEVER write code before writing a provable contract.**

All code changes MUST have a corresponding contract (YAML in ../provable-contracts/contracts/<project>/ or .pmat-work/<TICKET>/contract.json) BEFORE implementation. This is enforced by `pmat comply` CB-1400.

- Use `pmat comply check` to verify contract coverage
- Minimum verification level: L1 (recommended L3+)
- See docs/agent-instructions/provable-contract-first-agents.md for the full workflow

## Project Overview

**Entrenar** is a Rust training and optimization library for neural networks, part of the PAIML stack. It provides
autograd, optimizers, quantization (QAT/PTQ), LoRA/QLoRA, model merging (TIES/DARE/SLERP), and knowledge distillation.

**Status:** Specification phase - implementation not yet started.

**Stack Dependencies:**
- `trueno` - SIMD-accelerated tensor operations (compute layer)
- `realizar` - GGUF model I/O (inference only)
- `aprender` - Loss functions, APR format (training checkpoints)

**Critical Constraint:** Entrenar depends on backward propagation operations that do not yet exist in Trueno. Phase 1
cannot start until `trueno/src/ops/backward.rs` is implemented.

## LAYOUT-002: Row-Major Mandate

**All tensors in entrenar use ROW-MAJOR layout.** This matches the Sovereign AI Stack convention.

### Training Checkpoint Format

When saving model checkpoints:
- **Use APR format** (aprender's native format) - already row-major
- **Never save as GGUF** - GGUF is column-major, only for external consumption

### Gradient Tensors

Gradient tensors during backprop must maintain row-major layout:
```rust
// Gradient accumulation preserves layout
let grad = tensor.grad();  // Row-major, matches forward tensor
```

### Model Export for Inference

When exporting trained models for inference:
```bash
# Export to APR (row-major, native) - use for realizar
entrenar export model.apr --format apr

# Export to GGUF (column-major) - aprender handles transpose
entrenar export model.gguf --format gguf  # Internally calls apr export
```

See `aprender/CLAUDE.md` LAYOUT-002 and `realizar/CLAUDE.md` LAYOUT-002 for full details.

## Architecture

### Core Type System

The foundation is a tape-based autograd system with lifetime-tracked gradient computation:

```rust
pub struct Tensor<'g, T> {
    data: trueno::Tensor<'g, T>,
    grad: Option<trueno::Tensor<'g, T>>,
    op: Option<Box<dyn BackwardOp<'g, T>>>,
}
```

The lifetime `'g` ensures gradient tape validity throughout backpropagation.

### Layer Structure

1. **Autograd Engine** (`src/autograd/`) - Tape-based eager-mode differentiation with backward operations
2. **Optimizers** (`src/optim/`) - SGD, Adam, AdamW, learning rate schedulers
3. **LoRA** (`src/lora/`) - Low-rank adaptation with QLoRA (4-bit base) support
4. **Quantization** (`src/quant/`) - QAT (fake quantize + STE) and PTQ (calibration)
5. **Model Merging** (`src/merge/`) - TIES, DARE, SLERP methods
6. **Distillation** (`src/distill/`) - Knowledge distillation with temperature-scaled softmax
7. **Declarative Config** (`src/config/`) - Ludwig-inspired YAML → trained model pipeline
8. **Training Loop** (`src/train/`) - High-level trainer with epoch/step abstractions

## Commands

### Quality Gates (Tiered TDD Workflow)

**Tier 1 (Fast <5s)** - Run before every commit:
```bash
make tier1    # Format, clippy, unit tests
```

**Tier 2 (Integration <30s)** - Run before push:
```bash
make tier2    # Tier 1 + integration tests
```

**Tier 3 (Full <5m)** - Run before PR:
```bash
make tier3    # Tier 1+2 + property tests
```

**Pre-Commit** - All quality gates:
```bash
make pre-commit    # Tier 1 + PMAT TDG check
```

**Full CI Pipeline** - Complete validation:
```bash
make ci    # Tier 3 + coverage + mutants + PMAT + security
```

### Development

```bash
# Build
make build              # Debug build
make release            # Release build

# Testing
make test               # Fast tests
cargo test test_name    # Specific test

# Code Quality
make lint               # Clippy
make format             # Format code
make check              # Type check

# Coverage (>90% required)
make coverage           # Generate HTML report
make coverage-clean     # Clean coverage data

# Mutation Testing (>80% kill rate required)
make mutants            # Full mutation testing
make mutants-quick      # Quick check on git diff

# Dependency Security
make deny-check         # Check for vulnerabilities

# Clean
make clean              # Remove build artifacts
```

## Code Search (pmat query)

**NEVER use grep or rg for code discovery.** Use `pmat query` instead -- it returns quality-annotated, ranked results
with TDG scores and fault annotations.

```bash
# Find functions by intent
pmat query "gradient computation" --limit 10

# Find high-quality code
pmat query "optimizer step" --min-grade A --exclude-tests

# Find with fault annotations (unwrap, panic, unsafe, etc.)
pmat query "training loop" --faults

# Filter by complexity
pmat query "quantization" --max-complexity 10

# Cross-project search (e.g., find trueno backward ops)
pmat query "backward operation" --include-project ../trueno

# Search across the stack
pmat query "LoRA adapter" --include-project ../aprender
pmat query "GGUF export" --include-project ../realizar

# Git history search (find code by commit intent via RRF fusion)
pmat query "fix gradient computation" -G
pmat query "autograd refactor" --git-history

# Enrichment flags (combine freely)
pmat query "optimizer step" --churn                # git volatility (commit count, churn score)
pmat query "quantization" --duplicates             # code clone detection (MinHash+LSH)
pmat query "training loop" --entropy               # pattern diversity (repetitive vs unique)
pmat query "backward pass" --churn --duplicates --entropy --faults -G  # full audit
```

### Ticket-Based Development (Required)

**ALL WORK MUST BE TRACKED VIA TICKETS** (ENT-001 through ENT-040).

Roadmap is tracked in `roadmap.yaml`. To start work:

1. **Find next ticket** in roadmap.yaml (status: pending)
2. **Update status** to `in-progress`
3. **Implement** following EXTREME TDD
4. **Update status** to `complete` with actual hours
5. **Commit** changes mentioning ticket ID

**Example workflow for ENT-002 (Matmul backward)**:

```bash
# 1. Edit roadmap.yaml: Set ENT-002 status to in-progress
vim roadmap.yaml

# 2. Implement with TDD
make tier1              # Fast feedback loop
cargo test matmul       # Specific test

# 3. Verify quality gates
make tier3              # Full validation
make coverage           # Check >90% coverage
make mutants-quick      # Quick mutation check

# 4. Update roadmap.yaml with completion
vim roadmap.yaml        # Set status: complete, actual_hours: 4

# 5. Commit
git add . && git commit -m "feat: ENT-002 matmul backward with gradient checking

- Implement matmul forward/backward operations
- Add property tests (1000+ cases)
- Gradient validation via finite difference
- Coverage: 95%, Mutation score: 85%

Closes ENT-002"
```

**Check progress**:
```bash
make roadmap-status     # View summary
grep "status: complete" roadmap.yaml | wc -l  # Count completed
```

**PMAT quality analysis** (optional):
```bash
make pmat-complexity    # Check complexity (<10)
make pmat-tdg           # Check TDG score (>90)
```

## Testing Requirements

**Zero tolerance for defects.** All code must meet these quality metrics:

- **Test Coverage:** >90% (cargo llvm-cov)
- **Mutation Kill Rate:** >80% (cargo-mutants)
- **TDG Score:** >90 (A grade via PMAT)
- **Cyclomatic Complexity:** ≤10
- **Cognitive Complexity:** ≤15
- **Gradient Error:** <1e-3 (property tests with finite difference validation)
- **Property Test Iterations:** 200K+ (proptest)

### Test-First Workflow

1. Write property test with gradient checking (finite difference validation)
2. Implement function to pass tests
3. Run mutation testing
4. Refactor if needed

Example test structure:
```rust
proptest! {
    #[test]
    fn softmax_backward_gradient_check(x in prop::collection::vec(-10.0f32..10.0, 1..100)) {
        let y = softmax(&x);
        let dy = vec![1.0; x.len()];
        let analytical = softmax_backward(&y, &dy);
        let numerical = finite_diff(|x| softmax(x).sum(), &x, 1e-5);
        prop_assert!((analytical - numerical).abs() < 1e-3);
    }
}
```

## Critical Implementation Details

### Backward Operations Required in Trueno

Before Entrenar implementation can begin, Trueno needs these operations in `trueno/src/ops/backward.rs`:

- `softmax_backward`: `∂L/∂x = softmax(x) * (∂L/∂y - dot(∂L/∂y, softmax(x)))`
- `layer_norm_backward`: Returns gradients for input, gamma, and beta
- `attention_backward`: Returns gradients for Q, K, V matrices
- `relu_backward`, `gelu_backward`, `swish_backward`: Element-wise activation gradients

All backward ops must use SIMD for small batches and GPU for large batches (5× threshold rule from Trueno).

### Quantization Output Format

- QAT/PTQ must output Realizar-compatible GGUF files
- Supports Q4_0 and Q8_0 block formats
- Per-channel and per-tensor quantization modes

### LoRA Memory Requirements

QLoRA provides 4× memory reduction vs full fine-tuning:
- Base weights: 4-bit quantized (frozen)
- Adapters: FP16/FP32 (trainable, rank r << min(d_in, d_out))
- On-the-fly dequantization during forward pass

## Declarative Configuration

Entrenar supports Ludwig-inspired YAML configs for zero-code training:

```yaml
model:
  path: llama-3-7b.gguf
  layers: [q_proj, k_proj, v_proj, o_proj]

data:
  train: train.parquet
  batch_size: 8
  auto_infer_types: true

optimizer:
  name: adam
  lr: 1e-4

lora:
  rank: 64
  alpha: 16

quantize:
  bits: 4
  symmetric: true

merge:
  method: TIES
  density: 0.2
```

Single command execution: `entrenar train config.yaml`

## Development Roadmap

**Total Estimate:** 824 hours (103 days @ 8h/day)

### Phase 1: Autograd (200h)
Core gradient computation with tape-based backpropagation. Depends on Trueno backward ops.

### Phase 2: Optimizers (120h)
SGD, Adam, AdamW with momentum and learning rate scheduling.

### Phase 3: LoRA (144h)
Low-rank adaptation with 4-bit base weights (QLoRA).

### Phase 4: Quantization (136h)
QAT with straight-through estimator, PTQ calibration, GGUF export.

### Phase 5: Model Merging (96h)
TIES (trim + sign election), DARE (dropout), SLERP (spherical interpolation).

### Phase 6: Declarative Config (64h)
YAML schema with auto-feature inference.

### Phase 7: Distillation (64h)
KD loss with temperature-scaled softmax.

All tickets tracked with PMAT (`ENT-001` through `ENT-040`).

## Pre-Commit Requirements

```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test --lib  # Fast unit tests only
pmat analyze tdg src/ --min-score 90
```

## Benchmarks (Target Performance)

| Operation | Size | Target | Backend |
|-----------|------|--------|---------|
| Matmul backward | 512×512 | 3× forward | GPU |
| Adam step | 1M params | <10ms | SIMD |
| Softmax backward | 10K | 2× forward | SIMD |
| Q4_0 quantize | 1GB | <1s | Scalar |
| LoRA merge | 7B, r=64 | <5s | SIMD |

## Sovereign AI Stack Toolchain

**CRITICAL: Python is PROHIBITED.** This project uses only pure Rust tools from the Sovereign AI Stack.

### Required Tools (use instead of Python equivalents)

| Task | Use This | NOT This |
|------|----------|----------|
| HuggingFace downloads | `batuta hf pull` | `huggingface_hub` (Python) |
| Model inference | `realizar` | `transformers` (Python) |
| Tensor operations | `trueno` | `numpy`, `torch` (Python) |
| Data processing | `aprender` | `pandas` (Python) |
| Visualization | `trueno-viz` | `matplotlib` (Python) |

### Batuta Oracle Commands

Query the Sovereign AI Stack for tool recommendations:
```bash
# HuggingFace model download
batuta hf pull model Qwen/Qwen2.5-Coder-0.5B-Instruct -o ./models/

# Search HuggingFace Hub
batuta hf search --type model "qwen coder"

# Oracle recommendations
batuta oracle stack  # Show PAIML stack components
```

## Dependencies

```toml
[dependencies]
trueno = "0.x"  # SIMD tensor operations
realizar = "0.x"  # GGUF I/O
aprender = "0.x"  # Loss functions
serde = { version = "1", features = ["derive"] }
serde_yaml = "0.9"
serde_json = "1"

[dev-dependencies]
proptest = "1.4"  # Property-based testing
cargo-mutants = "25.3"  # Mutation testing
```

**Note:** Always use latest versions from crates.io. Never use git dependencies for PAIML stack libraries.


## Stack Documentation Search

**IMPORTANT: Proactively use the batuta RAG oracle when:**
- Looking up patterns from other stack components (trueno SIMD, aprender ML, realizar inference)
- Finding cross-language equivalents (PyTorch training patterns from Python ground truth corpora)
- Understanding how autograd/backprop is implemented in other frameworks
- Researching LoRA/QLoRA, quantization, and model merging approaches

```bash
# Index all stack documentation (run once, persists to ~/.cache/batuta/rag/)
batuta oracle --rag-index

# Search across the entire stack
batuta oracle --rag "your question here"

# Entrenar-specific examples
batuta oracle --rag "autograd tape-based differentiation"
batuta oracle --rag "LoRA QLoRA parameter-efficient fine-tuning"
batuta oracle --rag "GGUF Q4_0 Q8_0 quantization format"
batuta oracle --rag "model merging TIES DARE SLERP"
batuta oracle --rag "knowledge distillation multi-teacher"
```

The RAG index (341+ docs) includes CLAUDE.md, README.md, and source files from all stack components plus Python ground
truth corpora for cross-language pattern matching.

Index auto-updates via post-commit hooks and `ora-fresh` on shell login.
To manually check freshness: `ora-fresh`
To force full reindex: `batuta oracle --rag-index --force`

## cuBLAS Training Integration Status

### GEMM Parity: VERIFIED

cuBLAS GEMM parity with trueno's fused NF4 kernels has been verified: **4 out of 5 parity tests pass**. The math is correct -- training is blocked by a Blackwell JIT bug in trueno, NOT by wrong GEMM computations.

**Parity tests**:
```bash
cargo test --features cuda --test cuda_cublas_parity -- --ignored
```

### Training Blocked by Blackwell JIT (trueno#200)

Training (backward pass) fails on Blackwell GPUs (sm_121) because backward kernels trigger PTX JIT compilation (`cuModuleLoadDataEx`) during active GPU work. The NVIDIA driver's JIT compiler crashes with `CUDA_ERROR_UNKNOWN` under these conditions.

- **Forward kernels**: Work after pre-warming (all variants loaded before training loop)
- **Backward kernels**: Crash because they compile on-demand when GPU is already active
- **Inference (`apr run`)**: NOT affected -- uses cuBLAS and SIMD paths, no custom PTX

The permanent fix is trueno#203 (dimension-independent kernels with pre-compiled cubins).

### 8 Bugs Found and Fixed

Across multiple cuBLAS integration attempts, 8 bugs were discovered and fixed:

1. **Cache key collision**: Backward kernels with different shapes mapped to same cache entry
2. **Pre-warming incomplete**: Not all forward kernel variants loaded before training started
3. **GQA variant missing**: Grouped-query attention backward kernel not generated for Qwen3 head config
4. **NF4 dequant precision**: cuBLAS path used FP32 dequant vs fused kernel's FP16, causing gradient drift
5. **Transpose convention**: cuBLAS uses column-major (Fortran), trueno uses row-major -- needed explicit transpose
6. **Stream synchronization**: Backward kernel launch raced with forward kernel completion on same stream
7. **cuBLAS handle leak**: Handle not destroyed on error path, causing OOM after many failed compilations
8. **Gradient accumulation order**: cuBLAS GEMM accumulated in different order than fused kernel, causing non-determinism

### Local trueno Dependency

Until trueno 0.4.36 (with Blackwell fixes) is published to crates.io, use a `[patch]` section:

```toml
[patch.crates-io]
trueno = { path = "../trueno" }
```

Remove this patch once trueno 0.4.36 is published.

### Ship Gate Status

**PASS**: Canary evaluation achieves 90% accuracy on shell safety classification. The classifier ships with the current fused-kernel path (15.5 tok/s). cuBLAS integration (298 tok/s) is a performance improvement, not a correctness blocker.

## WGPU Training Support (AMD/Cross-Platform GPU)

### Overview

`WgpuTransformerTrainer` provides GPU-accelerated transformer training via WebGPU (WGSL shaders), enabling training on AMD, Intel, and other non-NVIDIA GPUs.

- **Implementation**: `src/train/transformer_trainer/wgpu_trainer.rs`
- **Feature gate**: `--features gpu` required to build
- **Backend**: Uses trueno's `GpuDevice` and WGSL backward shaders for gradient computation
- **Tested hardware**: AMD Radeon Pro W5700X (FALSIFY-WGPU-002 PASS)

### Building

```bash
# Build with WGPU support
cargo build --features gpu

# Run tests
cargo test --features gpu wgpu
```

### Local trueno Patch (Required)

WGPU training depends on trueno's WGSL backward shaders which may not yet be published to crates.io. Use a path patch:

```toml
[patch.crates-io]
trueno = { path = "../trueno" }
```

Remove this patch once the required trueno version is published.

### Current Status

- **Phase 2 COMPLETE**: Forward pass, loss computation, backward pass (gradient computation via WGSL shaders)
- **Phase 3 PENDING**: Full LoRA training loop integration (optimizer step, checkpoint save/restore)

### Provable Contracts

WGPU training correctness is verified via FALSIFY tests:

- `wgpu-training-v1.yaml` — 4 FALSIFY tests for AMD WGPU training (FALSIFY-WGPU-001..004)
- FALSIFY-WGPU-002 (forward-backward parity): PASS on AMD W5700X

### Relationship to cuBLAS Path

| Aspect | cuBLAS | WGPU |
|--------|--------|------|
| Vendor | NVIDIA only | AMD, Intel, NVIDIA (cross-platform) |
| Backend | CUDA PTX kernels | WGSL shaders via WebGPU |
| Maturity | 8 bugs found/fixed, Blackwell JIT blocked | Phase 2 complete |
| Performance | 298 tok/s (forward) | TBD (Phase 3) |
