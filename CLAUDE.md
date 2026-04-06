# Claude Code Development Guide for Realizar

## Project Overview

**Realizar** - Pure Rust ML inference engine built from scratch for GGUF and Safetensors model serving.

- **Philosophy:** Total control, zero compromise - build everything ourselves except HTTP infrastructure
- **Architecture:** Model parsers → Inference engine → Trueno compute primitives
- **Methodology:** EXTREME TDD with mutation testing, property-based testing, 85%+ coverage
- **Quality Target:** TDG Score ≥95.0/100 (A+)

## CRITICAL: Contract-First Design

**NEVER write code before writing a provable contract.**

All code changes MUST have a corresponding contract (YAML in ../provable-contracts/contracts/<project>/ or .pmat-work/<TICKET>/contract.json) BEFORE implementation. This is enforced by `pmat comply` CB-1400.

- Use `pmat comply check` to verify contract coverage
- Minimum verification level: L1 (recommended L3+)
- See docs/agent-instructions/provable-contract-first-agents.md for the full workflow

## Critical Dependencies - ALWAYS USE LATEST

### Trueno (SIMD/GPU Compute Primitives)

**IMPORTANT:** Trueno is actively developed and frequently updated. **ALWAYS check for the latest version.**

```bash
# Check trueno version before any development work
cd ../trueno && git pull && grep "^version" Cargo.toml
```

**Current Integration:**
- Path: `../trueno`
- Features: `["gpu"]` for GPU acceleration
- Status: v0.4.2 (2025-11-21) - SIMD attribute compliance, PMAT integration, zero warnings

**Update Workflow:**
1. Pull latest trueno: `cd ../trueno && git pull`
2. Check version: `grep "^version" Cargo.toml`
3. Update realizar's Cargo.toml with new version
4. Test integration: `cargo test --lib`
5. Commit with clear message about trueno version bump

**Trueno Capabilities:**
- Vector operations: add, sub, mul, div, dot, sum, norm_l1, norm_l2
- SIMD backends: AVX2, SSE2, NEON, WASM, Scalar
- GPU backend: wgpu-based (optional feature)
- Activation functions: ReLU, sigmoid, GELU, swish, mish, selu, hardswish
- Performance: 2-11x SIMD speedups on compute-bound operations

**Trueno GPU Kernels (trueno-gpu crate):**
- `GemmKernel` - Matrix multiplication (naive, tiled, tensor core)
- `AttentionKernel` - FlashAttention-style tiled attention with online softmax
- `SoftmaxKernel` - Numerically stable softmax with warp shuffle
- `LayerNormKernel` - Fused layer normalization
- `QuantizeKernel` - Q4_K dequantization fused with matmul
- `Q5KKernel` - Q5_K dequantization
- `Q6KKernel` - Q6_K dequantization

## ⚠️ CRITICAL ANTI-PATTERN: NO HAND-ROLLED PTX

**NEVER write PTX strings directly in realizar code.**

### Why This Is Forbidden

1. **Trueno exists** - The `trueno-gpu` crate has tested, optimized kernels
2. **PTX is fragile** - Syntax errors, wrong compute capabilities, shared memory limits
3. **Trueno has trueno-explain** - Static analysis tool to find PTX bugs
4. **Maintenance burden** - Hand-rolled PTX must be updated for each GPU generation
5. **Testing** - Trueno kernels have property tests; hand-rolled PTX does not

### The Anti-Pattern (DO NOT DO THIS)

```rust
// ❌ WRONG - Hand-rolled PTX string in realizar
fn generate_attention_ptx(seq_len: u32, head_dim: u32) -> String {
    format!(r"
.version 8.0
.target sm_89
.address_size 64
.visible .entry attention(...) {{
    // 200 lines of hand-written PTX
}}
")
}
```

### The Correct Pattern (DO THIS)

```rust
// ✅ CORRECT - Use trueno-gpu kernels
use trueno_gpu::kernels::{AttentionKernel, Kernel};

let kernel = AttentionKernel::new(seq_len, head_dim)
    .with_causal()
    .with_tiles(64, 64);
let ptx = kernel.emit_ptx();
```

### If Trueno Is Missing a Kernel

1. **Add it to trueno-gpu** - Push to `../trueno`, not realizar
2. **Use the PTX builder API** - `PtxKernel::new().param().build(|ctx| {...})`
3. **Add property tests** - Ensure kernel works for all valid dimensions
4. **Use trueno-explain** - Run `trueno-explain bugs --kernel <name>` to find issues

## ⚠️ CRITICAL: LAYOUT-002 Row-Major Mandate

**Realizar is EXCLUSIVELY row-major. All data from GGUF is transposed by aprender at import.**

### Why This Matters

GGUF uses column-major layout (GGML convention). Realizar's fused Q4K/Q6K kernels expect row-major layout. Using the wrong layout produces **garbage output**.

```
GGUF (column-major)     Realizar (row-major)
─────────────────────   ─────────────────────
W[i,j] at j*rows + i    W[i,j] at i*cols + j

Same bytes → WRONG interpretation → "olumbia+lsi nunca/localENTS" (garbage)
```

### The Architecture

```
┌─────────────────────────────────────────────────────────┐
│              REALIZAR DOMAIN (Row-Major Only)            │
│                                                          │
│  APR file ──► GGUF loader ──► fused_q4k_dot ──► output  │
│  (already row-major,         (expects row-major)         │
│   transposed by aprender)                                │
└─────────────────────────────────────────────────────────┘
```

**Realizar never handles layout conversion.** Aprender's converter (`src/format/converter/write.rs`) transposes GGUF data during import. By the time data reaches realizar, it's already row-major.

### FORBIDDEN: Trueno Column-Major Kernels

```rust
// ❌ NEVER USE - These expect column-major layout
use trueno::backends::q4k::matmul_q4k_f32_colmajor;
use trueno::backends::q6k::matmul_q6k_f32_colmajor;

// ✅ ALWAYS USE - Row-major kernels in realizar
use crate::quantize::fused_q4k_parallel_matvec;
use crate::quantize::fused_q6k_parallel_matvec;
```

### Key Implementation Files

| File | Responsibility |
|------|----------------|
| `src/quantize/fused_k.rs` | Row-major Q4K/Q6K matmul kernels |
| `src/quantize/parallel_k.rs` | Parallel row-major kernels (ONE WAY ONLY) |
| `src/gguf/loader.rs` | Loads APR (pre-transposed by aprender) |

### DELETED: Legacy Aliases (2026-02-03)

These confusing aliases were **purged** to enforce ONE WAY ONLY:
- ~~`fused_q6k_colmajor_matvec`~~ → Use `fused_q6k_parallel_matvec`
- ~~`fused_q4k_auto_matvec_into`~~ → Use `fused_q4k_parallel_matvec_into`

**If you see these function names in old code, they no longer exist.**

### Falsification Test (F-LAYOUT-001)

```bash
# Test that GGUF→APR→realizar produces coherent output
apr import model.gguf -o model.apr
realizar run model.apr --prompt "2+2=" --max-tokens 10
# Expected: "4" (coherent math)
# NOT: "olumbia+lsi" (garbage = layout bug)
```

## ⚠️ CRITICAL: PMAT-216 GPU Parity Mandate

**GPU inference MUST match CPU inference. This is enforced by CI.**

### Root Cause (Five Whys)

| Why | Answer |
|-----|--------|
| 1. Why garbage GPU output? | LM head produces wrong values |
| 2. Why wrong LM head? | Weight matrix not properly transposed |
| 3. Why not transposed? | `lm_head_weight_t` contained original data |
| 4. Why? | Argument order in `from_apr_weights` swapped |
| 5. Why? | No type safety on weight parameters |

### Fix Applied (2026-02-05)

1. **Type-safe wrappers** in `types.rs`:
   - `LmHeadWeight` - Original layout [vocab_size, hidden_dim]
   - `LmHeadWeightTransposed` - GPU layout [hidden_dim, vocab_size]

2. **Runtime validation** in `from_apr_weights`:
   - Checks first row of original == first column of transposed
   - Fails with `PMAT-216: Arguments may be swapped` on mismatch

3. **Mandatory parity test** (`tests/gpu_cpu_trace_compare.rs`):
   ```bash
   cargo test --features cuda --test gpu_cpu_trace_compare
   # Expected: CPU L2 ≈ GPU L2 (diff < 0.01%)
   ```

### Why Tracing Didn't Catch This

| Gap | Impact |
|-----|--------|
| `GpuModel` has no `forward_traced` | Can't trace GPU layer-by-layer |
| No `TracedForward` trait | CPU/GPU can diverge silently |
| No parity test in CI | GPU bugs ship undetected |

### Mandatory GPU Verification

```rust
// ALWAYS compare CPU vs GPU for new models:
let cpu_trace = apr_model.forward_traced(&tokens)?;
let gpu_logits = gpu_model.forward_gpu(&tokens)?;
let cpu_l2 = cpu_trace.logits.iter().map(|x| x * x).sum::<f32>().sqrt();
let gpu_l2 = gpu_logits.iter().map(|x| x * x).sum::<f32>().sqrt();
assert!((cpu_l2 - gpu_l2).abs() / cpu_l2 < 0.01, "GPU diverged from CPU!");
```

### Aprender (ML Library)

**IMPORTANT:** Aprender is actively developed and frequently released. **ALWAYS check for the latest version.**

```bash
# Check aprender version and status
cd ../aprender && git pull && grep "^version" Cargo.toml
```

**Current Status:**
- Version: v0.1.0 (released to crates.io 2024-11-18)
- TDG Score: 95.6/100 (A+)
- Test Coverage: 97.72%
- Path: `../aprender`

**Aprender Primitives (Fallback Option):**
- `Vector<T>` - Generic 1D array with sum, mean, dot, norm, variance
- `Matrix<T>` - Row-major 2D array with matmul, transpose, Cholesky
- **Pure Rust:** Forbids unsafe code entirely
- **Battle-tested:** 149 tests (127 unit + 22 property)

**When to Use Aprender:**
- If trueno has compilation issues (rare)
- For pure Rust fallback without SIMD/GPU
- Can swap implementations transparently

**Update Workflow:**
1. Pull latest aprender: `cd ../aprender && git pull`
2. Check if relevant for inference primitives
3. Consider integration if trueno unavailable
4. Document in commit message

## Python Usage Policy

**IMPORTANT: Avoid Python unless absolutely necessary. This is a pure Rust project.**

### When Python IS Acceptable
- Generating reference values from HuggingFace transformers for verification
- Quick one-off debugging comparisons (not permanent scripts)
- No Rust equivalent exists for the task

### When Python is NOT Acceptable
- Production code (use Rust)
- Build scripts (use Rust/Makefile)
- Tests (use Rust tests)
- Benchmarks (use Criterion)

### If Python Is Required, Use `uv`

**NEVER use pip, virtualenv, conda, or poetry. ONLY use `uv`.**

```bash
# Run Python script with dependencies
uv run --with torch --with transformers python script.py

# Or use inline script dependencies (PEP 723)
uv run script.py  # If script has # /// script metadata

# Interactive REPL with deps
uv run --with torch python
```

**Why uv:**
- Fast dependency resolution (10-100x faster than pip)
- Deterministic environments
- No need to manage venvs manually
- Works with pyproject.toml or inline deps

## Ground Truth Verification

**CRITICAL: Always verify inference outputs against multiple reference implementations.**

All reference implementations live in `~/src/`:

### Reference Implementations (Priority Order)

1. **llama.cpp** (`~/src/llama.cpp`) - Primary reference for GGUF inference
   ```bash
   cd ~/src/llama.cpp
   ./llama-cli -m /path/to/model.gguf -p "prompt" -n 1 --verbose
   # Or for embeddings/hidden states:
   ./llama-embedding -m /path/to/model.gguf -p "prompt"
   ```

2. **Ollama** (`~/src/ollama`) - Production GGUF serving reference
   ```bash
   ollama run tinyllama "prompt" --verbose
   # Check logs for token probabilities
   ```

3. **HuggingFace Transformers** - FP32 ground truth (via uv)
   ```bash
   uv run --with torch --with transformers python3 << 'EOF'
   from transformers import AutoModelForCausalLM, AutoTokenizer
   model = AutoModelForCausalLM.from_pretrained("model-name")
   # Get logits, hidden states, etc.
   EOF
   ```

4. **Candle** (`~/src/candle`) - Rust reference implementation
   ```bash
   cd ~/src/candle
   cargo run --release --example llama -- --model /path/to/model --prompt "test"
   ```

### Verification Checklist

When debugging inference issues, verify in order:

1. **Embedding lookup** - Token → embedding vector
   - Compare L2 norm and first 10 elements with HF
   - Note: GGUF may use Q4_K quantized embeddings

2. **RMSNorm** - Layer normalization
   - Compare L2 norm before/after norm
   - Verify weight values match

3. **Attention projections** (Q/K/V) - Per-layer
   - Compare Q output L2 with HF for same input
   - Check per-head L2 norms

4. **FFN projections** (gate/up/down) - Per-layer
   - Check FFN hidden (gate * up) L2
   - Verify FFN output doesn't cause catastrophic cancellation

5. **Layer-by-layer hidden state L2** - Track through all layers
   - Should closely match HF layer-by-layer
   - Watch for divergence accumulation

6. **Final logits** - Top-k comparison
   - Compare L2 norm (should be within 10%)
   - Verify top-5 tokens match HF top-5
   - Check cosine similarity > 0.99

### Quantization Tolerance

Expected differences due to quantization:
- **Q4_K**: ±5% element-wise, <1% L2 norm
- **Q6_K**: ±2% element-wise, <0.5% L2 norm
- **FP16**: ±0.1% element-wise

### Creating Verification Scripts

Store verification scripts in `examples/par_*` (parity tests):
```
examples/
  par_001_*.rs     # Token embedding verification
  par_002_*.rs     # Layer-by-layer hidden states
  par_003_*.rs     # Logit comparison
  debug_*.rs       # One-off debugging scripts
```

## Development Workflow

### Before Starting Any Work

```bash
# 1. Check ecosystem versions
cd ../trueno && git pull && grep "^version" Cargo.toml
cd ../aprender && git pull && grep "^version" Cargo.toml
cd realizar

# 2. Update dependencies if needed
# Edit Cargo.toml with new versions

# 3. Verify clean build
cargo clean
cargo test --lib

# 4. Check quality baselines
pmat analyze tdg
pmat analyze satd
pmat analyze complexity
```

## Code Search (pmat query)

**NEVER use grep or rg for code discovery.** Use `pmat query` instead -- it returns quality-annotated, ranked results with TDG scores and fault annotations.

```bash
# Find functions by intent
pmat query "inference forward pass" --limit 10

# Find high-quality code
pmat query "attention mechanism" --min-grade A --exclude-tests

# Find with fault annotations (unwrap, panic, unsafe, etc.)
pmat query "tokenizer decode" --faults

# Filter by complexity
pmat query "gguf loading" --max-complexity 10

# Cross-project search (e.g., find trueno SIMD kernels)
pmat query "simd matmul" --include-project ../trueno

# Search across the stack
pmat query "quantization Q4_K" --include-project ../aprender
pmat query "model checkpoint" --include-project ../entrenar

# Git history search (find code by commit intent via RRF fusion)
pmat query "fix inference output" -G
pmat query "kernel optimization" --git-history

# Enrichment flags (combine freely)
pmat query "attention mechanism" --churn           # git volatility (commit count, churn score)
pmat query "gguf loading" --duplicates             # code clone detection (MinHash+LSH)
pmat query "tokenizer" --entropy                   # pattern diversity (repetitive vs unique)
pmat query "forward pass" --churn --duplicates --entropy --faults -G  # full audit
```

### Coverage-Guided Search (pmat 3.0.0+)

**Use `pmat query --coverage` to find untested code. NEVER parse coverage JSON manually.**

```bash
# Find top uncovered functions (no query needed)
pmat query --coverage-gaps

# Find uncovered functions matching a semantic query
pmat query "quantization" --coverage --uncovered-only

# Use pre-existing coverage data (avoids re-running cargo llvm-cov)
pmat query --coverage-gaps --coverage-file /path/to/coverage.json

# Coverage auto-detection: runs `cargo llvm-cov report --json` automatically
# Prerequisite: run `cargo llvm-cov test --lib --no-report` first to generate data
```

**Workflow for coverage improvement:**
1. `cargo llvm-cov test --lib --no-report` — generate coverage data
2. `pmat query --coverage-gaps` — find top uncovered functions
3. Write tests targeting those functions
4. `make coverage` — verify improvement

### EXTREME TDD Methodology

**Follow RED-GREEN-REFACTOR:**

1. **RED:** Write failing tests first
   - Comprehensive test coverage (edge cases, errors, valid inputs)
   - Property-based tests for mathematical correctness
   - Document expected behavior

2. **GREEN:** Minimal implementation to pass tests
   - Focus on correctness, not optimization
   - Use clear, readable code
   - Leverage trueno primitives where applicable

3. **REFACTOR:** Clean up and optimize
   - Fix clippy warnings (zero tolerance)
   - Apply rustfmt formatting
   - Extract helper functions
   - Document with examples

**Quality Gates (all must pass):**
```bash
make fmt-check     # Format check
make clippy        # Zero warnings
make test          # All tests pass
make test-fast     # < 5 minutes
make coverage      # <10 minutes, aim for 85%+
```

### Trueno Integration Patterns

**Prefer Trueno for Compute:**
```rust
// Good: Use trueno for vector operations
use trueno::Vector;

let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
let result = a.dot(&b); // SIMD-accelerated
```

**Matrix Operations:**
```rust
// Good: Use trueno for matrix multiplication
use trueno::Matrix;

let weights = Matrix::from_slice(128, 256, &data);
let input = Matrix::from_slice(1, 128, &input_data);
let output = weights.matmul(&input); // GPU-accelerated if available
```

**Activation Functions:**
```rust
// Good: Use trueno activations for inference
use trueno::Vector;

let logits = Vector::from_slice(&[0.1, -0.5, 0.3]);
let activated = logits.relu(); // SIMD-accelerated ReLU
```

## Phase 1 Roadmap Progress

### Week 1-2: Model Parsers ✅ COMPLETE
- ✅ GGUF parser (header + metadata + tensor_info)
- ✅ Safetensors parser (JSON metadata + zero-copy data)
- ✅ 26 tests passing
- ✅ TDG Score: 96.2/100 (A+)
- ✅ Zero SATD violations

### Week 3-4: Transformer Components ✅ COMPLETE
- ✅ Layer normalization (7 tests, epsilon-based normalization)
- ✅ Linear layer (6 tests, weight/bias loading)
- ✅ GELU activation (5 tests, tanh approximation)
- ✅ Feed-forward networks (FFN) (6 tests, 2-layer with GELU)
- ✅ Softmax activation (6 tests, numerically stable)
- ✅ Attention mechanism (8 tests, scaled dot-product attention)
- ✅ RoPE position embeddings (11 tests, rotary position encoding)
- ✅ KV cache management (10 tests, efficient inference caching)

### Week 5-6: Quantization ✅ COMPLETE
- ✅ Q4_0 dequantization (4-bit, block size 32)
- ✅ Q8_0 dequantization (8-bit, block size 32)
- ✅ Dequantization for inference
- ✅ EXTREME TDD (5 comprehensive tests)
- [ ] Mixed precision support (deferred)

### Week 7-8: Tokenizer & Inference ✅ COMPLETE
- ✅ Basic tokenizer (10 tests, encode/decode)
- ✅ Embedding layer (6 tests, token to vector)
- ✅ Complete Model struct (5 tests, end-to-end inference)
- ✅ Generation loop (6 tests, token sampling)
- ✅ Sampling strategies (16 tests, greedy/top-k/top-p)
- ✅ BPE tokenizer (14 tests, byte pair encoding)
- ✅ SentencePiece tokenizer (14 tests, unigram model)
- ✅ HTTP API with axum (8 tests, REST endpoints)

## Quality Standards

**Mandatory Requirements:**
- **TDG Score:** ≥95.0/100 (A+ grade)
- **Test Coverage:** ≥85%
- **Mutation Score:** ≥80%
- **Cyclomatic Complexity:** ≤10 per function
- **Clippy Warnings:** 0 (zero tolerance)
- **SATD Comments:** 0 (implement or remove TODOs)

**Testing Requirements:**
- Unit tests for all public APIs
- Property-based tests for mathematical operations
- Integration tests for end-to-end workflows
- Benchmark tests for performance-critical paths

## Git Workflow

**Branch Policy:** Work directly on `main` branch (per CLAUDE.md in ~/.claude/)

**Commit Message Format:**
```
<type>: <subject>

<body>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `perf`: Performance improvement
- `refactor`: Code restructuring
- `test`: Add/update tests
- `docs`: Documentation
- `chore`: Maintenance (deps, config)

## Monitoring Ecosystem Updates

**Daily Checks (if actively developing):**
```bash
# Quick version check
cd ../trueno && git log --oneline -1 && grep "^version" Cargo.toml
cd ../aprender && git log --oneline -1 && grep "^version" Cargo.toml
```

**When to Update Realizar:**
- New trueno version with relevant features (vector ops, activations)
- Bug fixes in trueno that affect realizar
- Performance improvements in trueno SIMD/GPU backends
- New aprender primitives useful for inference

**Testing After Updates:**
1. `cargo clean` - Clear build artifacts
2. `cargo test --lib` - Verify all tests pass
3. `cargo clippy --lib -- -D warnings` - Zero warnings
4. `make quality-gates` - Full quality suite
5. Commit with version bump and rationale

## Architecture Principles

**1. Pure Rust from Scratch:**
- Build all ML components ourselves (parsers, transformer, quantization, tokenizer)
- Use trueno for compute primitives only
- HTTP server is swappable (axum default)

**2. Zero Unsafe in Public API:**
- All unsafe code isolated in trueno/aprender
- Realizar public API is 100% safe Rust

**3. Backend Agnostic:**
- Trueno handles SIMD/GPU dispatch automatically
- Fallback to scalar for unknown architectures
- WASM support via trueno

**4. Swappable HTTP Server:**
```rust
pub trait HttpServer {
    fn serve(&self, addr: &str) -> Result<()>;
}

// Currently: axum
// Future: hyper, actix-web, custom
```

## Performance Targets

**Inference Latency (1B models):**
- p50: <100ms
- p95: <200ms
- p99: <500ms

**Memory Usage:**
- Model: As loaded (no unnecessary copies)
- Runtime: <512MB overhead
- KV cache: Bounded and configurable

**Throughput:**
- Single request: Minimize latency
- Batch inference: Maximize throughput (Phase 2)

## Resources

**Documentation:**
- GGUF Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Safetensors Spec: https://github.com/huggingface/safetensors
- Trueno README: ../trueno/README.md
- Aprender README: ../aprender/README.md

**Related Projects:**
- [Trueno](https://github.com/paiml/trueno) - SIMD/GPU compute primitives
- [Aprender](https://github.com/paiml/aprender) - ML library in pure Rust
- [Renacer](https://github.com/paiml/renacer) - Profiling tools
- [paiml-mcp-agent-toolkit](https://github.com/paiml/paiml-mcp-agent-toolkit) - Quality gates
- [bashrs](https://github.com/paiml/bashrs) - Script enforcement

**Quality Tools:**
- `pmat`: Multi-dimensional analysis (TDG, complexity, SATD, coverage)
- `cargo-mutants`: Mutation testing
- `cargo-llvm-cov`: Code coverage
- `proptest`: Property-based testing

---

## Hardware Environment

**GPU:** NVIDIA GeForce RTX 4090
- CUDA Compute Capability: 8.9 (Ada Lovelace)
- VRAM: 24GB GDDR6X
- Tensor Cores: 4th Gen (FP16/BF16/INT8)
- CUDA Cores: 16384
- Memory Bandwidth: 1008 GB/s

**⚠️ CRITICAL: CUDA IS ALWAYS AVAILABLE**

99% of development happens on RTX 4090. CUDA hardware is ALWAYS present.

- **NEVER** say "CUDA not available" or "skipping because no GPU"
- **NEVER** use `CudaExecutor::is_available()` as a pre-check - it can return false even when GPU exists
- **ALWAYS** try `CudaExecutor::new(0)` directly and inspect the actual error if it fails
- If CUDA tests fail, **investigate the real error** - don't assume hardware is missing

```rust
// ❌ WRONG - Don't pre-check availability
if !CudaExecutor::is_available() {
    println!("Skipping: CUDA not available");  // WRONG ASSUMPTION
    return;
}

// ✅ CORRECT - Try to create and show actual error
match CudaExecutor::new(0) {
    Ok(exec) => { /* use it */ },
    Err(e) => {
        eprintln!("CUDA init failed: {:?}", e);  // Shows REAL problem
        // Investigate: library path? permissions? driver version?
    }
}
```

**⚠️ MANDATORY GPU TESTING:**
```bash
# ALWAYS run GPU tests - RTX 4090 is available
cargo test --lib --features cuda

# For integration tests with multiple CudaExecutor instances, use single thread
# to avoid CUDA_ERROR_NOT_INITIALIZED race condition:
cargo test --test cuda_combinatorial_coverage --features cuda -- --test-threads=1

# DO NOT use #[ignore] for GPU tests
# ALL GPU tests must execute, not be skipped
```

**Benchmark Targets (RTX 4090):**
- Ollama phi2:2.7b: ~225-266 tok/s (baseline)
- llama.cpp CUDA: ~256 tok/s
- Target: <1.25x gap to Ollama

**Development Iteration ("implement using pmat work"):**
1. `pmat analyze satd` - check SATD
2. `cargo clippy --lib --features cuda` - zero warnings
3. `cargo test --lib --features cuda` - **ALL tests including GPU**
4. Update spec with results

---

## CRITICAL: TUI Simulation Debugging (Probar-Style)

**⚠️ MANDATORY FOR ALL GPU/CUDA DEBUGGING**

When debugging GPU scheduler issues (CUDA vs wgpu parity, buffer management, kernel execution),
you MUST use TUI simulation workflow tests. This pattern was proven critical in PARITY-114 where
it detected a **state accumulation bug** that simple unit tests missed.

### Why TUI Simulation is Required

1. **Watches the Flow**: Step-by-step visualization of data through schedulers
2. **Catches State Bugs**: Sequential operations reveal accumulation/leakage issues
3. **Provides Diagnosis**: Automatic analysis of failure ratios (8x = accumulator bug, 4x = tile bug)
4. **Probar Alignment**: Matches probar's proven TUI testing methodology

### TUI Simulation Test Pattern

```rust
/// Example: TUI simulation for scheduler parity testing
#[test]
#[cfg(feature = "cuda")]
fn test_scheduler_parity_tui_simulation() {
    use realizar::gpu::{CudaScheduler, HybridScheduler};

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  TUI SIMULATION: Watch Data Flow Through Schedulers                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    let mut sim = MatmulSimulator::new();

    // Define steps
    let step_init = sim.add_step("INIT", "Initialize test matrices");
    let step_cpu = sim.add_step("CPU", "Compute reference");
    let step_cuda = sim.add_step("CUDA", "Execute via CudaScheduler");
    let step_check = sim.add_step("CHECK", "Verify parity");

    // Execute with visual feedback
    sim.start_step(step_init);
    println!("  ◐ Initializing...");
    // ... setup code ...
    sim.complete_step(step_init, values, None);
    println!("  ● Complete");

    // Render final TUI frame
    println!("{}", sim.render_final());
}
```

### State Isolation Test Pattern

**CRITICAL**: Always test sequential operations to catch state bugs:

```rust
/// Test for state accumulation bugs
#[test]
fn test_scheduler_state_isolation() {
    let mut scheduler = CudaScheduler::new().unwrap();

    // Same operation twice - results MUST be identical
    let r1 = scheduler.matmul(&a, &b, m, k, n).unwrap();
    let r2 = scheduler.matmul(&a, &b, m, k, n).unwrap();

    assert_eq!(r1[0], r2[0], "State leak detected: first={}, second={}", r1[0], r2[0]);
}
```

### Running TUI Workflow Tests

```bash
# Run all GPU parity workflow tests with visual output
cargo test --test gpu_parity_workflow --features cuda -- --nocapture

# Specific TUI simulation test
cargo test --test gpu_parity_workflow test_parity_114_tui_simulation --features cuda -- --nocapture
```

### Failure Analysis Guide

| Ratio | Diagnosis | Check |
|-------|-----------|-------|
| 8x | Accumulator/tile loop bug | Inner loop iterations, FMA instruction |
| 4x | Partial tile accumulation | n_tiles calculation, tile bounds |
| 2x | Half iterations | Loop termination condition |
| Varies | State accumulation | Output buffer not cleared between calls |

### Bug Discovery: PARITY-114 Case Study

The TUI simulation discovered that **the same operation produced different results**:

```
Op 1: 4×64×8, expected 64, got 8
Op 3: 4×64×8, expected 64, got 16  ← DIFFERENT from Op 1!
```

This proved the output buffer was accumulating between calls rather than being cleared.
Simple unit tests would NOT have caught this - only sequential TUI simulation revealed it.

---

**Last Updated:** 2026-01-21
**Realizar Version:** 0.8.0
**GPU Spec Version:** v5.2.0 (CUDA Monolith Shattered + Lint Zero)
**Trueno Version:** 0.16.0
**Aprender Version:** 0.27.0
**Entrenar Version:** 0.7.2
**paiml-mcp-agent-toolkit Version:** v2.200.0 (with Known Defects Scorer, SATD Detector, Defect Analyzer)
**TDG Score:** 93.9/100 (A)
**Rust Project Score:** 137.9/134 (103%, Grade A+)
**Test Coverage:** 80.97% (region), 88.75% (function), 80.08% (lines)
**Total Tests:** 6324 (all passing), 32 ignored
**Mutation Score:** 100% on api.rs (18/18 viable mutants caught)
**Documentation:** 15.0/15 (100%) ✅ Perfect score!
**Known Defects:** 20.0/20 (100%) ✅ Perfect score!
**Dependency Health:** 10.5/12 (87.5%) - Modular feature flags
**Benchmarks:** 4 suites (tensor_ops, inference, cache, tokenizer)
**Examples:** 7 (inference, api_server, tokenization, safetensors_loading, model_cache, gguf_loading, convert_and_bench_apr)
**Performance:**
  - **APR Q4_0: 17.0-17.3 tok/s (1.36x faster than GGUF)** ✅ v0.3.4
  - GGUF Q4_0: 12.5-13.0 tok/s (Candle parity exceeded)
  - APR F32: 0.1 tok/s (memory bandwidth limited)
  - <1ms p50 for 5-token generation
  - **38-41% of llama.cpp** (target: 100%+)
**CLI Binary:** ✅ `realizar serve --demo` (65% coverage)
**Quality Improvements:**
  - Added workspace-level lints (unsafe_op_in_unsafe_fn, unreachable_pub, checked_conversions)
  - Created .clippy.toml for cognitive complexity thresholds
  - Fixed critical unwrap() in safetensors.rs (replaced with expect())
  - Updated to latest trueno v0.4.2 with SIMD attribute compliance and PMAT integration
  - Integrated paiml-mcp-agent-toolkit v2.200.0 (Known Defects, SATD, Defect Analysis)
**GPU Performance Parity (M29-M32):**
  - M29: Error Recovery (ErrorRecoveryStrategy, DegradationManager, FailureIsolator)
  - M30: Resource Management (ConnectionPool, ResourceLimiter, ResourceMonitor)
  - M31: Resilience (RetryPolicy, CircuitBreaker, BulkheadManager)
  - M32: Diagnostics (Logger, PhaseTimer, MemoryTracker, DiagnosticsCollector, DebugMode)
**APR Q4_0 Format (v0.3.5):**
  - `QuantizedAprTransformerQ4` - Pure Rust quantized inference
  - RoPE (Rotary Position Embeddings) with configurable theta
  - Grouped Query Attention (GQA) for TinyLlama compatibility
  - SIMD matmul via `fused_q4_0_q8_0_parallel_matvec`
  - **Parallel attention heads** via rayon (32 heads parallelized)
  - **Parallel FFN up/gate** via rayon::join
  - **KV Cache** for efficient autoregressive generation
    - `AprKVCache` stores K/V per layer, avoids recomputation
    - `forward_with_cache()` for context-aware generation
    - `causal_attention_cached()` with parallel head processing
  - **13-19 tok/s** context-aware generation (32-45% of llama.cpp)
**CUDA Refactor (v5.2.0):**
  - Shattered 23K-line cuda.rs monolith into 9 atomic modules
  - Split 21K-line executor.rs into domain submodules (activations, core, gemm, layer, quantized, workspace)
  - Split 15K-line impl_main.rs into 9 focused submodules
  - 65 files cleaned for zero clippy warnings
  - Fixed broken benchmarks (GGUFTransformer → AprTransformer)
**Latest Achievement:** CUDA monolith shattered + comprehensive lint cleanup (65 files, 2089 insertions, 1040 deletions)
**Completed:** Weeks 1-8 + GPU parity M1-M32 + APR Q4_0 (M2) + Rayon (M3) + KV Cache (M4) + CUDA Refactor (v5.2.0)


## Stack Documentation Search

**IMPORTANT: Proactively use the batuta RAG oracle when:**
- Looking up SIMD/GPU patterns from trueno
- Finding inference patterns from TGI ground truth corpus
- Understanding quantization approaches (GGUF, APR formats)
- Researching KV cache, attention, or batching implementations

```bash
# Search across the entire Sovereign AI Stack
batuta oracle --rag "your question here"

# Examples for realizar development
batuta oracle --rag "KV cache optimization patterns"
batuta oracle --rag "continuous batching TGI"
batuta oracle --rag "CUDA kernel matmul implementation"
batuta oracle --rag "quantization Q4_K dequantization"
batuta oracle --rag "FlashAttention tiled attention"

# Reindex if needed (persists to ~/.cache/batuta/rag/)
batuta oracle --rag-index
```

The RAG index includes 335 documents across:
- All Sovereign AI Stack repos (trueno, aprender, entrenar, etc.)
- Python ground truth corpora (HuggingFace, JAX, vLLM patterns)
- Rust ground truth corpora (TGI inference patterns, MLOps)

Index auto-updates via post-commit hooks and `ora-fresh` on shell login.
To manually check freshness: `ora-fresh`
To force full reindex: `batuta oracle --rag-index --force`

## SSC Training / Blackwell: Inference NOT Affected (2026-03-22)

- **Inference is NOT affected** by the Blackwell training JIT bug (trueno#200)
- **realizar uses cuBLAS (GPU) or trueno SIMD (CPU)** for all GEMMs — pre-compiled kernels, no JIT
- **NF4 fused kernel and cuBLAS backward kernels** are training-only (entrenar) — realizar never calls them
- **When the SSC model ships**: realizar loads the LoRA adapter via standard PEFT/safetensors path — no special Blackwell handling needed
- **Trained model (LoRA adapter)**: Architecture-independent — works on any GPU or CPU
- **Key tickets**: trueno#200 (Blackwell JIT), trueno#203 (pre-compiled kernels), entrenar#300 (cuBLAS backward)
