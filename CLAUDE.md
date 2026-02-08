# CLAUDE.md

## Project Overview

Aprender is a next-generation machine learning library written in pure Rust. v0.25.1 implements the TOP 10 ML algorithms plus advanced modules (time series, NLP, Bayesian, GLM, graph, audio, format conversion). 11,251 tests with comprehensive quality gates.

## Build Commands

```bash
cargo build --release        # Optimized build
cargo test                   # Full test suite (11251 tests)
cargo test --lib             # Unit tests only
cargo fmt --check            # Check formatting
cargo clippy -- -D warnings  # Strict linting
cargo bench                  # Run criterion benchmarks

# Makefile tiered quality gates
make tier1                   # Fast feedback (<1s): fmt, clippy, check
make tier2                   # Pre-commit (<5s): tests + strict clippy
make tier3                   # Pre-push (1-5min): full validation + coverage
make tier4                   # CI/CD: includes pmat analysis
make coverage                # Coverage report (96.35% achieved, target ≥95%)
```

## Debugging: Use apr Tools First (MANDATORY)

**STOP. Before reading code or grepping, USE THE APR DIAGNOSTIC TOOLCHAIN.**

GH-202 lesson: we read code instead of running `apr qa` which would have instantly shown the failure.

```bash
# Step 1: ALWAYS start here (catches 80% of issues)
apr qa model.apr

# Step 2: Check tensor shapes/stats
apr tensors model.apr | head -20

# Step 3: Diff against known-good model
apr diff model.apr reference.gguf

# Step 4: Format/metadata integrity
apr validate model.apr --quality
apr lint model.apr

# Step 5: ONLY NOW read code
```

| Tool | Purpose |
|------|---------|
| `apr qa` | Falsifiable QA gates (first tool for ANY issue) |
| `apr tensors` | Tensor inspection (shapes/stats) |
| `apr validate` | Integrity check |
| `apr lint` | Best practices |
| `apr diff` | Model comparison (tensor-by-tensor) |
| `apr trace` | Layer-by-layer analysis |
| `apr profile` | Roofline analysis (memory vs compute bound) |
| `apr inspect` | Metadata inspection |
| `apr debug` | Quick debug output ("drama" mode for verbose) |

All tools support GGUF, APR, and SafeTensors formats. If a tool says "format not supported", that's a BUG.

### Realizar Inference Tracing

Located in `realizar/src/inference_trace.rs`:
```bash
realizar run model.safetensors --prompt "2+2?" --trace
realizar run model.gguf --prompt "Hi" --trace=tokenize,sample,decode
```

TraceSteps: `Tokenize`, `Embed`, `LayerNorm`, `Attention`, `FFN`, `TransformerBlock`, `LmHead`, `Sample`, `Decode`

## Architecture

1. **Trait-Based Multiple Dispatch** - Julia-inspired pattern
2. **Backend Agnostic** - CPU (SIMD), GPU, WASM via Trueno
3. **Three-Tier API**: High (`Estimator` trait), Mid (`Optimizer`/`Loss`/`Regularizer`), Low (Trueno primitives)

**Runtime:** `trueno = "0.4.0"` (SIMD-accelerated tensor ops)
**Dev Tools:** `proptest`, `criterion`, `pmat` v2.216.0, `renacer`, `cargo-mutants`
**Banned:** serde, rayon, tokio, thiserror, ndarray, polars, arrow (see spec)

## CRITICAL: Realizar-First Architecture

**ALL inference/serving MUST use `realizar`. The `aprender` crate is for TRAINING ONLY.**

| Responsibility | aprender | realizar | trueno |
|---------------|----------|----------|--------|
| Model Training / Autograd | Primary | Never | Compute |
| .apr Format R/W | Primary | Read-only | - |
| Model Serving / HTTP / KV Cache | **FORBIDDEN** | Primary | Compute/Storage |
| GGUF/SafeTensors Loading | Never | Primary | - |
| CUDA/GPU Inference | Never | Primary | Kernels |

```rust
// WRONG - bypasses realizar, 0.3 tok/s
use aprender::models::Qwen2Model;
let output = model.generate(&input_ids, 32, 0.7, 0.9);

// CORRECT - uses realizar, 225+ tok/s
use realizar::Model;
let model = Model::load_safetensors(&path)?;
let output = model.generate(&input_ids, config)?;
```

```bash
# BEST - apr CLI uses realizar automatically
cargo run --bin apr --features inference -- run model.safetensors \
    --prompt "What is 2+2?" --max-tokens 32
```

Feature flag: `inference = ["realizar", "tokio", "axum"]` (default-enabled in apr-cli).
Always profile with `apr profile`/`apr trace`/`apr bench` before optimizing.

### Performance Targets (Ollama Parity)

| Model | CPU (tok/s) | GPU (tok/s) | Memory |
|-------|-------------|-------------|--------|
| 1B Q4_K | 100+ | 500+ | 600MB |
| 7B Q4_K | 30+ | 150+ | 4GB |
| 13B Q4_K | 15+ | 80+ | 8GB |

Architecture: Trueno SIMD backend, realizar fused dequant+matmul kernels, PagedAttention KV cache, optional wgpu/CUDA.

## LAYOUT-001/002: Tensor Layout Safety

**CRITICAL: GGUF/APR use ROW-MAJOR layout. This bug has occurred 100+ times.**

APR and realizar are EXCLUSIVELY row-major. GGUF column-major data is transposed at import boundary.

```
GGUF (col-major) ──(TRANSPOSE at import)──► APR (row-major) ──► realizar ──► output
SafeTensors (native) ──────────────────────► APR (row-major) ──► realizar ──► output
```

**FORBIDDEN IMPORTS (produce garbage):**
```rust
// NEVER for GGUF/APR data:
use trueno::backends::q4k::matmul_q4k_f32_colmajor;
use trueno::backends::q6k::matmul_q6k_f32_colmajor;
// (and their _dispatch variants)
```

**REQUIRED IMPORTS (row-major):**
```rust
use crate::quantize::fused_q4k_parallel_matvec;
use crate::quantize::fused_q6k_parallel_matvec;
```

**Key Files:**
- `contracts/tensor-layout-v1.yaml` - **SOURCE OF TRUTH**
- `src/format/layout_contract.rs` - Rust validation API
- `src/format/converter/write.rs` - GGUF→APR import with transpose
- `src/format/converter/mod.rs` - `transpose_q4k_for_matmul()`, `transpose_q6k_for_matmul()`

```rust
use aprender::format::layout_contract::{CONTRACT, LayoutContract};
CONTRACT.should_transpose_gguf("output.weight");  // true for 2D, false for 1D
CONTRACT.validate_apr_shape("lm_head.weight", &[vocab, hidden], vocab, hidden)?;
```

### Code Scheduled for Deletion

- `src/models/qwen2/mod.rs::generate()` / `forward()` - DELETE
- `examples/qwen_inference.rs` - REWRITE to use apr CLI

## Shell Scripts: Use bashrs (NOT shellcheck)

```bash
bashrs lint scripts/*.sh          # Lint
bashrs purify scripts/ci.sh       # Determinism + idempotency
bashrs make lint Makefile          # Makefile linting
bashrs gate --strict .             # Full quality gate
```

Required: `set -euo pipefail`, no `ls` for iteration, quoted variables, explicit error handling.

## Testing

Target: 60% unit, 30% property, 10% integration. Coverage: 96.35% line (target ≥95%).

```bash
cargo test                    # All 11251 tests
cargo test --lib              # Unit only
cargo test --test integration # Integration
cargo test --doc              # Doctests
make coverage                 # Coverage report (disables mold linker, two-phase llvm-cov)
```

Mutation testing: `cargo mutants --no-times --timeout 300 --in-place -- --all-features` (or via CI).

## Linting

Workspace-level lints in `Cargo.toml` (`[workspace.lints.rust]` / `[workspace.lints.clippy]`).
Key: `unsafe_code = "forbid"`, `clippy::all + pedantic = "warn"`, ML-specific allows for casts/float_cmp.

## CI/CD (`.github/workflows/`)

- **ci.yml**: check, fmt, clippy, test, coverage (Codecov), mutation testing, security audit, docs, bashrs
- **benchmark.yml**: criterion benchmarks on PR/weekly, auto PR comments
- **security.yml**: cargo-audit, cargo-deny (license/banned crates), cargo-outdated (weekly)
- **dependabot.yml**: weekly Rust deps, monthly GH Actions
- **book.yml**: EXTREME TDD book to GitHub Pages
- **release.yml**: automated releases on version tags

## Modules

**v0.4.0 (TOP 10 ML):** LinearRegression, LogisticRegression, DecisionTree, RandomForest, GBM, NaiveBayes, KNN, SVM, KMeans, PCA + model selection + metrics

**v0.7.x (Advanced):** ARIMA time series, text processing (tokenizers, stop words, stemming, chat templates via minijinja), Bayesian inference (conjugate priors, BLR), GLMs (Poisson/Gamma/Binomial), ICA decomposition, graph algorithms (Dijkstra/A*/PageRank/community detection)

## Key Files

- `src/lib.rs` - Library entry, module exports
- `src/traits.rs` - Core traits (Estimator, UnsupervisedEstimator, Transformer)
- `src/primitives/` - Vector/Matrix with Cholesky solver
- `src/format/` - APR format, validation, lint, converter, export
- `src/text/chat_template.rs` - Chat template engine (6+ formats, sandboxed Jinja2)
- `crates/apr-cli/` - CLI tool
- `docs/specifications/APR-SPEC.md` - Full APR format spec

## APR CLI (`crates/apr-cli/`)

Commands: `run`, `serve`, `compile`, `inspect`, `debug`, `validate`, `diff`, `tensors`, `trace`, `lint`, `explain`, `canary`, `export`, `import`, `convert`, `merge`, `tui`, `probar`, `profile`, `qa`

```bash
apr run hf://openai/whisper-tiny --input audio.wav
apr validate model.apr --quality
apr convert model.safetensors --quantize int8 -o model-int8.apr
apr export model.apr --format gguf -o model.gguf
apr merge model1.apr model2.apr --strategy weighted --weights 0.7,0.3 -o merged.apr
apr import hf://openai/whisper-tiny -o whisper.apr --arch whisper
apr qa model.gguf --assert-tps 100 --json
```

## PMAT Quality Analysis (v2.216.0)

**Scores:** Project 124/134 (A+), TDG 95.2/100 (A+), Coverage 96.35%, Mutation 85.3%
**Thresholds:** Coverage ≥95%, Complexity ≤10/fn, SATD 0, TDG ≥95, Mutation ≥85%, 0 unwrap()

```bash
pmat quality-gates              # Run all gates (config: .pmat-gates.toml)
pmat rust-project-score         # Project analysis
pmat analyze complexity         # Cyclomatic/cognitive complexity
pmat analyze satd               # Zero TODO/FIXME/HACK
pmat tdg . --include-components # Technical debt grading
pmat query "error handling"     # Semantic code search with quality annotations (RAG-powered)
pmat embed sync                 # Sync embeddings for codebase (run before query)
```

unwrap() banned via `.clippy.toml` disallowed-methods. Use `expect()` or `ok_or_else(|| ...)?`. See Issue #41.

## CRITICAL: Code Search Policy

**NEVER use grep/glob for code search. ALWAYS use pmat query.**

### Decision Tree

| Task | Command |
|------|---------|
| Find functions by intent | `pmat query "error handling" --limit 10` |
| Find important functions | `pmat query "mcp" --rank-by pagerank --limit 5` |
| Find most-called utilities | `pmat query "format" --rank-by indegree --limit 5` |
| Find in specific path | `pmat query "validate" --path src/api/` |
| Find high-quality code only | `pmat query "parse" --min-grade B --max-complexity 15` |

### Examples

```bash
# BAD - Raw text search returns 500+ noisy matches with no context
# GOOD - Semantic search returns 10 ranked functions with quality metrics
pmat query "error handling" --limit 10
```

### Cross-Project Search

The index automatically includes sibling projects (aprender, trueno, realizar).
Query from any project to search 60k+ functions across all three codebases.

```bash
# Build index in each project first (one-time setup)
cd ~/src/aprender && pmat query "init" --rebuild-index --limit 1
cd ~/src/trueno && pmat query "init" --rebuild-index --limit 1
cd ~/src/realizar && pmat query "init" --rebuild-index --limit 1

# Now query from any project - siblings auto-merge
pmat query "matrix multiplication" --limit 5
```

### Output Formats

- Default (text): Human-readable with signatures and metrics
- `--format json`: For parsing/scripting
- `--format markdown`: For documentation
- `--include-source`: Include full source code in results

### Quick Reference

```bash
pmat query "<intent>"                    # Basic search
pmat query "<intent>" --rank-by pagerank # Most important functions
pmat query "<intent>" --format json      # Machine-readable
pmat query "<intent>" --include-source   # Include full source code
pmat query "<intent>" --exclude-tests    # Skip test functions

# Git history search (find code by commit intent via RRF fusion)
pmat query "fix serialization" -G
pmat query "apr format" --git-history

# Enrichment flags (combine freely)
pmat query "ml algorithm" --churn                  # git volatility (commit count, churn score)
pmat query "tensor operation" --duplicates          # code clone detection (MinHash+LSH)
pmat query "loss function" --entropy                # pattern diversity (repetitive vs unique)
pmat query "model training" --churn --duplicates --entropy --faults -G  # full audit
```

### Coverage-Guided Search (pmat 3.0.0+)

**Use `pmat query --coverage` to find untested code. NEVER parse coverage JSON manually.**

```bash
# Find top uncovered functions (no query needed)
pmat query --coverage-gaps

# Find uncovered functions matching a semantic query
pmat query "error handling" --coverage --uncovered-only

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

## Stack Documentation Search

```bash
batuta oracle --rag "your question here"    # Search entire Sovereign AI Stack
batuta oracle --rag-index                   # Reindex (335 docs)
```

Use proactively for trueno SIMD patterns, cross-language equivalents, and stack best practices.
