# CLAUDE.md

## Project Overview

Aprender is a next-generation ML framework in pure Rust — **monorepo with 82 workspace crates**. Install: `cargo install aprender` → `apr` binary (103 subcommands). 25,300+ tests, 1460 provable contracts. Core library in `crates/aprender-core/` ([lib] name = "aprender"). All 20 repos (trueno, realizar, entrenar, batuta, + 15 satellites) consolidated per APR-MONO spec. **v0.34.0 SHIPPED 2026-05-18: MODEL-2 §88 stack-existence-proof, paiml/albor-370m-v1 LIVE on HF Hub.**

## Git Workflow (Branch Protection)

**`main` is protected.** Required status checks: `ci / gate` + `workspace-test`. Direct pushes blocked.

1. Create feature branch: `git checkout -b <name>`
2. Commit on branch, push: `git push -u origin <name>`
3. Open PR: `gh pr create`
4. CI must pass before merge — enforced by GitHub

## Autonomous Operation Mode (DEFAULT)

**Operate autonomously by default. Ship PRs, don't ask permission for routine work.**

### Just do it (no check-in needed)
- Run any read-only diagnostic (`apr inspect`, `apr trace`, `pmat query`, `gh pr view`)
- Build / test / lint / clippy on any branch
- Create feature branches and commit work-in-progress
- Open PRs with auto-merge armed (`gh pr merge --squash --auto --delete-branch`)
- Re-run failed CI jobs (`gh run rerun --failed`) per memory rules
- Update PR branches when BEHIND (`gh pr update-branch`)
- Author spec amendments (§N entries) when findings warrant
- Capture evidence into `evidence/section-NN-*/findings.json`
- Update memory files (`memory/*.md`) with new lessons
- Continue cascades — when one PR lands, automatically start the next prioritized item per §80-class queues
- Surface defects as new spec sections rather than asking "should I file this?"
- Pivot strategies (e.g. when P0-A blocks, immediately try P0-B; when P0-B blocks, surface §81-class amendment and continue to next prereq)

### Check in BEFORE acting (real escalations only)
- Compute spend > 1hr on non-lambda-vector hosts (lambda-vector is pre-authorized per `feedback_compute_pre_authorized.md`)
- Destructive ops: `git push --force`, `gh release delete`, dropping branches/tags on main, `cargo yank`
- Modifying CI workflows (`.github/workflows/*.yml`)
- Crates.io publish cascade (always ask before `make publish`)
- Architectural pivots (changing model architecture, retraining from scratch, switching tokenizers)
- Anything contradicting an explicit user instruction earlier in the session

### Telemetry mode (one-line per PR)
After landing or arming a PR, emit ONE line:
```
✅ PR #1699 (P0-F arch case mapping) auto-merge armed — 9/9 unit tests pass, llama-cli now accepts the GGUF
```
Then immediately continue to the next prioritized work. Do NOT ask "should I continue?" — assume yes.

### When to stop autonomously
Stop and summarize when:
- All P0/P1 items in the current §80-class priority queue are landed OR blocked on external compute/auth
- A surfaced defect requires architectural decision (e.g. "should we add `--force-overfit` or hard-fail?")
- 5+ PRs in flight all stuck on the same flake — surface the flake pattern rather than keep rerunning

### Loop behavior
When using `/loop`, treat fallback wakeups as cheap and merge events as primary. Don't poll, don't echo "still waiting." Monitor scripts should emit terminal states only (PASS, FAIL, MERGED).

## Build Commands

```bash
cargo build --release              # Optimized build (all 82 crates)
cargo test --workspace --lib       # Full workspace lib tests (25,300+)
cargo test -p aprender-core --lib  # Core ML library only (12,975)
cargo test -p apr-cli --lib        # CLI tests only (4,158)
cargo check --workspace            # Type-check all 82 crates
cargo fmt --check                  # Check formatting
cargo clippy -- -D warnings        # Strict linting

# Install
cargo install aprender             # Install `apr` binary (like cargo install ollama)
apr --version                      # Verify

# Makefile tiered quality gates
make tier1                   # Fast feedback (<1s): fmt, clippy, check
make tier2                   # Pre-commit (<5s): tests + strict clippy
make tier3                   # Pre-push (1-5min): full validation + coverage
make tier4                   # CI/CD: includes pmat analysis
make coverage                # Coverage report (target ≥95%)
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

**Monorepo layout** (82 crates, flat `crates/aprender-*` per Polars/Burn/Nushell pattern):
- `crates/aprender-core/` — ML library ([lib] name = "aprender")
- `crates/aprender-compute/` — SIMD/GPU (was trueno, [lib] name = "trueno")
- `crates/aprender-serve/` — inference server (was realizar, [lib] name = "realizar")
- `crates/aprender-train/` — training (was entrenar, [lib] name = "entrenar")
- `crates/apr-cli/` — CLI logic (internal, `apr` binary from root facade)
- Root `Cargo.toml` — workspace + facade (`cargo install aprender` → `apr`)

**Runtime:** `aprender-compute` (SIMD), `aprender-contracts` (provable contracts)
**Dev Tools:** `proptest`, `criterion`, `pmat`, `cargo-mutants`

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

### FFN Gate+Up Kernel Fusion (PMAT-FFN-FUSION)

The SwiGLU FFN block fuses gate and up projections into a single rayon dispatch via
`generic_fused_gate_up_matvec_into<F>` (realizar `quantize/fused_gate_up.rs`). This halves
rayon spawn overhead (56→28 dispatches/token on 28-layer models) and improves L1/L2 cache
reuse by loading the activation vector once per midi-tile instead of twice.

- **Fused path**: Q4K, Q5K, Q6K when both gate+up weights share the same qtype and dims
- **Fallback**: `rayon::join` with two separate `fused_matmul_into` for mixed types
- **Q8K path**: Existing `fused_q4k_q8k_ffn_up_gate_into` still used when Q8K activations available
- **Key files**: `realizar/src/quantize/fused_gate_up.rs`, `fused_matmul_into.rs` (`fused_gate_up_matmul_into`)

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

- ~~`src/models/qwen2/mod.rs::generate()` / `forward()`~~ - DELETED (Refs #224, #1977). `Qwen2Model` has no inference path; all inference (incl. KV caching) goes through `realizar`. Only construction, weight loading, and introspection remain.
- ~~`examples/qwen_inference.rs`~~ - DELETED (Refs #224). Use the `apr` CLI / `realizar` for inference.

## Publishing Safety (CB-510 Lesson)

**CRITICAL: `.gitignore` and `Cargo.toml` exclude patterns must use root-anchored paths.**

The `models/` pattern silently matches `src/models/` — hiding source code from git and crates.io. Always use `/models/` (root-anchored).

```bash
# Pre-publish checks (also in make tier3)
bash scripts/check_include_files.sh     # All 562 include!() files tracked by git
bash scripts/check_package_includes.sh  # All 319 src/ include!() files in cargo package

# After creating new include!() files, verify they're not gitignored:
git ls-files --others --exclude-standard src/
git check-ignore -v src/path/to/new_file.rs  # Should return exit 1 (not ignored)
```

**After any `.gitignore` or `Cargo.toml` exclude change:** re-run both scripts.

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
cargo test                    # All tests (12,974 lib + 4,599 integration)
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

- `crates/aprender-core/src/lib.rs` - ML library entry, module exports
- `crates/aprender-core/src/traits.rs` - Core traits (Estimator, UnsupervisedEstimator, Transformer)
- `crates/aprender-core/src/primitives/` - Vector/Matrix with Cholesky solver
- `crates/aprender-core/src/format/` - APR format, validation, lint, converter, export
- `crates/aprender-core/src/text/chat_template.rs` - Chat template engine
- `crates/apr-cli/` - CLI logic (82 commands)
- `src/bin/apr.rs` - Root binary entry point (`cargo install aprender`)
- `contracts/` - 1460 provable contracts (merged from all 20 repos)
- `docs/specifications/aprender-monorepo-consolidation.md` - Monorepo spec

## APR CLI (`cargo install aprender`)

82 commands across 10 categories. Contract: `contracts/apr-cli-commands-v1.yaml`.
Key commands: `run`, `chat`, `serve`, `pull`, `finetune`, `prune`, `distill`, `merge`, `quantize`, `inspect`, `debug`, `validate`, `diff`, `tensors`, `trace`, `lint`, `explain`, `export`, `import`, `convert`, `compile`, `train`, `tune`, `eval`, `bench`, `profile`, `qa`, `mcp`, `probar`, `cbtop`, `tui`, `hex`, `tree`, `flow`, `qualify`

```bash
apr run hf://openai/whisper-tiny --input audio.wav
apr validate model.apr --quality
apr convert model.safetensors --quantize int8 -o model-int8.apr
apr export model.apr --format gguf -o model.gguf
apr merge model1.apr model2.apr --strategy weighted --weights 0.7,0.3 -o merged.apr
apr import hf://openai/whisper-tiny -o whisper.apr --arch whisper
apr qa model.gguf --assert-tps 100 --json
```

## PMAT Quality Analysis (v3.10.0)

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

## Contract Validation: DOGFOOD `pv`, NEVER bash

**`provable-contracts` is merged in-tree (APR-MONO Phase 2b).** It lives as three crates:
- `crates/aprender-contracts/` — evaluation engine
- `crates/aprender-contracts-macros/` — `#[contract]` derive
- `crates/aprender-contracts-cli/` — `pv` binary

**`pv` is THE dogfooded contract CLI.** When you need to validate, lint, score, scaffold, diff, audit, generate proofs, or run falsification tests on a YAML contract in `contracts/`, use `pv`. Writing a bash/yq/python script that re-implements what `pv` already does is **muda** (waste) and will be rejected.

```bash
pv validate contracts/apr-code-parity-v1.yaml    # schema + falsification gates
pv lint contracts/                               # validate + audit + score on all
pv status contracts/tensor-layout-v1.yaml        # equations, obligations, coverage
pv query "tensor layout" --limit 5               # search contracts by intent
pv diff contracts/apr-mcp-server-v1.yaml HEAD~3  # semver bump suggestion
pv coverage                                      # cross-contract obligation coverage
```

40+ `pv` subcommands: `validate, scaffold, codegen, kani, probar, status, audit, diff, coverage, generate, graph, equations, lean, proof-status, lint, score, query, invariants, coq, fuzz, mirai, flux, tla, book, infer, roofline, pipeline, kaizen, certify, verify-structure, verify-pipeline, ...`.

**If `pv validate` rejects a contract** (wrong kind, missing required fields), the fix is one of:
1. Restructure the contract to fit the existing schema (usually `KernelContract` shape with `equations`, `proof_obligations`, `falsification_tests`).
2. Extend `aprender-contracts/src/schema/` to add the new `kind` + validator rule (real engineering task, own PMAT ticket).
3. If it genuinely isn't a provable contract, use a different YAML schema under a different directory and a purpose-built `apr` subcommand — not `contracts/`.

**Never** work around `pv` with a shell script. The in-tree tool is the source of truth.

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

**Workflow for coverage improvement (MUST co-evolve with contracts):**
1. `cargo llvm-cov test --lib --no-report` — generate coverage data
2. `pmat query --coverage-gaps --exclude-tests` — find top uncovered functions by impact
3. For EACH function being tested, ALSO:
   a. Add `#[contract]` annotation if missing
   b. Add/strengthen falsification conditions in the relevant contract YAML
   c. Eliminate placeholder preconditions
4. Write tests targeting those functions
5. `make coverage` — verify improvement
6. `pmat comply check` — verify contract density improved

**RULE: Coverage without contracts is REJECTED. Both must improve together.**
See monorepo spec Rule 7: Coverage + Contracts Co-Evolution.

## Stack Documentation Search

```bash
batuta oracle --rag "your question here"    # Search entire Sovereign AI Stack
batuta oracle --rag-index                   # Reindex (335 docs)
```

Use proactively for trueno SIMD patterns, cross-language equivalents, and stack best practices.

## SSC Training Infrastructure Status (2026-03-22)

- **SSC canary eval**: 90% accuracy, SHIP gate PASS — classifier ready to ship
- **entrenar cuBLAS integration**: GEMM parity verified between CPU and GPU paths
- **Blackwell (GB10) training**: Blocked by JIT pre-warming bug in custom PTX kernels. Must use fused NF4 kernel path (15.5 tok/s) until trueno 0.4.36 ships with pre-compiled kernels
- **apr-cli inference NOT affected**: `apr run` / `apr serve` use cuBLAS (GPU) or trueno SIMD (CPU) — pre-compiled, no custom PTX involved
- **Trained model (LoRA adapter)**: Architecture-independent safetensors — works on any GPU or CPU via standard PEFT loading
- **Key tickets**: trueno#200 (Blackwell JIT), trueno#203 (pre-compiled kernels), entrenar#300 (cuBLAS backward)
