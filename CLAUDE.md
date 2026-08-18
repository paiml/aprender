# CLAUDE.md

## Project Overview

Aprender is a next-generation ML framework in pure Rust — a monorepo where the
workspace, not this file, is the source of truth. Install: `cargo install aprender` →
`apr` binary. Core library in `crates/aprender-core/` ([lib] name = "aprender").
20 repos (5 core + 15 satellites, per `docs/specifications/aprender-monorepo-consolidation.md`)
were merged in and archived 20/20.

### Derive the numbers; do not quote them (#2331)

Every count in this file drifted. The published figures were wrong by 3.2× on the
test count and 29 minor versions on the release line before they were checked. So
**the table gives the command, and the value is only a dated sample.** If a number
here disagrees with its command, the command wins.

| Fact | Derive with | Sample (2026-08-13) |
|------|-------------|---------------------|
| Workspace crates | `cargo metadata --no-deps --format-version 1 \| python3 -c "import json,sys;print(len(json.load(sys.stdin)['packages']))"` | 78 — 77 under `crates/` plus the root facade |
| Dirs under `crates/` | `ls -1d crates/*/ \| wc -l` | 82. **This is not the crate count**: 4 are `exclude`d in the root `Cargo.toml`, and `aprender-contracts-staging` has no manifest. A directory is not a crate |
| `apr` subcommands | `apr --help`; registry is `contracts/apr-cli-commands-v1.yaml` §`commands`, mirrored by `crates/apr-cli/tests/cli_commands.rs::registered_commands` | 111, in 10 categories |
| Provable contracts | `find contracts -name '*.yaml' \| wc -l` | 1768 |
| Workspace lib tests | the `Summary` line of CI's `workspace-test` job (see Build Commands for the exact nextest invocation) | **80,604 passed**, 130 skipped, across 69 binaries — CI run `31631488466`, `main` @ `d40756541`, 2026-08-12 |
| Released version | `git tag --sort=-creatordate \| head -1` · `gh release list` | **v0.63.0**, 2026-08-01 ("provenance") |

`crates/aprender-core/tests/readme_contract.rs` is the drift gate for all three published
docs: README.md's crate/contract counts, `docs/BEATS.md` vs the beat contracts, and every
repo-relative file path cited in **this file**. A path here that does not exist fails the
build (FALSIFY-DOCS-CLAUDE-001) — that is what caught the six pre-monorepo `realizar/…`
and `src/format/…` paths this file still advertised. Counts it cannot check, you re-derive.

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
cargo build --release              # Optimized build (every workspace member; no default-members)
cargo test -p aprender-core --lib  # Core ML library only
cargo test -p apr-cli --lib        # CLI tests only
cargo check --workspace            # Type-check the whole workspace
cargo fmt --all -- --check         # Check formatting (--all, not just the root package)
cargo clippy -p <crate> --lib -- -D warnings   # Strict lint. Bare `cargo clippy` only
                                   # lints the ROOT FACADE package, which is nearly empty

# What CI's `workspace-test` job actually runs (.github/workflows/ci.yml). Reproduce
# THIS, not `cargo test --workspace --lib` — the three excluded crates need a GPU
# toolchain and are gated separately:
cargo nextest run --profile ci --workspace --lib \
    --exclude aprender-gpu --exclude aprender-cuda-edge --exclude aprender-compute
cargo test -p aprender-compute --lib   # the SIMD crate, run as its own CI step

# Install
cargo install aprender             # Install `apr` binary (like cargo install ollama)
apr --version                      # Verify

# Makefile tiered quality gates
make tier1                   # Fast feedback (<1s): fmt, clippy, check
make tier2                   # Pre-commit (<5s): tests + strict clippy
make tier3                   # Pre-push (1-5min): full validation + coverage
make tier4                   # CI/CD: includes pmat analysis
make coverage                # Coverage report (enforced floor 88%, target ≥95%)
```

## Debugging: Use apr Tools First (MANDATORY)

**STOP. Before reading code or grepping, USE THE APR DIAGNOSTIC TOOLCHAIN.**

GH-202 lesson: we read code instead of running `apr qa` which would have instantly shown the failure.

**Step 0 — pin the binary, ALWAYS.** Never invoke a bare `apr`, and never hardcode
an absolute path to one. Four `apr` binaries were found coexisting on the dev box
(0.60.0 ×2, 0.61.0, 0.62.0); a bare `apr` resolved to a **26-day-old** copy, and
the path this file used to call "canonical" was two minor versions stale. There is
no correct path to hardcode — `.cargo/config.toml` [gitignored] redirects cargo's
target-dir, so the main checkout and a fresh worktree build to different places.

```bash
. scripts/apr_bin.sh || exit 1   # exports $APR, proves it was built from HEAD
```

Everything below uses `"$APR"`. A diagnostic run against the wrong binary is worse
than no diagnostic: it produces a confident answer about code you are not running.

```bash
# Step 1: ALWAYS start here (catches 80% of issues)
"$APR" qa model.apr

# Step 2: Check tensor shapes/stats
"$APR" tensors model.apr | head -20

# Step 3: Diff against known-good model
"$APR" diff model.apr reference.gguf

# Step 4: Format/metadata integrity
"$APR" validate model.apr --quality
"$APR" lint model.apr

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

**There is no `realizar` binary.** `realizar` is the *library* name of the
`aprender-serve` package (`[lib] name = "realizar"`, `crates/aprender-serve/Cargo.toml`);
that package ships no `[[bin]]`. Tracing is driven through `apr run`:

```bash
"$APR" run model.safetensors --prompt "2+2?" --trace
"$APR" run model.gguf --prompt "Hi" --trace --trace-steps tokenize,sample,decode
"$APR" run model.gguf --prompt "Hi" --trace --trace-level payload   # or --trace-payload
```

The flag is `--trace-steps <a,b,c>` (comma-delimited), not `--trace=<...>`.
`--trace-level` accepts `none|basic|layer|payload|chrome` and defaults to `basic`.

Implementation: `crates/aprender-serve/src/inference_trace/` (a DIRECTORY — `mod.rs`
plus `save_tensor*.rs`, `gpu_stage_dump.rs`, `tracer_contracts.rs`, …).

TraceSteps (`TraceStep` in `crates/aprender-serve/src/inference_trace/mod.rs`): `Tokenize`, `Embed`, `LayerNorm`,
`Attention`, `FFN`, `TransformerBlock`, `LmHead`, `Sample`, `Decode`, `KernelLaunch`
(PTX-level, GH-219), `BrickProfile` (trueno `BrickProfiler`).

## Architecture

1. **Trait-Based Multiple Dispatch** - Julia-inspired pattern
2. **Backend Agnostic** - CPU (SIMD), GPU, WASM via Trueno
3. **Three-Tier API**: High (`Estimator` trait), Mid (`Optimizer`/`Loss`/`Regularizer`), Low (Trueno primitives)

**Monorepo layout** (flat `crates/aprender-*` per the Polars/Burn/Nushell pattern; for
the crate count run the command in the Project Overview table, don't trust a number here):
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

Feature flag (`crates/apr-cli/Cargo.toml`): `inference = ["realizar", "trueno", "tokio",
"axum", "futures-util"]`, and `default = ["hf-hub", "safetensors-compare", "inference",
"training", "visualization", "zram"]` — so `inference` is on unless you pass
`--no-default-features`. GPU work needs `--features cuda`, which pulls in `inference`,
`realizar/cuda` and `entrenar/cuda`.
Always profile with `apr profile`/`apr trace`/`apr bench` before optimizing.

### Performance Targets (Ollama Parity)

| Model | CPU (tok/s) | GPU (tok/s) | Memory |
|-------|-------------|-------------|--------|
| 1B Q4_K | 100+ | 500+ | 600MB |
| 7B Q4_K | 30+ | 150+ | 4GB |
| 13B Q4_K | 15+ | 80+ | 8GB |

These are **targets**, not measurements. For what is actually measured against Ollama,
see `docs/BEATS.md`: GPU decode on RTX 4090 sm_89 is at **parity** (1.015–1.109×), and
`contracts/beat-ollama-decode-throughput-speed-v1.yaml` enforces `beat_threshold: 0.9000`
— a no-collapse floor. The old "apr beats Ollama 1.371×" headline is **withdrawn**.

Architecture: Trueno SIMD backend, realizar fused dequant+matmul kernels, PagedAttention KV cache, optional wgpu/CUDA.

### FFN Gate+Up Kernel Fusion (PMAT-FFN-FUSION)

The SwiGLU FFN block fuses gate and up projections into a single rayon dispatch via
`generic_fused_gate_up_matvec_into<F>` (`crates/aprender-serve/src/quantize/fused_gate_up.rs:63`). This halves
rayon spawn overhead (56→28 dispatches/token on 28-layer models) and improves L1/L2 cache
reuse by loading the activation vector once per midi-tile instead of twice.

- **Fused path**: Q4K, Q5K, Q6K when both gate+up weights share the same qtype and dims
- **Fallback**: `rayon::join` with two separate `fused_matmul_into` for mixed types
- **Q8K path**: Existing `fused_q4k_q8k_ffn_up_gate_into` still used when Q8K activations available
- **Key files**: `crates/aprender-serve/src/quantize/fused_gate_up.rs`,
  `crates/aprender-serve/src/gguf/inference/fused_matmul_into.rs` (`fused_gate_up_matmul_into`)

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

**Key Files** (all under `crates/aprender-core/` — the pre-monorepo top-level `src/`
paths this file used to give have not existed since APR-MONO):
- `contracts/tensor-layout-v1.yaml` - **SOURCE OF TRUTH**
- `crates/aprender-core/src/format/layout_contract.rs` - Rust validation API
- `crates/aprender-core/src/format/converter/write.rs` - GGUF→APR import with transpose
- `crates/aprender-core/src/format/converter/mod.rs` - `transpose_q4k_for_matmul()`, `transpose_q6k_for_matmul()`

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
# Pre-publish checks (also in make tier3). Both print the count they checked —
# read it off the output, don't quote a number from this file.
bash scripts/check_include_files.sh     # scans src/ AND crates/; printed 1771 on 2026-08-13
bash scripts/check_package_includes.sh  # scans src/ ONLY, against `cargo package -p aprender --list`

# After creating new include!() files, verify they're not gitignored:
git ls-files --others --exclude-standard crates/
git check-ignore -v crates/<crate>/src/path/to/new_file.rs  # exit 1 == not ignored (good)
```

**`check_package_includes.sh` is currently vacuous — know this before trusting it.**
It greps `include!(` in the root `src/` only, and the root `src/` is a two-file facade
(`lib.rs`, `bin/`) with **zero** `include!()` directives. So it reports
`OK: All 0 include!() files are included in cargo package` and can never fail. The
real coverage against CB-510 today comes from `check_include_files.sh` (1771 files,
`src/` + `crates/`). Extending the package check to the 70+ publishable member crates
is open work — do not treat its green as evidence.

**After any `.gitignore` or `Cargo.toml` exclude change:** re-run both scripts.

## Shell Scripts: Use bashrs (NOT shellcheck)

```bash
bashrs lint scripts/*.sh          # Lint
bashrs purify scripts/ci.sh       # Determinism + idempotency
bashrs make lint Makefile          # Makefile linting
bashrs gate --strict .             # Full quality gate
```

Required: `set -euo pipefail`, no `ls` for iteration, quoted variables, explicit error handling.

**Exception — a SOURCED library must be option-neutral.** `set` in a sourced file
mutates the *caller's* shell. `scripts/apr_bin.sh` opened with `set -euo pipefail`;
`qwen-story.sh` sources it and had deliberately chosen no `-e` so it could run every
beat and tally failures. The leak killed the nightly six lines in. Sourceable
libraries fail by **return status** instead: `. scripts/lib.sh || exit 1`. Enforced by
`scripts/check_sourced_libs_option_neutral.sh`.

## Verification Discipline (MANDATORY — read before reporting any result)

These are not style notes. Each cost real time, and in every case the general
principle was known and the specific instance still went wrong.

**1. Never read `$?` through a pipe.** It is the LAST command's status.
```bash
cmd > /tmp/out.log 2>&1; rc=$?               # correct
cmd | tee /tmp/out.log; rc=${PIPESTATUS[0]}  # correct (bash)
cmd | grep -E "^error"; echo "exit=$?"       # WRONG — that is grep's status
```
This shipped twice — `qwen-story-daily` captured `tee`'s status so its fail-the-job
step was unreachable and three green runs proved nothing (#2336); `make publish`'s
POST-PUBLISH VERIFICATION did the same, so it could never report a broken published
crate (#2360). **When a result looks good, check how it was measured.**

**2. Never label a run by intent — prove the mechanism engaged.** A repro harness
printed `device: GPU` while built without `--features cuda`; three findings were
reported from CPU runs. `CUDA_VISIBLE_DEVICES` says what is *visible*, never what was
*used*. Cite a trace line, a version+SHA, or a behaviour delta.

**3. Pin the binary.** Never a bare `apr`, never a hardcoded absolute path — see
Step 0 above. Four `apr` binaries once coexisted here and a bare `apr` resolved to a
26-day-old one.

**4. Extending a guard's SCOPE requires re-mutating in the new scope.** The old proof
does not transfer. Extending `check_apr_bin_pinned.sh` to the Makefile found real
violations *and* was still blind to `\t@apr …`, the most common Makefile form — caught
only by re-running the RED-turning mutation there (#2360).

**5. A guard that does not scan the surface where the DECISION is made is theater.**
Enumerate the decision surfaces — release, publish, certify, gate — then check
coverage. Covering "CI" is not covering "the release".

**6. One failing input is an anecdote.** Vary it before naming a cause, and
especially before blocking a release. Four neighbouring prompts once inverted a
diagnosis from "GPU correctness defect" to "the gate sampled a near-tie" (#2359).

**7. Guard regexes ship a case table.** The `apr`-invocation patterns were wrong five
times; every one was caught by a must-match/must-not-match table, none by review.
Re-run the table rather than re-reading the pattern.

**8. A shadowed artifact is worse than a missing one** — edits look effective and
change nothing. `~/.local/bin/apr` shadowed a fresh install; `~/.claude/skills/dogfood/`
shadowed the repo's release-certifying skill so hardening it edited a file that never
ran (#2361). When a fix seems to have no effect, ask what else claims that name
(`type -aP <cmd>`; user-scope vs repo-scope skills; always set an explicit `name:`).

## Testing

Target: 60% unit, 30% property, 10% integration. Coverage: **88.78% line** (786448/885829, measured 2026-07-29 by coverage-nightly on 95145584f; target ≥95%, enforced floor 88% via COV_FLOOR). The long-quoted "96.35%" predates the measurement ever working - the pipeline reported 0/0 until #2333.

```bash
cargo test -p <crate> --lib             # Unit tests for one crate (what you run while working)
cargo test -p <crate> --test <target>   # ONE integration target. There is NO root
                                        # `tests/` dir, so `cargo test --test integration`
                                        # fails at the workspace root — the `integration.rs`
                                        # targets live per-crate (aprender-core, aprender-data,
                                        # aprender-registry, …)
cargo test --doc                        # Doctests
make coverage                           # Coverage report (disables mold linker, single-phase llvm-cov)
```

For the workspace-wide number, reproduce CI's `workspace-test` nextest command (see
Build Commands) and read its `Summary` line — 80,604 tests on 2026-08-12. Only a
subset of integration targets is wired into CI: `.github/workflows/ci.yml` runs `--lib`
across the workspace, plus ONE explicit line listing the individual `--test` targets
(beats, `cli_commands`, `monorepo_invariants`, `readme_contract`, …). A new
`tests/*.rs` file is dark until it is added to that line.

Mutation testing: `cargo mutants --no-times --timeout 300 --in-place -- --all-features` (or via CI).

## Linting

Workspace-level lints in `Cargo.toml` (`[workspace.lints.rust]` / `[workspace.lints.clippy]`).
Key: `unsafe_code = "forbid"`, `clippy::all + pedantic = "warn"`, ML-specific allows for casts/float_cmp.

**Both ends of the toolchain range are gated (#2370).** `rust-toolchain.toml` pins one
exact release, so every gate we own — `make tier1/2/3`, sovereign-ci `lint` — lints
under that one clippy and nothing else.

| End | Guard | Runs |
|-----|-------|------|
| FLOOR — declared `rust-version` still builds | `scripts/check_msrv.sh` | on demand |
| CEILING — current stable clippy is clean | `scripts/check_clippy_current_stable.sh` / `make lint-current` | `toolchain-ceiling.yml`, daily 05:00 UTC |

Without the ceiling gate, findings from newer clippy releases accumulate invisibly:
#2370 was a fresh mbp whose homebrew rustc is not rustup-managed (so the pin is
silently inert) running plain `make` and getting 28 errors out of `tier2` — the tree
had 107 findings across 12 lints by then, none of which any gate had ever run.
The ceiling gate refuses to pass vacuously (stale `stable`, missing clippy component,
broken version comparator) and ships a comparator case table that `tier3` re-runs.

Clippy's lint set is **not monotonic**: the #2370 tree is clean on 1.93/1.96/1.97 and
1.95 *alone* reports 8 `collapsible_match` findings. A green ceiling gate means
"clean on the pin and on current stable", never "clean on every release between".

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
- `crates/aprender-core/src/text/chat_template/` - Chat template engine (a DIRECTORY: mod.rs + template.rs/raw_template.rs/ship_008.rs via include!)
- `crates/apr-cli/` - CLI logic (command registry: `contracts/apr-cli-commands-v1.yaml`)
- `src/bin/apr.rs` - Root binary entry point (`cargo install aprender`)
- `contracts/` - provable contracts, merged from all 20 repos (`find contracts -name '*.yaml' | wc -l`)
- `docs/specifications/aprender-monorepo-consolidation.md` - Monorepo spec
- `docs/BEATS.md` - the public beat scoreboard. Gated against `contracts/` by
  `crates/aprender-core/tests/readme_contract.rs`

## APR CLI (`cargo install aprender`)

111 commands across 10 categories as of 2026-08-15; the registry is
`contracts/apr-cli-commands-v1.yaml` (§`commands`), mirrored by
`crates/apr-cli/tests/cli_commands.rs::registered_commands` and enforced by
FALSIFY-CLI-001/002. The contract deliberately hand-maintains no count — parse
the `commands:` list, never `grep -c '^  - name:'` (117; other same-indent
`name:` keys exist in the file).
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

## PMAT Quality Analysis

Version: run `pmat --version` (3.30.0 on this box, 2026-08-13). `.pmat-gates.toml`
still carries a header comment claiming it was "Updated for PMAT v2.215.0".

**Scores.** Only one of these has a citable measurement, so only one is stated:

| Score | Value | Provenance |
|-------|-------|------------|
| Line coverage | **88.78%** (786448/885829) | coverage-nightly, 2026-07-29, commit `95145584f`. The long-quoted "96.35%"/"96.94%" predates the pipeline ever working — it reported 0/0 until #2333 |
| Project score / TDG / mutation % | **re-derive** — `pmat rust-project-score`, `pmat tdg . --include-components`, `cargo mutants` | The previously published "124/134", "TDG 95.2/100" and "Mutation 85.3%" carried no date or commit and could not be reproduced from the tree |

**Thresholds — read from the config, which does not say what this file used to say:**

| Gate | Configured as | Where |
|------|---------------|-------|
| Coverage (aspirational) | `min_coverage = 95.0` | `.pmat-gates.toml` |
| Coverage (**enforced**) | `COV_FLOOR := 88` — the last *measured* value, and the one that actually fails a build | `Makefile:287`, `.github/workflows/coverage-nightly.yml` |
| Cyclomatic complexity | `max_complexity = 10` per fn | `.pmat-gates.toml` |
| TDG | `min_grade = "B"` — **not** "≥95" | `.pmat-gates.toml` `[tdg]` |
| Mutation | `MUTANTS_MAX_MISSED` (default **0**) surviving mutants **on the PR diff** — not a global 85% score | `.github/workflows/ci.yml:517` |
| Verification ladder | `min_level = "L3"` | `.pmat-gates.toml` |
| `unwrap()` | banned outright | `.clippy.toml` disallowed-methods |

SATD is checked by `pmat analyze satd`, but no SATD threshold is configured in
`.pmat-gates.toml`; the pre-commit hook there is `pmat comply check --failures-only`.

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

`pv --help` lists the full set (42 subcommands + `help` in pv 0.49.0): `explain,
validate, check-parity, scaffold, extract-pytorch, codegen, kani, probar, status,
audit, diff, coverage, generate, graph, equations, lean, lean-status, proof-status,
lint, score, query, invariants, coq, fuzz, mirai, flux, tla, book, infer, unlock,
roofline, pipeline, kaizen, certify, verify-structure, verify-pipeline,
verify-bindings, migrate`.

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
batuta oracle --rag-index                   # Reindex (the command prints the doc count)
```

Use proactively for trueno SIMD patterns, cross-language equivalents, and stack best practices.

## SSC Training Infrastructure Status (snapshot 2026-03-22 — STALE, re-verify before acting)

This block has not been re-measured in ~5 months and one of its premises no longer
holds: it points at "trueno 0.4.36" as an external crate, but since APR-MONO trueno is
in-tree as `crates/aprender-compute` and has no independent version to wait on.

- **SSC canary eval**: 90% accuracy, SHIP gate PASS — classifier ready to ship
- **entrenar cuBLAS integration**: GEMM parity verified between CPU and GPU paths
- **Blackwell (GB10) training**: Blocked by JIT pre-warming bug in custom PTX kernels. Must use fused NF4 kernel path (15.5 tok/s) until trueno 0.4.36 ships with pre-compiled kernels
- **apr-cli inference NOT affected**: `apr run` / `apr serve` use cuBLAS (GPU) or trueno SIMD (CPU) — pre-compiled, no custom PTX involved
- **Trained model (LoRA adapter)**: Architecture-independent safetensors — works on any GPU or CPU via standard PEFT loading
- **Key tickets**: trueno#200 (Blackwell JIT), trueno#203 (pre-compiled kernels), entrenar#300 (cuBLAS backward)
