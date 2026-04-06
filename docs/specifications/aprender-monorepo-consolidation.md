# APR-MONO: Sovereign Stack Monorepo Consolidation

**Version**: 1.3
**Date**: 2026-04-06
**Status**: PROPOSAL — Ready for Review
**Priority**: P0 — Unblocks daily apr-cli releases
**Author**: PAIML Team + Claude
**Contract**: `contracts/cgp/cgp-monorepo-consolidation-v1.yaml`
**Falsification**: 10 conditions (FALSIFY-MONO-001 through 010)

---

## Architectural Invariant: apr-cli Is THE Binary

**This is a HARD REQUIREMENT. No exceptions.**

### Rule 1: One Binary, One Entry Point

`apr-cli` (binary name: `apr`) is the **only** user-facing CLI binary in the
monorepo. All functionality — inference, training, serving, profiling,
orchestration, data management, model registry — is accessed via `apr`
subcommands. No other crate in the workspace may produce a user-facing binary.

**Before (7+ binaries across repos):**
```
cargo install batuta        # batuta binary
cargo install entrenar      # entrenar binary
cargo install realizar      # realizar binary
cargo install cbtop         # cbtop binary
cargo install renacer       # renacer binary
cargo install presentar     # presentar binary
cargo install trueno-rag    # trueno-rag binary
```

**After (1 binary):**
```
cargo install apr-cli       # apr binary — THE entry point
apr run                     # inference (was: realizar)
apr train                   # training (was: entrenar)
apr serve                   # serving (was: realizar serve)
apr orchestrate             # agents, playbooks (was: batuta)
apr profile                 # profiling (was: renacer)
apr monitor                 # GPU/system monitoring (was: cbtop)
apr rag                     # RAG pipeline (was: trueno-rag)
apr registry                # model registry (was: pacha)
apr present                 # TUI dashboards (was: presentar)
apr test                    # WASM/browser testing (was: probar)
apr contracts               # provable contracts (was: pv)
```

### Rule 2: Libraries Are Libraries

Every `aprender-*` crate is a **library**. They expose `pub fn` APIs consumed
by `apr-cli` or by external Rust code via `use aprender_compute::*;`.
They do NOT produce binaries, CLIs, or executables.

Exception: `aprender-contracts-cli` may produce a `pv` binary for standalone
contract validation (build tooling, not user-facing ML tooling).

### Rule 3: CLI Contract Coverage

Every `apr` subcommand MUST have:

1. **A provable contract** in `contracts/` defining inputs, outputs, and
   falsification conditions
2. **A clap `#[derive(Parser)]`** with `#[command(about = "...")]` doc
3. **An integration test** in `apr-cli/tests/` verifying the subcommand
   runs without error
4. **A cookbook entry** in `cookbook/` showing usage with real models

**Enforcement**: CI checks that every `Commands` enum variant has a
matching contract YAML:

```bash
# Extract subcommand names from clap derive
grep -oP '^\s+(\w+)\s*[{,]' crates/apr-cli/src/commands_enum.rs \
  | tr -d ' {,' | tr '[:upper:]' '[:lower:]' > /tmp/subcommands.txt

# Check each has a contract
while read cmd; do
  if ! ls contracts/apr-cli-${cmd}-*.yaml 2>/dev/null | head -1 > /dev/null; then
    echo "MISSING CONTRACT: apr $cmd"
    FAIL=1
  fi
done < /tmp/subcommands.txt
[ -z "$FAIL" ] || exit 1
```

### Rule 4: Namespace Discipline

| Pattern | Allowed | Example |
|---------|---------|---------|
| `apr <subcommand>` | Yes — user-facing CLI | `apr run model.apr` |
| `aprender-*` crate name | Yes — library crate | `aprender-compute = "0.29"` |
| `aprender::*` Rust import | Yes — library API | `use aprender::format::AprFile;` |
| Standalone binary from `aprender-*` crate | **NO** | ~~`aprender-serve` binary~~ |
| Old binary names (`batuta`, `entrenar`, etc.) | **NO** (archived) | ~~`cargo install batuta`~~ |

---

### Citations

| # | Reference | Relevance |
|---|-----------|-----------|
| [1] | Potvin & Levenberg, "Why Google Stores Billions of Lines of Code in a Single Repository," CACM 59(7), July 2016. DOI: 10.1145/2854146 | Monorepo enables atomic changes, unified tooling. Scale: 2B lines, 45K commits/day. |
| [2] | Brousse, "The Issue of Monorepo and Polyrepo in Large Enterprises," ACM ICSE Companion 2019, pp. 150-159. DOI: 10.1109/ICSE-Companion.2019.00062 | Taxonomy: monorepo wins for tightly-coupled projects; polyrepo for independent products. |
| [3] | Brito et al., "On the Use of Monorepos in Open Source Projects," MSR 2023 | Empirical: 377 monorepos, median 8 packages. Motivation: shared deps, atomic changes. |
| [4] | Rastogi et al., "Dependency Smells in JavaScript Monorepo Projects," ICSME 2023 | Diamond dep elimination is the #1 measurable benefit. Version skew drops to zero. |
| [5] | PAIML clean-room-spec.md | 9 whack-a-mole patterns, 19 broken publishes from `[patch.crates-io]`. |
| [6] | PAIML release-system.md | Trusted Publishing, OIDC, tag-triggered releases. |
| [7] | PAIML unified-ci-pipeline.md | sovereign-ci.yml reusable workflow, 20/20 repos GREEN. |

---

## Executive Summary

Merge **19 repositories** (trueno, aprender, entrenar, realizar, batuta,
presentar, renacer, certeza, provable-contracts, trueno-{db,graph,rag,viz,zram},
alimentar, simular, repartir, verificar, probar) into
a **single `paiml/aprender` monorepo** with ~48 workspace crates under the
`aprender-*` namespace. This eliminates the cross-repo version sync problem
that has caused **19 broken crates.io publishes** (paiml/aprender#701) and
enables daily `apr-cli` releases from a single `cargo publish -p apr-cli`.

### Precedent

Every successful large Rust project uses this pattern:

| Project | Crates | Repo | Pattern |
|---------|--------|------|---------|
| Polars | 28 | 1 (`pola-rs/polars`) | `polars-{core,lazy,io,...}` |
| Burn (ML) | 33 | 1 (`tracel-ai/burn`) | `burn-{tensor,train,wgpu,...}` |
| Nushell | 30+ | 1 (`nushell/nushell`) | `nu-{cli,command,engine,...}` |
| DataFusion | 15 | 1 (`apache/datafusion`) | `datafusion-{common,expr,...}` |
| TiKV | 20+ | 1 (`tikv/tikv`) | `tikv-{client,server,...}` |
| **PAIML (current)** | **32+** | **5** | **4 namespaces, 19 broken publishes** |

---

## Current State (5 repos, 4752 .rs files, 32+ published crates)

### Repository Inventory

| Repo | Files | Version | Published Crates | Role |
|------|-------|---------|-----------------|------|
| trueno | 478 | 0.18.0 | 18 (`trueno-*`) | Compute: SIMD, GPU, WASM |
| aprender | 1179 | 0.27.8 | 4 (`aprender-*`) | ML format, tokenizers, model ops |
| entrenar | 1052 | 0.7.13 | 7 (`entrenar-*`) | Training loops |
| realizar | 1499 | 0.8.6 | 1 | Inference server |
| batuta | 544 | 0.7.3 | 2 | Orchestration, agents, RAG oracle |
| **Total** | **4752** | — | **32** | — |

### Satellite Crates (separate repos, stack-dependent)

These crates live in their own repos but depend on the core stack:

| Repo | Version | Files | Role | Disposition |
|------|---------|-------|------|-------------|
| presentar | 0.3.5 | 1 | TUI framework (workspace) | **MERGE** — core UI for cbtop, batuta |
| renacer | 0.10.2 | 119 | Profiling/tracing | **MERGE** — used by all 5 core crates |
| certeza | 0.1.1 | 9 | Quality validation | **MERGE** — tiny, used in CI |
| trueno-db | 0.3.17 | 27 | Embedded analytics DB | **MERGE** — already trueno-namespaced |
| trueno-graph | 0.1.18 | 23 | Graph database | **MERGE** — already trueno-namespaced |
| trueno-rag | 0.2.5 | 42 | RAG pipeline | **MERGE** — already trueno-namespaced |
| trueno-viz | 0.2.4 | 114 | Visualization | **MERGE** — already trueno-namespaced |
| trueno-zram | 0.3.1 | 3 | Compressed RAM (workspace) | **MERGE** — already trueno-namespaced |
| batuta-common | 0.1.0 | 6 | Shared batuta types | **MERGE** — folded into aprender-orchestrate |
| repartir | 2.0.4 | 23 | Distributed computing | **MERGE** — used by batuta |
| manzana | 0.1.0 | 10 | Apple hardware interfaces | KEEP SEPARATE — platform-specific |
| whisper.apr | 0.2.8 | 197 | Whisper speech model | KEEP SEPARATE — application, not framework |
| alimentar | 0.2.9 | 83 | Data loading/synthetic data | **MERGE** — core data pipeline |
| simular | 0.3.2 | 93 | Simulation framework | **MERGE** — used by training |
| verificar | 0.5.0 | 52 | Verification/testing | **MERGE** — used by CI/quality |
| probar | 1.0.3 | 1 (workspace: 4 crates) | WASM/browser test framework | **MERGE** — depends on trueno+presentar |
| provable-contracts | 0.2.2 | 1 (workspace: 3 crates) | Contract macros + YAML | **MERGE** — trueno build.rs reads its binding.yaml via path dep |
| pacha | 0.2.6 | 35 | Model/data registry + lineage | **MERGE** — depends on aprender+trueno-graph |

**Updated totals with satellites:**
- **Merge into monorepo**: 5 core + 15 satellites = 20 repos
- **Keep separate**: manzana, whisper.apr, forjar (+ pmat, which is its own product)
- **Total .rs files**: ~5500+
- **Total workspace crates**: ~48

### Dependency Graph (Current)

```
apr-cli ──→ aprender ──→ trueno 0.17, trueno-quant
       ──→ entrenar ──→ trueno 0.17, aprender 0.27, realizar(opt)
       ──→ realizar ──→ trueno 0.17, trueno-gpu, aprender(opt)
       ──→ batuta(?) ──→ trueno 0.16, aprender, entrenar, realizar
       ──→ trueno 0.17, trueno-explain, trueno-viz
```

**Problems:**
1. **Version skew**: trueno is 0.18.0 but all consumers pin 0.17 → diamond deps
2. **[patch.crates-io]** hacks required during development → leak to publishes
3. **Publishing order matters**: trueno → aprender → entrenar → realizar → apr-cli (5 sequential publishes, any can break)
4. **Circular deps**: aprender→trueno, but trueno's inference needs aprender's tokenizer
5. **19 broken publishes** documented in paiml/aprender#701

---

## Proposed Structure

```
paiml/aprender/                          # THE monorepo
├── Cargo.toml                           # workspace root
│   [workspace]
│   members = ["crates/*"]
│   [workspace.package]
│   version = "0.29.0"                   # ALL crates share one version
│
├── crates/
│   │
│   │ ── User-facing ──
│   ├── apr-cli/                         # Binary: `apr` command (DAILY releases)
│   │
│   │ ── Core ML ──
│   ├── aprender/                        # ML format (.apr), tokenizers, model ops
│   ├── aprender-train/                  # Was: entrenar (training loops)
│   ├── aprender-serve/                  # Was: realizar (inference server)
│   ├── aprender-orchestrate/            # Was: batuta (agents, RAG oracle, playbooks)
│   │
│   │ ── Compute primitives ──
│   ├── aprender-compute/                # Was: trueno (SIMD/GPU/WASM core)
│   ├── aprender-gpu/                    # Was: trueno-gpu (CUDA PTX, no nvcc)
│   ├── aprender-quant/                  # Was: trueno-quant
│   ├── aprender-gemm-codegen/           # Was: trueno-gemm-codegen
│   ├── aprender-inference/              # Was: trueno src/inference/ (GGUF, LlamaModel)
│   │
│   │ ── Data & Storage ──
│   ├── aprender-db/                     # Was: trueno-db
│   ├── aprender-rag/                    # Was: trueno-rag
│   ├── aprender-graph/                  # Was: trueno-graph
│   │
│   │ ── Visualization & Tooling ──
│   ├── aprender-viz/                    # Was: trueno-viz
│   ├── aprender-explain/                # Was: trueno-explain
│   ├── aprender-profile/               # Was: renacer (profiling/tracing)
│   ├── aprender-present/               # Was: presentar (TUI framework)
│   ├── aprender-shell/                  # REPL (already in aprender)
│   ├── aprender-verify/                # Was: certeza (quality validation)
│   │
│   │ ── Training sub-crates ──
│   ├── aprender-train-common/           # Was: entrenar-common
│   ├── aprender-train-lora/             # Was: entrenar-lora
│   │
│   │ ── Data & Simulation ──
│   ├── aprender-data/                   # Was: alimentar (data loading, synthetic data)
│   ├── aprender-simulate/              # Was: simular (simulation framework)
│   ├── aprender-distribute/            # Was: repartir (distributed computing)
│   │
│   │ ── Edge / Specialized ──
│   ├── aprender-cuda-edge/              # Was: trueno-cuda-edge
│   ├── aprender-zram/                   # Was: trueno-zram-core + trueno-zram-adaptive
│   ├── aprender-fft/                    # Was: trueno-fft
│   ├── aprender-sparse/                 # Was: trueno-sparse
│   ├── aprender-solve/                  # Was: trueno-solve
│   ├── aprender-rand/                   # Was: trueno-rand
│   ├── aprender-image/                  # Was: trueno-image
│   ├── aprender-tensor/                 # Was: trueno-tensor
│   │
│   │ ── Benchmarks & Testing ──
│   ├── aprender-bench-tokenizer/        # Already in aprender
│   ├── aprender-bench-compute/          # Already in aprender
│   └── aprender-tsp/                    # Already in aprender
│
├── contracts/                           # ALL provable contracts (merged)
├── book/                                # Unified mdbook documentation
├── cookbook/                             # apr-cookbook (merged in)
└── docs/specifications/                 # Specs (merged)
```

### Crate Count: ~42 workspace members

Comparable to Polars (28), Burn (33), Nushell (30+). Slightly larger
but includes infrastructure that Polars gets from external deps
(we own the full stack: DB, graph, profiler, TUI, distributed compute).

---

## Backward Compatibility

### Re-export shim crates

Old crate names continue to work via thin re-export crates:

```rust
// trueno/src/lib.rs (published as trueno 0.19.0)
//! trueno is now aprender-compute. This crate re-exports for backward compatibility.
pub use aprender_compute::*;
```

Same for `entrenar`, `realizar`, `batuta`, and all `trueno-*` sub-crates.
These shim crates are ~5 lines each, maintained indefinitely, never change.

### For existing users

| Current dependency | Migration |
|-------------------|-----------|
| `trueno = "0.18"` | Works forever (shim re-exports aprender-compute) |
| `aprender = "0.27"` | `aprender = "0.29"` (same crate, new version) |
| `entrenar = "0.7"` | `aprender-train = "0.29"` or keep `entrenar = "0.8"` (shim) |
| `realizar = "0.8"` | `aprender-serve = "0.29"` or keep `realizar = "0.9"` (shim) |
| `batuta = "0.7"` | `aprender-orchestrate = "0.29"` or keep `batuta = "0.8"` (shim) |

---

## Migration Plan

### Phase 1: Prepare (1 day)

1. Create `paiml/aprender` branch `monorepo-consolidation`
2. Add `crates/` directories for new workspace members
3. Set up `[workspace.package] version = "0.29.0"`
4. Set up `[workspace.dependencies]` for all shared deps (like Polars does)

### Phase 2: Move source (2 days)

For each repo (trueno, entrenar, realizar, batuta):

```bash
# Preserve git history with subtree merge
git subtree add --prefix=crates/aprender-compute \
  git@github.com:paiml/trueno.git main
```

Then:
- Rename `[package] name` in each moved Cargo.toml
- Update internal `use trueno::` → `use aprender_compute::`
- Update internal path deps to workspace-relative

### Phase 3: Wire workspace deps (1 day)

Replace all version-pinned cross-crate deps with workspace paths:

```toml
# Before (in entrenar/Cargo.toml):
trueno = { version = "0.17", features = ["parallel"] }
aprender = { version = "0.27" }

# After (in crates/aprender-train/Cargo.toml):
aprender-compute = { path = "../aprender-compute", features = ["parallel"] }
aprender = { path = "../aprender" }
```

### Phase 4: Publish & Shim (1 day)

1. Publish all `aprender-*` crates from the monorepo in topological order
2. Publish shim crates for old names (see Phase 4a below)
3. Verify `cargo install apr-cli` works from crates.io
4. Post-publish smoke test: `cargo install apr-cli --force` on clean machine

#### Phase 4a: Shim Crate Publishing

Each old crate name gets a final version that re-exports the new name:

```rust
// trueno 0.19.0/src/lib.rs — published to crates.io
//! `trueno` has moved to `aprender-compute`.
//! This crate re-exports `aprender-compute` for backward compatibility.
//! New code should depend on `aprender-compute` directly.
pub use aprender_compute::*;
```

```toml
# trueno 0.19.0/Cargo.toml
[package]
name = "trueno"
version = "0.19.0"
description = "DEPRECATED: Use aprender-compute instead. This crate re-exports aprender-compute."
repository = "https://github.com/paiml/aprender"
keywords = ["deprecated", "moved"]

[dependencies]
aprender-compute = "0.29"
```

Repeat for all 19+ old crate names (see Appendix A).
Shim crates are ~10 lines each. Publish once, never update again.

### Phase 5: Archive Old Repositories (1 day)

For each of the 19 merged repositories:

#### 5a. Update README with redirect

```markdown
# ⚠️ This repository has moved

**This project is now part of the [aprender monorepo](https://github.com/paiml/aprender).**

- New location: `paiml/aprender/crates/aprender-compute/` (was `paiml/trueno`)
- New crate name: `aprender-compute` (old name `trueno` still works via re-export)
- Issues: File at [paiml/aprender/issues](https://github.com/paiml/aprender/issues)

## For existing users

```toml
# This still works (re-export shim):
trueno = "0.19"

# Preferred (direct dependency):
aprender-compute = "0.29"
```
```

#### 5b. Archive repository

```bash
# Via GitHub API (or Settings → Danger Zone → Archive)
gh api -X PATCH repos/paiml/trueno -f archived=true
gh api -X PATCH repos/paiml/entrenar -f archived=true
gh api -X PATCH repos/paiml/realizar -f archived=true
gh api -X PATCH repos/paiml/Batuta -f archived=true
gh api -X PATCH repos/paiml/presentar -f archived=true
gh api -X PATCH repos/paiml/renacer -f archived=true
gh api -X PATCH repos/paiml/certeza -f archived=true
gh api -X PATCH repos/paiml/trueno-db -f archived=true
gh api -X PATCH repos/paiml/trueno-graph -f archived=true
gh api -X PATCH repos/paiml/trueno-rag -f archived=true
gh api -X PATCH repos/paiml/trueno-viz -f archived=true
gh api -X PATCH repos/paiml/trueno-zram -f archived=true
gh api -X PATCH repos/paiml/batuta-common -f archived=true
gh api -X PATCH repos/paiml/repartir -f archived=true
gh api -X PATCH repos/paiml/alimentar -f archived=true
gh api -X PATCH repos/paiml/simular -f archived=true
gh api -X PATCH repos/paiml/verificar -f archived=true
gh api -X PATCH repos/paiml/probar -f archived=true
gh api -X PATCH repos/paiml/provable-contracts -f archived=true
gh api -X PATCH repos/paiml/pacha -f archived=true
```

Archiving preserves: issues, PRs, stars, forks, git history, wiki.
Disables: push, new issues, new PRs. Read-only forever.

#### 5c. crates.io namespace reservation

Old crate names on crates.io remain owned by PAIML. The shim versions
(trueno 0.19, entrenar 0.8, etc.) ensure the names can't be squatted.
`cargo install` continues to work via re-export.

**crates.io ownership audit** — verify all old crate names list the
PAIML team as owner:

```bash
for crate in trueno trueno-gpu trueno-quant trueno-db trueno-viz \
             trueno-explain trueno-rag trueno-graph trueno-gemm-codegen \
             trueno-zram-core trueno-zram-adaptive trueno-cuda-edge \
             trueno-fft trueno-sparse trueno-solve trueno-rand \
             trueno-image trueno-tensor entrenar entrenar-common \
             entrenar-lora realizar batuta batuta-common repartir \
             presentar renacer certeza verificar probar \
             provable-contracts provable-contracts-macros; do
  echo -n "$crate: "
  cargo owner --list $crate 2>/dev/null | head -1
done
```

### Phase 6: Documentation Update (1 day)

#### 6a. Unified book

Merge book content from all repos into `aprender/book/`:

```
book/src/
├── introduction.md
├── getting-started/
│   └── installation.md          # cargo install apr-cli
├── compute/                     # was trueno book
│   ├── simd-backends.md
│   ├── gpu-compute.md
│   └── inference.md
├── training/                    # was entrenar docs
│   ├── training-loops.md
│   └── lora.md
├── serving/                     # was realizar docs
│   ├── inference-server.md
│   └── api-reference.md
├── orchestration/               # was batuta docs
│   ├── agents.md
│   └── rag-oracle.md
├── cli-reference/               # auto-generated from clap
│   ├── apr-run.md
│   ├── apr-serve.md
│   └── ...
└── appendix/
    ├── changelog.md             # unified changelog
    ├── migration-guide.md       # trueno → aprender-compute
    └── crate-rename-table.md
```

#### 6b. Auto-generated CLI reference

Add to CI/Makefile:

```makefile
docs-cli:
	@for cmd in run serve inspect debug validate diff tensors trace \
	            lint explain canary export import pull list rm convert \
	            compile merge quantize tui check gpu code; do \
	    echo "## apr $$cmd" > book/src/cli-reference/apr-$$cmd.md; \
	    echo '```' >> book/src/cli-reference/apr-$$cmd.md; \
	    cargo run -p apr-cli -- $$cmd --help >> book/src/cli-reference/apr-$$cmd.md 2>&1; \
	    echo '```' >> book/src/cli-reference/apr-$$cmd.md; \
	done
```

#### 6c. Update external references

- crates.io descriptions: all `aprender-*` crates link to monorepo
- docs.rs: ensure workspace docs build (`cargo doc --workspace`)
- GitHub topics: add "monorepo" tag to paiml/aprender
- README badges: update CI, coverage, crates.io links

### Phase 7: Daily workflow (ongoing)

```bash
# Daily apr-cli release (ONE command):
cargo publish -p apr-cli

# If a compute primitive changed too:
cargo publish -p aprender-compute && cargo publish -p apr-cli

# Workspace-wide test (catches ALL breakage):
cargo test --workspace

# Publish with topological ordering (when multiple crates changed):
cargo workspaces publish --from-git
```

---

## What This Fixes

| Problem | Before (5 repos) | After (1 repo) |
|---------|------------------|----------------|
| Version sync | Manual, 19 failures (#701) | Automatic (workspace) [4] |
| Daily apr-cli | 5-repo coordination | `cargo publish -p apr-cli` |
| Diamond deps | `trueno 0.17` vs `0.18` | Impossible (one version) [4] |
| `[patch.crates-io]` | Required, leaks to publish | Eliminated [5] |
| Circular deps | aprender↔trueno blocked | Workspace siblings [1] |
| CI coverage | 19 separate pipelines | 1 pipeline, 1 report [7] |
| New contributor setup | Clone 5+ repos | Clone 1 repo [1] |
| Cross-crate refactoring | 5+ PRs, coordinated merge | 1 PR [1] |
| Crate namespace | 4 prefixes (trueno/aprender/entrenar/realizar) | 1 prefix (aprender-*) |
| crates.io names | 32+ names, version sync hell | ~48 names, workspace-locked |
| Documentation | 5+ separate books | 1 unified book |
| Old repos | Active, diverging | Archived read-only, redirect READMEs |

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Large git repo (4752 files) | Certain | Low | Polars has 28 crates, Burn 33 — proven at scale |
| Compile time increase | Medium | Medium | `default-members` limits what builds by default; `cargo test -p apr-cli` for focused work |
| CI time increase | Medium | Medium | Use `cargo nextest` + `--partition` for parallel CI; cache `target/` |
| Migration breaks existing users | Low | High | Shim crates provide indefinite backward compat |
| Git history loss | Low | Medium | `git subtree` preserves full history; old repos archived read-only |
| Merge conflicts during migration | Medium | Low | Do it over a weekend freeze; migrate one repo at a time |

---

## Decision Matrix

| Option | Impact | Effort | Risk | Recommendation |
|--------|--------|--------|------|---------------|
| **A: Full monorepo** (this spec, 19 repos → 1) | **Critical** | **5-7 days** | **Low** | **RECOMMENDED — matches industry standard [1][2][3]** |
| B: Keep trueno separate | Medium | 2 days | Medium | Partial fix, version sync remains |
| C: Do nothing | — | 0 | **High** | 19 incidents → 30+ incidents |

---

## Success Criteria

1. `cargo test --workspace` passes (all 5500+ files compile together)
2. `cargo publish -p apr-cli` succeeds without `[patch.crates-io]`
3. `cargo install apr-cli` works from a clean machine
4. Old crate names (`trueno`, `entrenar`, `realizar`, `batuta`) still resolve via shims
5. Daily apr-cli releases take < 5 minutes (publish + verify)
6. Zero cross-crate version mismatch incidents for 90 days post-migration

---

## Falsification Conditions

**Contract**: `contracts/cgp/cgp-monorepo-consolidation-v1.yaml`

If ANY of these become true, the migration hypothesis is wrong:

| ID | Condition | Threshold | Mitigation |
|----|-----------|-----------|------------|
| FALSIFY-MONO-001 | Incremental compile time regression | > 3× baseline (> 15s for 1-file change) | `default-members`, dep graph pruning |
| FALSIFY-MONO-002 | CI gate time exceeds budget | > 10 min wall-clock for 1-file PR | `cargo nextest --partition`, sccache |
| FALSIFY-MONO-003 | Merge conflict rate increases | > 2 conflicts/week (baseline ~0) | CODEOWNERS, directory ownership [2] |
| FALSIFY-MONO-004 | Daily publish exceeds time budget | > 5 min for `make publish CRATE=apr-cli` | Topological publish ordering |
| FALSIFY-MONO-005 | Broken publishes continue | > 2 incidents in 90 days (baseline 19/5mo) | Workspace eliminates version skew [4] |
| FALSIFY-MONO-006 | Clone time exceeds threshold | > 30s for `git clone --depth 1` | .gitattributes LFS, shallow clone |
| FALSIFY-MONO-007 | Git history lost during migration | `git log --follow` doesn't show pre-merge commits | Verify `git subtree` preservation |
| FALSIFY-MONO-008 | Shim crates fail re-export | `trueno = "0.19"` produces type mismatches | Integration test shim crates in CI |
| FALSIFY-MONO-009 | Workspace version bump breaks downstream | Patch bump causes API incompatibility | Polars pattern: shared version [1] |
| FALSIFY-MONO-010 | Crate name not in Appendix A registry | Any `[package] name` not listed in spec | CI script validates against registry |
| FALSIFY-MONO-011 | Non-apr-cli binary found in workspace | Any `[[bin]]` section outside apr-cli | CI grep for `[[bin]]` in Cargo.toml files |
| FALSIFY-MONO-012 | apr subcommand missing contract | Any Commands enum variant without contract YAML | CI cross-checks enum vs contracts/ |

---

## Infrastructure Requirements (paiml/infra updates)

The following infra specs must be updated BEFORE or DURING migration:

### INFRA-CI-MONO: Workspace-aware CI pipeline

`unified-ci-pipeline.md` currently assumes single-crate repos. Changes:
- `cargo test --workspace` replaces `cargo test`
- `cargo clippy --workspace` replaces `cargo clippy`
- sccache warmup for ~48 crate build graph
- CI time budget: 30-90s → 3-5 min for full workspace
- `cargo nextest --partition` for parallel test execution

### INFRA-PUBLISH-MONO: Topological publish ordering

`release-system.md` must support workspace publish ordering:
- Cannot `cargo publish -p apr-cli` until all deps are published
- Need topological sort: provable-contracts → compute → aprender → train/serve → apr-cli
- Tool: `cargo-workspaces publish` or custom `xtask publish`
- Trusted Publishing OIDC must work for ~48 crate names

### INFRA-CLEAN-ROOM-MONO: Workspace resource budget

`clean-room-spec.md` container must handle full workspace:
- Disk: 2-3× current for ~48 crate build graph
- Memory: monitor for OOM on parallel compilation
- `cargo install apr-cli` post-publish smoke test unchanged

### INFRA-ARCHIVE: Old repo archival

19 repos archived as read-only (GitHub Settings → Archive):
- README updated: "This repo has moved to paiml/aprender/crates/..."
- No deletion — preserve issues, PRs, stars
- Branch protection removed (read-only)

---

## Appendix A: Definitive Crate Name Registry (ENFORCED BY CONTRACT)

**This table is the single source of truth for all crate names in the monorepo.**
Any crate not listed here MUST NOT be added without updating this spec.
Contract: `cgp-monorepo-consolidation-v1.yaml` FALSIFY-MONO-010.

### A.1 Core ML (unchanged names)

| # | Crate Name | Workspace Path | Source Repo | Description |
|---|-----------|---------------|-------------|-------------|
| 1 | `aprender` | `crates/aprender/` | paiml/aprender | ML format (.apr), tokenizers, model ops |
| 2 | `apr-cli` | `crates/apr-cli/` | paiml/aprender | `apr` binary — user-facing CLI |
| 3 | `aprender-shell` | `crates/aprender-shell/` | paiml/aprender | Interactive REPL |
| 4 | `aprender-tsp` | `crates/aprender-tsp/` | paiml/aprender | TSP solver examples |
| 5 | `aprender-monte-carlo` | `crates/aprender-monte-carlo/` | paiml/aprender | Monte Carlo simulations |

### A.2 Compute Primitives (was trueno)

| # | Crate Name | Workspace Path | Old Name | Shim Version |
|---|-----------|---------------|----------|-------------|
| 6 | `aprender-compute` | `crates/aprender-compute/` | `trueno` | trueno 0.19 |
| 7 | `aprender-gpu` | `crates/aprender-gpu/` | `trueno-gpu` | trueno-gpu 0.5 |
| 8 | `aprender-quant` | `crates/aprender-quant/` | `trueno-quant` | trueno-quant 0.2 |
| 9 | `aprender-gemm-codegen` | `crates/aprender-gemm-codegen/` | `trueno-gemm-codegen` | trueno-gemm-codegen 0.2 |
| 10 | `aprender-fft` | `crates/aprender-fft/` | `trueno-fft` | trueno-fft 0.2 |
| 11 | `aprender-sparse` | `crates/aprender-sparse/` | `trueno-sparse` | trueno-sparse 0.2 |
| 12 | `aprender-solve` | `crates/aprender-solve/` | `trueno-solve` | trueno-solve 0.2 |
| 13 | `aprender-rand` | `crates/aprender-rand/` | `trueno-rand` | trueno-rand 0.2 |
| 14 | `aprender-image` | `crates/aprender-image/` | `trueno-image` | trueno-image 0.2 |
| 15 | `aprender-tensor` | `crates/aprender-tensor/` | `trueno-tensor` | trueno-tensor 0.2 |
| 16 | `aprender-cuda-edge` | `crates/aprender-cuda-edge/` | `trueno-cuda-edge` | trueno-cuda-edge 0.2 |
| 17 | `aprender-ptx-debug` | `crates/aprender-ptx-debug/` | `trueno-ptx-debug` | No (internal only) |
| 18 | `aprender-explain` | `crates/aprender-explain/` | `trueno-explain` | trueno-explain 0.3 |
| 19 | `aprender-cbtop` | `crates/aprender-cbtop/` | `cbtop` | cbtop 0.2 |
| 20 | `aprender-cgp` | `crates/aprender-cgp/` | `cgp` | No (internal only) |

### A.3 Data & Storage

| # | Crate Name | Workspace Path | Old Name | Shim Version |
|---|-----------|---------------|----------|-------------|
| 21 | `aprender-db` | `crates/aprender-db/` | `trueno-db` | trueno-db 0.4 |
| 22 | `aprender-graph` | `crates/aprender-graph/` | `trueno-graph` | trueno-graph 0.2 |
| 23 | `aprender-rag` | `crates/aprender-rag/` | `trueno-rag` | trueno-rag 0.3 |
| 24 | `aprender-rag-cli` | `crates/aprender-rag-cli/` | `trueno-rag-cli` | trueno-rag-cli 0.2 |
| 25 | `aprender-data` | `crates/aprender-data/` | `alimentar` | alimentar 0.3 |
| 26 | `aprender-registry` | `crates/aprender-registry/` | `pacha` | pacha 0.3 |

### A.4 Training (was entrenar)

| # | Crate Name | Workspace Path | Old Name | Shim Version |
|---|-----------|---------------|----------|-------------|
| 27 | `aprender-train` | `crates/aprender-train/` | `entrenar` | entrenar 0.8 |
| 28 | `aprender-train-common` | `crates/aprender-train-common/` | `entrenar-common` | entrenar-common 0.2 |
| 29 | `aprender-train-lora` | `crates/aprender-train-lora/` | `entrenar-lora` | entrenar-lora 0.4 |
| 30 | `aprender-train-distill` | `crates/aprender-train-distill/` | `entrenar-distill` | entrenar-distill 0.2 |
| 31 | `aprender-train-inspect` | `crates/aprender-train-inspect/` | `entrenar-inspect` | entrenar-inspect 0.2 |
| 32 | `aprender-train-shell` | `crates/aprender-train-shell/` | `entrenar-shell` | entrenar-shell 0.2 |

### A.5 Serving (was realizar)

| # | Crate Name | Workspace Path | Old Name | Shim Version |
|---|-----------|---------------|----------|-------------|
| 33 | `aprender-serve` | `crates/aprender-serve/` | `realizar` | realizar 0.9 |

### A.6 Orchestration (was batuta)

| # | Crate Name | Workspace Path | Old Name | Shim Version |
|---|-----------|---------------|----------|-------------|
| 34 | `aprender-orchestrate` | `crates/aprender-orchestrate/` | `batuta` | batuta 0.8 |

### A.7 Visualization & TUI (was presentar + trueno-viz)

| # | Crate Name | Workspace Path | Old Name | Shim Version |
|---|-----------|---------------|----------|-------------|
| 35 | `aprender-viz` | `crates/aprender-viz/` | `trueno-viz` | trueno-viz 0.3 |
| 36 | `aprender-present-core` | `crates/aprender-present-core/` | `presentar-core` | presentar-core 0.4 |
| 37 | `aprender-present-terminal` | `crates/aprender-present-terminal/` | `presentar-terminal` | presentar-terminal 0.4 |
| 38 | `aprender-present-widgets` | `crates/aprender-present-widgets/` | `presentar-widgets` | presentar-widgets 0.4 |
| 39 | `aprender-present-layout` | `crates/aprender-present-layout/` | `presentar-layout` | presentar-layout 0.4 |
| 40 | `aprender-present-yaml` | `crates/aprender-present-yaml/` | `presentar-yaml` | presentar-yaml 0.4 |
| 41 | `aprender-present-cli` | `crates/aprender-present-cli/` | `presentar-cli` | presentar-cli 0.4 |
| 42 | `aprender-present` | `crates/aprender-present/` | `presentar` | presentar 0.4 |

### A.8 Profiling & Quality

| # | Crate Name | Workspace Path | Old Name | Shim Version |
|---|-----------|---------------|----------|-------------|
| 43 | `aprender-profile` | `crates/aprender-profile/` | `renacer` | renacer 0.11 |
| 44 | `aprender-profile-core` | `crates/aprender-profile-core/` | `renacer-core` | renacer-core 0.2 |
| 45 | `aprender-verify` | `crates/aprender-verify/` | `certeza` | certeza 0.2 |
| 46 | `aprender-verify-ml` | `crates/aprender-verify-ml/` | `verificar` | verificar 0.6 |
| 47 | `aprender-simulate` | `crates/aprender-simulate/` | `simular` | simular 0.4 |
| 48 | `aprender-distribute` | `crates/aprender-distribute/` | `repartir` | repartir 2.1 |

### A.9 Testing Framework (was probar)

| # | Crate Name | Workspace Path | Old Name | Shim Version |
|---|-----------|---------------|----------|-------------|
| 49 | `aprender-test` | `crates/aprender-test/` | `probar` | probar 1.1 |
| 50 | `aprender-test-derive` | `crates/aprender-test-derive/` | `probar-derive` | probar-derive 1.1 |
| 51 | `aprender-test-cli` | `crates/aprender-test-cli/` | `probar-cli` | probar-cli 1.1 |
| 52 | `aprender-test-js-gen` | `crates/aprender-test-js-gen/` | `probar-js-gen` | probar-js-gen 1.1 |

### A.10 Contracts & Build Infrastructure (was provable-contracts)

| # | Crate Name | Workspace Path | Old Name | Shim Version |
|---|-----------|---------------|----------|-------------|
| 53 | `aprender-contracts` | `crates/aprender-contracts/` | `provable-contracts` | provable-contracts 0.3 |
| 54 | `aprender-contracts-macros` | `crates/aprender-contracts-macros/` | `provable-contracts-macros` | provable-contracts-macros 0.3 |
| 55 | `aprender-contracts-cli` | `crates/aprender-contracts-cli/` | `provable-contracts-cli` | provable-contracts-cli 0.3 |

### A.11 Compressed Memory (was trueno-zram)

| # | Crate Name | Workspace Path | Old Name | Shim Version |
|---|-----------|---------------|----------|-------------|
| 56 | `aprender-zram` | `crates/aprender-zram/` | `trueno-zram-core` | trueno-zram-core 0.4 |
| 57 | `aprender-zram-adaptive` | `crates/aprender-zram-adaptive/` | `trueno-zram-adaptive` | trueno-zram-adaptive 0.4 |
| 58 | `aprender-zram-generator` | `crates/aprender-zram-generator/` | `trueno-zram-generator` | trueno-zram-generator 0.4 |
| 59 | `aprender-zram-cli` | `crates/aprender-zram-cli/` | `trueno-zram-cli` | trueno-zram-cli 0.4 |
| 60 | `aprender-ublk` | `crates/aprender-ublk/` | `trueno-ublk` | trueno-ublk 0.4 |

### A.12 Benchmarks (internal, not published)

| # | Crate Name | Workspace Path | Old Name | Published? |
|---|-----------|---------------|----------|-----------|
| 61 | `aprender-bench-tokenizer` | `crates/aprender-bench-tokenizer/` | (same) | No |
| 62 | `aprender-bench-compute` | `crates/aprender-bench-compute/` | (same) | No |
| 63 | `aprender-train-bench` | `crates/aprender-train-bench/` | `entrenar-bench` | No |

**Total: 63 workspace crates (49 published + 14 internal)**

### A.13 Shim Crate Count

- **Published shim crates needed**: ~45 (one per renamed crate)
- **Each shim**: ~10 lines (`pub use new_name::*;`)
- **Published once, never updated again**
- **Purpose**: backward compatibility + namespace reservation

### Appendix B: Kept Separate (NOT merged)

| Crate | Reason |
|-------|--------|
| pmat / paiml-mcp-agent-toolkit | Separate product, own release cycle, 3830 .rs files |
| manzana | Platform-specific (Apple only) |
| whisper.apr | Application built ON the stack, not part of it |
| forjar | Standalone IaC tool, zero stack deps (1180 files) |
| ruchy | Separate language/runtime project |
| apr-cookbook | Becomes `aprender/cookbook/` (content, not a crate) |

### Appendix C: Polars Reference Architecture

```
pola-rs/polars/Cargo.toml:
  [workspace.package]
  version = "0.53.0"           ← ALL 28 crates share this version
  
  [workspace.dependencies]     ← shared dep versions, DRY
  arrow = "53"
  serde = { version = "1", features = ["derive"] }
  
polars-core/Cargo.toml:
  [package]
  name = "polars-core"
  version.workspace = true     ← inherits from workspace
  
  [dependencies]
  polars-arrow = { workspace = true }   ← resolves to local path
```

This is the target state for `paiml/aprender`.
