---
tools: [pmat, pv, cargo, gh, bashrs]
---

# CLAUDE.md

Narratives, catalogs: `docs/agent-rules-history.md` (anchors named below).

## Project Overview

Aprender: pure-Rust ML monorepo; the workspace, not this file, is the source of truth. `cargo install aprender` → `apr`.
Core library `crates/aprender-core/` ([lib] name = "aprender"). Spec: `docs/specifications/aprender-monorepo-consolidation.md`.
**Derive numbers, never quote them (#2331):** if a number disagrees with its command, the command wins (#derive-the-numbers).
Crates: `cargo metadata --no-deps --format-version 1 | jq '.packages | length'` (`ls crates/*/` is NOT the count); `apr` commands: `apr --help`, registry `contracts/apr-cli-commands-v1.yaml` §`commands` (never `grep -c`); contracts: `find contracts -name '*.yaml' | wc -l`; tests: `Summary` of CI `workspace-test`; version: `git tag --sort=-creatordate | head -1`.
`crates/aprender-core/tests/readme_contract.rs` fails on a nonexistent repo path cited here (FALSIFY-DOCS-CLAUDE-001).

## Git Workflow (Branch Protection)

**`main` is protected.** Required: `ci / gate` + `gate` + `workspace-test` (protection and org ruleset each name one). No direct pushes: `git checkout -b <name>` → push → `gh pr create`; CI must pass.

## Intake and Pull Signal (RAX-INTERIM, operator, 2026-11-04)

- **Intake:** workers and seats never run `gh issue create`. Record a finding as one line in
  `docs/findings/<session>.jsonl` or in the cop inbox. Only cops mint tickets, under an epic with a budget.
- **Pull signal:** when the repo is over its cap (open PRs > 15 or open issues > 100), its cop opens no new
  feature PRs and spawns no new feature workers there. Only fixes, expedite and triage folds proceed.
- **Closes:** every new PR body carries `Closes #N` or `no-issue: <reason>`. Cops report created vs closed (24 h) per repo.

## Autonomous Operation Mode (DEFAULT)

**Operate autonomously. Ship PRs, don't ask permission for routine work:** diagnostics, build/test/lint, branches, commits, PRs with `gh pr merge --squash --auto --delete-branch`, `gh run rerun --failed`, `gh pr update-branch`, spec amendments, evidence in `evidence/section-NN-*/findings.json`, memory updates, the next item after a merge.
Record a defect as one line in `docs/findings/<session>.jsonl` (schema `contracts/findings-ledger-v1.yaml`, `scripts/findings_ledger.sh add`) or the cop inbox, not a new spec section.
**Check in BEFORE:** compute > 1hr off lambda-vector (`feedback_compute_pre_authorized.md`); `git push --force`, `gh release delete`, dropping branches/tags on main, `cargo yank`; editing `.github/workflows/*.yml`; architectural pivots; anything contradicting an explicit user instruction.
~~Crates.io publish cascade (always ask before `make publish`)~~ **WITHDRAWN** — T-4 is unattended (APR-RELEASE-001 rev. 2026-09-17 §T-4; operator, 2026-09-20, verbatim: *"tell agent to auto-publish and never wait for me"*). The pre-publish gates ARE the authorization: dogfood GO on the release commit, release assets verified, clean-room green on the tag, publish preflight R1–R6, and the `--check` dry-run (a receipt, not a stop).
After landing/arming a PR emit ONE line and continue; do NOT ask "should I continue?". Stop only when the P0/P1 queue is landed or externally blocked, a defect needs an architectural decision, or 5+ PRs are stuck on one flake. `/loop`: no polling.

## Build Commands

```text
cargo test -p aprender-core --lib; cargo check --workspace; cargo fmt --all -- --check
cargo clippy -p <crate> --lib -- -D warnings   # bare `cargo clippy` lints only the empty ROOT FACADE
# CI `workspace-test` (.github/workflows/ci.yml): reproduce THIS, not `cargo test --workspace --lib`:
cargo nextest run --profile ci --workspace --lib \
    --exclude aprender-gpu --exclude aprender-cuda-edge --exclude aprender-compute
cargo test -p aprender-compute --lib   # own CI step
make tier1 | tier2 | tier3 | tier4     # fast | pre-commit | pre-push | CI; `make coverage` (floor COV_FLOOR in Makefile)
```

## Debugging: Use apr Tools First (MANDATORY)

**STOP. Before reading code or grepping, USE THE APR DIAGNOSTIC TOOLCHAIN** (GH-202).
**Step 0 — pin the binary, ALWAYS.** Never a bare `apr`, never a hardcoded absolute path (gitignored `config.toml` under `.cargo/` redirects target-dir; #pin-the-binary):

```text
. scripts/apr_bin.sh || exit 1        # exports $APR, proves it was built from HEAD
"$APR" qa model.apr                   # Step 1: ALWAYS start here
"$APR" tensors model.apr | head -20   # Step 2;  Step 3: "$APR" diff model.apr reference.gguf
"$APR" validate model.apr --quality; "$APR" lint model.apr   # Step 4;  Step 5: ONLY NOW read code
```

Also `trace`, `profile`, `inspect`, `debug`. "format not supported" for GGUF/APR/SafeTensors is a BUG. No `realizar` binary: trace with `"$APR" run model.gguf --prompt "Hi" --trace` (`crates/aprender-serve/src/inference_trace/`; flags #realizar-tracing).

## Architecture

Flat `crates/aprender-*` (`aprender-core`, `aprender-compute` = "trueno", `aprender-serve` = "realizar", `aprender-train`, `apr-cli`; `apr` via root facade `src/bin/apr.rs`). Key: `crates/aprender-core/src/lib.rs`, `crates/aprender-core/src/format/`, `contracts/`, `docs/BEATS.md`.

## CRITICAL: Realizar-First Architecture

**ALL inference/serving MUST use `realizar`. The `aprender` crate is for TRAINING ONLY.** Serving / HTTP / KV cache: aprender **FORBIDDEN**; GGUF/SafeTensors loading and CUDA/GPU inference: aprender never; `.apr` R/W: aprender primary, realizar read-only.
`inference` is a default feature of `crates/apr-cli/Cargo.toml`; GPU needs `--features cuda`. Profile before optimizing. Perf figures are targets; measured: `docs/BEATS.md`, floor `contracts/beat-ollama-decode-throughput-speed-v1.yaml`. "apr beats Ollama 1.371x" is **withdrawn**. #ffn-fusion

## LAYOUT-001/002: Tensor Layout Safety

**CRITICAL: GGUF/APR use ROW-MAJOR layout. This bug has occurred 100+ times.** GGUF column-major data is transposed at import.
- **FORBIDDEN for GGUF/APR data:** `trueno::backends::q4k::matmul_q4k_f32_colmajor`, `trueno::backends::q6k::matmul_q6k_f32_colmajor` (and `_dispatch`). **REQUIRED:** `crate::quantize::fused_q4k_parallel_matvec`, `crate::quantize::fused_q6k_parallel_matvec`.
- `contracts/tensor-layout-v1.yaml` (**SOURCE OF TRUTH**), `crates/aprender-core/src/format/layout_contract.rs`, `crates/aprender-core/src/format/converter/write.rs`, `crates/aprender-core/src/format/converter/mod.rs`.
- Deleted (Refs #224, #1977): `Qwen2Model` generate()/forward(); inference goes through `realizar`.

## Publishing Safety (CB-510 Lesson)

**CRITICAL: `.gitignore` and `Cargo.toml` exclude patterns must be root-anchored** (unanchored **models/** hides **src/models/**; use **/models/**).
`bash scripts/check_include_files.sh` (pre-publish, tier3). After new `include!()` files: `git ls-files --others --exclude-standard crates/`; `git check-ignore -v <new file>` (exit 1 == not ignored). After any `.gitignore`/`Cargo.toml` exclude change re-run both scripts. The merge-path step stays a `bash` claim fence (ONT-4c, `contracts/claude-md.yaml`):

```bash
setsid --wait bash scripts/check_package_includes.sh  # as CI runs it; scans src/ ONLY, against `cargo package -p aprender --list`
```

It is vacuous today (zero `include!()` in root `src/`): its green is not evidence; real coverage is `check_include_files.sh`.

## Shell Scripts: Use bashrs (NOT shellcheck)

`bashrs lint scripts/*.sh`, `bashrs make lint Makefile`, `bashrs gate --strict .`. Required: `set -euo pipefail`, no `ls` for iteration, quoted variables, explicit error handling.
A SOURCED library must be option-neutral (fail by return status, `. scripts/lib.sh || exit 1`); enforced by `scripts/check_sourced_libs_option_neutral.sh` (Why: #sourced-libs).

## Verification Discipline (MANDATORY — read before reporting any result)

Incidents: #verification-discipline.
1. **Never read `$?` through a pipe** (last command's status): `cmd > out.log 2>&1; rc=$?` or `rc=${PIPESTATUS[0]}`. When a result looks good, check how it was measured.
2. **Never label a run by intent — prove the mechanism engaged** (trace line, version+SHA, behaviour delta).
3. **Pin the binary.** Never a bare `apr` or hardcoded path (Step 0).
4. **Extending a guard's SCOPE requires re-mutating in the new scope.**
5. **A guard that does not scan the surface where the DECISION is made is theater.**
6. **One failing input is an anecdote.** Vary it before naming a cause or blocking a release.
7. **Guard regexes ship a case table** (must-match / must-not-match).
8. **A shadowed artifact is worse than a missing one**: when a fix has no effect, ask what else claims the name (`type -aP <cmd>`).

## Testing

60% unit, 30% property, 10% integration. Coverage target >=95%, enforced floor 89% (`COV_FLOOR`); derive the measured value from coverage-nightly (#coverage-measurements).
`cargo test -p <crate> --test <target>` (NO root `tests/` dir: `integration.rs` targets are per-crate). CI runs `--lib` plus `ci/explicit-test-commands.d/` (`NNN-<slug>.cmd`, sorted): a new `tests/*.rs` is dark until a fragment names it; take a free ordinal between two others, never renumber.
Mutation testing runs ONLY in the scheduled `.github/workflows/mutants-nightly.yml` (infra#3828; operator, 2026-11-10 16:50Z, verbatim: "NO build can do mutation outside of nightly; the end. FOREVER"). No PR, push, merge queue, train or local seat runs `cargo mutants`, and no PR or merge waits on a mutation score. `make mutants*` refuses.

## Linting

`unsafe_code = "forbid"`. `unwrap()` banned (`.clippy.toml`; use `expect()` or `ok_or_else(|| ...)?`, Issue #41). Both toolchain ends gated (#2370): FLOOR `scripts/check_msrv.sh` (on demand); CEILING `scripts/check_clippy_current_stable.sh` / `make lint-current` (daily). Green ceiling = clean on the pin and current stable only.
PMAT: re-derive scores (`pmat rust-project-score`, `pmat tdg . --include-components`); config `.pmat-gates.toml`; `pmat analyze satd` (zero TODO/FIXME/HACK); pre-commit `pmat comply check --failures-only`. Workflows: `.github/workflows/` (#ci-cd); release cut by hand (`scripts/bump-version.sh`, tag, `gh release create`; `docs/specifications/PP-LLAMA-001-MASTER.md` §7.2).

## Contract Validation: DOGFOOD `pv`, NEVER bash

In-tree (APR-MONO Phase 2b): `crates/aprender-contracts/`, `crates/aprender-contracts-macros/`, `crates/aprender-contracts-cli/` (`pv`). **`pv` is THE contract CLI** (validate, lint, score, scaffold, diff, audit, proofs, falsification) for `contracts/`; a bash/yq/python re-implementation is **muda** and rejected.
If `pv validate` rejects a contract: (1) restructure to the existing schema (`KernelContract`: `equations`, `proof_obligations`, `falsification_tests`); (2) extend `crates/aprender-contracts/src/schema/` (own PMAT ticket); (3) if not a provable contract, use another schema/directory and a purpose-built `apr` subcommand, not `contracts/`.

**Never:** work around `pv` with a shell script. The in-tree tool is the source of truth.

## CRITICAL: Code Search Policy

**NEVER use grep/glob for code search. ALWAYS use `pmat query`** (`--limit 10 --include-source`; #pmat-query; `pmat query "init" --rebuild-index --limit 1` builds the index once per project). Full tree: author-local `~/.claude/CLAUDE.md`.
**Use `pmat query --coverage-gaps --exclude-tests` to find untested code. NEVER parse coverage JSON manually.** Prerequisite: `cargo llvm-cov test --lib --no-report`.
**Coverage work MUST co-evolve with contracts:** per function tested, add `#[contract]` if missing, strengthen falsification conditions in the contract YAML, eliminate placeholder preconditions; then `make coverage` and `pmat comply check`.
**RULE: Coverage without contracts is REJECTED. Both must improve together.** (monorepo spec Rule 7.) Stack docs: `batuta oracle --rag "<question>"`. Stale SSC snapshot: `docs/ssc-training-status-2026-03-22.md`.
