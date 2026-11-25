---
tools: [pmat, pv, cargo, gh, bashrs]
---

# CLAUDE.md

Rationale, command blocks, catalogs: `docs/agent-rules-history.md#second-trim-moved-blocks` (and the anchors it names). Global rules (git workflow, pmat query, coverage via `pmat query --coverage-gaps`, trueno in-tree) load from `~/.claude/CLAUDE.md`; not restated.

## Project Overview

- Pure-Rust ML monorepo; core `crates/aprender-core/` ([lib] name = "aprender"); `cargo install aprender` → `apr`.
- **Derive numbers, never quote them (#2331):** the command wins (history #derive-the-numbers).
- `crates/aprender-core/tests/readme_contract.rs` fails on a nonexistent repo path cited here (FALSIFY-DOCS-CLAUDE-001).

## Git Workflow

- **`main` is protected:** required `ci / gate` + `gate` + `workspace-test`; branch → PR, no direct push.
- PR body carries `Closes #N` or `no-issue: <reason>`; squash-arm `gh pr merge --squash --auto --delete-branch` at once.

## Intake and Pull Signal (RAX-INTERIM, operator, 2026-11-04)

- Workers and seats never run `gh issue create`; one line in `docs/findings/<session>.jsonl` (`scripts/findings_ledger.sh add`, schema `contracts/findings-ledger-v1.yaml`) or the cop inbox. Only cops mint tickets.
- Over cap (PRs > 15 or issues > 100): only fixes, expedite, triage folds.

## Autonomous Operation Mode (DEFAULT)

- Ship PRs, do not ask permission for routine work; after landing/arming emit ONE line and continue. Stop only when P0/P1 is landed or blocked, or 5+ PRs stick on one flake. `/loop`: no polling.
- **Check in BEFORE:** compute > 1hr off lambda-vector; `git push --force`; `gh release delete`; dropping branches/tags on main; `cargo yank`; editing `.github/workflows/*.yml`; architectural pivots; contradicting an explicit user instruction.
- Publish is unattended: operator, 2026-09-20, verbatim: *"tell agent to auto-publish and never wait for me"*. The pre-publish gates ARE the authorization (APR-RELEASE-001 §T-4).

## Build Commands

- Run `cargo test -p aprender-core --lib`, `cargo check --workspace`, `cargo fmt --all -- --check`, `cargo clippy -p <crate> --lib -- -D warnings` (bare `cargo clippy` lints only the empty root facade).
- CI `workspace-test` is `cargo nextest run --profile ci --workspace --lib` plus excludes: read history before reproducing it. `make tier1|tier2|tier3|tier4`.

## Debugging: Use apr Tools First (MANDATORY)

- **Before reading code or grepping, use the apr toolchain** (GH-202). Step 0, ALWAYS: `. scripts/apr_bin.sh || exit 1` exports `$APR`; never a bare `apr` or hardcoded path (#pin-the-binary).
- Then `"$APR" qa`, `tensors`, `diff`, `validate --quality`, `lint`; only now read code. "format not supported" for GGUF/APR/SafeTensors is a BUG. No `realizar` binary: `"$APR" run model.gguf --prompt "Hi" --trace`.

## Architecture

- Flat `crates/aprender-*`; `apr` via root facade `src/bin/apr.rs`. Key: `crates/aprender-core/src/lib.rs`, `crates/aprender-core/src/format/`, `contracts/`, `docs/BEATS.md`.
- **ALL inference/serving MUST use `realizar`; `aprender` is TRAINING ONLY:** serving/HTTP/KV cache FORBIDDEN in aprender; `.apr` R/W aprender primary. GPU needs `--features cuda`. "apr beats Ollama 1.371x" is **withdrawn**; measured: `docs/BEATS.md`.

## LAYOUT-001/002: Tensor Layout Safety

- **GGUF/APR are ROW-MAJOR (bug seen 100+ times).** FORBIDDEN: `matmul_q4k_f32_colmajor`, `matmul_q6k_f32_colmajor` (and `_dispatch`). REQUIRED: `fused_q4k_parallel_matvec`, `fused_q6k_parallel_matvec`.
- SOURCE OF TRUTH: `contracts/tensor-layout-v1.yaml`, `crates/aprender-core/src/format/layout_contract.rs`.

## Publishing Safety (CB-510 Lesson)

- **`.gitignore` and `Cargo.toml` exclude patterns must be root-anchored** (`/models/`, not `models/`). Run `bash scripts/check_include_files.sh`; re-run after any exclude change. Claim fence (ONT-4c, `contracts/claude-md.yaml`):

```bash
setsid --wait bash scripts/check_package_includes.sh  # as CI runs it; scans src/ ONLY, against `cargo package -p aprender --list`
```

## Shell, Verification, Testing, Linting

- **Shell:** `bashrs lint scripts/*.sh`, `bashrs gate --strict .`; `set -euo pipefail`, quoted vars; sourced libs option-neutral (`scripts/check_sourced_libs_option_neutral.sh`).
- **Verify (MANDATORY):** never read `$?` through a pipe (`rc=${PIPESTATUS[0]}`); prove the mechanism engaged, never label by intent; pin the binary; a guard must scan the surface where the DECISION is made, and extending its scope needs re-mutation; one failing input is an anecdote; guard regexes ship a case table; ask what shadows an artifact (`type -aP`). History #verification-discipline.
- **Testing:** 60/30/10 unit/property/integration; enforced floor `COV_FLOOR` 89%. No root `tests/` dir; a new `tests/*.rs` is dark until `ci/explicit-test-commands.d/NNN-<slug>.cmd` names it (never renumber).
- **Mutation testing ONLY in `.github/workflows/mutants-nightly.yml`** (operator, 2026-11-10 16:50Z, verbatim: "NO build can do mutation outside of nightly; the end. FOREVER"). No PR, push, queue, train or seat runs `cargo mutants`.
- **Lint:** `unsafe_code = "forbid"`; `unwrap()` banned (`.clippy.toml`; use `expect()` or `ok_or_else`). PMAT: `pmat comply check --failures-only`, zero TODO/FIXME/HACK. Release by hand (`scripts/bump-version.sh`).

## Contract Validation: DOGFOOD `pv`, NEVER bash

- **`pv` is THE contract CLI** for `contracts/` (`crates/aprender-contracts-cli/`); a bash/yq/python re-implementation is **muda**. If `pv validate` rejects: fit `KernelContract`, or extend `crates/aprender-contracts/src/schema/`, or use a purpose-built `apr` subcommand.
- **Coverage without contracts is REJECTED:** per function tested, add `#[contract]` and strengthen the YAML falsification conditions.

**Never:** work around `pv` with a shell script. The in-tree tool is the source of truth.
