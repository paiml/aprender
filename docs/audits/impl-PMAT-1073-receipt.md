---
status: partial
ticket: PMAT-1073
issue: 3002
kind: code
branch: agent/R-0b
model: claude-fable-5-1
tokens_used: ~300k (this session)
wall_clock_s: ~7200
turns: ~60
---
# impl receipt — PMAT-1073 / R-0b: backend resolution reads the registry

**Row R-0b (#3002, PP-066 claim 1).** Base: origin/main + R-0a (#3004) + L0-1a (#3026), merged locally
(R-0a is not on main yet — this PR is stacked and cannot arm until #3004 merges). `status: partial`:
the CLI resolution, the case table, the guard and the contract are done and verified; the last two
`cfg!(feature=…)` backend reads (`dispatch.rs:173`, `lib.rs:230`) are not yet converted — see Gaps.

## What lands (verified)
- `crates/apr-cli/src/registry.rs` (declared from the hook-clean `commands_enum.rs` via `#[path]`):
  `Request::wanted()` classifies `--gpu`/`--no-gpu`/`--backend <kind>`/`--gpu-layers`/default;
  `resolve(_in)` turns it into a `Resolved` or a refusal over `trueno::registry::BackendRegistry`.
  A forced accelerator that is not Ready **never** resolves to cpu: not compiled ⇒
  `CliError::FeatureDisabled` (exit 9), compiled but not Ready here ⇒ the new
  `CliError::BackendUnavailable` (exit 14). Only the default (no flag) may fall to cpu, with the
  registry's reason (REG-8). `selected_line()` joins the parity text (REG-15) to the registry reason.
- Reroutes to the registry, never `cfg!`: `accel.rs` (`build_has_accelerator`, `ensure_available`),
  `serve/mod.rs` (the accelerator gate + `list_devices` + `cli_build_features`),
  `bench.rs` (`compute_class`, provenance features), `handler_gpu_completion.rs`, `finetune.rs`.
- `scripts/check_backend_registry.sh` (`--static` scans `crates/apr-cli/src` for
  `cfg!(feature="cuda"|"wgpu")` backend reads outside `registry.rs`; `--self-test` 3 rows both
  polarities), wired in ci.yml `guard-runner-labels` as its case table.
- `contracts/apr-backend-registry-v1.yaml`: REG-OB-004 + REG-F-003 (resolution + the static guard);
  `pv validate` through the pin: valid.

## RED test + mutations (measured, this tree)
- `crates/apr-cli/tests/backend_refusal_case_table.rs` — 5 rows over R-0a's fixtures (cpu-only,
  one-cuda, two-vendors): a forced accelerator never resolves to cpu; none-Ready refuses with exit
  9|14; cpu/no-gpu/default resolve to cpu; the default takes a Ready accelerator; every
  BACKEND_VALUE resolvable, forced-gpu never cpu. 5/5 PASS.
- Mutation (a): make the not-Ready branch of `resolve_in` return cpu → the case table's
  `a_forced_accelerator_with_none_ready_refuses_and_never_downgrades` and
  `every_backend_value_is_resolvable…` turn RED (observed).
- Mutation (b): restore `cfg!(any(feature="cuda","wgpu"))` at `accel.rs:28` →
  `check_backend_registry.sh --static` RED naming the line (observed).
- CI RED→GREEN pair: owed once the PR runs (fleet starved; PR blocked on #3004 regardless).

## Acceptance (`.pr/R-0b/accept.sh`)
A2 (case table) 5/5, A3 (guard self-test) 3/3, A4 (registry unit tests), A5 (`pv validate`) all GREEN.
A1 (`--static` == 0) is **RED**: two reads remain (below) — the honest state of a partial row.

## Gaps / next (the row's remaining step)
Two backend `cfg!(feature=…)` reads are not yet converted, and both files are blocked by the
pre-commit complexity hook, which charges a staged file for every function in its `include!`
expansion:
- `dispatch.rs:173` (`--backend cuda && !cfg!(feature="cuda")`) — `dispatch.rs` carries
  pre-existing `dispatch_runtime_commands` (cognitive 43) and `dispatch_diagnostic_commands` (30);
- `lib.rs:230` (`cuda_feature = cfg!(feature="cuda")`, a version-json build report) — `lib.rs`
  `include!`s `dispatch.rs` and `help_producer_truth.rs` (`resolve`, cognitive 73).
Converting either read requires decomposing those three pre-existing over-threshold functions in the
same commit (the repo's same-commit rule; `--no-verify` is forbidden). That is a prerequisite
refactor of other features' hot code and is filed separately; when it lands, the two reads convert to
`crate::registry::compiled(…)` and ci.yml arms `check_backend_registry.sh --static` (A1 → GREEN).
Functionally, `apr run --backend cuda` on a non-cuda build still refuses today via the existing
`dispatch.rs:173` cfg check — claim 1 holds; only its *mechanism* (cfg vs registry) is the residual.
Also owed: A3's `GET /v1/effective-config.backend == resolved Selection` + `discovered_at` (REG-12)
on the served process, and `make fleet-verify ROW=R-0b` on four hosts.
