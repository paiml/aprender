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

## Acceptance (`.pr/R-0b/accept.sh`) — 5/5 GREEN (second commit)
A1 (`--static` == 0) GREEN, A2 (case table) 5/5, A3 (guard self-test) 3/3, A4 (registry unit
tests), A5 (`pv validate`). `cargo clippy -p apr-cli --lib -- -D warnings` clean; apr-cli lib
suite 7,228 passed / 0 failed (measured on this tree after the serve-test rewrite; the earlier run
had 4 failures — below).

## Second commit: the last two cfg reads converted (#3040 done inside this PR)
`dispatch.rs:173` and `lib.rs:230` are converted; `check_backend_registry.sh --static` is armed
in ci.yml as a live gate. The pre-commit complexity hook charges a staged file for every function
in its `include!` expansion, so this required decomposing three pre-existing over-threshold
functions in the same commit (the same-commit rule; `--no-verify` is forbidden):

| function | before | after |
|---|---|---|
| `dispatch_runtime_commands` (dispatch.rs) | cognitive 43 | `run_preflight` + `run_batch_if_requested` hoisted; arm ≤ 25 |
| `dispatch_diagnostic_commands` (dispatch.rs) | cognitive 30 | `trace_save_tensor_dispatch` + `DiffOpts`/`diff_dispatch` hoisted |
| `help_producer_truth::resolve` | cognitive 73 | a `Walk` cursor: `step`/`long_flag`/`short_takes_value`/`word` |

Oracles: the three `help_producer_truth` tests (3/3 before and after), the serve/accel/registry
tests (278/278), `cargo check` with default and `--features cuda`.

**`apr run --backend <kind>` now resolves against the registry** in `run_preflight` (cuda AND
wgpu — the old check special-cased cuda only, so `--backend wgpu` on a build that could not honour
it fell through; that gap closes here).

### The four serve-guard tests were the old premise, not a regression
`accelerator_guard_tests::{gpu_without_a_backend_is_an_error_not_a_silent_cpu_run,
an_unreachable_backend_is_also_an_error, the_refusal_is_total_over_every_input}` and
`gpu_layers_contract_tests::gpu_layers_is_refused_on_a_build_with_no_accelerator` failed on this
box after the conversion. They asserted "a build without the `cuda`/`wgpu` features is CPU-only",
but apr-cli's `wgpu` feature is `["inference"]` — an alias — so the wgpu inference path exists on
every default build, and the registry truthfully found this host's two AMD W5700X (RADV) adapters
Ready. The cfg gate had been refusing `--gpu` on a build that could serve on wgpu. The tests now
run over fixture registries through `ensure_accelerator_available_in(config, reg)`:
`no-accelerator-compiled` (new twin of cpu-only: cuda/wgpu `source: not-compiled`) ⇒
`FeatureDisabled` with the install remedy; `cpu-only` ⇒ `BackendUnavailable` quoting the flag;
the total-refusal table runs over four fixtures × every request the server accepts. Serve's own
precedence (`--no-gpu` beats `--gpu`; `--backend cpu` asks for nothing) is unchanged.

### Pre-existing, not touched (recorded, not fixed here)
- `cargo check -p apr-cli --no-default-features` does not build on origin/main either
  (`diff_05_aprt_stage.rs:100` uses `realizar` unconditionally; `lib.rs:74` `serve::auth`); no
  gate builds that configuration.
- `cargo clippy -p apr-cli --tests`: 36 `disallowed_methods` in `tests/falsification_crux_{a_22,k_08}.rs`
  (on main), 2 in `nf4_classifier.rs`, 5 in R-0a's `registry_failure_catalogue.rs`. The gated form
  is `--lib`; ci.yml's header claims `--all-targets` but carries no clippy step.
- `aprender-serve` `double_must_use` on `ParityReport::not_run` (L0-1a's code, #3026): fixed at its
  source on `agent/L0-1` (278417bbc, local, held while the queue is BSE-001's) and merged here.

## Gaps / next
Still owed before `status: complete`: the `selected:` line at every model-load site (run/chat/serve),
A3's `GET /v1/effective-config.backend == resolved Selection` + `discovered_at` (REG-12) on the
served process, `make fleet-verify ROW=R-0b` on four hosts, and the CI RED→GREEN mutation pair
(the fleet is BSE-001's until told otherwise; this PR is stacked on R-0a #3004 and cannot arm
before it merges).  A3's `GET /v1/effective-config.backend == resolved Selection` + `discovered_at` (REG-12)
on the served process, and `make fleet-verify ROW=R-0b` on four hosts.
