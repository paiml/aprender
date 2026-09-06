---
status: partial
partial_reason: "this PR is not yet merged on the required check; flip to complete with the DAG status write-back after merge"
ticket: PMAT-991
row: R-3
issue: 2906
epic: 2873
model: "orchestrator claude-fable-5-1; workers sonnet (paiml-impl-worker) x2 (P_1, P_2), each resumed once at maxTurns=40; P_2's last mile and P_3 by the orchestrator"
tokens_used: "workers 120478 + 120623 = 241101 (P_1, P_2 first passes; resumes not separately reported); orchestrator [U] (not instrumented)"
wall_clock_s: "[U] (not instrumented); P_1 dispatch ~02:52Z -> P_3 contract commit ~04:20Z on 2026-09-06 (about 5,300 s of orchestration, two worker runs inside)"
---
# impl-PMAT-991 — R-3 · T-6 honest training banner (#2906; spec §5 R-3, D-7; feedback_apr_finetune_lora_cpu_only)

## Identity
ticket PMAT-991 · kind code · branch `agent/R-3` (worktree, claim held) · base `65680cdf8` · `discover.json` at `$XDG_RUNTIME_DIR/paiml-implement/discover-R-3.json` (`gate_cmd_fallback=true`: `cargo test --workspace` is CI's job; crate-level `--lib` gates were run here) · quorum: 3-lane review-only at P3 (agy delegate) · K̂ = 3 (`basis=first-run[U]`, estimate.sh: 0 rows for this row class).

## What lands
- `entrenar` (crates/aprender-train): `BACKWARD_KERNEL_LAUNCHES` (AtomicU64) with `backward_kernel_launches()`, `reset_backward_kernel_launches()`, `pub(crate) note_backward_kernel_launch()` called at every cuBLAS backward launch (`cuda_forward/matmul.rs`: a, a_accumulate, b; `matmul_f16.rs`: a, b); `training_backend_banner(requested) -> Option<String>` is Some iff `requested == "cuda"` and the counter > 0. Test `tests/banner_truth.rs` (RED first at 51bdb5b66: the three symbols did not exist; GREEN at 7c1701431; both `--features cuda` and default arms).
- `apr finetune` (crates/apr-cli): `gpu_backend_decision(requested, build_has_cuda, build_has_wgpu) -> Result<GpuBackendChoice, CliError>` — `cuda`/`wgpu` on a build without that feature is `CliError::FeatureDisabled` (exit code 9, read from `error.rs:106`), `auto`/`cpu` never refuse; `pre_training_notice` never contains "cuBLAS backward"; `post_training_banner` delegates to `entrenar::training_backend_banner`, printed after training. The case table lives in `commands/finetune_gpu_backend_truth_tests.rs` (5 tests; RED first at c8d1cd386).
- `contracts/apr-train-banner-truth-v1.yaml` (kind: pattern; TBT-OB-001..003; tests = the two commands above); README contract count 1811 → 1812.
- `test_run_training_creates_adapter` (pre-existing) hard-coded `gpu_backend = "cuda"` and asserted the old silent CPU fallback; it now names the CPU path it always exercised (`"cpu"`). A test that asserts `is_ok()` on a request the build cannot honour locks the defect in (dogfood 0.63.0 lesson).

## Verification (orchestrator, every command re-run)
| check | result |
|---|---|
| `cargo test -p aprender-train --test banner_truth` | rc 0 (2 passed) |
| `cargo test -p aprender-train --test banner_truth --features cuda` (lambda, RTX 4090) | rc 0 (2 passed; the cuda arm drives one cuBLAS backward and sees the counter move) |
| `cargo test -p apr-cli --lib gpu_backend_truth` | rc 0 (5 passed) |
| `cargo test -p apr-cli --lib` (gate proxy) | rc 0: 7213 passed, 0 failed, 12 ignored |
| `cargo test -p aprender-train --lib` (gate proxy) | rc 101: 7621 passed, 3 failed — `prune::snapshot_tests::tests::{snapshot_all_prune_methods, snapshot_pipeline_stages, snapshot_schedule_validation_errors}`; **ENV, not this diff**: the same three fail identically on a pristine `origin/main` worktree at 65680cdf8 while CI's `workspace-test` on that commit is green (insta snapshot drift local to this box; `crates/aprender-train/src/prune` untouched by this branch) |
| `cargo fmt --all -- --check` · `cargo clippy -p aprender-train --lib -- -D warnings` · `cargo clippy -p apr-cli --lib -- -D warnings` | rc 0 · 0 · 0 |
| `pv validate` · `pv lint` on the contract · `check_contract_test_binding.sh` · `check_contract_enforcement.sh` · `check_readme_claims.sh` · `check_no_claim_literals.sh` | valid · 0/0 · rc 0 ×4 |

## Mutations (RED, then restored GREEN)
1. `training_backend_banner` returns Some for "cuda" regardless of the counter → `banner_truth`: `banner_is_none_for_cuda_with_zero_launches` FAILED and `cpu_only::cpu_backward_never_increments_the_device_counter` FAILED (0 passed, 2 failed). Restored → 2 passed.
2. `gpu_backend_decision` returns `Ok(Cpu)` for "cuda" when the build has no cuda → `cuda_request_on_cpu_only_build_is_a_refusal_not_a_fallback` FAILED and `gpu_backend_decision_case_table` FAILED (3 passed, 2 failed). Restored → 5 passed.

## Dispatch ledger
| phase | mode | agent | turns | maxTurns hit | resumed | outcome |
|---|---|---|---|---|---|---|
| P_1 | subagent:sonnet | a798f959da57eb45b | 40 + resume | yes | once | receipt partial=false; every command re-run green by the orchestrator |
| P_2 | subagent:sonnet | a5710155b2d926446 | 40 + resume | yes | once | receipt partial=true (two blockers below); finished by the orchestrator |
| P_3 | direct | — | — | — | — | contract, README, mutations, receipt |
| P3 review | delegate (agy quorum, width 3) | see the PR body | — | — | — | review-only |
Slots: ≤ 2 live at any instant (worker + delegate); denials 0.

## Jidoka
- **The brief named the crate `aprender_train`; its `[lib] name` is `entrenar`** (S0-10 class: the crate dir and the lib name differ). The worker used `entrenar::` and said so.
- **`crates/apr-cli/src/lib.rs` cannot be committed here**: the repo's pre-commit complexity hook follows lib.rs's `include!()` graph and refuses on three PRE-EXISTING violations (`dispatch.rs:103` cognitive 41, `dispatch.rs:508` cognitive 30, `help_producer_truth.rs:51` cognitive 73), none touched by this ticket. The integration-test seam the worker added there was dropped; the case table lives in `commands::finetune`'s own `#[cfg(test)]` module instead, reached as `cargo test -p apr-cli --lib gpu_backend_truth`. The debt is out of scope and is not discharged with `#[allow]` or `--no-verify`.
- **`git add` with one non-existent pathspec adds nothing** — the relocation commit first landed with only the deletion staged; amended (bae8a988e) with the three files.
- The three prune snapshot failures are an environment finding (see the table); no ticket filed from this row — they belong to whoever owns the insta snapshots, and CI does not see them.

## Gaps
- Not instrumented: custom-PTX backward kernels under `autograd/cuda_backward/` (gemm.rs PTX fallback, elementwise.rs, structured.rs) — P_1 counted the cuBLAS sites the spec names; a PTX-only backward would leave the banner None. Recorded as the row's open question for the 3-lane review.
- The gx10 leg (`host: lambda, gx10`) is not run here: the cuda arm was measured on lambda only. The dogfood row (C4) covers gx10.
- Receipt for this PR: advisory, not produced (driver A1).

## Estimates
K̂ 3 (`basis=first-run[U]`); actual: P_1 worker 40 turns + resume, P_2 worker 40 turns + resume, orchestrator ≈ 9 bash calls for P_2's last mile and P_3 (`basis=this receipt`). Rows appended to `docs/audits/impl-estimates.jsonl`.

## Verdict
PENDING-MERGE (`status: partial`).
