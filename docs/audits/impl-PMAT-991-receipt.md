---
status: partial
partial_reason: "this PR is not yet merged on the required check; flip to complete with the DAG status write-back after merge"
ticket: PMAT-991
row: R-3
issue: 2906
epic: 2873
model: "orchestrator claude-fable-5-1; workers sonnet (paiml-impl-worker) x2 (P_1, P_2), each resumed once at maxTurns=40; P_2's last mile and P_3 by the orchestrator"
tokens_used: "workers 120478 + 120623 = 241101 (P_1, P_2 first passes; resumes not separately reported); orchestrator [U] (not instrumented)"
wall_clock_s: "[U] (not instrumented); P_1 dispatch ~02:52Z -> review-round commit ~04:45Z on 2026-09-06 (about 6,800 s of orchestration, two worker runs and one 3-lane review inside)"
---
# impl-PMAT-991 — R-3 · T-6 honest training banner (#2906; spec §5 R-3, D-7; feedback_apr_finetune_lora_cpu_only)

## Identity
ticket PMAT-991 · kind code · branch `agent/R-3` (worktree, claim held) · base `65680cdf8` · `discover.json` at `$XDG_RUNTIME_DIR/paiml-implement/discover-R-3.json` (`gate_cmd_fallback=true`: `cargo test --workspace` is CI's job; crate-level `--lib` gates were run here) · quorum: 3-lane review-only at P3 (agy delegate) · K̂ = 3 (`basis=first-run[U]`, estimate.sh: 0 rows for this row class).

## What lands
- `entrenar` (crates/aprender-train): `BACKWARD_KERNEL_LAUNCHES` (AtomicU64, monotonic, no reset API) with `backward_kernel_launches()` and `pub(crate) note_backward_kernel_launch()` at every device-side backward launch — `cuda_forward/matmul.rs`: cuBLAS a, a_accumulate, b, NF4 cuBLAS a, NF4 PTX a, NF4 tensor-core a; `matmul_f16.rs`: f16 a, b, f16→f32 a (nine sites); `training_backend_banner(requested, launches_at_start) -> Option<String>` is Some iff `requested == "cuda"` and the counter rose above the caller's start snapshot — the banner is scoped to THIS run. Test `tests/banner_truth.rs` (RED first at 51bdb5b66: the three symbols did not exist; GREEN at 7c1701431; both `--features cuda` and default arms).
- `apr finetune` (crates/apr-cli): `gpu_backend_decision(requested, method_has_cuda_path, build_has_cuda, build_has_wgpu) -> Result<GpuBackendChoice, CliError>` — `cuda`/`wgpu` on a build without that feature is `CliError::FeatureDisabled` (exit code 9, `error.rs:106`); `cuda` for plain LoRA (no cuBLAS path) is `CliError::ValidationFailed` (exit code 5) — `-m lora --gpu-backend cuda` refuses or trains on the GPU, never a CPU run under a GPU flag; `auto`/`cpu` never refuse (auto + plain LoRA is the CPU path, the notice says so); `pre_training_notice` never contains "cuBLAS backward"; `post_training_banner(choice, launches_at_start)` delegates to `entrenar`, snapshot taken before `trainer.train()`. The dead request-derived `gpu_backend_notice`/`GpuBackendPlan` (still carrying the literal "CUDA selected — using cuBLAS backward path") and its five tests are removed. The case table lives in `commands/finetune_gpu_backend_truth_tests.rs` (8 tests; RED first at c8d1cd386).
- `contracts/apr-train-banner-truth-v1.yaml` (kind: pattern; TBT-OB-001..003; tests = the two commands above); README contract count 1811 → 1812.
- `test_run_training_creates_adapter` (pre-existing) hard-coded `gpu_backend = "cuda"` and asserted the old silent CPU fallback; it now names the CPU path it always exercised (`"cpu"`). A test that asserts `is_ok()` on a request the build cannot honour locks the defect in (dogfood 0.63.0 lesson).

## Verification (orchestrator, every command re-run)
| check | result |
|---|---|
| `cargo test -p aprender-train --test banner_truth` | rc 0 (2 passed) |
| `cargo test -p aprender-train --test banner_truth --features cuda` (lambda, RTX 4090) | rc 0 (2 passed; the cuda arm drives one cuBLAS backward and sees the counter move) |
| `cargo test -p apr-cli --lib gpu_backend_truth` | rc 0 (8 passed) |
| `cargo test -p apr-cli --lib` (gate proxy, after the review round) | rc 0: 7211 passed, 0 failed, 12 ignored (the five dead-notice tests removed) |
| `cargo test -p aprender-train --lib` (gate proxy) | rc 101: 7621 passed, 3 failed — `prune::snapshot_tests::tests::{snapshot_all_prune_methods, snapshot_pipeline_stages, snapshot_schedule_validation_errors}`; **ENV, not this diff**: the same three fail identically on a pristine `origin/main` worktree at 65680cdf8 while CI's `workspace-test` on that commit is green (insta snapshot drift local to this box; `crates/aprender-train/src/prune` untouched by this branch) |
| `cargo fmt --all -- --check` · `cargo clippy -p aprender-train --lib -- -D warnings` · `cargo clippy -p apr-cli --lib -- -D warnings` | rc 0 · 0 · 0 |
| `pv validate` · `pv lint` on the contract · `check_contract_test_binding.sh` · `check_contract_enforcement.sh` · `check_readme_claims.sh` · `check_no_claim_literals.sh` | valid · 0/0 · rc 0 ×4 |

## Mutations (RED, then restored GREEN — re-run after the review round, at a137ba684)
1. `training_backend_banner` returns Some for "cuda" regardless of the counter → `banner_truth`: `banner_is_none_for_cuda_with_zero_launches` FAILED, `cpu_only::cpu_backward_never_increments_the_device_counter` FAILED (0 passed, 2 failed). Restored → 2 passed.
2. `gpu_backend_decision` returns `Ok(Cpu)` for "cuda" when the build has no cuda → `cuda_request_on_cpu_only_build_is_a_refusal_not_a_fallback` FAILED, `gpu_backend_decision_case_table` FAILED (6 passed, 2 failed). Restored → 8 passed.
3. `gpu_backend_decision` returns `Ok(Cuda)` for plain LoRA + cuda (no refusal) → `plain_lora_with_explicit_cuda_is_a_refusal_not_a_cpu_run` FAILED (7 passed, 1 failed). Restored → 8 passed.

## Dispatch ledger
| phase | mode | agent | turns | maxTurns hit | resumed | outcome |
|---|---|---|---|---|---|---|
| P_1 | subagent:sonnet | a798f959da57eb45b | 40 + resume | yes | once | receipt partial=false; every command re-run green by the orchestrator |
| P_2 | subagent:sonnet | a5710155b2d926446 | 40 + resume | yes | once | receipt partial=true (two blockers below); finished by the orchestrator |
| P_3 | direct | — | — | — | — | contract, README, mutations, receipt |
| P3 review | delegate (agy quorum, width 3) | ab097fc576df910b1 (agy conversations 5f8882ec…, 5a62974e…, f333e610…) | — | — | — | 3/3 FAIL = mergeable-with-changes; changes applied at a137ba684 (below) |
Slots: ≤ 2 live at any instant (worker + delegate); denials 0.

## Review quorum (3-lane, review-only, 2026-09-06) and what changed
3/3 lanes: mergeable with changes (no lane: do-not-implement). All three accepted the core design (banner from a launch counter, never the request) and verified the evidence section (tests exist, mutations match, README 1812 = `find contracts -name '*.yaml' | wc -l`, `src/prune` untouched). Unanimous defects, both fixed at a137ba684: (a) the process-wide `AtomicU64` with a test-only reset leaked across fine-tunes — a second run in one process, or a long-lived server, would inherit `launches > 0` and re-introduce the ticket's own defect → since-run-start semantics (`training_backend_banner(requested, launches_at_start)`, snapshot before `trainer.train()`, no reset API); (b) `banner_truth.rs` reset the static under cargo's parallel tests → the tests now snapshot and compare, race-free. 2/3: dead `gpu_backend_notice`/`GpuBackendPlan` still carried the request-derived "CUDA selected — using cuBLAS backward path" → removed with its five tests. **Lane 1 alone claimed `gemm_f16_to_f32_backward_a` (matmul_f16.rs:155) launches without the counter; lanes 2 and 3 said all five named sites were covered — lane 1 was right (the function calls `cublas.gemm_f16_to_f32` directly), and the orchestrator found three more uncounted device-side backward launches in `matmul.rs` (`gemm_nf4_backward_a_cublas`, `gemm_nf4_backward_a` PTX, `gemm_nf4_tc_backward_a`) — the QLoRA path itself.** Nine sites are counted now. Lane 1's other singleton (the pre-training notice lost the plain-LoRA CPU warning) was also right in substance: the row's own rule ("-m lora --gpu-backend cuda refuses or trains on the GPU") is now enforced by `gpu_backend_decision` (ValidationFailed, exit 5) and the CPU notice names the plain-LoRA path. Lane 3's "banner on the error path" is moot: `trainer.train()` returns a result struct, and the banner prints after it returns; a training failure that propagates with `?` never reaches the banner.

## Jidoka
- **The brief named the crate `aprender_train`; its `[lib] name` is `entrenar`** (S0-10 class: the crate dir and the lib name differ). The worker used `entrenar::` and said so.
- **`crates/apr-cli/src/lib.rs` cannot be committed here**: the repo's pre-commit complexity hook follows lib.rs's `include!()` graph and refuses on three PRE-EXISTING violations (`dispatch.rs:103` cognitive 41, `dispatch.rs:508` cognitive 30, `help_producer_truth.rs:51` cognitive 73), none touched by this ticket. The integration-test seam the worker added there was dropped; the case table lives in `commands::finetune`'s own `#[cfg(test)]` module instead, reached as `cargo test -p apr-cli --lib gpu_backend_truth`. The debt is out of scope and is not discharged with `#[allow]` or `--no-verify`.
- **`git add` with one non-existent pathspec adds nothing** — the relocation commit first landed with only the deletion staged; amended (bae8a988e) with the three files.
- The three prune snapshot failures are an environment finding (see the table); no ticket filed from this row — they belong to whoever owns the insta snapshots, and CI does not see them.

## Gaps
- Not yet instrumented: the elementwise/structured backward kernels under `autograd/cuda_backward/` (`elementwise.rs`, `structured.rs`); every GEMM-class device backward (cuBLAS f32/f16, NF4 cuBLAS/PTX/tensor-core) is counted. A run whose only device work is elementwise would leave the banner None — it under-reports, never over-claims.
- The gx10 leg (`host: lambda, gx10`) is not run here: the cuda arm was measured on lambda only. The dogfood row (C4) covers gx10.
- Receipt for this PR: advisory, not produced (driver A1).

## Estimates
K̂ 3 (`basis=first-run[U]`); actual: P_1 worker 40 turns + resume, P_2 worker 40 turns + resume, orchestrator ≈ 9 bash calls for P_2's last mile and P_3, ≈ 9 more for the review round (`basis=this receipt`). Rows appended to `docs/audits/impl-estimates.jsonl`.

## Verdict
PENDING-MERGE (`status: partial`).
