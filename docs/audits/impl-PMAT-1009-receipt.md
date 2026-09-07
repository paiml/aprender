---
status: partial
partial_reason: "this PR is not yet merged on the required check; flip to complete with the DAG status write-back after merge"
ticket: PMAT-1009
row: T-2
issue: 2924
epic: 2873
model: "orchestrator claude-fable-5-1; worker sonnet (paiml-impl-worker), resumed once at maxTurns=40; P_2 (contract, receipt) by the orchestrator"
tokens_used: "worker 80365 (first pass; resume not separately reported); orchestrator [U] (not instrumented)"
wall_clock_s: "[U] (not instrumented); dispatch ~06:05Z -> contract commit ~06:30Z on 2026-09-06 (about 1,500 s)"
---
# impl-PMAT-1009 — T-2 · `--max-seq-len` honoured or refused, never clamped (#2924; spec §5 T-2; S0-11; #2526)

## Identity
ticket PMAT-1009 · kind code · branch `agent/T-2` (worktree, claim held) · base `027ed889d` · `discover.json` at `$XDG_RUNTIME_DIR/paiml-implement/discover-T-2.json` (`gate_cmd_fallback=true`) · quorum: review-only (`--quorum never`; the row is a one-function fix with a case table) · K̂ = 3 (`basis=first-run[U]`).

## What lands
- `effective_max_seq_len(requested: Option<usize>, path: SeqLenPath) -> Result<usize>` in `commands/finetune.rs` — the one place every path (Instruct, Wgpu, Classify) derives its effective value; `Some(n)` → `Ok(n)` or `ValidationFailed` (exit 5, read from `error.rs`), `None` → the path's default. The wgpu instruct pipeline no longer carries the `512, // max_seq_len` literal (finetune.rs:717 before); the instruct path keeps what #2247 gave it, routed through the same function. `Max seq len: <effective>` is printed on all three paths (the classify path already did).
- `finetune_seq_len_truth_tests.rs` (RED first at 64e3d0312 — the function and the enum did not exist; GREEN at 2beca0de1): the case table over {256, 512, 1024, 2048} × three paths, the default row, the refusal-code row (4 tests).
- `contracts/apr-finetune-config-truth-v1.yaml` (kind: pattern; FCT-OB-001; test = the case table); README 1811 → 1812.

## Verification (orchestrator, every command re-run at 2beca0de1)
| check | result |
|---|---|
| `cargo test -p apr-cli --lib finetune_seq_len_truth` | rc 0 (4 passed) |
| `cargo test -p apr-cli --lib finetune` (the module: 80 tests) | rc 0 |
| `cargo fmt --all -- --check` · `cargo clippy -p apr-cli --lib -- -D warnings` | rc 0 · 0 |
| `grep -n '512, // max_seq_len' finetune.rs` | only the doc comment that names the old defect |
| `pv validate` · `pv lint` · `check_contract_test_binding.sh` · `check_contract_enforcement.sh` · `check_readme_claims.sh` · `check_no_claim_literals.sh` · `check_roadmap_diff_additive.sh` | valid · PASS · rc 0 ×5 |

## Mutation (RED, then restored GREEN)
`effective_max_seq_len`'s Wgpu branch returns `Ok(512)` regardless of the request → `effective_max_seq_len_wgpu_never_clamps_to_the_old_512_literal` FAILED and `effective_max_seq_len_honours_every_requested_value_on_every_path` FAILED (2 passed, 2 failed). Restored → 4 passed. (The worker showed the same pair RED in its receipt; re-run by the orchestrator.)

## Dispatch ledger
| phase | mode | agent | turns | maxTurns hit | resumed | outcome |
|---|---|---|---|---|---|---|
| P_1 | subagent:sonnet | a4c04cefd17dbf90d | 40 + resume | yes | once | receipt partial=false; every command re-run green by the orchestrator |
| P_2 | direct | — | — | — | — | contract, README, receipt |
Slots ≤ 1 live; denials 0.

## Jidoka
- The case table's acceptance is `cargo test -p apr-cli --lib finetune_seq_len_truth` (a `#[cfg(test)]` module of `commands::finetune`), not the card's `--test finetune_seq_len_truth` integration target: an integration test cannot reach the crate-private `commands` tree and the `lib.rs` test seam is blocked by the pre-commit complexity gate on pre-existing debt (`dispatch.rs`, `help_producer_truth.rs`; same finding as R-3). Recorded for the DAG write-back.
- The test drives the PURE function, not a training run: the row's claim is that the configuration reaches the engine unclamped; an end-to-end wgpu training run over 2048 tokens is a host-dogfood row (it needs a wgpu adapter and a model).

## Gaps
- No end-to-end run on the wgpu adapter here (lambda has one — Vulkan on the RTX 4090 — but the row's A is the case table; a training run belongs to T-0's harness).
- Receipt for this PR: advisory, not produced (driver A1).

## Estimates
K̂ 3 (`basis=first-run[U]`); actual: worker 40 turns + resume (P_1), orchestrator ≈ 4 bash calls (verification, contract, receipt) (`basis=this receipt`).

## Verdict
PENDING-MERGE (`status: partial`).
