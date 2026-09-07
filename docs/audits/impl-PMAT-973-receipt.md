---
status: partial
partial_reason: "this PR is not yet merged on the required check; flip to complete with the DAG status write-back after merge"
ticket: PMAT-973
row: I-25
issue: 2888
epic: 2873
model: "orchestrator claude-fable-5-1; worker sonnet (paiml-impl-worker), resumed once at maxTurns=40 and stopped again with the fix uncommitted; the commit, the mutation, the contract and this receipt by the orchestrator"
tokens_used: "worker 92101 (first pass; resume not separately reported); orchestrator [U] (not instrumented)"
wall_clock_s: "[U] (not instrumented); dispatch ~06:50Z -> contract commit ~07:20Z on 2026-09-06 (about 1,800 s)"
---
# impl-PMAT-973 — I-25 · `--workload` bound to the prompt corpus (#2888; #2756)

## Identity
ticket PMAT-973 · kind code · branch `agent/I-25` (worktree, claim held) · base `027ed889d` · `discover.json` at `$XDG_RUNTIME_DIR/paiml-implement/discover-I-25.json` (`gate_cmd_fallback=true`) · quorum: review-only (`--quorum never`) · K̂ = 3 (`basis=first-run[U]`) · owner perf-gate.

## What lands
- `commands/test_llm_band.rs`: `WorkloadBinding { label, corpus_sha256, prompt_count }` and `bind_workload(label, prompts_file)` — accepted only when the file's first-line `_meta.corpus` equals the label (`ValidationFailed` naming both labels otherwise; `ValidationFailed` naming "not a labelled corpus" with no header); `corpus_sha256` = sha256 of the file bytes; `receipt_accepts_workload` (a corpus label without a digest is not accepted); the band receipt carries `corpus_sha256`; `--workload <corpus label>` with no prompts file (`--profile short` = one prompt sent N times) is refused before the first request with one line naming the corpus.
- `test_llm_band_workload_binding_tests.rs` (RED first at 24b077aa4 — the function did not exist): the real W1 corpus binds (256 prompts, 64-hex digest); a W2-labelled file refuses W1; a header-less file refuses; a receipt with a corpus label and no digest is refused (4 tests).
- `contracts/apr-workload-corpus-binding-v1.yaml` (kind: pattern; WCB-OB-001); README 1811 → 1812.

## Verification (orchestrator, every command re-run at e925fc058)
| check | result |
|---|---|
| `cargo test -p apr-cli --lib workload_corpus_binding` | rc 0 (4 passed) |
| `cargo test -p apr-cli --lib test_llm_band` (the module: 65 tests) | rc 0 |
| `cargo fmt --all -- --check` · `cargo clippy -p apr-cli --lib -- -D warnings` | rc 0 · 0 |
| `pv validate` · `pv lint` · `check_contract_test_binding.sh` · `check_contract_enforcement.sh` · `check_readme_claims.sh` · `check_no_claim_literals.sh` · `check_roadmap_diff_additive.sh` | valid · PASS · rc 0 ×5 |

## Mutation (RED, then restored GREEN)
The `_meta.corpus` comparison in `bind_workload` removed (any file accepted) → `a_corpus_labelled_w2_refuses_the_w1_label` FAILED, `a_prompt_set_with_no_meta_header_is_not_a_labelled_corpus` FAILED (2 passed, 2 failed). Restored → 4 passed.

## Dispatch ledger
| phase | mode | agent | turns | maxTurns hit | resumed | outcome |
|---|---|---|---|---|---|---|
| P_1 | subagent:sonnet | a8aca22b1e219ac2d | 40 + resume (40) | yes, twice | once | RED commit landed; the fix was complete but uncommitted (fmt clean, clippy pending) — committed by the orchestrator after re-running everything; lock removed by hand |
| P_2 | direct | — | — | — | — | mutation, contract, receipt |
Slots ≤ 1 live; denials 0.

## Jidoka
- The card's A names `cargo test -p apr-cli --test workload_corpus_binding`; the tests are a `#[cfg(test)]` module of `commands::test_llm_band` (`--lib workload_corpus_binding`) for the same reason as R-3/T-2: the crate-private `commands` tree and the pre-commit gate on `lib.rs`.
- The second A (`apr test llm band --workload W1 --profile short` refused with one line) is implemented as the refusal before the first request in the band-run setup; it is exercised by the unit rows, not by a live server run.
- Stale worker lock removed by hand after the turn limit (fifth time this campaign).

## Gaps
- No live `apr test llm band` run here (needs a server); the refusal branch is the unit-tested path.
- Receipt for this PR: advisory, not produced (driver A1).

## Estimates
K̂ 3 (`basis=first-run[U]`); actual: worker 40 + resume (40), orchestrator ≈ 4 bash calls (`basis=this receipt`).

## Verdict
PENDING-MERGE (`status: partial`).
