---
status: partial
partial_reason: "this PR is not yet merged on the required check; flip to complete with the DAG status write-back after merge"
ticket: PMAT-978
row: C0-4
issue: 2893
epic: 2873
model: "orchestrator claude-fable-5-1; worker sonnet (paiml-impl-worker), resumed once at maxTurns=40; P_2 (mutation re-run, contract, receipt) by the orchestrator"
tokens_used: "worker 137948 (first pass; resume not separately reported); orchestrator [U] (not instrumented)"
wall_clock_s: "[U] (not instrumented); dispatch ~08:50Z -> contract commit ~09:35Z on 2026-09-06 (about 2,700 s; the self-test alone takes ~96 s per run)"
---
# impl-PMAT-978 — C0-4 · perf_gate.sh: an Arm A that measured nothing on a c=1-only receipt is REPORT, never PASS (#2893; #2830)

## Identity
ticket PMAT-978 · kind code · branch `agent/C0-4` (worktree, claim held) · base `027ed889d` · `discover.json` at `$XDG_RUNTIME_DIR/paiml-implement/discover-C0-4.json` (`gate_cmd_fallback=true`; no Rust in the diff) · quorum: review-only (`--quorum never`) · K̂ = 3 (`basis=first-run[U]`) · owner perf-gate.

## What lands
- `scripts/perf_gate.sh`, Arm A (`arm_a_self_regression`, PP-31): every line the arm emits goes through one `say()` that records that the arm spoke; a receipt whose only band is c=1 (the denominator) used to walk every print and fire none, and the gate read the silence as "nothing failed" → `VERDICT PASS`. Now the arm prints `REPORT ArmA scaling: c=1 only, no scaling measured` when it measured nothing, and silence is never a PASS. The multi-band output is unchanged.
- `--selftest` rows `arm_a_c1_only_not_pass` (RED polarity: the c=1-only fixture must carry the REPORT line and never `VERDICT PASS`; RED first at 7e4b9ee11) and `arm_a_multi_band_ok` (the c=1 + c=8 fixture prints the normal Arm A lines).
- `contracts/pp-llama-001-perf-gate-v1.yaml`: the invariant under `a_ratchet_compares_quantities` and `FALSIFY-PP-LLAMA-001-PERF-GATE-012` (the existing contract; no new file, no README count change).

## Verification (orchestrator, every command re-run at c3308947c)
| check | result |
|---|---|
| `bash scripts/perf_gate.sh --selftest` (~96 s) | rc 0; `arm_a_c1_only_not_pass expect=fail ok`, `arm_a_multi_band_ok expect=pass ok` among the existing rows |
| `bashrs lint scripts/perf_gate.sh` | 0 errors |
| `pv validate` · `pv lint` on the contract · `check_contract_test_binding.sh` · `check_contract_enforcement.sh` · `check_no_claim_literals.sh` · `check_roadmap_diff_additive.sh` · `check_readme_claims.sh` | valid · PASS · rc 0 ×5 |

## Mutation (RED, then restored GREEN)
The `print("REPORT ArmA scaling: c=1 only, no scaling measured")` site replaced by a no-op → `arm_a_c1_only_not_pass` BROKE ("fail but never said REPORT ArmA scaling: c=1 only, no scaling measured"); the crude replacement also broke two neighbouring rows (`self_regress_fail`, `phase_guard_a_merge` — the silent-arm rule now demotes those verdicts too), which is the intended coupling: an arm that says nothing cannot PASS anything. Restored → rc 0, every row green. (The worker's own receipt shows the same row RED with the REPORT line removed.)

## Dispatch ledger
| phase | mode | agent | turns | maxTurns hit | resumed | outcome |
|---|---|---|---|---|---|---|
| P_1 | subagent:sonnet | a832747f058c09cc0 | 40 + resume (40) | yes, twice | once | both commits landed (7e4b9ee11 RED, c3308947c GREEN); its lock outlived it (removed by hand) |
| P_2 | direct | — | — | — | — | mutation re-run, contract, receipt |
Slots ≤ 1 live; denials 0.

## Jidoka
- The card names `arm_a_scaling`; PP-31 renamed the arm `arm_a_self_regression` ("self-regression, not scaling efficiency") — the defect is the same silence, recorded under the current name.
- The self-test takes ~96 s per run; the worker's combined verify command timed out and it split the commands — the receipt's gate is two commands.
- Stale worker lock removed by hand after the turn limit (fourth time this campaign).

## Gaps
- Receipt for this PR: advisory, not produced (driver A1).

## Estimates
K̂ 3 (`basis=first-run[U]`); actual: worker 40 + resume, orchestrator ≈ 4 bash calls (`basis=this receipt`).

## Verdict
PENDING-MERGE (`status: partial`).
