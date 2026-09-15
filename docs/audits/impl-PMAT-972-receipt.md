---
status: partial
partial_reason: "this PR is not yet merged on the required check; flip to complete with the DAG status write-back after merge"
ticket: PMAT-972
row: I-24
issue: 2887
epic: 2873
model: "orchestrator claude-fable-5-1; workers sonnet (paiml-impl-worker) x2 (P_1 guard + case table, P_1b decomposition), each resumed once at maxTurns=40; the faithful mutation, the docstring fix, the ci.yml wiring and P_2 by the orchestrator"
tokens_used: "workers 110697 + 104001 = 214698 (first passes; resumes not separately reported); orchestrator [U] (not instrumented)"
wall_clock_s: "[U] (not instrumented); dispatch ~07:40Z -> contract commit ~08:35Z on 2026-09-06 (about 3,300 s)"
---
# impl-PMAT-972 — I-24 · parity_block.py: a zero COMPARATOR band median is a named refusal, not a traceback (#2887; #2735)

## Identity
ticket PMAT-972 · kind code · branch `agent/I-24` (worktree, claim held) · base `027ed889d` · `discover.json` at `$XDG_RUNTIME_DIR/paiml-implement/discover-I-24.json` (`gate_cmd_fallback=true`; no Rust in the diff) · quorum: review-only (`--quorum never`) · K̂ = 3 (`basis=first-run[U]`).

## What lands
- `scripts/lib/parity_block.py`: `_comparator_median(values, label)` — the one guarded divisor helper; every median-as-divisor site goes through it; an empty comparator band refuses `band <c> comparator: no samples is not a measurement`, a non-positive median refuses `band <c> comparator: non-positive rate is not a measurement`, both through the script's existing refusal path (exit 1). The subject-side refusal is untouched. `_executor_side` / `_executor_lane` / `build` / `main` decomposed (pure code motion) so the file fits the pre-commit complexity gate (three of the four were over cognitive 25 on main already).
- `scripts/check_parity_block_refusals.sh --selftest` (RED first at 72383a9de: rows 2 and 4 died with the traceback): four generated fixtures in the HISTORICAL layout — control (rc 0), zero comparator (rc 1 + text), zero subject (rc 1 + the pre-existing wording), empty comparator (rc 1 + text); every row also asserts no `Traceback`. Wired into ci.yml `guard-runner-labels`.
- `contracts/apr-parity-block-refusals-v1.yaml` (kind: pattern; PBR-OB-001); README 1811 → 1812.

## Verification (orchestrator, every command re-run)
| check | result |
|---|---|
| `bash scripts/check_parity_block_refusals.sh --selftest` | PASS, 4 fixtures |
| `python3 -m py_compile scripts/lib/parity_block.py` | rc 0 |
| `pmat analyze complexity --file scripts/lib/parity_block.py --max-cyclomatic 30 --max-cognitive 25` (the hook's thresholds) | accepted by the pre-commit hook (three pre-existing violators decomposed) |
| `bash scripts/check_thresholds_in_matrix.sh --selftest` | 6 passed, 0 broken (a first docstring read as a float comparison to its CMP regex and broke it — reworded) |
| `bashrs lint scripts/check_parity_block_refusals.sh` | 0 errors |
| `pv validate` · `pv lint` · `check_contract_test_binding.sh` · `check_contract_enforcement.sh` · `check_readme_claims.sh` · `check_no_claim_literals.sh` · `check_roadmap_diff_additive.sh` · `check_guards_are_wired.sh` · `check_workflow_env_defined.sh` | valid · PASS · rc 0 ×7 |

## Mutation (RED, then restored GREEN)
The guard call and the guarded division replaced by the bare `statistics.median(subject_values) / statistics.median(comparator_values)` → row 2 FAILED: `ZeroDivisionError: float division by zero`, `contains forbidden text: Traceback`. Restored → PASS. (A first attempt that kept the guard call and only changed the divisor stayed GREEN — the guard raises before the division; the faithful mutation removes both, and that is the one recorded.)

## Dispatch ledger
| phase | mode | agent | turns | maxTurns hit | resumed | outcome |
|---|---|---|---|---|---|---|
| P_1 | subagent:sonnet | a9e6dadb94785aba9 | 40 + resume (40) | yes, twice | once | left the RED commit and the guard uncommitted with the self-test green; its lock outlived it (removed by hand, see Jidoka) |
| P_1b | subagent:sonnet | a660a2ea336b4f813 | 40 + resume | yes | once | decomposition committed (b11fd9ce4) |
| P_2 | direct | — | — | — | — | docstring fix, ci.yml wiring, mutation, contract, receipt |
Slots ≤ 1 live; denials 0.

## Jidoka
- **Stale worker lock**: the P_1 worker stopped at its turn limit twice and its `SubagentStart` lock outlived it, so the hook refused every orchestrator command containing `gh pr` ("push/PR is orchestrator-only; a worker is running") — removed by hand (`$XDG_RUNTIME_DIR/claude-subagent-<session>.lock/<agent>`), as in earlier rows.
- **The complexity gate on a file with pre-existing debt**: `parity_block.py` carried three functions over cognitive 25 on main (`_executor_side` 35, `_executor_lane` 43, `build` 31); the fix could not be committed without decomposing them in the same commit (pure code motion, self-test unchanged).
- **A docstring is code to a guard**: `check_thresholds_in_matrix.sh`'s CMP regex read the prose `0 / positive == 0.0` in `_comparator_median`'s docstring as a float-literal comparison and turned its own self-test RED; reworded.
- The card's mutation names line numbers (`:571/652/680/720`) that had drifted; the sites are now one helper, so the mutation is the helper's bare division.

## Gaps
- Receipt for this PR: advisory, not produced (driver A1).

## Estimates
K̂ 3 (`basis=first-run[U]`); actual: two workers × (40 + resume), orchestrator ≈ 6 bash calls (`basis=this receipt`).

## Verdict
PENDING-MERGE (`status: partial`).
