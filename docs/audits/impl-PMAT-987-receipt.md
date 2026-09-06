---
status: partial
partial_reason: "waiting on #2981 (the DAG file) and G-6 to merge — this branch's live check reads docs/specifications/pp-066-dag.yaml, which is on main only after #2981; flip to complete with the DAG write-back after merge"
ticket: PMAT-987
row: G-4
issue: 2902
epic: 2873
model: "orchestrator claude-fable-5-1 (direct; no worker dispatched — the three preceding sonnet workers all hit maxTurns=40)"
tokens_used: "orchestrator [U] (not instrumented)"
wall_clock_s: "[U] (not instrumented); 9 orchestrator bash calls for P_1–P_4"
---
# impl-PMAT-987 — G-4 · the obligation DAG as data, invariants in CI (#2902; spec §4 C10)

## Identity
ticket PMAT-987 · kind code · branch `agent/G-4` (worktree, claim held) · base `132ccda56` · HEAD at receipt time: the contract commit above · `discover.json` at `$XDG_RUNTIME_DIR/paiml-implement/discover-G-4.json` (`gate_cmd_fallback=true`; no Rust in the diff) · quorum `--quorum never` (review-only row) · K̂ = 184 (`basis=docs/audits/impl-estimates.jsonl:L1-L7`).

## What lands
- `scripts/lib/dag_invariants.py` — D1 no dangling blocker · D2 acyclic · D3 ≥ `--min-slack-days` (default 6, spec §2) blocker→blockee on the 0.66 lane · D4 per-host queues ordered by resolved expiry · D5 owner · D6 exactly one expiry form (date xor `{anchor, days}` with a real anchor) · past-expiry rows REPORTED by id (deterministic: `--today` defaults to the DAG's `generated` date).
- `scripts/check_dag_invariants.sh` — the wiring + an 11-row `--selftest` (one RED and one GREEN per rule on a synthetic 6-row DAG); a missing DAG is exit 2, never a pass.
- `scripts/render_dag.py` — renders the §5/§6 block from the yaml; `--check` exits 0 (match) / 1 (drift, unified diff) / 3 (NOT ARMED: the spec carries no marker pair yet — SPEC-1.6 inserts it and wires the check).
- `ci.yml` `guard-runner-labels`: case table, then the live DAG check, beside `spec_conformance.sh`.
- `contracts/apr-obligation-dag-v1.yaml` (`kind: pattern`; tests = the self-test and `check_guards_are_wired.sh`).
- Not `pmat comply check --rule obligation-dag`: pmat 3.37.0 has no such rule (measured); a pmat ticket is owed and the DAG row's notes say so.

## Verification (orchestrator, every command re-run)
| check | result |
|---|---|
| `bash scripts/check_dag_invariants.sh --selftest` | rc 0, `11/11 rows` |
| `bash scripts/check_dag_invariants.sh docs/specifications/pp-066-dag.yaml --min-slack-days 6` (the plan branch's DAG, copied in untracked for this run) | rc 0, `rows=90 violations=0 queues=2 past_expiry=0` |
| `bash scripts/check_dag_invariants.sh /nonexistent.yaml` | rc 2 |
| `python3 scripts/render_dag.py render` · `--check` against the v1.5 spec | 155-line block · rc 3 NOT ARMED (expected until SPEC-1.6) |
| `bashrs lint scripts/check_dag_invariants.sh` | 0 errors (one SEC011 finding fixed with the repo's `safe_rm_scratch` idiom) |
| `pmat analyze complexity` | dag_invariants.py max cyclomatic 9 (one cognitive-15 warning: `expiry_form` 23); render_dag.py max cyclomatic 7, no violation; the pre-commit gate accepted both |
| `check_guards_are_wired.sh` · `check_workflow_env_defined.sh` · `check_pr_review_wiring.sh` | rc 0 each |
| `pv validate contracts/apr-obligation-dag-v1.yaml` · `check_contract_test_binding.sh` · `check_contract_enforcement.sh` · `pv lint` | see the P_3 lines of the orchestrator log (rc 0 / rc 0 / rc 0 / 0 errors) |

## Mutations on the real DAG (RED, then restored GREEN)
1. `R-2.expiry := R-0.expiry` → rc 1, `D3 R-0 -> R-2: slack 0 d < 6 d`.
2. `host_queues.gx10` positions 2 and 4 swapped → rc 1, `D4 gx10: T-0 (2026-10-03) is queued before T-1 (2026-09-26)`.
3. `G-4.blockers += [G-4]` → rc 1, `D2 cycle: G-4 -> G-4`.
Restored → rc 0 PASS. The DAG copy stays untracked here (it lands via #2981; a second add would be a merge conflict, which the driver names a STOP).

## Jidoka
- bashrs SEC011 on `rm -rf "$TD"` → adopted `check_pr_review_arm4.sh`'s `safe_rm_scratch` idiom.
- The first `render_dag.py` recursed on a decorated string and parsed `--check` as a positional; both fixed before commit (the pre-commit gate does not run python tests — the guard's own runs are the check).
- Spec §5 G-4's acceptance names `pmat comply check --rule obligation-dag`, which pmat does not have; recorded as [U] in the DAG row and here.
- `render_dag.py` truncated titles to 140 characters, which cut the evidence citation off SPEC-1.6's rendered B-M3 row and turned `check_no_claim_literals.sh` RED on the render while the DAG was GREEN (e6687cb88: titles are never truncated). A first fix put a trailing comment on the tuple line and swallowed three cells (8 columns under an 11-column header); caught by counting columns in the render, not by `--check`, which compares a render against itself.

## Gaps
- This PR must merge after #2981 (the DAG file) and after G-6/C0-5 in the STEP B order; until then its live-check step is RED by design on a branch without the file.
- `render_dag.py --check` is armed by SPEC-1.6 (marker pair + the ci.yml step).
- Receipt for this PR: advisory, not produced (driver A1).

## Verdict
PENDING-MERGE (`status: partial`).
