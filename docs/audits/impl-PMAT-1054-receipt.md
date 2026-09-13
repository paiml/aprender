---
status: complete
merged: "PR #2985 squash cdc0acb99, 2026-09-05T16:44:33Z, through the merge queue (queue CI run 33975501157: ci / gate + workspace-test green)"
ticket: PMAT-1054
row: C0-5
issue: 2982
epic: 2873
model: "orchestrator claude-fable-5-1; workers sonnet (paiml-impl-worker) x2"
tokens_used: "workers 116952 + 132291 = 249243; orchestrator [U] (not instrumented)"
wall_clock_s: "974 (P_1 dispatch 12:04:47Z -> P_3 commit 12:21:01Z; P_4 excluded)"
---
# impl-PMAT-1054 — C0-5 · PRQ-013: the PR's own receipt is judged from the base

## Identity
ticket PMAT-1054 · kind code · branch `agent/C0-5` (worktree, claim held by `hold.sh`) · base `42be1560b` (`origin/main` after #2872) · HEAD at receipt time `9b0195ed8` · `discover.json` sha256 `4ee4f6b3e329fbed…` (`gate_cmd_fallback=true`: discovery found only `cargo test --workspace`; the diff carries no Rust, so the measured gates are the guards named below) · quorum `--quorum never` (driver: review-only row; the plan-quorum override is recorded in `docs/audits/pp-066-plan-quorum.md`).

## Finding (A1 of the PP-066 driver, measured 2026-09-05)
Ruleset *Green Main — unified gate enforcement* requires the bare `gate` context; `gate` (`ci.yml`) `needs: pr-review-present` and exits 1 on its `failure`; `pr-review-present` and the own-receipt step of `pr-review-receipt` run `if: github.event_name == 'pull_request'`, whose definition GitHub reads from the PR head. The check that judged a PR's own review receipt was editable by the PR under review. Branch protection contexts: `ci / gate`, `workspace-test`.

## Plan and routing
| phase | what | route | trigger | A_i |
|---|---|---|---|---|
| P_1 | `scripts/check_receipt_gate_base_owned.sh` (B1..B4, 17-row case table), RED-first at HEAD | subagent:sonnet | — | `bash scripts/check_receipt_gate_base_owned.sh --self-test`; guard at HEAD RED |
| P_2 | `.github/workflows/pr-review-quorum.yml` (`pull_request_target` + `merge_group`, base checkout, receipt dir materialised with `git archive`, step outputs bind PR identity); `ci.yml`: `gate` drops `pr-review-present`, the job is removed, the receipt job's own-receipt step is removed, the new guard is wired in `guard-runner-labels` | direct (workers may not edit workflows) | — | guard GREEN; wiring, guards-wired, env-defined, runner-labels, path-filters, shell-lint guards GREEN |
| P_3 | `contracts/apr-required-checks-v1.yaml` (`kind: pattern`, no `registry: true`, no `verification_summary`, `test:` = the self-test invocation) | subagent:sonnet | — | `pv validate` 0/0; `check_contract_test_binding.sh`; `check_contract_enforcement.sh`; `pv lint` 0 errors |
| P_4 | mutations RED→GREEN, DoD gates, receipt, PR | direct | — | see verification |
K̂ = 184 (`basis=docs/audits/impl-estimates.jsonl:L1-L7`, median 46/phase × 4), K = 368; orchestrator turns used ≈ 13.

## Dispatch ledger
| phase | mode | agent id | turns | maxTurns hit | resumed | receipt JSON |
|---|---|---|---|---|---|---|
| P_1 | subagent:sonnet | `a0611dc49dbf9cef5` | 40 | yes (after its commit `c7816c632`) | no | missing → `partial=true` per §6.2; verified by the orchestrator's own re-run below |
| P_3 | subagent:sonnet | `a4b517a547186733b` | 40 | yes (after its commit `9b0195ed8`) | no | missing → `partial=true` per §6.2; verified below |
Slots: 3; live peak 1; denials from `events-<session>.jsonl`: 0 (two stale lock entries left by the maxTurns exits were removed by hand before the push — a SubagentStop that never fired, recorded here). I-3: `PASS transcript-gate: attempted=10 denied=0 running_peak=1 slots=3` (session-wide: the count includes the pr-review reviewer agents dispatched earlier in the session for #2872/#2875).

## Verification (claimed vs re-run by the orchestrator)
| check | worker claim | orchestrator re-run |
|---|---|---|
| `bash scripts/check_receipt_gate_base_owned.sh --self-test` | (no receipt) | rc 0, `SELF-TEST PASSED (17/17 rows)` |
| guard at HEAD before P_2 | (no receipt) | rc 1: `FAIL B1` (two head-defined invocations: `ci.yml:pr-review-present`, `ci.yml:pr-review-receipt`), `FAIL B3` (`gate` needs `pr-review-present`) — the RED-first commit |
| guard after P_2 | — | rc 0, `ok  B1..B4 hold: pr-review-quorum.yml judges the PR's own receipt from the base` |
| `check_pr_review_wiring.sh` · `check_guards_are_wired.sh` · `check_workflow_env_defined.sh` · `check_runner_labels.sh` · `check_workflow_path_filters.sh` · `check_shell_lint_ratchet.sh` | — | rc 0 each |
| `pv validate contracts/apr-required-checks-v1.yaml` (via `scripts/pv_bin.sh`) | (no receipt) | rc 0, `0 error(s), 0 warning(s)` |
| `check_contract_test_binding.sh` · `check_contract_enforcement.sh` · `pv lint <file>` | (no receipt) | rc 0 · rc 0 · `0 errors, 0 warnings` |
| `check_no_claim_literals.sh` · `check_baseline_ratchets.sh` · `check_pass_grep_anchored.sh` · `check_roadmap_completion_is_cited.sh` · `spec_conformance.sh` | — | rc 0 each |
| `actionlint` on both workflows | — | only the pre-existing `clean-room` runner-label finding (ci.yml carries the same 18); no new shellcheck finding |
| kind-gate (DoD) · status-lint | — | `kind=code files=5` rc 0 · `3 blocks, all with basis=` |

## Mutations (RED then GREEN, on the working tree, restored)
1. `pr-review-quorum.yml` `on: pull_request_target` → `on: pull_request` → guard rc 1: `FAIL B2: pr-review-quorum.yml does not declare pull_request_target` → restored → rc 0.
2. `ci.yml` `gate` `needs:` gains `pr-review-present` → guard rc 1: `FAIL B3: job gate in ci.yml needs pr-review-present — a head-defined receipt job` → restored → rc 0.
Discrimination: under mutation 1 `scripts/check_pr_review_wiring.sh` stays rc 0 (it governs the receipt guard's own self-test job, which is legitimately head-defined).

## Jidoka
- `pmat hooks install --strict --force` fails in a git worktree (`Error: Not a directory (os error 20)` — `.git` is a file there); commits carry the `Pmat-Ticket:` trailer by hand. Owner: pmat (ticket owed).
- `check_workflow_env_defined.sh` fails closed on names set through `GITHUB_ENV` (by design); the workflow binds the resolved PR identity as step outputs and per-step `env:` instead.
- `merge_group` runs on the queue's merge ref, which contains the PR; only the `pull_request_target` run is base-defined. Stated in the workflow header and the contract.
- Both workers stopped at maxTurns (40) after committing; their locks outlived them (no SubagentStop) and blocked the orchestrator's push until removed.

## Gaps
- Receipt for this PR itself: **advisory, not produced** (A1: nothing this session produces is evidence for its own PR; regeneration stopped). The first PR judged by `pr-review-quorum.yml` is the next one — a `pull_request_target` workflow runs the base's definition, which does not carry this file until this PR merges.
- Ruleset change (make `pr-review-quorum / present` a required context and drop the `gate` dependency on receipts) — recorded for the operator, never applied by this session.
- C0-6 (mutation-step timeout) and C0-7 (receipt marker guard) are separate rows.

## Estimates
`K̂=184 basis=docs/audits/impl-estimates.jsonl:L1-L7`; actual orchestrator turns ≈ 13 for P_0–P_4; rows appended to `docs/audits/impl-estimates.jsonl`.

## Merge evidence (orchestrator, after the fact)
- Merged as `cdc0acb99` (#2985) at 2026-09-05T16:44:33Z through the merge queue; queue CI run 33975501157: `ci / gate` and `workspace-test` green; the pull_request run 33972942257 had every gate leg green (`ci / gate`, `workspace-test`, `mutants`, `guard-runner-labels`).
- The mechanism engaged on the queue: `pr-review-quorum.yml` fired on `merge_group` (run 33975500851, job `present`, runner intel-clean-room-8), resolved `pr=2985` from the queue ref and `head=cdc0acb99` (the queue's merge commit, of which the PR head is an ancestor; Arm 4 accepts an ancestor-bound receipt), and reported `A2 no receipt directory at evidence/pr-review/2985` → RED. That RED is the missing receipt this session does not produce (driver A1: receipts advisory), on a context no ruleset requires; it did not block the merge and is the expected shape until receipts are re-armed.
- A1–A3 re-run on the merged tree (main `cdc0acb99` + the plan docs): `check_receipt_gate_base_owned.sh --self-test` rc 0, live rc 0; `check_pr_review_wiring.sh` rc 0 (and `--selftest` rc 0); `gate.needs = [ci, workspace-test, mutants, guard-runner-labels]`; the quorum workflow's `on:` = `[pull_request_target, merge_group]`, job `present` gated on exactly those two events. A3: protection contexts `ci / gate`, `workspace-test` (strict) and ruleset "Green Main" → `gate`, unchanged by the session (read back after the merge).

## Verdict
DONE (`status: complete`): every A_i re-run green by the orchestrator on the merged tree; merged green on the required checks.
