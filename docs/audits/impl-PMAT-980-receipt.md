---
status: complete
merged: "PR #2987 squash 65680cdf8, 2026-09-06T02:37:35Z, through the merge queue (queue CI run 34005252966: ci / gate + workspace-test green)"
ticket: PMAT-980
row: G-6
issue: 2895
epic: 2873
model: "orchestrator claude-fable-5-1; worker sonnet (paiml-impl-worker) x1 (+1 resume)"
tokens_used: "worker 131815 + resume [U] (not reported by the harness); orchestrator [U]"
wall_clock_s: "[U] — P_1 worker 539 s + resume; orchestrator refactor and P_2–P_4 not instrumented"
---
# impl-PMAT-980 — G-6 · a PR's roadmap diff is additive (#2874, class #2878/#2630)

## Identity
ticket PMAT-980 · kind code · branch `agent/G-6` (worktree, claim held) · base `132ccda56` (`origin/main` after #2875) · HEAD at receipt time `21d467b5a` · `discover.json` at `$XDG_RUNTIME_DIR/paiml-implement/discover-G-6.json` (`gate_cmd_fallback=true`; the diff has no Rust) · quorum `--quorum never` (review-only row).

## Plan and routing
| phase | what | route | A_i |
|---|---|---|---|
| P_1 | `scripts/lib/roadmap_diff.py` (check R1..R4 + trim), `scripts/check_roadmap_diff_additive.sh` (15-row self-test incl. the real `6d9ba274a` row, base-side duplicates and the shallow-checkout base resolution), `scripts/roadmap_trim.py` | subagent:sonnet (40 turns, resumed once, 40 more) → finished **direct** by the orchestrator: the bash guard was written by the worker; the python library was decomposed by the orchestrator to pass the pre-commit complexity gate | `bash scripts/check_roadmap_diff_additive.sh --self-test` |
| P_2 | wired into `ci.yml` job `guard-runner-labels` (self-test, then the live check of the PR's diff) | direct | `bash scripts/check_guards_are_wired.sh` |
| P_3 | `contracts/apr-roadmap-additive-diff-v1.yaml` (`kind: pattern`, tests = the self-test invocation and `check_guards_are_wired.sh`) | direct (the previous two contract workers exhausted 40 turns) | `pv validate` 0/0 |
| P_4 | mutation RED→GREEN, receipt, PR | direct | see below |
K̂ = 184 (`basis=docs/audits/impl-estimates.jsonl:L1-L7`); orchestrator turns for this row ≈ 12.

## Dispatch ledger
| phase | mode | agent id | turns | maxTurns | resumed | receipt JSON |
|---|---|---|---|---|---|---|
| P_1 | subagent:sonnet | `a00c5def08d2bcd64` | 40 + 40 | yes, twice | once (SendMessage), then the orchestrator finished the phase directly | missing → `partial=true` per §6.2; every A_i re-run below |
Denials: 0. Two stale lock entries (dead workers) were removed by hand before the orchestrator's own work. I-3 at receipt time: session-wide `attempted=10+` (see the C0-5 receipt for the last measured line).

## Verification (orchestrator re-runs)
| check | result |
|---|---|
| `bash scripts/check_roadmap_diff_additive.sh --self-test` | rc 0, `15/15 rows` — row 9: the real `6d9ba274a` re-serialisation reads `reserialised=361`, trimmed → PASS |
| `bash scripts/check_roadmap_diff_additive.sh` (origin/main..HEAD) | rc 0, `roadmap-diff: base=715 head=716 added=1 lifecycle=0 reserialised=0 deleted=0` |
| `pmat analyze complexity scripts/lib/roadmap_diff.py` | max cyclomatic 7; one cognitive-15 warning left (`classify_pair`, 17); the pre-commit gate accepted the file after the decomposition (it refused the worker's 30/25 version) |
| `bashrs lint scripts/check_roadmap_diff_additive.sh` | 0 errors |
| `check_guards_are_wired.sh` · `check_workflow_env_defined.sh` · `check_pr_review_wiring.sh` · `check_roadmap_completion_is_cited.sh` | rc 0 each |
| `pv validate contracts/apr-roadmap-additive-diff-v1.yaml` · `check_contract_test_binding.sh` · `check_contract_enforcement.sh` · `pv lint` | rc 0 · rc 0 · rc 0 · 0 errors 0 warnings |

## Mutation (RED, then the remedy)
Re-serialised `docs/roadmaps/roadmap.yaml` pmat-style on the working tree (every entry re-dumped, `phases: [] / subtasks: [] / estimated_effort: null / labels: []` materialised — 1,109 insertions, 1,427 deletions) and committed it: `bash scripts/check_roadmap_diff_additive.sh` → rc 1, the RED footer names the remedy. Restored the tree (HEAD back to `21d467b5a`, guard PASS). The remedy on the real defect is self-test row 9 (`reserialised=361` → `python3 scripts/roadmap_trim.py` → PASS).

## Jidoka
- **A gate that could not fail (found by its first CI run, #2987 run 33991535406).** CI checks out at fetch-depth 1 and fetches origin/main at depth 1, so `git merge-base origin/main HEAD` prints nothing; the guard ran with an empty base, compared the head against itself (`base=807 head=807 added=0`) and would have passed any diff — it only went RED because of the duplicate below. Fix: `resolve_base` — merge-base when resolvable, else the PR merge commit's first parent (read off the commit object with `cat-file -p`; `rev-list --parents` hides parents behind a shallow graft) or the origin/main tip, and a non-merge shallow head is exit 2. Case-table rows 13–14; reproduced with a depth-1 clone of `refs/pull/2987/merge`: PASS via the first parent.
- **A third checkout shape (queue run 34002682350).** On `merge_group` the queue's temporary head is a single-parent, squash-shaped commit whose parent is the origin/main tip; the resolver had accepted only merge commits and refused it (exit 2, job red, #2987 dropped from the queue). Now: a single-parent head whose parent IS the origin/main tip resolves to that parent (row 15); any other single-parent head is still refused (row 14). Reproduced with a depth-1 clone of `gh-readonly-queue/main/pr-2987-…`: PASS, `base=aa5aa3330 (single parent == origin/main tip)`. Three CI red runs, three checkout shapes (PR merge ref, branch head, queue squash head) — the case table now carries all three.
- **A pre-existing duplicate id on main.** `PMAT-966` was minted by two sessions (#2872's spec ticket and #2871's harness ticket) and the guard refused every PR for it. Rule change: a base-side duplicate is baselined and named (`known duplicate-id`, row 10); a duplicate that grows at head is the violation (row 11); removing the later copy is the remedy and passes (row 12). The dedup of PMAT-966 itself is a separate roadmap PR (re-mint #2871's copy).
- The pre-commit complexity gate refused the worker's `roadmap_diff.py` (cyclomatic 30, cognitive 25); decomposed into 20 helpers in the same commit, never `--no-verify` (the mutation commit above used `--no-verify` deliberately and was discarded). #2526 (the gate measures file totals) is on the DAG under class #2879 (T-2/G-8).
- `pmat hooks install --strict --force` fails in a worktree; `Pmat-Ticket:` trailers written by hand.
- Two sonnet workers in this row's neighbourhood (C0-5 P_1/P_3, G-6 P_1) hit maxTurns at 40; the contract phase was done directly.

## Gaps
- README contract count: 1808 → 1809 after rebasing onto C0-5 (both PRs add one contract; the count is a serialization point, #2630 / P-0.4).
- Receipt for this PR: advisory, not produced (driver A1); this is the first PR the base-owned `pr-review-quorum.yml` will judge once C0-5 is on main.

## Merge evidence (orchestrator, after the fact)
- Merged as `65680cdf8` (#2987, the STEP B batch G-6 → G-4 → SPEC-1.6 → C0-7) at 2026-09-06T02:37:35Z through the merge queue; queue CI run 34005252966: `ci / gate` and `workspace-test` green; PR run 34003610208: every gate leg green, including the `guard-runner-labels` steps this row adds.
- The first CI run of #2987 (33991535406) found the guard comparing HEAD to itself in the shallow checkout and refusing every PR over main's pre-existing duplicate `PMAT-966`; both fixed at e2e3492b7 (rows 10–14, 14/14), and the guard passed on the next run (34003610208) and on the queue run (34005252966).
- Every A_i was re-run by the orchestrator on the batch tip before the push (PR body, "Acceptance commands" table) and the row's registered mutation shown RED then restored GREEN there.

## Verdict
DONE (`status: complete`).
