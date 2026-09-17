# impl-PMAT-3445 receipt — the milestone being cut is read at the cut

## Identity

| Field | Value |
|---|---|
| ticket | PMAT-3445 (GitHub #3445, P1, milestone 0.69.0) |
| kind | code (`kind-gate.sh PMAT-3445 docs/roadmaps/roadmap.yaml --base origin/main` → exit 0, `kind=code`) |
| branch | `PMAT-3445-t0-milestone-gate`, cut from `origin/main` @ `dc6fb687c` (the #3448 merge that filed the ticket) |
| discover.json sha256 (prefix) | `f1120b4c9bc805a9` — `gate_cmd=make gate`, `gate_cmd_fallback=false`, `required_check=ci / gate,workspace-test`, `contracts_dir=contracts`, `quorum_tool=agy` |
| orchestrator model | admitted `opus` (`model-gate.sh` → `model=opus-5 class=opus decision=admit`). An earlier Fable session was refused by the same gate (`model fable-5-1 not permitted`) and made no edit; the operator relaunched on Opus |
| phase boundary | `phase-boundary: ticket=PMAT-3445 phase=2 orch_model=opus change=none recorded=0` |

Status-line join (`[U]`, not measured in this run): statusLine `session_id` = hook `session_id`; `tasks[].id` = hook
`agent_id`; `transcript_path` present on subagentStatusLine stdin. NotRun, with no command executed for them.

## What landed

| File | Change |
|---|---|
| `scripts/check_milestone_cut.sh` | new gate: exit 0 only when the named milestone holds 0 open issues and PRs; 1 names each open item plus its carry/close remedy; 2 when it cannot judge |
| `docs/specifications/06x-release-schedule.md` | §4 step 1 reads the milestone at the freeze; step 4 re-reads it immediately before the tag (the binding read); §11 amendment row |
| `docs/specifications/APR-RELEASE-001-train-and-build-kaizen.md` | §4 T-0 and T-3 rows carry the same two reads; carry = re-milestone plus `slipped_from:` |
| `contracts/release-schedule-06x-v1.yaml` | equation `milestone_settled_at_cut`, RS0-INV-007, FALSIFY-REL-06X-009/010, qa_gate check |
| `docs/audits/impl-PMAT-3445-plan.md` | plan v2 plus the v1 grill record |

## Plan, routing, dispatch ledger

| Phase | Route (`route.sh`, verbatim) | Trigger | Executor | Outcome |
|---|---|---|---|---|
| P0 plan grill | `route=agy-plan w=1.00 basis=absent effort=1[U]` | Q2 (plan artifact) | `paiml-agy-delegate` (opus), quorum `grillme` width 3, writes=false | 3/3 `do-not-implement-as-written` (gemini-3.1-pro-high, gemini-3.8-flash-high, gemini-3.7-flash-high; author family claude); agy conversations `d22ad04f…`, `e8b9932a…`, `8959b9a4…`; children=3; `--concurrent-scope scripts/check_milestone_cut.sh` on every lane; no BLIND, no KEPT |
| P1 gate | `route=agy-goal w=1.00 basis=absent note=fable-binding effort=1[U] bucket_collision=true` | — | `paiml-agy-delegate` (opus), goal width 1, writes=true → **superseded**, then **direct** | the delegate hit maxTurns (30) and was not resumed; its lane (conversation `ba5e4680…`, gemini-3.1-pro-high) committed `d14914db8` in its own worktree under v1 semantics; the reducer read NO-VERDICT (goal output has no verdict field). Not merged: v1 design rejected, and `gh api --paginate` without `--jq` writes concatenated arrays that `json.load` rejects above 100 items. Rewritten directly for v2 (deviation from R-4, recorded) |
| P2 wiring | `route=self` (orchestration-class edits to specs and contract) | — | direct | all acceptance commands exit 0 (below) |
| P3 rehearsal + pre-PR review | `route=self` / quorum via `quorum-review.sh` | Phase 4 | rehearsal direct; review = the artifact `docs/audits/quorum-PMAT-3445.json` | see that artifact |

Slots and I-3: `transcript-gate.sh <session dir>` → `PASS transcript-gate: attempted=2 denied=0 stalled=0 running_peak=2 slots=3`.
(Run without the argument, it swept the worktree's project dir and reported a vacuous 0/0. Finding F-5.)

## Verification — claimed vs re-run by the orchestrator

| Acceptance | Lane/worker claim | Orchestrator rerun |
|---|---|---|
| `bash scripts/check_milestone_cut.sh --self-test` | goal lane: "exits 0" (v1 file, superseded) | exit 0, `self-test OK: 18 case(s).` (v2) |
| bare run = self-test | — | exit 0, same line |
| mutation (a) judge always passes | lane: "mutations turned self-test RED" (no output) | exit 1; RED S2 S3 S4 S12 S15 S16 |
| mutation (b) count cross-check deleted | — | exit 1; RED S6 S7 |
| mutation (c) `--paginate` dropped from the items read | — | exit 1; RED S15 |
| `bashrs lint --no-ignore --level error` | lane: "no errors" | exit 0 (plain lint: 0 errors, 26 warnings; precedent `check_reconcile.sh`: 0 errors, 76 warnings) |
| live `check_milestone_cut.sh 0.67.0` | lane: exit 0 | exit 0 — `0 open, 50 closed` |
| live `check_milestone_cut.sh 0.66.0` | — | exit 0 — `0 open, 10 closed` |
| live `check_milestone_cut.sh 0.68.0` (at `465cd96f2`) | lane: exit 1 naming #3091 | exit 1 — `#3091 issue [bug,P1]`, `#3450 pr [] release: 0.68.1` |
| live `check_milestone_cut.sh 0.69.0` | — | exit 1 — 48 open (milestone API `open_issues=48`) |
| `pv validate contracts/release-schedule-06x-v1.yaml` (pv pinned by `scripts/verifier_pin.sh`, 0.68.0) | lane 1: passes; lane 2: needs qa_gate wiring (both addressed) | exit 0, `0 error(s), 0 warning(s)`; `pv lint` PASS |
| FALSIFY-REL-06X-010 | — | exit 0 |
| `cargo test -p aprender-core --test readme_contract` | — | exit 0, 15 passed |
| `guard_tree.sh --dry-run --no-cargo` | — | prints `run: scripts/check_milestone_cut.sh` |
| `check_guards_are_wired.sh`, `check_no_timing_in_required.sh`, `check_assertions_exclude.sh`, `check_pass_grep_anchored.sh`, `check_sourced_libs_option_neutral.sh`, `check_bashrs_gate.sh` | lanes 2+3 predicted wired-guard refusal for a `release_tag.sh` (dropped) | all exit 0 |
| `cargo fmt --all -- --check`, `cargo test -p aprender-contracts --lib`, `cargo deny check advisories` | — | exit 0, exit 0 (1538 passed), exit 0 (`advisories ok`) |
| `make gate` (`gate_cmd`) | — | **NotRun locally**; CI's required checks `ci / gate` and `workspace-test` are the gate of record for this PR |

### Rehearsal (the ticket's falsifier)

A copy of `rel-068-1-autopilot/autopilot.sh:103-112` (the `tag` block, verbatim) with
`bash scripts/check_milestone_cut.sh "$MS_TITLE" … || die …` as its first line, against milestone 0.68.0.
`git tag`/`git push`/`gh release` were stubbed to a call log; every other `git`/`gh` call went to the real binaries.

- exit 1; STATUS: `STOP milestone 0.68.0 is not settled: #3450 pr [] release: 0.68.1;`
- write calls logged: **0**; `git ls-remote --tags origin 'v0.68.99*'` → 0 refs.
- #3091 was not named because between the `465cd96f2` read and the rehearsal, #3091 was **carried**:
  `2026-09-17T13:40:30Z demilestoned 0.68.0` / `milestoned 0.70.0` (issue events). This came after this run's
  warning on #3450. The rehearsal therefore ran on a real milestone holding exactly one open item and named it.

## Jidoka

| Phase | Defect | Owner | Whys |
|---|---|---|---|
| P0 | plan v1 admitted carried items that stay open, and proposed a dark `release_tag.sh` | PMAT-3445 (this plan) | the close step closes on `open_issues` → a comment-based carry leaves it non-zero → the plan read the ticket's "or carries `slipped_from:`" as a gate clause instead of the 06x step 1 re-milestone practice → the untracked autopilot was not read before planning → fixed in v2 by re-milestoning |

No gate went red in P1 or P2 after v2. `.pmat/jidoka.jsonl` is not a tracked file in this repository, so this row is its record.

## Estimates

K̂ = 24 (`basis=first-run[U]`). `estimate.sh aprender 3` exits ENV: 47 measured rows, none enters a total.
The ticket body's own Ĵ = 24 turns `[A]` was borrowed. K = 48. Actual: `k_measured` = 88 distinct main-thread
assistant message ids at receipt time, over the whole session, including the refused Fable turns before the
goal was declared. The overrun came from P0 (grill 3/3 against v1), the superseded goal lane, and the redesign.
Row appended to `docs/audits/impl-estimates.jsonl`.

## Findings and gaps

- **F-1 (gap, filed #3454, milestone 0.70.0).** The tag decision surface is an untracked per-train autopilot
  copy. This PR puts the read into both specs and the contract; no running autopilot calls it. The live 0.68.1
  train (pid 3021358, `autopilot.sh 3450 wait dryrun`) was warned on #3450.
- **F-2 (consequence).** Milestone 0.69.0 held 48 open items at the last read, and its cut is due
  2026-09-17T18:00Z. Where the gate is called, that cut stops until they are closed or carried.
- **F-3.** `pmat hooks install --strict --force` fails in a linked worktree (`Not a directory (os error 20)`,
  `.git` is a file). Commits ran the shared `core.hooksPath` pre-commit and carry `Pmat-Ticket:` by hand.
- **F-4.** The edit hook keys `active-ticket` by the session's cwd repository (`…/124170554`), not by the
  edited file's repository (`discover.sh --state-dir` in the worktree gives `…/3607376464`).
- **F-5.** `transcript-gate.sh` with no argument resolved the worktree's project dir, not the session's, and
  reported a vacuous 0/0. With the explicit session dir: attempted=2, running_peak=2.
- **F-6.** The goal-lane delegate hit maxTurns 30 while polling its lane. `lane-reduce` marks a goal lane
  NO-VERDICT/partial because the goal schema carries `outcome`, not `verdict`.
- **F-8 (environment, not this diff).** `guard_tree.sh --no-cargo` on this host: 62 checks, 2 FAIL, both
  `tool_version`. `scripts/cb200_baseline.txt` and `scripts/complexity_baseline.txt` were recorded under
  pmat 3.40.1, and this host runs pmat 3.40.2. `check_complexity_ratchet.sh` measured base `dc6fb687c` = 673
  and merge `ffc68fe47` = 673 functions over threshold, delta +0, same `.rs` universe. This PR changes no
  `.rs` file. Classified as a host instrument drift; CI's fleet pmat is the comparand of record.
- **F-7.** A peer session reported this worktree's base as predating #3448. Measured:
  `git merge-base --is-ancestor <#3448 merge> HEAD` → true.

## Verdict

`DONE` requires: merged green on `ci / gate` + `workspace-test`, and the quorum artifact
`docs/audits/quorum-PMAT-3445.json` agreed. At this commit the local gates above PASS, and the mutation was
observed RED. The `pv` contract and the invalidated spec claims ride in the same PR. Merge state is read from
the PR, not from this file.
