# impl-PMAT-3225 — APR-RELEASE-001 §5 P0·Instrument

## Identity

| Field | Value |
|---|---|
| ticket | PMAT-3225 (`kind:code`, kind-gate PASS) |
| target | `docs/specifications/APR-RELEASE-001-train-and-build-kaizen.md` (target-guard PASS) |
| branch | `PMAT-1108-build-report` off `origin/main@fa6e35f23` |
| HEAD | `3d819b8e0` |
| PR | #3271, milestone 0.69.0, auto-merge armed |
| discover.json sha256 | `e717fdd1a1269e52…` |
| session model | `opus-5`, class `opus`, model-gate `decision=admit basis=file` |
| gate_cmd | `make gate` (`gate_cmd_fallback=false`) |
| required_check | `ci / gate,workspace-test` |

## Selector — which row this is

APR-RELEASE-001 §0, first match wins.

| Row | Evaluated | Result |
|---|---|---|
| 0 — fleet under-utilised under intel pressure | packer sample 13:26:35Z: `jobs_running=0 jobs_waiting=0` ⇒ no intel pressure by §1 | no match |
| 1 — ≥ 48 h since last tag | `v0.67.0` at 2026-09-13T06:29:54Z = **31 h** | no match |
| 2 — first §5 row whose *Done* test fails | P0·Pack and P0·Reap need 10 trains of trailing data (unmeasurable, no missing mechanism). **P0·Instrument**: `make build-report` absent on `main` | **match** |

Row 0 was evaluated from the packer's own committed samples, not from `gh api …/actions/runners` — that call is refused here.

## Plan and routing

| Phase | What | route.sh (verbatim) | Executor actually used |
|---|---|---|---|
| P_1 | `build_report.sh` + `make build-report` | `route=agy-goal w=1.00 basis=absent note=fable-binding effort=1[U] bucket_collision=true` | `subagent:sonnet` (`paiml-impl-worker`) — R-4's named fallback. The `agy-goal` route needs `writes=true`, which the operator has refused durably; not retried. |
| P_2 | the binding-check correction + the four review defects | `route=agy-goal w=1.00 basis=absent …` | `self` — same refusal; findings arose during my own verification. |
| P_3 | pre-PR review | `route=agy-quorum w=1.00 basis=absent effort=1[U]` | `paiml-agy-delegate`, quorum width 3, `writes=false` |

## Dispatch ledger

| # | Mode | Agent id | Turns | maxTurns hit | Resumed |
|---|---|---|---|---|---|
| 1 | `paiml-impl-worker` sonnet, ph1 | `a819b3962aaf1951f` | 43 tool uses / 127,706 tok | **yes (40)** | no — A_1 verified green without it |
| 2 | `paiml-agy-delegate` opus, ph3.delegate, quorum w=3 | `ac4c53585f2577ff5` | 45 tool uses / 131,197 tok | **yes (30)** | no — lane artifacts read from disk |

Both stopped at their turn limits. Neither was resumed: the worker's claim was re-verified directly, and the delegate's three lanes had already written `lane-{1,2,3}.json` (status `SUCCESS`, turns 2/1/1) to `out_dir`, which is where the verdicts were read.

**Slots: `running_peak=1` of 3.** `transcript-gate.sh`: `attempted=14 denied=0 stalled=0 running_peak=1 slots=3 segments=377 files=11 (agent_calls=11 resumes=3)` — PASS.

Run inside the worktree it reads `attempted=0` and says so itself ("vacuous but honest"): its universe is the project dir of the cwd, and the subagents belong to the main checkout's session. The non-vacuous run above is the one that counts.

**Denials: 1.** `goal.sh set --ticket PMAT-3225` → `one ticket per session: PMAT-1098 was set here`. By design (one session, one goal). The statusline goal stayed PMAT-1098; PMAT-3225 is the row ticket and is the `Pmat-Ticket:` trailer on all four commits. Not retried.

## Verification — claimed vs my own re-run

| Claim | Source | My re-run | Agrees |
|---|---|---|---|
| `check_build_report.sh` 35/35 | worker | 35/35, exit 0 (was written as 16/16 when the table had 16 rows; re-counted at adoption, `grep -cE '^\s+ok '` on the self-test output) | yes |
| report runs, 1092 records | worker | 1092 valid / 1084 job / 8 skipped, 0.27 s | yes |
| every p50/p95 | `build_report.sh` | independent python reimplementation, byte-for-byte identical on all 9 figures | yes |
| byte-identical across runs | worker | `cmp` of two runs: identical | yes |
| `bashrs` clean | worker | 0 errors, 192 warnings | yes |
| non-object JSON → exit 1 | 3/3 lanes | **exit 5** (jq abort) — lanes right, now fixed | defect |
| empty/whitespace → exit 1 | 3/3 lanes | **exit 0**, silently dropped — lanes right, now fixed | defect |
| p7 of 1..100 = 7 | 1/3 lanes | **8** (float ceil) — lane right, now fixed | defect |
| `build_report.sh` has a caller | my own guard asserted it via a Makefile grep | `check_guards_are_wired.sh`: unwired 3 → 4 — **my assertion was weaker than the tree's standard**, now fixed | defect |

## Quorum (ph3)

Width 3, `writes=false`, lanes `SUCCESS`. Consensus on the two contract violations was **3/3 independent, `grounding=measured`** — not review prose. The planted trap (which of five claims looks mechanical and is not) was the binding-check claim; a lane settled it by citing `scripts/pr_review_quorum_arm.sh`, which already documents that branch protection names `ci / gate` and ruleset 13878864 names a bare `gate` — so both spellings are one check. That citation is why the required set now lists all three ledger names.

Lane verdict fields did not populate the schema's `verdict` key; findings were read from `.structured_output.findings`. `lane-reduce.sh` was not run (the delegate stopped first), so there is no reduced consensus artifact — the three lane JSONs were read directly. **Gap, named below.**

## jidoka

| Defect | Owner | Five-whys terminus |
|---|---|---|
| `has("total_s")` aborts jq on a non-object | this diff | the guard's only malformed fixture was plain text, which jq rejects cleanly — the contract looked kept because the fixture could not distinguish "rejected" from "crashed" |
| empty file silently dropped | this diff | `xargs -0 jq -c .` exits 0 and emits nothing; nothing compared record count to file count |
| p7 = 8 | this diff | `ceil` over IEEE754: `(7/100.0)*100 = 7.000000000000001`. k=50/95/100 all land on exact binary fractions, so the case table could not reach it |
| `build_report.sh` unwired | this diff | I asserted "has a caller" with a Makefile grep; the tree's standard is "a non-comment line in a workflow INVOKES it", because Makefile-only means `make tier3`, which CI does not run |

## Gate

`make gate` reports 4 failures. Each was run on this branch **and** on a branch without these changes, in the same minute:

| check | this branch | without changes | classification |
|---|---|---|---|
| `check_baseline_ratchets` | rc=1 | rc=1 | env: baselines under pmat 3.39.0 / bashrs 7.0.1, runner has 3.40.0 / 7.4.1 |
| `check_complexity_ratchet` | rc=1 | rc=1 | env: same instrument drift |
| `check_silicon_coverage` | rc=0 now, rc=1 15 min earlier | rc=0 | **wall-clock dependent** — verdict moved with no commit between |
| `check_guards_are_wired` | rc=0 | rc=0 | was mine (3 → 4), fixed |

**Zero of the four are caused by this diff.**

## Estimates

`estimate.sh aprender 2` → **exit 2**: "29 measured rows for repo=aprender and none enters a total — the writer and the reader disagree". Every aprender row in `docs/audits/impl-estimates.jsonl` lacks the mandatory `unit` field, so the pool can never produce a K̂ for this repo. basis is `first-run[U]` by force, not by novelty. **Gap, named below.**

## Gaps — each with the artifact that would close it

| Gap | Closed by |
|---|---|
| `lane-reduce.sh` never ran; no reduced consensus artifact | re-running the delegate to completion, or `lane-reduce.sh` over the three lane JSONs |
| `estimate.sh` cannot pool aprender rows | backfilling `unit:"turn"` on the 29 rows, or a writer fix — a facility with a ledger and no reader |
| `check_silicon_coverage` verdict moves on the wall clock | its own ticket; a required-path guard must not be time-dependent |
| §1 and §12.2.D still carry the `ci / gate` form of the throughput equation | a spec amendment on the APR-RELEASE-001 branch (#3268), now that 120 is measured |
| `pv` contract for this surface | `pv_lane=NotRun` — the §11.1 delta for the ledger-record schema waits on ONT-1 (`pv census`), epic #3269 |
| `peak_rss_mb` / `free_disk_gb` are `null` on every record | not the Actions REST API's to give; a runner-side collector, out of scope here |


## Machine-readable block (receipt-lint.sh)

```
orch_model: opus-5   orch_class: opus   orch_decision: admit
fable_binding: false   quota_age_h: 0   quota_mark: U   k_measured_at_set: 0

routes:
  ph1  class=impl        route=agy-goal    w=1.00  basis=absent  used=sonnet-worker  fallback=operator-refused-writes-true
  ph2  class=mechanical  route=agy-goal    w=1.00  basis=absent  used=self           fallback=operator-refused-writes-true
  ph3  class=review      route=agy-quorum  w=1.00  basis=absent  used=agy-delegate-quorum-w3

verification:
  cmd=bash scripts/check_build_report.sh  claimed_exit=0  rerun_exit=0  log_path=/tmp/claude-1000/-home-noah-src-aprender/b59147ae-201e-4299-9354-6a53738c822b/scratchpad/a.txt  sha256=65f5d3eba7b8e521
  cmd=bash scripts/build_report.sh        claimed_exit=0  rerun_exit=0  log_path=/tmp/claude-1000/-home-noah-src-aprender/b59147ae-201e-4299-9354-6a53738c822b/scratchpad/x.txt  sha256=87fcb41aea87861b
  cmd=make gate                           claimed_exit=1  rerun_exit=1  log_path=/tmp/claude-1000/-home-noah-src-aprender/b59147ae-201e-4299-9354-6a53738c822b/scratchpad/gate.log  sha256=d6c4c0562b54bc92

```

`quota_mark: U` — `quota.json` is absent, so route.sh printed `basis=absent` on every
phase and no quota-weighted comparison was possible. `k_measured_at_set: 0` because
`goal.sh set` was refused for this ticket (one session, one goal — PMAT-1098 holds it),
so no goal was ever declared for PMAT-3225 and the statusline measured none.

## Verdict

**PARTIAL(escalate)** — the row's mechanism is landed, verified and armed, and every defect found by the gate or the quorum is fixed. It is not DONE because two of the six DoD parts are open: `pv` contract `NotRun` (blocked on ONT-1), and the PR is not yet merged green on `ci / gate` + `workspace-test`.
