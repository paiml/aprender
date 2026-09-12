---
status: in-flight
merged: "[U] — this docs PR lands after the v0.67.0 tag; amended with the squash sha"
ticket: PMAT-1098
row: APR-RELEASE-001 §0 selector row 1 — the 0.67.0 train (session 1 of the spec; epic #3078)
epic: 3078
model: "orchestrator claude-opus-5 from mid-phase 4 (opus -> fable-5-1 at phase 4 entry, then fable-5-1 -> claude-opus-5 mid-phase; both movements are rows in docs/audits/impl-routing.jsonl via route.sh record-event)"
tokens_used: "orchestrator [U] (not instrumented); delegate ph4.teamwork 145,813 tokens / 42 tool uses; delegate ph4.g3ex 115,537 tokens / 38 tool uses (harness usage lines)"
wall_clock_s: "[U] — session spans a compaction; the train's own step timestamps are in rel-067-autopilot/STATUS"
orch_model: claude-opus-5
orch_class: orchestration
orch_decision: "self for every orchestration phase (queue, autopilot, tag, dry-run receipt, receipt); agy teamwork lane for the plan grill (Q2: spec artifact); agy goal lane (R-4, single module) for the G3.EX classifier fix, which failed isolation and was cherry-picked and re-verified"
fable_binding: false  # binding held while fable-5-1 drove phases 1-4; the harness moved the session to claude-opus-5 mid-phase-4 and the move is recorded, not asserted
quota_age_h: absent
quota_mark: U
k_measured_at_set: 32
---
# impl-PMAT-1098 — APR-RELEASE-001 session 1: the 0.67.0 train

Receipt for the first run of `docs/specifications/APR-RELEASE-001-train-and-build-kaizen.md`. The selector fired row 1 (last tag `v0.66.0` was 54 h old at HEAD `a1645cebf`, no `v0.67.0` tag, no SKIPPED record). Every number below names the command that produced it or carries `[U]`.

## Identity
ticket PMAT-1098 · kind code · labels `kind:code orch:fable release 0.67.0 epic-3078`, notes `orch-basis:release` · branch `PMAT-1098-apr-release-001` (this PR) plus `PMAT-1098-g3ex-needs-data` (#3163) and the bump `PMAT-1098-release-0.67.0` (#3145) · repo_root `/home/noah/src/aprender` for discovery (`discover.json`: gate_cmd `make gate`, gate_cmd_fallback=false, required_check `ci / gate,workspace-test` from branch-protection, quorum_tool agy, code_search `pmat query`, version 0.65.2 on the stale local branch — the release facts were read from `origin/main`).

Phase 0 gates: `kind-gate.sh PMAT-1098 … --base origin/main` → `kind=code files=0`; `model-gate.sh` → `model=fable-5-1 class=fable decision=admit basis=file` (tier 1 [A] meets tier 1 [A]); `config-lint.sh` → `slots=3 gh_calls_per_min=30 bank=3`; `target-guard.sh` PASS (the spec path is under repo_root).

Status-line join: `goal.sh show` → `{"ticket":"PMAT-1098","K":240,"K_hat":120,"basis":"first-run[U]","phase":4,"phases":5,"gate":"PASS","k_measured_at_set":32}`. `k_measured` at the spec's start: **312** (`jq -r 'select(.type=="assistant" and ((.isSidechain // false)|not)) | (.message.id // .uuid)' <transcript> | sort -u | wc -l`) — **K=240 was already crossed** when the operator issued `paiml-implement docs/specifications/APR-RELEASE-001-train-and-build-kaizen.md` (verbatim). The andon obligation (WIP committed, draft PR with the receipt-so-far) is this PR; the work continued on that instruction, and the gap is recorded here as a finding: the 312 turns include the fleet STOP/unclog work and PMAT-3138, which ran in this session under no `goal.sh set` of their own.

## Ground truth at HEAD vs the spec's §2 (recorded, then continued)
| Spec says | Measured at HEAD | Consequence |
|---|---|---|
| required check literally named `ci / gate` | branch protection requires TWO contexts: `ci / gate` and `workspace-test` (`discover.json` required_check) | both must be green; `gate` needs `[ci, workspace-test, mutants, guard-tree, guard-cargo]` — gpu-quick/cuda-unit are advisory |
| `ci / deep` green on the cut sha (T-1) | no job named `ci / deep` exists in `.github/workflows/ci.yml` (jobs: ci, workspace-test, guard-tree, guard-cargo, vendored-schemas, pr-review-*, gpu-touched, gpu-quick, cuda-unit, gate, mutants); `--no-default-features` occurs 0 times, doctests 0 times; FULL tier = schedule/workflow_dispatch + root-manifest rule (ii) (`scripts/ci_test_tier.sh`) | T-1 stand-in = the bump PR's FULL merge_group run + `dogfood.sh --phase pre-publish` + doctests and `--no-default-features` run by hand on the release commit (below); §8's "`ci / deep` red on a tag" can never fire until P1 lands — gap [U], named |
| `scripts/publish_cascade.sh` | absent; `scripts/cascade-publish.sh` (`--check` = dry-run report) and `scripts/cascade-drain.sh` (the real multi-pass publish), gated by `scripts/check_publish_preflight.sh` R1–R6 | the dry-run receipt is `--check` + preflight; the drain is the attended step |
| clean-room on the tag (T-3) | `paiml/infra` `.github/workflows/clean-room.yml` clones `--depth 1` the default branch; no ref/tag input; this repo has no tag-triggered clean-room run | stand-in: dispatch clean-room for aprender while `origin/main` HEAD == the release commit (the freeze holds), record the run id and the sha it built |
| intel 8 concurrent, memory-bound, 3.6 TB NVMe | `infra/machines/intel/forjar.yaml` description "32-core, 283GB, 3.6TB NVMe"; runner concurrency is not a single field (ci-runners.slice + runner-limits) — "8 concurrent" [U] | recorded |
| yoga RTX 4060 8 GB, 32 GB, 10G needs bolt.service | forjar.yaml lines 3, 297–301 confirm (10G is a Thunderbolt-tunnelled AQC113) | [V] |
| gx10 not a documented general runner | forjar.yaml declares ONE unit `actions.runner.paiml.gx10-blackwell` (labels gpu,gx10,cuda,blackwell,gb10); the box runs 6 units (blackwell, build, ephemeral, pool1, pool2, pool3) — pool1–3 added by SSH before the spec | [V]; the P2 infra PR must declare the live fleet; yoga likewise runs 5 units vs 1 declared |
| one PR in CI at a time (§3.4) | ruleset 17836320 `max_entries_to_build` = 3; the queue held 3–4 entries during this session | not changed this session — an operator decision, escalated below |
| ledger `docs/build-ledger/…` | absent | P0 row; the train record below is the first file |

## Plan (phases, routing, trigger)
Declared 5 phases at 06:36Z under the earlier plan (train #3127, fleet, post-train, cut, receipt); phases 1–3 landed before the spec arrived (train #3127 squash `cb829fcb9`, post-train, freeze, bump PR #3145). The spec's train is phase 4; the receipt is phase 5.

| phase | what | acceptance | route (route.sh verbatim) | trigger |
|---|---|---|---|---|
| 4.T-0 | bump PR #3145 merges behind #3136 #3046 #3068 #3139 | `gh pr view 3145 --json state -q .state` = MERGED and `git merge-base --is-ancestor <mc> origin/main` | `route=self w=100.00 basis=absent` | – |
| 4.T-1 | deep stand-in on the release commit | bump PR merge_group run FULL tier green; `cargo test --workspace --doc --exclude aprender-gpu --exclude aprender-cuda-edge --exclude aprender-compute` and `cargo check --workspace --no-default-features --locked` exit 0 in the release worktree | `route=self w=100.00 basis=absent` | – |
| 4.T-2 | dogfood | `dogfood.sh --phase pre-publish` VERDICT GO (autopilot `dogfood`); G3.EX `examples.tsv` trailer with fail=0 timeout=0 after the needs-data triage; G3.CB apr-cookbook #435 names 0.67.0; G3.RN CHANGELOG [0.67.0] non-empty | `route=self w=100.00 basis=absent` | – |
| 4.T-3 | promote | `gh release view v0.67.0`; `check_release_assets.sh v0.67.0` 16/16; `check_publish_preflight.sh` PASS; clean-room run at the release sha | `route=self w=100.00 basis=absent` | – |
| 4.T-4 | publish (ATTENDED) | `cascade-publish.sh --check` report written; NO drain run by this session; attended minutes [U] | `route=self w=100.00 basis=absent` | §3.7 |
| 4.grill | plan grill on the spec mapping + the G3.EX question | delegate receipt read; every finding re-checked | `route=agy-plan w=1.00 basis=absent effort=1[U]` | Q2 (spec artifact) |
| 4.g3ex | G3.EX classifier learns `needs-data` (#3163) | `bash scripts/dogfood_examples.sh --selftest` exit 0 with the nodata rows; mutation RED | `route=agy-goal w=1.00 basis=absent note=fable-binding effort=1[U]` | R-4 single module |
| 5 | train record + receipt + §7 report | `receipt-lint.sh`; `status-lint.sh` | `route=self w=100.00 basis=absent` | – |

phase-boundary line: `phase-boundary: ticket=PMAT-1098 phase=4 orch_model=fable-5-1 change=opus->fable-5-1 recorded=1`.

## Dispatch ledger
| dispatch | mode | agent id | lane / width | agy conversations | turns / maxTurns | resumed | outcome |
|---|---|---|---|---|---|---|---|
| PMAT-1098/ph4.delegate teamwork width 1 | paiml-agy-delegate (opus) | a2dae59cb33316b2d | teamwork, 1 lane, model_measured gemini-3.1-pro-high, 183 s | ec579c21-d691-45e0-8fdc-0b8d6154c0c5; child_conversations null (fanout `children=unknown method=none` — the third teamwork dispatch on this host with no measurable children: ONE model, not agreement) | 42 tool uses | no | verdict `do-not-implement-as-written` (schema has no `amend`); findings re-checked below; lane clone left byte-identical, no KEPT, no LANE BLIND |
| PMAT-1098/ph4.g3ex delegate goal width 1 | paiml-agy-delegate (opus) | a3e4e7b9c1b107a79 | goal, writes=true, 1 lane, model_measured gemini-3.1-pro-high, 449 s | 5aa7d8b2-11d6-4ad6-802a-8630b43abd45; child_conversations 1 | 38 tool uses | no | agy-lane exit 3: LANE ISOLATION VIOLATED ×2 — (a) `refs/heads/feat/linfa-burn-pareto-tickets` created at `f7dfd9561` by another actor in the main checkout (mtime 13:42:30 local; not the lane), (b) the lane wrote its SKILL.md hunk into the shared checkout as well as its worktree. Its detached commit `cf655b7cd` was cherry-picked (`7dfeed33e`), the README row added, author reset → `85bc96650`, PR #3163. R-4 fallback to sonnet-worker was not needed: the work product existed and every claim was re-run |

Slots: never more than 1 Claude subagent live (both dispatches sequential; `live=0` counted before each); `denied=0`; `stalled=1` (session count before the spec). I-3 line: [pending `transcript-gate.sh` at the receipt's final amendment].

## Verification table (claimed vs my rerun)
| what | claimed | my rerun |
|---|---|---|
| teamwork finding 1: FULL tier ≠ `ci / deep` (no `--no-default-features`, examples built not run, feature matrix contested) | lane: measured | `grep -c -- '--no-default-features' ci.yml` = 0; doctests 0; ci.yml:542 builds `--examples`; feature-gated suites do run (ci_test_tier.sh:219) — first two conjuncts TRUE, third partial |
| teamwork finding 2/3: an exit-1 G3.EX recorded green is a waiver (§3.8); any panic/build defect → SKIPPED (§4, §3.1) | lane: cited | spec lines re-read: §3 rule 8 "no waivers", §4 "Any step RED → SKIPPED". Applied: the classifier is FIXED (#3163) and the release-grade G3.EX is re-measured with it on the release commit — a corrected instrument, not a waiver; any `fail`/`timeout` row that survives the triage on a quiet host makes T-2 RED and the train SKIPPED |
| teamwork finding 4: no tag-triggered clean-room | lane: measured | no workflow in `.github/workflows` has a `tags:` trigger; clean-room lives in paiml/infra (clones main HEAD) — stand-in recorded above |
| teamwork finding 5: `scripts/publish_cascade.sh` absent | lane: measured | `ls scripts/publish_cascade.sh` → absent; `cascade-publish.sh`, `cascade-drain.sh` present |
| teamwork finding 6: train record shape satisfies §3.6/§0 | lane: cited | agreed; file written at phase 5 |
| goal lane: selftest exit 0 with nodata rows | lane: outcome=achieved, no command output returned | `bash scripts/dogfood_examples.sh --selftest` rc=0; `class: nodata -> needs-data (needs-data|1|Model not found at ../tiny-model-ground-truth/models/qwen2-0.5b-int8.apr)`, `cite: the three skip classes cite a line`, `trailer: summary counts`, `rows: every row carries one of the six classes` all PASS |
| goal lane: mutation proves NEEDS_DATA_RE | not returned | `NEEDS_DATA_RE='zzz-never-matches'` → rc=1, `FAIL class: nodata -> needs-data`, `FAIL trailer: summary counts`; restored → rc=0 |
| goal lane: bashrs no new findings | not returned | origin/main: 0 error 16 warning 60 info; branch: 0 error 16 warning 63 info; `comm` on normalised lines shows no new warning text |
| goal lane: "committed on the current branch" | claimed | FALSE as stated — branch head unchanged; the commit lived on the lane worktree's detached HEAD (`git branch --contains cf655b7cd` empty) until cherry-picked |
| G3.EX sweep on the train head (lambda, launched 07:03Z, `--out …/dogfood-examples`) | #3136's body: "classified three rows outside pass/args/hw" over 981 targets | at 937/981 rows: pass=542 fail=173 timeout=18 needs-args=26 — the body read a partial sweep; corrected by comment on #3136. Re-runs with stderr kept: `bench_bpe` rc=2 Usage (needs-args after #3136), `gpu_fallback_dogfood` rc=1 "Model not found at ../tiny-model-ground-truth/…", `bench_matmul_only` rc=1 "Failed to open /home/noah/models/TinyLlama-…gguf: No such file or directory" — needs-data, the class #3163 adds |

## Build ledger (§3.6 / §5 P0) — 182 records, written this train

`docs/build-ledger/2026-09-12/`, one JSON per (sha, host, job), from
`gh api repos/paiml/aprender/actions/runs/<run>/jobs`. `peak_rss_mb` and
`free_disk_gb` are `null` with `[U]` in `unmeasured[]`: the REST API does not
expose them and an absent measurement must not read as a zero. 182 > the 20 that
§8 requires before P0 stops being a stop condition.

| job | intel p50/p95 | gx10 p50/p95 | yoga p50/p95 |
|---|---|---|---|
| workspace-test | 1790 / 4148 s | 790 / 790 s | 259 / 3167 s |
| guard-cargo | 1201 / 1820 s | 422 / 516 s | 830 / 1046 s |
| guard-tree | 604 / 879 s | 192 / 213 s | 379 / 379 s |
| queue wait p95 | 2995 s | 233 s | 1654 s |

Per-run required-check wall clock by the host mix the run touched: gx10 only
9.4 min (n=4); gx10+yoga 11.3 min (n=4); gx10+intel 42.6 min (n=10);
gx10+intel+yoga 31.6 min (n=11). Every run that touched intel cost 32-43 min.

**§1 coupling is no longer `[U]`.** p95 required-check wall clock = 72.3 min over
29 runs, so `max PRs per train = 72 h / 72.3 min = 59.7`. §8's "stop cutting
trains below 10 PRs/train" does not fire.

## Queue actions taken on this train (§3.4 heijunka)

| action | why | evidence |
|---|---|---|
| dequeued #3046 | its merge-group CI was RED on `guard-tree` (`check_roadmap_diff_additive.sh`, 44 checks / 1 failed) and the entry held a build slot it could never use | run 34689617187; `dequeuePullRequest` mutation |
| cut #3046 from the 0.67.0 CHANGELOG | §4: scope is assigned to a train after the fact, and a PR that cannot precede the cut is not in it | bump commit e1548f34b |
| moved #3046 to milestone 0.68.0 | same | comment 5645837638 |
| cancelled CI on #3163 and #3164 | six aprender `workspace-test` jobs were live on intel at once against §3.4; two of them were mine and were not on the train | runs 34692405792, 34692277291 |

## T-2 evidence for this cut

| gate | verdict | how |
|---|---|---|
| G3.CB apr-cookbook names 0.67.0 | **PASS** | `gh api repos/paiml/apr-cookbook/commits?since=2026-09-10T04:55:52Z` returns "docs: aprender 0.67.0 - what changes for cookbook users (Refs #434) (#435)" |
| G3.RN CHANGELOG `[0.67.0]` non-empty | PASS | bump worktree, 4 sections |
| G3.EX every example runs | in flight | see the sweep section |

## Jidoka
- G3.EX classifier gap (173 rows read as defects): fixed in #3163, held until post-tag (`automerge-held.txt`); the release-grade sweep re-runs with it.
- agy goal lane isolation exit 3 (two causes above); the stray lane worktree under `fix-g3ex/.claude/worktrees/` was removed; the shared-checkout SKILL.md dirt reverted (byte-identical to the commit).
- A concurrent actor created `feat/linfa-burn-pareto-tickets` in the main checkout during the lane window — not this session's; named so the next reader does not attribute it to the lane.
- `pmat hooks install --strict --force` fails in a linked worktree ("Not a directory (os error 20)"): `.git` is a file there; the shared hooks dir carries `pre-commit` only, so `Pmat-Ticket:` trailers are written by hand.

## Estimates
K̂=120 [U] (estimate.sh: ENV — the 32 rows for repo=aprender in `docs/audits/impl-estimates.jsonl` carry no `unit`, none enters a total); K=240; k_measured=312 at the spec's start, final value in the last amendment. Delegate turns: 42 and 38 tool uses.

## Gaps (every NotRun lane and the artifact that closes it)
- `ci / deep` (P1) — closes the T-1 stand-in and §8's dead stop condition.
- P0 ledger (`make build-report`, ≥20 records) — this train writes record 1.
- P2 infra PR declaring yoga/gx10's live runner units; yoga-eph `clean-room` restore goes through it, not SSH.
- Ruleset `max_entries_to_build` 3 vs §3.4 — operator decision.
- Attended cascade minutes — measured only when Noah runs T-4.
- `pv` lane: NotRun (no contract touched by this PR).

## Verdict
IN-FLIGHT — amended at phase 5 with the train record, the §7 report, the I-3 line and the final k_measured.
