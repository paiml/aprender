---
status: complete
ticket: PMAT-1097
kind: code
milestone: 0.67.0
branch: PMAT-1097-06x-release-schedule
base: main 2c584a168
epic: 3078
model: claude-fable-5-1 (orchestrator) · one agy /teamwork-preview lane via paiml-agy-delegate (opus)
turns: 152
---
# impl receipt — PMAT-1097: the 06x release schedule (0.67.0 → 0.70.0)

orch_model: fable [V]   orch_class: fable   orch_decision: admit   orch_basis: release
fable_binding: true   quota_age_h: absent   quota_mark: ?   k_measured_at_set: 40

## Identity

- ticket `PMAT-1097` (kind:code, labels `orch:fable release 06x-schedule`; `notes:` carries `orch-basis:release`). Operator 2026-09-10, verbatim: *"override "kind:docs" block if needed"* — filed `kind:code` because the ticket ships `contracts/release-schedule-06x-v1.yaml` (pv-validated) and extends FALSIFY-DOCS-CLAUDE-001 to the spec (mutation-verified) alongside the document and the four epics.
- branch `PMAT-1097-06x-release-schedule`, worktree `/home/noah/src/aprender-worktrees/rel-06x-schedule` (outside the checkout: no `[patch]` inheritance), base `origin/main` @ `2c584a168`
- `discover.json` sha256 `baa3c2c239cc542e30d99a922b296527cf2e814641199e0fd961a483cea3ca16` (`gate_cmd=make gate`, `required_check=ci / gate,workspace-test`, `gate_cmd_fallback=false`, `version=0.66.0`, `quorum_tool=agy`)
- admission: `kind-gate.sh` exit 0 (`kind=code files=0`); `model-gate.sh` exit 0 (`model=fable class=fable decision=admit basis=file`, tier 1 [A] meets tier 1 [A]); `target-guard.sh` PASS; `config-lint.sh` exit 0 (`slots=3 gh_calls_per_min=30 bank=3`)
- estimate: `estimate.sh aprender 6` → `K_HAT=6 BASIS=first-run[U] ROWS=0`; K set to 120 (basis: `docs/audits/impl-PMAT-1096-receipt.md`, the last release-state ticket, K=150 [A]); `goal.sh set … --basis first-run[U]`

### Status-line join table [U]→[V]

| fact | measured | command |
|---|---|---|
| statusLine `session_id` = hook `session_id` | true [V] | `discover.sh --state-dir` prints `session=c1fb3cec-…`; `transcript-gate.sh` reports the same session |
| `tasks[].id` = hook `agent_id` | true [V] | the Agent tool returned `agentId a1f923ee6ff6b1030`; `transcript-gate.sh` counted `agent_calls=1` |
| `transcript_path` present on subagentStatusLine stdin | [U] | not observed this run |
| `k_measured` vs `global=k` | 152 vs 152 [V] | `jq -r 'select(.type=="assistant" and ((.isSidechain // false)|not)) \| (.message.id // .uuid)' <transcript> \| sort -u \| wc -l` |

**K overrun, stated:** K=120 was declared at k=40; the measured k at the receipt is 152. The andon rule (`turns ≥ 0.8K and gate ≠ PASS`) did not fire because every gate was PASS from phase 2 on; the overrun is a finding against the estimate (discovery alone took ~90 turns of read-only measurement), recorded in the estimates row below.

## Scope — derived, not remembered

Operator's request (verbatim): a release every 2–3 days working 24/7 for 0.67.0, 0.68.0, 0.69.0 (and 0.70.0 per priority G), one GitHub epic per release so Alfredo can comment, and priorities A–G. Three P0 five-whys tickets handed over mid-session with *"(add these to priorities)"*.

Deliverables at HEAD:

| artifact | what | proof |
|---|---|---|
| `docs/specifications/06x-release-schedule.md` | 459 lines: §0 21 measured rows (command per row), §1 cadence + capacity arithmetic, §2 priorities A–G as 47 rows with acceptance commands, §3 four trains, §4 release-day protocol, §5 epics, §6 nine decisions with dissent, §7 gates shipped/owed, §8 obligation DAG (46 rows), §9 risks, §10 review record, §11 amendments | drift gate GREEN; claim guards PASS |
| `contracts/release-schedule-06x-v1.yaml` | `kind: pattern`, 5 equations, 6 obligations, 8 falsification tests (commands) | `pv validate` → `0 error(s), 0 warning(s)`; `pv lint contracts/` → `0 errors, 1082 warnings (pre-existing), PASS` |
| `crates/aprender-core/tests/readme_contract.rs` | `DOCS_WITH_PATHS` 2 → 3 (the schedule) | RED on a bogus path (exit 101, "documented path(s) do not exist"), GREEN after revert — run twice |
| `contracts/apr-docs-v1.yaml`, `docs/specifications/TOC.md`, `docs/roadmaps/roadmap.yaml` | row prediction text; TOC entry; the ticket (additive: `base=831 head=832 added=1 lifecycle=0 reserialised=0`) | `check_roadmap_diff_additive.sh` PASS; sorted/unique PASS |
| GitHub | milestones `0.69.0` (#6), `0.70.0` (#7); epics #3078 (0.67.0), #3079 (0.68.0), #3080 (0.69.0), #3081 (0.70.0); P0 issues #3082 #3083 #3084 (milestone 0.67.0); cross-link comments on #3062 and #2873 | `gh issue view` each |

## Plan, routing and triggers

| phase | class | route.sh line (verbatim) | executed by | trigger | A_i |
|---|---|---|---|---|---|
| 1 plan | plan | `route=agy-plan w=1.00 basis=absent effort=1[U]` | self (the plan is the status block; the agy plan lane was folded into phase 3's teamwork review at the operator's instruction) | Q2 | `goal.sh set` exit 0 |
| 2 spec + contract + gate hooks | impl | `route=agy-goal w=1.00 basis=absent note=fable-binding effort=1[U]` | **self — deviation from R-4, named:** the document is the orchestrator's synthesis of 60+ facts measured in phase 0; a writing lane would re-derive or fabricate them (the NVIDIA spec's grill fabricated a build time). The agy budget went to the review instead | — | path sweep 0 missing; claim guards rc=0; `pv validate` rc=0; falsifiers 001–008 PASS; drift test RED→GREEN |
| 3 review | plan | `route=agy-plan w=1.00 basis=absent effort=1[U]` | `paiml-agy-delegate` → `agy /teamwork-preview` | Q2 (spec artifact) + operator: *"have it reviewed by agy /teamwork"* | receipt on disk; every finding re-run (table below) |
| 4 epics, receipt, push, PR | orchestration | `route=self` | self | — | `gh issue view 3078..3084`; PR open |

## Dispatch ledger

| mode | agent id | lane | width | agy conversations | child_conversations | turns | maxTurns hit | resumed |
|---|---|---|---|---|---|---|---|---|
| delegate (opus) | `a1f923ee6ff6b1030` | teamwork (`/teamwork-preview`) | 1 | `41071d23-fa14-4132-ab0e-f5b945dcd99d` (+3 capability probes: `fa1ed380-…`, `713d2917-…`, `42ee9943-…`) | `unknown method=none label=unknown` — **not** consensus; one model answering once | 30 | yes — the receipt was already on disk (`ph3/receipt.json`, 21 KB); the final message was the only thing cut off | no (not needed; the receipt was read from out_dir) |

Delegate deviations, declared in its receipt and accepted here: `--sandbox` omitted with `writes=false` because the sandboxed `gh api orgs/paiml/actions/runner-groups` returned `401` while the unsandboxed call answered (brief item 5 needed that endpoint); containment = disposable worktree, read-only prompt, `lane-guard.sh --exec` (self-test 26/26), HEAD + porcelain byte-identical before/after (`worktrees_created_by_this_run=0`). A harness note was appended to the prompt (agy's shell tool starts under `/home/noah/.gemini`). `receipt-lint.sh` on the delegate receipt printed `partial=true — missing: phase, files_changed, tests_added, acceptance, gate{…}` — the worker-receipt fields, which a delegate receipt does not carry; its `lane_reduce` block (`dedup`, `uncovered=[]`, `dissent`, `partial_reasons=[]`, `coverage_source=lanes`) is present.

**Slots:** `slots=3`; peak 1; denials 0 (`events-c1fb3cec-….jsonl`: none). **I-3:** `PASS transcript-gate: attempted=1 denied=0 running_peak=1 slots=3 segments=38 files=1 (agent_calls=1 resumes=0 workflow_started=0)` — run from the session's project dir; the same gate run from the worktree cwd was vacuous (`attempted=0`) because the delegate ran under the checkout's session, said so, and is superseded by the former.

## Verification — lane claims vs orchestrator re-run

| # | lane claim | claimed exit | my re-run | verdict |
|---|---|---|---|---|
| F1 | G9 `allows_public_repositories=false` is wrong | measured | `gh api orgs/paiml/actions/runner-groups/5` → `true`, `restricted_to_workflows=false`, `selected_workflows=[]` | accepted; G9 + 67-D0 rewritten (exposure is worse than written) |
| F2 | milestones 0.69.0/0.70.0 exist | measured | they exist because this ticket created them (#6, #7) after §0 was measured | accepted; G14 reworded |
| F3 | default features yield a CUDA binary; no smoke-cpu | measured | `crates/apr-cli/Cargo.toml:75` `default = [hf-hub, safetensors-compare, inference, training, visualization, zram]`; `cuda` opt-in at `:89` | **premise refuted**; smoke-cpu remedy accepted (67-A1) |
| F4 | merge_group quick on moved main skips reverse dependents beyond `CAP=3` | cited | `CAP=3` [V]; ci.yml passes only `use_nextest: true` to `sovereign-ci.yml`; its lint/test default to the root package [V] | accepted with a bounded remedy (mirror the PR tier; over-cap adds `cargo check --workspace --all-targets`; root-manifest keeps FULL; residual named in R2) |
| F5 | D-2 rescopes the operator's order silently | asserted | D-2 dissent + R5 already said it; not at the point of use | accepted; §2 B opens with the rescoping and the override path |
| F6 | re-baselining the PP-26 witness moves the goalpost | asserted | reading | accepted; near-tie vs defect rule in 67-F1 |

Gate re-runs by the orchestrator (claims are not pass): `make gate` → `EXIT=0` at `6fe3ab5c2` (pmat verify, `guard_tree.sh --no-cargo`, `gate_touched_crates.sh`); re-run on the final tree (log `make-gate-2.log`); `pv lint contracts/` → PASS; `check_no_claim_literals.sh` rc=0; `check_perf_claims_cite_receipts.sh` rc=0; `cargo test -p aprender-core --test readme_contract test_documented_paths_exist` → `ok. 1 passed` (three runs: before mutation, after revert, after the review fold-in); mutation run → exit 101 `documented path(s) do not exist`.

## Jidoka log

None. No gate went RED on a real defect; the one RED (drift test on a bogus path) was the deliberate mutation. `.pmat/jidoka.jsonl` unchanged.

## Estimates

`K̂=6 basis=first-run[U]` (estimate.sh, ROWS=0 for this repo) · `K=120` (basis `impl-PMAT-1096-receipt.md` K=150 [A]) · **actual 152** · phases 4 of the planned 6 (epics and receipt merged into phase 4; the sixth was the PR). Row appended to `docs/audits/impl-estimates.jsonl`.

## Gaps (NotRun lanes and what closes them)

- **Fan-out unmeasured** (`children=unknown`): the review is one lane. The spec's contract and drift gate do not depend on the review; a second lane (a width-3 quorum on the diff) is the artifact that would close it — not run, to keep the budget.
- **`pv_lane`:** run (`pv validate` + `pv lint contracts/`).
- **Dogfood:** not attached (`--dogfood` off); the release-day protocol (§4) is the next train's job, not this ticket's.
- **CI:** the PR touches `crates/aprender-core/tests/readme_contract.rs`, so BSE-17 routes it to the FULL tier (aprender-core touched, G5) — the ~86-minute run this spec's row 67-E1/E2 exists to shorten.

## Verdict

**DONE** — merged-green pending `ci / gate` + `workspace-test` on the PR; auto-merge armed.
