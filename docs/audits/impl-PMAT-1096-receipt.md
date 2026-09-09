---
status: partial
ticket: PMAT-1096
kind: code
milestone: 0.66.0
branch: PMAT-1096-release-0-66-0
base: main f34671a6b
epic: 2873
model: claude-fable-5-1 (orchestrator) · one agy quorum (width 3) via paiml-agy-delegate
turns: 62
---
# impl receipt — PMAT-1096: the 0.66.0 release cut

orch_model: fable [V]   orch_class: fable   orch_decision: admit   orch_basis: release
fable_binding: true   quota_age_h: 47   quota_mark: A   k_measured_at_set: 37

## Identity

- ticket `PMAT-1096` (kind:code, labels `orch:fable pp-066 release 0.66.0`; `notes:` carries `orch-basis:release`)
- branch `PMAT-1096-release-0-66-0`, worktree `/home/noah/src/aprender/.claude/worktrees/rel-066`, base `origin/main` @ `f34671a6b`
- `discover.json` sha256 `9fbca85b9c2970689c7ccfd0eabd8a0a2aba85bb94623e603164368cbe8a4d84` (`gate_cmd=make gate`, `required_check=ci / gate,workspace-test`, `gate_cmd_fallback=false`, `version=0.65.2` at discovery)
- admission: `kind-gate.sh` exit 0 (`kind=code files=0`); `model-gate.sh` exit 0 (`model=fable class=fable decision=admit basis=file`, tier 1 [A] meets tier 1 [A]); `target-guard.sh` PASS (ticket filed in the worktree's roadmap); `config-lint.sh` exit 0 (`slots=3 gh_calls_per_min=30 bank=3`)
- estimate: `estimate.sh aprender 6` → `K_HAT=6 BASIS=first-run[U] ROWS=0`; K set to 150 (basis: `docs/audits/impl-PMAT-742-receipt.md`, the last release-cut ticket, andon at K=150 [A]); `goal.sh set` recorded `k_measured_at_set: 37`

### Status-line join table [U]→[V]

| fact | measured | command |
|---|---|---|
| statusLine `session_id` = hook `session_id` | true [V] | `discover.sh --state-dir` prints `session=d8a83629-…` and `events-d8a83629-….jsonl` carries the same id |
| `tasks[].id` = hook `agent_id` | true [V] | events line `SubagentStart … agent_id=ab286307cd9f490cc` equals the Agent tool's returned `agentId` |
| `transcript_path` present on subagentStatusLine stdin | [U] | not measured this run (no subagent-statusline invocation observed) |
| `k_measured` vs `global=k` | 62 vs 62 [V] | the jq below over the session transcript |

```
jq -r 'select(.type=="assistant" and ((.isSidechain // false)|not)) | (.message.id // .uuid)' <transcript> | sort -u | wc -l
```

## Scope — what "0.66 work" is, derived not remembered

Milestone `0.66.0` (GitHub milestone 3, "PP-066 honest GPU release (epic #2873)") as the operator curated it:
2026-09-08T12:06Z #3022 milestoned 0.66.0 (with #2971); 2026-09-09T12:06Z every PP-066 row issue (#2890 #2891 #2893 #2904 #2905 #2906 #2908 #2909 #2910 #3002 #3013 #3018 #3019 #3024 #3045 #2869) demilestoned 0.67.0 → milestoned 0.68.0; 2026-09-09T12:20Z PR #3063 milestoned 0.66.0. Read back with `gh api repos/paiml/aprender/issues/<n>/timeline`. The v1.6/v2.1 PP-066 spec's TAG-0.66.0 row (rc1, four host receipts, R-5 assets) is therefore 0.68 scope; this cut is the narrow one.

| item | state at discovery | mechanism |
|---|---|---|
| #3022 F-1 `apr chat` demo-model substitution | PR #3050 open, `ci / gate` + `workspace-test` green, bare `gate` queued behind `mutants`, auto-merge armed | `gh pr checks 3050` |
| #2971 L0-1 GPU≠CPU on the 1.5B shape | fixed on main by #3026 (L0-1a) + #3032 (L0-1b); issue open | `gh pr list --search 2971 --state merged` |
| PR #3063 T0 un-dark `aprender-gpu`/`aprender-cuda-edge` in workspace-test | CI run queued (fleet saturated: 16/16 intel runners busy, shared across repos) | `gh api orgs/paiml/actions/runners` |

## Plan (phases, acceptance commands, routing)

| phase | what | A_i | mode | trigger |
|---|---|---|---|---|
| 1 | drain #3050 and #3063 through the fleet (two superseded runs on stale heads cancelled: 34352425473, 34368515383) | `gh pr view 3050 --json state -q .state` = MERGED and the same for 3063 | direct | - |
| 2 | close #2971 with a citation — only after a ledger quorum `agreed=true` (T-6) | `mutate.sh close … --quorum docs/audits/quorum-PMAT-1096-2971.json` exit 0 | quorum:agy width 3 | Q1 |
| 3 | bump `scripts/bump-version.sh 0.66.0`, CHANGELOG `[0.66.0]` (+ the missing `[0.65.1]`/`[0.65.2]`), local gates | `cargo fmt --all -- --check`, `cargo deny check advisories`, `cargo test -p aprender-contracts --lib`, `make gate` all exit 0 | direct | - |
| 4 | push once, PR (draft until phase 1 holds), merge behind `ci / gate`+`workspace-test`+`gate`, tag `v0.66.0`, `gh release create` | `gh release view v0.66.0 --json tagName` prints v0.66.0 | direct | - |
| 5 | crates.io cascade: `scripts/dogfood.sh --phase pre-publish` receipt → `scripts/check_publish_preflight.sh` R1–R6 → `scripts/cascade-drain.sh --target 0.66.0` (multi-pass) | `scripts/cascade-publish.sh --check` reports 0 behind | direct | - |
| 6 | post-publish QA: `cargo install aprender --version 0.66.0 --force`, `apr --version`, dogfood post-publish; receipt `status: complete`; milestone closed | `apr --version` prints 0.66.0 from the installed binary | direct | - |

routes:
  ph1  class=orchestration  route=self  w=11.11  basis=quota.json@46h
  ph2  class=review  route=agy-quorum  w=1.00  basis=quota.json@46h
  ph3  class=orchestration  route=self  w=11.11  basis=quota.json@46h
  ph4  class=orchestration  route=self  w=11.11  basis=quota.json@46h
  ph5  class=orchestration  route=self  w=11.11  basis=quota.json@46h
  ph6  class=orchestration  route=self  w=11.11  basis=quota.json@46h

## Dispatch ledger

| phase | mode | description | agent id | lane | width | turns | maxTurns hit | resumed | conversations |
|---|---|---|---|---|---|---|---|---|---|
| 2 | delegate:opus | PMAT-1096/ph2.delegate quorum width 3 on closing #2971 | ab286307cd9f490cc | quorum (mode=plan, writes=false) | 3 | 21 tool uses | no | no | 60e6651a-8cfb-42d6-b987-90efaad28c28, d84d00cc-d07f-446d-8460-311d7dc9ab26, c8ef19e9-57d5-42be-91ac-55ebfae2e020 (child_conversations=3) |

slots used: 1 of 3 (peak). Denials from `events-d8a83629-….jsonl`: 0 (`PreToolUse Agent decision=allow live=1`, `SubagentStart allow`, `SubagentStop released`).
I-3: `transcript-gate.sh` → `PASS attempted=0 denied=0 running_peak=0 slots=3` — **vacuous**: it swept the worktree's project dir (`-home-noah-src-aprender--claude-worktrees-rel-066`) while this session's transcript lives under `-home-noah-src-aprender` (the session started in the main checkout and moved into the worktree). The events file is the surviving witness: attempted=1 denied=0 running_peak=1 slots=3.

### Phase 2 outcome — the quorum did NOT agree; #2971 stays open

`lane-reduce.sh --width 3 --not-before 1757465470` exit 1, `agreed=false`: 2 lanes `close`, 1 lane `keep-open`. Artifact: `docs/audits/quorum-PMAT-1096-2971.json`. All three lanes measured (a) `scripts/check_model_parity.sh` implements min cosine ≥ 0.98 over ≥ 64 positions per manifest model and `SKIP_PARITY_GATE` never passes; (b) `evidence/models/supported.yaml` names `qwen2.5-coder-1.5b-instruct`; (c) the L0-1b change and its test exist; (d) `evidence/parity/l0-1/lambda/qwen2.5-coder-1.5b-instruct-q4_k_m.json` records the pre-fix 0.950827. The split is the issue's ask (3) — "a known-affected-shapes list surfaced before `--gpu`" — which lane 1 measured as absent and lanes 2–3 asserted as met by the runtime refusal; and every lane marked the step from same-shape Coder evidence to the reporter's base-Instruct file as `asserted`, never measured. Under T-6 a close needs `agreed=true` or the operator's own words; neither holds, so **no close is issued**. The release notes carry the fix as measured on the Coder file of the same shape. Follow-up in this receipt's gaps: measure the reporter's exact file (`qwen2.5-1.5b-instruct-q4_k_m.gguf`, downloaded to `~/models/` on lambda) with a cuda `apr` built from the release tree, then re-quorum. Delegate findings relayed: no lane could read the issue (gh 401 inside `--sandbox`; keyring token), so asks were judged against the brief's paraphrase; `<repo_root>/.claude/agent-memory/` is not gitignored (the delegate wrote there, saw it in `git status`, removed it).

## Verification (claimed vs my rerun)

verification:
  cmd="cargo fmt --all -- --check"  claimed_exit=n/a  rerun_exit=0  log_path=docs/audits/impl-PMAT-1096-logs/fmt.log  sha256=cbb7f05f0743eecf339ecf9ba95735258ce78709619d06375e1e0dd9c6e391f1
  cmd="cargo deny check advisories"  claimed_exit=n/a  rerun_exit=0  log_path=docs/audits/impl-PMAT-1096-logs/deny.log  sha256=5cdfbe38c43158e8a232460ff60c36ce8b008609925a7a2aaf77d59cd221d247
  cmd="cargo test -p aprender-contracts --lib"  claimed_exit=n/a  rerun_exit=0 (1501 passed, 5 ignored)  log_path=docs/audits/impl-PMAT-1096-logs/contracts.log  sha256=4d871214d2a714b50321aab2611ff4feb8487694a3769b59df788dbca54454fc
  cmd="make gate"  claimed_exit=n/a  rerun_exit=0 (pmat verify --skip satd --skip tests; guard_tree.sh --no-cargo; gate_touched_crates.sh → cargo check --workspace --tests, the fail-closed rule for a root Cargo.toml/Cargo.lock diff)  log_path=docs/audits/impl-PMAT-1096-logs/gate.log  sha256=7505d88f5a389b7a584fdf6d96625f730112e3c16707c8558cb89a2bbf608351
  cmd="bash scripts/bump-version.sh 0.66.0"  claimed_exit=n/a  rerun_exit=0 (root workspace 0.66.0; facades own version 0.4.0 left; facades upstream pins 0.66.0; facades lock --locked)  log_path=docs/audits/impl-PMAT-1096-logs/gate.log  sha256=7505d88f5a389b7a584fdf6d96625f730112e3c16707c8558cb89a2bbf608351
  cmd="cargo metadata --no-deps"  claimed_exit=n/a  rerun_exit=0 (79 packages, one version 0.66.0)  log_path=docs/audits/impl-PMAT-1096-logs/gate.log  sha256=7505d88f5a389b7a584fdf6d96625f730112e3c16707c8558cb89a2bbf608351

Every row above is the orchestrator's own run (no worker claimed anything this ticket); "claimed_exit=n/a" is that fact, not a missing column. Logs are `gate-reduce.sh` reductions (≤ 1 KB head + fail_tail); the full logs live only in the session scratchpad.

## Jidoka log

none (no red gate on this branch).

## Estimates

K̂=6 basis=first-run[U] (ROWS=0 — `docs/audits/impl-estimates.jsonl` has no qualifying `unit:turn phase:all` row) · K=150 basis=impl-PMAT-742-receipt.md [A] · actual at this write: 62 turns · appended to `docs/audits/impl-estimates.jsonl` (actual null until the ticket closes).

## DoD (six parts) and gaps

| part | state |
|---|---|
| merged green on `ci / gate`, `workspace-test`, `gate` | [U] — the PR is opened by this receipt's commit; result recorded post-merge |
| gate exists | `make gate` exit 0 at HEAD (above) |
| mutation observed RED | not applicable — a version bump carries no guard; the RED-turning evidence for the release content lives in #3050 (RED sha `9d0ddcf31`) and #3026/#3032 (`evidence/parity/l0-1/`) |
| `pv` contract same PR | `pv_lane=NotRun` — no contract changes in a bump; contracts_dir set; `pv validate` was exercised by `make gate`'s `pmat verify` only |
| discrimination confirmed | n/a for the bump; #3050's six-row case table is both-polarity |
| invalidated doc claims updated same PR | CHANGELOG gained `[0.66.0]`, `[0.65.2]`, `[0.65.1]`; no README literal cites 0.65.x (`grep -nE '0\.65\.[0-9]' README.md` → none) |

Gaps: (1) #2971 close — needs a re-quorum after the base-Instruct measurement; (2) phase 4–6 rows (tag, release, cascade, post-publish QA) are written into this receipt's follow-up on the post-publish docs PR, the pattern 0.65.2 used (#2868/#2871); (3) `pmat hooks install --strict --force` failed in the worktree (`Error: Not a directory` — `.git` is a file in a worktree), so the AD-03 commit-msg refusal was not installed here; every commit on the branch carries `Pmat-Ticket: PMAT-1096` by hand.

## Status blocks

[status] ticket=PMAT-1096 phase=1/6 global=8/6(K=150) k_measured=8 sub=0/0 basis=first-run[U]
         mode=direct trigger=- route=self w=11.11 basis=quota.json@46h q=fable_binding=true/age_h=46 gate=NOT-RUN slots=0/3 denied=0
         red=- filed=- blocker=- next=drain #3050 and #3063 through the fleet; bump on the worktree meanwhile

[status] ticket=PMAT-1096 phase=2/6 global=37/6(K=150) k_measured=37 sub=0/0 basis=first-run[U]
         mode=quorum:agy trigger=Q1 route=agy-quorum w=1.00 basis=quota.json@46h q=fable_binding=true/age_h=46 gate=NOT-RUN slots=1/3 denied=0
         red=- filed=- blocker=- next=delegate quorum width 3 on closing #2971 (T-6 close gate)

[status] ticket=PMAT-1096 phase=3/6 global=62/6(K=150) k_measured=62 sub=0/0 basis=first-run[U]
         mode=direct trigger=- route=self w=11.11 basis=quota.json@46h q=fable_binding=true/age_h=47 gate=PASS slots=0/3 denied=0
         red=- filed=- blocker=- next=push the release branch once; open the PR as a draft until #3050 and #3063 merge; measure the reporter's exact GGUF on lambda

verdict: PARTIAL(release in flight) — the bump is green locally and pushed; the tag, the cascade and the post-publish QA follow the merge.
