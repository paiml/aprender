---
status: partial
ticket: PMAT-1096
kind: code
milestone: 0.66.0
branch: PMAT-1096-release-0-66-0
base: main f34671a6b
epic: 2873
model: claude-fable-5-1 (orchestrator) · one agy quorum (width 3) via paiml-agy-delegate
turns: 213
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
| `k_measured` vs `global=k` | 213 vs 213 [V] | the jq below over the session transcript |

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
| 2 | delegate:opus | PMAT-1096/ph2.delegate2 quorum width 3 on closing #2971 with the exact-file record | a61181ec19370d084 | quorum (mode=plan, writes=false) | 3 | 23 tool uses | no | no | 82e5ea84-2efb-4a4e-b33e-7d088ef9df7f, e3068ae1-69af-4beb-935f-c39e7daaed04, b815c80a-21e5-45fd-98b4-31bb0916510e (child_conversations=3) |

slots used: 1 of 3 (peak). Denials from `events-d8a83629-….jsonl`: 0 (two dispatches, both `PreToolUse Agent decision=allow live=1` → `SubagentStart allow` → `SubagentStop released`; never two at once).
I-3: `transcript-gate.sh` → `PASS attempted=0 denied=0 running_peak=0 slots=3` — **vacuous**: it swept the worktree's project dir (`-home-noah-src-aprender--claude-worktrees-rel-066`) while this session's transcript lives under `-home-noah-src-aprender` (the session started in the main checkout and moved into the worktree). The events file is the surviving witness: attempted=2 denied=0 running_peak=1 slots=3.

### Phase 2 outcome — round 1 did NOT agree; round 2 agreed 3/3 after the exact file was measured; #2971 CLOSED

`lane-reduce.sh --width 3 --not-before 1757465470` exit 1, `agreed=false`: 2 lanes `close`, 1 lane `keep-open`. Artifact: `docs/audits/quorum-PMAT-1096-2971.json`. All three lanes measured (a) `scripts/check_model_parity.sh` implements min cosine ≥ 0.98 over ≥ 64 positions per manifest model and `SKIP_PARITY_GATE` never passes; (b) `evidence/models/supported.yaml` names `qwen2.5-coder-1.5b-instruct`; (c) the L0-1b change and its test exist; (d) `evidence/parity/l0-1/lambda/qwen2.5-coder-1.5b-instruct-q4_k_m.json` records the pre-fix 0.950827. The split is the issue's ask (3) — "a known-affected-shapes list surfaced before `--gpu`" — which lane 1 measured as absent and lanes 2–3 asserted as met by the runtime refusal; and every lane marked the step from same-shape Coder evidence to the reporter's base-Instruct file as `asserted`, never measured. Under T-6 a close needs `agreed=true` or the operator's own words; neither holds, so **no close is issued**. The release notes carry the fix as measured on the Coder file of the same shape. Follow-up in this receipt's gaps: measure the reporter's exact file (`qwen2.5-1.5b-instruct-q4_k_m.gguf`, downloaded to `~/models/` on lambda) with a cuda `apr` built from the release tree, then re-quorum. Delegate findings relayed: no lane could read the issue (gh 401 inside `--sandbox`; keyring token), so asks were judged against the brief's paraphrase; `<repo_root>/.claude/agent-memory/` is not gitignored (the delegate wrote there, saw it in `git status`, removed it).

**Round 2.** The gap every round-1 lane named was measured rather than argued: the reporter's exact file
(`qwen2.5-1.5b-instruct-q4_k_m.gguf` from `Qwen/Qwen2.5-1.5B-Instruct-GGUF`, sha256
`6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`, verified against the HF LFS oid) was
run on lambda (RTX 4090, driver 580.119.02) through the tree's own 78-token corpus prompt with a cuda
`apr` built from HEAD — `apr 0.66.0 (611989200)`, `scripts/apr_bin.sh` exit 0 (an earlier build at
`a25a575c5` was refused as STALE and rebuilt). `check_model_parity.sh --judge` → **PASS, 78 positions,
min cosine 0.9978 at position 4 ≥ 0.98**; the stderr carries the engaged-mechanism lines (`[GH-129]
Early kernel preload`, `[PMAT-082] cuBLASLt FP8 JIT warmed (1536×16×1536)`, `[trueno#243] Manual graph:
619 kernels`). Record: `evidence/parity/l0-1/lambda/qwen2.5-1.5b-instruct-q4_k_m.{json,err}` + the
appended section of `RECORD.md`. The same binary's `--manifest` on this host: measured=3
(coder-0.5b 0.9996 / coder-1.5b 0.9998 / coder-7b 0.9996), rc=0. Round 2 (delegate `a61181ec19370d084`,
the issue's ask section pasted verbatim because lanes cannot run `gh`): `lane-reduce.sh --width 3
--not-before 1788974190` exit 0, **`agreed=true`, 3/3 close, dissent=[]**; ask (1) grounded at
`crates/aprender-serve/src/gguf/inference/forward/ffn_block.rs:780` (lanes 1, 3) and
`crates/aprender-serve/src/quantize/mod.rs:348` (lane 2); ask (2) at
`crates/apr-cli/src/commands/parity_per_op.rs`; ask (3) judged moot by all three (grounded on
`parity_admission.rs` at three different lines — the delegate flagged that as the weakest item). Delegate
findings relayed: `uncovered[]` says no lane echoed the sha256 token or the build id, so identity was
re-verified by the orchestrator (`sha256sum` and the judge re-run, rows below) before the citation was
written; per-lane exit codes were not captured (bare `wait`) — agy `status: SUCCESS` and 0-byte stderr are
the substitutes. Artifacts: round 1 `docs/audits/quorum-PMAT-1096-2971-round1.json` (agreed=false), round 2
`docs/audits/quorum-PMAT-1096-2971.json` (agreed=true). **Close issued** through the T-6 gate:
`mutate.sh close --repo paiml/aprender --issue 2971 --cite … --quorum docs/audits/quorum-PMAT-1096-2971.json`
exit 0; read back `gh issue view 2971 --json state` → `CLOSED` (closedAt 2026-09-09T17:25:34Z);
`mutations.jsonl` carries the one `close` row.


### Phase 3b — the pre-publish dogfood, run early on the branch, NO-GO → fixed at the root → GO

`scripts/dogfood.sh --phase pre-publish` was run on the branch head (`c554a7731`) BEFORE the merge, to surface
red rows while the fleet queue drains (the R5 receipt itself must be re-taken on the merge commit). Round 1:
**NO-GO** on five rows, none of them introduced by the bump — every one came in on `main` since 0.65.2:

| row | cause | fix (this branch, `b891ab789`) |
|---|---|---|
| `declared:check_no_claim_literals` FAIL | the `[0.66.0]` CHANGELOG insertion shifted six baselined historical lines (file:line-keyed, shrink-only ratchet reads a move as growth) | the six numeric claims deleted from the historical lines; baseline pruned 452→446 (`--update`); `check_baseline_ratchets.sh` PASS |
| `declared:check_perf041_marker` FAIL | `evidence/perf041/lambda/marker.json` 7.1 days old (`witness.max_age_days=7`) | `scripts/perf041_batched_parity_probe.sh` re-run on lambda with the cuda `apr` at `c554a7731`: c=1/4/8/16 all PASS, `intra_agree_to=128`, `max_m=16`, `m1_agree_to=3` for c≥4 (the known kernel-family divergence at token 3, recorded not gated); marker + witness committed; guard PASS age=0.0d |
| `bashrs` FAIL 8 SEC/DET/IDEM over 271 files | `.pr/L0-1b/{accept,step0/sweep}.sh` (#3032), `check_hardcoded_paths.sh`, `check_roadmap_diff_additive.sh`, `predict_merge.sh`, `tests/guard_tree_job_test.sh` (BSE PRs) | eval→sed-read of `PROMPT` (byte-identical, 445 chars), `${R:?}`/validated `rm -rf`, `..`-refusal before `mkdir`, `SECONDS` instead of `date`, `yq e`; per-file and single-invocation gating count 0; each guard's own selftest/run re-executed (`check_hardcoded_paths --selftest` PASS, `guard_tree_job_test` 4/4, `predict_merge` PASS; `check_roadmap_diff_additive --selftest` rc=1 **before and after**, row verdicts identical — a pre-existing selftest defect, filed below) |
| `model-parity` FAIL "Feature not enabled: cuda" | dogfood's release-binary gate builds `--features cli` (no cuda) and C14 then ran that binary on a CUDA host — a tool defect read as a model defect | `scripts/dogfood.sh` C14 leg builds its own `--features cuda` apr into `<target>/dogfood-cuda` when `nvidia-smi -L` lists a device, records to a work dir (never `evidence/parity/<host>/`); GPU-less hosts unchanged |
| `git-clean` WARN | the C14 run's stray `evidence/parity/noah-Lambda-Vector/` and the delegate's `.claude/agent-memory/` | stray dir removed; `/.claude/agent-memory/` gitignored |

Round 2 at `b891ab789`: `make gate` exit 0; `scripts/dogfood.sh --phase pre-publish` → **GO** (43 rows; `model-parity PASS` 3 manifest models with `--features cuda`; `bashrs PASS`; only `reachability WARN`, informational). Receipt copied to
`docs/audits/impl-PMAT-1096-logs/dogfood-pre-publish-b891ab789.json` (sha256 `539a919203588b98a4d2c777f9bae64985ddaf9066d9320a18f70efd85e5b75c`); it is NOT the R5 receipt — that one is re-taken on the merge commit.

### Phase 3c — #3025 merged under the queue and made #3050 and this branch DIRTY

`#3025` (PP-066 SPEC-2.0: 866-line roadmap rewrite, README counts, ci.yml, five new scripts) merged at 21:09Z; the queue
then dropped `#3050` (`removed_from_merge_queue`, `mergeable_state: dirty`, auto-merge off) and this branch went dirty.
- **#3050**: the only conflict was README's derived contract count (branch 1818 vs main 1816). Resolved on a merge of
  `origin/main` into `agent/F-1` by re-deriving on the merged tree (`find contracts -name '*.yaml' | wc -l` = 1817),
  `check_readme_claims.sh` PASS, pushed `389d6b451`, auto-merge re-armed (queue method SQUASH, one entry at a time).
- **this branch**: two append-only conflicts (`docs/roadmaps/roadmap.yaml` — main's mints vs the PMAT-1096 mint;
  `docs/audits/impl-estimates.jsonl`) resolved as unions; `check_roadmap_diff_additive.sh` PASS (added=1),
  `pmat work validate` PASS. The merged tree then failed two things the autopilot's dogfood would have refused:
  `scripts/session_docs_commit.sh` (new on main) carried DET002 + SEC010 (fixed: SOURCE_DATE_EPOCH-derived date, the
  fleet pattern; named, `..`-validated temp paths), and `make gate` refused `check_no_tracked_ignored_files.sh`
  320→330 because **main now tracks `.claude/agent-memory/**` (10 subagent memory files committed by #3025)** and my
  `/.claude/agent-memory/` ignore rule declared them ignored — the rule is withdrawn (`.gitignore` = main's), the
  tracked agent memory is filed below as a finding. Merged tree: bashrs single-invocation gating 0 over 276 files,
  `make gate` 41 checks 0 failed, claim-literal + ratchet + README + PP-26 guards PASS.
- **#3070 filed**: BSE-17's quick tier ran 42 tree-reader targets serially for #3063 and hit its 60-minute step timeout
  under fleet load (zero failing tests; annotation is the surviving truth); #3063 re-run via `gh pr update-branch`, now
  in the queue.

### Phase 3d — four more reds on the merged trees, each a tool-or-environment defect fixed at its root

| PR | red | root cause | fix |
|---|---|---|---|
| #3063 | merge-queue `workspace-test` nextest exit 100: `driver::cublas_tests::*` panicked `CudaNotAvailable` on the clean-room | the PR measured `-p aprender-gpu -p aprender-cuda-edge` (per-package resolve, cuda off: 0 `driver::` tests) but changed the `--workspace` line, where cargo unifies features and **`aprender-explain` depends on `aprender-gpu` with `features=["cuda"]` non-optionally** (217 `driver::` tests listed) | `c6ce084ec`: the workspace line keeps its excludes; the two GPU crates run as their own `full`-tier per-package step (same container/mounts as the compute step); `tree_reader_tests.txt` re-derived (`a82983661`); tier case table 15/15; noted on #3067 |
| #3069 | `guard-cargo` "Cargo.lock must match" + `bump --check` "facades lock stale" | a worktree nested under `/home/noah/src/aprender/.claude/worktrees/` inherits the checkout's `.cargo/config.toml` `[patch.crates-io]`; the bump's regenerated locks carried 11 `[[patch.unused]]` entries CI's clean cargo strips → `--locked` refuses | `52d65b6f3`: both locks re-derived with the cwd outside the checkout (0 `patch.unused`; both `--locked` PASS) |
| #3050 | `guard-tree` G-4 `render_dag.py --check` DRIFT | the DAG status column is derived from receipts; F-1's receipt says `complete`, the committed block said `open` | `4ce257674`: block re-rendered and pasted (one row) |
| #3069 | `guard-tree` machine-specific-path ratchet +1 | the new `apr parity --json` record stored `model: /home/noah/models/…` | model field written as `~/models/…` (the judge reads positions, not the path); ratchet delta +0 |

Also: #3063's first PR-level run died on BSE-17's quick tier 60-minute step timeout (42 tree-reader targets serially; zero failing tests) → **#3070**. Every one of these was invisible to the PR-level checks and only surfaced on the merged tree — the 0.66 lesson is that a per-package measurement never proves a `--workspace` line, and a nested worktree is not a clean cargo environment.

## Verification (claimed vs my rerun)

verification:
  cmd="cargo fmt --all -- --check"  claimed_exit=n/a  rerun_exit=0  log_path=docs/audits/impl-PMAT-1096-logs/fmt.log  sha256=cbb7f05f0743eecf339ecf9ba95735258ce78709619d06375e1e0dd9c6e391f1
  cmd="cargo deny check advisories"  claimed_exit=n/a  rerun_exit=0  log_path=docs/audits/impl-PMAT-1096-logs/deny.log  sha256=5cdfbe38c43158e8a232460ff60c36ce8b008609925a7a2aaf77d59cd221d247
  cmd="cargo test -p aprender-contracts --lib"  claimed_exit=n/a  rerun_exit=0 (1501 passed, 5 ignored)  log_path=docs/audits/impl-PMAT-1096-logs/contracts.log  sha256=4d871214d2a714b50321aab2611ff4feb8487694a3769b59df788dbca54454fc
  cmd="make gate"  claimed_exit=n/a  rerun_exit=0 (pmat verify --skip satd --skip tests; guard_tree.sh --no-cargo; gate_touched_crates.sh → cargo check --workspace --tests, the fail-closed rule for a root Cargo.toml/Cargo.lock diff)  log_path=docs/audits/impl-PMAT-1096-logs/gate.log  sha256=7505d88f5a389b7a584fdf6d96625f730112e3c16707c8558cb89a2bbf608351
  cmd="bash scripts/bump-version.sh 0.66.0"  claimed_exit=n/a  rerun_exit=0 (root workspace 0.66.0; facades own version 0.4.0 left; facades upstream pins 0.66.0; facades lock --locked)  log_path=docs/audits/impl-PMAT-1096-logs/gate.log  sha256=7505d88f5a389b7a584fdf6d96625f730112e3c16707c8558cb89a2bbf608351
  cmd="cargo metadata --no-deps"  claimed_exit=n/a  rerun_exit=0 (79 packages, one version 0.66.0)  log_path=docs/audits/impl-PMAT-1096-logs/gate.log  sha256=7505d88f5a389b7a584fdf6d96625f730112e3c16707c8558cb89a2bbf608351
  cmd="sha256sum ~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"  claimed_exit=n/a  rerun_exit=0 (6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e = HF LFS oid)  log_path=evidence/parity/l0-1/lambda/RECORD.md  sha256=2032bf8b6222719e2a9e56cdf0b2b732a7fb2f5ad753214676a2efafbeb85bb6
  cmd="APR_BIN=/mnt/nvme-raid0/targets/rel-066-cuda/release/apr bash scripts/apr_bin.sh"  claimed_exit=n/a  rerun_exit=0 (apr 0.66.0 (611989200) = HEAD at build)  log_path=evidence/parity/l0-1/lambda/qwen2.5-1.5b-instruct-q4_k_m.err  sha256=06709a1366b62e7c6b3684a20c486b3eaad0e5848f2d7b546a44b44b4a120e72
  cmd="bash scripts/check_model_parity.sh --judge evidence/parity/l0-1/lambda/qwen2.5-1.5b-instruct-q4_k_m.json --model qwen2.5-1.5b-instruct"  claimed_exit=0 (three lanes)  rerun_exit=0 (PASS 78 positions, min cosine 0.9978 at position 4)  log_path=evidence/parity/l0-1/lambda/qwen2.5-1.5b-instruct-q4_k_m.json  sha256=3dbd650563f114208f9d11a5b67a1249b39150b3d486e56bd5510dce62014e11
  cmd="gh issue view 2971 --json state -q .state"  claimed_exit=n/a  rerun_exit=0 (CLOSED)  log_path=docs/audits/quorum-PMAT-1096-2971.json  sha256=ce51c0acadcb9e02b21756a2750a4328b9a3eb245fc5e61b1a798a50213764a5
  cmd="make gate (b891ab789)"  claimed_exit=n/a  rerun_exit=0  log_path=docs/audits/impl-PMAT-1096-logs/gate2.log  sha256=fab1037f1dabf52e75d19058b3afee00d16f75a562c20a2296c900374de53c31
  cmd="bash scripts/dogfood.sh --phase pre-publish (b891ab789)"  claimed_exit=n/a  rerun_exit=0 (GO)  log_path=docs/audits/impl-PMAT-1096-logs/dogfood-pre-publish-b891ab789.json  sha256=539a919203588b98a4d2c777f9bae64985ddaf9066d9320a18f70efd85e5b75c
  cmd="bash scripts/check_perf041_marker.sh"  claimed_exit=n/a  rerun_exit=0 (lambda PASS age=0.0d)  log_path=evidence/perf041/lambda/marker.json  sha256=fd254d58852f4a651e3d73dd5fb87128365e1385e92c12e074d1a933de54fc1b
  cmd="bash scripts/check_no_claim_literals.sh && bash scripts/check_baseline_ratchets.sh"  claimed_exit=n/a  rerun_exit=0  log_path=scripts/claim_literal_baseline.txt  sha256=921afca893ad399db8cdd7954ff24c527b0a230afaf6e4eb1d1b2e6335d719d9
  cmd="make gate (merged tree)"  claimed_exit=n/a  rerun_exit=0 (41 checks, 0 failed)  log_path=docs/audits/impl-PMAT-1096-logs/gate2.log  sha256=fab1037f1dabf52e75d19058b3afee00d16f75a562c20a2296c900374de53c31
  cmd="bash scripts/check_no_tracked_ignored_files.sh"  claimed_exit=n/a  rerun_exit=0 (PASS ratcheted, after withdrawing the ignore rule)  log_path=docs/audits/impl-PMAT-1096-logs/gate2.log  sha256=fab1037f1dabf52e75d19058b3afee00d16f75a562c20a2296c900374de53c31

Rows marked "claimed_exit=n/a" are the orchestrator's own runs with nothing claimed by a worker; the judge row's claim is the three lanes' PASS, re-run here. Logs are `gate-reduce.sh` reductions (≤ 1 KB head + fail_tail); the full logs live only in the session scratchpad.

## Jidoka log

- {ticket: PMAT-1096, phase: 3b, defect: pre-publish dogfood NO-GO on five rows (claim-literal shift, stale PP-26 witness, 8 bashrs findings, C14 on a non-cuda binary, stray untracked dirs), owner: release tooling + the merged PRs that introduced them, whys: (1) why NO-GO → five red rows; (2) why red → each row above; (3) why not caught on main → main's dogfood is not run per PR, only at release; (4) why the tool defect → the release-binary gate and the C14 gate disagree on features; (5) root fix → C14 builds its own cuda leg, findings fixed in their scripts} — resolved same branch.
- filed: `.claude/agent-memory/**` (10 subagent memory files) is TRACKED on main since #3025 — a scratch surface in the index; the delegate's writes now show as modifications of tracked files. Owner: the PP-066 driver session. Untracking is a decision for that owner, not the release.
- filed: #3070 — BSE-17 quick tier 60-minute timeout on a CI-only PR.
- filed: `scripts/check_roadmap_diff_additive.sh --selftest` exits 1 on `main` at `f34671a6b` (row 'push shape'/re-serialisation) — pre-existing, verdicts identical before and after the SEC011 edit; not a blocker for the cut (the guard's non-selftest path is what CI runs and it PASSes).

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

Gaps: (1) closed — #2971 was closed on the round-2 quorum (above); (2) phase 4–6 rows (tag, release, cascade, post-publish QA) are written into this receipt's follow-up on the post-publish docs PR, the pattern 0.65.2 used (#2868/#2871); (3) `pmat hooks install --strict --force` failed in the worktree (`Error: Not a directory` — `.git` is a file in a worktree), so the AD-03 commit-msg refusal was not installed here; every commit on the branch carries `Pmat-Ticket: PMAT-1096` by hand.

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

[status] ticket=PMAT-1096 phase=2/6 global=78/6(K=150) k_measured=78 sub=0/0 basis=first-run[U]
         mode=quorum:agy trigger=Q1 route=agy-quorum w=1.00 basis=quota.json@46h q=fable_binding=true/age_h=47 gate=PASS slots=1/3 denied=0
         red=- filed=- blocker=- next=#2971 closed on a 3/3 quorum; wait for the fleet on #3050 #3063 #3069, then ready+arm #3069

[status] ticket=PMAT-1096 phase=3/6 global=118/6(K=150) k_measured=118 sub=0/0 basis=first-run[U]
         mode=direct trigger=- route=self w=11.11 basis=quota.json@46h q=fable_binding=true/age_h=47 gate=PASS slots=0/3 denied=0
         red=- filed=check_roadmap_diff_additive-selftest blocker=fleet: clean-room pool 15/17 busy on other repos; #3063 PR run queued 5h, #3066 queue run queued 1h next=push the gate fixes once; wait for #3050/#3063; then ready+arm #3069

[status] ticket=PMAT-1096 phase=4/6 global=160/6(K=300) k_measured=160 sub=0/0 basis=first-run[U]
         mode=direct trigger=- route=self w=11.11 basis=quota.json@46h q=fable_binding=true/age_h=47 gate=PASS slots=0/3 denied=0
         red=- filed=#3070,tracked-agent-memory blocker=fleet queue (#3064 #3056 #3063 ahead; #3050 re-running CI after the README conflict) next=autopilot: merge→dogfood→tag→release→cascade; K raised 150→300 on the operator's re-issued instruction

[status] ticket=PMAT-1096 phase=4/6 global=213/6(K=300) k_measured=213 sub=0/0 basis=first-run[U]
         mode=direct trigger=- route=self w=11.11 basis=quota.json@46h q=fable_binding=true/age_h=47 gate=PASS slots=0/3 denied=0
         red=- filed=#3070,#3067-comment,tracked-agent-memory blocker=fleet: three PR runs in progress on fixed heads next=queue #3050 → #3063 → autopilot readies+queues #3069 → dogfood → tag → cascade

verdict: PARTIAL(release in flight) — the bump is green locally and pushed; the tag, the cascade and the post-publish QA follow the merge.
