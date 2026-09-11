# CI fleet hygiene: no manual sweeps, no manual queue surgery (PMAT-1105)

**Status:** draft v0.1, 2026-09-11. **Owner:** the 0.67 train orchestrator. **Operator rule (verbatim, 2026-09-11):**
"disk space, blocked queues, etc should never be manual, but automated with forjar and quorum based designs using
distributed computing research […] build servers gx10 and yoga should never be idle if intel is busy and queued with
aprender."

## §1 What went wrong (measured 2026-09-11, session 870bdc8d)

| # | Finding | Evidence |
|---|---------|----------|
| 1 | `ci.yml` keys every run's cargo target at `<targets>/aprender-ci/<PR>/run-<RUN_ID>`; nothing removes it | 629 GB under `intel-mirror/targets/aprender-ci` on yoga (root 100 %), 374 GB on intel, 29 dead `gh-readonly-queue/main/pr-*` dirs |
| 2 | ENOSPC on yoga evicted #3089 from the merge queue (guard-cargo + workspace-test) | run 34568220853 jobs on yoga-build / yoga-build3 |
| 3 | A dequeued merge-queue entry's `merge_group` run keeps running and holds runners | 14 orphan runs cancelled by hand at 05:49Z–05:55Z |
| 4 | `gh run cancel` on a *queued* run can silently no-op | 5 runs needed `POST …/runs/<id>/force-cancel` |
| 6 | The sweeper's first live run deleted a fresh EMPTY mountpoint (`3112/run-34568509271-guards`): a files-only age test cannot see a dir that is used by being mounted; dockerd recreated it root-owned and guard-cargo died EACCES | job 103165670305, 07:18Z; fixed by paiml/infra#507 (dir mtime counts; 14-row selftest) |
| 5 | intel 15/16 busy (another repo's CI + `trueno-rag index --jobs 16`), 11 aprender runs queued ≥ 2 h, yoga 0/5 and gx10 0/4 busy | org runner list 05:39Z; every queued job asked `clean-room`, which only intel carried |

## §2 Disk: the sweeper (phase 1, LIVE)

paiml/infra#506 — `machines/clean-room/ci_target_sweep.sh` + `ci-target-sweep.{service,timer}` + forjar resources on
intel, yoga, gx10 (tag `sweep`). Hourly, `Persistent=true`. Policy: a `gh-readonly-queue/*/*` dir with no file newer
than 120 min, or a `<PR>/run-*` dir with no file newer than 360 min, is removed; nothing else; `find -P -xdev`; root
allow-list. `--selftest` = 11-row case table both polarities; mutation (drop the `-mmin` guard) turns it RED.
First receipts: yoga 129 GB, intel 93 GB, gx10 3.4 GB.

**Phase 2a (this repo):** every `ci.yml` job that bind-mounts a target dir pre-creates it with `heal_path` — `run-<ID>-guards` exactly as `run-<ID>`, since a missing mountpoint is created root-owned by dockerd whoever removed it (§1 #6) — and every `ci.yml` job that bind-mounts `…/run-${GITHUB_RUN_ID}` removes it in an
`if: always()` step through the image (root-owned contents), so the sweeper only ever sees hard-killed jobs.
Falsifier: a run's target dir must not exist 5 min after its job completed (`ci_target_gc_check.sh`, NotRun until landed).

## §3 Queue: the steward (phase 2b)

A scheduled steward (`scripts/ci_queue_steward.sh`, forjar-managed timer, every 5 min, on a host with an
authenticated `gh`) that is **check-then-write and quorum-gated**. It observes three independent signals —
(S1) the merge-queue entries (`GraphQL mergeQueue`), (S2) the `merge_group` runs and their jobs (`actions/runs`),
(S3) the org runner list (`orgs/paiml/actions/runners`) — and acts only when a decision is supported by **two
consecutive samples ≥ 2 min apart that agree**, so one stale read never cancels live work (the classic
two-phase/quorum read used in distributed leases: act on the intersection of independent views, never on one view).

| Rule | Condition (both samples) | Action | Why |
|------|--------------------------|--------|-----|
| Q1 orphan run | a `merge_group` run's `gh-readonly-queue/main/pr-N-<sha>` ref is not in S1 | `force-cancel` the run | §1 #3, #4 |
| Q2 doomed entry | an S1 entry whose S2 run has a failed job that `gate` needs | dequeue it (`dequeuePullRequest`), record why on the PR | ALLGREEN would rebuild the entries behind it up to 3× |
| Q3 evicted-green | a PR with milestone = the leaving train, required checks SUCCESS on head, auto-merge on, not in S1, no entry in the last 10 min | `enqueuePullRequest` | #3089 sat green for 17 h because its `gate` was merely *queued* |
| Q4 idle box | S3: intel busy ≥ 80 % ∧ aprender jobs queued ∧ (gx10 idle ∨ yoga idle) for both samples | emit `IDLE-NEXT-TO-QUEUE <box> <labels asked>` to the receipt and the epic | the operator's invariant; the remedy is a label or a routing PR, named in the receipt |

Destructive writes (cancel, dequeue) additionally require the same verdict from the previous tick (three samples
total) and are capped at 5 per tick. Every tick prints one receipt line and a packing table (intel/gx10/yoga:
capacity, busy, aprender-busy, share vs target 80/80/50) and appends it to `evidence/fleet/queue-steward.jsonl`.
`--selftest` replays recorded S1/S2/S3 fixtures (both polarities: an orphan that must be cancelled, a live run that
must not; a doomed entry, a healthy one; a green evicted PR, a red one; an idle box with and without a queue) and
fails on any wrong action. Mutation: removing the two-sample agreement must turn the selftest RED.

## §4 Owed

- [ ] phase 2a `ci.yml` end-of-job GC + `ci_target_gc_check.sh` (this branch)
- [ ] phase 2b steward script + selftest + forjar timer (infra), first receipts in `evidence/fleet/`
- [ ] `scripts/fleet_utilization.sh` promoted from the session `pack-report.sh` (67-row artifact)
- [ ] contract `contracts/ci-fleet-hygiene-v1.yaml` (kind: pattern) binding §2/§3 falsifiers; `pv validate`

### §3.1 State Directory Layout & Fixture Format

The steward's state directory (`--state-dir`) organizes tick data chronologically.
Each tick produces a directory `sample-<epoch>/` containing `s1.json`, `s2.json`, and `s3.json`.
- `s1.json`: The GraphQL result for the merge queue and PRs in the open milestone.
- `s2.json`: A unified JSON array of `merge_group` runs augmented with their jobs.
- `s3.json`: A combined JSON object of `runners` and queued `jobs`.
The steward writes a summary to `receipts.jsonl`.
Under `--selftest`, fixtures follow the exact same format inside `tests/fixtures/ci_queue_steward/<case_name>/<epoch>/`. The selftest harnesses this by copying these files into a temporary state directory, tricking the steward into reading them as historical data.

## §5 Measured baseline for the build-time objective (2026-09-11 07:48Z, `scripts/fleet_history.sh --limit 60`)

Objective (operator, verbatim): "shortest total duration of build time for each tagged released until Pareto Optimal
by measured history of logs we capture and report". Variables: (A) the PR-tier test set, (B) placement across
intel / gx10 / yoga. `evidence/fleet/history.jsonl` is the ledger (one row per completed run, idempotent append).

| measure (p50 over the last 60 completed runs, orphaned/cancelled included) | value |
|---|---|
| run wall, `pull_request` | 1912 s |
| run wall, `merge_group` | 754 s |
| `workspace-test` duration on intel / on yoga | 3295 s / 1869 s — yoga is 1.76× faster for the same job on the shared intel box |
| job queue wait: intel / yoga / gx10 | 350 s / 171 s / 107 s |
| jobs placed: intel / yoga / gx10 / hosted | 128 / 66 / 13 / 22 (the 22 hosted are the book workflows, #3073) |

Reading: intel is both the slowest box for workspace-test and the one jobs wait longest for, because it is
shared across paiml repos; every X64 job moved to yoga saves ~24 min of test time and ~3 min of wait. gx10 takes
only arch-neutral jobs until #3104 lands. Next samples go beside this table; a change to (A) or (B) is judged by
the delta in these rows, never by intent.

## §6 The 80/20 PR tier, measured (2026-09-11 07:50Z)

Operator (verbatim): "Only 20% of tests should provide 80% of value. [Kaizen this]". Inputs, both measured:
(i) per-test seconds from one full `cargo nextest run --profile ci --workspace --lib` on lambda-vector (48 cores):
82,203 tests, 73 binaries, 67 crates, 4,391 CPU-test-seconds, 370 s wall; (ii) catch value = the number of
times a crate's test files were touched by the 141 `fix` commits of the last 60 days (366 touches in measured
crates). Table: `evidence/fleet/test-tier-2026-09-11.tsv` (crate, tests, seconds, touches, touches/second,
cumulative %).

| finding | value |
|---|---|
| crates that carry 80 % of fix-linked catches | 16 of 67, costing **43.4 %** of test seconds |
| the four heaviest low-value crates | aprender-train 680 s / 5 touches, aprender-contracts 484 s / 17, aprender-core 460 s / 24, aprender-orchestrate 344 s / 18 |
| crates with ZERO fix-linked touches | 46, costing 11.3 % of test seconds (aprender-qa-runner 201 s, aprender-registry 122 s, …) |
| the densest value | aprender-serve: 15,775 tests, 877 s, 170 touches (46 % of all catches for 21 % of seconds) |

**Rule for the PR tier (proposed; the ratchet is the falsifier):** PR/queue runs the E1 set (touched crates + direct
reverse dependents + the 41 tree-reader targets) **∪ the catch-dense set** (every crate above the 80 % line in the
table); everything below the line runs on `main` nightly and in the pre-publish FULL dogfood. Expected PR-tier
cost from this sample: ≤ 43 % of the FULL suite's seconds when nothing dense is touched; the E1 union keeps a
touched crate's own tests regardless of rank. Granularity is the crate; the next Kaizen step is per-test
(`testcase` rows in the same junit), which will move the 80 % line well below 43 % because value inside
aprender-serve and apr-cli is concentrated in their falsifier modules.

**Ratchet (owed, §4):** `scripts/check_test_tier.sh` regenerates the table from the latest junit + git log and
refuses a PR that moves a crate across the line without a new row in this section; a tier whose PR-tier seconds
exceed the FULL suite's 50 % is RED (D-1's 20-min budget is the wall-clock counterpart).

### §6.1 Per-module granularity (2026-09-11 08:15Z, one agy research lane, re-run by the orchestrator)

Same two inputs, aggregated at `crate::module` (junit `testcase` rows mapped to the longest valid module path; fix
commits' `#[test]` hunks mapped to their module). Artifacts: `evidence/fleet/test-tier-2026-09-11/`
(per-module.tsv, pr-tier-filterset.txt, README.md, simulate_filterset.py — the simulation was re-run in this tree:
identical numbers).

| line | modules | % of test seconds |
|---|---|---|
| 80 % of fix-linked catches | 222 | **0.28 %** |
| 90 % | 268 | 0.91 % |
| 95 % | 299 | 1.63 % |

PR tier = modules above the 80 % line ∪ every test whose name contains `falsif` (the designed catches): **3,920 of
82,203 tests (4.77 %) costing 737.7 s of 4,390.6 s (16.8 %)**. The falsifiers are ~670 s of that; the dense modules
themselves are ~12 s. Together with E1's touched-crate set this is the PR tier; the FULL suite stays nightly and
pre-publish. Caveats carried from the lane: tests in `[[bin]]` targets are not in the `--lib` junit (recorded as
not-measured); 46 crates have zero touches and are excluded until a fix lands in them — a reviewer may add a module
by hand with its row; root-module tests are matched by regex and deserve a per-test audit before the ratchet lands.
