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
| 7 | guard-cargo's host-side cargo steps use the SHARED `~/.cargo` (16 runners, every paiml repo) while its container steps use the per-PR `GUARD_CARGO_HOME`; a concurrent job re-extracted `registry/src` and rustc got ENOENT on `y4m` — #3089's fourth eviction | job 103197449766 on intel-clean-room-7, 09:10Z; remedy in phase 2a: `CARGO_HOME=$GUARD_CARGO_HOME` for every host-side cargo step |
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
- [x] `scripts/check_test_tier.sh` + `scripts/lib/test_tier.py` + `tests/fixtures/test_tier/` — the §6 ratchet, hermetic (temp dirs, committed fixtures), selftest 7 rows both polarities, mutation (rule b removed) turns the silent-move row RED. `scripts/test_tier_ledger.sh` (selftest 4 rows on a throwaway git repo) writes the ledger with the helper's convention; `evidence/fleet/test-tier.tsv` is the committed table of record (§6.2).
- [x] `scripts/fleet_utilization.sh` — the per-box packing table (runners, busy, aprender-busy, share vs 80/80/50, queued jobs with their label sets); selftest 6 rows; run it at the top of every iteration report
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

### §6.2 The tier of record (2026-09-11 09:30Z, ratchet + its own ledger)

`scripts/test_tier_ledger.sh --base origin/main --days 60` → `evidence/fleet/test-tier-ledger.json` (141 fix
commits, 241 touched modules, 3,523 `#[test]` touches, 26 integration-target modules recorded as not measured).
`scripts/check_test_tier.sh --junit <junit> --catch-ledger evidence/fleet/test-tier-ledger.json --update` →
`evidence/fleet/test-tier.tsv` (3,937 modules): **PR tier = 11,313 / 82,203 tests (13.8 %) costing 1,011.7 s of
4,390.6 s (23.04 %)**; the re-run without `--update` is green with no moves. The lane's earlier 3,920 / 16.8 % used a
different module mapping and is superseded by this table; both are inside the 50 % budget. Kaizen from here: every
PR that changes a tier carries its row; the table is regenerated from the nightly junit and the ledger.

### §6.3 Tier of record as a filterset (PMAT-3119, 2026-09-11)

`bash scripts/ci_test_tier.sh --tier-of-record [--tsv evidence/fleet/test-tier.tsv]` turns the §6.2 table into ONE
nextest filterset. A `tier=pr` lib module (column `kind` absent or `lib`) maps to
`package(=CRATE) & kind(lib) & test(/^MODULE::/)`; an integration binary (`kind`=`test`) maps to
`package(=CRATE) & binary(=MODULE)`. Atoms are grouped per package —
`(package(=a) & kind(lib) & test(/^(m1|m2)::/)) | (package(=b) & binary(=x))` — so the expression stays short;
module names are regex-escaped, and a parent-module row also matches its submodules' tests, which errs toward
running more. Printed with it: `tier_of_record_{tests,modules,packages,seconds}` (594 modules, 11,313 tests,
1,011.65 s on the committed table). `--union-touched` (with `--event pull_request --comparand REF`) ORs in
`package(=C)` for every touched crate and prints `union_touched_crates=`, so a PR runs its own crates in full plus
the cross-tree 20 %; when the quick tier falls closed it prints `tier=full` and NO filterset — the FULL suite runs.
It also ORs in the quick tier's tree-reader targets, and it does so by handing `targets=` to the SAME
`--filterset` token→clause translation ci.yml uses (§6.5), never a second implementation: exactly ONE
`filterset=` line is printed, or a consumer would have to guess which of two it owed.

**Exit 1 — never a silent full run, never a silent empty set:** missing/unreadable table, a first line that is not
the header, a malformed row, zero `tier=pr` rows, or a `pr` row with an empty crate/module. Misusing
`--union-touched` is exit 2. The mapping lives in `scripts/lib/test_tier.py::filterset_from_tsv`; 24 of the 76
`--self-test` rows cover it, hermetically, against `tests/fixtures/test_tier/tier-small.tsv` and its committed
golden (`tier-small.filterset.txt`) — deleting one `pr` row changes the expression, which is the mutation row. The
`.github/workflows/ci.yml` wiring is phase 2 and is NOT in this change: until then the filterset is measured only by
the self-test.

### §6.4 roadmap.yaml 3-way merge by id (PMAT-3118)

A squash-merge from the queue re-serialises all of `docs/roadmaps/roadmap.yaml`, so every stacked branch goes DIRTY
on that one file even when the two sides touched unrelated tickets (measured 3× in one hour, 2026-09-11).

- **Driver**: `scripts/lib/roadmap_merge.py BASE OURS THEIRS [--out F]` merges by ENTRY ID over the byte-exact
  `- id:` blocks `roadmap_diff.split_entries` already produces. Ours-only / theirs-only / identical changes resolve;
  both-changed-differently and deleted-vs-edited CONFLICT, naming the id on stderr. Output order is
  `check_roadmap_sorted.sh`'s order (ascending within each id-prefix, prefixes in first-appearance order), so a side
  that tail-appended out of order is re-seated rather than merged unsorted. `--selftest` is an 8-row case table.
- **Attribute**: `.gitattributes` carries `docs/roadmaps/roadmap.yaml merge=roadmap`. The attribute alone does
  nothing: GitHub's merge engine never runs a local driver, so this is a LOCAL resolution path, not a merge-queue one.
- **Runner**: `scripts/ci_resolve_dirty.sh` — `--list-only` prints the DIRTY selection, no args plans one line per
  DIRTY PR, `--apply` merges in a throwaway worktree and prints the `git push` to run; `--pr N` restricts it.
  `CI_RESOLVE_DIRTY_PRS_JSON=<file>` substitutes a canned `gh pr list --json …` payload, which is how `--selftest`
  stays hermetic (a `gh` shim on PATH proves gh is never reached).
- **Registration is per invocation**: the steward/orchestrator passes
  `git -c merge.roadmap.driver="python3 scripts/lib/roadmap_merge.py %O %A %B" merge …`. Never `git config` in the
  shared `.git` — a worktree fleet shares that file, and a driver written there leaks into every other lane.

### §6.5 Tree-reader rows at MODULE granularity (PMAT-3120, 2026-09-11)

Issue #3120 measured where the PR quick tier's bill actually is: **88.08 % of ALL test-seconds came from the 18
whole `--lib` crates in `scripts/tree_reader_tests.txt`** — the registry named a CRATE when one test FILE in it
read the tree — while the touched-crate expansion adds ≤ 0.26 %. So `scripts/check_tree_reader_tests.sh` now
derives `crate<TAB>--lib<TAB>module::path` (src/a/b.rs → `a::b`, src/a/mod.rs → `a`, src/lib.rs → `<root>`, an
`include!()`-pulled or `#[path]`-attached file → the INCLUDING/DECLARING file's module). A module it cannot
resolve falls back to the whole crate (2 columns) **and prints `WARN unresolved-include` on stderr** — a silent
fallback restores the 88 % without anyone noticing.

The narrowing is a **TOKEN** extension of the ONE build graph #3089 built (PMAT-1098 —
`cargo nextest run --workspace --lib --tests -E "$EXPR"`): `targets=` gains
`crate:--lib:module::path` beside `crate:--lib`, `crate:--bins` and `crate:--test:NAME`, and
`ci_test_tier.sh --filterset` — the operand ci.yml already hands `targets=` to — maps it to
`(package(C) & kind(lib) & test(/^module::/))`, one clause per token, UNIONed with `|` exactly as the other three
tokens are. nextest matches a test's full path, so `^module::` is that module and its descendants and nothing
else; the module is regex-escaped (`::` is not special, so it stays literal). `<root>` and a 2-column
whole-crate fallback stay `(package(C) & kind(lib))`, and a crate carrying a whole-lib row WINS over its own
module rows — the registry asking for the whole lib is never answered with a narrower atom.
**`.github/workflows/ci.yml` needed no change at all**: the step reads the same `targets=` key and makes the same
single `--filterset` call, so the 88 % → 25 % narrowing is live with this merge rather than a phase 2. The token
grammar is pinned textually in both polarities by two committed goldens
(`tests/fixtures/tree_reader/registry-small.{targets,filterset}.txt`) over a hand-written registry carrying one of
every column shape, with a mutation row that drops the third column from every lib row and must turn the golden
diff RED.

Re-measured read-only over the same last 25 merged PRs and the same `evidence/fleet/test-tier.tsv` prices
(method: the measurement lane's `method.sh` + `remeasure_3120.py`; a module prices by its own key plus its
submodule keys, which is what `test(/^M::/)` actually runs):

| tier | test-seconds | % of the FULL suite (4,390.6 s) |
|---|---|---|
| tree-reader half, OLD (18 whole `--lib` crates) | 3,867.1 s / 64,884 tests | **88.08 %** |
| tree-reader half, NEW (92 module rows + 3 whole-crate rows) | 1,083.3 s / 9,946 tests | **24.67 %** |
| E1 (touched crates ∪ tree-readers), 25-PR sum | 102,435.9 s → 63,462.3 s | **93.32 % → 57.82 %** |
| E1 on a PR that stays quick (14 of 25) | 3,867.1 s → 1,083.3 s each | **88.08 % → 24.67 %** |

E1's 57.82 % is floored by the 11 of 25 PRs whose selection falls CLOSED to the full suite (100 % each,
untouched by this change); on the 14 quick PRs the tier is 3.57× cheaper. Two findings ride along: **578.1 s
of the remaining 1,083.3 s (53 %) is ONE unresolvable reader** — `crates/apr-cli/src/bin/apr-corpus-ingest.rs`,
a `[[bin]]` target whose unit tests no tier runs today (neither `--workspace --lib` nor the quick tier's
`-p apr-cli --lib`), so the fallback pays for apr-cli's whole lib to test a target it does not contain; wiring
`src/bin/` readers needs a measured `--bins` run and is owed, not guessed. And the dir-scoped module resolution
fixed a **silent-miss**: a crate-wide `mod tests;` match resolved `aprender-serve/src/cli/tests.rs` to
`cli::tests`, an atom matching ZERO tests, where the real path is `cli::cli_tests` (10 rows, 18.1 s). 10 of 92
module rows have no row in the table of record (feature-gated: `setfit::*`, `tui::*`, `orbit::wasm`, …) and are
priced 0.

