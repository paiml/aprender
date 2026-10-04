# BLD-001 — Release factory: train 0.70.2 "build" plan

**Status:** plan for operator review. Nothing is applied: no tickets minted, no PRs opened, no CI runs started, no
gate changed. Branch-only until 0.70.1 is on crates.io (operator ruling C277 item 3, 2026-10-03).
**Ticket:** proposed to the cop, not yet minted · **kind:** docs · **Train:** 0.70.2 "build" (C277 items 2, 5, 6;
C280 item 12) · **Owner:** aprender-a7 (build-kaizen)

Baselines come from §6's commands, run 2026-10-03 10:15–10:35Z on `origin/main` @ `316dee2cd4`, unless a cell says
otherwise. Reads that need the GitHub API wait until §0 reads S1 (C277 item 3 keeps the API budget for the release);
until then such a cell reads `not_measured`, never pass.

The operator's exit bar for this train, verbatim (C277 item 6, 2026-10-03):

> 0.70.2 ships when all of these hold, each with a receipt:
> - a nightly rehearsal runs the real publish path, upload excluded, and is green three nights in a row;
> - no check has been red on main for more than 24 h;
> - a version-only bump reuses the saved results: bump to tag under 30 minutes;
> - "required to merge, to tag, to publish" is one list in code: one definition of release-ready, zero contradictions;
> - tag to crates.io for 0.70.2 itself is under 2 hours;
> - the known-failure P0 tickets from 0.70.1 are closed or re-ruled.

And the 0.70.2 P0 named in operator ruling C280 (2026-10-03, 10:00Z), verbatim:

> 12. P0 for 0.70.2: no check may be enforced at publish until its inputs have been produced green on main,
> nightly, three times. The full CRUX lanes and the readiness check go nightly. CRUX gets the GPU to itself.

C280's header says "Replaces C279.", and its item 6 withdraws C279's option B, verbatim:

> 6. C279's B is withdrawn: no change to release_readiness.sh, no known_red additions, no UD-IQ2_XXS swap, no
> hotfix_scope record. If written, leave them out. The old failures stay red in the printed evidence, in the release
> notes and in their P0 tickets.

C279 had named two 0.70.2 P0s: item 6 ("restore the red_model entry, and fix the fallback or refuse the file by
name") and item 11 (marks must not void measured results; the two checks use one definition of "same code"). C280
restates neither. The plan treats them as having no operator mandate and keeps them out of the exit bar: rows 7 and 8
are carried as measured defects, and row 9 is folded into row 6 (decision D4). Whether C279 item 11 still stands is
§4 Q7.

## 0. Live state: which part of this plan applies now

```bash
v=$(curl -s -m 20 -A 'bld-001 (paiml)' https://crates.io/api/v1/crates/aprender \
      | jq -r '.crate.max_stable_version // empty')
[ -n "$v" ] || { echo "not_measured: crates.io read failed"; exit 2; }
ge() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" = "$1" ]; }   # ge A B  <=>  A >= B
if ge "$v" 0.70.2; then echo "S2 ($v)"; elif ge "$v" 0.70.1; then echo "S1 ($v)"; else echo "S0 ($v)"; fi
```

| State | Means | What runs |
|---|---|---|
| S0 | 0.70.1 is not on crates.io yet | This file, on a branch. No PR armed, no CI, no runner, no GPU, no local build (C277 item 3) |
| S1 | 0.70.1 is live, 0.70.2 is not | The §2 rows. One row = one ticket (minted by the cop) = one PR of ≤ 200 files. 0.70.2 comes first for runners, queue and review (C277 item 5) |
| S2 | 0.70.2 is live | Closed. The §3 counters stay as ratchets |

Read 2026-10-03 10:15Z: `max_stable_version` = `0.69.1` → **S0**.

## 1. Exit bar, made measurable

| Id | Operator words | Metric → pass | Baseline (source) | Receipt |
|---|---|---|---|---|
| E1 | "a nightly rehearsal runs the real publish path, upload excluded, and is green three nights in a row" | Row 1's scheduled rehearsal concludes `success` on 3 consecutive nights, each on that night's `main` head | **0** scheduled workflows run `cargo publish`, `cargo package` or `make publish` (§6 C3; its one hit is a comment, `.github/workflows/coverage-nightly.yml:211`). The nightly clean-room job, which lives outside this repo, builds and tests; it has no publish, package or dry-run step | 3 run IDs and their head SHAs |
| E2 | "no check has been red on main for more than 24 h" | For every check that reports on a `main` commit, the longest red stretch in the 7 days before the tag is ≤ 24 h | Successful / total scheduled runs on `main`, 2026-09-26 to 10-02: `mutants-nightly.yml` 0/5, `toolchain-ceiling.yml` 0/7, `guards-nightly.yml` 1/7 (Actions API, read 2026-10-02 11:17–11:30Z; §6 C9). Longest red stretch per check: **not_measured** (needs the API; S1) | Row 2's meter table at the tag |
| E3 | "a version-only bump reuses the saved results: bump to tag under 30 minutes" | (a) a diff of version lines only is judged the same code and the saved receipts are reused; (b) bump commit → tag < 30 min | (a) **No.** `scripts/release/release_readiness.sh:58-97` (`receipts_commit`) withholds the receipts when the measured and release commits differ anywhere outside `evidence/` (:91), and a bump changes `Cargo.toml` and `Cargo.lock`. (b) 0.66.0 2.42 h · 0.67.0 15.78 h · 0.68.1 1.80 h · 0.68.2 0.93 h · 0.69.1 31.80 h → **0 of 5** under 30 min (§6 C2) | Bump SHA, tag time, and the check's reuse line |
| E4 | ""required to merge, to tag, to publish" is one list in code: one definition of release-ready, zero contradictions" | Requirements stated outside the one list = 0; contradictions = 0 | **151** gate definitions, **93** contradictions (build-kaizen baseline, 2026-10-02/03: 76 of the 151 are in this repo @ `989cb012e5`, 75 outside it — operator rulings 31, agent memory 23, the fleet infra repo 21). The 2 required checks on `main` (`ci / gate`, `workspace-test`) live in GitHub settings, not in code | Row 4's checker: 0 unlisted, 0 orphaned, 0 contradictions |
| E5 | "tag to crates.io for 0.70.2 itself is under 2 hours" | The newest crates.io `created_at` at 0.70.2 over the crates of `scripts/release/publish-order.txt` at the tag − the tag's tagger date < 2 h (row 5) | 0.66.0 0.65 h · 0.67.0 1.02 h · 0.68.1 7.57 h · 0.68.2 10.17 h · 0.69.1 4.01 h → **2 of 5** under 2 h (§6 C1, C2). These time the root crate `aprender` alone. At v0.69.1 two crates publish after it, and before v0.69.0 the order file does not exist (§6 C13) | Row 5's timer at the tag: one line per crate, and the E5 line |
| E6 | "the known-failure P0 tickets from 0.70.1 are closed or re-ruled" | Every P0 ticket of a failure that 0.70.1 ships red is closed with a red→green receipt, or carries a new operator ruling. The set: the tickets the `v0.70.1` release notes name for its old failures (C280 item 6), plus each `known_red` ticket on the tag | 6 tickets from the 2026-10-03 classification, #4661–#4666 (row 6); states not re-read (S0). C280 item 6 adds no `known_red` entry: the ladder at the measured commit `cc4463f` is identical to the one at `316dee2cd4`, and its `known_red` holds one entry, rung `qwen35-0.8b-q4km`, ticket #4030 (§6 C7). The release notes do not exist yet (§6 C8) | One line per ticket: the closing PR and its receipt, or the ruling id |
| P4 | "no check may be enforced at publish until its inputs have been produced green on main, nightly, three times" (C280 item 12) | Checks the publish preflight enforces whose inputs have not been produced green on `main` by a nightly run three times = 0 (whether the three must be consecutive is §4 Q8) | The preflight's verdict is the AND of R1–R8 (`scripts/check_publish_preflight.sh:12-45`). Three of them judge an input that an earlier run produced: R5 the dogfood receipt (:32-34), R7 the per-host model-matrix receipts (:35-39), R8 the committed evidence graded through `release_readiness.sh` (:40-45). Of the 17 scheduled workflows, **0** run `scripts/dogfood.sh`, `check_model_ladder.sh`, `release_readiness.sh` or the preflight (§6 C10); timers outside this repo are not measured here | Row 10's table at the 0.70.2 tag: per check, the run IDs of its three nightly greens, or the operator's sign-off |

P1–P3 (C279 items 6 and 11) left this table when C280 replaced C279 (decision D4): rows 7 and 8 are carried
outside the exit bar, and row 9 is folded into row 6. P4 is C280 item 12 (decision D5).

## 2. Rows

**Prior art:** the build-kaizen Phase 0 countermeasures (2026-10-02). Each row cites the ones it reuses.

| K | Countermeasure | Tickets (state at the 2026-10-02 read) |
|---|---|---|
| K1 | Wire `make gate` (BSE-16) and predict-check (BSE-14) into pre-push. Today `.githooks/pre-push` runs `scripts/ci_guards.sh` only when `APR_PREPUSH_GUARDS=1` (:29-31), and calls neither | — |
| K2 | Bind the review record to the patch-id, so a re-push of the same patch keeps its review | #4635 |
| K3 | Gate admission as code, plus a projected-finish alarm | #4647 (done); #4648, #4649, #4652, #4653, #4654 |
| K4 | One push per run on release PRs; cancel only stale runs; stop orphans | #4645; #4637, #4638, #4644 |
| K5 | A PR-size guard (≤ 4 rows / ≤ 200 files), plus a nightly clean-room and publish dry-run on `main` | — |
| K6 | `make regen`: one writer for derived files | — |
| K7 | Shard `x86-main` toward ≤ 30 min, plus a merge-queue skip (needs a ruling; §4 Q1) | — |

The Phase 0 measurements the rows lean on (2026-09-26 to 10-02, Actions API): 62% of CI runs cancelled; PR CI p50
62 min, p90 242 min; `x86-main` p50 44 min, p90 104 min; the clean-room leg waited up to 642 min for a runner; the
merge queue re-tested an already-green tree in 3 of 10 sampled PRs (#4613, #4632, #4647); about 18 of 34
release-path reds were bookkeeping; the mutants shard p90 was 720 min, which is its timeout.

Every row lands as a ratchet with a red→green falsifier and a first green on a real PR. A row that changes a
checker follows CHECKER-BOOTSTRAP.

### Row 1 — E1: a nightly rehearsal of the real publish path

- **Why it fails today.** No scheduled job runs the publish path (E1 baseline). `make publish` with `PUBLISH_DRY_RUN`
  passes `--dry-run --no-verify` (`Makefile:1412-1414`), so it never compiles the packaged tarballs.
  `scripts/release/publish_strict.sh --plan` prints the order and uploads nothing, but it does not package either.
- **Mechanism.** A scheduled workflow runs the release's own sequence — the preflight
  (`scripts/check_publish_preflight.sh`, PMAT-745), the tarball compile (`scripts/release/rc_publish_gate.sh --verify`,
  #4287, which drops `--no-verify` so cargo unpacks and builds each tarball), and `publish_strict.sh` in
  `scripts/release/publish-order.txt` order — with only the upload call replaced. A structure test pins that the
  rehearsal and the release share one entry and differ in that one flag.
- **Inputs.** A nightly `main` head has no tag and no release ref, so R3 (the tag `v<version>` points at HEAD) and R4
  (HEAD is an ancestor of `origin/release/<version>`; a missing release ref refuses) can hold only against a tag and a
  release ref that the rehearsal makes in its own clone and never pushes (`scripts/check_publish_preflight.sh:18-23`).
  R5, R7 and R8 need receipts for that head, which row 10's nightly producers make. A rule the rehearsal cannot run
  prints `not_measured`, never pass, and E1 counts only nights on which all eight ran.
- **Prior art.** K5, its nightly half.
- **Falsifier.** Plant a CB-510-class defect (an `include!()` file left out of the package) on a fixture branch: the
  rehearsal goes red at the compile step; remove it: green. Second plant: a crate missing from `publish-order.txt` → red.
- **First green.** The first scheduled run on `main` after merge. E1 needs three nights in a row.
- **Gate impact.** Adds a scheduled workflow; keeps every gate, with its red/green proof in the PR. Not a required check.
  Depends on row 10 for the receipts R5, R7 and R8 judge.

### Row 2 — E2: a red-age meter with a 24 h andon

- **Why it fails today.** Nothing measures how long a check has been red on `main`, and scheduled checks stay red for
  days (E2 baseline: 0/5, 0/7, 1/7).
- **Mechanism.** A meter computes, for each check on `main`, the age of its current red stretch (the first red after
  the last green, to now). At 24 h it exits red and prints the check and its age; its value at the tag is E2's
  receipt. Before any check becomes blocking, K3 admission applies (a first-green receipt, p90 runtime ≤ 50% of the
  check's time limit, a fleet-hours budget). K1 and K6 keep reds from reaching `main`: pre-push runs the gate, and one
  writer regenerates derived files.
- **Prior art.** K1, K3, K6.
- **Falsifier.** Fixture histories: red for 25 h → meter red; red for 23 h → green; a green result resets the age; a
  check with no result in the window → `not_measured`, never green.
- **Built (S0, branch only).** `build-kaizen/r2-red-age` @ `a0bc666676`, no PR: `scripts/release/red_age.sh` and
  `contracts/red-age-v1.yaml`. The exit code is the andon, each check's red age at `--as-of`; the table also prints
  each check's 7-day maximum as E2's receipt column (decision D6). A green at attempt 2 or later is red, as in row 10 (decision D7).
  One TSV holds every check, and the exit is worst-of (decision D8). Receipts: case table 71/71 rows, 27/27 planted mutants
  killed, the contracts gate 8 of 8 steps. Waits for S1: the run-history fetcher (Actions API), `cargo test` of the
  contract crates, CI wiring.
- **First green.** Seven days on `main` with every check ≤ 24 h.
- **Gate impact.** "Check" means every check that reports on a `main` commit, scheduled ones included: the literal
  reading of C277 item 6. Narrowing it is §4 Q2.

### Row 3 — E3: a version-only bump reuses the saved results

- **Why it fails today.** `receipts_commit()` accepts only a diff confined to `evidence/` (E3 baseline (a)). A bump
  that waits for a full PR CI cycle cannot fit in 30 min (PR CI p50 62 min, p90 242 min).
- **Mechanism.** One "version-only" definition, inside row 8's function: the diff touches only the `version` fields
  of workspace manifests and the matching `Cargo.lock` `version` lines, and no dependency is added, removed or
  re-pinned. It reuses `lock_dep_change` (`scripts/lib/ladder_equiv.py:35`) and `lock_delta` (:48). The bump-to-tag
  path re-runs only what a version change can change (§4 Q6) and reuses the saved receipts, printing a proof line.
- **Prior art.** K2 (a record bound to the patch, not the push), K4 (one push per run), K7 (its shard half).
- **Falsifier.** Planted pairs: version-only → accepted, with the proof line; version plus a dependency change in
  `Cargo.lock` → refused; version plus one source line → refused; no diff → accepted as same.
- **First green.** The 0.70.2 bump itself: bump SHA → tag in under 30 min, with the reuse line in the receipt.
- **Gate impact.** Changes what a release check accepts: two non-author reviewers and operator sign-off on the PR
  (§4 Q3). Depends on row 8.

### Row 4 — E4: one list of release requirements, in code

- **Why it fails today.** 151 definitions across scripts, workflows, contracts, specs, docs, rulings and memory, with
  93 contradictions (E4 baseline). The required checks live in GitHub settings. `release_readiness.sh` exists because
  two hand-copied call sites "would drift" (:11-13); the same holds for the whole list.
- **Mechanism.** One `pv`-validated contract (working name `contracts/release-ready-v1.yaml`). Each requirement has an
  id, its checker, `applies_to` (a subset of merge, tag, publish), its provenance (ruling, ticket or spec section)
  and, when it applies to publish, its nightly producer (row 10). Every consumer reads the list. A checker fails on a
  requirement found outside the list (unlisted), a listed requirement with no checker (orphaned), and two entries that
  disagree (contradiction). A read-only probe compares the merge set with GitHub's required checks; changing those
  stays with the operator.
- **Prior art.** K3 (admission as code), the 151-row inventory, `release_readiness.sh` (one wrapper for two call sites).
- **Falsifier.** Plant a requirement in a script that the list lacks → red; delete a listed checker → red; plant two
  entries with conflicting thresholds → red.
- **First green.** The checker on `main` reads 0 unlisted, 0 orphaned, 0 contradictions.
- **Gate impact.** A contradiction that no ruling settles is listed for the operator, not decided here. The first PR
  commits the in-repo inventory (76 rows @ `989cb012e5`), so the count re-runs from this repo.

### Row 5 — E5: tag to crates.io in under 2 h

- **Why it fails today.** 3 of the last 5 releases took 4.01–10.17 h (E5 baseline). The clean-room job runs in about
  19 min (C277 item 1), but its leg waited up to 642 min for a runner.
- **Mechanism.** Row 1 rehearses the exact publish path every night, so the tag meets no new failure class. A
  clean-room runner slot is reserved for release tags; that is an infra ticket, whose text goes to the cop, and this
  repo does not edit infra. A timer receipt runs from tag creation to the last crate's `created_at`.
- **Prior art.** K5, its nightly half.
- **Falsifier.** The timer's case table: 2 h 01 min → red; 1 h 59 min → green; a missing crates.io timestamp →
  `not_measured`.
- **Built (S0, branch only).** `build-kaizen/r5-tag-to-crates` @ `87b8bafdf4`, no PR:
  `scripts/release/tag_to_crates.sh` and `contracts/tag-to-crates-v1.yaml`. It judges crates.io reads that a fetcher
  took; it makes no network call. Inputs: `--version`, `--tag-time` (the annotated tag's tagger date; a lightweight
  tag gives no start), `--as-of`, the order file at the tag, and a TSV of reads (`crate`, `version`, `created_at`,
  `read_at`). The clock stops at the newest `created_at` over every crate of the order file, at `--version` (decision D9).
  The three facade crates that `publish_strict.sh` appends are not timed: they carry their own version, `0.4.0` at
  every tag from v0.66.0 to v0.70.0 (decision D11; §6 C13). Per crate: no row → `not_measured`; a `created_at` before the
  tag → `not_measured` (decision D10); published 2 h or more after the tag → red; not on crates.io at a read 2 h or more
  after the tag (the read clipped to `--as-of`) → red, under 2 h → `not_measured` (decision D12); published under 2 h after
  the tag → ok. One unreadable row makes the whole judgment `not_measured`. Exit 0 met, 1 red (red wins),
  2 `not_measured`, 3 caller error. `LIMIT = 7200` is a constant. A context line prints what `aprender` alone took,
  the baseline's method, never as the verdict. Receipts: case table 64/64 rows, 44/44 planted mutants killed,
  `bashrs` 0 findings, `pv validate` 0/0, the contracts gate 8 of 8 steps. Waits for S1: the reads fetcher (one
  request per second; a failed read is not a row), `cargo test` of the contract crates, CI wiring.
- **Baseline method.** The E5 baseline times the root crate `aprender` alone (§6 C1). `scripts/release/publish-order.txt`
  has 71 crates at v0.69.x (`aprender` at line 69) and 73 at v0.70.0 (line 71), so two crates publish after
  `aprender` from v0.69.0 on. The file is absent at v0.66.0 to v0.68.2, so the order-file method can re-measure
  v0.69.0 and later only (§6 C13).
- **A measuring defect on `main`.** `scripts/release/release_wall_time.py` turns any failed crates.io read (a
  timeout, a 5xx, a 429) into a crate with no such version: `_crates_io` returns `None` on any exception (:61-66),
  and `measure` adds that crate to `missing` (:82). "Not published" and "not known" become one fact there. Fixing
  it is a ticket for S1.
- **First green.** 0.70.2's own tag.
- **Gate impact.** None in this repo.

### Row 6 — E6: the known-failure P0 tickets from 0.70.1

| Ticket | Model file | Failure, as classified 2026-10-03 |
|---|---|---|
| #4661 | `Qwen2.5-0.5B-Instruct-f16.gguf` | `serve` `/api/chat` answers gibberish, with `stream` false and true |
| #4662 | `Qwen3-Coder-30B-A3B-Q4_K_M.gguf` | gibberish |
| #4663 | `Qwen3.5-0.8B-IQ4_XS.gguf` | GPU fell back to CPU; `qa` exit 5, `run` rc 14; no measured thinking budget |
| #4664 | `Qwen3.5-0.8B-UD-IQ2_XXS.gguf` | wrong answer (`red_model` #4004, kept: C280 item 6 withdrew the swap) and a GPU-to-CPU fallback; folded from row 9 |
| #4665 | `Qwen3.5-35B-A3B-UD-IQ4_XS.gguf` | `red_unsupported` #3977: a refusal, re-proven, not blocking |
| #4666 | `qwen35-0.8b-q4km` | the think block; the one `known_red` entry on the measured commit is this rung's, ticket #4030 (E6) |

- **Why it fails today.** Each is an old failure (classified under rulings C272 5b / C276) that ships red with
  0.70.1. Under C280 item 6 it stays red in the printed evidence, in the release notes and in its P0 ticket, and no
  `known_red` entry is added for it.
- **Mechanism.** One defect = one ticket = one PR, each with a red→green test on the named file (a pinned binary,
  the file's sha256), or a new operator ruling. The set is re-read from the `v0.70.1` release notes and tag (§6 C8);
  a ticket found there and missing here joins this table. #4664 (folded from row 9, decision D4) closes by fixing
  the fallback, or by refusing the file by name with an error that names the file and #4664, the two remedies C279
  item 6 named; the decision quorum picks at row start (§4 Q5).
- **Prior art.** None of K1–K7. The `known_red` schema at `scripts/check_model_ladder.sh:349-363`: each entry names a
  model, its sha256, a clause, a ticket and a ruling, and an entry that covers nothing is refused as stale.
- **Falsifier.** Per ticket, the named file's failing row on the pinned binary: red before, green after.
- **First green.** Per ticket.
- **Gate impact.** No gate changes: a fix turns a printed red green, and no mark is added. Fixing the think block also
  retires its `known_red` entry (#4030), because the ladder refuses an entry that covers nothing as STALE
  (`scripts/check_model_ladder.sh:353-354`). Needs the model files and a GPU host, so S1 only.

### Row 7 — carried, not in the exit bar: rulings and marks do not void measurements

No operator mandate since C280 replaced C279 (decision D4); a measured defect. A yes to §4 Q7 moves it back into §1
unchanged.

- **Why it fails today.** The marks live in `contracts/model-capability-ladder-v1.yaml` (`hotfix_scope:` :75,
  `emergency_scopes:` :99, `known_red:` :111, `red_model:` :187, `red_unsupported:` :200; §6 C7), and
  `crates/aprender-contracts/build.rs:93-120` reads every `contracts/*.yaml` whose name lacks `binding` and emits
  `rerun-if-changed` for each, so a mark is a build input (§6 C6). The release check treats any diff outside
  `evidence/` as different code (`release_readiness.sh:91`). For 0.70.1 this ruled out C279's scoped exception by its
  own condition (C279 item 7). C280 instead records an emergency scope: outside `evidence/`, the final commit differs
  from the measured commit `cc4463f` only in one ladder entry and in the preflight script and its self-test, and the
  binary measured is the one built at `cc4463f` (C280 items 2, 4, 5, 9).
- **Mechanism.** One of three options, decided by the decision quorum when the row starts (§4 Q4): (a) move the marks
  to a file no build reads, which row 8 classifies as marks; (b) keep the file and make the build script skip it,
  with a test that nothing compiled reads it; (c) define "same code" by cargo's dep-info build inputs, not by paths.
- **Prior art.** The ladder's hotfix arm (#3710 r3, #4022); the recorded emergency scopes (0.69.1, and 0.70.1 under
  C280 items 2–5); C279 items 1–4, withdrawn by C280 item 6.
- **Falsifier.** Edit a `known_red` entry after a measured commit → the receipts stay valid; edit a source line →
  withheld. Plus a test that fails when any compiled code reads the marks file.
- **First green.** The first mark edit after the merge keeps its receipts.
- **Gate impact.** Changes what a release check accepts (§4 Q3). Depends on row 8.

### Row 8 — carried, not in the exit bar: one definition of "same code"

Carried as row 7 is (decision D4, §4 Q7). Row 3 (E3) depends on it: the version-only class is one of its classes,
so it lands before row 3 whatever the answer to Q7.

- **Why it fails today.** Two implementations can disagree about the same diff: `scripts/lib/ladder_equiv.py:79`
  `classify()`, used by `scripts/check_model_ladder.sh`, and `release_readiness.sh:58-97` `receipts_commit()`, a
  plain path diff with `evidence/` excluded (§6 C4, C5). The cut, the commit the receipts must match, is chosen by
  each caller, and a caller that names none gets HEAD (`scripts/check_model_ladder.sh:1295`).
  `scripts/dogfood.sh:448` runs each declared gate with no arguments, `check_model_ladder.sh` among them
  (`Cargo.toml:621-627`), and `scripts/cascade-publish.sh:614` runs the preflight with no `--scope` or
  `--cut-commit` (§6 C11).
- **Mechanism.** One function returns one of `same`, `evidence-only`, `version-only`, `marks-only`, `scoped-hotfix` or
  `not-same`, with a proof line naming the paths that decided it. `check_model_ladder.sh` and `release_readiness.sh`
  both call it. The cut is part of the input: a caller that names none is refused, never given HEAD. New code is bash
  (the build-kaizen rule); whether `classify()` is ported or called is decided when the row starts.
- **Prior art.** `classify()`, `lock_dep_change` and `lock_delta` in `scripts/lib/ladder_equiv.py`.
- **Falsifier.** A parity table: the same planted diffs through both callers give the same class. A structure test
  fails if either caller computes a diff itself; the mutation that turns it red reverts one caller to its own
  `git diff`. A caller that names no cut → refused.
- **First green.** Both checks green on `main`, calling the shared function.
- **Gate impact.** §4 Q3.

### Row 9 — folded into row 6

C279 item 6's P0 has no operator mandate since C280 (decision D4), and C280 item 6 withdrew the swap. In the ladder
contract the final commit adds one `emergency_scopes` entry and nothing else (C280 items 2 and 5), and `cc4463f`'s
ladder is identical to `main`'s (§6 C7), so the `red_model` entry (#4004) ships unchanged and there is nothing to
restore. The file's failures stay red in its P0 ticket, #4664, which E6 covers (row 6).

### Row 10 — P4: nothing enforced at publish before three nightly greens

- **Why it fails today.** R5, R7 and R8 judge inputs that an earlier run produced, and no scheduled workflow in this
  repo produces them (P4 baseline). The preflight names the release path's own T-1 steps as their producers
  (`scripts/check_publish_preflight.sh:32-45`). For 0.70.1, C280 item 4 printed R8's verdict as evidence under a
  recorded scope instead of enforcing it.
- **Mechanism.** Nightly producers on `main` for each of the three: the full CRUX lanes (the receipts R7 judges), the
  readiness check over that night's receipts (R8), and the dogfood receipt (R5). The CRUX lanes run with the GPU to
  themselves: a lane takes the GPU lock first and records, at its start and end, that no other process held the
  device. In row 4's list, each requirement that applies to publish names its nightly producer, and the preflight
  prints, per check, the run IDs of its last three nightly greens. The GPU scheduling half runs on hosts outside
  this repo; its ticket text goes to the cop.
- **Prior art.** Row 1 (the rehearsal consumes these receipts), row 4 (the list), K3 (a first-green receipt before a
  check blocks), K5 (its nightly half).
- **Falsifier.** Fixture histories for one check: three green nightly runs → `ready`, with the three run IDs; two →
  `not ready`; a night with no run → `not_measured`, never green. A planted second process on the GPU during a CRUX
  lane → that lane's receipt is void.
- **Built (S0, branch only).** `build-kaizen/r10-nightly-greens` @ `3045add39f`, no PR:
  `scripts/release/nightly_greens.sh` and `contracts/nightly-greens-v1.yaml`. It reads a run-history TSV, counts
  only scheduled runs on `main`, and gives each UTC night one state. It says ready only when the newest three
  nights are green and the newest is no older than the day before `--as-of`. A green at attempt 2 or later is red.
  Receipts: case table 42/42 rows, 15/15 planted mutants killed. Waits for S1: the run-history fetcher, `cargo test`
  of the contract crates, CI and preflight wiring.
- **First green.** The 0.70.2 preflight prints three nightly run IDs for each of R5, R7 and R8.
- **Gate impact.** Moving an enforced check (R5, R7 or R8 today) to nightly-only, or to printed evidence, is a
  report-only change to an existing gate, so the operator signs off on that list first (§4 Q9). The preflight holds
  that "report-only is a waiver and a waiver is a stop" (`scripts/check_publish_preflight.sh:318`, operator
  2026-09-28). If all three producers have three greens before the publish, no check changes mode. R1–R4 and R6
  judge the commit itself and have no produced input; they stay enforced (§4 Q10). Depends on rows 1 and 4.

## 3. Ratchet

This follows #4647 (operator ruling L31): a measured baseline, regression-only head-vs-base in the same run with the
same scanner, no stored limit, and a fleet-hours budget before any counter blocks.

| Counter | Baseline | Direction |
|---|---|---|
| Release requirements stated outside the one list | 151 | down only |
| Contradictions among them | 93 | down only |
| Known-failure tickets from 0.70.1 still open | 6 (#4661–#4666; the set is fixed by the release notes and the tag, §6 C8) | down only |
| Implementations of "same code" (row 8, carried) | 2 | to 1, then never above 1 |
| Checks red on `main` for more than 24 h | not_measured | down only |
| Checks enforced at publish with no nightly producer on `main` | 3 (R5, R7, R8; §6 C10) | down only |

E1, E3 and E5 are per-release times. They are recorded as receipts, not counters.

## 4. Open questions

| Q | Question | Who decides | What it blocks |
|---|---|---|---|
| Q1 | K7's merge-queue skip: may the queue skip re-testing a tree it already tested green (3 of 10 sampled: #4613, #4632, #4647)? | The operator only: it changes a gate. The decision quorum voted 3/3 that it is not a row (D3) | Nothing in this plan |
| Q2 | E2 counts every check on `main`, scheduled ones included. Narrow it? | The operator only, if ever asked | Nothing: the literal reading applies |
| Q3 | Rows 3, 7 and 8 change what a release check accepts | Each PR: two reviewers, neither the author, plus operator sign-off (as C279 item 8 and C280 item 4 required) | The merge of those rows |
| Q4 | Row 7: option (a), (b) or (c)? | The decision quorum, at row start | Row 7 |
| Q5 | #4664 (row 6, folded from row 9): fix the fallback, or refuse the file by name? | The decision quorum, at row start | #4664 |
| Q6 | Which runs does a bump commit still need before the tag? | Measured on the 0.70.1 bump; decided at row 3 start | Row 3 |
| Q7 | Does C279 item 11 still stand? C280 replaces C279 and withdraws its option B (item 6), but names neither item 11 nor its P0s | The operator only | Nothing now: rows 7 and 8 are carried outside the exit bar (D4); a yes moves them into §1 |
| Q8 | C280 item 12 says "three times"; E1 says "three nights in a row". Must item 12's three be consecutive? | The operator only: it sets when a check may be enforced | Nothing now: row 10 prints both counts until answered |
| Q9 | A check enforced at publish whose producer lacks three nightly greens: is it printed as evidence (as C280 item 4 did for R8), or does the publish stop (`check_publish_preflight.sh:318`)? Which checks? | The operator only: a report-only change to an existing gate | The publish-side half of row 10; nothing if R5, R7 and R8 have three nightly greens before the 0.70.2 publish |
| Q10 | Does C280 item 12 cover R1–R4 and R6? They judge the commit itself, and R3 and R4 cannot hold on a nightly `main` head | The operator, if ever asked | Nothing: they stay enforced, so no gate changes |

## 5. Out of scope

- K5's PR-size half: from 0.71 the operator's PR rule already caps a PR at k ≤ 4 rows.
- K7's merge-queue skip (§4 Q1).
- Items from the 2026-10-02 review that serve no exit condition: the fake `##[error]` lines (#4596, #4610); mutation
  testing as non-blocking on a dedicated host; paging the operator within 5 min on a P0; rulings kept as a versioned
  file; porting `scripts/ci/fat_driver.py` off Python; a pmat complexity-hook ticket.
- Lean T5 at its 24 GB memory cap (C279 item 11: "ticket, no action today"); C280 item 10 keeps Lean T5 among the
  0.70.1 checks.
- The documentation-drift ticket (29 docs), filed after S1.
- C280 items 1–11: the 0.70.1 release itself, under its recorded emergency scope. This plan starts when 0.70.1 is
  live (§0).

## 6. Commands

All read-only. `<sha>` = `316dee2cd4`. C1 is one unauthenticated crates.io read; C8's second line and C9 are the
only GitHub API calls, and both wait for S1.

```bash
# C1  crates.io publish times (E5)
curl -s -A 'bld-001 (paiml)' https://crates.io/api/v1/crates/aprender \
  | jq -r '.versions[] | .num + " " + .created_at'
# C2  tag time and bump time of one release (E3, E5); repeat for 0.66.0, 0.67.0, 0.68.1, 0.68.2, 0.69.1.
#     The bump time is the committer date of the first commit in the tag's history that adds the version line.
git for-each-ref --format='%(creatordate:iso-strict)' refs/tags/v0.69.1
git log --reverse --format=%cI -S'version = "0.69.1"' v0.69.1 -- Cargo.toml | head -1
# C3  scheduled workflows that publish or package (E1)
for f in .github/workflows/*.yml; do
  grep -q -E '^\s+schedule:' "$f" || continue
  grep -n -E 'cargo publish|cargo package|make publish|publish[-_]dry|publish_check|publish-check' "$f" /dev/null
done
# C4  the receipts-commit rule (E3, row 8)
git show <sha>:scripts/release/release_readiness.sh | sed -n '58,97p'
# C5  the ladder's same-code function (row 8)
git show <sha>:scripts/lib/ladder_equiv.py | sed -n '79,107p'
# C6  the build script that reads contracts/*.yaml (row 7)
git show <sha>:crates/aprender-contracts/build.rs | sed -n '93,120p'
# C7  the marks in the ladder (E6, rows 7 and 9), and the ladder at the measured commit (the diff prints nothing)
git show <sha>:contracts/model-capability-ladder-v1.yaml \
  | grep -n -E '^\s*(hotfix_scope|emergency_scopes|known_red|red_model|red_unsupported):'
git diff <sha> cc4463f -- contracts/model-capability-ladder-v1.yaml
# C8  E6, once v0.70.1 exists: the known_red tickets on the tag (local git; on cc4463f it prints "#4030"), and
#     the tickets its release notes name (one API read, S1)
git show v0.70.1:contracts/model-capability-ladder-v1.yaml \
  | sed -n '/^  known_red:/,/^  [a-z_]*:/p' | grep -E 'ticket:'
gh release view v0.70.1 --repo paiml/aprender --json body --jq .body | grep -oE '#[0-9]+' | sort -u
# C9  E2, S1 only: the conclusions of one scheduled workflow on main over one week
gh api 'repos/paiml/aprender/actions/workflows/toolchain-ceiling.yml/runs?branch=main&created=2026-09-26..2026-10-02&per_page=100' \
  --jq '[.workflow_runs[].conclusion] | group_by(.) | map({(.[0] // "none"): length}) | add'
# C10 P4: scheduled workflows that run a producer of R5, R7 or R8, or the preflight (prints nothing at <sha>;
#     17 of the 25 workflows have a schedule)
for f in $(git ls-tree --name-only <sha> .github/workflows/ | grep -E '\.ya?ml$'); do
  git show "<sha>:$f" | grep -q -E '^\s+schedule:' || continue
  git show "<sha>:$f" | grep -n -E 'dogfood\.sh|check_model_ladder|release_readiness|crux|check_publish_preflight' \
    | sed "s|^|$f:|"
done
# C11 row 8: the cut defaults to HEAD, and two callers that name none
git show <sha>:scripts/check_model_ladder.sh | sed -n '1295p'
git show <sha>:scripts/dogfood.sh | sed -n '448p'
git show <sha>:Cargo.toml | sed -n '621,627p'
git show <sha>:scripts/cascade-publish.sh | sed -n '614p'
# C12 P4, row 10: the preflight's rules, and the line that makes report-only a stop
git show <sha>:scripts/check_publish_preflight.sh | sed -n '12,45p;318p'
# C13 E5, row 5: the crates the clock covers (local git). At v0.70.0: 73 lines, `aprender` at line 71; at v0.69.x:
#     71 lines, line 69; absent at v0.66.0, v0.67.0, v0.68.0, v0.68.1 and v0.68.2. The facades print version = "0.4.0".
git show v0.70.0:scripts/release/publish-order.txt | wc -l
git show v0.70.0:scripts/release/publish-order.txt | grep -nx aprender
for t in v0.66.0 v0.67.0 v0.68.0 v0.68.1 v0.68.2 v0.69.0 v0.69.1 v0.70.0; do
  git cat-file -e "${t}:scripts/release/publish-order.txt" 2>/dev/null && echo "$t present" || echo "$t absent"; done
git show v0.70.0:crates/facades/Cargo.toml | grep -m1 '^version'
TZ=UTC git for-each-ref --format='%(taggerdate:format-local:%Y-%m-%dT%H:%M:%SZ)' refs/tags/v0.70.0   # 2026-10-02T23:42:38Z
```

E4's 151 and 93 come from the build-kaizen baseline of 2026-10-02/03, an inventory kept outside this repo. Row 4's
first PR commits its in-repo part, so the count can be re-run from here.

## 7. BLD-002: release from green evidence (rows R0–R13)

BLD-002 (operator ruling C289, 2026-10-03) extends this plan: measure every night and release what is already green. Release day verifies evidence and uploads; it measures nothing new. The rows run in the order the ruling gives. Until 0.70.1 is live they stay on branches only (C277 item 3). Tickets are minted after that; until then, commits carry `BLD-002/Rn`. Each row gets a ticket, a contract, one planted falsifier that must go red, and before/after numbers from R0.

| BLD-002 row | Work | Rows of this plan it carries | State, 2026-10-04 |
|---|---|---|---|
| R0 | Baseline: step, duration, wait and first-pass yes/no for 0.70 and 0.70.1, plus the rolled first-pass yield | the baseline report | built on a branch: the step-table calculator (48 case rows, 16 planted mutants killed) and its contract; a step with no measured duration is listed UNMEASURED; the rolled first-pass yield over 0.70 and 0.70.1 is 0.000 (baseline data kept outside this repo) |
| R1 | One code identity H, used by the ladder judge, the readiness wrapper, the preflight and dogfood | row 8 | design; the path set waits on ruling request RQ-5 |
| R2 | An evidence store outside H, keyed by H | row 7 | built on a branch: the store script (82 case rows, 46 planted mutants killed) and its contract; the publish check reads the store only after RQ-6 rules on where it lives |
| R3 | The version bump moves to the start of a cycle | row 3 | not started |
| R4 | A nightly evidence train: every lane in parallel, one line out (`RELEASABLE H=…` or `NOT RELEASABLE: <check>`) | row 1 (the publish lane) | not started |
| R5 | Release = promote, rehearsed nightly up to the upload | row 5 | row 5's clock and fetcher are built on a branch |
| R6 | Split dogfood: coverage nightly, model tests in an optimized build, release-day dogfood in 15 min or less | none | waits on RQ-1 |
| R7 | Gate admission as code: a check may block only after three green nights | rows 2 and 10 | rows 2 and 10 are built on branches |
| R8 | A known-failure ratchet against the last release | rows 6 and 9 | not started |
| R9 | One definition of release-ready | row 4 | design decided by quorum; build in progress |
| R10 | Flow control as code | none | not started |
| R11 | The nightly train builds the binaries; a release is public only with every asset | none | not started |
| R12 | Python off the release path: inventory now, no new Python | none | not started |
| R13 | The tag-day coverage gate names a CI job that no longer exists | none | found 2026-10-03; see below |

### R13 — the tag-day coverage gate names a CI job that no longer exists

Found 2026-10-03 while answering RQ-1. It does not affect the 0.70.1 release, whose release steps never call the gate. Any 0.70.2 release driven by `scripts/release/autopilot.sh` hits it.

- **Fact.**
  - `scripts/release/tag_coverage_gate.sh:26` sets `JOB='ci / coverage'`, but `.github/workflows/ci.yml` has no job by that name: `git show <sha>:.github/workflows/ci.yml | grep -c 'ci / coverage'` prints 0, both at 316dee2cd4 and at the 0.70.1 release head. `evidence/fleet/history.jsonl:3` still records a job of that name.
  - The autopilot calls the gate in step 5 (`autopilot.sh:312`). By then, step 3 has already pushed the tag and published the GitHub release.
  - The gate polls the tag-push CI run for up to 90 × 60 s (`TCG_TRIES`, `TCG_SLEEP`, :57), then refuses.
  - The release then stops with its tag and its GitHub release public and no crate published.
- **The comment is wrong too.** The gate and its caller say it enforces `COV_FLOOR`. It cannot: the CI coverage section sets no floor (`ci/sections.yml:22-26`). Only `make coverage` (in `coverage-nightly.yml` and in dogfood) enforces the floor.
- **Five whys.**
  1. Why does the release stop after the tag? The gate waits for a job that never appears.
  2. Why does the job never appear? Its name left the workflow.
  3. Why did nobody notice? No check compares a gate's job names with the workflow at the commit it judges.
  4. Why is there no such check? Gates are admitted by hand, with no registry (R7).
  - Mechanism: R7's registry records each gate's inputs, and admission resolves them.
- **Countermeasure.**
  1. The gate resolves its job name in `.github/workflows/ci.yml` at the commit before it waits. The autopilot runs that check before step 3, and a missing name refuses at once, naming the job.
  2. With R6 and R2, the gate reads the nightly coverage result for H instead of waiting for a tag-push job.
- **Falsifier (must go RED today).** A case-table row where the workflow at the commit has no job of that name must refuse before the tag, within 1 minute, naming the job. Today the tag is pushed first and the refusal comes up to 90 minutes later.
- **Before/after.** Before: up to 90 min of waiting, then a stop after the tag. After: a refusal before the tag, within 1 min.

## Quorum record: decision quorum, 2026-10-03 (aprender-a7)

The operator's standing rule, verbatim (2026-10-03): "use quorum for decision, never block on me". The quorum may
never approve a waiver, a gate or threshold change, a ruleset, secret or credential change, a force-push, a tag or a
publish (C213). Those stay with the operator: §4 Q1, Q3, Q7, Q8 and Q9.

Lanes: gpt-oss-120b (medium) and claude-sonnet-5-5 ×2, in plan mode. Launched 10:09:26Z; all three exited 0 by
10:09:41Z. Each lane answered from the same brief of facts: the C277 and C279 text, the layout of the existing train
plans, and the K table.

Round 2 (D4, D5), after C280 replaced C279: the same three lanes, launched 10:53:46Z; all three exited 0 by
10:54:01Z. Their brief quoted C280's header and items 6 and 12, C279 items 6 and 11, and the baselines of rows 7–9.

Rounds 3–5 (D6–D12) shaped the row 2 meter and the row 5 timer, built on branches in S0. The author is Opus 5.5, so
the lanes were the same three models: claude-sonnet-5-5 ×2 and gpt-oss-120b. There was no Gemini lane, because C10
keeps Gemini for release votes. Round 3 (D6–D8) launched 12:23:12Z and all three lanes had answered by 12:23:26Z;
round 4 (D9, D10) ran 13:32:55Z to 13:33:10Z; round 5 (D11, D12) ran 13:54:09Z to 13:54:23Z. Each brief quoted the
operator condition (C277 item 6), the row's text, facts from local git, and what was already settled. Each lane
also named the strongest argument against one of its answers (D7, D9, D12).

| D | Question | Tally | Applied as | Dissent and risks named | Reversible by |
|---|---|---|---|---|---|
| D1 | Where does the plan live? | A 3/3 | A new file, `docs/specifications/BLD-001-release-factory.md`: the train-plan layout plus a live-state selector, scoped to 0.70.2 | None. Risk named: a later move could duplicate or diverge, so a pointer from APR-RELEASE-001 goes in the first PR after S1 | Moving the file and leaving a pointer |
| D2 | What are the rows? | B 2/3 | One row per C277 item 6 condition, plus the P0s of C279 items 5, 6 and 11. K items are cited as prior art; a K item that serves no row is out of scope | gpt-oss voted A (one row per K item). Risks named: B could drop a high-value K item, or savings the operator would have approved | Adding a row for a K item in a later PR |
| D3 | Is K7's merge-queue skip a row? | no 3/3 | Not a row: it changes a gate (C213). It is §4 Q1, for the operator | None | An operator ruling |
| D4 | What happens to P1–P3 (rows 7–9) now that C280 replaces C279? | A 2/3 | Out of the exit bar. Rows 7 and 8 are carried as measured defects with no operator mandate; row 9 is folded into row 6 (#4664); §4 Q7 asks whether C279 item 11 still stands | gpt-oss voted B (drop rows 7–9; its risk: dropping erases the trace of earlier rulings). Risks named for A: if C279 item 11 still stands, A drops a live P0 from the bar; if it does not, rows 7 and 8 are unmandated scope | An operator answer to Q7 moves rows 7 and 8 back into §1 unchanged |
| D5 | Where does C280 item 12 go? | A 3/3 | P4 in §1 with the operator's words, and row 10 with its own ticket, falsifier and first green; depends on rows 1 and 4; moving an enforced check to nightly-only or to printed evidence waits for the operator (§4 Q9) | None | Folding row 10 into rows 1 and 4 in a later PR |
| D6 | Row 2's meter: what sets its exit code? | A 3/3 | The andon: each check's red age at `--as-of` (≥ 24 h red; no result in the 7 days before `--as-of` → `not_measured`). The table also prints each check's 7-day maximum, E2's receipt column, and an E2 line | None | Editing row 2's meter |
| D7 | Row 2: a run that succeeded only at attempt 2 or later | A 3/3 | Not a green, as in row 10: it starts or continues a red stretch and never resets the age | None. Risk named by both claude lanes: a rerun that passes can hold a check red past 24 h on infrastructure flakiness, not code | Editing row 2's meter |
| D8 | Row 2's input | A 3/3 | One TSV for all checks: a `check` column, then row 10's six columns. One invocation; worst-of exit (any red 1, else any `not_measured` 2, else 0); a header-only file is `not_measured` | None | Editing row 2's meter |
| D9 | Row 5: where does the clock stop? | A 3/3 | At the newest `created_at` over all crates of `publish-order.txt` at the tag, not at the root crate `aprender`: two crates publish after it | None. Risks named: one crates.io read per crate (73) strains rate limits and makes `not_measured` likelier; the 5-release baseline used the one-crate method, so the two are not comparable | Editing row 5's timer |
| D10 | Row 5: a crate whose `created_at` is earlier than the tag | A 3/3 | `not_measured`, naming the crate: the clock has no valid start | None | Editing row 5's timer |
| D11 | Row 5: which crates does the clock cover? | A 3/3 | The crates of `publish-order.txt` at the tag, all at `--version`. The three facades are out: they carry their own version and are skipped when already live | None | Editing row 5's timer |
| D12 | Row 5: each read's time | A 3/3 | A fourth column, `read_at`. A crate missing at its read is known missing only up to the earlier of `read_at` and `--as-of`, and is red only when that is 2 h or more after the tag | None. Risks named: the fetcher supplies `read_at`, so a fetcher that reports it wrongly defeats the check; a fourth column adds input surface and fetcher work | Editing row 5's timer |

`quorum: rounds 5 | width 3 | verdicts D1 A 3/3, D2 B 2/3 (oss A), D3 no 3/3, D4 A 2/3 (oss B), D5 A 3/3, D6 A 3/3, D7 A 3/3, D8 A 3/3, D9 A 3/3, D10 A 3/3, D11 A 3/3, D12 A 3/3 | overridden no`
