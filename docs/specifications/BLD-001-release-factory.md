# BLD-001 — Release factory: train 0.70.2 "build" plan

**Status:** plan for operator review. Nothing is applied: no tickets minted, no PRs opened, no CI runs started, no
gate changed. Branch-only until 0.70.1 is on crates.io (operator ruling C277 item 3, 2026-10-03).
**Ticket:** proposed to the cop, not yet minted · **kind:** docs · **Train:** 0.70.2 "build" (C277 items 2, 5, 6) ·
**Owner:** aprender-a7 (build-kaizen)

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

And the 0.70.2 P0s named in operator ruling C279 (2026-10-03), verbatim:

> 6. … P0 ticket for 0.70.2: restore the red_model entry, and fix the fallback or refuse the file by name.
>
> 11. P0 for 0.70.2: rulings and known-failure marks must not void measured results, and the two checks use one
> definition of "same code".

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
| E5 | "tag to crates.io for 0.70.2 itself is under 2 hours" | crates.io `created_at` of 0.70.2 − tag creator date < 2 h | 0.66.0 0.65 h · 0.67.0 1.02 h · 0.68.1 7.57 h · 0.68.2 10.17 h · 0.69.1 4.01 h → **2 of 5** under 2 h (§6 C1, C2) | Tag time and crates.io `created_at` |
| E6 | "the known-failure P0 tickets from 0.70.1 are closed or re-ruled" | Every `known_red` ticket on the `v0.70.1` tag is closed with a red→green receipt, or carries a new operator ruling | 6 tickets from the 2026-10-03 classification, #4661–#4666 (row 6). States not re-read (S0). The full set is fixed by the tag, which does not exist yet (§6 C8) | One line per ticket: the closing PR and its receipt, or the ruling id |
| P1 | "rulings and known-failure marks must not void measured results" (C279 item 11) | Editing a mark (`hotfix_scope`, `known_red`, `red_model`, `red_unsupported`) leaves the measured receipts valid | The marks live in `contracts/model-capability-ladder-v1.yaml` (`hotfix_scope:` :75, `known_red:` :111, `red_model:` :187, `red_unsupported:` :200; §6 C7). `crates/aprender-contracts/build.rs:93-120` reads every `contracts/*.yaml` whose name lacks `binding` and emits `rerun-if-changed` for each, so a mark is a build input (§6 C6). For 0.70.1 this voided the scoped exception (C279 item 7), and the train ran the re-measure path (item 10) | Row 7's falsifier, green |
| P2 | "the two checks use one definition of "same code"" (C279 item 11) | Implementations of "same code" = 1, called by both checks | **2**: `scripts/lib/ladder_equiv.py:79` `classify()`, used by `scripts/check_model_ladder.sh`, and `release_readiness.sh:58-97` `receipts_commit()`, a plain path diff with `evidence/` excluded (§6 C4, C5) | Row 8's structure test, green |
| P3 | "restore the red_model entry, and fix the fallback or refuse the file by name" (C279 item 6) | On the 0.70.2 tag: `red_model` names `Qwen3.5-0.8B-UD-IQ2_XXS.gguf` → #4004, and its GPU-to-CPU fallback is fixed or the file is refused by name | Present on `main` @ `316dee2cd4` (:187-190). Swapped for a `known_red` entry for 0.70.1 only (C279 item 6) | The ladder on the tag, and a trace of the file on a pinned binary |

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
- **Prior art.** K5, its nightly half.
- **Falsifier.** Plant a CB-510-class defect (an `include!()` file left out of the package) on a fixture branch: the
  rehearsal goes red at the compile step; remove it: green. Second plant: a crate missing from `publish-order.txt` → red.
- **First green.** The first scheduled run on `main` after merge. E1 needs three nights in a row.
- **Gate impact.** Adds a scheduled workflow; keeps every gate, with its red/green proof in the PR. Not a required check.

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
- **Mechanism.** One `pv`-validated contract (working name `contracts/release-ready-v1.yaml`). Each requirement has
  an id, its checker, `applies_to` (a subset of merge, tag, publish) and its provenance (ruling, ticket or spec
  section). Every consumer reads the list. A checker fails on a requirement found outside the list (unlisted), a
  listed requirement with no checker (orphaned), and two entries that disagree (contradiction). A read-only probe
  compares the merge set with GitHub's required checks; changing those stays with the operator.
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
- **First green.** 0.70.2's own tag.
- **Gate impact.** None in this repo.

### Row 6 — E6: the known-failure P0 tickets from 0.70.1

| Ticket | Model file | Failure, as classified 2026-10-03 |
|---|---|---|
| #4661 | `Qwen2.5-0.5B-Instruct-f16.gguf` | `serve` `/api/chat` answers gibberish, with `stream` false and true |
| #4662 | `Qwen3-Coder-30B-A3B-Q4_K_M.gguf` | gibberish |
| #4663 | `Qwen3.5-0.8B-IQ4_XS.gguf` | GPU fell back to CPU; `qa` exit 5, `run` rc 14; no measured thinking budget |
| #4664 | `Qwen3.5-0.8B-UD-IQ2_XXS.gguf` | wrong answer (`red_model` #4004) and a GPU-to-CPU fallback; row 9 |
| #4665 | `Qwen3.5-35B-A3B-UD-IQ4_XS.gguf` | `red_unsupported` #3977: a refusal, re-proven, not blocking |
| #4666 | `qwen35-0.8b-q4km` | the think block (C279 item 5) |

- **Why it fails today.** Each is a known red that ships with 0.70.1 under rulings C272 5b / C276.
- **Mechanism.** One defect = one ticket = one PR, each with a red→green test on the named file (a pinned binary,
  the file's sha256), or a new operator ruling. The set is re-read from the `v0.70.1` tag (§6 C8); a ticket found
  there and missing here joins this table.
- **Prior art.** None of K1–K7. The `known_red` schema at `scripts/check_model_ladder.sh:349-363`: each entry names a
  model, its sha256, a clause, a ticket and a ruling, and an entry that covers nothing is refused as stale.
- **Falsifier.** Per ticket, the named file's failing row on the pinned binary: red before, green after.
- **First green.** Per ticket.
- **Gate impact.** Removes `known_red` entries only (tightens). Needs the model files and a GPU host, so S1 only.

### Row 7 — P1: rulings and marks do not void measurements

- **Why it fails today.** The marks are build inputs (P1 baseline), and the release check treats any diff outside
  `evidence/` as different code.
- **Mechanism.** One of three options, decided by the decision quorum when the row starts (§4 Q4): (a) move the marks
  to a file no build reads, which row 8 classifies as marks; (b) keep the file and make the build script skip it,
  with a test that nothing compiled reads it; (c) define "same code" by cargo's dep-info build inputs, not by paths.
- **Prior art.** The ladder's hotfix arm (#3710 r3, #4022) and C279 items 1–4, the one scoped exception for 0.70.1.
- **Falsifier.** Edit a `known_red` entry after a measured commit → the receipts stay valid; edit a source line →
  withheld. Plus a test that fails when any compiled code reads the marks file.
- **First green.** The first mark edit after the merge keeps its receipts.
- **Gate impact.** Changes what a release check accepts (§4 Q3). Depends on row 8.

### Row 8 — P2: one definition of "same code"

- **Why it fails today.** Two implementations (P2 baseline) can disagree about the same diff.
- **Mechanism.** One function returns one of `same`, `evidence-only`, `version-only`, `marks-only`, `scoped-hotfix`
  or `not-same`, with a proof line naming the paths that decided it. `check_model_ladder.sh` and
  `release_readiness.sh` both call it. New code is bash (the build-kaizen rule); whether `classify()` is ported or
  called is decided when the row starts.
- **Prior art.** `classify()`, `lock_dep_change` and `lock_delta` in `scripts/lib/ladder_equiv.py`.
- **Falsifier.** A parity table: the same planted diffs through both callers give the same class. A structure test
  fails if either caller computes a diff itself; the mutation that turns it red reverts one caller to its own
  `git diff`.
- **First green.** Both checks green on `main`, calling the shared function.
- **Gate impact.** §4 Q3.

### Row 9 — P3: UD-IQ2_XXS on the 0.70.2 tag

- **Why it fails today.** For 0.70.1 only, the file's `red_model` entry (#4004) is swapped for a `known_red` entry
  (C279 item 6), and the GPU-to-CPU fallback is not fixed (#4664).
- **Mechanism.** Restore the `red_model` entry. Then either fix the fallback, or refuse the file by name with an
  error that names the file and #4664; the decision quorum picks at row start (§4 Q5).
- **Prior art.** None of K1–K7.
- **Falsifier.** A trace of the file on a pinned binary: before, the fallback; after, no fallback, or the refusal by
  name.
- **First green.** The ladder on the 0.70.2 tag.
- **Gate impact.** Restores an entry (tightens). Needs a GPU host, so S1 only.

## 3. Ratchet

This follows #4647 (operator ruling L31): a measured baseline, regression-only head-vs-base in the same run with the
same scanner, no stored limit, and a fleet-hours budget before any counter blocks.

| Counter | Baseline | Direction |
|---|---|---|
| Release requirements stated outside the one list | 151 | down only |
| Contradictions among them | 93 | down only |
| Known-failure tickets from 0.70.1 still open | 6 (#4661–#4666; the set is fixed at the tag) | down only |
| Implementations of "same code" | 2 | to 1, then never above 1 |
| Checks red on `main` for more than 24 h | not_measured | down only |

E1, E3 and E5 are per-release times. They are recorded as receipts, not counters.

## 4. Open questions

| Q | Question | Who decides | What it blocks |
|---|---|---|---|
| Q1 | K7's merge-queue skip: may the queue skip re-testing a tree it already tested green (3 of 10 sampled: #4613, #4632, #4647)? | The operator only: it changes a gate. The decision quorum voted 3/3 that it is not a row (D3) | Nothing in this plan |
| Q2 | E2 counts every check on `main`, scheduled ones included. Narrow it? | The operator only, if ever asked | Nothing: the literal reading applies |
| Q3 | Rows 3, 7 and 8 change what a release check accepts | Each PR: two reviewers, neither the author, plus operator sign-off (as C279 item 8) | The merge of those rows |
| Q4 | Row 7: option (a), (b) or (c)? | The decision quorum, at row start | Row 7 |
| Q5 | Row 9: fix the fallback, or refuse the file by name? | The decision quorum, at row start | Row 9 |
| Q6 | Which runs does a bump commit still need before the tag? | Measured on the 0.70.1 bump; decided at row 3 start | Row 3 |

## 5. Out of scope

- K5's PR-size half: from 0.71 the operator's PR rule already caps a PR at k ≤ 4 rows.
- K7's merge-queue skip (§4 Q1).
- Items from the 2026-10-02 review that serve no exit condition: the fake `##[error]` lines (#4596, #4610); mutation
  testing as non-blocking on a dedicated host; paging the operator within 5 min on a P0; rulings kept as a versioned
  file; porting `scripts/ci/fat_driver.py` off Python; a pmat complexity-hook ticket.
- Lean T5 at its 24 GB memory cap (C279 item 11): a ticket, no action.
- The documentation-drift ticket (29 docs), filed after S1.

## 6. Commands

All read-only. `<sha>` = `316dee2cd4`. C1 is one unauthenticated crates.io read; C9 is the only GitHub API call and
waits for S1.

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
# C4  the receipts-commit rule (E3, P2)
git show <sha>:scripts/release/release_readiness.sh | sed -n '58,97p'
# C5  the ladder's same-code function (P2)
git show <sha>:scripts/lib/ladder_equiv.py | sed -n '79,107p'
# C6  the build script that reads contracts/*.yaml (P1)
git show <sha>:crates/aprender-contracts/build.rs | sed -n '93,120p'
# C7  the marks in the ladder (P1, P3)
git show <sha>:contracts/model-capability-ladder-v1.yaml \
  | grep -n -E '^\s*(hotfix_scope|known_red|red_model|red_unsupported):'
# C8  the known-failure tickets on the 0.70.1 tag (E6; once the tag exists)
git show v0.70.1:contracts/model-capability-ladder-v1.yaml \
  | sed -n '/^  known_red:/,/^  [a-z_]*:/p' | grep -E 'ticket:'
# C9  E2, S1 only: the conclusions of one scheduled workflow on main over one week
gh api 'repos/paiml/aprender/actions/workflows/toolchain-ceiling.yml/runs?branch=main&created=2026-09-26..2026-10-02&per_page=100' \
  --jq '[.workflow_runs[].conclusion] | group_by(.) | map({(.[0] // "none"): length}) | add'
```

E4's 151 and 93 come from the build-kaizen baseline of 2026-10-02/03, an inventory kept outside this repo. Row 4's
first PR commits its in-repo part, so the count can be re-run from here.

## Quorum record: decision quorum, 2026-10-03 (aprender-a7)

The operator's standing rule, verbatim (2026-10-03): "use quorum for decision, never block on me". The quorum may
never approve a waiver, a gate or threshold change, a ruleset, secret or credential change, a force-push, a tag or a
publish (C213). Those stay with the operator: §4 Q1 and Q3.

Lanes: gpt-oss-120b (medium) and claude-sonnet-5-5 ×2, in plan mode. Launched 10:09:26Z; all three exited 0 by
10:09:41Z. Each lane answered from the same brief of facts: the C277 and C279 text, the layout of the existing train
plans, and the K table.

| D | Question | Tally | Applied as | Dissent and risks named | Reversible by |
|---|---|---|---|---|---|
| D1 | Where does the plan live? | A 3/3 | A new file, `docs/specifications/BLD-001-release-factory.md`: the train-plan layout plus a live-state selector, scoped to 0.70.2 | None. Risk named: a later move could duplicate or diverge, so a pointer from APR-RELEASE-001 goes in the first PR after S1 | Moving the file and leaving a pointer |
| D2 | What are the rows? | B 2/3 | One row per C277 item 6 condition, plus the P0s of C279 items 5, 6 and 11. K items are cited as prior art; a K item that serves no row is out of scope | gpt-oss voted A (one row per K item). Risks named: B could drop a high-value K item, or savings the operator would have approved | Adding a row for a K item in a later PR |
| D3 | Is K7's merge-queue skip a row? | no 3/3 | Not a row: it changes a gate (C213). It is §4 Q1, for the operator | None | An operator ruling |

`quorum: rounds 1 | width 3 | verdicts D1 A 3/3, D2 B 2/3 (oss A), D3 no 3/3 | overridden no`
