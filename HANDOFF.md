# RQ-2 — release-day ladders move to the nightly (design, look-ahead)

Status: ruling A BUILT (§8), quorum round 2 folded (§9); branch `build-kaizen/rq2-ladders-night`, READY-NO-PR (freeze), off by default. Agent: aprender-w4578.
Target: release day stops running the long ladders; it reads last night's result **for its own commit**.
Saving claimed by aprender-a7: −160 min per release (a7's measurement, not re-derived here).

## 1. What runs on release day now (origin/main 316dee2cd4)

The train is `scripts/release/autopilot.sh`. The cut `MC` is the bump PR's merge commit (autopilot.sh:75).

| Step | Where | Runtime (as stated, not measured here) | Moves? |
|---|---|---|---|
| `deep`: doctests, `--no-default-features`, examples, all-bins smoke | autopilot.sh:92-122, lambda | not stated | **YES** |
| `dogfood`: `dogfood.sh --phase pre-publish` FULL + R5 `--receipt-only` | autopilot.sh:133-142 | ~90 min (06x-release-schedule.md:71) | **YES** |
| `models`: `models_t1.sh` → `model_ladder.sh` on lambda + gx10 | autopilot.sh:149-155 | "Hours" of GPU per host (scripts/model_ladder.sh:24) | **YES** |
| readiness (`pv` SHACL), preflight R1-R8, fleet_cells_gate | autopilot.sh:162-172, :308; rc_cut.sh:212 | seconds; judges only | no: they re-judge the receipts |
| clean-room T-3, binary-release, rc fleet stage, host + post-publish dogfood | autopilot.sh:239-492 | need the tag / the asset | no: cannot run before a tag |

## 2. The design: cut AT the commit the nightly measured

The release does not *inherit* a receipt measured at another commit. It **chooses its cut to be the commit
the nightly measured**. Then every existing judge holds unchanged:

- R5: `commit == HEAD` (scripts/check_publish_preflight.sh:119-173).
- R7: `apr_sha == cut`, or tree-equal outside evidence/ (scripts/check_model_ladder.sh:90-106).
- #3708's "never inherited" rule (autopilot.sh:125-131).

So **no gate changes what it accepts**; only *when* the three steps run changes.

1. **Nightly** `scripts/release/release_ladders_nightly.sh` (new) runs on lambda, the host autopilot already runs on.
   - Trigger: a forjar-managed systemd timer at **02:00 UTC**, under `choom` and the existing `apr-gpu.lock`.
   - Why not a GitHub workflow: lambda is not a runner (APR-RELEASE-001 §3); a timer needs no workflow edit and 0 GitHub calls.
   - It runs autopilot's existing `deep`, `dogfood` and `models` steps, unchanged, with `MC=$(git rev-parse origin/main)`, in a fresh worktree at that sha.
   - It writes `/mnt/nvme-raid0/release-ladders/<sha>/{deep,dogfood,models}.{log,rc}` plus the receipts the steps already write, and appends `<sha>\t<version>\t<deep rc>\t<dogfood rc>\t<models rc>\t<utc>` to `index.tsv`.
   - It runs every night, release pending or not: this is D-1 of 06x-release-schedule.md (FULL tier nightly on main), now feeding the release.
2. **Train order.**
   - Evening before release day: the bump PR merges.
   - Night: the nightly measures a main HEAD that carries version V.
   - Release day morning: `autopilot.sh --cut-from-nightly` picks the cut.
3. **The release reads** the newest index row that meets all three conditions:
   - version == V;
   - the sha descends from the bump merge;
   - deep, dogfood and models are all rc 0.

   That sha becomes `MC`. Then:
   - autopilot copies its receipts into the worktree and runs the **same** judges (R5 `--receipt-only`, `check_model_ladder.sh`, readiness);
   - it skips the three steps;
   - it tags `MC`.

   Commits that landed after `MC` ride the next train (main never freezes).
4. **No usable row means the old path, not a pass.** autopilot runs the three steps on release day exactly as today. The fallback costs time, never a gate.

## 3. Case table — what `--cut-from-nightly` decides

Planned as `scripts/release/check_nightly_cut.sh`, bash, with a planted-row table like
`scripts/ci/ladder_pr_skips_leanchecker.sh`. Not built in this design PR.

| # | Situation | Decision | Why |
|---|---|---|---|
| 1 | row at version V, descends from the bump, deep/dogfood/models rc 0, receipts present | cut = that sha; skip the 3 steps; judges re-run | the steps measured this exact commit |
| 2 | as 1, but a later main sha also has a green row | cut = the **newest** green row | most code in the train |
| 3 | newest row at V has models rc 1 (a red cell) | **STOP**, no tag; no fallback to an older green row unless it also meets 1 and the red one is triaged | a red ladder at V is a finding, not noise (FLAKE-0) |
| 4 | row has dogfood rc 2 (decline / host defect) | not usable; fallback = run on release day | decline ≠ pass (L25) |
| 5 | row's receipts missing or unreadable, index says rc 0 | not usable; fallback | the index is a pointer, the receipt is the evidence |
| 6 | receipt `commit`/`apr_sha` ≠ the row's sha | **STOP** (forged or corrupt) | binding broken (#3957 F2) |
| 7 | newest row is pre-bump (version ≠ V) | not usable; fallback (or wait one night) | receipts are version-keyed (`evidence/dogfood/models/<V>/`) |
| 8 | row sha does not descend from the bump merge | not usable | it is not this release's code |
| 9 | row older than 36 h | not usable; fallback | a stale host/toolchain state (yokoten L27) |
| 10 | only one host's models receipt present | run that host's ladder only on release day, judge both | partial reuse, never a partial pass |
| 11 | `/mnt/nvme-raid0/release-ladders` absent (host rebuilt) | fallback, and a NIGHTLY-RED line | not_measured (L23) |
| 12 | the nightly did not run (timer dead) | fallback + NIGHTLY-RED (APR-RELEASE-001 §7) | the saving is lost, the gate is not |

## 4. Open questions, to quorum (not to the operator, per C282)

- Q1 **gx10 contention at 02:00.** silicon-nightly 22:30, qwen-story 23:17, nightly-bench 00:30 and beat-speed 00:45 all touch gx10. `apr-gpu.lock` serialises them, but a late start pushes the row past the morning cut. Measure one week of finish times before fixing the hour.
- Q2 **Bump timing.** The design needs the bump PR merged the evening before. Today it merges on release day itself. This is a train-order change (APR-RELEASE-001 §4 T-0), not a gate change.
- Q3 **Index durability.** It is a local dir on lambda. Committing receipts happens later, in the bump/ledger PR, as today (autopilot.sh:547). A lost dir only means fallback (row 11).

## 5. Not in scope

- Moving clean-room, binary-release or host checks: they need the tag.
- Changing any judge.
- Changing the provable-ladder: since #4710 / C312 it runs in full on every push to main.

## 6. Quorum round 1 (2 Sonnet 5.5 lanes, author is Opus 5.5): both APPROVE-WITH-FIXES

Folded in here. These amendments override §2 and §3 where they conflict.

### F1. "Unchanged steps" is false as written

autopilot.sh:40-79 needs a PR, a milestone, an epic and `gh` before any step runs.

- **Fix:** factor deep/dogfood/models into `scripts/release/lib_ladders.sh <sha> <V>`, sourced and option-neutral.
- autopilot then calls the lib. This is a refactor PR with a planted-row diff of the worktree state, and no behaviour change.

### F2. Where the receipts live

R5 picks the newest `.dogfood/receipt-*.json`. R7/R8 read the receipts committed under `evidence/dogfood/models/<V>`. The dirty-tree stop is at autopilot.sh:138.

- **Invariant:** release day opens a worktree at S (the measured sha) and leaves it **byte-identical** to what today's steps leave at MC. It then follows today's path unchanged: the ledger/receipt commit and the judges.
- The dogfood receipt is selected by `commit == S`, not by mtime. A stale older receipt is never copied; the copy replaces `.dogfood/` whole.
- The planted test for this is a diff of the worktree after the "real steps" against the worktree after the "nightly copy".

### F3. MC is no longer the bump merge

autopilot.sh:75-88 must re-run at the new MC: version == V and `bump-version.sh --check`. `check_milestone_cut.sh` re-reads at T-3 against MC.

- S must descend from the bump merge.

### F4. Most nights measure the wrong version

Receipts are keyed by version. **Fix:** the nightly runs the ladders only when main's version is greater than the latest tag, i.e. a bump has landed and is not yet tagged. Otherwise it exits "no pending release" and writes no row. This also saves GPU on the other ~6 nights.

- Row 9 is now: the row must come from a night **after** the bump merge, which bounds it to ≤ 30 h, not 36 h.

### F5. Case table changes

- Drop row 10 (partial host reuse). models_t1.sh judges the two hosts as one batch (:167-179), and mixed provenance is not admitted. Any missing host means a full fallback.
- Row 3 loses "unless triaged". A red row at V is STOP, with no human-waiver path.
- New row 13: the receipt's `version` ≠ V, or its `phase` ≠ pre-publish → STOP.
- New row 14: OPEN obligations outside the pre-publish phase → STOP (check_publish_preflight.sh:153-165).
- New row 15: lock-busy model_ladder exit 75 (apr-gpu.lock) → decline → fallback, never a pass.
- New row 16: release day rebuilds `apr` at S. A nightly target dir is reused only when its build sha == S (the readiness re-grep at autopilot.sh:168-169).

### F6. Cite fixes

- R5 is check_publish_preflight.sh:119-179.
- R7 begins at check_model_ladder.sh:87.
- dogfood is autopilot.sh:125-143; models is :146-156.

### F7. The saving is fragile, and there is a simpler option for a7 to rule on

Today's release-day order is deep → dogfood → models, run serially after the bump merges.

**Option B:** launch the three at the bump merge, concurrently. dogfood and deep run on lambda; the model ladder runs on lambda and gx10 in parallel, as a sibling of `wait`.

- B needs no timer, no index, no evening bump, and no fallback table.
- It saves the serial sum minus the longest leg on the critical path.
- The nightly (option A) saves the whole duration, but only when the bump lands the evening before and the post-bump night is green.

**Recommendation:** ship B first, because it is small and has no train-order change. Then add A's nightly as a second step only if B's measured critical path still misses the target.

**Ruling asked of aprender-a7:** A, B, or B then A.

## 7. RULING (aprender-a7, 2026-10-04): A, with B as the fallback for a late bump

a7's words, verbatim: "RQ-2 F7 ruling: A (nightly 02:00Z; the release cuts at the measured sha), with B only as the fallback for a late version bump. Decided because A is the "release = a commit already green last night" rule; dissent B is faster for late bumps; reversible by re-ruling. Log it in your HANDOFF and build A."

- **When B applies:** there is no usable post-bump nightly row, i.e. rows 4, 5, 7-9, 11, 12 or 15 hold. B then runs the three steps concurrently at release time, in place of today's serial run.
- **Build order:**
  1. `lib_ladders.sh` (F1).
  2. `check_nightly_cut.sh` plus its planted rows (§3 + F5).
  3. `release_ladders_nightly.sh`, gated on the version (F4).
  4. The `autopilot --cut-from-nightly` wiring.

## 8. BUILD STATUS (aprender-w4578, 2026-10-04): A built, branch only

Supersedes §7's build order where they differ.

- **F1 is done by `autopilot.sh --ladders <V> <sha> deep|dogfood|models`, not `lib_ladders.sh`.** Moving the step bodies into a lib broke three guards that read autopilot's lines verbatim (check_release_models_t1, check_release_autopilot_dogfood_close, check_release_host_receipts). `--ladders` keeps one copy of each step in autopilot: it skips the milestone, epic and PR lookups, sets `MC=<sha>`, and runs only the named step. A malformed call exits 2 before any state dir, `gh` or `git` (5 rows).
- **`scripts/release/check_nightly_cut.sh`** picks the cut (§3 + F5). Table `check_nightly_cut_table.sh`: 20 rows, 6 mutants caught.
- **`scripts/release/release_ladders_nightly.sh`** is the nightly (F4 version gate, one row per sha, `flock`). Table `release_ladders_nightly_table.sh`: 12 rows + 5 `--ladders` rows, 4 mutants caught.
  - A step's index rc: exit 0 → 0; autopilot exit 2 → 2; last STOP line `NO-GO rc=2` or `rc=75` → 2; anything else → 1 (fail closed toward STOP).
  - The dogfood receipt is copied by `commit == sha`, never by mtime (F2).
- **autopilot wiring** is opt-in: `RELEASE_CUT_FROM_NIGHTLY=1` (not a `--cut-from-nightly` flag).
  - check_nightly_cut rc 0 → `MC` = the measured sha (must still be an ancestor of origin/main), then F3's version + `bump-version.sh --check` run at that MC as today.
  - rc 2 → FALLBACK: today's path (ruling B). Any other rc → STOP.
  - On a cut, deep/dogfood/models are skipped; at readiness the receipts are copied in (`.dogfood/` and `models-t1/` replaced whole) and the SAME judges run: R5 `--receipt-only` (row 14: OPEN obligations), `check_model_ladder.sh`, then `apr` is rebuilt at MC so readiness re-proves `(${MC:0:9}` (row 16).
- **Guards:** all 7 autopilot guards + check_sourced_libs_option_neutral green after the edits.
- **Not built here (follow-ups):**
  1. The forjar systemd timer, 02:00Z on lambda, from a dedicated checkout (paiml/infra; the timer is the only trigger, 0 GitHub calls).
  2. B's concurrent launch for a late bump: the fallback today is still the serial run.
  3. Turning `RELEASE_CUT_FROM_NIGHTLY=1` on by default, after one release cuts from a nightly row.

## 9. Quorum round 2 (2 Sonnet 5.5 lanes on e52ba8632f): both APPROVE-WITH-FIXES

Neither lane found a path where `RELEASE_CUT_FROM_NIGHTLY=1` tags a sha that was not measured GO, or where the unset path changes. Fixed in the next commit:

- **r1 F1. Deep was trusted from the index alone.** check_nightly_cut now also requires `DEEP GO at <S>` in `<root>/<S>/STATUS` (row 17; else FALLBACK). Mutant `no-deep-evidence` is caught.
- **r1 F3 = r2 F4. step_rc could read an earlier step's STOP line**, so a killed step became 2. It now reads only the STATUS lines written during its own step. Row + mutant `stale-stop-line`.
- **r1 F5. The stub's STOP text was not tied to autopilot.** Two rows grep autopilot for the exact `die "... NO-GO rc=$rc` lines step_rc parses.
- **r2 F2. Lock fd leaked to children; no timeout.** Steps now run `9>&-` under `timeout` (`RELEASE_LADDERS_STEP_TIMEOUT`, default 6 h). A timeout is 2, never 0. Rows + mutants `fd9-leak` and `timeout-is-red`.
- **r2 F3. A not-measured sha was never retried.** A sha is re-measured while its newest row holds a 2; a 0 or 1 row is final. Rows + mutant `no-retry-on-2`.
- **Found while folding these: the tables reused state across mutant runs.** `table` runs in a subshell, so the dir counter reset and mutant runs inherited earlier index rows, which gave spurious catches. Dirs now come from `mktemp`. Every mutant is still caught by its own target row.
- **r2 F7.** The status line above.

Open, not fixed here:

1. **r1 F2: the release-day block has no planted test.** It needs `gh` and judge stubs around a full autopilot run. That is the next item before `RELEASE_CUT_FROM_NIGHTLY` is used on a real release; until then it stays off.
2. **r1 F4 (to aprender-a7: changes what a gate reports).** models_t1.sh checks `env` before `nogo`. A RED cell plus an unreachable host exits 2 (not measured), not 1. The effect today is a fallback re-run, not a bad tag.
3. **r2 F1 + F5 (infra timer follow-up).**
   - The nightly must run from a dedicated checkout that the timer detaches to origin/main first, so the nightly runs the same autopilot release day will.
   - It must never share `target/` with a release train.
   - autopilot should take `<root>/.lock` while it runs the fallback ladders.
   - Add `git worktree prune` and retention for `<root>/<sha>/wt`.
