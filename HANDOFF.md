# RQ-2 — release-day ladders move to the nightly (design, look-ahead)

Status: DESIGN ONLY, branch `build-kaizen/rq2-ladders-night`, READY-NO-PR (freeze). Agent: aprender-w4578.
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
