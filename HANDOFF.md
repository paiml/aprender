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
