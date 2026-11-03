# Five whys: Release Rehearsal `pick` job red 4 of 4 (#6676)

Owner: aprender-w2. Measured 2026-11-03 from run logs and the models INDEX; every figure below names its source.
Falsifier of #6676: cause read within 24 h; green or waived within 72 h.

## Defect
`release-rehearsal-nightly.yml` job `pick` exited 2 (`not_measured`) on its scheduled runs of 10-30, 10-31 and 11-01.
The last red run read, from job log of run 41303875559 (11-01 12:20Z):

    not_measured: no commit in main's newest 200 first-parent commits was measured by both the models bundle (19 commits)
    and an infra clean-room run (3 commits)

A `not_measured` night resets the green-night count, so the 0.72.3 promote could not start.

## Whys
1. **Why did `pick` exit 2?** `pick_c` (scripts/release/rehearse.sh:576) needs one sha that is in the models bundle
   (`nightly-evidence:models-crux/INDEX`, 24 lines) AND measured by an infra clean-room run. The intersection was empty.
2. **Why was it empty?** The two lanes each measure "main's head at the time they start", on their own clocks.
   Tested shas of the infra Clean-Room runs (job step "Assert the commit under test", read 2026-11-03):

   | run | UTC | aprender sha | in models INDEX | position on main |
   |---|---|---|---|---|
   | 41158120704 | 10-31 02:02 | c2c2f5051 | no | off main |
   | 41172679453 | 10-31 05:27 | e575d31b2 | no | main~18 at 11-03 |
   | 41179372434 | 10-31 07:07 | 3f61ce0c8 | no | main~21 |
   | 41183652197 | 10-31 08:12 | 2f1951837 | no | main~17 |
   | 41262858664 | 11-01 01:52 | 613eead8e | no | off main |
   | 41376044974 | 11-02 02:07 | 9c34b9a91 | **yes** | main~0 |
   | 41400416163 | 11-02 06:31 | 9c34b9a91 | **yes** | main~0 |

3. **Why did their commits differ?** Main moved between the two reads. Main's head changed many times a day until
   11-01 12:15Z; the clean-room schedule (02:00Z) and the models bundle (its own timer) never sampled the same head.
4. **Why does the pick depend on that coincidence?** The design is two independent lanes that each pick "newest
   main", and `pick_c` only takes their intersection. Nothing makes one lane measure the other's sha: the clean-room
   workflow takes no sha input from the models INDEX.
5. **Why was it not caught?** The 168-mutant table proves `pick_c` refuses wrongly-measured commits; it has no case
   for "both sides measured, never the same sha", and no KPI counts `pick` not_measured nights by cause.

## Why it is green now
Main stopped moving at 9c34b9a91 (11-01 12:15Z). The 11-02 02:07Z clean-room run and the models bundle then measured
the same head. Run 41454580958 (11-02 13:37Z): `pick` success, `models` success, `lanes / pick C` success. The red
moved downstream: `publish-dryrun` ("A worktree of C", exit 128) and `rehearse` stage `bump`
("cargo set-version failed (is cargo-edit installed?)"). Those are separate causes, not this ticket's.

## Root cause and fix
Root cause: the pick intersects two moving-head samplings, so it is green only while main is idle. Merging faster (the
operator's 5 PRs/h target) makes this red more often, not less.
Fix (one ticket, not in this PR): make the clean-room dispatch take the models INDEX's newest green sha as input
(infra `clean-room.yml` `workflow_dispatch` input `sha`), run by the rehearsal workflow before `pick`. That is a change
to infra and to a release gate, so it needs a planted quorum and a non-author sign-off.

## Not measured
- Why the 10-30 runs failed: log not read here.
- Infra clean-room runs older than 10-31: not read.
