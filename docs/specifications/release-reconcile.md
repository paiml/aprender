# Release reconcile: milestones must reflect releases

Status: design (aprender#6471). No code in this PR.

## Problem

Milestone 0.72.0 was closed (193 closed, 0 open) while 0.72.0 was never published: the tag
and a draft GitHub Release exist and crates.io tops out at 0.71.4. The version that will
ship is 0.72.1, whose milestone holds the merged advisory work beside open release items.
A reader of the tickets was told a release happened that did not.

Cause (five whys in `docs/specifications/release-reconcile-five-whys.md`): the milestone was
closed by hand at 09:45Z (triage, after moving 8 leftovers), not by the train: the train
stopped at clean-room on 07:17Z and never reached its `close` step. That step
(`scripts/release/autopilot.sh:840-849`) closes the milestone when `open_issues = 0` but never
asks whether the version is published, and a hand close has no check at all.

## Definitions

A version V is **published** when all three hold, each measured, never inferred:

1. tag `vV` exists on origin;
2. a GitHub Release for `vV` exists and is not a draft;
3. crates.io lists `aprender` version V.

A missing read of any of the three is NOT_MEASURED (rc 2), never published.

## Rules

- **R-a, close.** A milestone titled V may be closed only if V is published. A milestone for
  a skipped version stays open until its leftover work is moved to its successor, and the
  successor's release notes list that work.
- **R-b, a delta to the existing `close` step.** `autopilot.sh` step 9 already lists the other
  open items, closes the epic, checks `open_issues = 0` and PATCHes the milestone
  (`autopilot.sh:840-849`). Add before the PATCH at line 849: `published(V)` (rc 2 or
  false = die, milestone stays open), and a release-URL comment on every closed issue of V.
  The open = 0 check is unchanged.
- **R-c, nightly.** A LAB lane diffs milestones against tags, Releases and crates.io and
  files one ticket per drift, owned by the release-train owner, then stays silent.

## Drift classes (R-c)

| id | drift | measured by |
|----|-------|-------------|
| D-1 | closed milestone V, V not published | milestone closed, any of the three missing |
| D-2 | published V, milestone V open | all three exist, milestone open |
| D-3 | published V, milestone V has open issues | open_issues > 0 |
| D-4 | closed issue in a published milestone without a release-URL comment | R-b comment absent |
| D-5 | tag or Release for V with no milestone titled V | zero milestones |

## GATE / LAB (C324)

R-c is **LAB**: it runs at night on an idle machine, never in a PR, merge-queue or release
job, and a red opens or updates one ticket and stays silent. R-b is a delta to a step of the release
train (`autopilot.sh close`), which already sits on the release path; it adds no new
check on a PR. R-a is enforced by R-b for the train and, for hand closes (how 0.72.0 closed), found by D-1 the next night.
It is not a PR gate: no merge is blocked by it.

## Falsifiers

1. Plant a milestone closed with a draft Release: D-1 fires. Publish the Release without
   crates.io: D-1 still fires. All three present: no D-1.
2. Plant a read failure (rate limit, 404): result is NOT_MEASURED rc 2, not "published".
3. R-b with one open issue left: the step lists it, exits non-zero and leaves the
   milestone open. R-b with a draft Release and open = 0: the step refuses before the PATCH.
4. A skipped version (tag, no Release) with its leftovers moved: no D-1 while the
   milestone stays open.

## Out of scope

The fleet-wide lane for other repos belongs to the infra seat. Repairing today's drift (the
0.72.0 milestone) is a ticket edit, not part of this change.

## Rollout

1. This design, sign-off round by a non-author.
2. `scripts/release/reconcile.sh` (read-only, `--self-test` with the falsifiers above), no
   workflow change.
3. Wire R-b into `autopilot.sh close`; add the nightly lane (workflow change: its own 3/3).
