# Five whys: milestone 0.72.0 CLOSED 193/0 but never published (aprender-w5404, 2026-10-27 12:15Z)

## Facts (each measured today, 12:06Z)
- Milestone #10 "0.72.0": state closed, open 0 / closed 193, closed_at 2026-10-27T09:45:06Z.
- Tag v0.72.0 exists at d162cc409e. GitHub Release "aprender 0.72.0" is a DRAFT (created 06:37Z). crates.io max version is 0.71.4.
- Milestone #29 "0.72.1": open, 4 open / 4 closed. Open: #6468 (changelog), #6438 (release PR), #6437 (release issue), #5326. Closed and merged: #6413, #6414, #6416, #6420.
- Inbox 09:47Z (aprender-wsweep, triage cycle 1): "0.72.0 closed (8 moved: #5326->0.72.1, 7->0.73.0)". The close came after open reached 0.
- 0.72.0 was SKIPPED by ruling (NOAH-0716Z, "0.72.1, not 0.72.0"); the clean-room run 40462182334 was cancelled and Binary Release run 40462178924 failed on `aprender-rollup --help` (#6439).

## Why 1: why is a milestone for an unpublished version closed?
A triage worker closed it at 09:45Z because GitHub showed open_issues = 0 after it moved the 8 leftovers away.

## Why 2: why did open = 0 count as "close it"?
Nothing states what closing a milestone means. The only signal the worker had was GitHub's own counter, and a counter says "no open issues", not "released".

## Why 3: why was there no second signal?
The release train DOES close milestones: `scripts/release/autopilot.sh` step 9 `close` (lines 840-849) lists the other open items, closes the epic, checks `open_issues = 0` and PATCHes the milestone closed. That step did not close 0.72.0: the train's log (`/mnt/nvme-raid0/tmp/rt-v0.72.0/autopilot.log`) has no `close`/`MILESTONE` line, and `STATUS` ends `2026-10-27T07:17:58Z STOP clean-room ... cancelled`, 2 h 27 min before the milestone closed. So the close was a hand/triage close (the 09:47Z inbox line from aprender-wsweep reports it). The one place that knows the publish state (the train) was bypassed, and the train's own close step checks only `open_issues = 0`, never that the version is published. No other script compares milestones to tags, Releases or crates.io.

## Why 4: why did 0.72.0 reach open = 0 while unpublished, and 0.72.1 hold the merged work?
0.72.0 was skipped by ruling. The merged work stayed in milestone 0.72.0 (193 closed issues, none shipped) while the version that will ship became 0.72.1. No rule moves or re-labels milestones when a version is skipped, and no rule says a skipped version's milestone stays open until its work ships under a successor.

## Why 5: why is there no such rule?
Milestones are treated as planning buckets. Release state lives in four other places (tag, Release, crates.io, train ledger), and no invariant ties a milestone's state to them. Nothing checks that tickets reflect releases: the E1 rehearsal checks the train, not the tickets.

## Root cause
No invariant and no checker tie milestone state to publish state. Counter-measure design (R-b is a delta inside the existing `close` step, before the PATCH at `autopilot.sh:849`): `docs/specifications/release-reconcile.md` (design PR to follow by 13:30Z):
(a) a milestone closes only if tag AND GitHub Release (not draft) AND crates.io version exist;
(b) in `autopilot.sh close`, before the milestone PATCH: require published(V) (rc 2 = do not close), then comment the release URL on every closed issue; the existing open = 0 check stays;
(c) a nightly LAB lane diffs milestones against tags/releases and opens one ticket per drift.

## Immediate repair (not done by me; needs the cop/operator because it edits live tickets)
Reopen milestone #10 "0.72.0" until a decision on where its 193 closed issues are accounted for (release 0.72.1 notes list them), or record the skip in the milestone description. Not changed by this session.
