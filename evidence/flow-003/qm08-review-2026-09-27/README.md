# FLOW-003 QM-08 residue: review cost and escape cost (#4519)

QM-01 (#4513, `89/4513-qm01-inputs` @ `8326f2e40e`) measured the queue inputs. It left the review
inputs of Prop 12 at `[A]` or `[U]`. This receipt measures what the window allows and says why the
rest stays unmeasured. The window is the same: 2026-09-20T07:09:01Z to 2026-09-27T07:09:01Z.

## Result

| Input | Value | n | Mark |
|---|---|---|---|
| Q_x (review cost, full class) | **46.2 min per merged PR** | 26 PRs, 71 timed rounds | [V] |
| rounds per PR | mean 3.69, median 3, max 9 | 26 | [V] |
| round wall-time | mean 12.5, median 10.2, p90 26.9 min | 71 | [V] |
| Q_d, Q_a | none | 0 | [U]. Both classes were empty in the window |
| D (escape cost) | 240 min | 0 | [A]. See below |
| κ_full, κ_cheap | none | 0 | [U] |

35 of the 61 merged PRs have no quorum comment. They merged without an AD-04 round in the PR thread.
Only 24 of the 26 quorum PRs ever posted an agreed round.

## D cannot be measured from revert timelines

The row asks for D "from revert timelines". This repo has none:

- 0 reverts on main since 2026-08-01.
- 1 regression-labelled issue (#2466), which names no culprit.
- 1 fix-forward escape: #4309 says "release was RED since #4308".

Fixes go forward and do not name a culprit, so there is no timeline to measure. To measure D, a fix
would need a `Fixes-escape: #N` trailer, or a revert would need to be the default response to a red
main. That is a process change for the FLOW-003 owner, not something this receipt can do.

## Prop 12

q_c\* = (Q_x − Q_c)/((κ_x − κ_c)·D). At the [A] inputs it is 0.25. With the measured Q_x and Q_c = 0
(an upper bound), it is at most **0.385**. κ and D are still assumed, so the verdict is
**NOT-DECIDED**. The measured review cost is 1.5× the assumed saving of 30 min, which makes a cheap
review class more attractive than §5.1 assumed.

## Reproduce

- `raw/rounds.sh` fetches the quorum comments on each merged PR into `raw/quorum-comments.jsonl`.
- `raw/units.sh` pairs each user-journal `Started … quorum-review.sh` line with its unit's `Consumed`
  or `Deactivated` line, and writes `raw/quorum-units.json`.

The round wall-time is how long a round is on the path. It is not the CPU time the reviewers spent.
