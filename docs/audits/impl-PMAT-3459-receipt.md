---
status: complete
ticket: PMAT-3459
github_issue: 3459
part: 2 (the must-carry label universe); part 1 (the in-repo gated tag path) landed in 6a657fc61 / PR #3617
kind: code
model: claude-opus-5-5 (author)
---
# implementation receipt: PMAT-3459 part 2 (#3459)

## Scope: where part 2 comes from

The roadmap title names only part 1. The issue #3459 carries part 2 in its own body, under "Tag-path defect 3
(operator ruling 2026-09-20 ~14:00Z) — `check_milestone_cut.sh` counts PRs":

> **The rule it must implement instead: count ISSUES carrying the must-carry label. Never PRs.**

The same section measures why: S3 asserted "one open PR -> RED", so every open PR on the milestone blocked the tag
(0.69.0: `open_issues=9` from the milestone API, 3 from `gh issue list`; the six PRs made up the difference).

dd's MUST-row sweep (#4159, 2026-09-24T07:11Z comment on #3459) records part 1 as DONE and part 2 as "OPEN.
Claimed by aprender-f5".

The cop (aprender-cf) ruled on 2026-09-24, and the ruling bounds this diff. This is the cop's ruling, not an
operator quotation:
- (a) Create the `must-carry` label, described as "Blocks the release cut of its milestone (check_milestone_cut.sh)".
- (b) The narrowing is approved on one condition: nothing is silently left behind. At the cut, every open item
  that is not must-carry is MOVED, with a one-line comment. It goes to the next release if that release's epic
  lists it, otherwise to `backlog`.
- A milestone that is tagged while items are still open is RED. Add a self-test row: an unlabelled open issue left
  in a tagged milestone gives RED.
- The move lives in the autopilot, before the tag. The gate only verifies.

So the carry script and the autopilot's move-before-tag step are the approval condition for narrowing the
blocker set. They are not extra scope. Without the move, `--must-carry` alone would let unlabelled items silently
ride a tagged milestone.

## What the diff does

| file | change |
|---|---|
| `scripts/check_milestone_cut.sh` | Adds a `--must-carry` mode. It is RED only on an OPEN ISSUE labelled `must-carry`, never on a PR. Every other open item is printed `TO CARRY`, and the JSON gains `mode` and `to_carry`. The strict default is unchanged. New self-test rows S24–S29; S27 is the cop's row (an unlabelled open issue under strict gives RED). |
| `scripts/release/carry_milestone_items.sh` | New. For each open non-must-carry item, it moves the item to the next open semver milestone when that milestone's "EPIC: release train <next>" lists it, otherwise to `backlog`, each move with a `slipped_from:` comment. It refuses (rc 1, zero writes) while any must-carry issue is open. rc 2 means "cannot act": a failed read, no next milestone, or a partial write. |
| `scripts/release/autopilot.sh` `cut_tag()` | Runs the must-carry verdict, then the carry step, then the EXISTING strict call (the line and its case block are byte-identical), then `git tag`. The strict gate still sees every open item, so anything the carry step missed turns the tag RED. |
| `scripts/check_tag_step_gated.sh` | Asserts that order. Must-carry rc 1 or 2 must neither carry nor tag, and carry rc 2 must not tag. New mutants: M4 (must-carry verdict discarded) and M5 (carry call deleted). It also runs the carry script's own self-test. |

The strict gate is not weakened. S3 still asserts "one open PR -> RED" under strict, and strict remains the last
thing before `git tag`. S24 inverts S3 only in `--must-carry` mode, which is exactly the rule the issue asks for.

## Measured (HEAD of this branch)

```
bash scripts/check_milestone_cut.sh --self-test          -> 31/31 ok; a mutant ignoring the label is killed by S26
bash scripts/release/carry_milestone_items.sh --self-test -> SELF-TEST PASSED (7 rows, stub gh)
bash scripts/check_tag_step_gated.sh --self-test          -> SELF-TEST PASSED (M1..M5 RED, real subject GREEN)
bash scripts/check_bashrs_gate.sh                         -> PASS, 0 SEC/DET/IDEM errors
check_shell_lint_ratchet / no_pipe_into_grep_q / no_hand_rolled_parsers / guards_are_wired -> PASS
gh api repos/paiml/aprender/labels/must-carry             -> exists, color B60205
```

Not done here, per the cop's ruling: applying `must-carry` to the 0.70 scope. dd does that at the 0.70 scope-cut GO.
