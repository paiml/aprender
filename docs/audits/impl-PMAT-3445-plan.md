# PMAT-3445 plan — the milestone being cut is read at the cut, not only at T-5

Ticket: #3445 (P1, milestone 0.69.0). Design discussion #3418, plan and evidence #3427.
Status: **v2**. v1 was grilled by a width-3 quorum and returned 3/3 `do-not-implement-as-written`
(§6). v2 applies every consensus edit.

## 1. Mechanism, measured

| Fact | Command | Value |
|---|---|---|
| 0.68.0's milestone at the tag | `gh api repos/paiml/aprender/milestones/5` (issue #3445 body) | open 2 / closed 93 at 06:35Z; #3091 reopened 05:10Z |
| What the 2 open items are today | `gh api 'repos/paiml/aprender/issues?milestone=5&state=open'` | #3091 (issue, P1) and #3450 (PR, `release: 0.68.1`). **The milestones API `open_issues` counts PRs** |
| Where the tag is cut | `/mnt/nvme-raid0/agent-wt/rel-068-1-autopilot/autopilot.sh:104` `if run_step tag` (untracked, one copy per train) | `git tag -a` → `git push origin <tag>` → `gh release create`; **no milestone read** |
| The only milestone read in the train | same file `:239`, the `close` step | `open_issues == 0` or refuse to close. This runs **after** tag, assets and publish |
| T-5 | `scripts/check_reconcile.sh` | R1–R4; no milestone predicate |
| Live | `ps` | the 0.68.1 autopilot (pid 3021358) waits on #3450 against milestone 5, which holds #3091. Warned on #3450 |
| Next cut | `docs/specifications/06x-release-schedule.md` §3 | 0.69.0 leaves 2026-09-17T18:00Z; milestone 6 holds 126 open (123 issues + 3 PRs) |

Why the defect exists: the reconcile reads the milestone at one instant. An item reopened after that
instant is invisible, and nothing re-reads it at the cut.

## 2. Design

### G1 — `scripts/check_milestone_cut.sh`

```
check_milestone_cut.sh <milestone-title> [--repo O/R] [--json OUT]
check_milestone_cut.sh --self-test          # also the bare run (guard_tree.sh runs guards bare)
```

**The rule: exit 0 only when the milestone has 0 open items.** Issues and pull requests both count,
because the close step (`autopilot.sh:239`) and 06x §4 step 8 close on `open_issues`, which counts
both. A gate that ignored PRs could pass while close still refused.

**Carry means re-milestoning** (grill consensus, 3/3). An item that does not ride this train is moved
to the next milestone (`gh issue edit N --milestone <next>` / `gh pr edit N --milestone <next>`), with
a `slipped_from: X.Y.0` comment as the record. That is the term 06x §4 step 1 and D-5 already use, and
it leaves the milestone at 0 open, which is what close needs. The gate reads no comments and no events.
A reopen before the cut shows up as an open item, which is the #3091 shape exactly.

**When it runs.** Twice, and never while the train's own bump PR is open:
1. At the freeze (06x §4 step 1, APR-RELEASE-001 T-0), before the bump PR is opened.
2. Immediately before `git tag` (06x §4 step 4, APR-RELEASE-001 T-3), after the bump PR has merged.
   This is the binding read.

**Live reads**, each paginated as JSON lines (`gh api --paginate --jq '.[]'`) so that more than 100
items cannot turn into concatenated arrays:
`repos/O/R/milestones?state=all&per_page=100` and `repos/O/R/issues?milestone=<n>&state=open&per_page=100`.

**Exit 2, never a pass**, when: gh or python3 is missing; gh is unauthenticated; a gh read fails; the
title matches 0 or 2+ milestones (compared as a literal string, so `0.69.0` never matches `0a69b0` and
`0.69` never matches `0.69.0`); `open_issues + closed_issues == 0` (vacuous: the wrong milestone was
named); the item list length differs from `open_issues`; any listed item is not open or carries a
different milestone number.

**Exit 1** prints one line per open item, `#N <issue|pr> [labels] title`, then the remedy for each
item (close it, or run the exact `gh … edit N --milestone <next>` command), then a summary.
**Exit 0** prints `PASS milestone <M> (#n): 0 open, <c> closed`.

`--json OUT` writes `{milestone, number, open, closed, items:[{number, kind, title, labels, url}], verdict}`.

**Structure:** the judge reads two FILES (`milestones.jsonl`, `items.jsonl`). The live path fetches into
a temp dir and calls the same judge. The self-test builds fixtures and stubs `gh` on PATH; it never
touches the network. This is the `check_reconcile.sh` pattern.

Self-test case table:

| Case | Input | Exit |
|---|---|---|
| S1 | open 0, closed 5 | 0, prints PASS |
| S2 | one open issue | 1, names `#10 issue` |
| S3 | one open PR | 1, names `#11 pr`, remedy is `gh pr edit` |
| S4 | two open (issue + PR) with labels | 1, names both with their labels |
| S5 | open 0, closed 0 | 2 (vacuous) |
| S6 | `open_issues` 3, list holds 2 | 2 |
| S7 | `open_issues` 0, list holds 1 | 2 (a stale count cannot pass) |
| S8 | no milestone with the title | 2 |
| S9 | two milestones with the title | 2 |
| S10 | titles `0a69b0` (1 open) and `0.69.0` (0 open): judge `0.69.0` → 0; judge `0.69` → 2 | literal match |
| S11 | a listed item carries another milestone number, or is closed | 2 |
| S12 | 150 open items as JSON lines (the pagination shape) | 1, summary counts 150 |
| S13 | live path, `gh` absent from PATH | 2 |
| S14 | live path, stub `gh` that fails `auth status` | 2 |
| S15 | live path, stub `gh` serving the S2 fixture; the stub refuses a call without `--paginate` or a query without `state=open` | 1, names `#10` |
| S16 | S2 with `--json` | verdict `RED`, `items[0].number == 10` |

Mutation proof, recorded in the commit body: (a) the judge always passes → S2, S3, S4, S12, S15 go RED;
(b) delete the count cross-check → S6, S7 go RED; (c) drop `--paginate` from the live read → S15 goes RED.

### G2 — dropped

v1 proposed `scripts/release_tag.sh`. All three lanes rejected it: `guard_tree.sh` runs only
`scripts/check_*.sh` (`scripts/guard_tree.sh:18`), and `scripts/check_guards_are_wired.sh:122-126`
treats any `scripts/*.sh` with a `--self-test)` arm as a guard that must be wired. So it would be either
refused as unwired or silently dark. No CI home exists without a workflow edit.

The tag's decision surface therefore stays in the untracked per-train autopilot. That is a **named
gap**, not a claim of wiring. What this PR does about it:
- The spec text names the exact call and its position (§G3).
- The live 0.68.1 train is warned on its bump PR (#3450).
- A follow-up ticket versions the autopilot's tag step so the next per-train copy cannot omit the read.

### G3 — wiring and contract

- `docs/specifications/APR-RELEASE-001-train-and-build-kaizen.md` §4: the T-0 row reads the milestone
  at the cut; the T-3 row re-reads it immediately before the tag. A RED read makes the step RED
  (SKIPPED, like any red step). Carry means re-milestoning.
- `docs/specifications/06x-release-schedule.md` §4 step 1 (read at freeze; carry = re-milestone plus
  `slipped_from:`) and step 4 (re-read immediately before the tag), plus a §11 amendment row.
- `contracts/release-schedule-06x-v1.yaml`: equation `milestone_settled_at_cut`; falsification tests
  (a) the self-test and (b) both specs name the gate at the cut; `qa_gate.checks` entry.
- `scripts/guard_tree.sh` picks up `check_milestone_cut.sh` bare, which runs the self-test.

## 3. First-green proof before it may block (live, recorded in the receipt)

| Run | Expected |
|---|---|
| `check_milestone_cut.sh 0.67.0` (closed, 0 open / 50 closed) | 0 |
| `check_milestone_cut.sh 0.66.0` (0 open / 10 closed) | 0 |
| `check_milestone_cut.sh 0.68.0` | 1, names #3091 and #3450 |
| `check_milestone_cut.sh 0.69.0` | 1, 126 items (more than 100: the paginated read) |
| Rehearsal: a copy of the autopilot's `tag` block with the gate as its first line, `git`/`gh` write verbs stubbed to a call log, milestone 0.68.0 | `STOP` names #3091; the call log holds 0 write calls |

## 4. Phases

| Phase | Scope | Acceptance |
|---|---|---|
| P1 gate | `scripts/check_milestone_cut.sh` | `bash scripts/check_milestone_cut.sh --self-test` = 0; mutations (a)(b)(c) each turn it RED; `bashrs lint` has no errors; the four live rows of §3 |
| P2 wiring | the two specs, the contract | `pv validate contracts/release-schedule-06x-v1.yaml` = 0; `cargo test -p aprender-core --test readme_contract test_documented_paths_exist` = 0; `bash scripts/guard_tree.sh --dry-run --no-cargo` prints `run: scripts/check_milestone_cut.sh`; `bash scripts/check_guards_are_wired.sh` = 0 |
| P3 rehearsal + review | scratch only | the §3 rehearsal row; the width-3 quorum on the diff |

## 5. Consequence that must be surfaced, not hidden

The 0.69.0 cut is due today at 18:00Z with 126 open items in milestone 6. Where this gate is called,
that cut stops until every item is closed or carried. That is the gate working. The remedy is the
existing freeze/spill (06x §4 step 1, APR-RELEASE-001 §6.3), not a waiver.

## 6. Grill record (v1)

Width 3, `grillme`, review-only; lanes gemini-3.1-pro-high, gemini-3.8-flash-high,
gemini-3.7-flash-high; author family claude. Verdicts 3/3 `do-not-implement-as-written`. Consensus
edits, all applied above: carry = re-milestone and require open == 0; drop `release_tag.sh`;
paginate every read; add the metacharacter, zero-item and pagination cases; run the gate only when the
bump PR is not open. A width-1 goal lane had meanwhile implemented v1 (commit `d14914db8`, lane worktree
only). It was superseded, not merged: v1 semantics, and `gh api --paginate` without `--jq` writes
concatenated arrays that `json.load` rejects above 100 items. That would be an exit 2 on 0.69.0 itself.

## 7. Amendment v2.1 — pre-merge review

The review delegate found that each train's release epic (06x §5: label `epic`, milestone X.Y.0) is closed
at §4 step 8, after publish. #3078 was open in milestone 0.67.0 at the v0.67.0 tag. v2's rule (0 open items)
would therefore have read RED at every cut. v2.1 admits exactly one item: an issue labelled `epic` whose title
is literally `EPIC: release train <M>` followed by end or whitespace. It is printed as `ADMITTED`, and two
claimants exit 2. Cases S17–S23; mutations (d) admission removed → S17 S22 S23 RED, (e) admission widened to
any `epic` label → S18 S20 S21 RED.

