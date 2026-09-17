# PMAT-3445 plan — the milestone being cut is read at the cut, not only at T-5

Ticket: #3445 (P1, milestone 0.69.0). Design discussion #3418, plan and evidence #3427.
Status: PLAN, grilled by a width-3 quorum before code lands.

## 1. Mechanism, measured

| Fact | Command | Value |
|---|---|---|
| 0.68.0's milestone at the tag | `gh api repos/paiml/aprender/milestones/5` (issue #3445 body) | open 2 / closed 93 at 06:35Z; #3091 reopened 05:10Z |
| What the 2 open items are today | `gh api 'repos/paiml/aprender/issues?milestone=5&state=open'` | #3091 (issue, P1) and #3450 (PR, `release: 0.68.1`) — **the milestones API `open_issues` counts PRs** |
| Where the tag is cut | `/mnt/nvme-raid0/agent-wt/rel-068-1-autopilot/autopilot.sh`, `if run_step tag` block (untracked, one copy per train) | `git tag -a` → `git push origin <tag>` → `gh release create`; **no milestone read** |
| The only milestone read in the train | same file, `close` step | `open_issues == 0` or refuse to close the milestone — **after** tag, assets and publish |
| T-5 | `scripts/check_reconcile.sh` | R1–R4; no milestone predicate |
| Live now | `ps` | the 0.68.1 autopilot (pid 3021358) waits on #3450 against milestone 5, which holds #3091 open |
| Next cut | `docs/specifications/06x-release-schedule.md` §3 | 0.69.0 leaves 2026-09-17T18:00Z; milestone 6 holds 126 open (123 issues + 3 PRs) |

Why the defect exists: the reconcile reads the milestone at one instant; an item reopened after that instant is invisible, and nothing re-reads at the cut.

## 2. Design

### G1 — `scripts/check_milestone_cut.sh`

```
check_milestone_cut.sh <milestone-title> [--repo O/R] [--json OUT]
check_milestone_cut.sh --self-test          # also the bare run (guard_tree.sh runs guards bare)
```

- **Universe:** every open item in the milestone, issues AND pull requests —
  `gh api --paginate 'repos/O/R/issues?milestone=<number>&state=open&per_page=100'`. PRs count because
  the milestone's own `open_issues` counts them, and that is the number the ticket quotes.
- **Title → number** via `milestones?state=all`. Zero or two matches → exit 2.
- **Count cross-check:** `len(universe) != milestone.open_issues` → exit 2. A paginated read that dropped
  rows must not be able to pass.
- **Vacuity:** `open_issues + closed_issues == 0` → exit 2. An empty milestone means the wrong one was named.
- **Carry record** (the ticket's "explicit `slipped_from:` decision"): a comment on the item with a line
  `slipped_from: <M>` — anchored at line start, the milestone title matched whole (so `0.69.00` and
  `0.69.0-rc` are not `0.69.0`), optional reason after whitespace. The term is the one
  `06x-release-schedule.md` §4 step 1 and D-5 already use.
- **A reopen voids an earlier carry:** the comment must be newer than the item's latest `reopened`
  event (`issues/<n>/events`). This is the #3091 shape exactly.
- **Verdict:** exit 0 when every open item is carried (vacuously when none are open); exit 1 naming each
  uncarried item `#N <issue|pr> [labels] title`; exit 2 on any environment or consistency failure
  (gh missing/unauthenticated, unresolvable milestone, count mismatch, vacuity). Never a silent pass.
- **Receipt** (`--json`): `{milestone, number, open, closed, carried:[{number, comment_url, created_at}],
  uncarried:[{number, kind, title, labels, reopened_at}], verdict}`.
- **Structure:** the predicate reads FILES (`milestone.json`, `items.json`, `comments/<n>.json`,
  `events/<n>.json`); the live path fetches into a temp dir and calls the same function. The self-test
  builds fixtures and never touches the network (the `check_reconcile.sh` pattern).

Self-test case table:

| Case | Input | Exit |
|---|---|---|
| C1 | open 0, closed 5 | 0 |
| C2 | one open issue, no comments | 1, names it |
| C3 | one open PR, no comments | 1, names it as `pr` |
| C4 | open issue, comment `slipped_from: M` | 0 |
| C5 | comment `slipped_from: <other milestone>` | 1 |
| C6 | carry comment OLDER than the latest reopen (#3091) | 1 |
| C7 | carry comment NEWER than the latest reopen | 0 |
| C8 | `slipped_from: M` mid-line in prose | 1 |
| C9 | `slipped_from: M0` / `slipped_from: M-rc` (prefix trap) | 1 |
| C10 | open 0, closed 0 | 2 |
| C11 | milestone says open 3, list holds 2 | 2 |
| C12 | two open, one carried, one not | 1, names only the uncarried |

Mutation proof, recorded in the commit body: neuter the predicate to admit everything → the self-test
goes RED (C2, C3, C5, C6, C8, C9, C12); neuter the reopen rule → C6 goes RED.

### G2 — the decision surface: `scripts/release_tag.sh`

The tag is decided in an untracked per-train autopilot copy. A precondition written only into the spec
depends on the next copy remembering it — the defect class this ticket is. So the tag step itself
becomes a versioned script the autopilot calls:

```
release_tag.sh --version X.Y.Z --commit <sha> --milestone <title> --notes <file> [--repo O/R] [--dry-run]
release_tag.sh --self-test
```

Order, each step fail-closed: `check_milestone_cut.sh <milestone>` (exit ≠ 0 → `STOP` naming the items,
no tag) → the tag does not already exist locally or on the remote → `git tag -a` → `git push` →
`gh release create --verify-tag`. `--dry-run` runs the gate and prints the three write commands without
running them. Its self-test stubs `gh`/`git` on PATH and asserts that a RED gate means zero write calls.

### G3 — wiring and contract

- `docs/specifications/APR-RELEASE-001-train-and-build-kaizen.md` §4: T-0 row reads the milestone
  (`check_milestone_cut.sh`) at the freeze; T-3 Promote runs it again immediately before the tag,
  through `release_tag.sh`. A RED read is SKIPPED, like any red step.
- `docs/specifications/06x-release-schedule.md` §4 step 1 (read at freeze), step 4 (tag through
  `release_tag.sh`), §7 ships-table row.
- `contracts/release-schedule-06x-v1.yaml`: equation `milestone_settled_at_cut`, falsification tests
  running both self-tests.
- `scripts/guard_tree.sh` picks both scripts up bare (both are cargo-free and self-test when bare).
- The running 0.68.1 autopilot is another session's file and is running. It is not edited here. Its
  owner gets the one-line replacement for its tag block. This is a named gap in the receipt, not a claim
  that it is wired.

## 3. First-green proof before it may block (live, recorded in the receipt)

| Run | Expected |
|---|---|
| `check_milestone_cut.sh 0.67.0` (closed, 0 open / 50 closed) | 0 |
| `check_milestone_cut.sh 0.66.0` (0 open / 10 closed) | 0 |
| `check_milestone_cut.sh 0.68.0` | 1, names #3091 and #3450 |
| `release_tag.sh --dry-run --version 0.68.99 --milestone 0.68.0 …` (the rehearsal the ticket names) | non-zero, `STOP` names #3091, no tag on the remote |

## 4. Phases

| Phase | Scope | Acceptance |
|---|---|---|
| P1 gate | `scripts/check_milestone_cut.sh` | `bash scripts/check_milestone_cut.sh --self-test` = 0; the mutation turns it RED; `bashrs lint` clean; the three live rows of §3 |
| P2 decision surface | `scripts/release_tag.sh` | `bash scripts/release_tag.sh --self-test` = 0; the dry-run rehearsal row of §3 |
| P3 wiring | the two specs, the contract | `pv validate contracts/release-schedule-06x-v1.yaml`; `cargo test -p aprender-core --test readme_contract test_documented_paths_exist`; `bash scripts/guard_tree.sh --dry-run --no-cargo` lists both as `run:` |

## 5. Consequence that must be surfaced, not hidden

The 0.69.0 cut is due today at 18:00Z with 126 open items in milestone 6. When this gate is wired,
that cut stops until every item is closed or carried. That is the gate working. The remedy is the
existing freeze/spill (06x §4 step 1, APR-RELEASE-001 §6.3), not a waiver.
