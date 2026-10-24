# EMF G3: receipts keyed by patch-id (PMAT-6134)

Source: EMF-001 §4 G3 / mechanism M2 / falsifier EMF-3. Operator, 2026-10-23 10:28Z (verbatim, relayed by
aprender-b9): "EMF G3 (no push voids receipts for unchanged patch-ids) moves up: restart storms are the demand."
Class: **gate content** (seats + quorum + arbiter sign). Door: two-way.

## Problem (measured surfaces)

A seat or quorum receipt is a GitHub review whose `commit_id` is the head sha it judged. Three readers bind to that sha:

| Reader | Binding today | Consequence of a push |
| --- | --- | --- |
| `scripts/mq_feeder.sh` `needs_seats` | `rv` = reviews with `commit_id == HEAD`; seats since PMAT-6124 also match `patch-id <whole-PR id>` | A no-op rebase or a merge of main voids the quorum receipt; the seat match is by the id of the WHOLE PR diff, so any changed commit voids every seat |
| `pmat-merge` / quorum artifact | `head` / `parent` / `diff_sha256` (content hash of the whole diff) | Same: one changed commit lapses the verdict for all |
| `scripts/arm_pr_automerge.sh` | `guard-tree` check-run with `head_sha == HEAD` | Correct to stay sha-bound: CI must pass on the exact tree that merges. **Out of scope** (result reuse is G9 / FLOW-010) |

The seat text itself is also unreliable: seats write `patch-ids: a b`, the feeder matches only `patch-id <id>` (PMAT-6124 seat-2 finding).

## Minimal change

1. **One key.** `scripts/pr_patch_ids.sh BASE HEAD` prints `git patch-id --stable` (first 12 hex) of every non-merge commit in
   `BASE..HEAD`, oldest first. It is the only producer; the feeder, the quorum artifact and any lint call it. A merge commit
   contributes no id, so merging main changes no id. An unreadable range, or an empty set, is a refusal (fail closed).
2. **One receipt line.** A seat or quorum review carries `patch-ids: <id> <id> ...` (the ids it judged). The `@<sha>` form stays accepted.
3. **Coverage, not equality.** Let `P` = the ids of HEAD. An id is *covered* by a seat when that seat's `patch-ids:` line names it
   (on any commit's review). Gate arming needs: every id in `P` covered by at least 2 distinct seats, and a `quorum 3/3 PASS`
   receipt covering every id in `P`, with no request-changes / FAIL review naming any id in `P`.
4. **A push re-reviews only what changed.** A no-op rebase or merge of main leaves `P` unchanged, so it voids 0 receipts. An
   amended commit replaces one id; only the ids new to `P` lack coverage, and a re-seat names just those (it may restate the rest).
   A negative review binds to the ids it names and stops blocking once none of them is in `P`.
5. **The sha rule is never loosened.** Every path that arms today on head/parent/diff_sha256 still does; patch-id coverage is an
   additional way to be current. `arm_pr_automerge.sh` keeps its `head_sha` rule unchanged.
6. **Duplicates.** Two commits with one id count once; the duplicate is reported (it signals a squashed-in repeat).

## Falsifiers (case table, in `mq_feeder.sh --self-test`; each has a planted mutant)

| # | Plant | Expect |
| --- | --- | --- |
| F1 | no-op rebase: same ids, new sha, receipt + 2 seats on the old sha | armable (0 receipts voided) |
| F2 | merge of main on top: same ids | armable |
| F3 | one commit amended: its id changes, seats name the old set | skip; names the uncovered id |
| F4 | re-seat naming only the new id | armable |
| F5 | seat covering an id not in `P` | contributes 0 |
| F6 | `patch-ids:` plural line and the singular `patch-id <id>` | both parse |
| F7 | request-changes naming an id still in `P` / one no longer in `P` | blocks / ignored |
| F8 | helper cannot read the range, or `P` is empty | skip (fail closed) |
| F9 | same seat number twice | counts once |

## PR plan (one ticket, one PR, EMF-001: a fix is a new commit on the same PR under a child ticket)

* **PR-A, PMAT-6134 (this ticket):** `scripts/pr_patch_ids.sh`, the `mq_feeder.sh` `needs_seats` coverage rule replacing the
  whole-PR-id match (builds on #6112 / PMAT-6124), the case table above. Gate content:
  3/3 quorum + 2 seats + arbiter sign.
* **PR-B, child ticket:** the quorum artifact gains `patch_ids[]`; `pmat-merge` accepts when `patch_ids ⊇ P` as one more rule
  beside head/parent/diff_sha256. The skill's source repo is confirmed before this PR is opened.
* **Not in scope:** CI check-run reuse (G9), the arbiter-sign format, the deployed `batcher-local-gate.sh`.

Exit (EMF-3): a no-op rebase voids 0 seats, and a changed commit voids only its own, measured on a planted PR.
