# PR Batch Cleanup — aprender (PRB-001)

Oct 21, 2026 · @Noah Gift

## 1. Status and scope

The goal: cut aprender's open PRs from 45 to ≤10, with 0 older than 24 h, by 2026-10-22 07:40Z. A batch-sweep worker under aprender-cop does this on lambda.

- **Status:** draft for review; the §10 rulings are recorded as relayed (see §10).
- **Parents:** FLOW-008 §8.7 (batch merge), FLOW-009 (lambda batch flow), and the 2026-10-20 stop-the-line ruling, whose timebox ends 10-23 12:00Z.
- **In scope:** paiml/aprender open PRs only.
- **Out of scope:** other repos. They get the same pattern later, after the exit in §8 holds.
- **Snapshot:** taken 2026-10-21 07:45Z. 45 open PRs, 37 of them drafts, 29 older than 24 h.
- **Grounding:** the project guide was not available. The triage CSV of 2026-10-12 is empty. Live measurements come from `gh pr view --json files,isDraft,mergeable` \[A\].

Source-of-truth paths:

- `infra/docs/specifications/FLOW-008-fleet-release-cadence.md` §8.7
- `infra/docs/specifications/FLOW-009-lambda-batch-flow.md` \[U\] (not on infra main)
- aprender `.github/workflows/ci.yml` (`ci / gate`)
- `scripts/contracts_resolve.sh` (#5762) and `scripts/contracts_gate.sh regen`: regenerate `contracts.nt`, `shapes.ttl` and `witness/*.json` from `contracts/*.yaml` \[A\]
- `scripts/readme_sync.sh --write`: regenerates the README `CONTRACT_COUNT` blocks \[A\]
- `scripts/batch_fold.sh`: the existing batch tool, which needs the fix in §5 \[A\]
- `scripts/arm_pr_automerge.sh` \[A\]

Tool paths checked with `git show origin/main:<path>` on 2026-10-21: the five aprender scripts exist. `infra/docs/specifications/FLOW-009-lambda-batch-flow.md` does NOT exist on paiml/infra main (only `fleet/flow009/gh-wrapper.sh` and `test-gh-wrapper.sh`), so that path is \[U\] until the spec lands.

## 2. Problem and root cause

The PRs go stale because they share files, so they merge one at a time. Batching them as they stand would not help.

Five whys:

1. **Why are the fold PRs stale?** Each one waits for its own `ci / gate` slot.
2. **Why are the merges serial?** Each merge forces the rest to rebase.
3. **Why do the rebases conflict?** The PRs write the same files. Five files are each touched by 3 or more lane A PRs (the batch-able test PRs, §3) \[A\]:
   - `contracts/witness/811c7818…json`: 6 PRs
   - `contracts/contracts.nt`: 5 PRs
   - `README.md`: 4 PRs
   - `docs/findings/aprender-w5404.jsonl`: 3 PRs
   - `classify_trainer_tests.rs`: 3 PRs
4. **Why didn't batching already fix it?** #5572, #5558, #5723, #5724 and #5781 are already batches. They overlap each other and went stale too.
5. **Root mechanism:** generated artifacts travel on the PR path, and each has many writers. This is the same failure as `roadmap.yaml` in 2026-09-15, where one shared file forced every PR into a single line.

## 3. Lanes

Every open PR goes in exactly one lane. Only lane A is batched by the sweep.

| Lane | PRs | Action |
| --- | --- | --- |
| A: test-only folds | 5735, 5728, 5724, 5723, 5672, 5657, 5591, 5590, 5572, 5558, 5527, 5784, 5781, 5827 | De-duplicate first (§5 step 2). Then batch, at most 7 per batch (cap ≤7, ruling 09:17Z, relayed). Drafts are allowed. |
| B: CI and gate changes | 5835, 5826, 5815, 5812, 5804, 5802, 5801, 5744, 5793, 5760, 5754, 5798, 5758, 5757, 5756, 5738, 5824, 5533, 5514, 5440, 5561, 5560 | These are one-way changes, so they need quorum. Never batched (amendment 3): each lane B PR runs alone. Close duplicates. |
| C: release | 5816 (0.71.5) | Runs solo, first in the queue. |
| D: external contributors | 5710, 5235 (guyernest) | Never absorbed. Reviewed one at a time. |
| E: held | 5830, 5829, 5834 (LAB), 5711 (HELD), 5562 (draft) | Stay in draft until stop-the-line ends. |

Total: 14 + 22 + 1 + 2 + 5 = 44 lane entries. #5832 is the 45th PR; it goes in the enabler queue (§4).

Known overlaps in lane A \[A\]:

- \#5724 and #5558 both fold #5539 and #5540.
- \#5572 overlaps #5723, #5724 and #5728.
- \#5827 is split c of #5590.
- \#5781 overlaps #5724 and #5558 on `classify_trainer_tests.rs`.
- \#5657, #5591 and #5572 are already CONFLICTING with main.

## 4. Queue order

The merge queue runs one entry at a time, and each run takes 41 min. That gives about 1.46 runs per hour, so every run should land as many tickets as it can. A fixed order would waste runs; this rule replaces it.

1. **Slot rule:** among entries ready to enter the merge queue, the one with the most tickets goes first.
2. **Fixed deadlines:**
   - \#5832 (K1) lands by 2026-10-22 06:00Z, two hours ahead of kfocus K2.
   - \#5816 (0.71.5): no fixed "today" deadline here. It conflicts with stop-the-line; its date comes from the release train, not this spec.
3. **Already in flight:** #5824 and #5829 finish their runs. Next comes lane A batch 1 if it is ready; otherwise #5812.
4. **Set before lane A:** #5812 (andon cleared once 2 seats confirm no continue-on-error). Close #5744 as superseded by #5801. Run `gh pr ready` on #5758, then #5798.
5. **Enabler effects:** #5812 frees machine time but adds 0 min to and removes 0 min from the 41-min critical path, so it does not raise merges per hour. #5758 and #5798 shorten PR CI, not the merge queue.

Throughput = tickets per run × runs per hour. At 1.46 runs per hour, 20 merges an hour needs 14 tickets per run, or 7 per run at about 21 min per run, or more merge-queue machines (kfocus). Native batching by GitHub's group-size setting: tested, NO (relayed by aprender-57: 231 merge_group runs, at most 1 PR each). Local batching (§5) is the only batching; the sweep must carry the tickets itself.

## 5. Batch-sweep worker

The sweep is a worker under aprender-cop on lambda, not a new top-level agent; this follows the 2026-10-18 ruling. It squashes each absorbed PR into one commit, then rebase-merges the batch. Main gets one commit per ticket, so any single ticket can be reverted on its own.

1. **Ticket.** Open a pmat ticket before starting. Work in a fresh worktree off `origin/main`, never the primary checkout.
2. **De-duplicate.** Give each folded ticket one owner PR. Close the others as "superseded by #N". Rebase or close the PRs that are CONFLICTING.
3. **Select.** Take lane A only, oldest first, at most 7 per batch. Drafts held by stop-the-line are allowed. Skip any PR whose diff is already on main.
4. **Absorb.** Fix `scripts/batch_fold.sh` in place: it uses `git merge --no-ff` today, and rebase-merge cannot carry merge commits. Squash cherry-pick each PR into one linear commit. Keep the original author and the `Refs #ticket` line. Strip the generated paths listed in §6.
5. **Drop on failure.** If a PR conflicts or its tests fail locally, drop it and label it `batch-reject`. Continue with the rest.
6. **Regenerate.** Run `scripts/contracts_resolve.sh` and `scripts/readme_sync.sh --write` once, as a single trailing commit titled `batch: regenerate`. Never amend this into a ticket commit, or reverting that ticket would also revert the regeneration.
7. **Verify.** Run affected-set nextest on lambda (#5802 once landed, otherwise the full workspace) plus coverage. Produce a signed receipt.
8. **Open the batch PR.** List each absorbed PR in the body as "Supersedes #N". Never use a closing keyword (close, fix, resolve) before #N.
9. **Merge.** `ci / gate` (clean-room) runs, then the merge queue rebase-merges the batch.
10. **Close out.** Once the merge sha exists on main, comment it on each absorbed PR, close the PR, and delete its branch.
11. **On a red batch.** Split it locally (7 into 4 and 3). Never resubmit the whole batch.

When searching for prior art, use `pmat query` (for example, "batch receipt" or "steward enqueue"), not grep.

## 6. Generated artifacts

Regenerating the shared files once per batch is only a stopgap. The real fix is to take generated files off the PR path entirely.

Rules for each file until then:

| Path | Rule |
| --- | --- |
| `contracts/contracts.nt`, `contracts/shapes.ttl` | Strip from each absorbed PR. Regenerate once per batch with `scripts/contracts_resolve.sh`. `ci / gate` re-derives them and fails on any mismatch. |
| `contracts/witness/*.json` | Same as above. A witness is gate evidence, and the producer is never the gate. |
| `docs/findings/*.jsonl` | Use `merge=union`. Never strip it: it is an append log, and stripping loses findings. |
| `README.md` `CONTRACT_COUNT` blocks | Strip, then regenerate with `scripts/readme_sync.sh --write`. |
| The rest of `README.md` | Hand-written. Resolve conflicts by hand; never strip. |

The structural fix is a separate pmat ticket: generated artifacts come off the PR path.

- PRs carry sources only.
- Main regenerates the artifacts after each merge, deterministically, with a single writer.
- The gate checks that regeneration reproduces the committed output.
- This is the same pattern as the `roadmap.yaml` fragments fix.

**Target:** 0 open PRs touch `contracts.nt` or `witness/*.json` after the ticket lands.

## 7. Contract: batch-sweep-v1

The contract is `contracts/batch-sweep-v1.yaml` (valid under `pv validate`). Every obligation has a test that can fail: `bash scripts/check_batch_sweep.sh --self-test` runs each on a fixture repo, RED then GREEN (15 rows, LAB, wired into no workflow).

| Id | Obligation | How it is falsified |
| --- | --- | --- |
| BS-1 | Every absorbed PR's tree diff is on main after the merge | Diff the PR's merge-base..head against main; any missing hunk fails |
| BS-2 | Main gets N ticket commits plus 1 trailing `batch: regenerate` commit; no merge commits | Commit count in the batch range must equal N + 1; any commit with 2 parents fails |
| BS-3 | No PR is closed before its merge sha exists | A PR closed while its sha is not an ancestor of main fails |
| BS-4 | No external-author PR or gate-file PR is ever absorbed | Check each absorbed PR's author and paths against a deny-list |
| BS-5 | No closing keyword appears before #N in a batch PR body | A regex guard on the body |
| BS-6 | Generated artifacts equal their regenerated output | `ci / gate` re-derives them and diffs |

## 8. KPIs and exit

The spec is done when ≤10 PRs are open (drafts included) and none is older than 24 h, by 2026-10-22 07:40Z.

| KPI | Now | Target | Basis |
| --- | --- | --- | --- |
| Open PRs, drafts included | 45 | ≤10 | Ruling 2026-10-12 |
| Open PRs older than 24 h | 29 | 0 | Ruling 2026-10-12 |
| Lane A PRs landed | 0 of 14 | all, in ≤3 gate runs, within 4 h of start | Finding-to-live ≤4 h, ruling 2026-10-18 |
| Merges per hour (tickets landed) | not measured | ≥10/h held for 4 h, then 20/h | Rulings 2026-10-19 and 2026-10-20 |
| Little's Law check (WIP = throughput × lead time) | FAIL (err 54%) | err ≤10% | Printed as FAIL, never "not measured" |
| Lane A reject rate | not measured | ≤30% | This spec's falsifier (§9) |
| Batches green on first gate run | not measured | Measure 5 batches, then set a floor | \[U\] No invented thresholds |
| `ci / gate` wall clock, p50 / p95 | 31 / 62 min (measured 2026-09-17, now stale) | Re-measure today; may only shrink after that | Shrink-only ratchet |

## 9. Falsifiers and andon

Each falsifier, if it trips, means part of this spec is wrong.

| If this is observed | Then this is wrong | Act |
| --- | --- | --- |
| The lane A reject rate stays above 30% after the §6 ticket lands | The root cause in §2: the hotspot is somewhere else | Rerun the file intersection; find the new hotspot |
| A batch's gate wall clock is more than 2× a single PR's | The assumption that batching saves time | Fix PR time first: sharding, gx10, kfocus |
| More than 10 PRs are open again within 48 h of exit | The assumption that a sweep can keep up | Cap intake: lambda opens no single-ticket PRs (FLOW-009) |
| An absorbed ticket is missing on main | BS-1, the absorption step | Stop and run five-whys |

Andon: stop the line on 2 consecutive red batches, or on any BS-1 or BS-3 violation.

## 10. Rulings

Relayed by aprender-57 on 2026-10-21 as the operator's answers; the verbatim text was not seen by the author of this amendment, so this section records the relay, not a quote. The author of the draft recommended yes to both.

- [x] **Do drafts count toward the ≤10 open-PR limit?** YES (relayed).
- [x] **May the sweep absorb stop-the-line drafts that are test-only?** YES (relayed).
- Q3 and Q4: relayed as ruled; their text is not in the relay. Fill in from the cop's record.
- Batch cap: ≤7 tickets per batch (09:17Z ruling, relayed). Lane B is never batched (amendment 3, relayed).

## 11. Amendment log

- 2026-10-21: cap ≤7; lane B never batched; native batching test reads NO; `batch_fold.sh` fix and README `CONTRACT_COUNT`-only rule folded in; tool paths verified on origin/main; `batch-sweep-v1.yaml` and its self-test added.
