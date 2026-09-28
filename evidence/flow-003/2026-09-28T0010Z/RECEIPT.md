# FLOW-003 QM-01 / QM-08 re-measure — 7-day window ending 2026-09-28T00:10Z (paiml/aprender)

This is an overnight refresh of `../2026-09-27T1802Z/RECEIPT.md` (89, @c99c6c3aa5). It uses the same script
at the same commit and the same commands, with only the window moved. Every number below is re-derivable
from the raw files in this directory.

## Commands (run from the repo root; R = this directory)

```bash
S=scripts/release/queue_inputs.sh            # unchanged from c99c6c3aa5
bash $S dora-fetch $R/dora 7                 # window 2026-09-21T00:10:42Z .. 2026-09-28T00:10:42Z
bash $S dora $R/dora > $R/dora.json          # rc 1 = some target MISSED (expected)
bash $S fetch $R/qm01 7                      # window 2026-09-21T00:11:39Z .. 2026-09-28T00:11:39Z (41 min)
bash $S ident $R/qm01
bash $S readset <detached origin/main worktree @ readset_main_sha.txt> $R/qm01
bash $S compute $R/qm01 > $R/queue-inputs.json
bash $S prop12 $R/queue-inputs.json > $R/prop12.txt
```

Every step returned rc 0 except `dora`, which returned rc 1 (targets MISSED, as in the baseline).
`readset_main_sha.txt` = aca6f2d7f6, the same main commit as the baseline.

## What changed since 18:02Z: only the window edge

**Zero PRs merged into main between the two receipts.** The last merge into main is 2026-09-27T10:38:14Z,
so main has had no merge for 13.5 h at window end. Every delta below comes from the window's start moving
forward from 09-20T18:03Z to 09-21T00:10Z. Those 6 h dropped 4 merges and 8 queue entries. Nothing was
added at the end, because nothing merged after 18:02Z. So this re-measure is a no-new-flow reading, not a trend.

## #4439 (asked for by d1)

#4439 is inside both windows, so it is counted in both receipts. It was added to the queue at
2026-09-25T23:59:25Z and merged at 2026-09-26T00:42:44Z, a merge-queue wait of **43.3 min** (below p50). It
appears in `qm01/mq_prs.txt`, and one merge-from-main resolution (09-25T19:54Z) is charged to it in
`dora.json`.

**#4439 is NOT the first merge-queue merge.** `AddedToMergeQueueEvent`-timed merges run through the whole
window, starting with #3491 at 2026-09-21T00:41:10Z. The baseline window has earlier ones, #3540 and #3581
at 09-20T18:36Z. If "first real" means something other than "first queued" (for example, the first merge
under the new queue-group-1 config), then 00:42Z has to be tied to a config change. This data cannot show that.

## Headline numbers: baseline vs now

| Metric | 09-27T18:02Z | 09-28T00:10Z | Target |
|---|---|---|---|
| Merge-queue wait p50 | 60.8 min (n35) | 54.3 min (n31) | report |
| Merge-queue wait p90 | 332.9 min (n35) | **373.7 min** (n31) | <= 120 — **RED** |
| MQ entry build p50 (= QM-01 T) | 43.1 min (n43) | 39.9 min (n35) | report |
| MQ entry build p90 | 122.3 min (n43) | 102.8 min (n35) | report |
| PR age p90, merged | 13.95 h (n57) | 13.95 h (n53) | < 24 |
| PR age p90, open | 51.7 h (n20) | **57.6 h** (n20) | < 24 — **RED** (the same open PRs, 6 h older) |
| Open PRs | 21 | **21** | <= 10 — **RED** |
| Conflicted > 4 h | 5 | 5 (#4512 #4459 #4457 #4428 #4414) | 0 |

## QM-01 queue-model inputs (queue-inputs.json)

| Input | 18:02Z | 00:10Z |
|---|---|---|
| T | 43.1333 (n43) | 39.9333 (n35) |
| C | 44.0583 (n96) | 46.3417 (n98) |
| q | 0.1395 (n49) | 0.1143 (n39) |
| phi | 0.0197 (n152) | 0.0203 (n148) |
| q_eff | 0.1224 (n49) | 0.1026 (n39) |
| lambda_eff_per_h | 0.3214 (n54) | 0.2738 (n46) |
| lambda_per_h | 1.25 (n210) | 1.2083 (n203) |
| f / F | 1 / 4.0167 (n1) | 1 / 4.0167 (n1) |
| rho_mq | 0.0014 (n1) | 0.0014 (n1) |
| rho_rel | 0.1259 (n83) | 0.2569 (n89) |
| R | 410.1667 (n2) | 779.8167 (n1) |
| ident_rate | 0.2841 (n88) | 0.2778 (n90) |
| n_runs | 53 | 42 |

Derived: rho_HOL (Prop 11, lower bound) = **0.0187** (was 0.0283), against the operator stop rule 0.6.
Verdict: **GREEN**. Low-n inputs, which should not be over-read: f, F, rho_mq, R (n <= 2). rho_rel doubled on
n = 89. R is now a single sample.

## QM-08 service classes (Prop 12)

```
PROP12 class=d n=0 q=null q_upper95=null q*=0.25 NOT-DECIDED
PROP12 class=a n=0 q=null q_upper95=null q*=0.25 NOT-DECIDED
PROP12 class=x n=39 q=0.1143 q_upper95=0.3085 q*=0.25 NOT-DECIDED
```

Unchanged in shape: no class-d or class-a PR in the window, so pi_d = pi_a = 0. The cheap-lane decision
stays NOT-DECIDED (S-2: an unmeasured input never decides). The class x upper bound fell from 0.32 to 0.31,
which is still above q* = 0.25.

## Full DORA table (dora.json)

| Metric | 18:02Z | 00:10Z | Target | ok |
|---|---|---|---|---|
| lead_time_p50_min | 254.97 (n35) | 278.45 (n31) | < 60 | false |
| lead_time_p90_min | 705.48 (n35) | 868.77 (n31) | report | true |
| ci_p50_code_min | 62.75 (n74) | 58.18 (n65) | <= 10 | false |
| ci_p50_docs_min | 143.02 (n1) | 143.02 (n1) | <= 2 | false |
| pr_age_p90_h_merged | 13.95 (n57) | 13.95 (n53) | < 24 | true |
| pr_age_p90_h_open | 51.66 (n20) | 57.56 (n20) | < 24 | false |
| mq_wait_p50_min | 60.83 (n35) | 54.25 (n31) | report | true |
| mq_wait_p90_min | 332.92 (n35) | 373.65 (n31) | <= 120 | false |
| mq_entry_p50_min | 43.13 (n43) | 39.93 (n35) | report | true |
| mq_entry_p90_min | 122.32 (n43) | 102.82 (n35) | report | true |
| open_prs | 21 (n21) | 21 (n21) | <= 10 | false |
| conflicted_over_4h | 5 (n21) | 5 (n21) | 0 | false |
| change_fail_rate | 0 (n33) | 0 (n30) | < 0.05 | true |
| release_cycle_p50_min | 43.57 (n96) | 44.55 (n98) | <= 30 | false |
| merge_commit_resolutions | 108 (n78) | 97 (n74) | <= 2 (-> 0) | false |
| fold_merges | 139 (n78) | 121 (n74) | report | true |

Verdict (unchanged set): MISSED lead_time_p50_min, ci_p50_code_min, ci_p50_docs_min, pr_age_p90_h_open,
mq_wait_p90_min, open_prs, conflicted_over_4h, release_cycle_p50_min, merge_commit_resolutions.
