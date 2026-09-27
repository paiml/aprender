# FLOW-003 QM-01 / QM-08 receipt — 7-day window ending 2026-09-27T18:03Z (paiml/aprender)

Every number below is re-derivable from the raw files in this directory. `compute`, `dora` and `prop12`
read nothing else. Script: `scripts/release/queue_inputs.sh` at the commit that adds this file.

## Commands (run from the repo root; R = this directory)

```bash
S=scripts/release/queue_inputs.sh
bash $S dora-fetch $R/dora 7                 # read-only gh API: merged/open PRs, queue-add events, ci.yml runs
bash $S dora $R/dora > $R/dora.json          # rc 1 = some target MISSED (expected here)
bash $S fetch $R/qm01 7                      # merge_group runs, jobs, nextest logs (~45 min)
bash $S ident $R/qm01
bash $S readset <main worktree @ readset_main_sha.txt> $R/qm01   # QM-08 class d read set
bash $S compute $R/qm01 > $R/queue-inputs.json
bash $S prop12 $R/queue-inputs.json > $R/prop12.txt
```

## Headline numbers (the ones asked for)

| Metric | Value | n | Target | Definition |
|---|---|---|---|---|
| Merge-queue wait p50 | 60.8 min | 35 | report | first AddedToMergeQueue -> mergedAt, merged PRs into main |
| Merge-queue wait p90 | **332.9 min** | 35 | <= 120 (operator stop rule) | same, p90 — **RED** |
| Merge-queue entry build p50 | 43.1 min | 43 | report | successful merge_group ci.yml runs, created -> updated (= QM-01 T) |
| Merge-queue entry build p90 | 122.3 min | 43 | report | same, p90 |
| PR age p90, merged | 13.95 h | 57 | < 24 | createdAt -> mergedAt |
| PR age p90, open | **51.7 h** | 20 | < 24 | open non-draft PRs, createdAt -> window end — **RED** |
| Open PRs | **21** (1 draft) | — | <= 10 (operator cap) | at fetch time — **RED** |

Percentiles are lower-rank: element floor((n-1)·p) of the sorted list.

## QM-01 queue-model inputs (queue-inputs.json)

| Input | Value | n |
|---|---|---|
| T | 43.1333 | 43 |
| C | 44.0583 | 96 |
| q | 0.1395 | 49 |
| phi | 0.0197 | 152 |
| q_eff | 0.1224 | 49 |
| lambda_eff_per_h | 0.3214 | 54 |
| lambda_per_h | 1.25 | 210 |
| f | 1 | 1 |
| F | 4.0167 | 1 |
| rho_mq | 0.0014 | 1 |
| rho_rel | 0.1259 | 83 |
| R | 410.1667 | 2 |
| ident_rate | 0.2841 | 88 |
| window_days | 7 | 1 |
| n_runs | 53 | 53 |

Derived: rho_HOL (Prop 11, lower bound) = 0.0283 (operator stop rule 0.6). Verdict: GREEN.

## QM-08 service classes (Prop 12)

```
PROP12 class=d n=0 q=null q_upper95=null q*=0.25 NOT-DECIDED
PROP12 class=a n=0 q=null q_upper95=null q*=0.25 NOT-DECIDED
PROP12 class=x n=49 q=0.1395 q_upper95=0.32 q*=0.25 NOT-DECIDED
```

No PR in the window qualified as class d (docs-only outside the read set), and none as class a, because the attestation label does not exist.
So pi_d = pi_a = 0. The cheap-lane decision stays NOT-DECIDED: an unmeasured input never decides (S-2).
Class x q = 0.14, 95% upper bound 0.32 against q* = 0.25.

## Full DORA table (dora.json)

| Metric | Value | n | Target | ok |
|---|---|---|---|---|
| lead_time_p50_min | 254.97 | 35 | < 60 | false |
| lead_time_p90_min | 705.48 | 35 | report | true |
| ci_p50_code_min | 62.75 | 74 | <= 10 | false |
| ci_p50_docs_min | 143.02 | 1 | <= 2 | false |
| pr_age_p90_h_merged | 13.95 | 57 | < 24 | true |
| pr_age_p90_h_open | 51.66 | 20 | < 24 | false |
| mq_wait_p50_min | 60.83 | 35 | report | true |
| mq_wait_p90_min | 332.92 | 35 | <= 120 | false |
| mq_entry_p50_min | 43.13 | 43 | report | true |
| mq_entry_p90_min | 122.32 | 43 | report | true |
| open_prs | 21 | 21 | <= 10 | false |
| conflicted_over_4h | 5 | 21 | 0 | false |
| change_fail_rate | 0 | 33 | < 0.05 | true |
| release_cycle_p50_min | 43.57 | 96 | <= 30 | false |
| merge_commit_resolutions | 108 | 78 | <= 2 (-> 0) | false |
| fold_merges | 139 | 78 | report | true |

Verdict: MISSED lead_time_p50_min,ci_p50_code_min,ci_p50_docs_min,pr_age_p90_h_open,mq_wait_p90_min,open_prs,conflicted_over_4h,release_cycle_p50_min,merge_commit_resolutions

Thin inputs, not to be over-read: f and F have n = 1 (one ejection -> push -> re-entry in the window), rho_mq n = 1.
