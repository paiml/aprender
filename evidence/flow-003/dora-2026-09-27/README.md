# FLOW-003 QM-01: weekly DORA metrics, 7-day window (aprender#4513)

Window `2026-09-20T09:57:55Z` → `2026-09-27T09:57:55Z` (`raw/window.txt`).

```bash
scripts/release/queue_inputs.sh self-test                                  # planted week + empty week (NO-DATA, never MET)
scripts/release/queue_inputs.sh dora evidence/flow-003/dora-2026-09-27/raw  # re-derives dora.json; rc 0 only when MET
```

`raw/` is the receipt, and `dora` reads only those files. `dora-fetch` produced them read-only: ci.yml
runs, merged and open PRs, each PR's arm/queue events, files and branch commits, and main's commits.

| metric | value | n | target | met |
|---|---|---|---|---|
| lead time p50 (arm → merged, into main) | **223.8 min** | 41 | < 60 | no |
| lead time p90 | 705.5 min | 41 | report | |
| CI p50, code PRs | **61.5 min** | 88 runs | ≤ 10 | no |
| CI p50, docs PRs (every file `.md`) | 15.3 min | **1** run | ≤ 2 | no (low n) |
| PR age p90, merged | 13.1 h | 63 | < 24 h | yes |
| PR age p90, open non-draft at window end | **43.6 h** | 15 | < 24 h | no |
| conflicted > 4 h (open) | **5** (#4463 #4459 #4457 #4431 #4414) | 16 open | 0 | no |
| change-fail (reverts / main commits) | 0 | 41 | < 5% | yes |
| release cycle p50 (release-branch CI) | **43.6 min** | 95 | ≤ 30 | no |
| merge-commit resolutions (main pulled into a PR branch) | **104** | 79 PRs | ≤ 2 → 0 | no |
| fold merges (PR branch into a batch) | 135 | | report | |

`verdict: MISSED`. Seven targets are missed, and two are met.

## Reading it

- **Merge-commit resolutions are the biggest outlier: 104 against a target of ≤ 2.** Two PRs account for 41
  of them: #3689 (21) and `release/0.69-batch` #3669 (20). Squash-merge hides them from main, so only the
  branch commits show them.
- **CI p50 for code is 61.5 min.** It is ci.yml created → updated, which includes runner queue wait. It is
  also higher than C = 43.6 min (the release-PR cycle), because PR runs share runners with the batches.
- **Lead time is 3.7× the target.** Arming comes long before merging, and most of the wait sits in the
  batch/fold path rather than the queue. The QM-01 table has median mq_wait of 54 min.
- **Change-fail is 0 by subject.** No `Revert` commit landed on main this week. A fix-forward counts as
  nothing here, so treat this number as a lower bound.
- Docs CI has n = 1. §5.1 routes almost every `.md` PR to class x (see qm01 README, QM-08).

The method for each metric is written in `dora.json` `.metrics.<key>.method`. A metric with n = 0 is
`ok: null` (NO-DATA), never a pass.

## Baseline: the exact command behind each number

This week is the **baseline**: 7 of 9 targets missed. The raw inputs were fetched once, read-only:

```bash
scripts/release/queue_inputs.sh dora-fetch evidence/flow-003/dora-2026-09-27/raw 7
#  ci_runs.jsonl      gh api repos/paiml/aprender/actions/workflows/ci.yml/runs?created=<12h slice>   (fetch_ci_runs)
#  merged.json        gh pr list --state merged --search "merged:<start>..<end>" --limit 1000 --json number,createdAt,mergedAt,headRefName,baseRefName,title
#  open.json          gh pr list --state open --limit 500 --json number,createdAt,headRefName,baseRefName,isDraft,mergeable
#  main_commits.jsonl gh api --paginate "repos/paiml/aprender/commits?sha=main&since=<start>&until=<end>"
#  pr_dora.jsonl      gh api graphql: per PR files, AUTO_MERGE_ENABLED / ADDED_TO_MERGE_QUEUE events, last 100 commits (parents)
```

Each number below comes from `Q=scripts/release/queue_inputs.sh; R=evidence/flow-003/dora-2026-09-27/raw`. It is re-derived
offline by `bash $Q dora $R | jq '<filter>'`, and the rule behind it is in `dora_compute`:

| number | filter | rule |
|---|---|---|
| 223.8 min | `.metrics.lead_time_p50_min` | p50 of mergedAt − (first AutoMergeEnabled, else first AddedToMergeQueue), base = main |
| 61.5 min | `.metrics.ci_p50_code_min` | p50 of updated_at − created_at, ci.yml `pull_request` runs, success/failure, non-release branch, PR not all-`.md` |
| 15.3 min | `.metrics.ci_p50_docs_min` | same, PR whose every file is `.md` |
| 13.1 h / 43.6 h | `.metrics.pr_age_p90_h_merged` / `_open` | p90 of mergedAt − createdAt; open non-draft: window end − createdAt |
| 5 | `.metrics.conflicted_over_4h`, `.detail.conflicted_prs` | open, `mergeable == CONFLICTING`, last branch commit > 4 h before window end |
| 0 | `.metrics.change_fail_rate`, `.detail.reverts` | main commits whose subject starts `Revert` / all main commits in the window |
| 43.6 min | `.metrics.release_cycle_p50_min` | p50 ci.yml `pull_request` run time on `RELEASE_BRANCH_RE` branches |
| 104 | `.metrics.merge_commit_resolutions`, `.detail.merge_resolutions_by_pr` | two-parent PR-branch commits in the window whose headline names `main` |

## Weekly re-measure and next checkpoint

The `untangle-weekly.timer` (Mon 07:05 Madrid) runs `dora-fetch` + `dora` on a fresh 7-day window. It posts one
`dora weekly` inbox line and keeps the raw receipt under `~/.local/state/dora/<date>/raw`.

**Checkpoint QM-09 (#4527):** code CI p50 should drop from 61.5 min to about 15 min. The weekly line after #4527 lands
measures it with `.metrics.ci_p50_code_min`, the same filter and the same rule as above.
