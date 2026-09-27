# infra#1237: fw16 runner CI wait, BEFORE the wrapper (baseline)

Window `2026-09-20T11:12:16Z` → `2026-09-27T11:12:16Z`. fw16 took its first job on **2026-09-26**, so the data covers
about 2 days.

```bash
Q=scripts/release/queue_inputs.sh
bash $Q runner-wait-fetch <raw-dir> 7        # every workflow run in the window (12 h slices, 1000-row cap refused),
                                             # then /actions/runs/<id>/jobs?filter=all for each (3479 runs, 24986 jobs)
bash $Q runner-wait evidence/infra-1237/fw16-wait-2026-09-27/raw            # -> fw16-wait.json
bash $Q runner-wait evidence/infra-1237/fw16-wait-2026-09-27/raw | jq .all.p50_s   # the number: 35
```

The rule: take the jobs whose `runner_name` matches `^framework16` and compute wait = `started_at − created_at`
(queued → started). Jobs from every workflow and every attempt count. `raw/jobs.jsonl` keeps only the fw16 jobs,
and the full 7.5 MB job dump stays on lambda at `~/.local/state/fw16-wait/2026-09-27/`. Both inputs give the same result.

| runner | n | p50 | p90 | max |
|---|---|---|---|---|
| **all fw16** | 232 | **35 s** | 548 s | 2010 s |
| framework16 | 99 | 14 s | 296 s | |
| framework16-2 | 96 | 51 s | 619 s | |
| framework16-3 | 37 | 78 s | 495 s | |

10 jobs are excluded. They are re-run attempts whose `started_at` comes from the earlier attempt (started < created).

**The infra#1237 target** is AFTER within 10% of BEFORE, so AFTER p50 must be ≤ **38.5 s**. To measure AFTER, run the same
two commands on a window that starts when the wrapper switches on. Use a window of similar length: n = 232 over 2 days
makes the p50 noisy.
