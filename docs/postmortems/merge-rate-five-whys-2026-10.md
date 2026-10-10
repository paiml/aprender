# Post-mortem — the `main` merge rate, 2026-09-26 to 2026-10-10

Refs #5057. Every number here is measured. The query and the time are given for each, and inferences are marked as such.

## What was asked

The question was: what caused the build-system slowdown that we fixed? Merges to `main` ran 1–2 a day in early October.
On 10-09 and 10-10 they ran 13–18 a day.

## What was measured (gh and `git log --first-parent origin/main`, 2026-10-10 12:27–12:30Z)

| period | PRs opened/day | merged/day | open PRs at start | p50 open→armed | p50 armed→merged |
|---|---|---|---|---|---|
| 09-26 .. 10-04 19:04Z | 10.2 | 2.6 (first-parent 1.25 from 09-30) | 11 | 2.3h | 1.65h |
| .. 10-06 17:26Z | 24.3 | 18.1 | 34 | 0.8h | 2.7h |
| .. 10-08 00:33Z | 7.7 | 6.9 (first-parent 8.5) | 19 | 6.2h | 1.2h |
| .. 10-10 12:00Z | 18.2 | 15.7 (first-parent 15.4) | 10 | 0.4h | 5.3h |

The open→armed and armed→merged medians come from a sample of about 8 PRs per period.

- **Supply.** PRs opened per day against `main` were 1–3 on 09-30..10-03, 34 on 10-04, 21–23 on 10-05/06, 8 on 10-07, then 14–17 from 10-08.
- **CI time.** The median `ci.yml` pull_request run was 59.8 min before #4912 and 55.8 min after it.
- **#4753** (merge_group reuses the PR head's x86-main result, merged 10-04 19:32Z). The merge-group runs sampled after it still ran x86-main, so the reuse did not show in that sample.
- **#4912** (mutants and the provable ladder off the PR path, 10-08 00:32Z). The jobs it removed were already skipped, or finished in under a minute, in the sampled PR runs.

## Five whys

1. **Why did merges/day go about 1 → 18 → 7 → 15?** Merges tracked supply, the PRs opened per day. Neither open→armed nor armed→merged moved in step with the rate. Armed→merged was longest, 5.3h, when the rate was highest. That is the opposite of a CI or queue bottleneck.
2. **Why did supply move?** It stepped at release holds and staffing changes:
   - a release-prep hold, 09-30..10-03;
   - a fleet-wide start, 10-04;
   - a hold that parked PRs and limited pushes, 10-06 10:45Z;
   - a freeze of `main` until the release tag, 10-07;
   - the hold lifted, 10-07 ~17:40Z.
3. **Why did a hold cut supply, not only merges?** The holds applied to every worker, not only to release-path files. *(Inferred from the hold text and the session log, not measured per PR.)*
4. **Why every worker?** The release was cut by freezing `main`, so a release stopped all work on `main`. *(Inferred.)*
5. **Root cause.** The release process stopped work on `main`; this follows from 1–4. The build system, meaning CI time and the merge queue, was not what limited the rate in this window.

## What the first analysis got wrong

The first answer named #4912, then #4753, as the cause. It rested on before/after merge-rate windows only, and it claimed that PRs spent "hours in CI". Measuring the stages refuted both. A rate change that lines up with a commit is an anecdote until the stage that changed has been measured.

## Countermeasure and guard

| row | countermeasure | guard (red when) |
|---|---|---|
| 1 | Never freeze `main` for a release: cut a release branch, and keep `main` merging. *(Proposed; needs a ruling.)* | `main` has 0 merges for 12h while any PR is armed |
| 2 | Keep #4912's state: heavy checks stay off the PR and merge-queue path. | any `mutants*` or `*ladder*` job appears in a `merge_group` run |
| 3 | Watch the rate, not anecdotes. | 3-day merges/day < 8 while armed PRs exist |
| 4 | Decompose before attributing. A throughput claim names its stage: supply, decision wait, or queue+CI. | — (review rule) |
