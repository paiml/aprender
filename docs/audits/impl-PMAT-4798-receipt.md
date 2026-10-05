# Receipt: PMAT-4798 (aprender#4798) nightly judge: newest commit every verdict lane ran on

Change: `scripts/release/nightly_train.sh`, the matching amendment of `contracts/nightly-train-v1.yaml` (C becomes J), one
`Makefile` comment, and the roadmap entry `docs/roadmaps/entries/PMAT-4798.yaml`. Authored fresh on the c310 pick-window
proof (#4749, `a7w/4749-c310` @ 067e2e0646). Not touched: `.github/workflows/nightly-train.yml` (its step label still
says "main's head"; a workflow edit needs its own gate-keeping proof).

## Ticket intent
The nightly judged C = main's head at the read and counted a lane only from a run on C, so any merge after the producers
fired voided the night. Now it judges J = the newest commit, at or before the read, at most 6 commits behind the head, on
which EVERY verdict lane has a measured (green or red) run. None such = judged on the head, where a lane without a run
reads not_measured. There is no other fallback. The line prints `(judged <sha10>, head <sha10>, lag k commits)` when J is
not the head.

## How c310's rule carries to J
- A plain (schedule, push) run stands for the commit it ran on: it counts for J when its head_sha is J.
- A chained (`workflow_run`) run's tree is the pick of its night, never its head_sha, so it stands for that pick only:
  it counts for J when `nightc(created_at)` == J, and for a run created 12:00Z-17:59Z also `prevc` == J. The J search
  applies the same rule when it asks whether a commit has a measured run on every lane.
- Candidates are main's head (rank 0) and its history from the SAME GraphQL query (`history(first: 7)`), written to
  `RAW/rank.tsv`; lag is the rank of J (exact). One J for all lanes, never a per-lane mix. A rank list whose rank 0 is
  not the head read is ignored.

## Two unlike failures (a7's condition)
- "No pick for that night": the nightly refs were read and that night has none. The chained lane is not_measured with the
  reason "measured the pick no pick for that night" (c310's behaviour); plain lanes are unaffected. Case row
  `a_chained_run_without_a_read_pick_is_not_measured`; read-level row `an_absent_pick_is_a_good_read`.
- "Pick read failed": the refs field is not an array (fetch/parse error). `picks_read` writes `pickfail`; `evaluate` turns
  the read into `failed: pick read: ...` and EVERY lane is not_measured, with no fallback. Rows
  `a_failed_pick_read_is_not_measured_never_a_fallback`, `a_pick_read_with_no_refs_field_failed`.
- Mutants that merge them: `m66_failed_pick_read_falls_back` (evaluate ignores `pickfail`) and
  `m67_failed_pick_read_is_an_absent_pick` (the array check is dropped, so a failed read looks like an absent pick).

## What is measured
- `--self-test`: 87/87 rows. `--mutants`: 67/67 killed (m43-m67; m55-m65 are the J search, m66 and m67 the pick split).
- Merge-queue history on main is not strictly first-parent; those commits carry no main runs and only shorten the
  lookback. Conservative.

## Not claimed
- No replay of 2026-10-05: the saved raw predates the history read, so a replay reads only the head. The first
  measurement is the next live read.
- If the verdict is ever wired to the release, require tree(J) == tree(MC). Not done here.
