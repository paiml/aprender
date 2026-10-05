# Receipt: PMAT-4798 (aprender#4798) nightly judge: newest commit every verdict lane ran on

Change: `scripts/release/nightly_train.sh`, plus the matching amendment of `contracts/nightly-train-v1.yaml` (C becomes J), one `Makefile` comment, and the roadmap entry `docs/roadmaps/entries/PMAT-4798.yaml`. Not touched: `.github/workflows/nightly-train.yml`, whose step label still says "main's head" (cosmetic; a workflow edit needs its own gate-keeping proof).

## Ticket intent
The nightly judged C = main's head at the read and counted a lane only from a run on C, so any merge after the
producers fired voided the night. Now it judges C' = the newest commit, at or before the read, at most 6 commits behind
the head, on which EVERY verdict lane has a measured (green or red) run. No such commit = judged on the head, where a
lane without a run reads not_measured. There is no other fallback. The line prints
`(judged <sha10>, head <sha10>, lag k commits)` when C' is not the head. The release does not read C.

## What is measured
- Candidates are main's head (rank 0) and its history rollups from the SAME GraphQL query (`history(first: 7)`),
  newest first, written to `RAW/rank.tsv`. Lag is the rank of C' (exact; a commit with no runs still counts).
  The prototype ordered by first-run time and counted commits seen in runs, a lower bound that could exceed the cap
  unseen. This replaces it.
- One commit for all lanes, never a per-lane mix. A rank list whose rank 0 is not the head read is ignored.
- `--self-test`: 70/70 rows. `--mutants`: 53/53 killed (m43-m49 prototype, m50-m53 new: cap off by one,
  head of the rank list unchecked, oldest wins, history dropped from the rank list).
- Merge-queue history on main is not strictly first-parent (a merge commit lists the merged branch's commits too);
  those carry no main runs, only shorten the lookback. Conservative.

## Not claimed
- No replay of 2026-10-05: the saved raw predates the history read (no `rank.tsv`, no history rollups), so a replay
  reads only the head, which is the old behaviour. The first measurement is the next live 04:45Z read.
- If the verdict is ever wired to the release, require tree(C') == tree(MC). Not done here.
