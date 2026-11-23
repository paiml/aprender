# Release train table, ruled 2026-11-13

Ruled by the operator on 2026-11-13 at 11:48Z: the train table stands. Release master: #5306. Facts below were read from the GitHub API at 12:50Z (UTC).

This replaces the morning meaning of 0.74.0 ("the plot release", due 19 Nov) written in `0.74.0-notes-draft.md` by #6997.

## Table

| Date | Version | Carries |
|------|---------|---------|
| 14 Nov | v0.74.0 | What is green on main: plot slice 1, the first #6909 rows |
| 16 Nov | v0.74.1 | present-lib P0s (#6728: #6890 #6891 #6892 #6893 #6895 #6896), plus #6880 |
| 18 Nov | v0.74.2 | Plot slice 2: the usable plot core, proven by the pv ontology (30 plot issues + PRs #6938 #6968 #6976 #6986 #6992 #6996) |
| 20 Nov | v0.75.0 | Ships with what is done |
| 22 Nov | v0.75.1 | |
| 24 Nov | v0.75.2 | |
| 26 Nov | v0.76.0 | |

## Rule

- A train leaves on its date with the last green nightly.
- Unmerged work rides the next train.
- A train whose nightly is not green on its date does not move the table: it leaves with the last green nightly, or it is skipped and the skip is said.
- Clean-room CI is green on the tagged commit before every publish (unchanged).

## Plot preview notice

The release notes of v0.74.0 and v0.74.1 say in their first lines: plot verbs are a preview, NOT pv-proven, not for course demos. Courses pin plots only from v0.74.2.

## Milestones on GitHub (12:50Z)

| Milestone | Due | Open |
|-----------|-----|------|
| 0.74.0 | 14 Nov | 1 (#6879) |
| 0.74.1 | 16 Nov | 8 |
| 0.74.2 | 18 Nov | 36 |
| 0.75.0 | 20 Nov | 31 |
| 0.75.1 | 22 Nov | not measured |
| 0.75.2 | 24 Nov | not measured |
| 0.76.0 | 26 Nov | not measured |
