# Cop rulings register (#4436)

Operator, verbatim (2026-09-25 ~17:05 Madrid): "Make today's rulings permanent — each gets a ticket,
and "done" = merged + enforced".

Step 1 of `SKILL.md` reads this file. A ruling is in force for a fresh session only when its row
is `[M]`: merged, with a merge sha and a merged path that enforces it. `[U]` means not merged, so
not in force, however the ruling was worded in chat. The cop flips a row to `[M]` only with the
merge sha in hand, and the path column names merged files, never chat or `cop-state.md`.

| state | ruling (date, Madrid) | ticket | PR | merge sha | merged path |
|---|---|---|---|---|---|
| [U] | rc.N = tag on a queue-green main sha within 5 min; receipts on the tag gate promotion; promotion failure → rc.N+1; CI = 5 fat jobs, Σ executed unchanged (2026-09-25 16:58) | #4434 | — | — | `docs/specifications/APR-RELEASE-001-train-and-build-kaizen.md` §3 rule 17 |
| [U] | this register: step 1 reads it; only `[M]` rows are in force (2026-09-25 ~17:05) | #4436 | — | — | `.claude/skills/cop/SKILL.md` §1, `.claude/skills/cop/RULINGS.md` |
