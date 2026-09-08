# docs/roadmaps/roadmap.yaml

This file is the `pmat work` ticket database for aprender. It is edited by
`pmat work add` / `pmat work complete` and read by CI's `roadmap-valid` step
(`pmat work validate`).

## Insert in order — never append to the tail

New top-level (0-indent) `- id:` entries must be inserted in **sorted
position among their own id-prefix peers** (e.g. a new `PMAT-1069` goes next
to the other `PMAT-NNNN` entries, in ascending numeric order — not appended
after the last entry in the file). Blindly appending to the tail is how two
unrelated PRs both edit the same handful of lines and conflict on merge even
though their tickets have nothing to do with each other (measured, BSE-09b:
two PRs appending to the tail both edited the same lines; `.gitattributes
merge=union` does not help because GitHub's merge engine does not honour a
local-only `merge=union` driver). Inserting at your ticket's own sorted
position spreads concurrent edits across different lines instead of the same
one, which is what actually avoids the conflict.

`scripts/check_roadmap_sorted.sh` enforces this: for every id of the shape
`PREFIX-NUMBER` (letters/digits, hyphen, digits — e.g. `PMAT-9`, `GH-279`),
each prefix's own numeric sequence must be ascending in file order. It also
refuses a duplicate id and a tracked `docs/roadmaps/*.bak` or `*.lock` file
(`pmat work` writes both as scratch — never `git add` them; see
`.gitignore`). Ids that predate the `PREFIX-NUMBER` convention (roughly 100
legacy entries whose `id:` is a free-text title) are exempt from the sort
check — there is no numeric position for them to violate — but still count
toward the duplicate-id and tracked-file checks.

```bash
bash scripts/check_roadmap_sorted.sh              # check the real file
bash scripts/check_roadmap_sorted.sh --self-test  # fixture case table
```
