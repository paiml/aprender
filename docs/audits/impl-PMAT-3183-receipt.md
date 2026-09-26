# impl receipt — PMAT-3183 (aprender#3183)

**Defect.** `scripts/dogfood.sh` row 3 was `grep -qF "$VERSION" CHANGELOG.md`: PASS on a
bare `## [x.y.z]` heading, or on any line mentioning the version string.

**Change.**
- `scripts/check_changelog_covers_merged.sh` — both-direction predicate over
  `git log --first-parent LAST_TAG..CUT` (MISSING / FALSE-ROW), `--notes`, `--self-test`
  (6-row case table, each RED row matched by its printed reason, not just exit code).
- `scripts/dogfood.sh` row 3 — calls the guard. FAIL in pre-/post-publish, WARN in `full`.
  Last tag: `git describe` (excluding `v$VERSION` and rc tags), else the highest non-rc tag
  below `$VERSION` by `sort -V` — release tags sit on release branches and are not ancestors
  of main (measured: `git describe HEAD` on main → "No tags can describe").
- `Makefile` tier3 — runs `--self-test`.

**Evidence.**
- `--self-test`: 6/6 ok.
- Mutant 1 (false-row set never populated): "a false row … is RED" FAILs.
- Mutant 2 (`covered=1` forced): 3 rows FAIL (deleted row, uncited PR, bare heading).
- Real history: `check_changelog_covers_merged.sh v0.69.1 v0.69.3 0.69.3` → exit 1,
  17 MISSING, 0 FALSE-ROW. The old row said PASS for that release.
- dogfood row block exercised on main (VERSION 0.69.3, fallback tag v0.69.1):
  `full` → WARN "misses 7 row(s)", `pre-publish` → FAIL.
- `bashrs lint` new script: 0 errors; `bashrs make lint Makefile`: 0 errors;
  `scripts/check_dogfood_version_row.sh`: PASS 33 rows (section marker unchanged).

**Not run.** A full `dogfood.sh --phase pre-publish` end to end (it builds/publishes-dry-runs
the crate); the row block was exercised in isolation with a stub `mark`.

**Consequence.** This reds current release practice (prose CHANGELOG with ~2 refs per
release). The 0.70.1 captain must write CHANGELOG coverage before pre-publish.
