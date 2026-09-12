---
name: project_spec_conformance_strip_md_quirk
description: scripts/lib/spec_conformance.py strips ALL markdown ** and * from every table cell before any regex sees it — a regex requiring literal ** never matches
metadata:
  type: project
---

`scripts/lib/spec_conformance.py` (the §6/Appendix-C/§12 join behind
`scripts/spec_conformance.sh`) runs every `|`-delimited table cell through
`strip_md()` (`cell.replace("**", "").replace("*", "").strip()`) inside
`table_rows()`/`_pipe_cells()`/`ledger_rows_outside_table()` — this happens
for BOTH §6 and §12 tables, before `parse_dag()` or any downstream regex
(e.g. an "Expires marker" pattern) ever sees the cell text. A regex written
against the raw markdown (`r"Expires\s*\*\*(\d{4}-\d{2}-\d{2})\*\*"`) will
silently never match; write it against the post-strip text instead
(`r"Expires\s+(\d{4}-\d{2}-\d{2})"`).

**Why:** PMAT-974/I-26 (S0-2, issue #2889) needed the scanner to prefer the
`Expires **YYYY-MM-DD**` marker over the first bare date in a §12 root cell.
The first regex attempt (with `**` baked in) passed a standalone Python
unit-test with an unstripped string but failed silently end-to-end through
`spec_conformance.sh --selftest`, because the real code path always strips
markdown first.

**How to apply:** when adding any regex that inspects §6/§12/ledger cell
text in `scripts/lib/spec_conformance.py`, test it via
`python3 scripts/lib/spec_conformance.py <root> --no-out` against a real
fixture tree (or `bash scripts/spec_conformance.sh --selftest`), never a
bare `re.search` against the literal markdown string — the two inputs
differ by design.

Also useful: `--selftest`'s synthetic §12/§6 fixtures are built entirely in
bash inside `selftest()` (`scripts/spec_conformance.sh`) via a `mk_root`
helper that writes a throwaway repo tree under `mktemp -d`; a new case is
added as either a `row <name> <expected-violation-codes-CSV-or-CLEAN> <root>`
call (asserts the VIOLATION rule set) or, for asserting a specific derived
value (not just a violation code), a manual `_reg name; then; scan ...; awk
'$1=="DAG" ...'` block like `dag_derived_equals_max_blocker`.

`SPEC_CONFORMANCE_TODAY` (env var, read by `_today()` in the python scanner)
overrides "today" for the D6 (row past expiry) andon — set a fixture-wide
default early in `selftest()` well before every literal date any existing
fixture types (e.g. `2020-01-01`), then override per-row with a
`VAR=val row name ...` prefix for tests that need to exercise expiry. Bash's
temporary env-assignment prefix on a function call IS visible to nested
`scan()`/`python3` invocations inside that function for the duration of that
one command — no `export`/`unset` dance needed per-test.
