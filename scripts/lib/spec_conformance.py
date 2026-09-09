#!/usr/bin/env python3
"""Join PP-LLAMA-001-MASTER.md to the tree, and refuse the three ways it can lie.

Owned by scripts/spec_conformance.sh. It replaces scripts/lib/mutation_registry.py,
which located its input by glob over `APR-PERF-GATE-001-v*.md` and went silently
vacuous the moment that document was archived -- the shape it existed to catch.

Kept as a library rather than inlined because the guard's --selftest drives it
against synthetic roots, and a checker that can only be run against the real
repository cannot be shown to turn RED.

THREE JOINS, one per way the spec can drift from the tree:

  §6  Every row whose status starts with ARMED names selftest cases that EXIST,
      by name, on the SURFACE the row declares. A backticked name is not
      evidence; the name has to appear in a case table something runs.
  §Appendix C  PP-9: a cell, once run, is SPENT. No two ledger rows may share
      (host, workload, model quant, commit, interleaved) with conformance
      RECORDED -- re-running a cell until it comes out green is the defect, and
      it is only refusable against a written record.
  §12 Expiries are DERIVED from the blocked_by DAG, not typed. A root row carries
      a date; every other row's expiry is the LATEST among its transitive
      blockers. Cycles are refused, and so is a non-root row that types a date --
      that is how an expiry comes to be earlier than the work it waits on.

Emits one TSV record per line on stdout, so the shell caller does no parsing:

    SPEC        <path>
    ROW         <status>            <id>
    CASE        <surface>  <name>   <found|missing>   <id>
    LEDGER      <rows>     <recorded>
    DAG         <row>      <expires>          <derived-from>
    NOTE        <text>
    VIOLATION   <rule>     <key>    <detail>

Exit status is 0 unless the scan itself broke; the CALLER decides, so that
"found nothing" is a loud verdict row and never a swallowed error.
"""

from __future__ import annotations

import datetime
import glob
import json
import os
import unicodedata
import re
import subprocess
import sys

SPEC_GLOB = os.path.join("docs", "specifications", "PP-LLAMA-001-MASTER*.md")
# `--write` regenerates evidence/parity/derived_expiries.json; without it the
# scan COMPARES the derivation against the committed file and refuses drift
# (D5), so the guard never dirties a checkout and a stale committed derivation
# cannot pass.
WRITE_DERIVED = False
ID_ROW_RE = re.compile(r"^\|\s*\**(PP-(\d+))\**\s*\|")
LEDGER_REL = os.path.join("evidence", "parity", "LEDGER.md")
DERIVED_REL = os.path.join("evidence", "parity", "derived_expiries.json")

BACKTICKED = re.compile(r"`([^`]+)`")
SECTION_6 = re.compile(r"^#{1,4}\s+.*§\s*6\b", re.IGNORECASE)
SECTION_12 = re.compile(r"^#{1,4}\s+.*§\s*12\b", re.IGNORECASE)
NEXT_HEAD = re.compile(r"^#{1,4}\s")
# A bare case name: lowercase, digits, underscores. Anything with a slash or a
# dot is a producer PATH, and the producer column is not the join.
BARE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
SURFACES = ("pg", "sh", "rs")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
# I-26 (PMAT-974, S0-2): a §12 cell's expiry is this MARKER, never the first
# date the cell happens to mention. A root row narrates work in prose before
# its actual expiry ("witness taken 2026-09-02 ... Expires **2026-09-19**"),
# and DATE.search alone took the earlier prose date every time. table_rows()
# runs every cell through strip_md() before this ever sees it, so the `**`
# bold markers around the date are already gone -- match on the word alone.
EXPIRES_MARKER = re.compile(r"Expires\s+(\d{4}-\d{2}-\d{2})")
# `^\s+(ok|BROKE)\s+<name>` is the shape the master names, and the tree carries
# BOTH indentations: perf_gate.sh and check_perf_concurrency_groups.sh indent
# their rows, check_comparator_flags.sh and check_llama_pin.sh start at column 0.
# `\s*` rather than `\s+` so a guard is not reported as missing its own cases
# because of two spaces. It cannot invent a name: the line still has to BE a
# case row.
CASE_LINE = re.compile(r"^\s*(?:ok|BROKE)\s+(\S+)")
LIST_TOKEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ROW_ID = re.compile(r"(PP-\d+|row[- ]?\d+|\d+)", re.IGNORECASE)


def emit(*fields) -> None:
    sys.stdout.write("\t".join(str(f).replace("\t", " ") for f in fields) + "\n")


def strip_md(cell: str) -> str:
    return cell.replace("**", "").replace("*", "").strip()


# --------------------------------------------------------------- markdown ---
def section_lines(path: str, head: re.Pattern) -> list:
    out, inside = [], False
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not inside:
                inside = bool(head.match(line))
                continue
            if NEXT_HEAD.match(line):
                break
            out.append(line)
    return out


def _table_cells(line: str):
    """None for a line that is not a table row, [] for a separator row,
    else the row's cells."""
    if not line.lstrip().startswith("|"):
        return None
    cells = [strip_md(c) for c in line.strip().strip("|").split("|")]
    if set("".join(cells)) <= set("-: "):
        return []
    return cells


def table_rows(lines: list) -> tuple:
    """(header cells, data rows) of the FIRST pipe table in `lines`."""
    header, rows = [], []
    for line in lines:
        cells = _table_cells(line)
        if cells is None and rows:
            break
        if not cells:
            continue
        if header:
            rows.append(cells)
        else:
            header = [c.lower() for c in cells]
    return header, rows


def column(header: list, *needles) -> int:
    for i, name in enumerate(header):
        if all(n in name for n in needles):
            return i
    return -1


def _first_table_end(lines: list) -> int:
    """Line index (0-based) of the line that makes table_rows() stop.

    Mirrors table_rows()'s own break condition exactly, without changing what
    table_rows() returns, so a caller can find the lines the FIRST table
    never reached: a blank line or a prose line right after the last data row
    it accumulated.
    """
    header_seen = False
    rows_seen = False
    for i, line in enumerate(lines):
        if not line.lstrip().startswith("|"):
            if rows_seen:
                return i
            continue
        cells = [strip_md(c) for c in line.strip().strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        if not header_seen:
            header_seen = True
            continue
        rows_seen = True
    return len(lines)


ROW_ID_CELL = re.compile(r"^[0-9]+[a-z]?$")
SUPERSEDED_HEAD = re.compile(r"^#{1,4}\s*Superseded rows\b")


def _pipe_cells(line: str) -> list:
    """Cells of a pipe-table line, the leading and trailing pipe optional --
    [] when the line holds fewer than two pipes and so is not a table line."""
    if line.count("|") < 2:
        return []
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [strip_md(c) for c in body.split("|")]


def _is_separator(cells: list) -> bool:
    return bool(cells) and set("".join(cells)) <= set("-: ")


def _row_id(cells: list) -> str:
    """The first cell as a ledger row id, or "" -- backticks are stripped
    first, so `7` and 7 are the same id."""
    if not cells:
        return ""
    rid = _norm(cells[0])
    return rid if ROW_ID_CELL.match(rid) else ""


SPEND_TIERS = ("RECORDED", "CONFORMANT")


_WRAP = "`*_ \t"
_CODE_TAG = re.compile(r"</?code>", re.IGNORECASE)


def _norm(cell: str) -> str:
    """One normalisation for every cell the ledger rules read. What a
    formatting habit can put around or inside a value comes off: code tags,
    format characters (zero-width joiners and spaces), no-break spaces,
    runs of whitespace, and wrapping backticks, asterisks and underscores.
    What is left is the value a person meant; comparisons casefold it.
    THREAT MODEL, stated so the class is bounded: PP-9 is a discipline rule
    against the honest re-roll. A key disguised past this normaliser
    (homoglyphs, a changed digit) is a forged ledger row, and a forged row is
    a line the pull request's diff shows to its reviewers (PMAT-935)."""
    text = _CODE_TAG.sub("", cell)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    text = " ".join(text.replace("\u00a0", " ").split())
    return text.strip(_WRAP)


def _claims_spend(cells: list) -> bool:
    """A row that carries a conformance tier claims a spend. PP-9 binds on
    RECORDED (CONFORMANT implies it), so the universe of L2 is every row
    that claims one -- whatever its id, its width, its leading pipe, the
    table it was pasted under or the heading it sits below. RECORDED and
    CONFORMANT are therefore reserved words in the ledger file: a cell that
    starts with one is a spend claim wherever it is (PMAT-934)."""
    return any(_norm(c).upper().startswith(SPEND_TIERS) for c in cells)


def _outside_row(line: str):
    """(row id or the raw first cell, cells) when `line` is a pipe row that
    claims a spend, else None. The id is reported, never required: an id an
    author chooses must not be able to take a row out of the universe."""
    cells = _pipe_cells(line)
    if not cells or _is_separator(cells):
        return None
    if not _claims_spend(cells):
        return None
    return _row_id(cells) or _norm(cells[0]) or "?", cells


def ledger_rows_outside_table(lines: list) -> tuple:
    """(table_end, ((line index, row id, cells), ...)) -- every ledger row
    that sits after the first table breaks: L2's universe. A ledger row is
    any pipe line after the first table, to the end of the file, that
    claims a conformance tier; no id, run, header, width or region
    condition sits between such a row and the rule (four review rounds on
    #2861 each removed one that an author could satisfy)."""
    table_end = _first_table_end(lines)
    found = []
    for i in range(table_end, len(lines)):
        hit = _outside_row(lines[i])
        if hit is not None:
            found.append((i, hit[0], hit[1]))
    return table_end, tuple(found)


def _emit_malformed(rid: str, n: int, want: int) -> None:
    emit("VIOLATION", "L3", rid,
         "ledger row %s has %d cell(s) against a %d-cell header: a malformed "
         "row shifts every column the spend key reads, so it is refused "
         "rather than mis-keyed" % (rid, n, want))


def _check_ledger_shapes(rows: list, header: list) -> None:
    """Emit L3 for every first-table row whose cell count differs from the header's."""
    for cells in rows:
        if len(cells) != len(header):
            _emit_malformed(_row_id(cells) or (cells[0] if cells else "?"),
                            len(cells), len(header))


# ------------------------------------------------------------- §6: the join --
SURFACE_TOKEN = re.compile(r"(?<![\w:.-])(pg|sh:[^\s`,;)]+|rs:[^\s`,;)]+)(?![\w.-])")
PAREN = re.compile(r"\(([^()]*)\)")


def _tokens(text: str, default: str) -> list:
    r'''(surface, name) for the backticked names in one span.

    A surface token seen in the span RE-BINDS the surface for the names that
    follow it, which is what makes `(pg + rs:aprender-test-lib \`a\`, \`b\`)`
    read the way it looks: the two long names belong to the Rust crate, not to
    perf_gate.sh.
    '''
    out = []
    current = default
    for match in re.finditer(r"`([^`]+)`|" + SURFACE_TOKEN.pattern, text):
        if match.group(1) is None:
            current = match.group(2)
            continue
        token = match.group(1).strip()
        if ":" in token:
            # The prefixed spelling: `pg:name`, `sh:scripts/x.sh:name`. The NAME
            # is always after the LAST colon, so a script path parses
            # unambiguously.
            prefix, name = token.rsplit(":", 1)
            if prefix.split(":", 1)[0] in SURFACES:
                out.append((prefix, name))
            continue
        if BARE_NAME.match(token):
            out.append((current, token))
    return out


def _outside_default(spans: list) -> str:
    """The surface the names OUTSIDE every parenthetical belong to: the first
    surface token any parenthetical declares, else `pg`."""
    for span in spans:
        found = SURFACE_TOKEN.search(span)
        if found:
            return found.group(1)
    return "pg"


def _dedupe(pairs: list) -> list:
    seen, unique = set(), []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            unique.append(pair)
    return unique


def selftest_names(cell: str) -> list:
    """(surface, name) for every SELFTEST case a §6 row names.

    TWO SPELLINGS, both accepted, because the column has to be readable by a
    person and joinable by a program:

      prefixed    `pg:cellset_missing` / `sh:scripts/check_llama_pin.sh:pin_stale`
      annotated   `cellset_missing` / `cellset_na_ok` (pg)
                  `join_mismatch` / `join_ok` (pg + rs:aprender-test-lib `a`, `b`)

    In the annotated form the parenthetical declares the surface for the names
    BEFORE it, and any names INSIDE it belong to the last surface named inside.
    A cell with neither spelling defaults to `pg`, the perf_gate.sh case table.

    The producer half of a `producer · selftest` cell is dropped first: it names
    FILES, and a file is not a case name.
    """
    if "\u00b7" in cell:
        cell = cell.rsplit("\u00b7", 1)[1]
    spans = PAREN.findall(cell)
    outside_default = _outside_default(spans)
    inner = []
    for span in spans:
        found = SURFACE_TOKEN.search(span)
        inner.extend(_tokens(span, found.group(1) if found else outside_default))
    out = _tokens(PAREN.sub(" ", cell), outside_default)
    out.extend(inner)
    return _dedupe(out)


def _proc(args, cwd=None):
    try:
        return subprocess.run(args, capture_output=True, text=True, check=False, cwd=cwd)
    except OSError:
        return None


def _run(args, cwd=None):
    proc = _proc(args, cwd=cwd)
    return None if proc is None else proc.stdout + proc.stderr


class Surfaces:
    """Every case name a surface can produce, resolved once and cached."""

    def __init__(self, root: str):
        self.root = root
        self._cache = {}

    def names(self, prefix: str):
        if prefix not in self._cache:
            self._cache[prefix] = self._resolve(prefix)
        return self._cache[prefix]

    def _resolve(self, prefix: str):
        kind, _, rest = prefix.partition(":")
        if kind == "pg":
            return self._shell(os.path.join("scripts", "perf_gate.sh"))
        if kind == "sh":
            return self._shell(rest)
        if kind == "rs":
            return self._rust(rest)
        return None

    def _interpreter(self, path):
        return [sys.executable] if path.endswith(".py") else ["bash"]

    @staticmethod
    def _list_mode(run: list):
        """The names `--list-selftests` prints -- accepted only when it looks
        like a list: rc 0 and every non-empty line a bare identifier. A guard
        without list mode answers with a usage error or its whole case table,
        and either would be read as a set of names that happens to contain
        none of the ones being joined: a silent miss where the guard must be
        loud."""
        proc = _proc(run + ["--list-selftests"])
        if proc is None or proc.returncode != 0:
            return None
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        if lines and all(LIST_TOKEN.match(ln) for ln in lines):
            return set(lines)
        return None

    @staticmethod
    def _case_table(run: list):
        """The names on the case table's own `ok`/`BROKE` lines. Both spellings
        of the flag are tried because the tree carries both, and a guard that
        recognised only one would report half its siblings missing."""
        for flag in ("--selftest", "--self-test"):
            text = _run(run + [flag])
            if not text:
                continue
            matches = (CASE_LINE.match(line) for line in text.splitlines())
            found = {m.group(1) for m in matches if m}
            if found:
                return found
        return None

    def _shell(self, rel: str):
        path = os.path.join(self.root, rel)
        if not os.path.exists(path):
            return None
        run = self._interpreter(path) + [path]
        # LIST MODE FIRST; the case table only when there is no list mode.
        listed = self._list_mode(run)
        if listed is not None:
            return listed
        return self._case_table(run)

    @staticmethod
    def _test_fns(path: str) -> set:
        """Names of the `#[test]` functions in one Rust file: a `fn` whose
        three preceding lines carry the attribute."""
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        found = set()
        for i, line in enumerate(lines):
            match = re.match(r"^\s*(?:pub\s+)?fn\s+(\w+)\s*\(", line)
            if match and "#[test]" in "".join(lines[max(0, i - 3):i]):
                found.add(match.group(1))
        return found

    def _rust(self, crate: str):
        base = os.path.join(self.root, "crates", crate, "src")
        if not os.path.isdir(base):
            return None
        found = set()
        for dirpath, _dirs, files in os.walk(base):
            for name in files:
                if name.endswith(".rs"):
                    found |= self._test_fns(os.path.join(dirpath, name))
        return found


def _row_status(cells: list, status_at: int) -> str:
    if 0 <= status_at < len(cells) and cells[status_at]:
        return cells[status_at].split()[0]
    return ""


def _aliased(name: str, found_names: set) -> bool:
    return any(other.startswith(name + "__") for other in found_names)


def _check_case(key: str, prefix: str, name: str, surfaces: Surfaces,
                resolved: dict, found_names: set) -> None:
    available = surfaces.names(prefix)
    if not resolved[(prefix, name)] and _aliased(name, found_names):
        emit("CASE", prefix, name, "found", key)
        return
    if available is None:
        emit("VIOLATION", "C3", key,
             "names surface %r, which this tree cannot enumerate (the "
             "script, crate or list mode is absent)" % prefix)
        emit("CASE", prefix, name, "missing", key)
        return
    if name in available:
        emit("CASE", prefix, name, "found", key)
        return
    emit("CASE", prefix, name, "missing", key)
    emit("VIOLATION", "C1", key,
         "is ARMED and names `%s` on surface %s, which that case "
         "table does not contain. Rename the case or downgrade the "
         "row; a name in a table nobody runs is the thing this guard "
         "exists to refuse" % (name, prefix))


def _check_row_cases(key: str, names: list, surfaces: Surfaces) -> None:
    # A SHORT NAME MAY BE A ROW'S SHORTHAND FOR A LONGER CASE.
    # PP-32 reads `abrecord_comparator` / `abrecord_ok`
    # (rs:aprender-test-lib `abrecord_comparator__a_comparator_field_does_not_parse`,
    # `abrecord_ok__a_code_delta_with_two_shas_parses`). The two long names
    # are the tests; demanding `#[test] fn abrecord_comparator` as well would
    # demand a function nobody meant to write. A short name is therefore
    # satisfied by a longer one in the SAME row that extends it with the `__`
    # convention -- and only when that longer one was actually FOUND, so the
    # allowance cannot launder a missing case.
    resolved = {}
    for prefix, name in names:
        available = surfaces.names(prefix)
        resolved[(prefix, name)] = bool(available and name in available)
    found_names = {name for (_p, name), ok in resolved.items() if ok}
    for prefix, name in names:
        _check_case(key, prefix, name, surfaces, resolved, found_names)


def check_section_6(root: str, spec: str, surfaces: Surfaces) -> int:
    """Returns the number of rows parsed."""
    lines = section_lines(spec, SECTION_6)
    header, rows = table_rows(lines)
    if not rows:
        emit("VIOLATION", "C0", "-",
             "no §6 table parsed from %s -- the scan is broken, or §6 was emptied"
             % os.path.relpath(spec, root))
        return 0
    status_at = column(header, "status")
    for cells in rows:
        key = cells[0] or "?"
        status = _row_status(cells, status_at)
        emit("ROW", status or "(empty)", key)
        if not status.upper().startswith("ARMED"):
            continue
        names = selftest_names(cells[-1])
        if len(names) < 2:
            emit("VIOLATION", "C2", key,
                 "is ARMED and names %d selftest case(s). An armed rule has a "
                 "must-fire and a must-not-fire, both by name, or it is a claim "
                 "about a table nobody has run" % len(names))
            continue
        _check_row_cases(key, names, surfaces)
    return len(rows)


def _ledger_column_index(header: list) -> dict:
    idx = {
        "host": column(header, "host"),
        "workload": column(header, "workload"),
        "model": column(header, "model"),
        "commit": column(header, "commit"),
        "interleaved": column(header, "interleav"),
        "conformance": column(header, "conformance"),
    }
    missing = [k for k, v in idx.items() if v < 0]
    if missing:
        emit("VIOLATION", "L0", "-",
             "the first pipe table of the ledger carries no %s column: it is not "
             "the table PP-9 reads, so the spend check has nothing to key on -- a "
             "table written above the ledger shadows it" % ", ".join(sorted(missing)))
    return idx


def _ledger_cell(cells: list, idx: dict, name: str) -> str:
    i = idx[name]
    return _norm(cells[i]) if 0 <= i < len(cells) else ""


def _ledger_spend_key(cells: list, idx: dict) -> tuple:
    return tuple(_ledger_cell(cells, idx, k).casefold()
                 for k in ("host", "workload", "model", "commit", "interleaved"))


def _emit_respend(key: tuple) -> None:
    emit("VIOLATION", "L1", " ".join(k for k in key if k),
         "two ledger rows share the spend key (host, workload, model "
         "quant, commit, interleaved) with conformance RECORDED. PP-9: a "
         "cell, once run, is SPENT -- the second run is a re-roll, and "
         "the only legal move is a new commit")


def _check_ledger_spends(rows: list, idx: dict) -> int:
    """Emit L1 for every re-spent key among RECORDED rows; return the RECORDED count."""
    recorded = 0
    seen = set()
    for cells in rows:
        if not _ledger_cell(cells, idx, "conformance").upper().startswith(SPEND_TIERS):
            continue
        recorded += 1
        key = _ledger_spend_key(cells, idx)
        if key in seen:
            _emit_respend(key)
        seen.add(key)
    return recorded


def _emit_ledger_split(table_end: int, outside: tuple) -> None:
    emit("VIOLATION", "L2", " ".join(rid for _, rid, _ in outside),
         "%d ledger row(s) sit outside the table PP-9 reads: the table "
         "ends at line %d (a blank or prose line splits it) and a row "
         "with the same columns continues at line %d; every spent row "
         "must be contiguous with the header or the re-spend check "
         "never sees it"
         % (len(outside), table_end + 1, outside[0][0] + 1))


def _ledger_universe(lines: list, header: list, rows: list) -> list:
    """The rows PP-9's spend check reads: the first table's rows (L3 for a
    malformed one) plus every row L2 finds outside that table."""
    table_end, outside = ledger_rows_outside_table(lines)
    if outside:
        _emit_ledger_split(table_end, outside)
    _check_ledger_shapes(rows, header)
    return rows + [c for _, _, c in outside]


def check_ledger(root: str, ledger: str) -> None:
    if not os.path.exists(ledger):
        emit("VIOLATION", "L0", "-",
             "%s is absent -- PP-9 ('a cell, once run, is spent') is only "
             "enforceable against a written record of what has been spent"
             % os.path.relpath(ledger, root))
        return
    with open(ledger, encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    header, rows = table_rows(lines)
    if not rows:
        emit("VIOLATION", "L0", "-", "no row table parsed from the ledger")
        return
    universe = _ledger_universe(lines, header, rows)
    recorded = _check_ledger_spends(universe, _ledger_column_index(header))
    emit("LEDGER", len(universe), recorded)
    if recorded == 0:
        emit("NOTE", "no ledger row is marked conformance RECORDED, so the PP-9 "
                     "duplicate rule matched nothing on this tree; its must-fire "
                     "lives in the fixture rows of --selftest")


# ------------------------------------------------------- §12: the expiry DAG --
def _dag_columns(header: list):
    """(row, blocked_by, expires) column indexes of the §12 table, or None
    with D0 emitted when the two derived-from columns are absent."""
    row_at = column(header, "row")
    if row_at < 0:
        row_at = column(header, "id")
    if row_at < 0:
        row_at = 0
    blocked_at = column(header, "blocked")
    expires_at = column(header, "expir")
    if blocked_at < 0 or expires_at < 0:
        emit("VIOLATION", "D0", "-",
             "§12 has no `blocked_by` and/or `expires` column (header: %s)" % header)
        return None
    return row_at, blocked_at, expires_at


def _dag_cell(cells: list, i: int) -> str:
    return cells[i].strip().strip("`") if 0 <= i < len(cells) else ""


def _blockers(key: str, text: str, ids) -> list:
    """The row ids `text` names, as whole tokens. Matching whole tokens
    against the known ids is what lets a cell say "15 clean at `--phase
    merge`" or "— (needs a gx10 window)" and still be parsed exactly: prose
    around an id is prose, and a digit inside another word (`gx10`) is not
    an id."""
    return [other for other in ids
            if other != key
            and re.search(r"(?<![\w-])%s(?![\w-])" % re.escape(other), text)]


def parse_dag(root: str, spec: str):
    lines = section_lines(spec, SECTION_12)
    header, rows = table_rows(lines)
    if not rows:
        emit("VIOLATION", "D0", "-",
             "no §12 table parsed -- every non-root expiry is DERIVED from the "
             "blocked_by column, and with no table there is nothing to derive from")
        return None
    cols = _dag_columns(header)
    if cols is None:
        return None
    row_at, blocked_at, expires_at = cols
    table, raw = {}, {}
    for cells in rows:
        key = _dag_cell(cells, row_at)
        if not key:
            continue
        raw[key] = _dag_cell(cells, blocked_at)
        table[key] = {"blocked_by": [], "expires_cell": _dag_cell(cells, expires_at)}
    # TWO PASSES, because a blocker is named by ROW ID and the id set is only
    # known once every row is read.
    for key, text in raw.items():
        table[key]["blocked_by"] = _blockers(key, text, table)
    return table


class _Expiries:
    """Derived expiry per row: max over transitive blockers. Refuses cycles."""

    def __init__(self, table: dict):
        self.table = table
        self.out = {}
        self.state = {}

    def _finish(self, key, value, via):
        self.out[key] = value
        self.table[key]["derived_from"] = via
        self.state[key] = "done"
        return value

    def _cycle(self, key, stack):
        emit("VIOLATION", "D1", key,
             "blocked_by forms a CYCLE (%s). A cycle has no latest blocker, "
             "so every row in it would wait for itself" % " -> ".join(stack + [key]))
        self.out[key] = None
        return None

    def _live(self, row) -> list:
        return [b for b in row["blocked_by"]
                if b in self.table and "LANDED" not in self.table[b]["expires_cell"].upper()]

    def _unblocked(self, key, literal):
        # A row every one of whose blockers has LANDED is unblocked: there is
        # nothing left to derive an expiry from, so it is a root again and its
        # date is the one a person must write. Refusing a date there would leave
        # the row with no deadline at all, which is the failure mode the whole
        # derivation exists to prevent.
        if literal is None:
            emit("VIOLATION", "D2", key,
                 "is blocked only by rows that have LANDED, so nothing derives "
                 "its expiry any more, and it carries no date. An unblocked "
                 "obligation with no deadline never expires")
        return self._finish(key, literal, [])

    def _root(self, key, literal, cell):
        if literal is None:
            emit("VIOLATION", "D2", key,
                 "is a ROOT row (nothing blocks it) and carries no literal "
                 "expiry %r. A root has nothing to derive from, so its date "
                 "is the one date a person must write" % cell)
        return self._finish(key, literal, [])

    @staticmethod
    def _typed_on_blocked(key, row, live, literal):
        if literal is None:
            return
        emit("VIOLATION", "D3", key,
             "is blocked by %s (still live: %s) and still types the literal "
             "date %s. §12's own preamble says `expires` is a date only on "
             "root rows; a typed expiry on a blocked row can fall BEFORE the "
             "work it waits on, which is how a gate comes to be red for a "
             "reason nobody can clear"
             % (", ".join(row["blocked_by"]), ", ".join(live), literal))

    def _blocker_value(self, key, blocker, stack):
        if blocker not in self.table:
            emit("VIOLATION", "D4", key,
                 "is blocked_by %r, which is not a row in §12" % blocker)
            return None
        return self.visit(blocker, stack + [key])

    def _best_blocker(self, key, row, stack):
        best, via = None, []
        for blocker in row["blocked_by"]:
            value = self._blocker_value(key, blocker, stack)
            if value is None:
                continue
            if best is None or value > best:
                best, via = value, [blocker]
            elif value == best:
                via.append(blocker)
        return best, via

    def _from_blockers(self, key, row, stack, live, literal):
        self._typed_on_blocked(key, row, live, literal)
        best, via = self._best_blocker(key, row, stack)
        return self._finish(key, best, via)

    def _resolve(self, key, row, stack):
        cell = row["expires_cell"]
        marker = EXPIRES_MARKER.search(cell)
        if marker:
            literal = marker.group(1)
        else:
            found = DATE.search(cell)
            literal = found.group(0) if found else None
        live = self._live(row)
        if row["blocked_by"] and not live:
            return self._unblocked(key, literal)
        if not row["blocked_by"]:
            return self._root(key, literal, cell)
        return self._from_blockers(key, row, stack, live, literal)

    def visit(self, key, stack):
        if key in self.out:
            return self.out[key]
        if self.state.get(key) == "open":
            return self._cycle(key, stack)
        self.state[key] = "open"
        row = self.table[key]
        cell = row["expires_cell"]
        # A DISCHARGED row has no deadline to derive: the work landed, so there
        # is nothing left to wait for and nothing left to expire. Treating it as
        # a root with a missing date would demand a date for finished work, and
        # treating it as a live blocker would keep its dependents waiting on it
        # forever.
        row["discharged"] = "LANDED" in cell.upper()
        if row["discharged"]:
            return self._finish(key, None, [])
        return self._resolve(key, row, stack)


def derive_expiries(table: dict) -> dict:
    """Derived expiry per row: max over transitive blockers. Refuses cycles."""
    walk = _Expiries(table)
    for key in table:
        walk.visit(key, [])
    return walk.out


def _dag_reason(row: dict) -> str:
    if row.get("derived_from"):
        return ",".join(row["derived_from"])
    if row.get("discharged"):
        return "discharged"
    return "root" if not row["blocked_by"] else "unblocked"


def _today() -> str:
    """"Today", for deciding whether a §12 row is past its derived expiry
    (D6). SPEC_CONFORMANCE_TODAY overrides it -- read by the --selftest case
    table so the andon is exercised deterministically, never by the wall
    clock at whatever moment the suite happens to run."""
    override = os.environ.get("SPEC_CONFORMANCE_TODAY", "").strip()
    return override or datetime.date.today().isoformat()


def _dag_document(table: dict, derived: dict, today: str) -> dict:
    document = {}
    for key in sorted(table):
        row = table[key]
        expires = derived.get(key)
        document[key] = {
            "expires": expires,
            "root": not row["blocked_by"],
            "discharged": bool(row.get("discharged")),
            "blocked_by": row["blocked_by"],
            "derived_from": row.get("derived_from", []),
        }
        emit("DAG", key, expires or "-", _dag_reason(row))
        # §4 andon: a row past its derived expiry, and not discharged (no
        # LANDED marker), must turn the whole scan RED. A row with no
        # derivable expiry (D2/D1 already fired for it) has nothing to
        # compare and is skipped here, not double-counted.
        if expires and not row.get("discharged") and expires < today:
            emit("VIOLATION", "D6", key,
                 "expired %s (today is %s) and is not discharged. §4's andon "
                 "refuses a row past its expiry with nothing red for it"
                 % (expires, today))
    return document


def _first_difference(have: list, want: list) -> int:
    return next((i for i, (a, b) in enumerate(zip(have, want), 1) if a != b),
                min(len(have), len(want)) + 1)


def _write_derived(out_path: str, text: str) -> None:
    directory = os.path.dirname(out_path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _compare_derived(out_path: str, rel: str, text: str) -> None:
    if not os.path.exists(out_path):
        emit("VIOLATION", "D5", "-",
             "%s is absent. The derived expiries are an artifact other gates "
             "read; run `bash scripts/spec_conformance.sh --write` and commit it"
             % rel)
        return
    with open(out_path, encoding="utf-8") as fh:
        committed = fh.read()
    if committed == text:
        return
    first = _first_difference(committed.splitlines(), text.splitlines())
    emit("VIOLATION", "D5", "-",
         "%s no longer matches the derivation from §12 (first difference "
         "at line %d). A committed derivation that drifts from the table "
         "is the expiry nobody re-derived; run `bash "
         "scripts/spec_conformance.sh --write` and commit the result"
         % (rel, first))


def check_dag(root: str, spec: str, out_path: str) -> None:
    table = parse_dag(root, spec)
    if table is None:
        return
    document = _dag_document(table, derive_expiries(table), _today())
    if not out_path:
        return
    text = json.dumps({"source": os.path.relpath(spec, root),
                       "rows": document}, indent=2, sort_keys=True) + "\n"
    if WRITE_DERIVED:
        _write_derived(out_path, text)
    else:
        _compare_derived(out_path, os.path.relpath(out_path, root), text)


def find_spec(root: str, override: str):
    if override:
        return [override] if os.path.exists(override) else []
    return sorted(glob.glob(os.path.join(root, SPEC_GLOB)))


def check(root: str, spec_override: str, ledger_override: str, out_path: str) -> int:
    specs = find_spec(root, spec_override)
    if len(specs) != 1:
        emit("VIOLATION", "C0", "-",
             "found %d file(s) matching %s -- the master is the ONE document this "
             "guard joins; zero means the scan is broken and more than one means "
             "two specs disagree" % (len(specs), spec_override or SPEC_GLOB))
        return 0
    spec = specs[0]
    emit("SPEC", os.path.relpath(spec, root))
    surfaces = Surfaces(root)
    parsed = check_section_6(root, spec, surfaces)
    check_id_contiguity(spec)
    check_ledger(root, ledger_override or os.path.join(root, LEDGER_REL))
    check_dag(root, spec, out_path)
    return parsed


def check_id_contiguity(spec: str) -> None:
    """§0.6: `PP-nn` are stable and a retired rule KEEPS its number with the
    status RETIRED. So the ids in §6 must be exactly PP-1..PP-max: a gap is a
    deleted invariant, and a vacuity floor alone cannot see one deletion inside
    a table that is still above the floor."""
    nums = set()
    with open(spec, encoding="utf-8") as fh:
        for line in fh:
            m = ID_ROW_RE.match(line)
            if m:
                nums.add(int(m.group(2)))
    if not nums:
        return
    gaps = sorted(set(range(1, max(nums) + 1)) - nums)
    if gaps:
        emit("VIOLATION", "C5", "PP-%d" % gaps[0],
             "§6 is missing %s (ids run PP-1..PP-%d). §0.6 keeps a retired rule's "
             "number with status RETIRED, so a gap is an invariant that was DELETED "
             "rather than retired"
             % (", ".join("PP-%d" % g for g in gaps), max(nums)))


def _parse_args(rest: list, out: str) -> tuple:
    """(spec, ledger, out) from the flags after the root argument."""
    global WRITE_DERIVED
    paths = {"--spec": "", "--ledger": "", "--out": out}
    for i, arg in enumerate(rest):
        if arg in paths and i + 1 < len(rest):
            paths[arg] = os.path.abspath(rest[i + 1])
        elif arg == "--no-out":
            paths["--out"] = ""
        elif arg == "--write":
            WRITE_DERIVED = True
    return paths["--spec"], paths["--ledger"], paths["--out"]


def main(argv) -> int:
    root = os.path.abspath(argv[1] if len(argv) > 1 else ".")
    spec, ledger, out = _parse_args(argv[2:], os.path.join(root, DERIVED_REL))
    emit("PARSED", check(root, spec, ledger, out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
