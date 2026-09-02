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

import glob
import json
import os
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


def table_rows(lines: list) -> tuple:
    """(header cells, data rows) of the FIRST pipe table in `lines`."""
    header, rows = [], []
    for line in lines:
        if not line.lstrip().startswith("|"):
            if rows:
                break
            continue
        cells = [strip_md(c) for c in line.strip().strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        if not header:
            header = [c.lower() for c in cells]
            continue
        rows.append(cells)
    return header, rows


def column(header: list, *needles) -> int:
    for i, name in enumerate(header):
        if all(n in name for n in needles):
            return i
    return -1


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
    outside_default = "pg"
    for span in spans:
        found = SURFACE_TOKEN.search(span)
        if found:
            outside_default = found.group(1)
            break
    inner = []
    for span in spans:
        found = SURFACE_TOKEN.search(span)
        inner.extend(_tokens(span, found.group(1) if found else outside_default))
    out = _tokens(PAREN.sub(" ", cell), outside_default)
    out.extend(inner)
    seen, unique = set(), []
    for pair in out:
        if pair not in seen:
            seen.add(pair)
            unique.append(pair)
    return unique


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

    def _shell(self, rel: str):
        path = os.path.join(self.root, rel)
        if not os.path.exists(path):
            return None
        run = self._interpreter(path) + [path]
        # LIST MODE FIRST, and it is accepted only when it looks like a list: rc
        # 0 and every non-empty line a bare identifier. A guard without list mode
        # answers `--list-selftests` with a usage error or with its whole case
        # table, and either would be read as a set of names that happens to
        # contain none of the ones being joined -- a silent miss where the guard
        # must be loud.
        proc = _proc(run + ["--list-selftests"])
        if proc is not None and proc.returncode == 0:
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            if lines and all(LIST_TOKEN.match(ln) for ln in lines):
                return set(lines)
        # No list mode: read the case table's own `ok`/`BROKE` lines. Both
        # spellings of the flag are tried because the tree carries both, and a
        # guard that recognised only one would report half its siblings missing.
        found = set()
        for flag in ("--selftest", "--self-test"):
            text = _run(run + [flag])
            if not text:
                continue
            for line in text.splitlines():
                match = CASE_LINE.match(line)
                if match:
                    found.add(match.group(1))
            if found:
                break
        return found or None

    def _rust(self, crate: str):
        base = os.path.join(self.root, "crates", crate, "src")
        if not os.path.isdir(base):
            return None
        found = set()
        for dirpath, _dirs, files in os.walk(base):
            for name in files:
                if not name.endswith(".rs"):
                    continue
                with open(os.path.join(dirpath, name), encoding="utf-8", errors="replace") as fh:
                    lines = fh.readlines()
                for i, line in enumerate(lines):
                    match = re.match(r"^\s*(?:pub\s+)?fn\s+(\w+)\s*\(", line)
                    if not match:
                        continue
                    window = "".join(lines[max(0, i - 3):i])
                    if "#[test]" in window:
                        found.add(match.group(1))
        return found


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
    parsed = 0
    for cells in rows:
        key = cells[0] or "?"
        parsed += 1
        status = ""
        if 0 <= status_at < len(cells):
            status = cells[status_at].split()[0] if cells[status_at] else ""
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

        def aliased(name):
            return any(other.startswith(name + "__") for other in found_names)

        for prefix, name in names:
            available = surfaces.names(prefix)
            if not resolved[(prefix, name)] and aliased(name):
                emit("CASE", prefix, name, "found", key)
                continue
            if available is None:
                emit("VIOLATION", "C3", key,
                     "names surface %r, which this tree cannot enumerate (the "
                     "script, crate or list mode is absent)" % prefix)
                emit("CASE", prefix, name, "missing", key)
                continue
            if name in available:
                emit("CASE", prefix, name, "found", key)
            else:
                emit("CASE", prefix, name, "missing", key)
                emit("VIOLATION", "C1", key,
                     "is ARMED and names `%s` on surface %s, which that case "
                     "table does not contain. Rename the case or downgrade the "
                     "row; a name in a table nobody runs is the thing this guard "
                     "exists to refuse" % (name, prefix))
    return parsed


# ------------------------------------------------- Appendix C: PP-9 spending --
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
        emit("NOTE", "the ledger table carries no %s column, so those key "
                     "components are absent from every spend key" % ", ".join(sorted(missing)))
    recorded = 0
    seen = {}
    for cells in rows:
        def cell(name):
            i = idx[name]
            return cells[i].strip().strip("`") if 0 <= i < len(cells) else ""
        conformance = cell("conformance")
        if not conformance.upper().startswith("RECORDED"):
            continue
        recorded += 1
        key = tuple(cell(k) for k in ("host", "workload", "model", "commit", "interleaved"))
        if key in seen:
            emit("VIOLATION", "L1", " ".join(k for k in key if k),
                 "two ledger rows share the spend key (host, workload, model "
                 "quant, commit, interleaved) with conformance RECORDED. PP-9: a "
                 "cell, once run, is SPENT -- the second run is a re-roll, and "
                 "the only legal move is a new commit")
        seen[key] = True
    emit("LEDGER", len(rows), recorded)
    if recorded == 0:
        emit("NOTE", "no ledger row is marked conformance RECORDED, so the PP-9 "
                     "duplicate rule matched nothing on this tree; its must-fire "
                     "lives in the fixture rows of --selftest")


# ------------------------------------------------------- §12: the expiry DAG --
def parse_dag(root: str, spec: str):
    lines = section_lines(spec, SECTION_12)
    header, rows = table_rows(lines)
    if not rows:
        emit("VIOLATION", "D0", "-",
             "no §12 table parsed -- every non-root expiry is DERIVED from the "
             "blocked_by column, and with no table there is nothing to derive from")
        return None
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
    table, raw = {}, {}
    for cells in rows:
        def cell(i):
            return cells[i].strip().strip("`") if 0 <= i < len(cells) else ""
        key = cell(row_at)
        if not key:
            continue
        raw[key] = cell(blocked_at)
        table[key] = {"blocked_by": [], "expires_cell": cell(expires_at)}
    # TWO PASSES, because a blocker is named by ROW ID and the id set is only
    # known once every row is read. Matching whole tokens against the known ids
    # is what lets a cell say "15 clean at `--phase merge`" or "— (needs a gx10
    # window)" and still be parsed exactly: prose around an id is prose, and a
    # digit inside another word (`gx10`) is not an id.
    for key, text in raw.items():
        table[key]["blocked_by"] = [
            other for other in table
            if other != key and re.search(r"(?<![\w-])%s(?![\w-])" % re.escape(other), text)]
    return table


def derive_expiries(table: dict) -> dict:
    """Derived expiry per row: max over transitive blockers. Refuses cycles."""
    out = {}
    state = {}

    def visit(key, stack):
        if key in out:
            return out[key]
        if state.get(key) == "open":
            emit("VIOLATION", "D1", key,
                 "blocked_by forms a CYCLE (%s). A cycle has no latest blocker, "
                 "so every row in it would wait for itself" % " -> ".join(stack + [key]))
            out[key] = None
            return None
        state[key] = "open"
        row = table[key]
        cell = row["expires_cell"]
        found = DATE.search(cell)
        literal = found.group(0) if found else None
        # A DISCHARGED row has no deadline to derive: the work landed, so there
        # is nothing left to wait for and nothing left to expire. Treating it as
        # a root with a missing date would demand a date for finished work, and
        # treating it as a live blocker would keep its dependents waiting on it
        # forever.
        row["discharged"] = "LANDED" in cell.upper()
        if row["discharged"]:
            out[key] = None
            row["derived_from"] = []
            state[key] = "done"
            return None
        # A row every one of whose blockers has LANDED is unblocked: there is
        # nothing left to derive an expiry from, so it is a root again and its
        # date is the one a person must write. Refusing a date there would leave
        # the row with no deadline at all, which is the failure mode the whole
        # derivation exists to prevent.
        live = [b for b in row["blocked_by"]
                if b in table and "LANDED" not in table[b]["expires_cell"].upper()]
        if row["blocked_by"] and not live:
            if literal is None:
                emit("VIOLATION", "D2", key,
                     "is blocked only by rows that have LANDED, so nothing derives "
                     "its expiry any more, and it carries no date. An unblocked "
                     "obligation with no deadline never expires")
            out[key] = literal
            row["derived_from"] = []
            state[key] = "done"
            return out[key]
        if not row["blocked_by"]:
            if literal is None:
                emit("VIOLATION", "D2", key,
                     "is a ROOT row (nothing blocks it) and carries no literal "
                     "expiry %r. A root has nothing to derive from, so its date "
                     "is the one date a person must write" % cell)
            out[key] = literal
            row["derived_from"] = []
            state[key] = "done"
            return out[key]
        if literal is not None:
            emit("VIOLATION", "D3", key,
                 "is blocked by %s (still live: %s) and still types the literal "
                 "date %s. §12's own preamble says `expires` is a date only on "
                 "root rows; a typed expiry on a blocked row can fall BEFORE the "
                 "work it waits on, which is how a gate comes to be red for a "
                 "reason nobody can clear"
                 % (", ".join(row["blocked_by"]), ", ".join(live), literal))
        best, via = None, []
        for blocker in row["blocked_by"]:
            if blocker not in table:
                emit("VIOLATION", "D4", key,
                     "is blocked_by %r, which is not a row in §12" % blocker)
                continue
            value = visit(blocker, stack + [key])
            if value is not None and (best is None or value > best):
                best, via = value, [blocker]
            elif value is not None and value == best:
                via.append(blocker)
        out[key] = best
        row["derived_from"] = via
        state[key] = "done"
        return best

    for key in table:
        visit(key, [])
    return out


def check_dag(root: str, spec: str, out_path: str) -> None:
    table = parse_dag(root, spec)
    if table is None:
        return
    derived = derive_expiries(table)
    document = {}
    for key in sorted(table):
        document[key] = {
            "expires": derived.get(key),
            "root": not table[key]["blocked_by"],
            "discharged": bool(table[key].get("discharged")),
            "blocked_by": table[key]["blocked_by"],
            "derived_from": table[key].get("derived_from", []),
        }
        emit("DAG", key, derived.get(key) or "-",
             ",".join(table[key].get("derived_from") or [])
             or ("discharged" if table[key].get("discharged")
                 else "root" if not table[key]["blocked_by"] else "unblocked"))
    if out_path:
        text = json.dumps({"source": os.path.relpath(spec, root),
                           "rows": document}, indent=2, sort_keys=True) + "\n"
        rel = os.path.relpath(out_path, root)
        if WRITE_DERIVED:
            directory = os.path.dirname(out_path)
            if directory and not os.path.isdir(directory):
                os.makedirs(directory)
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(text)
        elif not os.path.exists(out_path):
            emit("VIOLATION", "D5", "-",
                 "%s is absent. The derived expiries are an artifact other gates "
                 "read; run `bash scripts/spec_conformance.sh --write` and commit it"
                 % rel)
        else:
            with open(out_path, encoding="utf-8") as fh:
                committed = fh.read()
            if committed != text:
                have = committed.splitlines()
                want = text.splitlines()
                first = next((i for i, (a, b) in enumerate(zip(have, want), 1) if a != b),
                             min(len(have), len(want)) + 1)
                emit("VIOLATION", "D5", "-",
                     "%s no longer matches the derivation from §12 (first difference "
                     "at line %d). A committed derivation that drifts from the table "
                     "is the expiry nobody re-derived; run `bash "
                     "scripts/spec_conformance.sh --write` and commit the result"
                     % (rel, first))


# --------------------------------------------------------------------- main --
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


def main(argv) -> int:
    root = os.path.abspath(argv[1] if len(argv) > 1 else ".")
    spec = ""
    ledger = ""
    out = os.path.join(root, DERIVED_REL)
    rest = argv[2:]
    for i, arg in enumerate(rest):
        if arg == "--spec" and i + 1 < len(rest):
            spec = os.path.abspath(rest[i + 1])
        elif arg == "--ledger" and i + 1 < len(rest):
            ledger = os.path.abspath(rest[i + 1])
        elif arg == "--out" and i + 1 < len(rest):
            out = os.path.abspath(rest[i + 1])
        elif arg == "--no-out":
            out = ""
        elif arg == "--write":
            global WRITE_DERIVED
            WRITE_DERIVED = True
    emit("PARSED", check(root, spec, ledger, out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
