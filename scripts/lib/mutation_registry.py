#!/usr/bin/env python3
"""Parse APR-PERF-GATE-001 §5 and check each row against the tree.

Owned by scripts/check_mutation_registry.sh (PERF-047, aprender#2752). Kept as a
library rather than inlined because the guard's --self-test drives it against a
synthetic root, and a checker that can only be run against the real repository
cannot be shown to turn RED.

Emits one TSV record per line on stdout, so the shell caller does no parsing:

    SPEC        <path>
    ROW         <status>        <key>
    VIOLATION   <rule>  <key>   <detail>

Exit status is always 0 unless the scan itself broke; the CALLER decides, so
that "found nothing" is a loud verdict row and never a swallowed error.

THE UNIVERSE IS DERIVED, NOT NAMED. The spec file is found by glob over
docs/specifications/APR-PERF-GATE-001-v*.md, because hardcoding v2.2 means a
v2.3 rename makes this guard silently vacuous -- the shape already fixed three
times in this epic (cascade TIERS[], book.yml paths, the check_*.sh glob in
check_guards_are_wired.sh).
"""

from __future__ import annotations

import glob
import os
import re
import sys

STATUSES = ("PROVEN", "PARTIAL", "UNPROVEN", "UNCOVERED")
CLAIMING = ("PROVEN", "PARTIAL", "UNCOVERED")

# A backticked token is a FILE token only if it looks like a path with a known
# extension. Prose cells ("verdict job (§4.9.1)") name no file, and that is
# legal for an UNPROVEN row and illegal for a claiming one -- which is the rule,
# not an omission.
FILE_TOKEN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:sh|py|rs|ya?ml|toml|md)$")
BACKTICKED = re.compile(r"`([^`]+)`")
SECTION_HEAD = re.compile(r"^##\s+.*§5\s+Mutation registry", re.IGNORECASE)
NEXT_HEAD = re.compile(r"^##\s")
SELFTEST_FLAG = re.compile(r"--self-?test")
SCRIPT_TOKEN = re.compile(r"[A-Za-z0-9_./-]+\.(?:sh|py)")
# An invocation, not a bare mention: the token starts a command, or follows
# bash/sh/./ or a shell separator.
INVOCATION_PREFIX = re.compile(r"(^|[\s;&|(])((ba)?sh\s+|\./)?$")


def emit(*fields: str) -> None:
    sys.stdout.write("\t".join(str(f).replace("\t", " ") for f in fields) + "\n")


def strip_md(cell: str) -> str:
    """Drop bold/italic markers so a status is compared as text, not markup."""
    return cell.replace("**", "").replace("*", "").strip()


def find_specs(root: str) -> list[str]:
    pattern = os.path.join(root, "docs", "specifications", "APR-PERF-GATE-001-v*.md")
    return sorted(glob.glob(pattern))


def section_lines(path: str) -> list[str]:
    """The lines of the §5 section: heading and the next §N heading excluded."""
    out: list[str] = []
    inside = False
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not inside:
                inside = bool(SECTION_HEAD.match(line))
                continue
            if NEXT_HEAD.match(line):
                break
            out.append(line)
    return out


def is_table_row(cells: list[str]) -> bool:
    """A data row: not the |---|---| rule line and not the column header."""
    if len(cells) < 2:
        return False
    if set("".join(cells)) <= set("-: "):
        return False
    return not cells[0].lower().startswith("gate / control")


def registry_rows(path: str) -> list[list[str]]:
    """The §5 table, as a list of cell lists. Header and rule line dropped."""
    rows = []
    for line in section_lines(path):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if is_table_row(cells):
            rows.append(cells)
    return rows


def invoked_in_line(line: str) -> list[str]:
    """Script basenames this line INVOKES with a self-test flag.

    Mention is not execution: check_guards_are_wired.sh learned that a guard
    named inside a `#` comment read as wired. `#` is stripped from the FIRST
    occurrence, not only on whole-line comments, because a TRAILING comment
    defeated the whole-line version there.
    """
    stripped = line.split("#", 1)[0]
    if not SELFTEST_FLAG.search(stripped):
        return []
    return [
        os.path.basename(m.group(0))
        for m in SCRIPT_TOKEN.finditer(stripped)
        if INVOCATION_PREFIX.search(stripped[: m.start()])
    ]


def workflow_selftest_invocations(root: str) -> set[str]:
    """Basenames a workflow invokes with a self-test flag."""
    found: set[str] = set()
    wf_dir = os.path.join(root, ".github", "workflows")
    paths: list[str] = []
    for pattern in ("*.yml", "*.yaml"):
        paths.extend(sorted(glob.glob(os.path.join(wf_dir, pattern))))
    for wf in paths:
        try:
            with open(wf, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    found.update(invoked_in_line(line))
        except OSError:
            continue
    return found


def row_status(cells: list[str], key: str) -> str:
    """The row's status token, or "" once the row has been reported as broken."""
    if len(cells) != 5:
        emit(
            "VIOLATION",
            "R3",
            key,
            f"row has {len(cells)} cell(s), expected 5"
            " (gate | file | mutation | discrimination | status)",
        )
        return ""
    text = strip_md(cells[4])
    status = text.split()[0] if text else ""
    if status in STATUSES:
        emit("ROW", status, key)
        return status
    emit(
        "VIOLATION",
        "R1",
        key,
        f"status starts with <{status or '(empty)'}>;"
        f" must be one of {', '.join(STATUSES)}",
    )
    # Every rule below is conditioned on the status, so an unclassifiable row is
    # reported once rather than three times.
    emit("ROW", "INVALID", key)
    return ""


def check_cells_present(cells: list[str], key: str) -> None:
    if not strip_md(cells[2]):
        emit("VIOLATION", "R3", key, "the Mutation cell is empty")
    if not strip_md(cells[3]):
        emit("VIOLATION", "R3", key, "the Discrimination cell is empty")


def resolve_token(root: str, token: str) -> str | None:
    for candidate in (os.path.join(root, token), os.path.join(root, "scripts", token)):
        if os.path.exists(candidate):
            return candidate
    return None


def file_tokens(cell: str) -> list[str]:
    return [t for t in BACKTICKED.findall(cell) if FILE_TOKEN.match(t)]


def check_one_token(root: str, token: str, key: str, selftested: set[str]) -> bool:
    """R2 for one token. Returns True if it names a file a workflow self-tests."""
    if resolve_token(root, token) is None:
        emit("VIOLATION", "R2", key, f"names `{token}`, which is not in the tree")
        return False
    return os.path.basename(token) in selftested


def check_files(
    root: str, cells: list[str], key: str, status: str, selftested: set[str]
) -> None:
    """R2 and R4: the named files must exist, and must agree with the status."""
    tokens = file_tokens(cells[1])
    # `any` would short-circuit and skip the R2 report on later tokens.
    covered = True in [check_one_token(root, t, key, selftested) for t in tokens]

    if status in CLAIMING and not tokens:
        emit(
            "VIOLATION",
            "R2",
            key,
            f"is {status} but names no backticked file — a row that claims"
            " anything about a mutation must name something a person can run",
        )
    if status == "UNPROVEN" and covered:
        emit(
            "VIOLATION",
            "R4",
            key,
            "marked UNPROVEN, but a workflow runs a file it names with a"
            " self-test flag. That combination is the PERF-047 drift verbatim:"
            " if the table does not cover this rule, say UNCOVERED and quote"
            " the mutation that stays green",
        )
    if status == "UNCOVERED" and tokens and not covered:
        emit(
            "VIOLATION",
            "R4",
            key,
            "marked UNCOVERED, but no workflow runs a self-test for any file it"
            " names. UNCOVERED means a case table exists and skips this rule;"
            " with no table the honest status is UNPROVEN",
        )


def check_spec(root: str, spec: str, selftested: set[str]) -> None:
    emit("SPEC", os.path.relpath(spec, root))
    for cells in registry_rows(spec):
        key = strip_md(cells[0])
        status = row_status(cells, key)
        if status:
            check_cells_present(cells, key)
            check_files(root, cells, key, status, selftested)


def check(root: str) -> None:
    specs = [p for p in find_specs(root) if registry_rows(p)]
    if not specs:
        emit(
            "VIOLATION",
            "R0",
            "-",
            "no §5 mutation registry table found under docs/specifications/"
            " APR-PERF-GATE-001-v*.md — the scan is broken, not the registry",
        )
        return
    selftested = workflow_selftest_invocations(root)
    for spec in specs:
        check_spec(root, spec, selftested)


def main(argv: list[str]) -> int:
    root = argv[1] if len(argv) > 1 else "."
    check(os.path.abspath(root))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
