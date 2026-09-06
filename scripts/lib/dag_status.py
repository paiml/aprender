"""dag_status.py — a DAG row's status is DERIVED from its receipt, never typed
(PP-066 row G-11, PMAT-1062, epic #2873; driver R2 of 2026-09-06).

The record of completion is docs/audits/impl-<pmat_id>-receipt.md and its
leading front-matter `status:` marker (the rule of C0-7, PMAT-1056,
scripts/check_receipt_complete.sh). A row is `complete` iff that marker says
complete; otherwise it is `open`. A `status:` key typed into the DAG is at
most a cache: render_dag.py ignores it and dag_invariants.py D7 refuses one
that disagrees with the receipt, so a row PR never has to (and never may)
edit docs/specifications/pp-066-dag.yaml to be counted.

    derived_status(root, row)   -> "complete" | "open"
    receipt_marker(path)        -> "complete" | "partial" | "none" | "torn"   (same rule as check_receipt_complete.sh)
    d7_typed_status(root, rows) -> ["D7 <id>: typed status `x` disagrees with the receipt (derived `y`)", ...]
"""
from __future__ import annotations

import os
import re

_STATUS = re.compile(r"^status:\s*(.*?)\s*$")


def receipt_path(root: str, row: dict):
    pid = row.get("pmat_id")
    return os.path.join(root, "docs", "audits", f"impl-{pid}-receipt.md") if pid else None


def _leading_front_matter(path):
    """The lines inside the FIRST `---` block, or None when the file does not open with one."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    if not lines or lines[0] != "---":
        return None
    body = []
    for line in lines[1:]:
        if line == "---":
            break
        body.append(line)
    return body


def _marker_value(front_matter) -> str:
    for line in front_matter or []:
        m = _STATUS.match(line)
        if m:
            v = m.group(1).strip().strip("\"'")
            return v if v in ("complete", "partial") else "none"
    return "none"


def receipt_marker(path) -> str:
    """The `status:` value inside the LEADING `---` block, nothing else (complete|partial|none|torn)."""
    if path is None or not os.path.isfile(path):
        return "none"
    if str(path).endswith(".tmp"):
        return "torn"
    return _marker_value(_leading_front_matter(path))


def derived_status(root: str, row: dict) -> str:
    return "complete" if receipt_marker(receipt_path(root, row)) == "complete" else "open"


def d7_typed_status(root: str, rows: dict) -> list:
    """A typed `status:` is tolerated only while it agrees with the receipt."""
    out = []
    for rid, r in rows.items():
        if "status" not in r:
            continue
        typed, derived = str(r.get("status")), derived_status(root, r)
        if typed != derived:
            out.append(f"D7 {rid}: typed status `{typed}` disagrees with the receipt (derived `{derived}` from docs/audits/impl-{r.get('pmat_id')}-receipt.md); status is derived, never typed")
    return out
