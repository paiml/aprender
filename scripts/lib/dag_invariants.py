#!/usr/bin/env python3
"""dag_invariants.py — the PP-066 obligation DAG is data, and its invariants
are checked here rather than re-read by every reader (PMAT-987, G-4, #2902).

    python3 scripts/lib/dag_invariants.py check docs/specifications/pp-066-dag.yaml [--min-slack-days 6] [--today YYYY-MM-DD]

Rules (each has a RED and a GREEN row in check_dag_invariants.sh --selftest):
  D1  every blocker names a row that exists (no dangling edge)
  D2  the blocker graph is acyclic
  D3  slack: for every 0.66-lane row R and every blocker B of R,
      expires(R) - expires(B) >= min_slack_days  (the review of 2026-09-04
      found W-H expiring the day of the row it should block; the rule is
      spec §2's 6-day floor, never a number invented here)
  D4  per-host queues (host_queues in the header) are ordered by expiry:
      queue_pos(h, a) < queue_pos(h, b)  =>  expires(a) <= expires(b)
  D5  every row has a non-empty owner
  D6  every row has exactly one expiry form: a YYYY-MM-DD date, or
      {anchor: <row id>, days: <int>} — never both, never neither, and an
      anchor must name a row that exists (resolved recursively)
Reported, never a violation: rows whose resolved expiry is before --today
(the §4 andon is a separate obligation; this checker only makes the list
mechanical so no row expires silently).

Exit 0 all rules hold; 1 any violation; 2 a usage/read error (the box cannot
answer — a missing file is not a passing DAG).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

try:
    import yaml
except ImportError:  # pragma: no cover
    print("dag_invariants: PyYAML is required", file=sys.stderr)
    sys.exit(2)


class UsageError(Exception):
    """A read/parse failure — exit 2, never a rule verdict."""


def load_dag(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as e:
        raise UsageError(f"cannot read {path}: {e}") from e
    if not isinstance(doc, dict) or not isinstance(doc.get("rows"), list):
        raise UsageError(f"{path}: no top-level `rows:` list")
    return doc


def rows_by_id(doc: dict) -> dict:
    out: dict = {}
    for r in doc["rows"]:
        rid = r.get("id")
        if not isinstance(rid, str) or not rid:
            raise UsageError("a row has no string `id`")
        if rid in out:
            raise UsageError(f"duplicate row id {rid}")
        out[rid] = r
    return out


def blockers_of(row: dict) -> list:
    return row.get("blockers") or []


# --- expiry ---------------------------------------------------------------
def _is_iso_date(s: str) -> bool:
    try:
        date.fromisoformat(s)
        return True
    except ValueError:
        return False


def _is_anchor(e: dict) -> bool:
    return set(e.keys()) == {"anchor", "days"} and isinstance(e.get("anchor"), str) and isinstance(e.get("days"), int)


def expiry_form(row: dict) -> str:
    """'date' | 'anchor' | 'none' | 'malformed'."""
    e = row.get("expiry")
    if e is None:
        return "none"
    if isinstance(e, date):  # PyYAML parses bare dates
        return "date"
    if isinstance(e, str):
        return "date" if _is_iso_date(e) else "malformed"
    if isinstance(e, dict):
        return "anchor" if _is_anchor(e) else "malformed"
    return "malformed"


def _as_date(e) -> date:
    return e if isinstance(e, date) else date.fromisoformat(e)


def resolve_expiry(rid: str, rows: dict, _seen: tuple = ()) -> date:
    if rid in _seen:
        raise ValueError(f"anchor cycle through {rid}")
    row = rows[rid]
    form = expiry_form(row)
    if form == "date":
        return _as_date(row["expiry"])
    if form != "anchor":
        raise ValueError(f"{rid}: expiry form {form}")
    anchor = row["expiry"]["anchor"]
    if anchor not in rows:
        raise ValueError(f"{rid}: anchor {anchor} is not a row")
    return resolve_expiry(anchor, rows, _seen + (rid,)) + timedelta(days=row["expiry"]["days"])


# --- rules ----------------------------------------------------------------
def d1_dangling(rows: dict) -> list:
    return [f"D1 {rid}: blocker `{b}` is not a row" for rid, r in rows.items() for b in blockers_of(r) if b not in rows]


def _visit(u: str, path: list, rows: dict, colour: dict, out: list) -> None:
    colour[u] = 1
    for v in blockers_of(rows[u]):
        if v not in rows:
            continue
        state = colour.get(v)
        if state == 1:
            out.append("D2 cycle: " + " -> ".join(path + [u, v]))
        elif state is None:
            _visit(v, path + [u], rows, colour, out)
    colour[u] = 2


def d2_cycles(rows: dict) -> list:
    colour: dict = {}
    out: list = []
    for rid in rows:
        if colour.get(rid) is None:
            _visit(rid, [], rows, colour, out)
    return out


def _d6_one(rid: str, r: dict, rows: dict):
    form = expiry_form(r)
    if form not in ("date", "anchor"):
        return f"D6 {rid}: expiry must be a date or {{anchor, days}} (got {form})"
    if form == "anchor" and r["expiry"]["anchor"] not in rows:
        return f"D6 {rid}: anchor `{r['expiry']['anchor']}` is not a row"
    return None


def d6_expiry_forms(rows: dict) -> list:
    return [m for rid, r in rows.items() for m in [_d6_one(rid, r, rows)] if m]


def resolved_expiries(rows: dict) -> "tuple[dict, list]":
    exp: dict = {}
    out: list = []
    for rid in rows:
        try:
            exp[rid] = resolve_expiry(rid, rows)
        except ValueError as e:
            out.append(f"D6 {e}")
    return exp, out


def _slack_of(rid: str, r: dict, exp: dict, min_days: int) -> list:
    return [
        f"D3 {b} -> {rid}: slack {(exp[rid] - exp[b]).days} d < {min_days} d"
        for b in blockers_of(r)
        if b in exp and (exp[rid] - exp[b]).days < min_days
    ]


def d3_slack(rows: dict, exp: dict, min_days: int) -> list:
    gated = [(rid, r) for rid, r in rows.items() if str(r.get("lane")) == "0.66" and rid in exp]
    return [m for rid, r in gated for m in _slack_of(rid, r, exp, min_days)]


def _queue_lines(host: str, q, rows: dict, exp: dict) -> list:
    if not isinstance(q, list):
        return [f"D4 host_queues.{host} is not a list"]
    out = [f"D4 {host}: `{x}` is not a row" for x in q if x not in rows]
    for a, b in zip(q, q[1:]):
        if a in exp and b in exp and exp[a] > exp[b]:
            out.append(f"D4 {host}: {a} ({exp[a]}) is queued before {b} ({exp[b]})")
    return out


def d4_queues(doc: dict, rows: dict, exp: dict) -> list:
    return [m for host, q in (doc.get("host_queues") or {}).items() for m in _queue_lines(host, q, rows, exp)]


def d5_owner(rows: dict) -> list:
    return [f"D5 {rid}: no owner" for rid, r in rows.items() if not str(r.get("owner") or "").strip()]


def past_expiry(rows: dict, exp: dict, today: date) -> list:
    return [f"{rid} expired {exp[rid]}" for rid in rows if rid in exp and exp[rid] < today and rows[rid].get("status") != "complete"]


def run_check(doc: dict, min_days: int, today: date) -> "tuple[int, list]":
    rows = rows_by_id(doc)
    violations = d1_dangling(rows) + d2_cycles(rows) + d6_expiry_forms(rows) + d5_owner(rows)
    exp, more = resolved_expiries(rows)
    violations += more + d3_slack(rows, exp, min_days) + d4_queues(doc, rows, exp)
    lines = list(violations)
    expired = past_expiry(rows, exp, today)
    if expired:
        lines.append("REPORT past-expiry (not a violation here; the §4 andon owns it): " + "; ".join(expired))
    lines.append(
        f"dag-invariants: rows={len(rows)} violations={len(violations)} "
        f"min_slack_days={min_days} queues={len(doc.get('host_queues') or {})} past_expiry={len(expired)}"
    )
    return (1 if violations else 0), lines


def _today(arg, doc: dict) -> date:
    if arg:
        return date.fromisoformat(arg)
    g = doc.get("generated")
    if isinstance(g, (str, date)):
        return _as_date(str(g)) if isinstance(g, str) else g
    return date.today()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dag_invariants.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("dag")
    c.add_argument("--min-slack-days", type=int, default=6)
    c.add_argument("--today", default=None, help="YYYY-MM-DD; default = the DAG header's `generated` date if present, else today")
    args = ap.parse_args(argv)
    try:
        doc = load_dag(args.dag)
        rc, lines = run_check(doc, args.min_slack_days, _today(args.today, doc))
    except UsageError as e:
        print(f"dag_invariants: {e}", file=sys.stderr)
        return 2
    print("\n".join(lines))
    return rc


if __name__ == "__main__":
    sys.exit(main())
