#!/usr/bin/env python3
"""Render a look-ahead Definition-of-Ready ledger (schema lookahead-dor-v1).

APR-LOOKAHEAD-001 §4 L1 item 2. A row is READY only when spec, contract,
falsifier, baseline and a proposed owner are all present and every path that
claims `on_main: true` exists at the checked git ref. Claims are verified,
never trusted. A `staged_at: origin/<branch>` claim must exist at that branch;
when its path already exists at the ref, existing proves nothing, so the claim
names a `staged_marker` that is in the branch's file and not in the ref's.

Exit codes: 0 rendered, 1 --require-ready and a row is NOT READY,
2 RED (empty ledger, schema error, or a false on_main claim).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

SCHEMA = "lookahead-dor-v1"
MAX_ROWS = 10
HERE = Path(__file__).resolve().parent


class Red(Exception):
    pass


def exists_at(ref, path):
    r = subprocess.run(["git", "cat-file", "-e", f"{ref}:{path}"],
                       capture_output=True, check=False)
    return r.returncode == 0


def text_at(ref, path):
    r = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True,
                       encoding="utf-8", errors="replace", check=False)
    return r.stdout if r.returncode == 0 else None


def claims(row):
    for field in ("spec", "contract", "baseline"):
        v = row.get(field)
        if isinstance(v, dict) and v.get("on_main"):
            yield field, v.get("path")


def staged(v):
    return isinstance(v, dict) and str(v.get("staged_at") or "").startswith("origin/")


def staged_claims(row):
    for field in ("spec", "contract", "baseline"):
        v = row.get(field)
        if staged(v):
            yield (field, v["staged_at"], v.get("staged_path") or v.get("path"),
                   v.get("staged_marker"))


def marker_fault(marker, at_branch, at_ref):
    """Why a staged claim proves nothing, or None."""
    if not marker:
        return "the path already exists at the ref; name the text the branch adds as staged_marker"
    if marker not in at_branch:
        return f"staged_marker {marker!r} is absent at the branch"
    if marker in at_ref:
        return f"staged_marker {marker!r} is already at the ref, so the content is on main"
    return None


def on_main(v):
    return isinstance(v, dict) and bool(v.get("path")) and bool(v.get("on_main"))


def ids_unique(rows):
    ids = [r.get("id") for r in rows]
    return None not in ids and len(set(ids)) == len(ids)


def ranks_dense(rows):
    return sorted(r.get("ev_rank") or 0 for r in rows) == list(range(1, len(rows) + 1))


ROW_RULES = (
    (bool, "empty ledger (R-2: empty is RED)"),
    (lambda rows: len(rows) <= MAX_ROWS, f"more than {MAX_ROWS} rows"),
    (ids_unique, "row ids missing or duplicated"),
    (ranks_dense, "ev_rank must be a permutation of 1..len(rows)"),
)


def validate(doc):
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise Red(f"schema is not {SCHEMA}")
    rows = doc.get("rows") or []
    for ok, message in ROW_RULES:
        if not ok(rows):
            raise Red(message)
    return rows


def staged_fault(ref, branch, path, marker):
    """Why a staged claim does not hold, or None."""
    if not path or not exists_at(branch, path):
        return f"but {path!r} is absent there"
    at_ref = text_at(ref, path)
    if at_ref is None and not marker:
        return None  # a new file: existing at the branch is the claim
    fault = marker_fault(marker, text_at(branch, path) or "", at_ref or "")
    return fault and f"in {path!r}, but {fault}"


def verify_claims(row, ref):
    for field, path in claims(row):
        if not path or not exists_at(ref, path):
            raise Red(f"{row['id']}: {field} claims on_main but {path!r} is absent at {ref}")
    for field, branch, path, marker in staged_claims(row):
        fault = staged_fault(ref, branch, path, marker)
        if fault:
            raise Red(f"{row['id']}: {field} claims staged at {branch} {fault}")


def missing_fields(row):
    checks = (
        ("spec", on_main(row.get("spec"))),
        ("contract", on_main(row.get("contract"))),
        ("falsifier", bool(row.get("falsifier"))),
        ("baseline", on_main(row.get("baseline"))),
        ("owner", bool(row.get("owner_proposed"))),
    )
    # A missing field that is staged on a branch renders as "name*": still NOT READY.
    return [name + ("*" if staged(row.get(name)) else "") for name, ok in checks if not ok]


def evaluate(doc, ref):
    out = []
    for r in sorted(validate(doc), key=lambda r: r["ev_rank"]):
        verify_claims(r, ref)
        out.append((r["ev_rank"], r["id"], r.get("exit_criterion"), missing_fields(r)))
    return out


def render(rows, train):
    ready = sum(1 for *_, m in rows if not m)
    for rank, rid, crit, missing in rows:
        state = "READY" if not missing else "NOT READY (" + ",".join(missing) + ")"
        print(f"{rank:>2} {rid:<20} {str(crit):<12} {state}")
    print(f"train {train}: top10_ready: {ready}/{len(rows)}  (* = staged on a branch, verified)")
    return ready


def run(path, ref, require_ready=False):
    doc = yaml.safe_load(Path(path).read_text()) if Path(path).stat().st_size else None
    rows = evaluate(doc, ref)
    ready = render(rows, (doc or {}).get("train"))
    return 1 if require_ready and ready < len(rows) else 0


# Each planted ledger must be RED for its own reason; RED for another reason
# means the rule it plants was never reached.
RED_FIXTURES = {
    "empty.yaml": "empty ledger",
    "false-claim.yaml": "claims on_main",
    "false-staged.yaml": "is absent there",
    "vacuous-staged.yaml": "already exists at the ref",
    "absent-marker.yaml": "is absent at the branch",
    "marker-on-ref.yaml": "is already at the ref",
}


def red_fault(fx, name, reason, ref):
    try:
        evaluate(yaml.safe_load((fx / name).read_text()), ref)
        return f"{name}: expected RED, got a render"
    except Red as e:
        return None if reason in str(e) else f"{name}: RED for the wrong reason: {e}"


def expect_red(fx, ref):
    failures = [f for name, why in RED_FIXTURES.items() if (f := red_fault(fx, name, why, ref))]
    # Control: a RED for another reason is caught (an empty ledger is not a marker fault).
    if not red_fault(fx, "empty.yaml", RED_FIXTURES["marker-on-ref.yaml"], ref):
        failures.append("reason check is vacuous: empty.yaml passed as a marker fault")
    return failures


def expect_marker_control():
    # Control: text the branch adds and the ref lacks is accepted.
    fault = marker_fault("v3.2", "PP-LLAMA-001 v3.2", "PP-LLAMA-001 v3.1")
    return [f"marker control rejected: {fault}"] if fault else []


def expect_planted(fx, ref):
    rows = {rid: m for _, rid, _, m in evaluate(yaml.safe_load((fx / "planted.yaml").read_text()), ref)}
    want = {"NO-CONTRACT": ["contract"], "ALL-PRESENT": []}
    return [f"planted {rid} rendered {rows.get(rid)!r}, want {w!r}"
            for rid, w in want.items() if rows.get(rid) != w]


def self_test(ref):
    """Planted falsifiers: each must fire, or the checker is lying."""
    fx = HERE / "fixtures"
    failures = expect_red(fx, ref) + expect_planted(fx, ref) + expect_marker_control()
    for f in failures:
        print("FALSIFIER SURVIVED:", f)
    fired = len(RED_FIXTURES) + 1  # + planted NO-CONTRACT
    print("self-test:", "RED" if failures else f"ok ({fired} falsifiers fire, 3 controls pass)")
    return 2 if failures else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ledger", nargs="?", default="docs/lookahead/0.71-dor.yaml")
    ap.add_argument("--ref", default="HEAD", help="git ref the on_main claims are checked against")
    ap.add_argument("--require-ready", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    try:
        return self_test(a.ref) if a.self_test else run(a.ledger, a.ref, a.require_ready)
    except Red as e:
        print("RED:", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
