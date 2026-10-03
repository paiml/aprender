#!/usr/bin/env python3
"""Render a look-ahead Definition-of-Ready ledger (schema lookahead-dor-v1).

APR-LOOKAHEAD-001 §4 L1 item 2. A row is READY only when spec, contract,
falsifier, baseline and a proposed owner are all present and every path that
claims `on_main: true` exists at the checked git ref. Claims are verified,
never trusted. A `staged_at: origin/<branch>` claim must exist at that branch;
when its path already exists at the ref, existing proves nothing, so the claim
names a `staged_marker` that is in the branch's file and not in the ref's.

A falsifier is a list of claims, `{ids, path, on_main: true}` or
`{ids, staged_at, staged_path}`, and the named file must define every id: as
an `id:` value in YAML, as a token-bounded occurrence in any other file. A
bare id list is RED, because nothing would check it. `PREFIX-001..012` is a
range of ids.

Exit codes: 0 rendered, 1 --require-ready and a row is NOT READY,
2 RED (empty ledger, schema error, or a false on_main claim).
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

SCHEMA = "lookahead-dor-v1"
MAX_ROWS = 10
HERE = Path(__file__).resolve().parent
RANGE = re.compile(r"^(?P<stem>.*?)(?P<lo>\d+)\.\.(?P<hi>\d+)$")


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


def expand(token):
    """`FALSIFY-X-001..003` is three ids; any other token is one."""
    m = RANGE.match(token)
    if not m:
        return [token]
    lo, hi = int(m["lo"]), int(m["hi"])
    if hi <= lo:
        raise Red(f"id range {token!r} does not ascend")
    return [f"{m['stem']}{n:0{len(m['lo'])}d}" for n in range(lo, hi + 1)]


def yaml_ids(node):
    """Every string `id:` value in a parsed YAML tree, at any depth."""
    if isinstance(node, dict):
        if isinstance(node.get("id"), str):
            yield node["id"]
        node = list(node.values())
    for child in node if isinstance(node, list) else ():
        yield from yaml_ids(child)


def defined_ids(text, path, where):
    """The `id:` values a YAML file defines, at any depth; None for any other file."""
    if not str(path).endswith((".yaml", ".yml")):
        return None
    try:
        return set(yaml_ids(yaml.safe_load(text)))
    except yaml.YAMLError as e:
        raise Red(f"{path!r} at {where} is not valid YAML") from e


def absent_ids(ids, text, path, where):
    """The ids the file does not define. Outside YAML a token-bounded
    occurrence counts, so there a mention satisfies it too."""
    known = defined_ids(text, path, where)
    if known is not None:
        return [i for i in ids if i not in known]
    return [i for i in ids if not re.search(
        r"(?<![A-Za-z0-9_-])" + re.escape(i) + r"(?![A-Za-z0-9_])", text)]


def claim_shaped(c):
    return isinstance(c, dict) and isinstance(c.get("ids"), list) and bool(c["ids"])


def falsifier_claims(row):
    """The row's falsifier claims, ranges expanded; [] when it has none."""
    v = row.get("falsifier") or []
    if not isinstance(v, list) or not all(claim_shaped(c) for c in v):
        raise Red(f"{row['id']}: falsifier must be a list of {{ids, path}} claims; "
                  f"a bare id is trusted, never verified: {v!r}")
    return [dict(c, ids=[i for t in c["ids"] for i in expand(str(t))]) for c in v]


def staged_ids_fault(ids, path, at_branch, at_ref):
    """Why ids staged in a file prove nothing, or None. at_ref is None when the
    ref lacks the file."""
    absent = absent_ids(ids, at_branch, path, "the branch")
    if absent:
        return f"{path!r} there does not define {', '.join(absent)}"
    if at_ref is not None and not absent_ids(ids, at_ref, path, "the ref"):
        return f"{path!r} at the ref already defines every id, so the claim is on main"
    return None


def on_main_ids_fault(claim, ref):
    """Why an on_main falsifier claim does not hold at the ref, or None."""
    path = claim.get("path")
    text = text_at(ref, path) if path else None
    if text is None:
        return f"claims on_main but {path!r} is absent at {ref}"
    absent = absent_ids(claim["ids"], text, path, ref)
    return f"claims on_main but {path!r} at {ref} does not define {', '.join(absent)}" if absent else None


def staged_ids_claim_fault(claim, ref):
    """Why a staged falsifier claim does not hold at its branch, or None."""
    branch, path = claim["staged_at"], claim.get("staged_path") or claim.get("path")
    text = text_at(branch, path) if path else None
    if text is None:
        return f"claims staged at {branch} but {path!r} is absent there"
    fault = staged_ids_fault(claim["ids"], path, text, text_at(ref, path))
    return fault and f"claims staged at {branch} but {fault}"


def falsifier_fault(claim, ref):
    """Why a falsifier claim does not hold, or None."""
    if claim.get("on_main"):
        return on_main_ids_fault(claim, ref)
    if not staged(claim):
        return "claim is neither on_main nor staged_at origin/..."
    return staged_ids_claim_fault(claim, ref)


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


def on_main_faults(row, ref):
    return [f"{field} claims on_main but {path!r} is absent at {ref}"
            for field, path in claims(row) if not path or not exists_at(ref, path)]


def staged_faults(row, ref):
    faults = ((field, branch, staged_fault(ref, branch, path, marker))
              for field, branch, path, marker in staged_claims(row))
    return [f"{field} claims staged at {branch} {why}" for field, branch, why in faults if why]


def falsifier_faults(row, ref):
    faults = (falsifier_fault(c, ref) for c in falsifier_claims(row))
    return [f"falsifier {why}" for why in faults if why]


def verify_claims(row, ref):
    """RED on the first claim the row makes that does not hold."""
    for check in (on_main_faults, staged_faults, falsifier_faults):
        faults = check(row, ref)
        if faults:
            raise Red(f"{row['id']}: {faults[0]}")


def field_staged(row, name):
    v = row.get(name)
    if name == "falsifier":
        return any(staged(c) for c in falsifier_claims(row))
    return staged(v)


def missing_fields(row):
    falsifiers = falsifier_claims(row)
    checks = (
        ("spec", on_main(row.get("spec"))),
        ("contract", on_main(row.get("contract"))),
        # Ready only when every claim is on main: one staged id set is enough to wait on.
        ("falsifier", bool(falsifiers) and all(on_main(c) for c in falsifiers)),
        ("baseline", on_main(row.get("baseline"))),
        ("owner", bool(row.get("owner_proposed"))),
    )
    # A missing field that is staged on a branch renders as "name*": still NOT READY.
    return [name + ("*" if field_staged(row, name) else "") for name, ok in checks if not ok]


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
    "bare-falsifier.yaml": "a bare id is trusted",
    "false-falsifier.yaml": "does not define",
    "absent-falsifier.yaml": "there does not define",
    "falsifier-on-ref.yaml": "already defines every id",
    "unanchored-falsifier.yaml": "neither on_main nor staged",
}

# Planted rows of planted.yaml and the render each must get: a non-empty one is
# a falsifier firing, the empty one is the READY control.
PLANTED = {"NO-CONTRACT": ["contract"], "NO-FALSIFIER": ["falsifier"], "ALL-PRESENT": []}


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


def raises_red(thunk):
    try:
        thunk()
        return False
    except Red:
        return True


def unit_checks():
    """(what, holds, kind) for the rules that need no git: a falsifier is a
    near miss that must be refused, a control is the case that must pass."""
    on = {"path": "p", "on_main": True}
    staged_row = {"id": "STAGED", "spec": on, "contract": on, "baseline": on, "owner_proposed": "L1",
                  "falsifier": [{"ids": ["FALSIFY-X-001"], "staged_at": "origin/x", "staged_path": "c.yaml"}]}
    return (
        ("a marker the branch adds and the ref lacks is accepted",
         marker_fault("v3.2", "PP-LLAMA-001 v3.2", "PP-LLAMA-001 v3.1") is None, "control"),
        ("an exact token outside YAML is defined",
         absent_ids(["F-11"], "| F-11 | x |", "k.md", "t") == [], "control"),
        ("XF-11 and F-110 do not define F-11",
         absent_ids(["F-11"], "| XF-11 | F-110 |", "k.md", "t") == ["F-11"], "falsifier"),
        ("a YAML id under any key is defined",
         absent_ids(["FALSIFY-X-001"], "falsification:\n- id: FALSIFY-X-001\n", "c.yaml", "t") == [], "control"),
        ("a YAML comment or prose mention defines nothing",
         absent_ids(["FALSIFY-X-001"], "tests:\n- id: FALSIFY-X-002\n  rule: was FALSIFY-X-001\n"
                    "# FALSIFY-X-001 retired\n", "c.yaml", "t") == ["FALSIFY-X-001"], "falsifier"),
        ("ids the branch adds and the ref lacks are accepted",
         staged_ids_fault(["FALSIFY-X-002"], "c.yaml", "- id: FALSIFY-X-001\n- id: FALSIFY-X-002\n",
                          "- id: FALSIFY-X-001\n") is None, "control"),
        ("a range expands to every id in it",
         expand("FALSIFY-P-001..003") == ["FALSIFY-P-001", "FALSIFY-P-002", "FALSIFY-P-003"], "control"),
        ("a descending range is RED", raises_red(lambda: expand("FALSIFY-P-003..001")), "falsifier"),
        ("a staged falsifier renders falsifier*, not READY",
         missing_fields(staged_row) == ["falsifier*"], "falsifier"),
    )


def expect_planted(fx, ref):
    rows = {rid: m for _, rid, _, m in evaluate(yaml.safe_load((fx / "planted.yaml").read_text()), ref)}
    return [f"planted {rid} rendered {rows.get(rid)!r}, want {w!r}"
            for rid, w in PLANTED.items() if rows.get(rid) != w]


def tally(units):
    """(falsifiers, controls) the self-test runs. Every RED fixture and every planted
    row that must miss a field fires; the reason check and the READY row are controls."""
    kinds = [kind for *_, kind in units]
    misses = sum(1 for want in PLANTED.values() if want)
    return (len(RED_FIXTURES) + misses + kinds.count("falsifier"),
            1 + len(PLANTED) - misses + kinds.count("control"))


def self_test(ref):
    """Planted falsifiers: each must fire, or the checker is lying."""
    fx = HERE / "fixtures"
    units = unit_checks()
    failures = (expect_red(fx, ref) + expect_planted(fx, ref)
                + [what for what, holds, _ in units if not holds])
    for f in failures:
        print("FALSIFIER SURVIVED:", f)
    fired, controls = tally(units)
    print("self-test:", "RED" if failures else f"ok ({fired} falsifiers fire, {controls} controls pass)")
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
