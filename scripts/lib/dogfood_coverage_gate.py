#!/usr/bin/env python3
"""Arithmetic half of the apr-dogfood blocking coverage gate (G2.2/G2.3/G2.4).

This module is deliberately git-free. It is handed two ledgers -- the COMPARAND,
which check_dogfood_coverage.sh extracts from protected `origin/main`, and the
CURRENT working-tree one -- and it reports where the second is worse than the
first. All git plumbing lives in the shell wrapper.

WHY THE COMPARAND IS NOT A LITERAL IN THIS REPO
-----------------------------------------------
The multi-platform dogfood gate had a floor and a universe that were both
literals in the same file, so ONE commit editing both defeated it. A gate whose
floor a PR can rewrite in the commit that breaks the floor is not a gate.

So there is no `142` and no `830` anywhere in this file. Every floor is DERIVED
from the comparand ledger at run time. To lower a floor you must land a commit on
`main`, where the required checks have already run.

WHAT IS CHECKED
---------------
  G2.2 reconciliation  no feature present in the comparand may vanish
  G2.3 floors          covered / rows / per-binary covered may not fall;
                       broken-and-ungated, UNKNOWN-hardware and
                       low-confidence-and-ungated may not rise
  G2.4 waivers         every quality<=4 ungated feature has a triage entry, and
                       any feature NEWLY in that state carries an issue or a
                       written waiver. DOGFOOD_RELEASE=1 demands it of all.

Usage:
    python3 scripts/lib/dogfood_coverage_gate.py \
        --base <comparand.csv> --head <current.csv> --the44 <triage.yaml>
"""
import argparse
import collections
import csv
import os
import re
import sys

COLUMNS = ["binary", "feature", "quality_1_10", "verified_hardware",
           "top_competitor", "in_dogfood_skill", "evidence_path", "confidence"]

# A ledger this small is not a ledger; it is a truncation. Refusing to compare
# against an empty or near-empty comparand is what stops "the file went missing"
# from reading as "every floor is satisfied".
MIN_ROWS = 100


class Fail(Exception):
    """A gate condition that must stop the run."""


def read_ledger(path, label):
    if not os.path.exists(path):
        raise Fail(f"{label} ledger not found: {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        if rdr.fieldnames != COLUMNS:
            raise Fail(f"{label} ledger has unexpected columns: {rdr.fieldnames}")
        rows = list(rdr)
    if len(rows) < MIN_ROWS:
        raise Fail(f"{label} ledger has only {len(rows)} rows (< {MIN_ROWS}); "
                   "refusing to treat a truncated ledger as a satisfied floor")
    return rows


def covered(row):
    return (row["in_dogfood_skill"] or "").strip().lower() == "yes"


def quality(row):
    return int(row["quality_1_10"])


def key(row):
    """Identity of a ledger row. binary+feature, because `POST /v1/models` is a
    different row for `apr` and for `aprender-orchestrate`."""
    return (row["binary"], row["feature"])


def broken_and_ungated(rows):
    return {key(r) for r in rows if quality(r) <= 4 and not covered(r)}


def unknown_hardware(rows):
    return sum(1 for r in rows if r["verified_hardware"].strip().upper() == "UNKNOWN")


def low_and_ungated(rows):
    return sum(1 for r in rows
               if r["confidence"].strip().lower() == "low" and not covered(r))


def covered_by_binary(rows):
    out = collections.Counter()
    for r in rows:
        out[r["binary"]] += 1 if covered(r) else 0
    return out


# --------------------------------------------------------------------------
# G2.2 -- reconciliation. A row that disappears takes its defect with it.
# --------------------------------------------------------------------------
def check_reconciliation(base, head, findings):
    gone = sorted({key(r) for r in base} - {key(r) for r in head})
    if gone:
        findings.append(
            "G2.2 reconciliation FAIL: {} feature(s) present on the comparand are "
            "absent from the working ledger. Deleting a row deletes its defect "
            "along with it; if the surface genuinely shrank, that is a finding to "
            "record, not a row to drop:\n".format(len(gone))
            + "\n".join(f"        - {b}: {f}" for b, f in gone[:20])
            + ("\n        ... and {} more".format(len(gone) - 20) if len(gone) > 20 else ""))
        return False
    print(f"  G2.2 reconciliation  PASS  no row lost ({len(base)} comparand rows all present)")
    return True


# --------------------------------------------------------------------------
# G2.3 -- floors, every one derived from the comparand.
# --------------------------------------------------------------------------
def _ratchet(findings, label, now, floor, direction):
    """direction 'up' = now must be >= floor; 'down' = now must be <= floor."""
    ok = now >= floor if direction == "up" else now <= floor
    arrow = ">=" if direction == "up" else "<="
    if not ok:
        findings.append(f"G2.3 floors FAIL: {label} is {now}, must be {arrow} {floor} "
                        f"(floor derived from the comparand, not from this tree)")
    return ok


def check_floors(base, head, findings):
    ok = True
    ok &= _ratchet(findings, "total rows (the denominator)", len(head), len(base), "up")
    ok &= _ratchet(findings, "covered rows", sum(1 for r in head if covered(r)),
                   sum(1 for r in base if covered(r)), "up")
    ok &= _ratchet(findings, "broken-and-ungated", len(broken_and_ungated(head)),
                   len(broken_and_ungated(base)), "down")
    ok &= _ratchet(findings, "UNKNOWN verified_hardware", unknown_hardware(head),
                   unknown_hardware(base), "down")
    ok &= _ratchet(findings, "low-confidence AND ungated", low_and_ungated(head),
                   low_and_ungated(base), "down")

    # Per-binary, because the aggregate hides a swap: dropping a gate on `apr`
    # while adding one elsewhere leaves the ratio untouched. Per-binary is what
    # keeps "27 of 28 binaries at zero" from being quietly traded away.
    base_cov, head_cov = covered_by_binary(base), covered_by_binary(head)
    for binary in sorted(base_cov):
        ok &= _ratchet(findings, f"covered in `{binary}`",
                       head_cov[binary], base_cov[binary], "up")
    if ok:
        n, d = sum(1 for r in head if covered(r)), len(head)
        print(f"  G2.3 floors          PASS  {n}/{d} covered "
              f"({100.0 * n / d:.1f}%), {len(base_cov)} per-binary floors held")
    return ok


# --------------------------------------------------------------------------
# G2.4 -- the 44. A known-broken ungated feature needs an issue or a waiver.
# --------------------------------------------------------------------------
ENTRY_RE = re.compile(
    r"^- feature: (?P<feature>.*?)$"
    r"(?P<body>(?:\n(?!- feature:).*)*)", re.M)


def parse_triage(path):
    """-> {feature: (has_issue, has_waiver)}. A deliberately small parser: the
    file is generated by scripts/dogfood_the44.py and its shape is fixed."""
    if not os.path.exists(path):
        raise Fail(f"triage file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    out = {}
    for m in ENTRY_RE.finditer(text):
        feature = m.group("feature").strip().strip("'\"")
        body = m.group("body")
        has_issue = bool(re.search(r"^\s+issue:\s*(?!null\s*$)\S", body, re.M))
        has_waiver = bool(re.search(r"^\s+waiver:\s*(?!null\s*$)\S", body, re.M))
        out[feature] = (has_issue, has_waiver)
    return out


def _bullets(features):
    return "\n".join(f"        - {f}" for f in features[:20])


def _check_membership(broken_now, triage, findings):
    """Both directions -- an untracked defect and a stale entry are both ways
    for the triage file to stop describing reality."""
    ok = True
    missing = sorted({f for _, f in broken_now} - set(triage))
    if missing:
        findings.append("G2.4 waivers FAIL: quality<=4 ungated feature(s) with no "
                        "triage entry:\n" + _bullets(missing))
        ok = False
    stale = sorted(set(triage) - {f for _, f in broken_now})
    if stale:
        findings.append("G2.4 waivers FAIL: triage entries for feature(s) no longer "
                        "broken-and-ungated:\n" + _bullets(stale))
        ok = False
    return ok


def _check_new_breakage(broken_now, broken_before, triage, findings):
    """A feature NEWLY quality<=4-and-ungated needs an issue or a waiver at
    once. The pre-existing set is grandfathered by the comparand, which is what
    makes this a ratchet rather than a wall that is red on arrival."""
    newly = sorted({f for _, f in broken_now} - {f for _, f in broken_before})
    untriaged = [f for f in newly if not any(triage.get(f, (False, False)))]
    if untriaged:
        findings.append(
            "G2.4 waivers FAIL: feature(s) newly quality<=4 and ungated, with "
            "neither an issue nor a written waiver:\n" + _bullets(untriaged))
    return not untriaged, newly


def _check_release_arm(triage, findings):
    """Off mid-cycle so the gate is usable; the release checklist sets
    DOGFOOD_RELEASE=1 and then every entry must be triaged."""
    untriaged = sorted(f for f, v in triage.items() if not any(v))
    if untriaged:
        findings.append(
            f"G2.4 waivers FAIL (DOGFOOD_RELEASE=1): {len(untriaged)} of "
            f"{len(triage)} triage entries carry neither an issue nor a waiver. "
            "A waiver is a sentence saying why shipping this ungated is "
            "acceptable; \"low priority\" is not one:\n" + _bullets(untriaged))
    return not untriaged


def check_waivers(base, head, triage_path, findings, release):
    triage = parse_triage(triage_path)
    broken_now = broken_and_ungated(head)
    broken_before = broken_and_ungated(base)

    ok = _check_membership(broken_now, triage, findings)
    new_ok, newly = _check_new_breakage(broken_now, broken_before, triage, findings)
    ok = new_ok and ok
    if release:
        ok = _check_release_arm(triage, findings) and ok
    if ok:
        arm = "release arm ON" if release else "release arm off"
        print(f"  G2.4 waivers         PASS  {len(broken_now)} broken-and-ungated, "
              f"{len(triage)} triaged, {len(newly)} new ({arm})")
    return ok


def run(args):
    base = read_ledger(args.base, "comparand")
    head = read_ledger(args.head, "working-tree")
    release = bool(os.environ.get("DOGFOOD_RELEASE"))
    findings = []
    ok = check_reconciliation(base, head, findings)
    ok = check_floors(base, head, findings) and ok
    ok = check_waivers(base, head, args.the44, findings, release) and ok
    for f in findings:
        print(f"  {f}")
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, help="comparand ledger (from origin/main)")
    p.add_argument("--head", required=True, help="working-tree ledger")
    p.add_argument("--the44", required=True, help="triage yaml")
    args = p.parse_args(argv)
    try:
        return run(args)
    except Fail as exc:
        print(f"  GATE ABORTED: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
