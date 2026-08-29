#!/usr/bin/env python3
"""Re-derive the G2.3 coverage baselines from the ledger.

This is the command cited by every value in the `baselines:` block of
contracts/apr-dogfood-coverage-v1.yaml. It exists so that no number in that
contract has to be trusted: run this, diff against the contract, and any drift
is visible.

    python3 scripts/dogfood_baseline.py              # print the YAML block
    python3 scripts/dogfood_baseline.py --check      # compare to the contract

A baseline nobody can re-derive is a number someone eventually "corrects" to a
rounder one. 17.1% is 142/830; it is not "about 17%".
"""
import csv
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
# The T2 pairing rule is imported, not re-implemented. This printer is one of the
# channels a cluster-level ratio can leave the repository through -- the gate
# report, the receipt, this, and the skill -- and a pairing rule enforced at one
# call site is a pairing rule with three ways around it.
# untriaged_broken() and parse_triage() come from the same place for the same
# reason: the UNTRIAGED count is the quantity G2.3 actually ratchets (#2757), and
# a baseline printer that re-implemented "untriaged" could drift from the gate
# that enforces it -- leaving a contract number nobody re-derives, which is the
# one thing this script exists to prevent.
from dogfood_coverage_gate import (  # noqa: E402
    enforce_pairing, parse_triage, untriaged_broken)

CSV_PATH = "docs/audits/surface_audit.csv"
THE44_PATH = "docs/audits/dogfood-the-44.yaml"
CONTRACT_PATH = "contracts/apr-dogfood-coverage-v1.yaml"

BANDS = (("q1_2", 1, 2), ("q3_4", 3, 4), ("q5_6", 5, 6),
         ("q7_8", 7, 8), ("q9_10", 9, 10))


def load(path=CSV_PATH):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def covered(row):
    return (row["in_dogfood_skill"] or "").strip().lower() == "yes"


def quality(row):
    return int(row["quality_1_10"])


def group_by_binary(rows):
    by_binary = collections.OrderedDict()
    for r in rows:
        by_binary.setdefault(r["binary"], []).append(r)
    return by_binary


def count_covered(rows):
    return sum(1 for r in rows if covered(r))


def per_band(rows):
    bands = collections.OrderedDict()
    for name, lo, hi in BANDS:
        sel = [r for r in rows if lo <= quality(r) <= hi]
        bands[name] = (len(sel), count_covered(sel))
    return bands


def per_cluster(rows):
    """Grouped by `cluster_label` -- NEVER by `cluster_id`. k-means ids permute on
    re-run, so a baseline keyed on one would silently re-point at a different set
    of features. The label is human-owned; see scripts/check_no_cluster_id_keys.sh.

    Both numbers are reported for every cluster, always: the FEATURE fraction and
    the share of total gate effort. Cluster coverage alone is a proxy -- one gate
    in a 95-member cluster is 1%, not "covered" -- and reporting the proxy without
    the feature fraction beneath it is how a coverage gate becomes theatre while
    looking stricter than before."""
    groups = collections.OrderedDict()
    for r in rows:
        groups.setdefault(r["cluster_label"].strip(), []).append(r)
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    return collections.OrderedDict(
        (lab, (len(rs), count_covered(rs))) for lab, rs in ordered)


def per_binary(by_binary):
    ordered = sorted(by_binary.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    return collections.OrderedDict(
        (b, (len(rs), count_covered(rs))) for b, rs in ordered)


def is_unknown_hardware(row):
    return row["verified_hardware"].strip() == "UNKNOWN"


def is_low_and_uncovered(row):
    return row["confidence"].strip().lower() == "low" and not covered(row)


def is_broken_and_ungated(row):
    return quality(row) <= 4 and not covered(row)


def compute(rows):
    by_binary = group_by_binary(rows)
    return {
        "rows": len(rows),
        "covered": count_covered(rows),
        "binaries": len(by_binary),
        "binaries_at_zero": sum(1 for rs in by_binary.values()
                                if count_covered(rs) == 0),
        "unknown_hardware": sum(1 for r in rows if is_unknown_hardware(r)),
        "low_and_uncovered": sum(1 for r in rows if is_low_and_uncovered(r)),
        "broken_and_ungated": sum(1 for r in rows if is_broken_and_ungated(r)),
        "untriaged_broken_and_ungated": len(
            untriaged_broken(rows, parse_triage(THE44_PATH))),
        "per_binary": per_binary(by_binary),
        "per_band": per_band(rows),
        "per_cluster": per_cluster(rows),
        "clusters": len(per_cluster(rows)),
        "clusters_at_zero": sum(1 for _, c in per_cluster(rows).values() if c == 0),
    }


def report(m):
    """Build the report as TEXT, hold it to the T2 pairing rule, and only then
    print it. Returns the lines so `main` can fail the run on an unpaired
    cluster ratio -- reading your own output back is the only way a reporting
    rule survives the next person who edits the emitter."""
    lines = report_lines(m)
    findings = []
    ok = enforce_pairing(lines, findings, m["clusters"], m["rows"],
                         where="scripts/dogfood_baseline.py output")
    for ln in lines:
        print(ln)
    for f in findings:
        print(f)
    return ok


def report_lines(m):
    out = []

    def print(*args, **kwargs):  # noqa: A001 -- collect instead of emit
        out.append(" ".join(str(a) for a in args))

    ratio = m["covered"] / m["rows"] if m["rows"] else 0.0
    print("# totals")
    print(f"rows: {m['rows']}")
    print(f"covered: {m['covered']}")
    print(f"ratio: {ratio:.4f}            "
          f"# {m['covered']}/{m['rows']} -- {100 * ratio:.1f}%")
    print(f"binaries: {m['binaries']}")
    print(f"binaries_at_zero_coverage: {m['binaries_at_zero']}")
    print(f"clusters: {m['clusters']}")
    print(f"clusters_at_zero_coverage: {m['clusters_at_zero']}")
    # The paired line. It exists so this printer is not VACUOUSLY compliant: a
    # pairing rule applied to output that states no cluster ratio proves nothing,
    # which is the same failure one level up that T2 is about.
    print("# clusters gated {}/{}, features gated {}/{}".format(
        m["clusters"] - m["clusters_at_zero"], m["clusters"],
        m["covered"], m["rows"]))
    print(f"unknown_hardware: {m['unknown_hardware']}")
    print(f"low_and_uncovered: {m['low_and_uncovered']}")
    print(f"broken_and_ungated: {m['broken_and_ungated']}")
    # The ratcheted quantity, printed beside the total it is a subset of. Naming
    # only the total would leave a reader assuming the total is the floor, which
    # is what it used to be and no longer is.
    print(f"untriaged: {m['untriaged_broken_and_ungated']}")
    print()
    print("# per_binary")
    for b, (total, cov) in m["per_binary"].items():
        print(f"{b}: {{total: {total}, covered: {cov}}}")
    print()
    print("# per_band")
    for name, (total, cov) in m["per_band"].items():
        pct = 100 * cov / total if total else 0.0
        print(f"{name}: {{total: {total}, covered: {cov}}}     # {pct:.1f}%")
    print()
    print("# per_cluster  -- ALLOCATION OF GATE EFFORT, the reason clustering earns")
    print("#                 its place. Nobody chose this allocation; it accreted.")
    tot_gates = m["covered"] or 1
    for lab, (total, cov) in m["per_cluster"].items():
        print(f"{lab}: {{total: {total}, covered: {cov}}}"
              f"     # {100 * cov / total if total else 0.0:.1f}% of the cluster, "
              f"{100 * cov / tot_gates:.1f}% of all gate effort")
    return out


def expected_lines(m):
    """Every literal the contract must carry, as (label, line) pairs."""
    out = []
    for b, (total, cov) in m["per_binary"].items():
        out.append((f"per_binary {b}", f"{b}: {{total: {total}, covered: {cov}}}"))
    for name, (total, cov) in m["per_band"].items():
        out.append((f"per_band {name}", f"{name}: {{total: {total}, covered: {cov}}}"))
    for lab, (total, cov) in m["per_cluster"].items():
        out.append((f"per_cluster {lab}", f"{lab}: {{total: {total}, covered: {cov}}}"))
    for label, key in (("rows", "rows"), ("covered", "covered"),
                       ("unknown", "unknown_hardware"),
                       ("low_and_uncovered", "low_and_uncovered"),
                       ("count", "broken_and_ungated"),
                       ("untriaged", "untriaged_broken_and_ungated"),
                       ("clusters", "clusters"),
                       ("clusters_at_zero_coverage", "clusters_at_zero")):
        out.append((f"totals {label}", f"{label}: {m[key]}"))
    return out


def check(m, path=CONTRACT_PATH):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    bad = [f"{label} -> {line}" for label, line in expected_lines(m)
           if line not in text]
    print()
    if bad:
        print("CONTRACT DRIFT -- the ledger no longer matches the committed baseline:")
        for x in bad:
            print("   ", x)
        return 1
    print("CHECK PASSED: every baseline in the contract matches the ledger.")
    return 0


def main():
    m = compute(load())
    rc = 0 if report(m) else 1
    if "--check" in sys.argv:
        rc = check(m) or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
