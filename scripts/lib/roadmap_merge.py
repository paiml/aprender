#!/usr/bin/env python3
"""roadmap_merge.py — a git merge driver for docs/roadmaps/roadmap.yaml that
merges by ENTRY ID, not by line (PMAT-3118; spec §6.4).

WHY THIS EXISTS
---------------
A squash-merge from the queue makes `pmat work` re-serialise the whole roadmap,
so every stacked branch goes DIRTY on that one file even when the two sides
touched unrelated tickets. Line-based 3-way merge cannot reconcile that; an
id-keyed one can: an entry is a BYTE-EXACT block starting at a column-0
`- id: <ID>` line (the split `roadmap_diff.split_entries` already implements),
and an entry only conflicts with ITSELF.

    ours-only change  -> take ours        theirs-only change -> take theirs
    identical change  -> take either      both changed differently -> CONFLICT
    one deleted, other untouched -> delete   deleted vs edited -> CONFLICT

ORDER IS THE CHECKER'S ORDER, not an anchor guess. `scripts/check_roadmap_sorted.sh`
requires: within each id-prefix, numerically ascending; prefixes keep their
first-appearance order; legacy free-form ids are exempt and stay where they sit.
An anchor-based placement (put a new id after its nearest preceding base id) is
NOT enough — a side that tail-appended its entry out of order then yields an
unsorted merge, which is why `sort_entries` permutes each prefix's entries
within the slots that prefix already occupies.

    git -c merge.roadmap.driver="python3 scripts/lib/roadmap_merge.py %O %A %B" merge …
    python3 scripts/lib/roadmap_merge.py BASE OURS THEIRS [--out FILE]
    python3 scripts/lib/roadmap_merge.py --selftest    # case table, one line per row

Exit: 0 merged · 1 conflict (or a failing selftest row) · 2 usage.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from scripts.lib.roadmap_diff import split_entries  # noqa: E402

CHECKER = os.path.join(REPO_ROOT, "scripts", "check_roadmap_sorted.sh")

# The SAME shape check_roadmap_sorted.sh's rule 1 recognises, so this driver and
# that guard agree on which ids are sortable and which are legacy/exempt.
ID_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)-([0-9]+)([^0-9A-Za-z_].*)?$", re.S)


def parse_id(eid):
    """-> (prefix, numeral) for a PREFIX-NUMBER id, else None (legacy id)."""
    m = ID_RE.match(eid)
    if not m:
        return None
    numeral = m.group(2)
    if os.environ.get("ROADMAP_MERGE_MUTATE") == "1":
        # Mutation hook: compare the numeral as a STRING, which puts PMAT-10
        # before PMAT-2. The `mut` selftest row requires the checker to catch it.
        return (m.group(1), numeral)
    return (m.group(1), int(numeral))


def _conflict(msg):
    sys.stderr.write("conflict: %s\n" % msg)
    sys.exit(1)


def resolve_header(base_pre, ours_pre, theirs_pre):
    if ours_pre == base_pre:
        return theirs_pre
    if theirs_pre == base_pre:
        return ours_pre
    if ours_pre == theirs_pre:
        return ours_pre
    return _conflict("header (everything before the first `- id:`)")


def _resolve_both_present(eid, b, o, t):
    if o == b:
        return t
    if t == b:
        return o
    if o == t:
        return o
    return _conflict(eid)


def _resolve_deletion(eid, b, o, t):
    """One side deleted the entry: safe only if the other side left it
    byte-identical to base, otherwise delete-vs-edit and a conflict."""
    survivor = o if o is not None else t
    if survivor is not None and survivor != b:
        return _conflict(eid)
    return None


def _resolve_existing(eid, b, o, t):
    if o is not None and t is not None:
        return _resolve_both_present(eid, b, o, t)
    return _resolve_deletion(eid, b, o, t)


def resolve_entry(eid, b, o, t):
    if b is not None:
        return _resolve_existing(eid, b, o, t)
    if o is not None and t is not None and o != t:
        return _conflict(eid)
    return o if o is not None else t


def enforce_prefix_order(ref):
    """Permute each id-prefix's entries within the slots that prefix already
    occupies in `ref`: ascending numeral inside the prefix, every other
    position (legacy ids, cross-prefix layout, first-appearance order) left
    exactly as given."""
    slots = {}
    for i, eid in enumerate(ref):
        key = parse_id(eid)
        if key is None:
            continue
        slots.setdefault(key[0], []).append(i)
    out = list(ref)
    for positions in slots.values():
        ordered = sorted((ref[i] for i in positions), key=lambda e: parse_id(e)[1])
        for pos, eid in zip(positions, ordered):
            out[pos] = eid
    return out


def sort_entries(final_entries, base_ids, ours_ids, theirs_ids):
    """Reference layout = base order, then ids only ours added, then ids only
    theirs added (each in its own file order); then the checker's order is
    enforced over that layout."""
    ref = []
    seen = set()
    for seq in (base_ids, ours_ids, theirs_ids):
        for eid in seq:
            if eid in final_entries and eid not in seen:
                seen.add(eid)
                ref.append(eid)
    return enforce_prefix_order(ref)


def merge(b_txt, o_txt, t_txt):
    b_pre, b_ent = split_entries(b_txt)
    o_pre, o_ent = split_entries(o_txt)
    t_pre, t_ent = split_entries(t_txt)

    f_pre = resolve_header(b_pre, o_pre, t_pre)

    b_dict = dict(b_ent)
    o_dict = dict(o_ent)
    t_dict = dict(t_ent)

    f_ent = {}
    for eid in set(b_dict) | set(o_dict) | set(t_dict):
        val = resolve_entry(eid, b_dict.get(eid), o_dict.get(eid), t_dict.get(eid))
        if val is not None:
            f_ent[eid] = val

    order = sort_entries(
        f_ent, [k for k, _ in b_ent], [k for k, _ in o_ent], [k for k, _ in t_ent]
    )
    return f_pre + "".join(f_ent[k] for k in order)


# ---------------------------------------------------------------------------
# --selftest: one `PASS  <row>` / `FAIL  <row>` line per row, exit 1 on any FAIL
# ---------------------------------------------------------------------------
H = "roadmap_version: '1.0'\n# a header comment\nroadmap:\n"
E1 = "- id: PMAT-1\n  a: 1\n"
E2 = "- id: PMAT-2\n  a: 2\n"
E3 = "- id: PMAT-3\n  a: 3\n"
E4 = "- id: PMAT-4\n  a: 4\n"
E5 = "- id: PMAT-5\n  a: 5\n"
E10 = "- id: PMAT-10\n  a: 10\n"
E1_EDIT = "- id: PMAT-1\n  a: 11\n  note: edited by ours\n"


def _ids_in_order(text):
    return [eid for eid, _ in split_entries(text)[1]]


def _a_both_appended(out, _err):
    ids = _ids_in_order(out)
    if ids != ["PMAT-1", "PMAT-2", "PMAT-3"]:
        return ["both ids must be present and ascending, got %s" % ids]
    return []


def _a_edit_verbatim(out, _err):
    if E1_EDIT not in out:
        return ["ours' edited block was not taken byte-for-byte: %r" % out]
    return []


def _a_conflict_names_id(_out, err):
    if "PMAT-1" not in err:
        return ["stderr must name the conflicting id, got %r" % err]
    return []


def _a_conflict_names_id2(_out, err):
    if "PMAT-2" not in err:
        return ["stderr must name the conflicting id, got %r" % err]
    return []


def _a_entry_gone(out, _err):
    if "PMAT-2" in out:
        return ["the uncontested deletion did not remove PMAT-2: %r" % out]
    if _ids_in_order(out) != ["PMAT-1"]:
        return ["surviving ids: %s" % _ids_in_order(out)]
    return []


def _a_header_verbatim(out, _err):
    first = out.find("- id:")
    if first == -1:
        return ["no entry in the output at all"]
    if out[:first] != H:
        return ["header changed: %r != %r" % (out[:first], H)]
    return []


def _a_tail_append_sorted(out, _err):
    ids = _ids_in_order(out)
    if ids != ["PMAT-1", "PMAT-3", "PMAT-4", "PMAT-5"]:
        return ["out-of-order tail appends must be re-seated, got %s" % ids]
    return []


# name, base, ours, theirs, expect_conflict, expect_checker_fail, assertion
CASES = [
    ("both-append", H + E1, H + E1 + E2, H + E1 + E3, False, False, _a_both_appended),
    ("edit", H + E1, H + E1_EDIT, H + E1, False, False, _a_edit_verbatim),
    ("conflict", H + E1, H + E1_EDIT, H + E1.replace("a: 1", "a: 12"), True, False,
     _a_conflict_names_id),
    ("del-unc", H + E1 + E2, H + E1, H + E1 + E2, False, False, _a_entry_gone),
    ("del-edit", H + E1 + E2, H + E1, H + E1 + E2.replace("a: 2", "a: 22"), True, False,
     _a_conflict_names_id2),
    ("header-untouched", H + E1, H + E1 + E2, H + E1 + E3, False, False,
     _a_header_verbatim),
    ("tail-append-unsorted-input", H + E1 + E5, H + E1 + E5 + E3, H + E1 + E5 + E4,
     False, False, _a_tail_append_sorted),
    ("mut", H + E1, H + E1 + E10, H + E1 + E2, False, True, None),
]


def _run_driver(td, base, ours, theirs, mutate):
    """-> (returncode, stdout+stderr, merged text or None, out path)"""
    paths = []
    for name, text in (("base", base), ("ours", ours), ("theirs", theirs)):
        p = os.path.join(td, name)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text)
        paths.append(p)
    out_path = os.path.join(td, "roadmap.yaml")
    env = os.environ.copy()
    if mutate:
        env["ROADMAP_MERGE_MUTATE"] = "1"
    else:
        env.pop("ROADMAP_MERGE_MUTATE", None)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__)] + paths + ["--out", out_path],
        env=env, capture_output=True, text=True,
    )
    merged = None
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as fh:
            merged = fh.read()
    return proc.returncode, proc.stdout + proc.stderr, merged, out_path


def _run_checker(td, out_path):
    """-> (rc, output). The checker is resolved against THIS file's repo root,
    never against the caller's cwd — an abspath() of a relative path made a 127
    (`No such file`) satisfy the mutation row vacuously."""
    proc = subprocess.run(
        ["bash", CHECKER, out_path], cwd=td, capture_output=True, text=True
    )
    return proc.returncode, proc.stdout + proc.stderr


def _git_init(td):
    empty = os.path.join(td, ".empty-template")
    os.makedirs(empty, exist_ok=True)
    subprocess.run(["git", "init", "-q", "--template=" + empty, td],
                   capture_output=True, check=False)


def _checker_verdict(td, out_path, exp_chk_fail):
    """Run check_roadmap_sorted.sh over the merged file and judge its verdict.
    The mutation row demands EXACTLY 1 (a violation): a 2 (env) or 127 (never
    ran) satisfies "nonzero" vacuously, which is how the cwd-relative checker
    path passed the mutation row while the guard had not executed at all."""
    _git_init(td)
    rc, out = _run_checker(td, out_path)
    if exp_chk_fail:
        if rc != 1:
            return ["check_roadmap_sorted.sh rc=%s, expected EXACTLY 1 "
                    "(a 2/127 would satisfy 'nonzero' vacuously): %r" % (rc, out)]
        if "FAIL" not in out:
            return ["checker exited 1 without a FAIL line: %r" % out]
        return []
    if rc != 0:
        return ["check_roadmap_sorted.sh rc=%s on the merged file: %r" % (rc, out)]
    return []


def _driver_rc_verdict(rc, exp_err, err):
    if (rc != 0) != exp_err:
        return ["driver rc=%s (expected %s), output: %r"
                % (rc, "nonzero" if exp_err else 0, err)]
    return []


def _check_row(row, td):
    """-> list of failure strings for one case row."""
    _name, base, ours, theirs, exp_err, exp_chk_fail, assertion = row
    rc, err, merged, out_path = _run_driver(td, base, ours, theirs, exp_chk_fail)
    fails = _driver_rc_verdict(rc, exp_err, err)
    if fails:
        return fails
    if exp_err:
        return assertion(merged or "", err) if assertion else []
    if merged is None:
        return ["the driver wrote no output file"]
    if assertion:
        fails += assertion(merged, err)
    return fails + _checker_verdict(td, out_path, exp_chk_fail)


def _report_row(name, fails):
    """One line per row: `PASS  <row>` or `FAIL  <row>` (+ indented detail).
    -> 1 if the row failed, 0 if it passed."""
    if not fails:
        print("PASS  %s" % name)
        return 0
    print("FAIL  %s" % name)
    for line in fails:
        for sub in str(line).splitlines():
            print("        %s" % sub)
    return 1


def run_selftest():
    if not os.path.isfile(CHECKER):
        sys.stderr.write("ENV: %s is missing; the order rows cannot be decided\n" % CHECKER)
        return 2
    red = 0
    for row in CASES:
        with tempfile.TemporaryDirectory() as td:
            fails = _check_row(row, td)
        red += _report_row(row[0], fails)
    print("%s/%s rows, %s failed" % (len(CASES) - red, len(CASES), red))
    return 1 if red else 0


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--selftest", "--self-test", action="store_true", dest="selftest")
    p.add_argument("--out")
    p.add_argument("b", nargs="?")
    p.add_argument("o", nargs="?")
    p.add_argument("t", nargs="?")
    args, _ = p.parse_known_args()

    if args.selftest:
        return run_selftest()
    if not all([args.b, args.o, args.t]):
        sys.stderr.write("usage: %s BASE OURS THEIRS [--out FILE] | --selftest\n"
                         % os.path.basename(__file__))
        return 2

    txts = []
    for f in (args.b, args.o, args.t):
        with open(f, "r", encoding="utf-8") as fh:
            txts.append(fh.read())

    out_text = merge(*txts)
    with open(args.out or args.o, "w", encoding="utf-8") as fh:
        fh.write(out_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
