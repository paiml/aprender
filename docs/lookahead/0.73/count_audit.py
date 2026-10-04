#!/usr/bin/env python3
"""Count audit for docs/lookahead/0.73 (#3999).

Each count that these drafts state is derived here from its source, and every sentence that
states it is checked against the derived value. Per-contract totals come from `pv status`.
The falsifier ids and their `test:` text come from the YAML, because pv does not print them.

Sources: the four contracts in contracts-draft/, the FALSIFY-R4-NNN ids in
R4-moe-gpu-wiring.md (R4 has no contract), the Assignment, Bundles and Ranking tables in
falsifier-landing-map.md (ranks 1 to 5, and rows 6 to 20 with the exit criteria they serve), and the fixture table in P1-receipt-checker-spec.md §4.

Usage: python3 docs/lookahead/0.73/count_audit.py [--self-test]   (needs PyYAML, and pv on PATH or in $PV)
Exit 0 when every check passes, 1 when one fails, 2 when a source is missing.
--self-test plants a wrong count in a copy of the drafts for each case, and checks that the audit catches it.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
NYW = "NOT YET WRITTEN"
CONTRACTS = {
    "BPM": "backend-parity-matrix-v1.yaml",
    "NEON": "neon-q4k-q6k-v1.yaml",
    "WGF": "wgpu-forward-v1.yaml",
    "AQ": "cpu-q4k-activation-quant-v1.yaml",  # the side fixes S1/S2, outside the gate set
}
LANDING, SPEC, BODIES = "falsifier-landing-map.md", "P1-receipt-checker-spec.md", "ticket-bodies-P1-P5.md"
WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
VERDICT = {True: "PASS", False: "FAIL"}
ID_RE = re.compile(r"\b(BPM|NEON(?:-Q4K)?|WGF|R4|AQ)-(\d{3})((?:\s*(?:\.\.|,|/|,?\s+and)\s*\d{3}\b)*)")
TAIL_RE = re.compile(r"(\.\.|,|/|and)\s*(\d{3})")
FIX_RE = re.compile(r"\bf\d{3}[a-z]?\b")
E_RE = re.compile(r"\bE([1-6])\b")
UNRANKED = {"M", "F3", "S1", "S2", "S3", "S5"}
failures = []


def die(msg):
    print(f"count_audit: {msg}", file=sys.stderr)
    sys.exit(2)


def ids_in(text):
    """Expand id mentions such as 'BPM-001..006, 008..018' or 'WGF-001, 002, 004 and 008'."""
    out = set()
    for m in ID_RE.finditer(text):
        prefix = "NEON" if m.group(1).startswith("NEON") else m.group(1)
        prev = int(m.group(2))
        out.add(f"{prefix}-{prev:03d}")
        for sep, num in TAIL_RE.findall(m.group(3)):
            n = int(num)
            out.update(f"{prefix}-{k:03d}" for k in (range(prev, n + 1) if sep == ".." else (n,)))
            prev = n
    return out


def shown(derived, stated, ok):
    if not isinstance(derived, set):
        return f"derived {derived}, stated {stated}"
    if ok:
        return f"{len(derived)} ids"
    return f"missing {sorted(derived - stated)}, extra {sorted(stated - derived)}"


def check(name, derived, stated, where):
    ok = derived == stated
    print(f"{VERDICT[ok]} {name}: {shown(derived, stated, ok)} ({where})")
    if not ok:
        failures.append(name)


def text(name):
    path = HERE / name
    if not path.exists():
        die(f"missing {path}")
    return path.read_text()


def find(name, pattern, flags=0):
    """First match of pattern in file name, with its line number for the report."""
    body = text(name)
    m = re.search(pattern, body, flags)
    if not m:
        die(f"no match for {pattern!r} in {name}")
    return m, f"{name}:{body.count(chr(10), 0, m.start()) + 1}"


def section(name, start, stop=r"^## "):
    """The lines of file name from the heading matching start up to the next stop heading."""
    m, _ = find(name, start, re.M)
    rest = text(name)[m.end():]
    end = re.search(stop, rest, re.M)
    return rest[: end.start()] if end else rest


def table_rows(name, heading):
    """The cells of each body row of the tables in one section; a header row is the line above a |--- line."""
    lines = section(name, heading).splitlines()
    heads = {i - 1 for i, ln in enumerate(lines) if re.match(r"\|\s*-", ln)}
    return [[c.strip() for c in ln.strip().strip("|").split("|")]
            for i, ln in enumerate(lines) if ln.startswith("| ") and i not in heads]


def pv_status(path):
    pv = os.environ.get("PV") or shutil.which("pv")
    if not pv:
        die("pv not found; set $PV")
    run = subprocess.run([pv, "status", str(path)], capture_output=True, text=True, check=False)
    if run.returncode != 0:
        die(f"pv status {path.name} exited {run.returncode}")
    keys = ("Equations", "Proof obligations", "Falsification tests", "Kani harnesses")
    return {k: int(m.group(1)) if (m := re.search(rf"^{k}: (\d+)$", run.stdout, re.M)) else 0 for k in keys}


def audit_contracts():
    """pv totals cross-checked against the YAML, and the unwritten falsifiers of each contract and of R4."""
    unwritten, ntests = {}, {}
    for prefix, fname in CONTRACTS.items():
        doc = yaml.safe_load(text(f"contracts-draft/{fname}"))
        tests = doc.get("falsification_tests") or []
        ntests[prefix] = len(tests)
        pv = pv_status(HERE / "contracts-draft" / fname)
        for what, key, n in (("falsifiers", "Falsification tests", len(tests)),
                             ("obligations", "Proof obligations", len(doc.get("proof_obligations") or [])),
                             ("equations", "Equations", len(doc.get("equations") or {}))):
            check(f"{prefix} {what}, pv vs yaml", n, pv[key], f"pv status {fname}")
        unwritten[prefix] = set().union(*(ids_in(t["id"]) for t in tests if NYW in str(t.get("test", ""))))
    unwritten["R4"] = {i for i in ids_in(text("R4-moe-gpu-wiring.md")) if i.startswith("R4-")}
    m, where = find("contracts-draft/cpu-q4k-activation-quant-v1.yaml", r"All (\d+) falsification tests pass")
    check("AQ qa_gate test count", ntests["AQ"], int(m.group(1)), where)
    return unwritten


def assignment():
    """How often each id is assigned in the landing map, its bundle, and the ids that also need P2."""
    rows, bundle_of, needs_p2 = Counter(), {}, set()
    for cells in table_rows(LANDING, r"^## Assignment$"):
        label = re.sub(r"\s*\(.*?\)", "", cells[1]).strip()
        ids = ids_in(cells[0])
        rows.update(ids)
        needs_p2 |= ids if label.endswith(" + P2") else set()
        bundle_of.update(dict.fromkeys(ids, label.removesuffix(" + P2")))  # P2 adds a trace field; it lands no falsifier
    return rows, bundle_of, needs_p2


def audit_landing_map(unwritten, gate):
    """Every gate falsifier is assigned to exactly one bundle, and the header counts them right."""
    rows, bundle_of, needs_p2 = assignment()
    check("landing map ids assigned once", {i for i, n in rows.items() if n == 1}, set(rows), f"{LANDING} Assignment")
    check("landing map covers the gate set", gate, set(rows), f"{LANDING} Assignment")
    m, where = find(LANDING, r"There are (\d+) falsifiers not yet written: "
                    r"(\d+) in backend-parity-matrix-v1 \(BPM\), (\d+) in neon-q4k-q6k-v1 \(NEON\),\s+"
                    r"(\d+) in wgpu-forward-v1 \(WGF\) and (\d+) in R4-moe-gpu-wiring.md \(R4\)")
    stated = [int(g) for g in m.groups()]
    check("landing map header total", len(gate), stated[0], where)
    for prefix, n in zip(("BPM", "NEON", "WGF", "R4"), stated[1:], strict=True):
        check(f"landing map header {prefix}", len(unwritten[prefix]), n, where)
    per_bundle = {b: {i for i, x in bundle_of.items() if x == b} for b in set(bundle_of.values())}
    return per_bundle, needs_p2


def audit_count_line(gate, per_bundle, needs_p2):
    m, where = find(LANDING, r"\*\*Main finding:\*\* (\d+) of the (\d+)")
    check("main finding P1", len(per_bundle["P1"]), int(m.group(1)), where)
    check("main finding total", len(gate), int(m.group(2)), where)
    m, where = find(LANDING, r"Count \((\d+)\): (\d+) in P1 \(([^)]*)\)")
    check("Count line total", len(gate), int(m.group(1)), where)
    check("Count line P1", len(per_bundle["P1"]), int(m.group(2)), where)
    check("Count line P1 ids", per_bundle["P1"], ids_in(m.group(3)), where)
    m, where = find(LANDING, r"\. ([^.]*?) also need P2")
    check("Count line P1 ids that also need P2", needs_p2 & per_bundle["P1"], ids_in(m.group(1)), where)
    for label in ("P3", "P4"):
        m, where = find(LANDING, rf"(\d+) in {label}\b")
        check(f"Count line {label}", len(per_bundle[label]), int(m.group(1)), where)
    for pattern, key in (("M alone", "M"), (r"P5 \+ M", "P5 + M")):
        m, where = find(LANDING, rf"(\d+) in {pattern} \(([^)]*)\)")
        check(f"Count line {key}", len(per_bundle[key]), int(m.group(1)), where)
        check(f"Count line {key} ids", per_bundle[key], ids_in(m.group(2)), where)


def landed(per_bundle, label):
    """The gate falsifiers a ranked bundle lands: its own, and those it lands together with M runs."""
    return set().union(*(ids for b, ids in per_bundle.items() if b == label or b.startswith(f"{label} + ")))


def research_rows(cell):
    return set(re.findall(r"\bR[1-5]\b", cell))


def audit_ranking(per_bundle, needs_p2):
    """The Ranking table ranks each bundle once, keeps the research row the Bundles table names, and states
    the falsifiers each bundle lands."""
    where = f"{LANDING} Ranking"
    named = {c[0].strip("*").split()[0]: research_rows(c[0]) for c in table_rows(LANDING, r"^## Bundles$")}
    rows = table_rows(LANDING, r"^## Ranking$")
    labels = [c[1].split()[0] for c in rows]
    check("ranking ranks", list(range(1, len(rows) + 1)), [int(c[0]) if c[0].isdigit() else -1 for c in rows], where)
    check("ranking bundles", sorted(set(named) - {"M"}), sorted(labels), where)
    check("ranking research rows", {(b, r) for b, rs in named.items() for r in rs},
          {(b, r) for b, c in zip(labels, rows, strict=True) if named.get(b) for r in research_rows(c[2])}, where)
    for label, c in zip(labels, rows, strict=True):
        m = re.match(r"\d+", c[3])
        check(f"ranking {label} gate falsifiers", len(landed(per_bundle, label)), int(m.group()) if m else -1, where)
    p2 = [c[3] for b, c in zip(labels, rows, strict=True) if b == "P2"]
    check("ranking P2 ids that need its fields", needs_p2, ids_in(p2[0]) if p2 else set(), where)



def criteria(cell):
    """The exit criteria E1..E6 that a cell names."""
    return {f"E{n}" for n in E_RE.findall(cell)}


def audit_ranking_tail():
    """Rows 6 to 20 continue the Ranking: ranks 6..20 in order, no bundle or unranked item again, each row
    serves an exit criterion, and with what ranks 1 to 5 serve they cover E1..E6."""
    where = f"{LANDING} Ranking, rows 6 to 20"
    rows = table_rows(LANDING, r"^## Ranking, rows 6 to 20$")
    labels = {c[1].split()[0] for c in table_rows(LANDING, r"^## Ranking$")}
    check("ranking rows 6 to 20 ranks", list(range(6, 21)), [int(c[0]) if c[0].isdigit() else -1 for c in rows], where)
    check("ranking rows 6 to 20 repeat nothing", [], [c[0] for c in rows if c[1].split()[0] in labels | UNRANKED], where)
    check("ranking rows 6 to 20 serve a criterion", [], [c[0] for c in rows if not criteria(c[2])], where)
    m, at = find(LANDING, r"^Ranks 1 to 5 serve: (.+)$", re.M)
    served = dict(re.findall(r"\b(P\d) ((?:E\d(?:, )?)+)", m.group(1)))
    check("ranks 1 to 5 serve line", sorted(labels), sorted(served), at)
    covered = set().union(*map(criteria, served.values()), *(criteria(c[2]) for c in rows))
    check("ranking covers E1 to E6", {f"E{n}" for n in range(1, 7)}, covered, where)


def audit_side_fixes(unwritten, gate):
    m, where = find(LANDING, r"(\w+) falsifiers sit outside the (\d+)")
    check("side fixes outside the gate set", len(unwritten["AQ"]), WORDS.get(m.group(1), -1), where)
    check("side fixes total they sit outside", len(gate), int(m.group(2)), where)
    side = {i for i in ids_in(section(LANDING, r"^## Side fixes")) if i.startswith("AQ-")}
    check("side-fix ids in the landing map", unwritten["AQ"], side, f"{LANDING} Side fixes")


def p1_fixtures():
    """The rows of the spec §4 fixture table, and the planted files they imply.

    The files are base.json, one per row, the extra names of a row such as f017a/b/c, each
    "a second fixture" a row adds, and the fixtures that only §5 (mutations) names, such as f002b.
    """
    table = [c for c in table_rows(SPEC, r"^## 4\. ") if re.match(r"f\w", c[0])]
    names = {c[0] for c in table}
    multi = {re.match(r"f\d{3}", n).group(0): len(n.split("/")) for n in names if "/" in n}
    second = {c[0] for c in table if "a second fixture" in "|".join(c)}
    only5 = {n for n in FIX_RE.findall(section(SPEC, r"^## 5\. ")) if not any(n in x for x in names)}
    base = int("`base.json`" in section(SPEC, r"^## 4\. "))
    files = base + len(table) + sum(k - 1 for k in multi.values()) + len(second) + len(only5)
    return table, multi, second, only5, files


def audit_p1(per_bundle):
    m, where = find(SPEC, r"It holds (\d+) falsifiers")
    check("P1 spec falsifier count", len(per_bundle["P1"]), int(m.group(1)), where)
    table, multi, second, only5, files = p1_fixtures()
    covered = set().union(*(ids_in(c[-1]) for c in table))
    check("P1 fixtures cover the P1 bundle", per_bundle["P1"], covered & per_bundle["P1"], f"{SPEC} §4")
    m, where = find(BODIES, r"\*\*Planted receipts:\*\* (\d+) files .*?the (\d+) rows of spec §4 "
                    r"with the second fixtures of ([^)]*?) and the (\w+) of (f\d{3}), and ([^)]*?)\)")
    check("P1 body planted files", files, int(m.group(1)), where)
    check("P1 body spec §4 rows", len(table), int(m.group(2)), where)
    check("P1 body second-fixture rows", second, set(FIX_RE.findall(m.group(3))), where)
    check("P1 body multi-fixture rows", multi, {m.group(5): WORDS.get(m.group(4).capitalize(), -1)}, where)
    check("P1 body §5-only fixtures", only5, set(FIX_RE.findall(m.group(6))), where)
    m, where = find(BODIES, r"They cover ([^\n]*?) and the uncaptured-stderr case")
    check("P1 body coverage ids", per_bundle["P1"], ids_in(m.group(1)), where)
    m, where = find(BODIES, r"the (\d+) checker mutations in spec §5")
    check("P1 body mutations", len(table_rows(SPEC, r"^## 5\. ")), int(m.group(1)), where)
    return files


def audit_bodies(per_bundle):
    """The P3..P5 ticket bodies name every falsifier of their bundle."""
    for head, key in (("P3", "P3"), ("P4", "P4"), ("P5", "P5 + M")):
        named = ids_in(section(BODIES, rf"^## {head} — "))
        check(f"{head} body names its bundle", per_bundle[key], named & per_bundle[key], f"{BODIES} {head}")


# --self-test plants one wrong count per case in a copy of the drafts. The audit must exit 1 and name
# the check the case breaks. The unchanged copy must exit 0, and a deleted source must exit 2.
MUTATIONS = [
    (None, None, None, 0),
    ("falsifier-landing-map.md", "19 in backend-parity-matrix-v1", "18 in backend-parity-matrix-v1", "landing map header BPM"),
    ("falsifier-landing-map.md", "WGF-005, WGF-009, R4-003)", "WGF-005, R4-003)", "Count line P1 ids"),
    ("falsifier-landing-map.md", "| NEON-000 cross-check | P3 |", "| NEON-000 cross-check | P4 |", "Count line P3"),
    ("falsifier-landing-map.md", "Three falsifiers sit outside", "Two falsifiers sit outside", "side fixes outside the gate set"),
    ("contracts-draft/wgpu-forward-v1.yaml", "'NOT YET WRITTEN — lands with R2 item 1'", "'tests/wgf.rs'",
     "landing map header WGF"),
    ("R4-moe-gpu-wiring.md", "Falsifier FALSIFY-R4-003", "Falsifier FALSIFY-R4-004", "landing map covers the gate set"),
    ("P1-receipt-checker-spec.md", "It holds 22 falsifiers", "It holds 21 falsifiers", "P1 spec falsifier count"),
    ("P1-receipt-checker-spec.md", "| fW09 | op_placement without the attention key", "fW09 was here", "P1 body spec §4 rows"),
    ("ticket-bodies-P1-P5.md", "**Planted receipts:** 47 files", "**Planted receipts:** 46 files", "P1 body planted files"),
    ("P1-receipt-checker-spec.md", "; and a second fixture with only 7 decode positions", "", "P1 body second-fixture rows"),
    ("falsifier-landing-map.md", "**Main finding:** 22 of the 42", "**Main finding:** 21 of the 42", "main finding P1"),
    ("falsifier-landing-map.md", "| 4 | P3 NEON kernels |", "| 3 | P3 NEON kernels |", "ranking ranks"),
    ("falsifier-landing-map.md", "| 5 | P5 MoE dispatch |", "| 5 | P6 MoE dispatch |", "ranking bundles"),
    ("falsifier-landing-map.md", "| P4 wgpu fixes | R2 |", "| P4 wgpu fixes | R3 |", "ranking research rows"),
    ("falsifier-landing-map.md", "| R3 | 8 |", "| R3 | 9 |", "ranking P3 gate falsifiers"),
    ("falsifier-landing-map.md", "WGF-009 and WGF-004 need", "WGF-009 need", "ranking P2 ids that need its fields"),
    ("falsifier-landing-map.md", "| 7 | OBS-18", "| 8 | OBS-18", "ranking rows 6 to 20 ranks"),
    ("falsifier-landing-map.md", "| 20 | Batched MoE prefill", "| 20 | P5 batched MoE prefill", "ranking rows 6 to 20 repeat nothing"),
    ("falsifier-landing-map.md", "| 17 | Sampling on the wgpu decoder (#3760) | E4 |", "| 17 | Sampling on the wgpu decoder (#3760) | none |",
     "ranking rows 6 to 20 serve a criterion"),
    ("falsifier-landing-map.md", "P5 E3, E4.", "P6 E3, E4.", "ranks 1 to 5 serve line"),
    ("falsifier-landing-map.md", "| E5 |", "| E4 |", "ranking covers E1 to E6"),
    ("ticket-bodies-P1-P5.md", "the 29 checker mutations", "the 28 checker mutations", "P1 body mutations"),
    ("R4-moe-gpu-wiring.md", None, None, 2),
]


def plant(path, old, new):
    """Replace the one occurrence of old in path by new, or delete path when old is None."""
    if old is None:
        path.unlink()
        return
    body = path.read_text()
    if body.count(old) != 1:
        die(f"self-test: {old!r} is not unique in {path.name}")
    path.write_text(body.replace(old, new))


def audit_copy(fname, old, new):
    """Run the audit on a copy of the drafts with one case planted; no fname means the unchanged copy."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / HERE.name
        shutil.copytree(HERE, root)
        if fname:
            plant(root / fname, old, new)
        return subprocess.run([sys.executable, str(root / Path(__file__).name)], capture_output=True, text=True, check=False)


def caught(run, expect):
    """An int is the exit code wanted. A name is the check that must FAIL, with exit 1."""
    if isinstance(expect, int):
        return run.returncode == expect
    return run.returncode == 1 and f"FAIL {expect}:" in run.stdout


def describe(fname, old):
    if not fname:
        return "unchanged copy"
    return f"{fname}: {'deleted' if old is None else repr(old)}"


def self_test():
    hits = 0
    for fname, old, new, expect in MUTATIONS:
        run = audit_copy(fname, old, new)
        ok = caught(run, expect)
        hits += ok
        print(f"{VERDICT[ok]} self-test {describe(fname, old)} -> exit {run.returncode}, want {expect}")
    print(f"{VERDICT[hits == len(MUTATIONS)]}: self-test caught {hits} of {len(MUTATIONS)} cases")
    return 0 if hits == len(MUTATIONS) else 1


def main():
    if sys.argv[1:] == ["--self-test"]:
        return self_test()
    unwritten = audit_contracts()
    gate = unwritten["BPM"] | unwritten["NEON"] | unwritten["WGF"] | unwritten["R4"]
    per_bundle, needs_p2 = audit_landing_map(unwritten, gate)
    audit_count_line(gate, per_bundle, needs_p2)
    audit_ranking(per_bundle, needs_p2)
    audit_ranking_tail()
    audit_side_fixes(unwritten, gate)
    files = audit_p1(per_bundle)
    audit_bodies(per_bundle)
    bundles = ", ".join(f"{k} {len(v)}" for k, v in sorted(per_bundle.items()))
    print(f"{VERDICT[not failures]}: {len(failures)} failed. Derived: gate {len(gate)}; {bundles}; P1 planted files {files}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
