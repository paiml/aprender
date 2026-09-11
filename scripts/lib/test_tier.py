"""test_tier.py — the 80/20 PR-tier table from a nextest junit and a catch ledger (PMAT-1105, spec §6).

usage: python3 scripts/lib/test_tier.py JUNIT LEDGER_JSON OUT_DIR
  JUNIT       nextest junit.xml (testsuite name '<crate>::<binary>' or classname '<crate>[/<binary>]';
              testcase name = 'module::path::fn'; time = seconds)
  LEDGER_JSON {"crate::module": touches} — how often fix commits touched that module's tests
  OUT_DIR     receives tier.tsv (key, module, crate, tests, seconds, touches, f_tests, tier) and stats.json

A module is 'crate::' + everything before the LAST '::' of the test name (the test's own module path), so the
universe comes from the junit itself — never from a committed table (the lane's version read the touched-module
list as the universe and folded every untouched module into the crate root, which moved the 80 % line).
tier = 'pr' when the module is above the cumulative-80 %-of-touches line (sorted by touches per second) OR the
module owns at least one test whose name contains 'falsif' (a designed catch); otherwise 'nightly'.
"""
import csv, json, sys
import xml.etree.ElementTree as ET
from collections import defaultdict


def crate_of(tc, ts_name):
    cls = tc.get("classname", "") or ""
    if cls:
        return cls.split("/")[0].split("::")[0]
    return (ts_name or "").split("::")[0]


def build_rows(junit_path, touches):
    stats = defaultdict(lambda: {"tests": 0, "seconds": 0.0, "f_tests": 0, "f_seconds": 0.0})
    root = ET.parse(junit_path).getroot()
    for ts in root.iter("testsuite"):
        for tc in ts.iter("testcase"):
            name = tc.get("name", "") or ""
            module = name.rsplit("::", 1)[0] if "::" in name else ""
            key = f"{crate_of(tc, ts.get('name'))}::{module}"
            sec = float(tc.get("time", "0") or 0)
            s = stats[key]
            s["tests"] += 1
            s["seconds"] += sec
            if "falsif" in name:
                s["f_tests"] += 1
                s["f_seconds"] += sec
    rows = []
    for key, s in stats.items():
        crate, _, module = key.partition("::")
        t = int(touches.get(key, 0))
        density = (t / s["seconds"]) if s["seconds"] > 0 else (float("inf") if t > 0 else 0.0)
        rows.append({"key": key, "module": module, "crate": crate, "tests": s["tests"],
                     "seconds": round(s["seconds"], 3), "touches": t, "f_tests": s["f_tests"],
                     "f_seconds": s["f_seconds"], "density": density})
    rows.sort(key=lambda r: (r["density"], r["touches"], -r["seconds"], r["key"]), reverse=True)
    total_t = sum(r["touches"] for r in rows)
    cum = 0
    cut = len(rows) - 1
    for i, r in enumerate(rows):
        cum += r["touches"]
        if total_t > 0 and cum * 100.0 / total_t >= 80.0:
            cut = i
            break
    if total_t == 0:
        cut = -1
    for i, r in enumerate(rows):
        r["tier"] = "pr" if (i <= cut or r["f_tests"] > 0) else "nightly"
    tot_s = sum(r["seconds"] for r in rows)
    pr_s = sum(r["seconds"] if r["tier"] == "pr" else 0.0 for r in rows)
    pr_n = sum(r["tests"] for r in rows if r["tier"] == "pr")
    tot_n = sum(r["tests"] for r in rows)
    return rows, {"pr_seconds": round(pr_s, 3), "total_seconds": round(tot_s, 3), "pr_tests": pr_n,
                  "total_tests": tot_n, "pr_pct_seconds": round(pr_s * 100.0 / tot_s, 2) if tot_s else 0.0,
                  "cut_index": cut, "modules": len(rows)}


def main():
    junit, ledger, out = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(ledger) as f:
        touches = json.load(f)
    rows, stats = build_rows(junit, touches)
    with open(f"{out}/tier.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "module", "crate", "tests", "seconds", "touches", "f_tests", "tier"],
                           delimiter="\t", extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with open(f"{out}/stats.json", "w") as f:
        json.dump(stats, f)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
