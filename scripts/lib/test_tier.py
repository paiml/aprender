"""test_tier.py — the 80/20 PR-tier table from a nextest junit and a catch ledger (PMAT-1105, spec §6).

usage: python3 scripts/lib/test_tier.py JUNIT LEDGER_JSON OUT_DIR
       python3 scripts/lib/test_tier.py filterset [--tsv evidence/fleet/test-tier.tsv]
         -> filterset= (a nextest DNF expression covering every tier=pr row) plus
            tier_of_record_{tests,modules,packages,seconds}; exit 1 on any unusable table (PMAT-3119).
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
import csv, json, re, sys
import xml.etree.ElementTree as ET
from collections import defaultdict


def crate_of(tc, ts_name):
    cls = tc.get("classname", "") or ""
    if cls:
        return cls.split("/")[0].split("::")[0]
    return (ts_name or "").split("::")[0]


def _tally(stats, tc, ts_name):
    """Fold one junit testcase into its `crate::module` bucket."""
    name = tc.get("name", "") or ""
    module = name.rsplit("::", 1)[0] if "::" in name else ""
    key = f"{crate_of(tc, ts_name)}::{module}"
    sec = float(tc.get("time", "0") or 0)
    s = stats[key]
    s["tests"] += 1
    s["seconds"] += sec
    if "falsif" in name:
        s["f_tests"] += 1
        s["f_seconds"] += sec


def _junit_stats(junit_path):
    stats = defaultdict(lambda: {"tests": 0, "seconds": 0.0, "f_tests": 0, "f_seconds": 0.0})
    root = ET.parse(junit_path).getroot()
    for ts in root.iter("testsuite"):
        for tc in ts.iter("testcase"):
            _tally(stats, tc, ts.get("name"))
    return stats


def _rows_from_stats(stats, touches):
    """The table rows, sorted by touches per second — the 80/20 order."""
    rows = []
    for key, s in stats.items():
        crate, _, module = key.partition("::")
        t = int(touches.get(key, 0))
        density = (t / s["seconds"]) if s["seconds"] > 0 else (float("inf") if t > 0 else 0.0)
        rows.append({"key": key, "module": module, "crate": crate, "tests": s["tests"],
                     "seconds": round(s["seconds"], 3), "touches": t, "f_tests": s["f_tests"],
                     "f_seconds": s["f_seconds"], "density": density})
    rows.sort(key=lambda r: (r["density"], r["touches"], -r["seconds"], r["key"]), reverse=True)
    return rows


def _cut_index(rows):
    """Last index inside the cumulative 80 % of fix-linked touches; -1 when nothing was ever touched."""
    total_t = sum(r["touches"] for r in rows)
    if total_t == 0:
        return -1
    cum = 0
    for i, r in enumerate(rows):
        cum += r["touches"]
        if cum * 100.0 / total_t >= 80.0:
            return i
    return len(rows) - 1


def _summary(rows, cut):
    tot_s = sum(r["seconds"] for r in rows)
    pr_s = sum(r["seconds"] if r["tier"] == "pr" else 0.0 for r in rows)
    return {"pr_seconds": round(pr_s, 3), "total_seconds": round(tot_s, 3),
            "pr_tests": sum(r["tests"] for r in rows if r["tier"] == "pr"),
            "total_tests": sum(r["tests"] for r in rows),
            "pr_pct_seconds": round(pr_s * 100.0 / tot_s, 2) if tot_s else 0.0,
            "cut_index": cut, "modules": len(rows)}


def build_rows(junit_path, touches):
    rows = _rows_from_stats(_junit_stats(junit_path), touches)
    cut = _cut_index(rows)
    for i, r in enumerate(rows):
        r["tier"] = "pr" if (i <= cut or r["f_tests"] > 0) else "nightly"
    return rows, _summary(rows, cut)


# --- PMAT-3119: the tier of record as a nextest filterset -------------------------------------
#
# A `tier=pr` row maps to a nextest filterset atom by kind:
#   lib module        -> package(=CRATE) & kind(lib) & test(/^MODULE::/)
#   integration bin   -> package(=CRATE) & binary(=MODULE)
# The optional 9th column `kind` (values `lib` | `test`) says which; absent or `lib` means a lib
# module, which is what every row of the table of record is (it was measured from a `--lib` junit).
# Rows are grouped per package so the DNF expression stays compact. Anything that is not a `pr`
# row is excluded. Every failure is exit 1 — never a silent full run, never a silent empty set.

class TierError(Exception):
    """The TSV cannot be turned into a filterset. Always fatal (exit 1)."""


_HEADER = ["key", "module", "crate", "tests", "seconds", "touches", "f_tests", "tier"]


def _read_tsv(path):
    try:
        with open(path, newline="") as f:
            rows = [r for r in csv.reader(f, delimiter="\t") if r]
    except OSError as e:
        raise TierError(f"tier table not readable: {path}: {e}") from e
    if not rows:
        raise TierError(f"tier table is empty (no header): {path}")
    head = rows[0]
    if head[: len(_HEADER)] != _HEADER:
        raise TierError(f"tier table header is not {'|'.join(_HEADER)}: {path}: got {'|'.join(head)}")
    return head, rows[1:]


def _rx(module):
    """Escape a module path for a nextest test(/.../) regex."""
    return re.escape(module)


def _row_kind(path, i, row, kind_at):
    """`lib` (a module) or `test` (an integration binary) for one row."""
    kind = (row[kind_at].strip() if 0 <= kind_at < len(row) else "") or "lib"
    if kind not in ("lib", "test"):
        raise TierError(f"{path}:{i}: unknown kind {kind!r} (want lib|test)")
    return kind


def _row_cost(path, i, row):
    try:
        return int(row[3]), float(row[4])
    except ValueError as e:
        raise TierError(f"{path}:{i}: tests/seconds are not numeric: {e}") from e


def _pr_rows(path, head, body):
    """Yield (crate, module, kind, tests, seconds) for every tier=pr row. Raises TierError."""
    kind_at = head.index("kind") if "kind" in head else -1
    for i, row in enumerate(body, start=2):
        if len(row) < len(_HEADER):
            raise TierError(f"{path}:{i}: row has {len(row)} field(s), want at least {len(_HEADER)}")
        module, crate, tier = row[1].strip(), row[2].strip(), row[7].strip()
        if tier != "pr":
            continue
        if not crate or not module:
            raise TierError(f"{path}:{i}: tier=pr row with an empty crate/module (crate={crate!r} module={module!r})")
        tests, seconds = _row_cost(path, i, row)
        yield crate, module, _row_kind(path, i, row, kind_at), tests, seconds


def _groups_for(crate, libs, bins):
    """The per-package filterset group(s): the lib modules first, then the integration binaries."""
    out = []
    if crate in libs:
        alts = "|".join(_rx(m) for m in sorted(set(libs[crate])))
        out.append(f"(package(={crate}) & kind(lib) & test(/^({alts})::/))")
    if crate in bins:
        names = sorted(set(bins[crate]))
        one = f"binary(={names[0]})" if len(names) == 1 else "(" + " | ".join(f"binary(={n})" for n in names) + ")"
        out.append(f"(package(={crate}) & {one})")
    return out


def filterset_from_tsv(path):
    """(expr, stats) for every tier=pr row of the tier table at `path`. Raises TierError."""
    head, body = _read_tsv(path)
    libs, bins = defaultdict(list), defaultdict(list)
    tests, seconds, modules = 0, 0.0, 0
    for crate, module, kind, n, sec in _pr_rows(path, head, body):
        (bins if kind == "test" else libs)[crate].append(module)
        modules += 1
        tests += n
        seconds += sec
    if modules == 0:
        raise TierError(f"{path}: zero tier=pr rows — refusing to emit an empty filterset")
    packages = sorted(set(libs) | set(bins))
    groups = [g for crate in packages for g in _groups_for(crate, libs, bins)]
    stats = {"tier_of_record_tests": tests, "tier_of_record_modules": modules,
             "tier_of_record_packages": len(packages),
             "tier_of_record_seconds": f"{seconds:.2f}"}
    return " | ".join(groups), stats


def filterset_main(argv):
    path = "evidence/fleet/test-tier.tsv"
    i = 0
    while i < len(argv):
        if argv[i] == "--tsv" and i + 1 < len(argv):
            path = argv[i + 1]
            i += 2
        else:
            raise TierError(f"usage: test_tier.py filterset [--tsv PATH] (got {argv[i]!r})")
    expr, stats = filterset_from_tsv(path)
    print(f"filterset={expr}")
    for k, v in stats.items():
        print(f"{k}={v}")
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "filterset":
        try:
            return filterset_main(sys.argv[2:])
        except TierError as e:
            print(f"test_tier: {e}", file=sys.stderr)
            return 1
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
