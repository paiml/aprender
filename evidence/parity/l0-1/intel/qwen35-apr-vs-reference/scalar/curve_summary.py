#!/usr/bin/env python3
"""PMAT-3091 scalar: summarise walk_points TSVs. Per position: first callback point whose rel_l2 exceeds 1e-6/1e-5/1e-4/1e-3
(reporting crossings, not judging), l_out-N rel_l2 per layer, and the 6 largest one-step growth ratios along the stream."""
import sys, csv, collections


def crossings(rs):
    out = []
    for t in (1e-6, 1e-5, 1e-4, 1e-3):
        hit = next((r for r in rs if float(r["rel_l2"]) > t), None)
        out.append(f">{t:g}:{hit['tensor']}({float(hit['rel_l2']):.2e})" if hit else f">{t:g}:none")
    return out


def steps(rs):
    found, prev = [], None
    for r in rs:
        v = float(r["rel_l2"])
        if prev and prev[1] > 0 and v > 0:
            found.append((v / prev[1], f"{prev[0]}->{r['tensor']}", v))
        prev = (r["tensor"], v)
    return sorted(found, reverse=True)


def summarise(path):
    rows = [r for r in csv.DictReader(open(path), delimiter="\t") if r["rel_l2"] and not r["rel_l2"].startswith("SIZE")]
    bypos = collections.defaultdict(list)
    for r in rows:
        bypos[int(r["pos"])].append(r)
    for p, rs in sorted(bypos.items()):
        lout = [(int(r["layer"]), float(r["rel_l2"])) for r in rs if r["tensor"].startswith("l_out-")]
        print(f"{rs[0]['label']} pos{p} crossings " + " ".join(crossings(rs)))
        print("  l_out rel_l2: " + " ".join(f"L{l}={v:.1e}" for l, v in lout))
        print("  top step ratios: " + "; ".join(f"{a:.0f}x {b} ({v:.1e})" for a, b, v in steps(rs)[:6]))


if __name__ == "__main__":
    for path in sys.argv[1:]:
        summarise(path)
