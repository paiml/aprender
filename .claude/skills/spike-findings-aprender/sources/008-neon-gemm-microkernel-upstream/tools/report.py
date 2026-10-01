#!/usr/bin/env python3
"""Render report.html (self-contained SVG) from results-before.json, results-after.json and the
criterion logs. Run from the spike directory: python3 tools/report.py"""
import json, re, html

before = {r["shape"]: r for r in json.load(open("results-before.json"))}
after = {r["shape"]: r for r in json.load(open("results-after.json"))}

def crit(path):
    out = {}
    for line in open(path, errors="replace"):
        m = re.match(r"^gemm/(\w+)/(\d+)\s+time:\s+\[\S+ \S+ (\S+) (\S+) ", line)
        if m:
            lib, n, val, unit = m.groups()
            v = float(val) * {"ns": 1e-6, "µs": 1e-3, "ms": 1.0, "s": 1e3}[unit]
            out[(lib, int(n))] = v
    return out

cb, ca = crit("bench-before.log"), crit("bench-after.log")

shapes = [s for s in before if not s.startswith(("k=", "odd 7"))]
W, H, L = 900, 34, 250
rows = []
maxg = max(max(after[s]["faer_gfs"], after[s]["blis_gfs"]) for s in shapes) * 1.05
def bar(y, v, color, label):
    w = v / maxg * (W - L - 90)
    return (f'<rect x="{L}" y="{y}" width="{w:.1f}" height="9" fill="{color}"/>'
            f'<text x="{L + w + 4:.1f}" y="{y + 8}" font-size="10" fill="#333">{label}</text>')
svg = []
y = 20
for s in shapes:
    b, a = before[s], after[s]
    svg.append(f'<text x="8" y="{y + 20}" font-size="12" fill="#222">{html.escape(s)}</text>')
    svg.append(bar(y + 2, b["blis_gfs"], "#c0392b", f'before {b["blis_gfs"]:.1f}'))
    svg.append(bar(y + 13, a["blis_gfs"], "#27ae60", f'after {a["blis_gfs"]:.1f}  ({a["blis_gfs"]/b["blis_gfs"]:.1f}×)'))
    svg.append(bar(y + 24, a["faer_gfs"], "#7f8c8d", f'faer {a["faer_gfs"]:.1f}'))
    y += H + 6
svg_h = y + 10

crit_rows = "".join(
    f"<tr><td>{n}³</td><td>{cb.get(('trueno', n), float('nan')):.3f}</td><td>{ca.get(('trueno', n), float('nan')):.3f}</td>"
    f"<td>{cb.get(('trueno', n), 1) / max(ca.get(('trueno', n), 1), 1e-9):.1f}×</td><td>{ca.get(('faer', n), float('nan')):.3f}</td>"
    f"<td>{ca.get(('trueno', n), 0) / max(ca.get(('faer', n), 1), 1e-9):.2f}×</td></tr>"
    for n in (64, 128, 256, 512, 1024))

table = "".join(
    f"<tr><td>{html.escape(s)}</td><td>{before[s]['blis_gfs']:.1f}</td><td>{after[s]['blis_gfs']:.1f}</td>"
    f"<td>{after[s]['blis_gfs']/before[s]['blis_gfs']:.1f}×</td><td>{after[s]['plain_gfs']:.1f}</td><td>{after[s]['faer_gfs']:.1f}</td>"
    f"<td>{after[s]['blis_err']:.1e}</td></tr>" for s in before)

page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Spike 008 — NEON GEMM microkernel</title>
<style>body{{font:14px/1.45 system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;font-size:13px;margin:12px 0}}td,th{{border:1px solid #ddd;padding:4px 8px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}h1{{font-size:20px}}h2{{font-size:16px;margin-top:28px}}.note{{color:#555}}</style></head><body>
<h1>Spike 008 — <code>gemm_blis</code> on aarch64: scalar microkernel → NEON 8×6</h1>
<p class="note">Apple M4 Pro, single thread. GFLOP/s, higher is better. Red = before (scalar kernel), green = after (NEON 8×6), grey = faer (pure-Rust reference BLAS).</p>
<svg width="{W}" height="{svg_h}" font-family="system-ui,sans-serif">{''.join(svg)}</svg>
<h2>Same driver, all shapes</h2>
<table><tr><th>shape</th><th>before GF/s</th><th>after GF/s</th><th>speed-up</th><th>plain8 GF/s</th><th>faer GF/s</th><th>after rel. err</th></tr>{table}</table>
<h2>The crate's own instrument: <code>benches/gemm_comparison.rs</code> (criterion mean, ms)</h2>
<table><tr><th>n</th><th>trueno before</th><th>trueno after</th><th>speed-up</th><th>faer</th><th>trueno/faer time</th></tr>{crit_rows}</table>
<p class="note">Errors are relative to an f64 reference; the 1e-5 bar is contract C-NEON-BLIS-001. Shapes with K ≤ 3 are omitted from the chart (packing dominates; unchanged by the kernel).</p>
</body></html>"""
open("report.html", "w").write(page)
print("wrote report.html")
