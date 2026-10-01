"""Render results.json into a self-contained SVG page (no JS libraries)."""
import json, html

R = json.load(open("results.json"))
W, H, PAD = 1040, 260, 48

def path(vals, lo, hi, n, x0=PAD, w=W-2*PAD, y0=PAD*0.4, h=H-PAD*1.4):
    if hi == lo: hi = lo + 1
    pts = []
    for i, v in enumerate(vals):
        x = x0 + w * (i / max(n - 1, 1))
        y = y0 + h * (1 - (v - lo) / (hi - lo))
        pts.append(f"{x:.2f},{y:.2f}")
    return "M" + " L".join(pts)

def chart(title, series, n_hist, sub=""):
    allv = [v for _, vs, _ in series for v in vs if v == v]
    lo, hi = min(allv), max(allv)
    pad = (hi - lo) * 0.08 or 1
    lo, hi = lo - pad, hi + pad
    n = max(len(vs) for _, vs, _ in series)
    xh = PAD + (W - 2*PAD) * (n_hist - 1) / max(n - 1, 1)
    out = [f'<figure><figcaption><b>{html.escape(title)}</b>{" — " + sub if sub else ""}</figcaption>',
           f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">',
           f'<rect x="0" y="0" width="{W}" height="{H}" fill="var(--panel)"/>',
           f'<line x1="{xh:.1f}" y1="{PAD*0.4:.0f}" x2="{xh:.1f}" y2="{H-PAD:.0f}" stroke="var(--rule)" stroke-dasharray="4 4"/>',
           f'<text x="{xh+6:.1f}" y="{PAD*0.4+12:.0f}" class="ann">forecast starts</text>']
    for (label, vs, color) in series:
        dash = ' stroke-dasharray="6 4"' if label.startswith("rust") else ""
        out.append(f'<path d="{path(vs, lo, hi, n)}" fill="none" stroke="{color}" stroke-width="1.8"{dash}/>')
    lx = PAD
    for (label, _, color) in series:
        out.append(f'<rect x="{lx}" y="{H-30}" width="11" height="11" fill="{color}"/>'
                   f'<text x="{lx+16}" y="{H-20}" class="lg">{html.escape(label)}</text>')
        lx += 22 + 7.2 * len(label)
    out.append(f'<text x="{PAD}" y="{H-PAD+30:.0f}" class="ax">{lo:,.0f} … {hi:,.0f}</text>')
    out.append('</svg></figure>')
    return "\n".join(out)

def comp(block, name):
    for c in block["components"]:
        if c["name"] == name:
            return c
    return None

def section(key, heading):
    b = R[key]
    r = b["rungs"]
    n = len(b["ds"]); nh = b["n_history"]
    s = [f"<h2>{html.escape(heading)}</h2>"]
    s.append(chart("Forecast: Python Prophet 1.4.0 vs the Rust port",
        [("python yhat", b["py_yhat"], "var(--c1)"),
         ("rust yhat @ python MAP", b["rust_yhat_at_py_params"], "var(--c2)"),
         ("rust yhat @ rust fit", b["rust_yhat_fit"], "var(--c3)")], nh,
        f'max |d| at Python MAP = {r["d_yhat"]:.1e} ({r["d_yhat"]/b["y_scale"]:.1e} of y_scale)'))
    for nm in ["extra_regressors_additive", "extra_regressors_multiplicative"]:
        c = comp(b, nm)
        if c and c["python"]:
            d = max(abs(x - y) for x, y in zip(c["rust"], c["python"]))
            s.append(chart(f"Component: {nm}",
                [("python", c["python"], "var(--c1)"), ("rust", c["rust"], "var(--c2)")], nh,
                f"max |d| = {d:.1e}"))
    s.append("<h3>Identifiability — why two optimisers disagree on a coefficient</h3>")
    s.append("<table><tr><th>regressor</th><th>mode</th><th>python β</th><th>rust β</th>"
             "<th>r vs trend</th><th>max r vs another column</th><th>identifiable</th></tr>")
    for row in b["identifiability"]:
        cls = ' class="bad"' if row["collinear"] else ""
        s.append(f'<tr{cls}><td>{row["name"]}</td><td>{row["mode"]}</td>'
                 f'<td>{row["py_beta"]:+.6f}</td><td>{row["rust_beta"]:+.6f}</td>'
                 f'<td>{row["r_trend"]:+.3f}</td><td>{row["r_other"]:+.3f} <code>{row["r_other_col"]}</code></td>'
                 f'<td>{"no — collinear" if row["collinear"] else "yes"}</td></tr>')
    s.append("</table>")
    s.append(f'<p class="note">Rung 1 column order identical to Python: <b>{"yes" if r["col_order_ok"] else "NO"}</b> '
             f'({r["n_cols"]} columns). Δprior_scales {r["d_prior_scales"]:.0e}, Δs_a {r["d_s_a"]:.0e}, '
             f'Δs_m {r["d_s_m"]:.0e}, ΔX {r["d_x"]:.1e}. Rung 3: {r["n_components"]} components, '
             f'worst {r["worst_component"]:.1e}. Rung 4: status {r["fit_status"]}, '
             f'objective slack <b>{r["slack"]:+.3f}</b> (contract bar 0.5; negative = Rust found a better optimum).</p>')
    return "\n".join(s)

page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spike 011 — Prophet external regressors</title><style>
:root {{ --bg:#fbfaf9; --fg:#1d1b19; --mut:#6b6560; --panel:#fff; --rule:#d9d4cf;
  --c1:#2f6f9f; --c2:#c8672a; --c3:#5b8c3e; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#17151a; --fg:#ece8e3; --mut:#a49e98;
  --panel:#201d23; --rule:#3a353f; --c1:#7fb6de; --c2:#e9975c; --c3:#93c46f; }} }}
* {{ box-sizing:border-box }}
body {{ margin:0; padding:24px 16px 64px; background:var(--bg); color:var(--fg);
  font:15px/1.55 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif; }}
main {{ max-width:1100px; margin:0 auto }}
h1 {{ font-size:1.5rem; margin:0 0 4px }} h2 {{ font-size:1.15rem; margin:36px 0 8px }}
h3 {{ font-size:1rem; margin:24px 0 8px }}
.sub {{ color:var(--mut); margin:0 0 8px }}
figure {{ margin:16px 0 }} figcaption {{ color:var(--mut); font-size:.88rem; margin-bottom:6px }}
svg {{ width:100%; height:auto; border:1px solid var(--rule); border-radius:8px }}
.lg,.ax,.ann {{ font:11px ui-monospace,monospace; fill:var(--mut) }}
table {{ border-collapse:collapse; width:100%; font-size:.9rem; margin:8px 0 }}
th,td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--rule) }}
th {{ color:var(--mut); font-weight:600 }}
tr.bad td {{ background:color-mix(in srgb, var(--c2) 12%, transparent) }}
code {{ font:12px ui-monospace,monospace; color:var(--mut) }}
.note {{ color:var(--mut); font-size:.88rem; border-left:3px solid var(--rule); padding-left:12px }}
</style></head><body><main>
<h1>Spike 011 — Prophet external regressors</h1>
<p class="sub">Does the shipped <code>aprender-forecast</code> design-matrix path reach Prophet 1.4.0
parity with <code>add_regressor</code>, and does it need restructuring to get there?
Oracle: <code>prophet==1.4.0</code> on <code>retail_sales</code>.</p>
{section("regressors_only", "retail_sales + 4 regressors")}
{section("regressors_and_holidays", "retail_sales + 2 holidays (with windows) + 4 regressors")}
</main></body></html>"""
open("report.html", "w").write(page)
print("wrote report.html", len(page), "bytes")
