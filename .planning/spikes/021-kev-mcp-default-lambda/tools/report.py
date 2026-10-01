"""Render report.html: one stacked bar per host = where the time goes from 'request arrives at a cold host' to
'first MCP answer', from results/*.jsonl. Self-contained SVG, no JS libraries (spike convention).

  python3 tools/report.py && open report.html
"""
import json, statistics as st
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
rows = lambda f: [json.loads(l) for l in open(R / f) if l.strip()]
def step(tl, name):
    return next((x["t_ms"] / 1e3 for x in tl if x["step"] == name), None)

bars = []  # (label, [(phase, seconds)], note)

# default Lambda, weights from S3: controlled rounds (16 x 64 MB parts, row 1) with a load timeline
ls3 = [r for r in rows("lambda-s3.jsonl") if r["label"].startswith("cold") and "load_timeline" in r.get("server", {})]
ls3 = [r for r in ls3 if "x 16" in (next(x.get("detail", "") for x in r["server"]["load_timeline"] if x["step"] == "s3 download"))]
def lambda_phases(r):
    tl = r["server"]["load_timeline"]; dl = step(tl, "s3 download"); ready = step(tl, "engine ready")
    return [("platform init", r.get("init_ms", 0) / 1e3), ("weights: S3 -> /tmp", dl), ("model build", ready - dl),
            ("first decision", r["server"]["decision_ms"] / 1e3)]
med = sorted(ls3, key=lambda r: r["client_wall_ms"])[len(ls3) // 2]
bars.append(("Default Lambda · S3 weights", lambda_phases(med), f"{len(ls3)} cold starts: {min(r['client_wall_ms'] for r in ls3)/1e3:.1f}-{max(r['client_wall_ms'] for r in ls3)/1e3:.1f} s · Graviton2 6 vCPU · 80 MB/s S3 cap"))

# default Lambda, baked image: from CloudWatch (the client timed out); both measured runs
bars.append(("Default Lambda · baked image", [("platform init", 0.39), ("weights: read via lazy image FS", 304.0), ("model build", 4.1), ("first decision", 1.2)],
             "2nd cold start 309 s; 1st ever 800 s (794 s reading 3 GB through the image store)"))

def fargate(f, label):
    rs = [r for r in rows(f) if "timeline_s" in r]
    r = sorted(rs, key=lambda r: r["timeline_s"]["first_answer"])[len(rs) // 2]
    t = r["timeline_s"]; tl = r["first"]["server"]["load_timeline"]; dl = step(tl, "s3 download") or 0
    ready = step(tl, "engine ready")
    ph = [("provisioning + ENI", t["pull_started"]), ("image pull + unpack", t["container_started"] - t["pull_started"])]
    ph += ([("weights: S3 -> disk", dl)] if dl else []) + [("model build", ready - dl), ("first decision", t["first_answer"] - t["container_started"] - ready)]
    spread = f"{min(x['timeline_s']['first_answer'] for x in rs):.0f}-{max(x['timeline_s']['first_answer'] for x in rs):.0f} s"
    return (label, ph, f"{len(rs)} scale-from-zero runs: {spread} · Graviton3/4 8 vCPU")
bars.append(fargate("fargate-s3.jsonl", "Fargate · S3 weights"))
bars.append(fargate("fargate-baked.jsonl", "Fargate · baked image"))

lmi = [r for r in rows("lmi-calls.jsonl") if r.get("server", {}).get("load_timeline")]
first = [r for r in lmi if r["server"]["first_call_in_process"]]
r = first[0]; tl = r["server"]["load_timeline"]; t0 = step(tl, "load start"); dl = step(tl, "s3 download")
bars.append(("Managed Instances · S3 weights", [("weights: S3 -> /tmp", dl - t0), ("model build", step(tl, "engine ready") - dl),
             ("first decision", r["server"]["decision_ms"] / 1e3)],
             "no request-path cold start: environments pre-provisioned (version Active in 48-88 s); this is the lazy first call per env · c9g 8 vCPU"))

colors = {"platform init": "#9aa5b1", "provisioning + ENI": "#9aa5b1", "image pull + unpack": "#c98a2b",
          "weights: S3 -> /tmp": "#3b82c4", "weights: S3 -> disk": "#3b82c4", "weights: read via lazy image FS": "#c0392b",
          "model build": "#5aa469", "first decision": "#7e57c2"}

def svg(bars, scale_max, width=880):
    left, bar_h, gap = 250, 26, 34; W = width; plot = W - left - 20
    h = 40 + len(bars) * (bar_h + gap)
    out = [f'<svg viewBox="0 0 {W} {h}" width="100%" role="img">']
    for tick in [0, 10, 20, 30, 40, 50] if scale_max <= 60 else [0, 60, 120, 180, 240, 300]:
        x = left + plot * tick / scale_max
        out.append(f'<line x1="{x:.1f}" y1="20" x2="{x:.1f}" y2="{h-10}" class="grid"/><text x="{x:.1f}" y="14" class="tick">{tick}s</text>')
    for i, (label, ph, note) in enumerate(bars):
        y = 28 + i * (bar_h + gap); x = left; total = sum(p[1] for p in ph)
        out.append(f'<text x="{left-10}" y="{y+17}" class="lbl">{label}</text>')
        for name, s in ph:
            w = plot * min(s, scale_max) / scale_max
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{max(w,0.8):.1f}" height="{bar_h}" fill="{colors[name]}"><title>{name}: {s:.2f} s</title></rect>')
            x += w
        clip = " ▶ (off scale)" if total > scale_max else ""
        out.append(f'<text x="{min(x+6, W-8):.1f}" y="{y+17}" class="tot" text-anchor="{"end" if total>scale_max else "start"}">{total:.1f} s{clip}</text>')
        out.append(f'<text x="{left}" y="{y+bar_h+13}" class="note">{note}</text>')
    out.append("</svg>")
    return "\n".join(out)

legend = "".join(f'<span><i style="background:{c}"></i>{n}</span>' for n, c in colors.items() if n not in ("weights: S3 -> disk", "provisioning + ENI"))
warm = [("Default Lambda (Graviton2, 6 vCPU)", "1.13-1.16 s"), ("Fargate (Graviton3 / Graviton4, 8 vCPU)", "0.63 s / 0.52 s"),
        ("Managed Instances (c9g, Neoverse-V3, 8 vCPU)", "0.34 s"), ("M4 Pro laptop (6 threads)", "0.36 s")]
html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kev MCP Cold Starts</title><style>
:root{{--bg:#fbfbfa;--fg:#1d1f21;--mute:#5f6b76;--grid:#e3e6e8;--card:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16181a;--fg:#e6e8ea;--mute:#9aa5b1;--grid:#2c3034;--card:#1e2124}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}} main{{max-width:960px;margin:auto}}
h1{{font-size:22px;margin:0 0 4px}} p{{color:var(--mute);margin:4px 0 16px}} .card{{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:16px;margin:16px 0}}
.grid{{stroke:var(--grid)}} .tick,.note{{fill:var(--mute);font-size:11px}} .tick{{text-anchor:middle}} .lbl{{fill:var(--fg);font-size:13px;text-anchor:end}} .tot{{fill:var(--fg);font-size:12px;font-weight:600}}
.legend span{{display:inline-flex;align-items:center;gap:6px;margin:0 14px 6px 0;font-size:12px;color:var(--mute)}} .legend i{{width:12px;height:12px;border-radius:2px;display:inline-block}}
table{{border-collapse:collapse;width:100%;font-size:14px}} td,th{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}}
</style></head><body><main>
<h1>Kev-0.8B as an MCP server on AWS: where a cold start goes</h1>
<p>From a request reaching a cold host to the first <code>tools/call decide</code> answer. 3 GB f32 GGUF, us-east-1, measured 2026-09-24 (spikes 021-023). Parity vs Python fp32: 4.8e-7 on every host.</p>
<div class="card"><div class="legend">{legend}</div>{svg([b for b in bars if b[0] != "Default Lambda · baked image"], 50)}</div>
<div class="card"><p>Off the first chart's scale: weights baked into a Lambda container image.</p>{svg([b for b in bars if b[0] == "Default Lambda · baked image"], 320)}</div>
<div class="card"><table><tr><th>Warm decision, 81-token row</th><th>p50</th></tr>{"".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a,b in warm)}</table></div>
</main></body></html>"""
(R.parent / "report.html").write_text(html)
print("wrote report.html;", [(b[0], round(sum(p[1] for p in b[1]), 1)) for b in bars])
