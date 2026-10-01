//! Self-contained SVG report: y, Rust yhat, Python yhat, plus a residual panel.

fn poly(xs: &[f64], ys: &[f64], x0: f64, x1: f64, y0: f64, y1: f64, w: f64, h: f64, pad: f64) -> String {
    let mut s = String::new();
    for (x, y) in xs.iter().zip(ys) {
        let px = pad + (x - x0) / (x1 - x0) * (w - 2.0 * pad);
        let py = h - pad - (y - y0) / (y1 - y0) * (h - 2.0 * pad);
        if !s.is_empty() { s.push(' '); }
        s.push_str(&format!("{px:.1},{py:.1}"));
    }
    s
}

#[allow(clippy::too_many_arguments)]
pub fn render(ds: &[String], y: &[f64], yhat_rs: &[f64], yhat_py: &[f64], trend_rs: &[f64], trend_py: &[f64], n_hist: usize, label: &str, secs_rs: f64, secs_py: f64, delta_py: &[f64], delta_rs: &[f64]) -> String {
    let n = yhat_rs.len();
    let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let (w, h, pad) = (1100.0, 420.0, 40.0);
    let all: Vec<f64> = y.iter().chain(yhat_rs).chain(yhat_py).copied().collect();
    let ymin = all.iter().cloned().fold(f64::INFINITY, f64::min) - 0.2;
    let ymax = all.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 0.2;
    let x0 = 0.0; let x1 = (n - 1) as f64;
    let split_px = pad + (n_hist as f64) / x1 * (w - 2.0 * pad);
    let y_poly = poly(&xs[..n_hist], y, x0, x1, ymin, ymax, w, h, pad);
    let rs_poly = poly(&xs, yhat_rs, x0, x1, ymin, ymax, w, h, pad);
    let py_poly = poly(&xs, yhat_py, x0, x1, ymin, ymax, w, h, pad);
    let trs_poly = poly(&xs, trend_rs, x0, x1, ymin, ymax, w, h, pad);
    let tpy_poly = poly(&xs, trend_py, x0, x1, ymin, ymax, w, h, pad);
    let resid: Vec<f64> = yhat_rs.iter().zip(yhat_py).map(|(a, b)| a - b).collect();
    let rmax = resid.iter().map(|v| v.abs()).fold(1e-6, f64::max);
    let (rh, rpad) = (160.0, 30.0);
    let r_poly = poly(&xs, &resid, x0, x1, -rmax, rmax, w, rh, rpad);
    let zero_y = rh - rpad - (0.0 + rmax) / (2.0 * rmax) * (rh - 2.0 * rpad);
    let mut delta_rows = String::new();
    for (i, (a, b)) in delta_py.iter().zip(delta_rs).enumerate() {
        delta_rows.push_str(&format!("<tr><td>{i}</td><td>{a:+.4}</td><td>{b:+.4}</td><td>{:.1e}</td></tr>", (a - b).abs()));
    }
    let max_resid_hist = resid[..n_hist].iter().map(|v| v.abs()).fold(0.0, f64::max);
    let max_resid_fut = resid[n_hist..].iter().map(|v| v.abs()).fold(0.0, f64::max);
    format!(r##"<!doctype html><html><head><meta charset="utf-8"><title>Spike 001 — Prophet MAP: aprender L-BFGS vs Python Prophet</title>
<style>body{{font:14px system-ui;margin:24px;color:#222;background:#fafafa}} .card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px;margin-bottom:16px}} table{{border-collapse:collapse;font-size:12px}} td,th{{border:1px solid #ddd;padding:2px 8px;text-align:right}} .legend span{{display:inline-block;margin-right:16px}} .sw{{display:inline-block;width:24px;height:3px;vertical-align:middle;margin-right:6px}}</style></head><body>
<h2>Spike 001 — Prophet MAP fit: aprender <code>LbfgsF64</code> vs Python Prophet (Stan L-BFGS)</h2>
<div class="card"><b>Peyton Manning</b> (log daily Wikipedia views, {n_hist} rows) + 365-day forecast. Rust variant: <b>{label}</b>. Fit time: Rust {secs_rs:.3}s vs Python {secs_py:.2}s (Python time includes cmdstan process spawn).
<div class="legend" style="margin-top:8px"><span><i class="sw" style="background:#bbb"></i>y (observed)</span><span><i class="sw" style="background:#d62728"></i>Rust yhat</span><span><i class="sw" style="background:#1f77b4;border-top:2px dashed #1f77b4;height:0"></i>Python yhat</span><span><i class="sw" style="background:#ff9896"></i>Rust trend</span><span><i class="sw" style="background:#aec7e8"></i>Python trend</span><span>shaded = forecast horizon</span></div>
<svg width="{w}" height="{h}" style="display:block;margin-top:8px">
<rect x="{split_px:.1}" y="{pad}" width="{:.1}" height="{:.1}" fill="#f0f4ff"/>
<polyline points="{y_poly}" fill="none" stroke="#bbb" stroke-width="1"/>
<polyline points="{tpy_poly}" fill="none" stroke="#aec7e8" stroke-width="2"/>
<polyline points="{trs_poly}" fill="none" stroke="#ff9896" stroke-width="1.5"/>
<polyline points="{py_poly}" fill="none" stroke="#1f77b4" stroke-width="2" stroke-dasharray="6,4"/>
<polyline points="{rs_poly}" fill="none" stroke="#d62728" stroke-width="1.2"/>
<text x="{pad}" y="{:.1}" font-size="11" fill="#666">{}</text><text x="{:.1}" y="{:.1}" font-size="11" fill="#666" text-anchor="end">{}</text>
</svg></div>
<div class="card"><b>Rust yhat − Python yhat</b> (max |Δ| history {max_resid_hist:.4}, forecast {max_resid_fut:.4}; y spans ~5.3…12.8)
<svg width="{w}" height="{rh}" style="display:block"><line x1="{rpad}" y1="{zero_y:.1}" x2="{:.1}" y2="{zero_y:.1}" stroke="#999" stroke-dasharray="3,3"/><rect x="{split_px:.1}" y="{rpad}" width="{:.1}" height="{:.1}" fill="#f0f4ff"/><polyline points="{r_poly}" fill="none" stroke="#2ca02c" stroke-width="1.2"/><text x="{rpad}" y="14" font-size="11" fill="#666">±{rmax:.4}</text></svg></div>
<div class="card"><b>Changepoint deltas</b> (Laplace prior, τ = 0.05) — Python vs Rust<table><tr><th>#</th><th>Python δ</th><th>Rust δ</th><th>|Δ|</th></tr>{delta_rows}</table></div>
</body></html>"##,
        w - pad - split_px, h - 2.0 * pad, h - pad + 14.0, ds[0], w - pad, h - pad + 14.0, ds[n - 1], w - rpad, w - rpad - split_px, rh - 2.0 * rpad)
}
