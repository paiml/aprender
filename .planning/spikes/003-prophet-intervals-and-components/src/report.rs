//! SVG panels: y, yhat, band, trend — one per fixture.
fn px(x: f64, x0: f64, x1: f64, w: f64, pad: f64) -> f64 { pad + (x - x0) / (x1 - x0) * (w - 2.0 * pad) }
fn py(y: f64, y0: f64, y1: f64, h: f64, pad: f64) -> f64 { h - pad - (y - y0) / (y1 - y0) * (h - 2.0 * pad) }
fn poly(xs: &[f64], ys: &[f64], x0: f64, x1: f64, y0: f64, y1: f64, w: f64, h: f64, pad: f64) -> String {
    xs.iter().zip(ys).filter(|(_, y)| y.is_finite()).map(|(x, y)| format!("{:.1},{:.1}", px(*x, x0, x1, w, pad), py(*y, y0, y1, h, pad))).collect::<Vec<_>>().join(" ")
}
pub struct Panel { pub title: String, pub xs_hist: Vec<f64>, pub y: Vec<f64>, pub xs: Vec<f64>, pub yhat: Vec<f64>, pub lower: Vec<f64>, pub upper: Vec<f64>, pub py_lower: Vec<f64>, pub py_upper: Vec<f64>, pub py_yhat: Vec<f64>, pub note: String }
pub fn render(title: &str, panels: &[Panel]) -> String {
    let (w, h, pad) = (1100.0, 360.0, 40.0);
    let mut body = String::new();
    for p in panels {
        let x0 = p.xs[0]; let x1 = p.xs[p.xs.len() - 1];
        let all: Vec<f64> = p.y.iter().chain(&p.lower).chain(&p.upper).chain(&p.py_lower).chain(&p.py_upper).cloned().filter(|v| v.is_finite()).collect();
        let y0 = all.iter().cloned().fold(f64::INFINITY, f64::min); let y1 = all.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let (y0, y1) = (y0 - 0.05 * (y1 - y0), y1 + 0.05 * (y1 - y0));
        let split = px(*p.xs_hist.last().expect("hist"), x0, x1, w, pad);
        let band: String = { let up = poly(&p.xs, &p.upper, x0, x1, y0, y1, w, h, pad); let xs_r: Vec<f64> = p.xs.iter().rev().cloned().collect(); let lo_r: Vec<f64> = p.lower.iter().rev().cloned().collect(); format!("{up} {}", poly(&xs_r, &lo_r, x0, x1, y0, y1, w, h, pad)) };
        body.push_str(&format!(r##"<div class="card"><b>{}</b><br><span class="note">{}</span>
<div class="legend"><span><i class="sw" style="background:#999"></i>y</span><span><i class="sw" style="background:#d62728"></i>Rust yhat</span><span><i class="sw" style="background:#f4b6b6;height:10px"></i>Rust 80% band</span><span><i class="sw" style="background:#1f77b4;border-top:2px dashed #1f77b4;height:0"></i>Python yhat_lower / yhat_upper</span></div>
<svg width="{w}" height="{h}" style="display:block"><rect x="{split:.1}" y="{pad}" width="{:.1}" height="{:.1}" fill="#f0f4ff"/>
<polygon points="{band}" fill="#f4b6b6" stroke="none" opacity="0.8"/>
<polyline points="{}" fill="none" stroke="#1f77b4" stroke-width="1.2" stroke-dasharray="5,4"/><polyline points="{}" fill="none" stroke="#1f77b4" stroke-width="1.2" stroke-dasharray="5,4"/>
<polyline points="{}" fill="none" stroke="#999" stroke-width="1"/>
<polyline points="{}" fill="none" stroke="#d62728" stroke-width="1.5"/></svg></div>
"##, p.title, p.note, w - pad - split, h - 2.0 * pad,
            poly(&p.xs, &p.py_lower, x0, x1, y0, y1, w, h, pad), poly(&p.xs, &p.py_upper, x0, x1, y0, y1, w, h, pad),
            poly(&p.xs_hist, &p.y, x0, x1, y0, y1, w, h, pad), poly(&p.xs, &p.yhat, x0, x1, y0, y1, w, h, pad)));
    }
    format!(r##"<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font:14px system-ui;margin:24px;color:#222;background:#fafafa}} .card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px;margin-bottom:16px}} .note{{color:#555;font-size:13px}} .legend span{{display:inline-block;margin:6px 16px 6px 0}} .sw{{display:inline-block;width:24px;height:3px;vertical-align:middle;margin-right:6px}}</style></head><body><h2>{title}</h2>{body}</body></html>"##)
}
