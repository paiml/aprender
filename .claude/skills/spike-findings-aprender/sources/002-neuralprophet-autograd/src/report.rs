//! Self-contained SVG report for spike 002.
pub struct Series<'a> { pub name: &'a str, pub color: &'a str, pub dash: bool, pub x: Vec<f64>, pub y: Vec<f64>, pub width: f64 }

fn poly(xs: &[f64], ys: &[f64], x0: f64, x1: f64, y0: f64, y1: f64, w: f64, h: f64, pad: f64) -> String {
    let mut s = String::new();
    for (x, y) in xs.iter().zip(ys) {
        if !y.is_finite() { continue; }
        let px = pad + (x - x0) / (x1 - x0) * (w - 2.0 * pad);
        let py = h - pad - (y - y0) / (y1 - y0) * (h - 2.0 * pad);
        if !s.is_empty() { s.push(' '); }
        s.push_str(&format!("{px:.1},{py:.1}"));
    }
    s
}

pub fn render(title: &str, subtitle: &str, series: &[Series], x_split: f64, x_min: f64, x_max: f64, table_html: &str) -> String {
    let (w, h, pad) = (1100.0, 440.0, 40.0);
    let ys: Vec<f64> = series.iter().flat_map(|s| s.x.iter().zip(&s.y).filter(|(x, y)| **x >= x_min && **x <= x_max && y.is_finite()).map(|(_, y)| *y)).collect();
    let ymin = ys.iter().cloned().fold(f64::INFINITY, f64::min) - 0.2;
    let ymax = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 0.2;
    let split_px = pad + (x_split - x_min) / (x_max - x_min) * (w - 2.0 * pad);
    let mut lines = String::new();
    let mut legend = String::new();
    for s in series {
        let pts = poly(&s.x, &s.y, x_min, x_max, ymin, ymax, w, h, pad);
        let dash = if s.dash { " stroke-dasharray=\"6,4\"" } else { "" };
        lines.push_str(&format!("<polyline points=\"{pts}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{}\"{dash}/>\n", s.color, s.width));
        legend.push_str(&format!("<span><i class=\"sw\" style=\"background:{}\"></i>{}</span>", s.color, s.name));
    }
    format!(r##"<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font:14px system-ui;margin:24px;color:#222;background:#fafafa}} .card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px;margin-bottom:16px}} table{{border-collapse:collapse;font-size:13px}} td,th{{border:1px solid #ddd;padding:3px 10px;text-align:right}} td:first-child,th:first-child{{text-align:left}} .legend span{{display:inline-block;margin-right:16px}} .sw{{display:inline-block;width:24px;height:3px;vertical-align:middle;margin-right:6px}}</style></head><body>
<h2>{title}</h2><div class="card">{subtitle}<div class="legend" style="margin-top:8px">{legend}<span>shaded = held-out test year</span></div>
<svg width="{w}" height="{h}" style="display:block;margin-top:8px"><rect x="{split_px:.1}" y="{pad}" width="{:.1}" height="{:.1}" fill="#f0f4ff"/>
{lines}</svg></div>
<div class="card">{table_html}</div></body></html>"##, w - pad - split_px, h - 2.0 * pad)
}
