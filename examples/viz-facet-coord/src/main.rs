//! #3552: study data -> `panels` -> `Coord::apply` -> SVG, against the published
//! `aprender-viz` crate (lib name `trueno_viz`). EV-15/EV-16 build against this.
//!
//! Usage: viz-facet-coord [OUT.svg]   (default: stdout)
//! Exits non-zero if a facet or coord law it relies on does not hold.

use trueno_viz::color::Rgba;
use trueno_viz::grammar::{apply, apply_limits, panels, Coord, DataFrame, Facet};
use trueno_viz::output::{SvgEncoder, TextAnchor};

const PANEL_W: f32 = 240.0;
const PANEL_H: f32 = 180.0;
const PAD: f32 = 24.0;
const NCOL: usize = 2;

/// A small study: dose vs response for three lineages.
fn study() -> DataFrame {
    let lineage = ["b", "a", "c", "a", "b", "c", "a", "b", "c", "a", "b", "c"];
    let dose = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 8.0, 8.0, 8.0];
    let response = [2.0, 1.5, 3.0, 3.1, 2.4, 4.2, 4.0, 3.5, 5.9, 5.2, 4.1, 7.5];
    let mut df = DataFrame::new();
    df.add_column_str("lineage", &lineage);
    df.add_column_f32("dose", &dose);
    df.add_column_f32("response", &response);
    df
}

fn range(v: &[f32]) -> (f64, f64) {
    let lo = v.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (f64::from(lo), f64::from(hi))
}

fn scale(v: f64, (lo, hi): (f64, f64), px0: f32, px1: f32) -> f32 {
    let t = if hi > lo { (v - lo) / (hi - lo) } else { 0.5 };
    px0 + (t as f32) * (px1 - px0)
}

fn run(out: Option<&str>) -> Result<String, String> {
    let df = study();
    let x = df.get_f32_present("dose").ok_or("no dose column")?;
    let y = df.get_f32_present("response").ok_or("no response column")?;

    // Facet: one panel per lineage, wrapped two to a row.
    let ps = panels(&Facet::wrap("lineage", NCOL), &df).map_err(|e| e.to_string())?;
    let covered: usize = ps.iter().map(|p| p.rows.len()).sum();
    if ps.len() != 3 || covered != df.nrow() {
        return Err(format!("facet law: {} panels covering {covered}/{} rows", ps.len(), df.nrow()));
    }

    // Coord: flipped cartesian with a declared x limit, so dose runs vertically.
    let coord = Coord::cartesian().xlim(0.0, 8.0).flip();
    let (dx, dy) = apply_limits(&coord, range(&x), range(&y)).map_err(|e| e.to_string())?;
    if dy != (0.0, 8.0) {
        return Err(format!("coord law: flip must move xlim to the vertical axis, got {dy:?}"));
    }

    let nrow = ps.iter().map(|p| p.row).max().unwrap_or(0) + 1;
    let w = (NCOL as f32) * (PANEL_W + PAD) + PAD;
    let h = (nrow as f32) * (PANEL_H + PAD) + PAD;
    let mut svg = SvgEncoder::new(w as u32, h as u32).background(Some(Rgba::WHITE));
    let mut points = 0usize;
    for p in &ps {
        let x0 = PAD + (p.col as f32) * (PANEL_W + PAD);
        let y0 = PAD + (p.row as f32) * (PANEL_H + PAD);
        let (x1, y1) = (x0 + PANEL_W, y0 + PANEL_H);
        svg = svg
            .rect(x0, y0, PANEL_W, PANEL_H, Rgba::rgb(240, 240, 240))
            .text_anchored((x0 + x1) / 2.0, y0 - 6.0, &p.levels.join(" / "), 12.0, Rgba::BLACK, TextAnchor::Middle);
        let mut line = Vec::with_capacity(p.rows.len());
        for &i in &p.rows {
            let (px, py) = apply(&coord, f64::from(x[i]), f64::from(y[i])).map_err(|e| e.to_string())?;
            // SVG y grows downward: the vertical domain maps y1 -> y0.
            let (sx, sy) = (scale(px, dx, x0, x1), scale(py, dy, y1, y0));
            if !(x0..=x1).contains(&sx) || !(y0..=y1).contains(&sy) {
                return Err(format!("point {i} ({sx},{sy}) escaped panel {:?}", p.levels));
            }
            line.push((sx, sy));
            svg = svg.circle(sx, sy, 3.5, Rgba::rgb(31, 119, 180));
            points += 1;
        }
        svg = svg.polyline(&line, Rgba::rgb(31, 119, 180), 1.0);
    }
    if points != df.nrow() {
        return Err(format!("drew {points} points for {} rows", df.nrow()));
    }

    let doc = svg.render();
    match out {
        Some(path) => svg.write_to_file(path).map_err(|e| e.to_string())?,
        None => print!("{doc}"),
    }
    Ok(format!("panels={} points={points} bytes={}", ps.len(), doc.len()))
}

fn main() {
    let out = std::env::args().nth(1);
    match run(out.as_deref()) {
        Ok(summary) => eprintln!("viz-facet-coord: {summary}"),
        Err(e) => {
            eprintln!("viz-facet-coord: FAIL: {e}");
            std::process::exit(1);
        }
    }
}
