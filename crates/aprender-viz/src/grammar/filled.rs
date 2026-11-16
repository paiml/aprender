//! Filled position layers: `geom_area` and `geom_ribbon` as `ggplot_build()` in ggplot2 4.0.3
//! resolves them (PLOT-GG-001, paiml/aprender#6909). Proven against R by
//! `ggplot2_geom_area_oracle.rs` and `ggplot2_geom_ribbon_oracle.rs`.
//!
//! # What is resolved
//!
//! * `geom_area()`: `stat_align` (the group-aligning stat), `GeomArea$setup_data` (`ymin = 0`,
//!   `ymax = y`, rows ordered by panel, group, x) and `position_stack()`, whose stacking engages
//!   only when an `x` repeats somewhere in the layer.
//! * `geom_ribbon()`: `stat_identity`, `GeomRibbon$setup_data` (rows ordered by panel, group, x with
//!   a missing `x` last; `y` is `ymin`), `position_identity`. Rows with missing values are kept.
//!
//! # What is refused (typed error, never a plausible wrong answer)
//!
//! A data-mapped colour, fill, size, shape, alpha or label; a stat other than the geom's own; an
//! infinite or text position; a layer without its required mapping. The same coordinate, companion
//! and degenerate-scale refusals as the path family apply (see `path`).
//!
//! The panel's trained scales include everything ggplot2 trains on after position adjustment:
//! `x`, `y`, `ymin`, `ymax` and, where stacking ran, `xmin`/`xmax`, padding rows included.

use super::aes::Aes;
use super::build::BuiltLayer;
use super::data::DataFrame;
use super::facet::Facet;
use super::geom::GeomType;
use super::ggplot::Layer;
use super::path::{cmp_na_last, global_panels, group_ids, read_column, refuse, row_panels};
use super::position::{apply_panels, BarRow, Position};
use super::stat::Stat;
use crate::error::{Error, Result};

/// The resolved columns of a filled layer beside `BuiltLayer::x` / `BuiltLayer::y`, one entry per
/// row of `ggplot_build()` output.
#[derive(Debug, Clone, PartialEq)]
pub struct BuiltFilled {
    /// 0-based panel index of each row.
    pub panel: Vec<usize>,
    /// ggplot2's group id of each row (`-1` when no group is mapped and one group is implied).
    pub group: Vec<i64>,
    /// Lower edge of each row.
    pub ymin: Vec<f64>,
    /// Upper edge of each row.
    pub ymax: Vec<f64>,
    /// `xmin`, present only when `position_stack` ran (then equal to `x`).
    pub xmin: Option<Vec<f64>>,
    /// `xmax`, present only when `position_stack` ran (then equal to `x`).
    pub xmax: Option<Vec<f64>>,
    /// `stat_align`'s `align_padding`: `Some(true)` on the two padding rows of an aligned group,
    /// `Some(false)` on the others, `None` (R's `NA`) on a panel `stat_align` did not touch. The
    /// column itself is absent (`None`) when `stat_align` ran on no panel.
    pub align_padding: Option<Vec<Option<bool>>>,
    /// ggplot2's `flipped_aes`; always `false` here (the vertical orientation is the only one).
    pub flipped_aes: bool,
}

/// Is this a geom this module builds?
pub(super) fn is_filled(g: &GeomType) -> bool {
    matches!(g, GeomType::Area { .. } | GeomType::Ribbon)
}

/// Refuse what the module cannot resolve on one filled layer.
fn require_supported(layer: &Layer, aes: &Aes) -> Result<()> {
    let name =
        if matches!(layer.geom.geom_type, GeomType::Ribbon) { "geom_ribbon" } else { "geom_area" };
    if aes.color.is_some()
        || aes.size.is_some()
        || aes.shape.is_some()
        || aes.alpha.is_some()
        || aes.fill.is_some()
        || aes.label.is_some()
    {
        return Err(refuse(format!(
            "a data-mapped colour, size, shape, alpha, fill or label is not supported on {name}"
        )));
    }
    let own_stat = match layer.geom.stat.as_ref() {
        None => true,
        Some(Stat::Identity) => matches!(layer.geom.geom_type, GeomType::Ribbon),
        Some(_) => false,
    };
    if own_stat {
        return Ok(());
    }
    Err(Error::UnsupportedStat(
        "an explicit stat on geom_area or geom_ribbon: layout() resolves only the geom's own",
    ))
}

/// Build one `geom_area` or `geom_ribbon` layer.
pub(super) fn build_layer(
    layer: &Layer,
    plot_aes: &Aes,
    plot_data: &DataFrame,
    facet: &Facet,
) -> Result<BuiltLayer> {
    let aes = plot_aes.merge(&layer.aes);
    require_supported(layer, &aes)?;
    let global = global_panels(facet, plot_data)?;
    let data = layer.data.as_ref().unwrap_or(plot_data);
    let panel = row_panels(facet, &global, data, layer.data.is_some())?;
    let group = group_ids(&aes, data)?;
    let x = read_column(data, aes.x.as_deref().unwrap_or("x"), "x")?;
    match layer.geom.geom_type {
        GeomType::Ribbon => {
            let lo = read_edge(data, aes.ymin.as_deref(), "ymin")?;
            let hi = read_edge(data, aes.ymax.as_deref(), "ymax")?;
            Ok(ribbon(&x, &lo, &hi, &panel, &group))
        }
        _ => {
            let y = read_column(data, aes.y.as_deref().unwrap_or("y"), "y")?;
            Ok(area(&x, &y, &panel, &group))
        }
    }
}

fn read_edge(data: &DataFrame, name: Option<&str>, what: &str) -> Result<Vec<f64>> {
    let name = name.ok_or_else(|| refuse(format!("geom_ribbon needs a mapped {what}")))?;
    read_column(data, name, what)
}

/// Row order of `GeomRibbon$setup_data` / `GeomArea$setup_data`: `order(PANEL, group, x)`, ties in
/// row order, a missing `x` last.
fn order_rows(x: &[f64], panel: &[usize], group: &[i64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| {
        panel[a].cmp(&panel[b]).then(group[a].cmp(&group[b])).then_with(|| cmp_na_last(x[a], x[b]))
    });
    idx
}

/// `geom_ribbon()`: identity stat, rows ordered, `y = ymin`.
fn ribbon(x: &[f64], lo: &[f64], hi: &[f64], panel: &[usize], group: &[i64]) -> BuiltLayer {
    let idx = order_rows(x, panel, group);
    let pick = |v: &[f64]| idx.iter().map(|&i| v[i]).collect::<Vec<f64>>();
    let ymin = pick(lo);
    BuiltLayer {
        x: pick(x),
        y: ymin.clone(),
        points: None,
        path: None,
        rects: None,
        bins: None,
        density: None,
        filled: Some(BuiltFilled {
            panel: idx.iter().map(|&i| panel[i]).collect(),
            group: idx.iter().map(|&i| group[i]).collect(),
            ymin,
            ymax: pick(hi),
            xmin: None,
            xmax: None,
            align_padding: None,
            flipped_aes: false,
        }),
    }
}

/// One row of `stat_align` output.
struct Row {
    panel: usize,
    group: i64,
    x: f64,
    y: f64,
    pad: Option<bool>,
}

/// `geom_area()`: `stat_align`, then `GeomArea$setup_data`, then `position_stack()`.
fn area(x: &[f64], y: &[f64], panel: &[usize], group: &[i64]) -> BuiltLayer {
    // Stat$compute_layer: remove_missing() drops a row whose x or y is missing.
    let keep: Vec<usize> = (0..x.len()).filter(|&i| !x[i].is_nan() && !y[i].is_nan()).collect();
    let mut panels: Vec<usize> = keep.iter().map(|&i| panel[i]).collect();
    panels.sort_unstable();
    panels.dedup();
    let mut rows: Vec<Row> = Vec::new();
    for p in panels {
        let rs: Vec<usize> = keep.iter().copied().filter(|&i| panel[i] == p).collect();
        let (px, py, pg): (Vec<f64>, Vec<f64>, Vec<i64>) = (
            rs.iter().map(|&i| x[i]).collect(),
            rs.iter().map(|&i| y[i]).collect(),
            rs.iter().map(|&i| group[i]).collect(),
        );
        rows.extend(align_panel(&px, &py, &pg).into_iter().map(|(g, rx, ry, pad)| Row {
            panel: p,
            group: g,
            x: rx,
            y: ry,
            pad,
        }));
    }
    let rx: Vec<f64> = rows.iter().map(|r| r.x).collect();
    let (rp, rg): (Vec<usize>, Vec<i64>) = rows.iter().map(|r| (r.panel, r.group)).unzip();
    // GeomArea$setup_data
    let idx = order_rows(&rx, &rp, &rg);
    let ran = rows.iter().any(|r| r.pad.is_some());
    let mut by_panel: Vec<Vec<BarRow>> = Vec::new();
    let mut slot: Vec<(usize, usize)> = Vec::new(); // (panel slot, index in it) per sorted row
    let mut seen: Vec<usize> = Vec::new();
    for &i in &idx {
        let r = &rows[i];
        let s = seen.iter().position(|&p| p == r.panel).unwrap_or_else(|| {
            seen.push(r.panel);
            by_panel.push(Vec::new());
            seen.len() - 1
        });
        let g = u32::try_from(r.group.saturating_add(2)).unwrap_or(u32::MAX);
        by_panel[s].push(BarRow {
            x: r.x,
            y: r.y,
            ymin: 0.0,
            ymax: r.y,
            xmin: r.x,
            xmax: r.x,
            width: 0.0,
            group: g,
            count: None,
            prop: None,
        });
        slot.push((s, by_panel[s].len() - 1));
    }
    let mut seen_x = std::collections::HashSet::new();
    let stacked = by_panel.iter().flatten().any(|b| !seen_x.insert((b.x + 0.0).to_bits()));
    apply_panels(&mut by_panel, Position::stack());
    let out: Vec<&BarRow> = slot.iter().map(|&(s, k)| &by_panel[s][k]).collect();
    BuiltLayer {
        x: out.iter().map(|b| b.x).collect(),
        y: out.iter().map(|b| b.y).collect(),
        points: None,
        path: None,
        rects: None,
        bins: None,
        density: None,
        filled: Some(BuiltFilled {
            panel: idx.iter().map(|&i| rows[i].panel).collect(),
            group: idx.iter().map(|&i| rows[i].group).collect(),
            ymin: out.iter().map(|b| b.ymin).collect(),
            ymax: out.iter().map(|b| b.ymax).collect(),
            xmin: stacked.then(|| out.iter().map(|b| b.xmin).collect()),
            xmax: stacked.then(|| out.iter().map(|b| b.xmax).collect()),
            align_padding: ran.then(|| idx.iter().map(|&i| rows[i].pad).collect()),
            flipped_aes: false,
        }),
    }
}

/// `StatAlign$compute_panel` over one panel's rows (layer row order, no missing x or y). Returns
/// `(group, x, y, align_padding)` rows: the panel unchanged when it holds one group.
fn align_panel(x: &[f64], y: &[f64], group: &[i64]) -> Vec<(i64, f64, f64, Option<bool>)> {
    let mut groups: Vec<i64> = group.to_vec();
    groups.sort_unstable();
    groups.dedup();
    if groups.len() < 2 {
        return (0..x.len()).map(|i| (group[i], x[i], y[i], None)).collect();
    }
    let cross = crossings(x, y, group);
    let mut loc: Vec<f64> = x.iter().copied().chain(cross).filter(|v| !v.is_nan()).collect();
    sort_unique(&mut loc);
    let range = match (loc.first(), loc.last()) {
        (Some(lo), Some(hi)) => hi - lo,
        _ => return Vec::new(),
    };
    let min_diff = loc.windows(2).map(|w| w[1] - w[0]).fold(f64::INFINITY, f64::min);
    let adjust = (range * 0.001).min(min_diff / 3.0);
    let mut all: Vec<f64> = loc.iter().flat_map(|v| [v - adjust, *v, v + adjust]).collect();
    sort_unique(&mut all);
    let mut out = Vec::new();
    for g in groups {
        let rows: Vec<usize> = (0..x.len()).filter(|&i| group[i] == g).collect();
        out.extend(align_group(g, &rows, x, y, &all, adjust));
    }
    out
}

/// The x where each group's line crosses zero between two consecutive rows (`vec_unrep` runs of
/// `(group, y < 0)`, every run end that is not also a group end).
fn crossings(x: &[f64], y: &[f64], group: &[i64]) -> Vec<f64> {
    // runs of (group, y < 0): (group, end row index, inclusive)
    let mut runs: Vec<(i64, usize)> = Vec::new();
    for i in 0..x.len() {
        let key = (group[i], y[i] < 0.0);
        let same = i > 0 && (group[i - 1], y[i - 1] < 0.0) == key;
        if same {
            if let Some(last) = runs.last_mut() {
                last.1 = i;
            }
        } else {
            runs.push((key.0, i));
        }
    }
    let mut out = Vec::new();
    for (k, &(g, end)) in runs.iter().enumerate() {
        let group_end = runs.get(k + 1).map_or(true, |n| n.0 != g);
        if group_end || end + 1 >= x.len() {
            continue;
        }
        let p = end;
        out.push((-y[p]) * (x[p + 1] - x[p]) / (y[p + 1] - y[p]) + x[p]);
    }
    out
}

fn sort_unique(v: &mut Vec<f64>) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v.dedup_by(|a, b| *a == *b);
}

/// `StatAlign$compute_group` for one group's rows.
fn align_group(
    g: i64,
    rows: &[usize],
    x: &[f64],
    y: &[f64],
    loc: &[f64],
    adjust: f64,
) -> Vec<(i64, f64, f64, Option<bool>)> {
    let mut pts: Vec<(f64, f64)> = rows.iter().map(|&i| (x[i], y[i])).collect();
    let first_x = pts.first().map(|p| p.0);
    if pts.iter().all(|p| Some(p.0) == first_x) {
        return Vec::new(); // is_unique(x): an empty frame, the group vanishes
    }
    let mut xs: Vec<f64> = pts.iter().map(|p| p.0).collect();
    sort_unique(&mut xs);
    if xs.len() < pts.len() {
        // dapply(data, "x"): ascending x, first and last row of a repeated x, first x -= adjust
        let mut kept = Vec::new();
        for v in xs {
            let same: Vec<(f64, f64)> = pts.iter().copied().filter(|p| p.0 == v).collect();
            if same.len() == 1 {
                kept.push(same[0]);
            } else if let (Some(f), Some(l)) = (same.first(), same.last()) {
                kept.push((f.0 - adjust, f.1));
                kept.push(*l);
            }
        }
        pts = kept;
    }
    let ys: Vec<Option<f64>> = loc.iter().map(|&v| approx(&pts, v)).collect();
    let (kx, ky): (Vec<f64>, Vec<f64>) =
        loc.iter().zip(&ys).filter_map(|(l, v)| v.map(|v| (*l, v))).unzip();
    let (Some(lo), Some(hi)) = (
        kx.iter().copied().fold(None, |a: Option<f64>, v| Some(a.map_or(v, |a| a.min(v)))),
        kx.iter().copied().fold(None, |a: Option<f64>, v| Some(a.map_or(v, |a| a.max(v)))),
    ) else {
        return Vec::new();
    };
    let n = kx.len();
    let mut out = vec![(g, lo - adjust, 0.0, Some(true))];
    out.extend((0..n).map(|k| (g, kx[k], ky[k], Some(false))));
    out.push((g, hi + adjust, 0.0, Some(true)));
    out
}

/// `stats::approxfun(x, y)(v)` with `rule = 1`, `ties = mean`, linear: `None` is `NA`.
fn approx(pts: &[(f64, f64)], v: f64) -> Option<f64> {
    let mut sorted = pts.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    let mut i = 0;
    while i < sorted.len() {
        let mut j = i;
        while j + 1 < sorted.len() && sorted[j + 1].0 == sorted[i].0 {
            j += 1;
        }
        let tie = &sorted[i..=j];
        xs.push(sorted[i].0);
        ys.push(tie.iter().map(|p| p.1).sum::<f64>() / tie.len() as f64);
        i = j + 1;
    }
    let n = xs.len();
    if n == 0 || v.is_nan() || v < xs[0] || v > xs[n - 1] {
        return None;
    }
    let (mut lo, mut hi) = (0, n - 1);
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if v < xs[mid] {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    if v == xs[hi] {
        return Some(ys[hi]);
    }
    if v == xs[lo] {
        return Some(ys[lo]);
    }
    Some(ys[lo] + (ys[hi] - ys[lo]) * ((v - xs[lo]) / (xs[hi] - xs[lo])))
}

/// The finite extents of a built filled layer: `(x_lo, x_hi, y_lo, y_hi)` over every position
/// column ggplot2 trains the scales on, `None` for an axis with no finite value.
pub(super) fn extents(l: &BuiltLayer) -> (Option<(f64, f64)>, Option<(f64, f64)>) {
    let Some(f) = &l.filled else { return (None, None) };
    let range = |cols: &[&[f64]]| {
        let mut r: Option<(f64, f64)> = None;
        for v in cols.iter().flat_map(|c| c.iter()).filter(|v| v.is_finite()) {
            r = Some(r.map_or((*v, *v), |(a, b)| (a.min(*v), b.max(*v))));
        }
        r
    };
    let empty: &[f64] = &[];
    let xmin = f.xmin.as_deref().unwrap_or(empty);
    let xmax = f.xmax.as_deref().unwrap_or(empty);
    (range(&[&l.x, xmin, xmax]), range(&[&l.y, &f.ymin, &f.ymax]))
}
