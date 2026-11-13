//! Path-family and reference-line layers: `geom_line`, `geom_path`, `geom_step`, `geom_hline`,
//! `geom_vline` and `geom_abline` as `ggplot_build()` in ggplot2 4.0.3 resolves them
//! (PLOT-GG-001, paiml/aprender#6909). Proven against R by `ggplot2_lines_oracle.rs`.
//!
//! # Scope (what this module promises, and what it refuses with a typed error)
//!
//! * Continuous numeric x and y. A text x or y column, or an infinite value, is an
//!   [`Error::Rendering`]. A missing column is [`Error::UnknownColumn`].
//! * Rows keep `NaN` where a cell is `NA`/`NaN`/null or a column is shorter than its siblings,
//!   as `ggplot_build()` keeps them. `geom_step` refuses such rows (its stair vertices are
//!   undefined there).
//! * Order. `geom_line` orders rows by (panel, group, x) with `NaN` x last and ties in data
//!   order (ggplot2's `setup_data`). `geom_path` and `geom_step` keep data order.
//! * Groups. Only an explicit `group` aesthetic is supported: a column that is all numbers or all
//!   text with no missing cell; ids are the 1-based rank of the sorted distinct value (C
//!   collation for text). Without it every row is group -1. Any other data-mapped aesthetic
//!   (`color`, `size`, `shape`, `alpha`, `fill`, `label`) would make ggplot2 split groups and
//!   train a colour scale; that is refused rather than answered wrongly.
//! * Facets. [`Facet::None`] and [`Facet::Wrap`] over an all-text column, whose levels sort in C
//!   collation as in R. A layer's own data is matched to the plot's panels by level. Any other
//!   facet, or a numeric facet column, is refused.
//! * Reference lines train a scale like ggplot2: `geom_hline` trains y with its intercept,
//!   `geom_vline` trains x, `geom_abline` trains neither. None of them inherits the plot's
//!   aesthetics (`inherit.aes = FALSE`). Each is replicated once per panel.
//! * Coordinate systems. Only the default `coord_cartesian()` (no limits, no flip) is supported:
//!   `layout()` applies no coordinate system, so `coord_flip`, `coord_cartesian(xlim, ylim)`,
//!   `coord_polar` and `coord_fixed` are an [`Error::UnsupportedCoord`] while one of these layers
//!   is present (`require_default_coord`).
//! * Stats. Only the identity stat (`stat = None` or `Stat::Identity`) is supported, on every
//!   layer of a plot that has one of these layers: `layout()` computes no stat, and ggplot2 would
//!   (`stat_smooth`, `stat_bin`, `stat_count`, `stat_density`, `stat_boxplot`, `stat_bin_2d`,
//!   `stat_summary`). Any other stat is an [`Error::UnsupportedStat`] (`require_identity_stats`).
//! * Companion layers. Besides these six geoms only `geom_point` is supported in the same plot.
//!   Any other geom is an [`Error::UnsupportedLayer`]: `geom_bar`, `geom_histogram`,
//!   `geom_area`, `geom_tile`, `geom_boxplot`, `geom_violin` and `geom_smooth` change ggplot2's
//!   trained scales or layer data through their stat or setup (`stat_align`, bar/tile widths, a
//!   zero baseline), and `geom_text` is not proven here.
//! * Position adjustments, `na.rm` and `orientation` have no field in the builder, so every
//!   layer is `position = "identity"`, `na.rm = FALSE` and `orientation = NA` as in ggplot2's
//!   defaults; `ggplot_build()` keeps `NA` rows either way (they are removed only when drawn).
//! * Degenerate scales (no finite value on an axis, or a single distinct value) are an
//!   [`Error::ScaleDomain`] while a path or reference layer is present: ggplot2 widens those
//!   ranges in a way the shared scale code does not yet reproduce.
//!
//! Colour and alpha come from `Aes::color_value` / `Aes::alpha_value`; `linewidth` is ggplot2's
//! default 0.5 and `linetype` 1.

use std::cmp::Ordering;

use crate::color::Rgba;
use crate::error::{Error, Result};

use super::aes::Aes;
use super::build::{BuiltLayer, PositionScale};
use super::coord::Coord;
use super::data::{DataFrame, DataValue};
use super::facet::{panels, Facet, Panel};
use super::geom::{GeomType, StepDirection};
use super::ggplot::Layer;
use super::stat::Stat;

/// `GeomPath$default_aes` / `GeomHline$default_aes` in ggplot2 4.0.3 (the stroke columns).
#[derive(Debug, Clone, PartialEq)]
pub struct StrokeAes {
    /// `colour = "black"` unless `color_value` is set.
    pub colour: Rgba,
    /// `linewidth = 0.5`.
    pub linewidth: f64,
    /// `linetype = 1` (solid).
    pub linetype: u8,
    /// `alpha = NA` unless `alpha_value` is set.
    pub alpha: Option<f64>,
}

/// Which reference line a layer is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceKind {
    /// `geom_hline`: `y = yintercept`.
    Horizontal,
    /// `geom_vline`: `x = xintercept`.
    Vertical,
    /// `geom_abline`: `y = slope * x + intercept`.
    Diagonal,
}

/// The per-panel rows of a reference layer (one row per panel, as `layer_data()` has them).
#[derive(Debug, Clone, PartialEq)]
pub struct Reference {
    /// Which line.
    pub kind: ReferenceKind,
    /// `yintercept`, `xintercept` or `intercept`, per panel.
    pub intercept: Vec<f64>,
    /// `slope` per panel; empty unless [`ReferenceKind::Diagonal`].
    pub slope: Vec<f64>,
}

/// The vertices `geom_step` draws: `stairstep()` per panel, then per group.
#[derive(Debug, Clone, PartialEq)]
pub struct Stairs {
    /// The direction these were made for.
    pub direction: StepDirection,
    /// Vertex x.
    pub x: Vec<f64>,
    /// Vertex y.
    pub y: Vec<f64>,
    /// 0-based panel of each vertex.
    pub panel: Vec<usize>,
    /// Group of each vertex.
    pub group: Vec<i64>,
}

/// What a path-family or reference layer adds to a [`BuiltLayer`].
#[derive(Debug, Clone, PartialEq)]
pub struct BuiltPath {
    /// 0-based panel of each row.
    pub panel: Vec<usize>,
    /// Group id of each row (-1 when ungrouped).
    pub group: Vec<i64>,
    /// The layer's stroke aesthetics (one value for every row).
    pub aes: StrokeAes,
    /// Present for `geom_step`.
    pub stairs: Option<Stairs>,
    /// Present for `geom_hline`, `geom_vline` and `geom_abline`.
    pub reference: Option<Reference>,
}

/// Is this one of the geoms this module builds?
pub(super) fn is_path_family(g: &GeomType) -> bool {
    matches!(
        g,
        GeomType::Line { .. }
            | GeomType::Path
            | GeomType::Step { .. }
            | GeomType::Hline { .. }
            | GeomType::Vline { .. }
            | GeomType::Abline { .. }
    )
}

fn is_reference(g: &GeomType) -> bool {
    matches!(g, GeomType::Hline { .. } | GeomType::Vline { .. } | GeomType::Abline { .. })
}

fn refuse(msg: impl Into<String>) -> Error {
    Error::Rendering(msg.into())
}

/// Refuse any coordinate system but the default `coord_cartesian()` while a path-family or
/// reference layer is present. `layout()` applies no coordinate system (limits, flip, polar and
/// fixed all change ggplot2's `panel_params`), so answering with the default numbers would be
/// silently wrong.
pub(super) fn require_default_coord(layers: &[Layer], coord: &Coord) -> Result<()> {
    if !layers.iter().any(|l| is_path_family(&l.geom.geom_type)) {
        return Ok(());
    }
    match coord {
        Coord::Cartesian { xlim: None, ylim: None, flip: false } => Ok(()),
        Coord::Cartesian { flip: true, .. } => Err(Error::UnsupportedCoord(
            "coord_flip with line, path, step or reference layers: layout() does not flip",
        )),
        Coord::Cartesian { .. } => Err(Error::UnsupportedCoord(
            "coord_cartesian limits with line, path, step or reference layers: layout() does not apply them",
        )),
        Coord::Polar { .. } => Err(Error::UnsupportedCoord(
            "coord_polar with line, path, step or reference layers",
        )),
        Coord::Fixed { .. } => Err(Error::UnsupportedCoord(
            "coord_fixed with line, path, step or reference layers",
        )),
    }
}

/// Refuse, while a path-family or reference layer is present, every layer whose stat `layout()`
/// would silently skip, and every companion layer other than `geom_point` (its ggplot2 stat or
/// setup changes the shared scales). Answering with the raw data would be silently wrong.
/// A plot with a `geom_bar()`/`geom_col()` layer over a text x never reaches this check: `layout()`
/// routes it to the bar layout, which admits `geom_hline`/`geom_vline` beside the bars with guards of
/// its own.
pub(super) fn require_identity_stats(layers: &[Layer]) -> Result<()> {
    if !layers.iter().any(|l| is_path_family(&l.geom.geom_type)) {
        return Ok(());
    }
    for l in layers {
        let g = &l.geom.geom_type;
        if !is_path_family(g) && !matches!(g, GeomType::Point { .. }) {
            return Err(Error::UnsupportedLayer(
                "only geom_point may share a plot with line, path, step or reference layers: \
                 layout() does not resolve other geoms with them (their ggplot2 stat or setup \
                 can change the shared scales)",
            ));
        }
        if let Some(msg) = l.geom.stat.as_ref().and_then(unsupported_stat) {
            return Err(Error::UnsupportedStat(msg));
        }
    }
    Ok(())
}

/// The refusal for a stat `layout()` does not compute; `None` for the identity stat.
fn unsupported_stat(s: &Stat) -> Option<&'static str> {
    let msg = match s {
        Stat::Identity => return None,
        Stat::Count => "stat_count with line, path, step or reference layers: layout() computes no stat",
        Stat::Bin { .. } => "stat_bin with line, path, step or reference layers: layout() computes no stat",
        Stat::Density => {
            "stat_density with line, path, step or reference layers: layout() computes no stat"
        }
        Stat::Boxplot => {
            "stat_boxplot with line, path, step or reference layers: layout() computes no stat"
        }
        Stat::Smooth => {
            "stat_smooth with line, path, step or reference layers: layout() computes no stat"
        }
        Stat::Bin2d { .. } => {
            "stat_bin_2d with line, path, step or reference layers: layout() computes no stat"
        }
        Stat::Sum => {
            "stat_summary(fun = sum) with line, path, step or reference layers: layout() computes no stat"
        }
        Stat::Mean => {
            "stat_summary(fun = mean) with line, path, step or reference layers: layout() computes no stat"
        }
    };
    Some(msg)
}

/// Refuse a scale that ggplot2 would widen differently from the shared scale code.
pub(super) fn require_trained(
    layers: &[Layer],
    x: &PositionScale,
    y: &PositionScale,
) -> Result<()> {
    if !layers.iter().any(|l| is_path_family(&l.geom.geom_type)) {
        return Ok(());
    }
    for (axis, s) in [("x", x), ("y", y)] {
        if s.limits.0 >= s.limits.1 {
            return Err(Error::ScaleDomain(format!(
                "{axis} scale has {} (limits {:?}); a degenerate range is not supported with \
                 line, path, step or reference layers",
                if s.limits.0 > s.limits.1 { "no finite value" } else { "a single value" },
                s.limits
            )));
        }
    }
    Ok(())
}

/// Build one path-family or reference layer.
pub(super) fn build_layer(
    layer: &Layer,
    plot_aes: &Aes,
    plot_data: &DataFrame,
    facet: &Facet,
) -> Result<BuiltLayer> {
    let geom = &layer.geom.geom_type;
    let reference = is_reference(geom);
    let aes = if reference { layer.aes.clone() } else { plot_aes.merge(&layer.aes) };
    let stroke = stroke_aes(geom, &aes, reference)?;
    let global = global_panels(facet, plot_data)?;

    if reference {
        let n = global.len();
        let (kind, intercept, slope) =
            reference_parts(geom).ok_or_else(|| refuse("not a reference geom"))?;
        if !intercept.is_finite() || slope.is_some_and(|s| !s.is_finite()) {
            return Err(refuse("a reference line needs a finite intercept and slope"));
        }
        let path = BuiltPath {
            panel: (0..n).collect(),
            group: vec![-1; n],
            aes: stroke,
            stairs: None,
            reference: Some(Reference {
                kind,
                intercept: vec![intercept; n],
                slope: slope.map_or_else(Vec::new, |s| vec![s; n]),
            }),
        };
        return Ok(BuiltLayer { x: vec![], y: vec![], points: None, bars: None, path: Some(path) });
    }

    let own = layer.data.is_some();
    let data = layer.data.as_ref().unwrap_or(plot_data);
    let panel = row_panels(facet, &global, data, own)?;
    let group = group_ids(&aes, data)?;
    let mut x = read_column(data, aes.x.as_deref().unwrap_or("x"), "x")?;
    let mut y = read_column(data, aes.y.as_deref().unwrap_or("y"), "y")?;
    let (mut panel, mut group) = (panel, group);

    let mut stairs = None;
    match geom {
        GeomType::Line { .. } => {
            let mut idx: Vec<usize> = (0..x.len()).collect();
            idx.sort_by(|&a, &b| {
                panel[a]
                    .cmp(&panel[b])
                    .then(group[a].cmp(&group[b]))
                    .then_with(|| cmp_na_last(x[a], x[b]))
            });
            x = idx.iter().map(|&i| x[i]).collect();
            y = idx.iter().map(|&i| y[i]).collect();
            panel = idx.iter().map(|&i| panel[i]).collect();
            group = idx.iter().map(|&i| group[i]).collect();
        }
        GeomType::Step { direction } => {
            if x.iter().chain(&y).any(|v| v.is_nan()) {
                return Err(refuse("geom_step does not support NA or NaN positions"));
            }
            stairs = Some(make_stairs(*direction, &x, &y, &panel, &group));
        }
        _ => {}
    }
    let path = BuiltPath { panel, group, aes: stroke, stairs, reference: None };
    Ok(BuiltLayer { x, y, points: None, bars: None, path: Some(path) })
}

fn stroke_aes(geom: &GeomType, aes: &Aes, reference: bool) -> Result<StrokeAes> {
    if aes.color.is_some()
        || aes.size.is_some()
        || aes.shape.is_some()
        || aes.alpha.is_some()
        || aes.fill.is_some()
        || aes.label.is_some()
    {
        return Err(refuse(
            "a data-mapped colour, size, shape, alpha, fill or label is not supported on a \
             line, path, step or reference layer",
        ));
    }
    if aes.size_value.is_some() {
        return Err(refuse(
            "a fixed size is not supported on a line, path, step or reference layer",
        ));
    }
    if reference && (aes.x.is_some() || aes.y.is_some() || aes.group.is_some()) {
        return Err(refuse("a reference line takes no x, y or group aesthetic"));
    }
    if matches!(geom, GeomType::Line { width } if (*width - 1.0).abs() > f32::EPSILON) {
        return Err(refuse("geom_line width is not ggplot2's linewidth; leave it at its default"));
    }
    Ok(StrokeAes {
        colour: aes.color_value.unwrap_or(Rgba::BLACK),
        linewidth: 0.5,
        linetype: 1,
        alpha: aes.alpha_value.map(f64::from),
    })
}

/// A facet column must be all text so its levels sort as R's C-collation `factor()` does.
pub(super) fn require_text_facet(data: &DataFrame, var: &str) -> Result<()> {
    let col = data
        .get(var)
        .ok_or_else(|| Error::UnknownColumn(format!("no column {var:?} to facet by")))?;
    if col.len() < data.nrow() || !col.iter().all(|c| matches!(c, DataValue::Text(_))) {
        return Err(refuse(format!(
            "facet column {var:?} must be text with no missing cell for path, reference and bar layers"
        )));
    }
    Ok(())
}

fn global_panels(facet: &Facet, plot_data: &DataFrame) -> Result<Vec<Panel>> {
    match facet {
        Facet::None => Ok(vec![Panel { levels: vec![], row: 0, col: 0, rows: vec![] }]),
        Facet::Wrap { var, .. } => {
            require_text_facet(plot_data, var)?;
            panels(facet, plot_data)
        }
        _ => Err(refuse("path and reference layers support facet_wrap or no facet only")),
    }
}

/// The 0-based panel index of every row of `data`.
fn row_panels(facet: &Facet, global: &[Panel], data: &DataFrame, own: bool) -> Result<Vec<usize>> {
    let mut out = vec![0; data.nrow()];
    let Facet::Wrap { var, .. } = facet else { return Ok(out) };
    let mine;
    let source: &[Panel] = if own {
        require_text_facet(data, var)?;
        mine = panels(facet, data)?;
        &mine
    } else {
        global
    };
    for q in source {
        let pi = global.iter().position(|p| p.levels == q.levels).ok_or_else(|| {
            refuse(format!("layer data has facet level {:?} that the plot data lacks", q.levels))
        })?;
        for &r in &q.rows {
            out[r] = pi;
        }
    }
    Ok(out)
}

/// One column as f64, one entry per row of `data`: `NA`, null and missing cells are `NaN`.
fn read_column(data: &DataFrame, name: &str, what: &str) -> Result<Vec<f64>> {
    let col =
        data.get(name).ok_or_else(|| Error::UnknownColumn(format!("no {what} column {name:?}")))?;
    (0..data.nrow())
        .map(|i| match col.get(i) {
            Some(DataValue::Number(v)) if v.is_infinite() => {
                Err(refuse(format!("{what} column {name:?} holds an infinite value")))
            }
            Some(DataValue::Number(v)) => Ok(f64::from(*v)),
            Some(DataValue::Null) | None => Ok(f64::NAN),
            Some(DataValue::Text(_)) => {
                Err(refuse(format!("{what} column {name:?} is text; only continuous positions")))
            }
        })
        .collect()
}

/// ggplot2's group ids: 1-based rank of the sorted distinct value, or -1 when unmapped.
fn group_ids(aes: &Aes, data: &DataFrame) -> Result<Vec<i64>> {
    let n = data.nrow();
    let Some(name) = aes.group.as_deref() else { return Ok(vec![-1; n]) };
    let col =
        data.get(name).ok_or_else(|| Error::UnknownColumn(format!("no group column {name:?}")))?;
    let bad = || refuse(format!("group column {name:?} must be all numbers or all text, no gaps"));
    if col.len() < n {
        return Err(bad());
    }
    let rank = |i: usize| i64::try_from(i).map_or(i64::MAX, |v| v + 1);
    if col.iter().all(|c| matches!(c, DataValue::Number(v) if v.is_finite())) {
        let vals: Vec<f32> = col.iter().filter_map(|c| c.as_f32()).map(|v| v + 0.0).collect();
        let mut uniq = vals.clone();
        uniq.sort_by(f32::total_cmp);
        uniq.dedup();
        return vals
            .iter()
            .map(|v| uniq.binary_search_by(|p| p.total_cmp(v)).map(rank).map_err(|_| bad()))
            .collect();
    }
    if col.iter().all(|c| matches!(c, DataValue::Text(_))) {
        let vals: Vec<&str> = col.iter().filter_map(|c| c.as_str()).collect();
        let mut uniq = vals.clone();
        uniq.sort_unstable();
        uniq.dedup();
        return vals.iter().map(|v| uniq.binary_search(v).map(rank).map_err(|_| bad())).collect();
    }
    Err(bad())
}

/// Ascending with `NaN` last and equal values tied, as R's `order()`.
fn cmp_na_last(a: f64, b: f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}

/// `GeomStep$draw_panel`: `stairstep()` on every group of every panel, panels then groups ascending.
fn make_stairs(
    direction: StepDirection,
    x: &[f64],
    y: &[f64],
    panel: &[usize],
    group: &[i64],
) -> Stairs {
    let mut keys: Vec<(usize, i64)> = panel.iter().copied().zip(group.iter().copied()).collect();
    keys.sort_unstable();
    keys.dedup();
    let mut out = Stairs { direction, x: vec![], y: vec![], panel: vec![], group: vec![] };
    for (p, g) in keys {
        let mut rows: Vec<usize> =
            (0..x.len()).filter(|&i| panel[i] == p && group[i] == g).collect();
        rows.sort_by(|&a, &b| cmp_na_last(x[a], x[b]));
        let n = rows.len();
        if n <= 1 {
            continue;
        }
        let (sx, sy): (Vec<f64>, Vec<f64>) = rows.iter().map(|&i| (x[i], y[i])).unzip();
        let verts: Vec<(f64, f64)> = match direction {
            StepDirection::Hv => (0..2 * n - 1).map(|k| (sx[(k + 1) / 2], sy[k / 2])).collect(),
            StepDirection::Vh => (0..2 * n - 1).map(|k| (sx[k / 2], sy[(k + 1) / 2])).collect(),
            StepDirection::Mid => (0..2 * n)
                .map(|k| {
                    let xv = if k == 0 {
                        sx[0]
                    } else if k == 2 * n - 1 {
                        sx[n - 1]
                    } else {
                        let j = (k - 1) / 2;
                        sx[j] + (sx[j + 1] - sx[j]) / 2.0
                    };
                    (xv, sy[k / 2])
                })
                .collect(),
        };
        for (vx, vy) in verts {
            out.x.push(vx);
            out.y.push(vy);
            out.panel.push(p);
            out.group.push(g);
        }
    }
    out
}

/// `(kind, intercept, slope)` of a reference geom; `None` for any other geom.
fn reference_parts(geom: &GeomType) -> Option<(ReferenceKind, f64, Option<f64>)> {
    match geom {
        GeomType::Hline { yintercept } => {
            Some((ReferenceKind::Horizontal, f64::from(*yintercept), None))
        }
        GeomType::Vline { xintercept } => {
            Some((ReferenceKind::Vertical, f64::from(*xintercept), None))
        }
        GeomType::Abline { slope, intercept } => {
            Some((ReferenceKind::Diagonal, f64::from(*intercept), Some(f64::from(*slope))))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::geom::Geom;
    use super::*;

    #[test]
    fn only_reference_geoms_have_reference_parts() {
        assert!(reference_parts(&Geom::hline(1.0).geom_type).is_some());
        assert!(reference_parts(&Geom::vline(1.0).geom_type).is_some());
        assert!(reference_parts(&Geom::abline(2.0, 1.0).geom_type).is_some());
        assert!(reference_parts(&Geom::line().geom_type).is_none());
        assert!(reference_parts(&Geom::step().geom_type).is_none());
    }
}
