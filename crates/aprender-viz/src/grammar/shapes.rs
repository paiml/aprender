//! `geom_segment`, `geom_polygon`, `geom_raster`, `geom_smooth(method = "lm")` and the typed-error
//! scope of `geom_violin`, as `ggplot_build()` in ggplot2 4.0.3 resolves them (PLOT-GG-001,
//! paiml/aprender#6909). Proven against R by `ggplot2_shapes_oracle.rs`.
//!
//! # Scope (what this module promises, and what it refuses with a typed error)
//!
//! * Continuous numeric positions. A text position column, an infinite value, a missing column
//!   ([`Error::UnknownColumn`]) and a column shorter than its siblings are an [`Error::Rendering`]
//!   (never padded with `NA`). A frame with no rows is refused. `NA`/`NaN`/null cells are kept
//!   where `ggplot_build()` keeps them (segment, polygon, raster) and removed by the smooth stat
//!   as `stat_smooth` removes them.
//! * Rows keep data order (segment, polygon, raster; ggplot2 does not sort them), including under
//!   `facet_wrap`.
//! * Groups, facets: as `path.rs` (an explicit all-number or all-text `group` column; ids are the
//!   1-based rank of the sorted distinct value, -1 without one; [`Facet::None`] and
//!   [`Facet::Wrap`] over an all-text column; a layer's own data is matched to the plot's panels).
//!   Any other facet is refused.
//! * Aesthetics. A data-mapped `colour`, `size`, `shape`, `alpha`, `fill` or `label`, and a fixed
//!   `size`, are refused. A fixed `colour` and `alpha` are read from `Aes::color_value` /
//!   `Aes::alpha_value`.
//! * Stat. Only the geom's own default stat: identity for segment, polygon and raster,
//!   `stat_smooth` for smooth. Any other `Stat` set on the geom is refused.
//! * Coordinate systems. Only the default `coord_cartesian()`; any other is an
//!   [`Error::UnsupportedCoord`] (`path::require_default_coord`).
//! * Degenerate scales are an [`Error::ScaleDomain`] (`path::require_trained`).
//! * `geom_segment` needs `xend` and `yend`. Both train their scale like ggplot2.
//! * `geom_raster` is refused when its `x` or `y` values are unevenly spaced (R warns and shifts
//!   the pixels). Evenly spaced values, with gaps, give the width and height R derives.
//! * `geom_smooth` is `method = "lm"`, formula `y ~ x`, `n = 80`, `level = 0.95`, weights 1, no
//!   `fullrange`. `SmoothMethod::Loess` (R's `"auto"`) is refused. `x` is a continuous (double)
//!   column: R builds the grid from the sorted distinct values for an *integer* column, which
//!   this crate cannot express. A group with fewer than two distinct finite `x` (R drops it
//!   silently), and `se = TRUE` with fewer than three finite rows in a group (R returns `NaN`),
//!   are refused.
//! * `geom_violin` is refused: `stat_ydensity` (`bw.nrd0`, 512 density points per group, trimming,
//!   `scale = "area"`) is not implemented here. The refusal is the row's whole claim.
//! * Tick labels are claimed only for non-zero data magnitude from 2^-8 to 10^4 (#6958).

use std::collections::BTreeMap;

use crate::color::Rgba;
use crate::error::{Error, Result};

use super::aes::Aes;
use super::build::BuiltLayer;
use super::data::{DataFrame, DataValue};
use super::facet::Facet;
use super::geom::{GeomType, SmoothMethod};
use super::ggplot::Layer;
use super::path::{global_panels, group_ids, refuse, row_panels};
use super::stat::Stat;
use super::tdist::qt_upper;

/// `GeomPolygon` / `GeomRaster` `fill = "grey20"`, as ggplot2 4.0.3 resolves it (`#333333FF`).
const GREY20: Rgba = Rgba::new(51, 51, 51, 255);
/// `GeomSmooth$default_aes`: `colour = "#3366FF"`.
const SMOOTH_COLOUR: Rgba = Rgba::new(0x33, 0x66, 0xFF, 255);
/// `GeomSmooth$default_aes`: `fill = "grey60"` (`#999999FF`).
const SMOOTH_FILL: Rgba = Rgba::new(0x99, 0x99, 0x99, 255);
/// `stat_smooth(n = 80)`.
const SMOOTH_N: usize = 80;
/// `stat_smooth(level = 0.95)`: each tail is `(1 - level) / 2`.
const SMOOTH_TAIL: f64 = 0.025;

/// Which geom a [`BuiltShape`] belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeKind {
    /// `geom_segment`.
    Segment,
    /// `geom_polygon`.
    Polygon,
    /// `geom_raster`.
    Raster,
    /// `geom_smooth(method = "lm")`.
    Smooth,
}

/// What a segment, polygon, raster or smooth layer adds to a [`BuiltLayer`] (whose `x`/`y` hold
/// the layer's `x`/`y` columns).
#[derive(Debug, Clone, PartialEq)]
pub struct BuiltShape {
    /// Which geom.
    pub kind: ShapeKind,
    /// 0-based panel of each row.
    pub panel: Vec<usize>,
    /// Group id of each row (-1 when ungrouped).
    pub group: Vec<i64>,
    /// The other numeric columns, by ggplot2's name: `xend`, `yend` (segment); `xmin`, `xmax`,
    /// `ymin`, `ymax` (raster); `ymin`, `ymax`, `se` (smooth with `se = TRUE`).
    pub cols: BTreeMap<&'static str, Vec<f64>>,
    /// `colour`; `None` is R's `NA`.
    pub colour: Option<Rgba>,
    /// `fill`; `None` when the geom has none.
    pub fill: Option<Rgba>,
    /// `linewidth`; `None` when the geom has none.
    pub linewidth: Option<f64>,
    /// `linetype`; `None` when the geom has none.
    pub linetype: Option<u8>,
    /// `alpha`; `None` is R's `NA`.
    pub alpha: Option<f64>,
}

/// Is this one of the geoms this module builds or refuses?
pub(super) fn is_shape_geom(g: &GeomType) -> bool {
    matches!(
        g,
        GeomType::Segment
            | GeomType::Polygon
            | GeomType::Raster
            | GeomType::Smooth { .. }
            | GeomType::Violin
    )
}

/// The x-family and y-family column names that train their position scale.
const X_TRAINING: [&str; 4] = ["x", "xend", "xmin", "xmax"];
const Y_TRAINING: [&str; 4] = ["y", "yend", "ymin", "ymax"];

/// The finite range of everything a built shape layer trains the x and y scales with (ggplot2
/// trains after the stat, so a smooth layer trains with its fitted curve, never its raw points).
pub(super) fn training_ranges(b: &BuiltLayer) -> Option<[Option<(f64, f64)>; 2]> {
    let s = b.shape.as_ref()?;
    let range = |base: &[f64], names: &[&str; 4]| {
        let it = base.iter().chain(
            names
                .iter()
                .filter(|n| **n != "x" && **n != "y")
                .filter_map(|n| s.cols.get(n))
                .flatten(),
        );
        it.filter(|v| v.is_finite()).fold(None, |acc: Option<(f64, f64)>, &v| match acc {
            None => Some((v, v)),
            Some((lo, hi)) => Some((lo.min(v), hi.max(v))),
        })
    };
    Some([range(&b.x, &X_TRAINING), range(&b.y, &Y_TRAINING)])
}

/// Build one layer of a geom [`is_shape_geom`] accepts.
pub(super) fn build_layer(
    layer: &Layer,
    plot_aes: &Aes,
    plot_data: &DataFrame,
    facet: &Facet,
) -> Result<BuiltLayer> {
    let geom = &layer.geom.geom_type;
    if matches!(geom, GeomType::Violin) {
        return Err(refuse(
            "geom_violin is not supported: stat_ydensity (bw.nrd0 density, 512 points per group, \
             trim, scale = \"area\") is not implemented",
        ));
    }
    require_default_stat(geom, layer.geom.stat.as_ref())?;
    let aes = plot_aes.merge(&layer.aes);
    require_plain_aes(&aes)?;
    let global = global_panels(facet, plot_data)?;
    let own = layer.data.is_some();
    let data = layer.data.as_ref().unwrap_or(plot_data);
    if data.nrow() == 0 {
        return Err(refuse("the layer's data has no rows"));
    }
    let panel = row_panels(facet, &global, data, own)?;
    let group = group_ids(&aes, data)?;
    let x = position(data, aes.x.as_deref().unwrap_or("x"), "x")?;
    let y = position(data, aes.y.as_deref().unwrap_or("y"), "y")?;
    let alpha = aes.alpha_value.map(f64::from);
    match geom {
        GeomType::Segment => {
            let xend = position(data, aes.xend.as_deref().unwrap_or("xend"), "xend")?;
            let yend = position(data, aes.yend.as_deref().unwrap_or("yend"), "yend")?;
            let cols = BTreeMap::from([("xend", xend), ("yend", yend)]);
            let shape = BuiltShape {
                kind: ShapeKind::Segment,
                panel,
                group,
                cols,
                colour: Some(aes.color_value.unwrap_or(Rgba::BLACK)),
                fill: None,
                linewidth: Some(0.5),
                linetype: Some(1),
                alpha,
            };
            Ok(BuiltLayer { x, y, points: None, path: None, shape: Some(shape) })
        }
        GeomType::Polygon => {
            let shape = BuiltShape {
                kind: ShapeKind::Polygon,
                panel,
                group,
                cols: BTreeMap::new(),
                colour: aes.color_value,
                fill: Some(GREY20),
                linewidth: Some(0.5),
                linetype: Some(1),
                alpha,
            };
            Ok(BuiltLayer { x, y, points: None, path: None, shape: Some(shape) })
        }
        GeomType::Raster => {
            let (w, h) = (pixel_size(&x, "x")?, pixel_size(&y, "y")?);
            let cols = BTreeMap::from([
                ("xmin", x.iter().map(|v| v - w * 0.5).collect()),
                ("xmax", x.iter().map(|v| v + w * 0.5).collect()),
                ("ymin", y.iter().map(|v| v - h * 0.5).collect()),
                ("ymax", y.iter().map(|v| v + h * 0.5).collect()),
            ]);
            let shape = BuiltShape {
                kind: ShapeKind::Raster,
                panel,
                group,
                cols,
                colour: None,
                fill: Some(GREY20),
                linewidth: None,
                linetype: None,
                alpha,
            };
            Ok(BuiltLayer { x, y, points: None, path: None, shape: Some(shape) })
        }
        GeomType::Smooth { method, se } => {
            if *method != SmoothMethod::Linear {
                return Err(refuse(
                    "geom_smooth supports method = \"lm\" only (formula y ~ x); loess and gam are \
                     not implemented",
                ));
            }
            smooth(&x, &y, &panel, &group, *se, aes.color_value, alpha)
        }
        _ => Err(refuse("not a segment, polygon, raster or smooth geom")),
    }
}

/// Refuse a stat other than the geom's own default.
fn require_default_stat(geom: &GeomType, stat: Option<&Stat>) -> Result<()> {
    let ok = matches!(
        (geom, stat),
        (_, None)
            | (GeomType::Smooth { .. }, Some(Stat::Smooth))
            | (GeomType::Segment | GeomType::Polygon | GeomType::Raster, Some(Stat::Identity))
    );
    if ok {
        Ok(())
    } else {
        Err(refuse(format!(
            "a stat other than the geom's default ({stat:?} on {geom:?}) changes ggplot2's numbers \
             and is not supported"
        )))
    }
}

/// Refuse every aesthetic that would make ggplot2 split groups, train a non-position scale or
/// change a number this module does not compute.
fn require_plain_aes(aes: &Aes) -> Result<()> {
    if aes.color.is_some()
        || aes.size.is_some()
        || aes.shape.is_some()
        || aes.alpha.is_some()
        || aes.fill.is_some()
        || aes.label.is_some()
    {
        return Err(refuse(
            "a data-mapped colour, size, shape, alpha, fill or label is not supported on a \
             segment, polygon, raster or smooth layer",
        ));
    }
    if aes.size_value.is_some() {
        return Err(refuse(
            "a fixed size is not supported on a segment, polygon, raster or smooth layer",
        ));
    }
    Ok(())
}

/// One position column as f64, exactly one entry per row: `NA`, null and `NaN` cells are `NaN`.
/// A short column, text and an infinite value are typed errors.
fn position(data: &DataFrame, name: &str, what: &str) -> Result<Vec<f64>> {
    let col =
        data.get(name).ok_or_else(|| Error::UnknownColumn(format!("no {what} column {name:?}")))?;
    if col.len() != data.nrow() {
        return Err(refuse(format!(
            "{what} column {name:?} has {} cells but the frame has {} rows",
            col.len(),
            data.nrow()
        )));
    }
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

/// `GeomRaster$setup_data`: the pixel width (or height) from the sorted distinct finite values.
/// Even spacing (within sqrt(eps)) gives the first difference; uneven spacing is refused.
fn pixel_size(values: &[f64], what: &str) -> Result<f64> {
    let mut u: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    u.sort_by(f64::total_cmp);
    u.dedup();
    let diff: Vec<f64> = u.windows(2).map(|w| w[1] - w[0]).collect();
    let Some(first) = diff.first().copied() else { return Ok(1.0) };
    let precision = f64::EPSILON.sqrt();
    if diff.windows(2).any(|w| (w[1] - w[0]).abs() > precision) {
        return Err(refuse(format!(
            "geom_raster {what} values are unevenly spaced; ggplot2 shifts those pixels, which \
             is not supported (use evenly spaced values)"
        )));
    }
    Ok(first)
}

/// `stat_smooth(method = "lm")` per (panel, group), in ggplot2's order: panels, then groups,
/// ascending.
fn smooth(
    x: &[f64],
    y: &[f64],
    panel: &[usize],
    group: &[i64],
    se: bool,
    colour: Option<Rgba>,
    alpha: Option<f64>,
) -> Result<BuiltLayer> {
    let mut keys: Vec<(usize, i64)> = panel.iter().copied().zip(group.iter().copied()).collect();
    keys.sort_unstable();
    keys.dedup();
    let (mut ox, mut oy, mut op, mut og) = (vec![], vec![], vec![], vec![]);
    let (mut lo, mut hi, mut ses) = (vec![], vec![], vec![]);
    for (p, g) in keys {
        let rows: Vec<(f64, f64)> = (0..x.len())
            .filter(|&i| panel[i] == p && group[i] == g && x[i].is_finite() && y[i].is_finite())
            .map(|i| (x[i], y[i]))
            .collect();
        let fit = LmFit::new(&rows, se).ok_or_else(|| {
            refuse(
                "geom_smooth needs at least two distinct x per group (and three rows for \
                 se = TRUE); ggplot2 drops or NaNs such a group, which is not supported",
            )
        })?;
        let (first, last) =
            rows.iter().fold((f64::MAX, f64::MIN), |(a, b), (v, _)| (a.min(*v), b.max(*v)));
        for k in 0..SMOOTH_N {
            // `seq(first, last, length.out = n)`
            let step = (last - first) / (SMOOTH_N as f64 - 1.0);
            let xv = if k == SMOOTH_N - 1 { last } else { first + k as f64 * step };
            let (yv, band) = fit.predict(xv);
            ox.push(xv);
            oy.push(yv);
            op.push(p);
            og.push(g);
            if let Some((s, half)) = band {
                lo.push(yv - half);
                hi.push(yv + half);
                ses.push(s);
            }
        }
    }
    let mut cols = BTreeMap::new();
    if se {
        cols.insert("ymin", lo);
        cols.insert("ymax", hi);
        cols.insert("se", ses);
    }
    let shape = BuiltShape {
        kind: ShapeKind::Smooth,
        panel: op,
        group: og,
        cols,
        colour: Some(colour.unwrap_or(SMOOTH_COLOUR)),
        fill: Some(SMOOTH_FILL),
        linewidth: Some(1.0),
        linetype: Some(1),
        alpha: Some(alpha.unwrap_or(0.4)),
    };
    Ok(BuiltLayer { x: ox, y: oy, points: None, path: None, shape: Some(shape) })
}

/// A simple linear fit `y ~ x` with unit weights, and the confidence band of `predict.lm`.
struct LmFit {
    n: f64,
    xbar: f64,
    ybar: f64,
    sxx: f64,
    slope: f64,
    /// `(residual variance, qt(0.975, n - 2))` when the band was asked for.
    band: Option<(f64, f64)>,
}

impl LmFit {
    /// `None` when fewer than two distinct x remain, or (with `band`) fewer than three rows.
    fn new(rows: &[(f64, f64)], band: bool) -> Option<Self> {
        let n = rows.len();
        let first = rows.first()?.0;
        if rows.iter().all(|(x, _)| *x == first) || (band && n < 3) {
            return None;
        }
        let nf = n as f64;
        let xbar = rows.iter().map(|(x, _)| x).sum::<f64>() / nf;
        let ybar = rows.iter().map(|(_, y)| y).sum::<f64>() / nf;
        let sxx: f64 = rows.iter().map(|(x, _)| (x - xbar) * (x - xbar)).sum();
        let sxy: f64 = rows.iter().map(|(x, y)| (x - xbar) * (y - ybar)).sum();
        let slope = sxy / sxx;
        let band = if band {
            let rss: f64 = rows
                .iter()
                .map(|(x, y)| {
                    let r = y - (ybar + slope * (x - xbar));
                    r * r
                })
                .sum();
            Some((rss / (nf - 2.0), qt_upper(SMOOTH_TAIL, nf - 2.0)?))
        } else {
            None
        };
        Some(Self { n: nf, xbar, ybar, sxx, slope, band })
    }

    /// The fitted value at `x`, and `(se, half-width of the band)` when the band was asked for.
    fn predict(&self, x: f64) -> (f64, Option<(f64, f64)>) {
        let d = x - self.xbar;
        let fit = self.ybar + self.slope * d;
        let band = self.band.map(|(var, t)| {
            let se = (var * (1.0 / self.n + d * d / self.sxx)).sqrt();
            (se, t * se)
        });
        (fit, band)
    }
}
