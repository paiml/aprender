//! `geom_smooth(method = "lm")` as `ggplot_build()` in ggplot2 4.0.3 resolves it (PLOT-GG-001,
//! paiml/aprender#6909). Proven against R by `ggplot2_geom_smooth_oracle.rs`.
//!
//! # What is resolved
//!
//! `stat_smooth(method = "lm", formula = y ~ x)` per group: a least-squares line fitted to the
//! group's rows, evaluated at `n` points spread evenly over the group's own x range
//! (`fullrange = FALSE`), with the standard error of the mean prediction and, for `se = TRUE`, the
//! confidence band `y -+ qt((1 + level) / 2, n - 2) * se`. The position scales are trained on the
//! stat's output (`x`, `y`, `ymin`, `ymax`), as ggplot2 retrains them after the stat.
//!
//! # What is refused (typed error, never a plausible wrong answer)
//!
//! * a method other than `lm`, so a bare `geom_smooth()` (ggplot2 picks loess below 1000 rows) is
//!   never silently fitted with lm; a stat other than `stat_smooth`;
//! * `fullrange = TRUE`, a `level` outside `(0, 1)`, `n` of 0 or past [`MAX_N`];
//! * a facet; a data-mapped colour, size, shape, alpha, fill, label, `ymin` or `ymax`;
//! * a layer where no group has two distinct x values (ggplot2 returns an empty layer there);
//! * the same coordinate, companion and degenerate-x refusals as the path family (see `path`).
//!
//! Not expressible in this crate's API, so there is nothing to refuse: a `formula` other than
//! `y ~ x`, `weight`, `orientation = "y"`, `xseq`, `span`.
//!
//! The renderer does not draw a smooth layer; `to_svg` and `to_framebuffer` refuse it with
//! [`NOT_DRAWN`].

use super::aes::Aes;
use super::build::BuiltLayer;
use super::data::{DataFrame, DataValue};
use super::facet::Facet;
use super::geom::{GeomType, SmoothMethod};
use super::ggplot::Layer;
use super::path::{group_ids, refuse};
use super::stat::Stat;
use super::t_dist::qt_upper;
use crate::error::{Error, Result};

/// The refusal `to_svg` / `to_framebuffer` give a plot with a smooth layer.
pub(super) const NOT_DRAWN: &str = "geom_smooth is not drawn by the renderer yet; use layout()";

/// The most points `n` may ask of one group (a memory guard, not a ggplot2 limit).
pub const MAX_N: usize = 1_000_000;

/// `se`, `level`, `n` and `fullrange` of `stat_smooth`. Defaults are ggplot2's: `se = TRUE`,
/// `level = 0.95`, `n = 80`, `fullrange = FALSE`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmoothSpec {
    /// Compute the confidence band and the standard error.
    pub se: bool,
    /// Confidence level of the band, in `(0, 1)`.
    pub level: f64,
    /// Number of evaluation points per group.
    pub n: usize,
    /// Evaluate over the full panel range instead of each group's range (refused).
    pub fullrange: bool,
}

impl Default for SmoothSpec {
    fn default() -> Self {
        Self { se: true, level: 0.95, n: 80, fullrange: false }
    }
}

/// The resolved columns of a smooth layer beside `BuiltLayer::x` / `BuiltLayer::y` (the fitted
/// values), one entry per row. Rows are ordered by group, then x.
#[derive(Debug, Clone, PartialEq)]
pub struct BuiltSmooth {
    /// ggplot2's group id of each row (`-1` when no group is mapped and one group is implied).
    pub group: Vec<i64>,
    /// Lower edge of the band; `None` for `se = FALSE`. `NaN` is R's `NA`.
    pub ymin: Option<Vec<f64>>,
    /// Upper edge of the band; `None` for `se = FALSE`.
    pub ymax: Option<Vec<f64>>,
    /// Standard error of the mean prediction; `None` for `se = FALSE`.
    pub se: Option<Vec<f64>>,
    /// ggplot2's `flipped_aes`; always `false` here (`orientation = "y"` is not expressible).
    pub flipped_aes: bool,
}

/// Is this a geom this module builds?
pub(super) fn is_smooth(g: &GeomType) -> bool {
    matches!(g, GeomType::Smooth { .. })
}

/// Refuse what the module cannot resolve; the layer's spec when it can.
fn require_supported(layer: &Layer, aes: &Aes, facet: &Facet) -> Result<SmoothSpec> {
    let GeomType::Smooth { method, spec } = layer.geom.geom_type else {
        return Err(refuse("not a smooth layer"));
    };
    if method != SmoothMethod::Linear {
        return Err(Error::UnsupportedStat(
            "geom_smooth with a method other than lm: layout() fits lm only (ggplot2's default \
             method is loess for under 1000 rows, so a bare geom_smooth() is refused)",
        ));
    }
    if !matches!(layer.geom.stat, None | Some(Stat::Smooth)) {
        return Err(Error::UnsupportedStat("geom_smooth with a stat other than stat_smooth"));
    }
    if !matches!(facet, Facet::None) {
        return Err(refuse("geom_smooth supports no facet: layout() fits one panel"));
    }
    require_aes(aes)?;
    require_spec(&spec)?;
    Ok(spec)
}

fn require_aes(aes: &Aes) -> Result<()> {
    let mapped = [&aes.color, &aes.size, &aes.shape, &aes.alpha, &aes.fill, &aes.label]
        .iter()
        .any(|a| a.is_some())
        || aes.ymin.is_some()
        || aes.ymax.is_some();
    if mapped {
        return Err(refuse(
            "a data-mapped colour, size, shape, alpha, fill, label, ymin or ymax is not supported \
             on geom_smooth",
        ));
    }
    Ok(())
}

fn require_spec(spec: &SmoothSpec) -> Result<()> {
    if spec.fullrange {
        return Err(refuse("geom_smooth(fullrange = TRUE) is not supported"));
    }
    if !(spec.level > 0.0 && spec.level < 1.0) {
        return Err(refuse(format!(
            "geom_smooth level {} is outside (0, 1); ggplot2 gives infinite or empty bands there",
            spec.level
        )));
    }
    if spec.n == 0 || spec.n > MAX_N {
        return Err(refuse(format!("geom_smooth n {} is outside 1..={MAX_N}", spec.n)));
    }
    Ok(())
}

/// One position column as f64, one entry per row: `NA`, null and a missing cell are `NaN`, an
/// infinite value is kept (ggplot2 drops it with a warning, see [`finite_rows`]).
fn read_position(data: &DataFrame, name: &str, what: &str) -> Result<Vec<f64>> {
    let col =
        data.get(name).ok_or_else(|| Error::UnknownColumn(format!("no {what} column {name:?}")))?;
    (0..data.nrow())
        .map(|i| match col.get(i) {
            Some(DataValue::Number(v)) => Ok(f64::from(*v)),
            Some(DataValue::Null) | None => Ok(f64::NAN),
            Some(DataValue::Text(_)) => {
                Err(refuse(format!("{what} column {name:?} is text; only continuous positions")))
            }
        })
        .collect()
}

/// The `(x, y)` pairs of group `g` whose x and y are both finite: ggplot2 removes the others
/// before fitting (measured on 4.0.3: a missing or infinite value drops the row, with a warning).
fn finite_rows(xs: &[f64], ys: &[f64], group: &[i64], g: i64) -> (Vec<f64>, Vec<f64>) {
    xs.iter()
        .zip(ys)
        .zip(group)
        .filter(|((x, y), gi)| **gi == g && x.is_finite() && y.is_finite())
        .map(|((x, y), _)| (*x, *y))
        .unzip()
}

/// A least-squares line `y = ybar + slope * (x - xbar)` and what its band needs.
#[derive(Debug, Clone, Copy)]
struct Fit {
    m: f64,
    xbar: f64,
    ybar: f64,
    sxx: f64,
    slope: f64,
    /// Residual variance `rss / (m - 2)`; `NaN` with no residual degree of freedom.
    var: f64,
    df: f64,
}

/// Fit one group, on x centred at its mean so an offset in x costs no digits. `None` when the
/// group has fewer than two distinct x (`StatSmooth` returns an empty frame for it).
fn fit_group(xs: &[f64], ys: &[f64]) -> Option<Fit> {
    let first = *xs.first()?;
    if xs.iter().all(|x| *x == first) {
        return None;
    }
    let m = xs.len() as f64;
    let xbar = xs.iter().sum::<f64>() / m;
    let ybar = ys.iter().sum::<f64>() / m;
    let sxx = xs.iter().map(|x| (x - xbar) * (x - xbar)).sum::<f64>();
    let sxy = xs.iter().zip(ys).map(|(x, y)| (x - xbar) * (y - ybar)).sum::<f64>();
    let slope = sxy / sxx;
    let rss = xs
        .iter()
        .zip(ys)
        .map(|(x, y)| {
            let r = y - (ybar + slope * (x - xbar));
            r * r
        })
        .sum::<f64>();
    let df = m - 2.0;
    let var = if df > 0.0 { rss / df } else { f64::NAN };
    Some(Fit { m, xbar, ybar, sxx, slope, var, df })
}

/// The fitted value and the standard error of the mean prediction at `x0`.
fn predict(f: &Fit, x0: f64) -> (f64, f64) {
    let d = x0 - f.xbar;
    let se = (f.var * (1.0 / f.m + d * d / f.sxx)).sqrt();
    (f.ybar + f.slope * d, se)
}

/// `seq(lo, hi, length.out = n)`: both ends exact, the others `lo + i * by`.
fn x_grid(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![lo],
        _ => {
            let by = (hi - lo) / (n - 1) as f64;
            (0..n).map(|i| if i + 1 == n { hi } else { lo + i as f64 * by }).collect()
        }
    }
}

/// The columns of the rows built so far.
#[derive(Default)]
struct Rows {
    x: Vec<f64>,
    y: Vec<f64>,
    group: Vec<i64>,
    se: Vec<f64>,
    lo: Vec<f64>,
    hi: Vec<f64>,
}

/// Append group `g`'s fitted rows: nothing when the group has too few distinct x.
fn append_group(rows: &mut Rows, spec: &SmoothSpec, g: i64, xs: &[f64], ys: &[f64]) {
    let Some(fit) = fit_group(xs, ys) else { return };
    let lo = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let t = qt_upper((1.0 - spec.level) / 2.0, fit.df);
    for x0 in x_grid(lo, hi, spec.n) {
        let (y, se) = predict(&fit, x0);
        rows.x.push(x0);
        rows.y.push(y);
        rows.group.push(g);
        rows.se.push(se);
        rows.lo.push(y - t * se);
        rows.hi.push(y + t * se);
    }
}

/// Build one `geom_smooth(method = "lm")` layer.
pub(super) fn build_layer(
    layer: &Layer,
    plot_aes: &Aes,
    plot_data: &DataFrame,
    facet: &Facet,
) -> Result<BuiltLayer> {
    let aes = plot_aes.merge(&layer.aes);
    let spec = require_supported(layer, &aes, facet)?;
    let data = layer.data.as_ref().unwrap_or(plot_data);
    let xs = read_position(data, aes.x.as_deref().unwrap_or("x"), "x")?;
    let ys = read_position(data, aes.y.as_deref().unwrap_or("y"), "y")?;
    let group = group_ids(&aes, data)?;
    let mut ids = group.clone();
    ids.sort_unstable();
    ids.dedup();
    let mut rows = Rows::default();
    for g in ids {
        let (gx, gy) = finite_rows(&xs, &ys, &group, g);
        append_group(&mut rows, &spec, g, &gx, &gy);
    }
    if rows.x.is_empty() {
        return Err(refuse(
            "geom_smooth: no group has two distinct finite x values, so ggplot2 returns an empty \
             layer; layout() refuses it instead of returning one",
        ));
    }
    Ok(assemble(rows, &spec))
}

fn assemble(rows: Rows, spec: &SmoothSpec) -> BuiltLayer {
    let band = spec.se;
    BuiltLayer {
        x: rows.x,
        y: rows.y,
        points: None,
        path: None,
        filled: None,
        smooth: Some(BuiltSmooth {
            group: rows.group,
            ymin: band.then_some(rows.lo),
            ymax: band.then_some(rows.hi),
            se: band.then_some(rows.se),
            flipped_aes: false,
        }),
    }
}

/// The finite extents of a built smooth layer over every column ggplot2 trains the position scales
/// on (`x`, `y`, `ymin`, `ymax`), `None` for an axis with no finite value.
pub(super) fn extents(l: &BuiltLayer) -> (Option<(f64, f64)>, Option<(f64, f64)>) {
    let Some(s) = &l.smooth else { return (None, None) };
    let range = |cols: &[&[f64]]| {
        let mut r: Option<(f64, f64)> = None;
        for v in cols.iter().flat_map(|c| c.iter()).filter(|v| v.is_finite()) {
            r = Some(r.map_or((*v, *v), |(a, b)| (a.min(*v), b.max(*v))));
        }
        r
    };
    let empty: &[f64] = &[];
    let lo = s.ymin.as_deref().unwrap_or(empty);
    let hi = s.ymax.as_deref().unwrap_or(empty);
    (range(&[&l.x]), range(&[&l.y, lo, hi]))
}
