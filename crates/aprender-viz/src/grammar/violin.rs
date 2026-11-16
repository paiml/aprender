//! ggplot2 4.0.3's `geom_violin()` through [`GGPlot::layout`](super::GGPlot::layout)
//! (PLOT-GG-001, paiml/aprender#6909): `stat_ydensity`, `GeomViolin`'s `setup_data` and the
//! default `position_dodge`, with the discrete x scale they train.
//!
//! Per group (rows sharing a ggplot2 `group`): the density of `y` at 512 points over the group's
//! range (`trim = TRUE`) or three bandwidths past it, `bw.nrd0` times `adjust`, then the three
//! quantile rows ggplot2 4.0.3 appends. Per panel: `violinwidth` by `scale`. Then `width`,
//! `xmin`/`xmax` and the dodge of groups that share an `x`.
//!
//! Refused with a typed error, never approximated: a `colour` mapping, an `x` that is not one
//! value per group (R pools it into one violin at the mid-range), a missing text `x`, a layer
//! with no group of two or more rows (R returns an empty layer), and everything
//! [`dist`](super::dist) refuses (facets, coordinate systems, scale specs, foreign stats, other
//! geoms in the plot).

use crate::error::{Error, Result};

use super::aes::Aes;
use super::build::{BuiltLayer, BuiltLayout, BuiltViolin, PositionScale};
use super::data::{DataFrame, DataValue};
use super::dist::{
    column, is_text, key_order, keys, numeric_lenient, refuse, DistSpec, Key, ViolinScale,
};
use super::dist_math::{
    approx1, bw_nrd0, quantile7, stat_density_adjusted, DensityCols, DENSITY_POINTS,
};
use super::scale_discrete::{position, train_levels};
use super::stat_count::discrete_axis;

/// `GeomViolin`'s default width, as a share of the resolution of `x`.
const DEFAULT_WIDTH: f64 = 0.9;
/// `stat_ydensity`'s `quantiles = c(0.25, 0.5, 0.75)`.
const QUANTILES: [f64; 3] = [0.25, 0.5, 0.75];
/// Bandwidths past the range when `trim = FALSE`.
const UNTRIMMED_EXT: f64 = 3.0;

/// One group's stat output: grid rows, then its quantile rows.
struct GroupRows {
    id: i64,
    x: f64,
    n: usize,
    y: Vec<f64>,
    density: Vec<f64>,
    scaled: Vec<f64>,
    count: Vec<f64>,
    wdensity: Vec<f64>,
    quantile: Vec<f64>,
}

/// The mapped columns, one entry per input row.
struct Input {
    /// `x` as a position: the level number for text, the value (`NaN` if missing) otherwise.
    x: Vec<f64>,
    /// The levels of a text `x`, trained on every row (ggplot2 trains before the stat).
    levels: Option<Vec<String>>,
    y: Vec<f64>,
    group: Vec<i64>,
}

fn check_aesthetics(aes: &Aes) -> Result<()> {
    if aes.color.is_some() || aes.color_value.is_some() {
        return Err(refuse("a `colour` mapping or value is not supported with geom_violin"));
    }
    Ok(())
}

fn check_spec(spec: &DistSpec) -> Result<()> {
    if !(spec.adjust.is_finite() && spec.adjust > 0.0) {
        return Err(refuse("geom_violin: `adjust` must be finite and positive"));
    }
    if let Some(w) = spec.width {
        if !(w.is_finite() && w > 0.0) {
            return Err(refuse("geom_violin: `width` must be finite and positive"));
        }
    }
    Ok(())
}

/// `x` as positions, and the levels when it is text.
fn positions(data: &DataFrame, name: &str) -> Result<(Vec<f64>, Option<Vec<String>>)> {
    if !is_text(data, name)? {
        return Ok((numeric_lenient(data, name)?, None));
    }
    let cells = column(data, name)?;
    let mut text: Vec<&str> = Vec::with_capacity(cells.len());
    for cell in cells {
        match cell {
            DataValue::Text(s) => text.push(s),
            _ => return Err(refuse(format!("column `{name}` has a missing or non-text cell"))),
        }
    }
    let levels = train_levels(text.iter().copied());
    let x = text.iter().map(|s| position(&levels, s).unwrap_or(f64::NAN)).collect();
    Ok((x, Some(levels)))
}

/// The discrete columns `add_group()` interacts, in the order it reads them: `x` (when text),
/// then `fill`. A `group` aesthetic replaces them all.
fn grouping_columns(aes: &Aes, text_x: bool) -> Vec<&str> {
    if let Some(g) = aes.group.as_deref() {
        return vec![g];
    }
    let mut cols: Vec<&str> = Vec::new();
    if let (true, Some(x)) = (text_x, aes.x.as_deref()) {
        cols.push(x);
    }
    if let Some(f) = aes.fill.as_deref() {
        if !cols.contains(&f) {
            cols.push(f);
        }
    }
    cols
}

/// ggplot2's `add_group()`: the rank of each row's value combination (first column slowest), or
/// `-1` when nothing is discrete.
fn group_column(data: &DataFrame, aes: &Aes, text_x: bool) -> Result<Vec<i64>> {
    let cols = grouping_columns(aes, text_x);
    if cols.is_empty() {
        return Ok(vec![-1; data.nrow()]);
    }
    let per_col: Vec<Vec<Key>> = cols.iter().map(|c| keys(data, c)).collect::<Result<_>>()?;
    let row = |i: usize| -> Vec<&Key> { per_col.iter().map(|k| &k[i]).collect() };
    let cmp = |a: &[&Key], b: &[&Key]| {
        a.iter()
            .zip(b)
            .map(|(p, q)| key_order(p, q))
            .find(|o| o.is_ne())
            .unwrap_or(std::cmp::Ordering::Equal)
    };
    let mut distinct: Vec<Vec<&Key>> = (0..data.nrow()).map(row).collect();
    distinct.sort_by(|a, b| cmp(a, b));
    distinct.dedup_by(|a, b| cmp(a, b).is_eq());
    Ok((0..data.nrow())
        .map(|i| {
            let r = row(i);
            distinct
                .binary_search_by(|d| cmp(d, &r))
                .map_or(-1, |p| i64::try_from(p + 1).unwrap_or(i64::MAX))
        })
        .collect())
}

fn read_input(data: &DataFrame, aes: &Aes) -> Result<Input> {
    let xname = aes.x.as_deref().ok_or_else(|| refuse("geom_violin needs an `x` aesthetic"))?;
    let yname = aes.y.as_deref().ok_or_else(|| refuse("geom_violin needs a `y` aesthetic"))?;
    let (x, levels) = positions(data, xname)?;
    let y = numeric_lenient(data, yname)?;
    let group = group_column(data, aes, levels.is_some())?;
    Ok(Input { x, levels, y, group })
}

/// The `(x, y)` pairs with finite `x` and `y` (`remove_missing(finite = TRUE)`: a missing or
/// infinite value drops its row, the group ids having been numbered before), by ascending group id.
fn rows_by_group(input: &Input) -> Vec<(i64, Vec<(f64, f64)>)> {
    let mut ids: Vec<i64> = input.group.clone();
    ids.sort_unstable();
    ids.dedup();
    ids.into_iter()
        .map(|id| {
            let rows = input
                .group
                .iter()
                .zip(input.x.iter().zip(&input.y))
                .filter(|(g, (x, y))| **g == id && x.is_finite() && y.is_finite())
                .map(|(_, (x, y))| (*x, *y))
                .collect();
            (id, rows)
        })
        .collect()
}

/// The one `x` of a group; refused when the group spans several (R would pool them).
fn group_x(xs: &[f64]) -> Result<f64> {
    let first = xs.first().copied().unwrap_or(f64::NAN);
    if xs.iter().any(|v| *v != first) {
        return Err(refuse(
            "geom_violin: `x` varies within a group (ggplot2 draws one violin at the mid-range \
             of x); map a `group` with one `x` per group",
        ));
    }
    Ok(first)
}

/// What R's FFT leaves behind, relative to the largest `count`: `density()` convolves with an FFT
/// whose rounding noise (measured at up to 9.1e-17 of the peak, `pmax.int(0, ...)` turning half of
/// it into exact zeros) lands in the tails, where the true value is smaller. This port sums the
/// convolution directly and is accurate there, so it differs from R by that noise, which no
/// algorithm reproduces. Ten times the measured figure.
const FFT_NOISE: f64 = 1e-15;
/// The tolerance the oracle holds every case to: `|got - want| <= REL * |want| + ABS`.
const TOL_REL: f64 = 1e-9;
const TOL_ABS: f64 = 1e-12;

/// Refuse a group whose R output would be tail noise this port cannot reproduce within tolerance:
/// the noise `FFT_NOISE * max(count)` exceeds what the smallest `count` is allowed to differ by.
/// `count` is the binding column: `density` and `scaled` pass whenever it does.
fn check_noise_floor(dens: &DensityCols) -> Result<()> {
    let peak = dens.count.iter().copied().fold(0.0, f64::max);
    let low = dens.count.iter().copied().fold(f64::INFINITY, f64::min);
    if FFT_NOISE * peak > TOL_ABS + TOL_REL * low {
        return Err(refuse(format!(
            "geom_violin: this group's density is too concentrated to match R within tolerance \
             (largest count {peak:e}, smallest {low:e}): R's FFT leaves rounding noise of about \
             {:e} in the tails, which this port cannot reproduce; rescale `y` (a unit where the \
             values are of order 1)",
            FFT_NOISE * peak
        )));
    }
    Ok(())
}

/// `stat_ydensity`'s `compute_group()` on the `y` of one group of at least two rows.
fn group_stat(spec: &DistSpec, ys: &[f64]) -> Result<(DensityCols, Vec<f64>)> {
    let bw = bw_nrd0(ys)?;
    let lo = ys.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ext = if spec.trim { 0.0 } else { UNTRIMMED_EXT * bw };
    let dens = stat_density_adjusted(ys, lo - ext, hi + ext, DENSITY_POINTS, spec.adjust)?;
    check_noise_floor(&dens)?;
    let mut sorted = ys.to_vec();
    sorted.sort_by(f64::total_cmp);
    let quants = QUANTILES.iter().map(|p| quantile7(&sorted, *p)).collect();
    Ok((dens, quants))
}

/// The group's rows: the grid minus every point equal to a quantile, then the quantile rows with
/// each column linearly interpolated at the quantile's `y` (`approx(ties = "ordered")`).
fn assemble_group(id: i64, x: f64, dens: &DensityCols, quants: &[f64]) -> GroupRows {
    let at = |series: &[f64]| -> Vec<f64> {
        quants.iter().map(|q| approx1(&dens.x, series, *q)).collect()
    };
    let grid = |series: &[f64]| -> Vec<f64> {
        dens.x.iter().zip(series).filter(|(g, _)| !quants.contains(*g)).map(|(_, v)| *v).collect()
    };
    let join =
        |series: &[f64]| -> Vec<f64> { grid(series).into_iter().chain(at(series)).collect() };
    let rows = grid(&dens.x).len();
    GroupRows {
        id,
        x,
        n: dens.n,
        y: grid(&dens.x).into_iter().chain(quants.iter().copied()).collect(),
        density: join(&dens.density),
        scaled: join(&dens.scaled),
        count: join(&dens.count),
        wdensity: join(&dens.wdensity),
        quantile: std::iter::repeat(f64::NAN).take(rows).chain(QUANTILES).collect(),
    }
}

/// Every group with two or more present rows, ascending by id; smaller groups are dropped, as
/// ggplot2 does (with a warning).
fn stat_groups(spec: &DistSpec, input: &Input) -> Result<Vec<GroupRows>> {
    let mut out = Vec::new();
    for (id, rows) in rows_by_group(input) {
        if rows.len() < 2 {
            continue;
        }
        let (xs, ys): (Vec<f64>, Vec<f64>) = rows.into_iter().unzip();
        let x = group_x(&xs)?;
        let (dens, quants) = group_stat(spec, &ys)?;
        out.push(assemble_group(id, x, &dens, &quants));
    }
    if out.is_empty() {
        return Err(refuse(
            "geom_violin: no group has two rows with finite x and y (ggplot2 returns an empty layer)",
        ));
    }
    Ok(out)
}

/// `violinwidth` for each row, by `scale` (`compute_panel()`).
fn violin_widths(scale: ViolinScale, density: &[f64], scaled: &[f64], n: &[f64]) -> Vec<f64> {
    let top = density.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let top_n = n.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    match scale {
        ViolinScale::Area => density.iter().map(|d| d / top).collect(),
        ViolinScale::Count => density.iter().zip(n).map(|(d, k)| d / top * k / top_n).collect(),
        ViolinScale::Width => scaled.to_vec(),
    }
}

/// `resolution(x, zero = TRUE, discrete = FALSE)`: the smallest gap between distinct values
/// (with 0 among them) that exceeds `sqrt(eps)`, or 1 when `x` has no spread or no such gap.
fn resolution(xs: &[f64]) -> f64 {
    let lo = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if zero_range(lo, hi) {
        return 1.0;
    }
    let mut v: Vec<f64> = xs.iter().copied().chain(std::iter::once(0.0)).collect();
    v.sort_by(f64::total_cmp);
    v.dedup();
    let tol = f64::EPSILON.sqrt();
    let gap = v.windows(2).map(|w| w[1] - w[0]).filter(|d| *d > tol).fold(f64::INFINITY, f64::min);
    if gap.is_finite() {
        gap
    } else {
        1.0
    }
}

/// `scales::zero_range()` with its default tolerance.
fn zero_range(lo: f64, hi: f64) -> bool {
    if lo == hi {
        return true;
    }
    let m = lo.abs().min(hi.abs());
    m != 0.0 && ((lo - hi) / m).abs() < 1000.0 * f64::EPSILON
}

/// Per-row `x`, `xmin`, `xmax` and `order` after `position_dodge()` (`preserve = "total"`).
struct Dodged {
    x: Vec<f64>,
    xmin: Vec<f64>,
    xmax: Vec<f64>,
    order: Vec<f64>,
}

/// `PositionDodge`: groups sharing an `x` split its width; `order` is a group's rank there.
/// Rows are sorted by `x` once, so each run of equal `x` is found in `O(N log N)` overall.
fn dodge(x: &[f64], group: &[i64], width: f64) -> Dodged {
    let (xmin0, xmax0): (Vec<f64>, Vec<f64>) =
        x.iter().map(|v| (v - width / 2.0, v + width / 2.0)).unzip();
    let span = xmax0.first().zip(xmin0.first()).map_or(width, |(hi, lo)| hi - lo);
    let n_rows = x.len();
    let mut out = Dodged { x: x.to_vec(), xmin: xmin0, xmax: xmax0, order: vec![1.0; n_rows] };
    // `+ 0.0` makes -0.0 and 0.0 one key, as `==` does.
    let mut by_x: Vec<usize> = (0..n_rows).collect();
    by_x.sort_by(|a, b| (x[*a] + 0.0).total_cmp(&(x[*b] + 0.0)));
    let mut start = 0;
    while start < n_rows {
        // `max(1)`: a NaN is never equal to itself, and still belongs to a run of its own.
        let run = by_x[start..].iter().take_while(|i| x[**i] == x[by_x[start]]).count().max(1);
        dodge_run(&mut out, &by_x[start..start + run], x, group, span);
        start += run;
    }
    out
}

/// The rows (indices) that share one `x`: each is placed among the groups present there.
fn dodge_run(out: &mut Dodged, rows: &[usize], x: &[f64], group: &[i64], span: f64) {
    let mut peers: Vec<i64> = rows.iter().map(|i| group[*i]).collect();
    peers.sort_unstable();
    peers.dedup();
    let n = peers.len() as f64;
    for i in rows {
        let idx = peers.binary_search(&group[*i]).map_or(1.0, |p| (p + 1) as f64);
        out.order[*i] = idx;
        if peers.len() > 1 {
            out.x[*i] = x[*i] + span * ((idx - 0.5) / n - 0.5);
            out.xmin[*i] = out.x[*i] - span / n / 2.0;
            out.xmax[*i] = out.x[*i] + span / n / 2.0;
        }
    }
}

fn extent(lo: &[f64], hi: &[f64]) -> (f64, f64) {
    (
        lo.iter().copied().fold(f64::INFINITY, f64::min),
        hi.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    )
}

/// The x scale after the stat: levels trained before it (kept), the continuous range retrained on
/// `xmin`/`xmax`.
fn x_scale(levels: Option<Vec<String>>, lo: f64, hi: f64) -> PositionScale {
    let Some(levels) = levels else {
        return PositionScale::continuous((lo, hi));
    };
    let axis = discrete_axis(&levels, Some((lo, hi)));
    PositionScale {
        limits: (1.0, levels.len().max(1) as f64),
        continuous_range: axis.continuous_range,
        breaks: axis.breaks,
        labels: axis.labels,
        n_breaks_censored: 0,
        minor_breaks: Vec::new(),
        levels: Some(levels),
    }
}

/// Every group's columns as one set, in group order.
fn flatten(groups: &[GroupRows]) -> Flat {
    let mut f = Flat::default();
    for g in groups {
        let rows = g.y.len();
        f.group.extend(std::iter::repeat(g.id).take(rows));
        f.x.extend(std::iter::repeat(g.x).take(rows));
        f.n.extend(std::iter::repeat(g.n as f64).take(rows));
        f.y.extend(&g.y);
        f.density.extend(&g.density);
        f.scaled.extend(&g.scaled);
        f.count.extend(&g.count);
        f.wdensity.extend(&g.wdensity);
        f.quantile.extend(&g.quantile);
    }
    f
}

#[derive(Default)]
struct Flat {
    group: Vec<i64>,
    x: Vec<f64>,
    n: Vec<f64>,
    y: Vec<f64>,
    density: Vec<f64>,
    scaled: Vec<f64>,
    count: Vec<f64>,
    wdensity: Vec<f64>,
    quantile: Vec<f64>,
}

/// One `geom_violin()`: the stat per group, the panel-wide `violinwidth`, the width and the dodge,
/// then the scales retrained on the output.
///
/// # Errors
///
/// [`Error::Rendering`] for the inputs the module docs list, [`Error::UnknownColumn`] for a
/// missing column.
pub(super) fn violin(spec: &DistSpec, data: &DataFrame, aes: &Aes) -> Result<BuiltLayout> {
    check_aesthetics(aes)?;
    check_spec(spec)?;
    let input = read_input(data, aes)?;
    let groups = stat_groups(spec, &input)?;
    let f = flatten(&groups);
    let violinwidth = violin_widths(spec.violin_scale, &f.density, &f.scaled, &f.n);
    let width = spec.width.unwrap_or_else(|| DEFAULT_WIDTH * resolution(&f.x));
    let d = dodge(&f.x, &f.group, width);
    let (xlo, xhi) = extent(&d.xmin, &d.xmax);
    let (ylo, yhi) = extent(&f.y, &f.y);
    let built = BuiltViolin {
        density: f.density,
        scaled: f.scaled,
        count: f.count,
        wdensity: f.wdensity,
        n: f.n,
        quantile: f.quantile,
        violinwidth,
        width,
        xmin: d.xmin,
        xmax: d.xmax,
        order: d.order,
    };
    let layer = BuiltLayer {
        x: d.x,
        y: f.y,
        points: None,
        path: None,
        bins: None,
        density: None,
        violin: Some(built),
    };
    if !(xlo.is_finite() && xhi.is_finite() && ylo.is_finite() && yhi.is_finite()) {
        return Err(Error::Rendering("geom_violin: the output range is not finite".into()));
    }
    Ok(BuiltLayout {
        layers: vec![layer],
        x: x_scale(input.levels, xlo, xhi),
        y: PositionScale::continuous((ylo, yhi)),
    })
}

#[cfg(test)]
mod tests {
    use super::super::dist::NOT_DRAWN;
    use super::super::{scale_continuous, Coord, Facet, GGPlot, Geom, Layer, TerminalPlot};
    use super::*;

    /// Two groups (`a`, `b`) of four, `f` alternating, `xn` one value per group.
    fn small() -> DataFrame {
        let mut d = DataFrame::new();
        d.add_column_str("g", &["a", "a", "a", "a", "b", "b", "b", "b"]);
        d.add_column_str("f", &["p", "q", "p", "q", "p", "q", "p", "q"]);
        d.add_column_f32("xn", &[1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0]);
        d.add_column_f32("y", &[1.0, 2.0, 4.0, 7.5, 2.0, 3.0, 5.0, 8.5]);
        d
    }

    fn gy() -> Aes {
        Aes::new().x("g").y("y")
    }

    fn plot(data: DataFrame, aes: Aes, geom: Geom) -> GGPlot {
        GGPlot::new().data(data).aes(aes).layer(Layer::new(geom))
    }

    fn run(p: GGPlot) -> Result<BuiltLayout> {
        p.build().and_then(|p| p.layout())
    }

    fn refused_with(r: &Result<BuiltLayout>, needle: &str) -> bool {
        matches!(r, Err(Error::Rendering(m)) if m.contains(needle))
    }

    #[test]
    fn a_group_gets_515_rows_and_three_appended_quantile_rows() {
        let b = run(plot(small(), gy(), Geom::gg_violin())).expect("layout");
        let v = b.layers[0].violin.as_ref().expect("violin columns");
        assert_eq!(b.layers[0].x.len(), 2 * 515);
        let (grid, tail) = v.quantile.split_at(512);
        assert!(grid.iter().all(|q| q.is_nan()));
        assert_eq!(tail.get(..3), Some(&[0.25, 0.5, 0.75][..]), "the probabilities");
        assert_eq!(b.layers[0].y.get(512..515), Some(&[1.75, 3.0, 4.875][..]), "their y");
        assert!(v.quantile.get(515..1024).is_some_and(|s| s.iter().take(512).all(|q| q.is_nan())));
    }

    #[test]
    fn numeric_x_without_a_group_is_refused() {
        let r = run(plot(small(), Aes::new().x("y").y("y"), Geom::gg_violin()));
        assert!(refused_with(&r, "varies within a group"), "{r:?}");
        // The same x, one value per group through a `group` aesthetic, is built.
        let ok = run(plot(small(), Aes::new().x("xn").y("y").group("g"), Geom::gg_violin()));
        assert!(ok.is_ok(), "{ok:?}");
    }

    #[test]
    fn colour_and_missing_aesthetics_are_refused() {
        let colour = run(plot(small(), gy().color("f"), Geom::gg_violin()));
        assert!(refused_with(&colour, "colour"), "{colour:?}");
        let two = run(plot(small(), gy().fill("f").color("g"), Geom::gg_violin()));
        assert!(refused_with(&two, "interaction"), "{two:?}");
        let no_x = run(plot(small(), Aes::new().y("y"), Geom::gg_violin()));
        assert!(refused_with(&no_x, "`x`"), "{no_x:?}");
        let no_y = run(plot(small(), Aes::new().x("g"), Geom::gg_violin()));
        assert!(refused_with(&no_y, "`y`"), "{no_y:?}");
        let unknown = run(plot(small(), Aes::new().x("g").y("nope"), Geom::gg_violin()));
        assert!(matches!(unknown, Err(Error::UnknownColumn(_))), "{unknown:?}");
    }

    #[test]
    fn invalid_adjust_and_width_are_refused() {
        for a in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let r = run(plot(small(), gy(), Geom::gg_violin().adjust(a)));
            assert!(refused_with(&r, "`adjust`"), "adjust {a}: {r:?}");
        }
        for w in [0.0_f32, -0.5, f32::NAN, f32::INFINITY] {
            let r = run(plot(small(), gy(), Geom::gg_violin().width(w)));
            assert!(refused_with(&r, "`width`"), "width {w}: {r:?}");
        }
    }

    #[test]
    fn a_layer_without_a_group_of_two_rows_is_refused_not_returned_empty() {
        let mut d = DataFrame::new();
        d.add_column_str("g", &["a", "b", "c"]);
        d.add_column_f32("y", &[1.0, 2.0, 3.0]);
        let r = run(plot(d, gy(), Geom::gg_violin()));
        assert!(refused_with(&r, "no group has two rows"), "{r:?}");
        let mut na = DataFrame::new();
        na.add_column_str("g", &["a", "a"]);
        na.add_column_f32("y", &[1.0, f32::NAN]);
        let r = run(plot(na, gy(), Geom::gg_violin()));
        assert!(refused_with(&r, "no group has two rows"), "NA rows do not count: {r:?}");
    }

    #[test]
    fn infinite_rows_are_dropped_as_r_does_and_hostile_values_return_errors_not_panics() {
        let mut d = small();
        d.add_column_f32("inf", &[1.0, f32::INFINITY, 2.0, 3.0, 1.0, 2.0, 3.0, f32::NEG_INFINITY]);
        let b = run(plot(d, Aes::new().x("g").y("inf"), Geom::gg_violin())).expect("drops rows");
        let v = b.layers[0].violin.as_ref().expect("violin columns");
        assert_eq!(v.n.iter().copied().fold(0.0, f64::max), 3.0, "a keeps 3 rows, b keeps 3");
        let mut all = DataFrame::new();
        all.add_column_str("g", &["a", "a"]);
        all.add_column_f32("y", &[f32::INFINITY, f32::NEG_INFINITY]);
        let r = run(plot(all, gy(), Geom::gg_violin()));
        assert!(r.is_err(), "{r:?}");
        let mut big = DataFrame::new();
        big.add_column_str("g", &["a", "a"]);
        big.add_column_f32("y", &[f32::MAX, -f32::MAX]);
        let _ = run(plot(big, gy(), Geom::gg_violin()));
        let empty = DataFrame::new();
        let r = run(plot(empty, gy(), Geom::gg_violin()));
        assert!(r.is_err(), "{r:?}");
    }

    #[test]
    fn infinite_x_rows_are_dropped_and_the_group_ids_stay() {
        let mut d = small();
        d.add_column_f32("xi", &[1.0, 1.0, f32::INFINITY, 1.0, 3.0, 3.0, 3.0, 3.0]);
        let aes = Aes::new().x("xi").y("y").group("g");
        let b = run(plot(d, aes, Geom::gg_violin())).expect("drops rows");
        let v = b.layers[0].violin.as_ref().expect("violin columns");
        assert_eq!(v.n.iter().copied().fold(0.0, f64::max), 4.0);
        assert!(v.n.contains(&3.0), "group a lost the infinite row");
    }

    /// A dense group is refused by name; the same data in unit scale is not.
    #[test]
    fn a_group_too_concentrated_for_r_to_be_matched_is_refused_by_name() {
        let tiny = |k: f32| {
            let mut d = small();
            let y: Vec<f32> =
                [1.0f32, 2.0, 4.0, 7.5, 2.0, 3.0, 5.0, 8.5].iter().map(|v| v * k).collect();
            d.add_column_f32("ys", &y);
            run(plot(d, Aes::new().x("g").y("ys"), Geom::gg_violin().adjust(0.05)))
        };
        assert!(tiny(1.0).is_ok());
        let r = tiny(1.0 / 1024.0);
        assert!(refused_with(&r, "too concentrated"), "{r:?}");
    }

    /// With violin as the ONLY layer: every coord, facet and scale spec is refused by name.
    #[test]
    fn coords_facets_scales_and_mixed_layers_are_refused_for_a_lone_violin() {
        let g = Geom::gg_violin();
        for c in [
            Coord::cartesian().xlim(0.0, 1.0),
            Coord::cartesian().ylim(0.0, 1.0),
            Coord::cartesian().flip(),
            Coord::polar(),
            Coord::fixed(1.0),
        ] {
            let r = run(plot(small(), gy(), g.clone()).coord(c.clone()));
            assert!(matches!(r, Err(Error::UnsupportedCoord(_))), "{c:?}");
        }
        for f in [Facet::row("g"), Facet::col("g"), Facet::wrap("g", 2)] {
            let r = run(plot(small(), gy(), g.clone()).facet(f));
            assert!(matches!(r, Err(Error::Rendering(_))), "facet");
        }
        let scaled = |m: &Result<BuiltLayout>| matches!(m, Err(Error::UnsupportedLayer(m)) if m.contains("scale_x/scale_y"));
        assert!(scaled(&run(plot(small(), gy(), g.clone()).scale_x(scale_continuous()))));
        assert!(scaled(&run(plot(small(), gy(), g.clone()).scale_y(scale_continuous()))));
        let mixed =
            plot(small(), gy(), g).layer(Layer::new(Geom::point()).aes(Aes::new().x("xn").y("y")));
        assert!(refused_with(&run(mixed), "combined"));
    }

    #[test]
    fn every_renderer_refuses_the_undrawn_layer() {
        let p = plot(small(), gy(), Geom::gg_violin()).build().expect("builds");
        let drawn = |r: Result<()>| matches!(r, Err(Error::UnsupportedLayer(m)) if m == NOT_DRAWN);
        assert!(drawn(p.to_framebuffer().map(drop)), "framebuffer");
        assert!(drawn(p.to_svg().map(drop)), "svg");
        let term = p.to_terminal(TerminalPlot { width: 60, height: 16, color: false });
        assert!(matches!(term, Err(Error::Rendering(m)) if m.contains("point, line")), "terminal");
        let numeric = plot(small(), Aes::new().x("xn").y("y").group("g"), Geom::gg_violin())
            .build()
            .expect("builds");
        let term = numeric.to_terminal(TerminalPlot { width: 60, height: 16, color: false });
        assert!(matches!(term, Err(Error::Rendering(m)) if m.contains("point, line")), "terminal");
    }

    #[test]
    fn resolution_is_the_smallest_gap_with_zero_among_the_values() {
        assert_eq!(resolution(&[3.0, 3.0]), 1.0, "no spread");
        // No gap above sqrt(eps): R returns 1 (measured: width 0.9), not infinity.
        assert_eq!(
            resolution(&[
                (1.0 / 1_073_741_824.0),
                (1.0 / 536_870_912.0),
                3.0 * (1.0 / 1_073_741_824.0)
            ]),
            1.0
        );
        assert_eq!(resolution(&[1.5, 5.0, 9.0]), 1.5, "zero joins: 0, 1.5, 5, 9");
        assert_eq!(resolution(&[1.0, 2.0, 3.0]), 1.0);
        assert!((resolution(&[2.0, 2.25, 4.0]) - 0.25).abs() < 1e-15);
    }

    /// The quadratic definition `position_dodge` states: peers are the distinct groups among the
    /// rows whose `x` is equal (`-0.0 == 0.0`). The sorted walk must give the same numbers.
    #[test]
    fn the_sorted_dodge_equals_the_quadratic_definition() {
        let x = [1.0, -0.0, 2.0, 0.0, 1.0, 2.0, 2.0, 3.5, 0.0, 1.0];
        let group = [3, 1, 2, 5, 1, 2, 9, 4, 1, 3];
        let width = 0.9;
        let d = dodge(&x, &group, width);
        // The span `position_dodge` takes: the first row's xmax - xmin, rounded as R rounds it.
        let span = (x[0] + width / 2.0) - (x[0] - width / 2.0);
        for i in 0..x.len() {
            let mut peers: Vec<i64> =
                (0..x.len()).filter(|j| x[*j] == x[i]).map(|j| group[j]).collect();
            peers.sort_unstable();
            peers.dedup();
            let n = peers.len() as f64;
            let idx = (peers.iter().position(|g| *g == group[i]).expect("own group") + 1) as f64;
            let (cx, lo, hi) = if peers.len() > 1 {
                let c = x[i] + span * ((idx - 0.5) / n - 0.5);
                (c, c - span / n / 2.0, c + span / n / 2.0)
            } else {
                (x[i], x[i] - width / 2.0, x[i] + width / 2.0)
            };
            assert_eq!(d.order[i], idx, "row {i}");
            assert_eq!((d.x[i], d.xmin[i], d.xmax[i]), (cx, lo, hi), "row {i}");
        }
    }

    #[test]
    fn dodge_splits_only_the_groups_that_share_an_x() {
        let d = dodge(&[1.0, 1.0, 2.0], &[1, 2, 3], 0.9);
        assert_eq!(d.order, [1.0, 2.0, 1.0]);
        assert!((d.x[0] - 0.775).abs() < 1e-12 && (d.x[1] - 1.225).abs() < 1e-12);
        assert!((d.xmin[0] - 0.55).abs() < 1e-12 && (d.xmax[0] - 1.0).abs() < 1e-12);
        assert!((d.x[2] - 2.0).abs() < 1e-12);
        assert!((d.xmin[2] - 1.55).abs() < 1e-12 && (d.xmax[2] - 2.45).abs() < 1e-12);
        let lone = dodge(&[], &[], 0.9);
        assert!(lone.x.is_empty());
    }

    #[test]
    fn grouping_interacts_x_then_fill_and_group_replaces_both() {
        let d = small();
        let both = group_column(&d, &gy().fill("f"), true).expect("groups");
        assert_eq!(both, [1, 2, 1, 2, 3, 4, 3, 4]);
        let by_x = group_column(&d, &gy(), true).expect("groups");
        assert_eq!(by_x, [1, 1, 1, 1, 2, 2, 2, 2]);
        let own = group_column(&d, &gy().fill("f").group("f"), true).expect("groups");
        assert_eq!(own, [1, 2, 1, 2, 1, 2, 1, 2]);
        let none = group_column(&d, &Aes::new().x("xn").y("y"), false).expect("groups");
        assert_eq!(none, [-1; 8]);
    }
}
