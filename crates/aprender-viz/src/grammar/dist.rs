//! ggplot2 4.0.3's distribution geoms through [`GGPlot::layout`](super::GGPlot::layout)
//! (PLOT-GG-001, paiml/aprender#6909): `geom_histogram`, `geom_freqpoly`, `geom_density`,
//! `geom_boxplot` and `geom_errorbar`, each with its stat and position.
//!
//! These layers are built by [`layout`], which mirrors `ggplot_build()`: group ids, the stat
//! (`stat_bin`, `stat_density`, `stat_boxplot`, identity), the position (`stack`, `dodge2`,
//! identity), then the position scales retrained on the stat's output. The numeric kernels are in
//! `dist_math.rs`. The renderer does not draw these geoms yet; they exist for `layout()`.
//!
//! Out of scope, and refused with a typed error instead of drawn wrongly: facets, a coordinate
//! system other than plain Cartesian, text `x` (a discrete scale), `y`-only histograms and
//! densities (flipped orientation), two or more discrete mappings without a `group`, a continuous
//! `colour` or `fill` mapping (the palette is not resolved: a mapped aesthetic is
//! [`Paint::Mapped`]), non-finite input, and layers mixed with non-distribution layers.

use crate::color::Rgba;
use crate::error::{Error, Result};

use super::aes::Aes;
use super::build::{BuiltLayer, BuiltLayout, PositionScale};
use super::coord::Coord;
use super::data::{DataFrame, DataValue};
pub use super::dist_math::Closed;
use super::dist_math::{
    bin_breaks, bin_breaks_bins, bin_breaks_width, bin_vector, bw_nrd0, cmp_na_last, density,
    dodge2, quantile7, resolution, Bins, Box2, MAX_DENSITY_N, MAX_OUTPUT_ROWS,
};
use super::facet::Facet;
use super::geom::{Geom, GeomType};
use super::ggplot::Layer;
use super::stat::Stat;

/// Which ggplot2 layer a [`DistSpec`] stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistKind {
    /// `geom_histogram()`: `stat_bin` with `pad = FALSE`, stacked.
    Histogram,
    /// `geom_freqpoly()`: `stat_bin` with `pad = TRUE`, as a path.
    Freqpoly,
    /// `geom_density()`: `stat_density`, gaussian kernel.
    Density,
    /// `geom_boxplot()`: `stat_boxplot`, `position_dodge2`.
    Boxplot,
    /// `geom_errorbar()`: identity stat, identity position.
    Errorbar,
}

/// The `bw` argument of `stat_density`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bandwidth {
    /// `"nrd0"`: Silverman's rule of thumb, `stats::bw.nrd0`.
    Nrd0,
    /// A number.
    Value(f64),
}

/// A distribution layer's parameters, named as ggplot2 names them.
#[derive(Debug, Clone, PartialEq)]
pub struct DistSpec {
    /// Which layer.
    pub kind: DistKind,
    /// `bins` (default 30).
    pub bins: usize,
    /// `binwidth`.
    pub binwidth: Option<f64>,
    /// `breaks`.
    pub breaks: Option<Vec<f64>>,
    /// `center`.
    pub center: Option<f64>,
    /// `boundary`.
    pub boundary: Option<f64>,
    /// `closed`.
    pub closed: Closed,
    /// `pad`; `None` is the layer's default (`TRUE` for the frequency polygon, else `FALSE`).
    pub pad: Option<bool>,
    /// `bw`.
    pub bw: Bandwidth,
    /// `adjust`.
    pub adjust: f64,
    /// `n`, the density grid size (default 512).
    pub n: usize,
    /// `trim`.
    pub trim: bool,
    /// `coef`, the whisker multiple of the IQR (default 1.5).
    pub coef: f64,
    /// `width` of a box or an error bar.
    pub width: Option<f64>,
    /// `linewidth` (default 0.5).
    pub linewidth: Option<f64>,
}

impl DistSpec {
    fn new(kind: DistKind) -> Self {
        Self {
            kind,
            bins: 30,
            binwidth: None,
            breaks: None,
            center: None,
            boundary: None,
            closed: Closed::Right,
            pad: None,
            bw: Bandwidth::Nrd0,
            adjust: 1.0,
            n: 512,
            trim: false,
            coef: 1.5,
            width: None,
            linewidth: None,
        }
    }
}

impl Geom {
    fn dist(kind: DistKind) -> Self {
        Self { geom_type: GeomType::Dist(DistSpec::new(kind)), aes: None, stat: None }
    }

    /// ggplot2 4.0.3 `geom_histogram()`, as built by `layout()`.
    #[must_use]
    pub fn gg_histogram() -> Self {
        Self::dist(DistKind::Histogram)
    }

    /// ggplot2 4.0.3 `geom_freqpoly()`, as built by `layout()`.
    #[must_use]
    pub fn gg_freqpoly() -> Self {
        Self::dist(DistKind::Freqpoly)
    }

    /// ggplot2 4.0.3 `geom_density()`, as built by `layout()`.
    #[must_use]
    pub fn gg_density() -> Self {
        Self::dist(DistKind::Density)
    }

    /// ggplot2 4.0.3 `geom_boxplot()`, as built by `layout()`.
    #[must_use]
    pub fn gg_boxplot() -> Self {
        Self::dist(DistKind::Boxplot)
    }

    /// ggplot2 4.0.3 `geom_errorbar()`, as built by `layout()`.
    #[must_use]
    pub fn gg_errorbar() -> Self {
        Self::dist(DistKind::Errorbar)
    }

    fn with_spec(mut self, f: impl FnOnce(&mut DistSpec)) -> Self {
        if let GeomType::Dist(spec) = &mut self.geom_type {
            f(spec);
        }
        self
    }

    /// `binwidth`.
    #[must_use]
    pub fn binwidth(self, v: f64) -> Self {
        self.with_spec(|s| s.binwidth = Some(v))
    }

    /// `boundary`.
    #[must_use]
    pub fn boundary(self, v: f64) -> Self {
        self.with_spec(|s| s.boundary = Some(v))
    }

    /// `center`.
    #[must_use]
    pub fn center(self, v: f64) -> Self {
        self.with_spec(|s| s.center = Some(v))
    }

    /// `closed`.
    #[must_use]
    pub fn closed(self, v: Closed) -> Self {
        self.with_spec(|s| s.closed = v)
    }

    /// `breaks`.
    #[must_use]
    pub fn breaks(self, v: &[f64]) -> Self {
        self.with_spec(|s| s.breaks = Some(v.to_vec()))
    }

    /// `pad`.
    #[must_use]
    pub fn pad(self, v: bool) -> Self {
        self.with_spec(|s| s.pad = Some(v))
    }

    /// `bw` as a number.
    #[must_use]
    pub fn bw(self, v: f64) -> Self {
        self.with_spec(|s| s.bw = Bandwidth::Value(v))
    }

    /// `adjust`.
    #[must_use]
    pub fn adjust(self, v: f64) -> Self {
        self.with_spec(|s| s.adjust = v)
    }

    /// `n`, the density grid size.
    #[must_use]
    pub fn n(self, v: usize) -> Self {
        self.with_spec(|s| s.n = v)
    }

    /// `trim`.
    #[must_use]
    pub fn trim(self, v: bool) -> Self {
        self.with_spec(|s| s.trim = v)
    }

    /// `coef`.
    #[must_use]
    pub fn coef(self, v: f64) -> Self {
        self.with_spec(|s| s.coef = v)
    }

    /// `linewidth`.
    #[must_use]
    pub fn linewidth(self, v: f64) -> Self {
        self.with_spec(|s| s.linewidth = Some(v))
    }
}

/// A colour or fill after building: ggplot2 resolves a mapped one through a scale, which this
/// crate does not do yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Paint {
    /// `NA`.
    Na,
    /// A fixed colour.
    Fixed(Rgba),
    /// Mapped to a variable: its palette is not resolved.
    Mapped,
}

/// A distribution layer after the stat and position: ggplot2's `layer_data()` columns.
#[derive(Debug, Clone, PartialEq)]
pub struct BuiltDist {
    /// The ggplot2 geom class, e.g. `"GeomBar"`.
    pub geom: &'static str,
    /// Numeric columns by ggplot2's name; `NaN` is `NA`. Every column has [`nrow`](Self::nrow)
    /// entries.
    pub cols: Vec<(&'static str, Vec<f64>)>,
    /// The `group` column.
    pub group: Vec<i64>,
    /// A box plot's `outliers` list column (empty for other layers).
    pub outliers: Vec<Vec<f64>>,
    /// The `colour` column.
    pub colour: Vec<Paint>,
    /// The `fill` column.
    pub fill: Vec<Paint>,
}

impl BuiltDist {
    /// One column by name.
    #[must_use]
    pub fn col(&self, name: &str) -> Option<&[f64]> {
        self.cols.iter().find(|(n, _)| *n == name).map(|(_, v)| v.as_slice())
    }

    /// Number of rows.
    #[must_use]
    pub fn nrow(&self) -> usize {
        self.group.len()
    }
}

fn refuse(msg: impl Into<String>) -> Error {
    Error::Rendering(msg.into())
}

/// `true` when this layer is a distribution layer.
pub(super) fn is_dist(layer: &Layer) -> bool {
    matches!(layer.geom.geom_type, GeomType::Dist(_))
}

/// Accumulates columns.
#[derive(Default)]
struct Table {
    cols: Vec<(&'static str, Vec<f64>)>,
}

impl Table {
    fn extend(&mut self, name: &'static str, v: &[f64]) {
        if let Some((_, c)) = self.cols.iter_mut().find(|(n, _)| *n == name) {
            c.extend_from_slice(v);
        } else {
            self.cols.push((name, v.to_vec()));
        }
    }

    fn get(&self, name: &str) -> &[f64] {
        self.cols.iter().find(|(n, _)| *n == name).map_or(&[], |(_, v)| v.as_slice())
    }

    fn set(&mut self, name: &'static str, v: Vec<f64>) {
        if let Some((_, c)) = self.cols.iter_mut().find(|(n, _)| *n == name) {
            *c = v;
        } else {
            self.cols.push((name, v));
        }
    }

    fn constant(&mut self, name: &'static str, v: f64, n: usize) {
        self.set(name, vec![v; n]);
    }
}

fn column<'a>(data: &'a DataFrame, name: &str) -> Result<&'a [DataValue]> {
    let cells = data.get(name).ok_or_else(|| Error::UnknownColumn(name.to_string()))?;
    // R cannot build a data.frame from columns of unequal length, so there is no ggplot2
    // behaviour to match: padding the short column with `NA` would be a guess.
    if cells.len() != data.nrow() {
        return Err(refuse(format!(
            "column `{name}` has {} cells but the data has {} rows (ragged columns)",
            cells.len(),
            data.nrow()
        )));
    }
    Ok(cells)
}

/// A numeric column, one entry per row. Missing is `NaN`; text and infinite values are refused.
fn numeric(data: &DataFrame, name: &str) -> Result<Vec<f64>> {
    let cells = column(data, name)?;
    cells
        .iter()
        .map(|cell| match cell {
            DataValue::Text(_) => Err(refuse(format!(
                "column `{name}` is text: a discrete position scale is not supported here"
            ))),
            // `NaN` is the missing value (R's `NA`); a missing cell reads the same way.
            other => match other.as_f32() {
                Some(v) if v.is_infinite() => {
                    Err(refuse(format!("column `{name}` holds an infinite value")))
                }
                v => Ok(v.map_or(f64::NAN, f64::from)),
            },
        })
        .collect()
}

/// A grouping key `(class, number, text)`: class 0 is a number, 1 text, 2 missing, so values sort
/// before `NA` (R's `order()`), numbers by value and text byte-wise.
type Key = (u8, f64, String);

fn key_order(a: &Key, b: &Key) -> std::cmp::Ordering {
    a.0.cmp(&b.0).then(cmp_na_last(a.1, b.1)).then_with(|| a.2.as_bytes().cmp(b.2.as_bytes()))
}

fn keys(data: &DataFrame, name: &str) -> Result<Vec<Key>> {
    let cells = column(data, name)?;
    Ok((0..data.nrow())
        .map(|i| match cells.get(i) {
            Some(DataValue::Text(s)) => (1, 0.0, s.clone()),
            Some(DataValue::Number(v)) if !v.is_nan() => (0, f64::from(*v), String::new()),
            _ => (2, 0.0, String::new()),
        })
        .collect())
}

fn is_text(data: &DataFrame, name: &str) -> Result<bool> {
    Ok(column(data, name)?.iter().any(|c| matches!(c, DataValue::Text(_))))
}

/// ggplot2's `add_group()`: `-1` without a discrete mapping, else the rank of the (sorted) value.
fn group_ids(data: &DataFrame, aes: &Aes) -> Result<Vec<i64>> {
    let nrow = data.nrow();
    let mut discrete: Vec<&str> = Vec::new();
    for name in [aes.fill.as_deref(), aes.color.as_deref()].into_iter().flatten() {
        if !is_text(data, name)? {
            return Err(refuse(format!(
                "`{name}` is a continuous colour or fill mapping: continuous colour scales are not supported"
            )));
        }
        if !discrete.contains(&name) {
            discrete.push(name);
        }
    }
    let name = match (&aes.group, discrete.as_slice()) {
        (Some(g), _) => g.as_str(),
        (None, []) => return Ok(vec![-1; nrow]),
        (None, [one]) => one,
        (None, _) => {
            return Err(refuse(
                "two discrete mappings without a `group`: their interaction is not supported",
            ))
        }
    };
    let ks = keys(data, name)?;
    let mut distinct = ks.clone();
    distinct.sort_by(key_order);
    distinct.dedup_by(|a, b| key_order(a, b).is_eq());
    Ok(ks
        .iter()
        .map(|k| {
            distinct
                .iter()
                .position(|d| key_order(d, k).is_eq())
                .map_or(-1, |p| i64::try_from(p + 1).unwrap_or(i64::MAX))
        })
        .collect())
}

/// Refuse every input `layout()` would otherwise ignore or misread: a `stat` other than the
/// geom's own, and an aesthetic the layer does not use (ggplot2 errors on a `y` for `stat_bin`
/// and `stat_density`, and keeps an error bar's `y` as a column, which this crate does not).
pub(super) fn check_inputs(kind: DistKind, aes: &Aes, stat: Option<&Stat>) -> Result<()> {
    let own_stat = matches!(
        (kind, stat),
        (_, None)
            | (DistKind::Errorbar, Some(Stat::Identity))
            | (DistKind::Density, Some(Stat::Density))
            | (DistKind::Boxplot, Some(Stat::Boxplot))
    );
    if !own_stat {
        return Err(Error::UnsupportedStat(
            "a stat other than the geom's own with a distribution layer: layout() computes only \
             stat_bin, stat_density, stat_boxplot and identity, each as its geom names it",
        ));
    }
    if aes.alpha.is_some() {
        return Err(refuse("a mapped `alpha` is not supported"));
    }
    let unused: [(&str, bool); 7] = [
        ("fill", aes.fill.is_some() && matches!(kind, DistKind::Freqpoly | DistKind::Errorbar)),
        ("size", aes.size.is_some() || aes.size_value.is_some()),
        ("shape", aes.shape.is_some()),
        ("label", aes.label.is_some()),
        ("y", aes.y.is_some() && kind != DistKind::Boxplot),
        ("ymin", aes.ymin.is_some() && kind != DistKind::Errorbar),
        ("ymax", aes.ymax.is_some() && kind != DistKind::Errorbar),
    ];
    if let Some((name, _)) = unused.iter().find(|(_, set)| *set) {
        return Err(refuse(format!(
            "the `{name}` aesthetic is not used by this layer and would be silently ignored"
        )));
    }
    Ok(())
}

fn paint_colour(aes: &Aes, default: Paint) -> Paint {
    if aes.color.is_some() {
        Paint::Mapped
    } else {
        aes.color_value.map_or(default, Paint::Fixed)
    }
}

fn paint_fill(aes: &Aes, default: Paint) -> Paint {
    if aes.fill.is_some() {
        Paint::Mapped
    } else {
        default
    }
}

/// Distinct group ids of `rows`, ascending.
fn groups_of(gid: &[i64], rows: &[usize]) -> Vec<i64> {
    let mut g: Vec<i64> = rows.iter().filter_map(|i| gid.get(*i).copied()).collect();
    g.sort_unstable();
    g.dedup();
    g
}

fn rows_in(gid: &[i64], rows: &[usize], g: i64) -> Vec<usize> {
    rows.iter().copied().filter(|i| gid.get(*i) == Some(&g)).collect()
}

fn pick(v: &[f64], rows: &[usize]) -> Vec<f64> {
    rows.iter().map(|i| v.get(*i).copied().unwrap_or(f64::NAN)).collect()
}

/// One layer's inputs.
struct Input<'a> {
    spec: &'a DistSpec,
    aes: Aes,
    data: &'a DataFrame,
    gid: Vec<i64>,
}

fn finite_range(v: &[f64]) -> Option<(f64, f64)> {
    v.iter().filter(|x| x.is_finite()).fold(None, |acc, &x| match acc {
        None => Some((x, x)),
        Some((lo, hi)) => Some((lo.min(x), hi.max(x))),
    })
}

fn x_name(inp: &Input<'_>) -> Result<String> {
    // A `y` mapping (a flipped histogram or density) is refused earlier, by `check_inputs`.
    inp.aes.x.clone().ok_or_else(|| refuse("this layer needs an `x` mapping"))
}

fn bins_for(spec: &DistSpec, range: (f64, f64)) -> Result<Bins> {
    if let Some(b) = &spec.breaks {
        if b.len() < 2 {
            return Err(refuse("`breaks` needs at least two values"));
        }
        let bins = bin_breaks(b.clone(), spec.closed)?;
        if bins.breaks.windows(2).any(|w| w[1] <= w[0]) {
            return Err(refuse("`breaks` must be strictly increasing"));
        }
        return Ok(bins);
    }
    if spec.boundary.is_some() && spec.center.is_some() {
        return Err(refuse("only one of `boundary` and `center` may be set"));
    }
    if let Some(w) = spec.binwidth {
        return bin_breaks_width(range, w, spec.center, spec.boundary, spec.closed);
    }
    bin_breaks_bins(range, spec.bins, spec.center, spec.boundary, spec.closed)
}

fn stat_bin(inp: &Input<'_>, range: (f64, f64), freqpoly: bool) -> Result<(Table, Vec<i64>)> {
    let xs = numeric(inp.data, &x_name(inp)?)?;
    let rows: Vec<usize> = (0..xs.len()).filter(|i| xs[*i].is_finite()).collect();
    if rows.is_empty() {
        return Err(Error::EmptyData);
    }
    let bins = bins_for(inp.spec, range)?;
    let groups = groups_of(&inp.gid, &rows);
    let nbin = bins.breaks.len().saturating_sub(1)
        + if freqpoly || inp.spec.pad.unwrap_or(false) { 2 } else { 0 };
    if nbin.saturating_mul(groups.len()) > MAX_OUTPUT_ROWS {
        return Err(refuse("the layer would build too many rows"));
    }
    let pad = freqpoly || inp.spec.pad.unwrap_or(false);
    let mut t = Table::default();
    let mut out_groups = Vec::new();
    for g in &groups {
        let gx = pick(&xs, &rows_in(&inp.gid, &rows, *g));
        let b = bin_vector(&gx, &bins, pad);
        t.extend("count", &b.count);
        t.extend("x", &b.x);
        t.extend("xmin", &b.xmin);
        t.extend("xmax", &b.xmax);
        if freqpoly {
            t.extend("width", &b.width);
        }
        t.extend("density", &b.density);
        t.extend("ncount", &b.ncount);
        t.extend("ndensity", &b.ndensity);
        out_groups.extend(std::iter::repeat(*g).take(b.count.len()));
    }
    let count = t.get("count").to_vec();
    t.set("y", count.clone());
    if !freqpoly {
        // position_stack(): per bin, the highest group id sits at the bottom.
        let per = count.len() / groups.len().max(1);
        let mut ymin = vec![0.0; count.len()];
        let mut ymax = vec![0.0; count.len()];
        for b in 0..per {
            let mut cum = 0.0;
            for gi in (0..groups.len()).rev() {
                let idx = gi * per + b;
                ymin[idx] = cum;
                cum += count[idx];
                ymax[idx] = cum;
            }
        }
        t.set("y", ymax.clone());
        t.set("ymin", ymin);
        t.set("ymax", ymax);
    }
    Ok((t, out_groups))
}

/// One group's density curve appended to `t`; returns the row count.
fn density_group(
    spec: &DistSpec,
    gx: &[f64],
    g: i64,
    range: (f64, f64),
    t: &mut Table,
) -> Result<usize> {
    if gx.len() < 2 {
        return Err(refuse(format!(
            "group {g} has fewer than 2 points: ggplot2 returns an NA row, which is not supported"
        )));
    }
    let raw = match spec.bw {
        Bandwidth::Nrd0 => bw_nrd0(gx),
        Bandwidth::Value(v) => v,
    };
    let bw = raw * spec.adjust;
    if !bw.is_finite() || bw <= 0.0 {
        return Err(refuse("the bandwidth must be a positive finite number"));
    }
    let (from, to) = if spec.trim { finite_range(gx).unwrap_or(range) } else { range };
    let (grid, y) = density(gx, bw, from, to, spec.n);
    let nx = gx.len() as f64;
    let ymax = y.iter().copied().filter(|v| v.is_finite()).fold(f64::NEG_INFINITY, f64::max);
    let scaled: Vec<f64> = y.iter().map(|v| v / ymax).collect();
    let count: Vec<f64> = y.iter().map(|v| v * nx).collect();
    t.extend("x", &grid);
    t.extend("density", &y);
    t.extend("scaled", &scaled);
    t.extend("ndensity", &scaled);
    t.extend("count", &count);
    t.extend("wdensity", &count);
    t.extend("n", &vec![nx; y.len()]);
    t.extend("y", &y);
    t.extend("ymax", &y);
    Ok(y.len())
}

fn stat_density(inp: &Input<'_>, range: (f64, f64)) -> Result<(Table, Vec<i64>)> {
    let spec = inp.spec;
    if spec.n == 0 || spec.n > MAX_DENSITY_N {
        return Err(refuse(format!("`n` must be between 1 and {MAX_DENSITY_N}")));
    }
    if !spec.adjust.is_finite() || spec.adjust <= 0.0 {
        return Err(refuse("`adjust` must be a positive finite number"));
    }
    let xs = numeric(inp.data, &x_name(inp)?)?;
    let rows: Vec<usize> = (0..xs.len()).filter(|i| xs[*i].is_finite()).collect();
    if rows.is_empty() {
        return Err(Error::EmptyData);
    }
    let groups = groups_of(&inp.gid, &rows);
    if spec.n.saturating_mul(groups.len()) > MAX_OUTPUT_ROWS {
        return Err(refuse("the layer would build too many rows"));
    }
    let mut t = Table::default();
    let mut out_groups = Vec::new();
    for g in &groups {
        let gx = pick(&xs, &rows_in(&inp.gid, &rows, *g));
        let len = density_group(spec, &gx, *g, range, &mut t)?;
        out_groups.extend(std::iter::repeat(*g).take(len));
    }
    let n = out_groups.len();
    t.constant("ymin", 0.0, n);
    t.constant("weight", 1.0, n);
    Ok((t, out_groups))
}

/// Per-group box statistics.
struct BoxStat {
    q: [f64; 5],
    notch: (f64, f64),
    outliers: Vec<f64>,
    ymin_final: f64,
    ymax_final: f64,
    group: i64,
}

/// One group's box statistics, centre x and width.
fn box_stat(gy: &[f64], gx: &[f64], g: i64, spec: &DistSpec, width: f64) -> (BoxStat, f64, f64) {
    let mut sorted = gy.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut q = [0.0; 5];
    for (slot, p) in q.iter_mut().zip([0.0, 0.25, 0.5, 0.75, 1.0]) {
        *slot = quantile7(&sorted, p);
    }
    let iqr = q[3] - q[1];
    let (lo_fence, hi_fence) = (q[1] - spec.coef * iqr, q[3] + spec.coef * iqr);
    let outside = |y: &f64| *y < lo_fence || *y > hi_fence;
    let outliers: Vec<f64> = gy.iter().copied().filter(outside).collect();
    if !outliers.is_empty() {
        let kept = [q[1], q[2], q[3]].into_iter().chain(gy.iter().copied().filter(|y| !outside(y)));
        let (lo, hi) = finite_range(&kept.collect::<Vec<_>>()).unwrap_or((q[0], q[4]));
        q[0] = lo;
        q[4] = hi;
    }
    let half = 1.58 * iqr / (gy.len() as f64).sqrt();
    let (xlo, xhi) = finite_range(gx).unwrap_or((0.0, 0.0));
    let mut u = gx.to_vec();
    u.sort_by(f64::total_cmp);
    u.dedup();
    let w = if u.len() > 1 { (xhi - xlo) * 0.9 } else { width };
    let ymin_final = outliers.iter().copied().fold(q[0], f64::min);
    let ymax_final = outliers.iter().copied().fold(q[4], f64::max);
    let stat = BoxStat {
        q,
        notch: (q[2] + half, q[2] - half),
        outliers,
        ymin_final,
        ymax_final,
        group: g,
    };
    (stat, (xlo + xhi) / 2.0, w)
}

fn box_table(
    boxes: &[Box2],
    stats: &[(BoxStat, f64, f64)],
    padded: bool,
) -> (Table, Vec<i64>, Vec<Vec<f64>>) {
    let mut t = Table::default();
    let mut groups = Vec::new();
    let mut outliers = Vec::new();
    for (b, (s, _, _)) in boxes.iter().filter_map(|b| stats.get(b.idx).map(|st| (b, st))) {
        for (name, v) in [
            ("ymin", s.q[0]),
            ("lower", s.q[1]),
            ("middle", s.q[2]),
            ("upper", s.q[3]),
            ("ymax", s.q[4]),
            ("notchupper", s.notch.0),
            ("notchlower", s.notch.1),
            ("x", b.x),
            ("ymin_final", s.ymin_final),
            ("ymax_final", s.ymax_final),
            ("xmin", b.xmin),
            ("xmax", b.xmax),
            ("order", b.order),
        ] {
            t.extend(name, &[v]);
        }
        if !padded {
            t.extend("xid", &[b.xid as f64]);
            t.extend("newx", &[b.newx]);
            t.extend("new_width", &[b.new_width]);
        }
        groups.push(s.group);
        outliers.push(s.outliers.clone());
    }
    (t, groups, outliers)
}

fn stat_boxplot(inp: &Input<'_>) -> Result<(Table, Vec<i64>, Vec<Vec<f64>>)> {
    let spec = inp.spec;
    if !spec.coef.is_finite() || spec.coef < 0.0 {
        return Err(refuse("`coef` must be a non-negative finite number"));
    }
    let y_name = inp.aes.y.clone().ok_or_else(|| {
        refuse("geom_boxplot needs a `y` mapping (an x-only box plot is flipped and not supported)")
    })?;
    let ys = numeric(inp.data, &y_name)?;
    let xs = match &inp.aes.x {
        Some(n) => numeric(inp.data, n)?,
        None => vec![0.0; ys.len()],
    };
    let width = spec.width.unwrap_or_else(|| resolution(&xs, true) * 0.75);
    let rows: Vec<usize> =
        (0..ys.len()).filter(|i| ys[*i].is_finite() && xs[*i].is_finite()).collect();
    if rows.is_empty() {
        return Err(Error::EmptyData);
    }
    let mut stats: Vec<(BoxStat, f64, f64)> = Vec::new();
    for g in groups_of(&inp.gid, &rows) {
        let grows = rows_in(&inp.gid, &rows, g);
        let gy = pick(&ys, &grows);
        let gx = pick(&xs, &grows);
        stats.push(box_stat(&gy, &gx, g, spec, width));
    }
    let boxes: Vec<Box2> = stats
        .iter()
        .enumerate()
        .map(|(idx, (s, x, w))| Box2 {
            idx,
            x: *x,
            xmin: x - w / 2.0,
            xmax: x + w / 2.0,
            group: s.group,
            order: 0.0,
            xid: 0,
            newx: 0.0,
            new_width: 0.0,
        })
        .collect();
    let (boxes, padded) = dodge2(boxes, 0.1)?;
    let (mut t, groups, outliers) = box_table(&boxes, &stats, padded);
    let n = groups.len();
    t.constant("weight", 1.0, n);
    t.constant("size", 1.5, n);
    t.constant("shape", 19.0, n);
    t.constant("width", spec.width.unwrap_or(0.9), n);
    Ok((t, groups, outliers))
}

fn stat_errorbar(inp: &Input<'_>) -> Result<(Table, Vec<i64>)> {
    let need = |o: &Option<String>, what: &str| {
        o.clone().ok_or_else(|| {
            refuse(format!(
                "geom_errorbar needs a `{what}` mapping (a horizontal error bar is not supported)"
            ))
        })
    };
    let x = numeric(inp.data, &need(&inp.aes.x, "x")?)?;
    let ymin = numeric(inp.data, &need(&inp.aes.ymin, "ymin")?)?;
    let ymax = numeric(inp.data, &need(&inp.aes.ymax, "ymax")?)?;
    if x.is_empty() {
        return Err(Error::EmptyData);
    }
    // ggplot2 4.0.3 `GeomErrorbar$setup_data`: the width comes from this layer's own x.
    let width = inp.spec.width.unwrap_or_else(|| resolution(&x, false) * 0.9);
    if !width.is_finite() || width < 0.0 {
        return Err(refuse("`width` must be a non-negative finite number"));
    }
    let mut t = Table::default();
    t.set("x", x.clone());
    t.set("ymin", ymin);
    t.set("ymax", ymax);
    t.set("xmin", x.iter().map(|v| v - width / 2.0).collect());
    t.set("xmax", x.iter().map(|v| v + width / 2.0).collect());
    t.constant("width", inp.spec.width.unwrap_or(0.9), x.len());
    Ok((t, inp.gid.clone()))
}

/// Largest magnitude a trained position scale may reach. Break search over limits near 1e200
/// does not return (reviewed on #6976: `binwidth(1e300)`, `bw(1e-300)`), so a stat that produces
/// such limits is refused before the scale is trained. Below this the search takes microseconds.
pub(super) const MAX_SCALE_MAGNITUDE: f64 = 1e100;

const X_AES: [&str; 5] = ["x", "xmin", "xmax", "xmin_final", "xmax_final"];
const Y_AES: [&str; 8] =
    ["y", "ymin", "ymax", "lower", "middle", "upper", "ymin_final", "ymax_final"];

fn train(layers: &[BuiltDist], names: &[&str], axis: &str) -> Result<PositionScale> {
    let mut acc: Option<(f64, f64)> = None;
    for l in layers {
        for n in names {
            if let Some(r) = l.col(n).and_then(finite_range) {
                acc = Some(acc.map_or(r, |(lo, hi)| (lo.min(r.0), hi.max(r.1))));
            }
        }
    }
    let (lo, hi) = acc.ok_or(Error::EmptyData)?;
    if lo.abs() > MAX_SCALE_MAGNITUDE || hi.abs() > MAX_SCALE_MAGNITUDE {
        return Err(Error::ScaleDomain(format!(
            "the {axis} scale would span ({lo}, {hi}), beyond +-{MAX_SCALE_MAGNITUDE:e}: break search is refused at that size"
        )));
    }
    if lo >= hi {
        return Err(Error::ScaleDomain(format!(
            "the {axis} scale range is degenerate ({lo}); ggplot2 widens it, this crate does not yet"
        )));
    }
    Ok(PositionScale::continuous((lo, hi)))
}

fn check_scope(coord: &Coord, facet: &Facet) -> Result<()> {
    match coord {
        Coord::Cartesian { xlim: None, ylim: None, flip: false } => {}
        Coord::Cartesian { flip: true, .. } => {
            return Err(Error::UnsupportedCoord("coord_flip with distribution layers"))
        }
        Coord::Cartesian { .. } => {
            return Err(Error::UnsupportedCoord("coord_cartesian limits with distribution layers"))
        }
        Coord::Polar { .. } => {
            return Err(Error::UnsupportedCoord("coord_polar with distribution layers"))
        }
        Coord::Fixed { .. } => {
            return Err(Error::UnsupportedCoord("coord_fixed with distribution layers"))
        }
    }
    if !matches!(facet, Facet::None) {
        return Err(refuse("facets with distribution layers are not supported"));
    }
    Ok(())
}

/// Build a plot made only of distribution layers.
///
/// # Errors
///
/// [`Error::UnsupportedCoord`] for any coordinate system but plain Cartesian,
/// [`Error::Rendering`] for out-of-scope input (see the module docs), [`Error::UnknownColumn`],
/// [`Error::EmptyData`] when no row survives, [`Error::ScaleDomain`] for a degenerate scale.
pub(super) fn layout(
    plot_data: &DataFrame,
    plot_aes: &Aes,
    layers: &[Layer],
    coord: &Coord,
    facet: &Facet,
) -> Result<BuiltLayout> {
    if !layers.iter().all(is_dist) {
        return Err(refuse("distribution layers cannot be combined with other geoms yet"));
    }
    check_scope(coord, facet)?;
    let mut inputs = Vec::new();
    for layer in layers {
        let GeomType::Dist(spec) = &layer.geom.geom_type else { continue };
        let data = layer.data.as_ref().unwrap_or(plot_data);
        let aes = plot_aes.merge(&layer.aes);
        check_inputs(spec.kind, &aes, layer.geom.stat.as_ref())?;
        let gid = group_ids(data, &aes)?;
        inputs.push(Input { spec, aes, data, gid });
    }
    // The pre-stat x scale: every finite x of every layer, which sets stat_bin's and
    // stat_density's range.
    let mut all_x = Vec::new();
    for inp in &inputs {
        if let Some(n) = &inp.aes.x {
            all_x.extend(numeric(inp.data, n)?);
        }
    }
    let range = finite_range(&all_x);
    let mut built = Vec::new();
    for inp in &inputs {
        built.push(build_one(inp, range)?);
    }
    let x = train(&built, &X_AES, "x")?;
    let y = train(&built, &Y_AES, "y")?;
    let out = built
        .into_iter()
        .map(|d| BuiltLayer {
            path: None,
            x: d.col("x").map(<[f64]>::to_vec).unwrap_or_default(),
            y: d.col("y").map(<[f64]>::to_vec).unwrap_or_default(),
            points: None,
            dist: Some(d),
        })
        .collect();
    Ok(BuiltLayout { layers: out, x, y })
}

fn build_one(inp: &Input<'_>, range: Option<(f64, f64)>) -> Result<BuiltDist> {
    let spec = inp.spec;
    let need_range = || range.ok_or(Error::EmptyData);
    if matches!(spec.kind, DistKind::Histogram | DistKind::Freqpoly | DistKind::Density) {
        x_name(inp)?;
    }
    let (mut t, group, outliers) = match spec.kind {
        DistKind::Histogram => {
            let (t, g) = stat_bin(inp, need_range()?, false)?;
            (t, g, Vec::new())
        }
        DistKind::Freqpoly => {
            let (t, g) = stat_bin(inp, need_range()?, true)?;
            (t, g, Vec::new())
        }
        DistKind::Density => {
            let (t, g) = stat_density(inp, need_range()?)?;
            (t, g, Vec::new())
        }
        DistKind::Boxplot => stat_boxplot(inp)?,
        DistKind::Errorbar => {
            let (t, g) = stat_errorbar(inp)?;
            (t, g, Vec::new())
        }
    };
    let n = group.len();
    let lw = spec.linewidth.unwrap_or(0.5);
    if !lw.is_finite() || lw < 0.0 {
        return Err(refuse("`linewidth` must be a non-negative finite number"));
    }
    let alpha = inp.aes.alpha_value.map_or(f64::NAN, f64::from);
    t.constant("linewidth", lw, n);
    t.constant("linetype", 1.0, n);
    t.constant("alpha", alpha, n);
    t.constant("panel", 1.0, n);
    let (geom, colour, fill) = match spec.kind {
        DistKind::Histogram => (
            "GeomBar",
            paint_colour(&inp.aes, Paint::Na),
            paint_fill(&inp.aes, Paint::Fixed(Rgba::new(89, 89, 89, 255))),
        ),
        DistKind::Freqpoly => {
            ("GeomPath", paint_colour(&inp.aes, Paint::Fixed(Rgba::BLACK)), Paint::Na)
        }
        DistKind::Density => (
            "GeomDensity",
            paint_colour(&inp.aes, Paint::Fixed(Rgba::BLACK)),
            paint_fill(&inp.aes, Paint::Na),
        ),
        DistKind::Boxplot => (
            "GeomBoxplot",
            paint_colour(&inp.aes, Paint::Fixed(Rgba::new(51, 51, 51, 255))),
            paint_fill(&inp.aes, Paint::Fixed(Rgba::WHITE)),
        ),
        DistKind::Errorbar => {
            ("GeomErrorbar", paint_colour(&inp.aes, Paint::Fixed(Rgba::BLACK)), Paint::Na)
        }
    };
    Ok(BuiltDist {
        geom,
        cols: t.cols,
        group,
        outliers,
        colour: vec![colour; n],
        fill: vec![fill; n],
    })
}
