//! ggplot2 4.0.3's distribution geoms through [`GGPlot::layout`](super::GGPlot::layout)
//! (PLOT-GG-001, paiml/aprender#6909): `geom_histogram`, `geom_freqpoly`, `geom_density`,
//! `geom_boxplot` and `geom_errorbar`, each with its stat and position.
//!
//! This slice holds the layer types and the input checks `layout()` runs before any stat: the
//! group ids (ggplot2's `add_group()`), the refusals below, and the column reads. The stats, the
//! positions and the scales retrained on their output land with their rows; until then
//! `layout` returns `NOT_BUILT` for input that passes every check, and the renderers refuse
//! every distribution layer with `NOT_DRAWN`.
//!
//! Out of scope, and refused with a typed error instead of drawn wrongly: facets, a coordinate
//! system other than plain Cartesian, text `x` (a discrete scale), `y`-only histograms and
//! densities (flipped orientation), two or more discrete mappings without a `group`, a continuous
//! `colour` or `fill` mapping (the palette is not resolved), non-finite input, layers mixed with
//! non-distribution layers, and a `scale_x`/`scale_y` spec (refused in `GGPlot::layout`, which
//! owns the scale specs).

use crate::error::{Error, Result};

use super::aes::Aes;
use super::build::{BuiltBins, BuiltDensity, BuiltLayer, BuiltLayout, PositionScale};
use super::coord::Coord;
use super::data::{DataFrame, DataValue};
use super::dist_math::{
    cmp_na_last, stat_bin_groups, stat_bin_padded, stat_density, BinArgs, DENSITY_POINTS,
};
use super::facet::Facet;
use super::geom::{Geom, GeomType};
use super::ggplot::Layer;
use super::position::{apply as apply_position, BarRow, Position};
use super::stat::Stat;
use super::violin;

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
    /// `geom_violin()`: `stat_ydensity`, `position_dodge`.
    Violin,
}

/// `scale =` of `geom_violin()`: how the violins' widths are scaled against each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolinScale {
    /// `"area"` (the default): every violin has the same area.
    Area,
    /// `"count"`: areas are proportional to the number of observations.
    Count,
    /// `"width"`: every violin has the same maximum width.
    Width,
}

/// A distribution layer's parameters, named as ggplot2 names them.
#[derive(Debug, Clone, PartialEq)]
pub struct DistSpec {
    /// Which layer.
    pub kind: DistKind,
    /// `bins` (default 30).
    pub bins: usize,
    /// `width` of a box or an error bar.
    pub width: Option<f64>,
    /// `alpha`, the layer's fixed alpha (`geom_histogram(alpha = 0.5)`), set by [`Geom::alpha`].
    pub alpha: Option<f64>,
    /// `binwidth` of `stat_bin`; wins over `bins`.
    pub binwidth: Option<f64>,
    /// `boundary` of `stat_bin`: an edge of a bin.
    pub boundary: Option<f64>,
    /// `center` of `stat_bin`: the middle of a bin.
    pub center: Option<f64>,
    /// `trim` of `stat_ydensity` (default `true`).
    pub trim: bool,
    /// `adjust` of `stat_ydensity` (default 1).
    pub adjust: f64,
    /// `scale` of `stat_ydensity`.
    pub violin_scale: ViolinScale,
}

impl DistSpec {
    fn new(kind: DistKind) -> Self {
        Self {
            kind,
            bins: 30,
            width: None,
            alpha: None,
            binwidth: None,
            boundary: None,
            center: None,
            trim: true,
            adjust: 1.0,
            violin_scale: ViolinScale::Area,
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

    /// ggplot2 4.0.3 `geom_violin()`, as built by `layout()`.
    #[must_use]
    pub fn gg_violin() -> Self {
        Self::dist(DistKind::Violin)
    }
}

/// `stat_bin` and `stat_ydensity` parameters on a distribution layer; no-ops on any other geom.
/// `trim`, `adjust` and `violin_scale` on a distribution layer other than `geom_violin` are
/// refused by `layout()`, not ignored.
impl Geom {
    fn with_spec(mut self, f: impl FnOnce(&mut DistSpec)) -> Self {
        if let GeomType::Dist(spec) = &mut self.geom_type {
            f(spec);
        }
        self
    }

    /// `binwidth =` of `stat_bin`.
    #[must_use]
    pub fn binwidth(self, w: f64) -> Self {
        self.with_spec(|s| s.binwidth = Some(w))
    }

    /// `boundary =` of `stat_bin`.
    #[must_use]
    pub fn boundary(self, b: f64) -> Self {
        self.with_spec(|s| s.boundary = Some(b))
    }

    /// `center =` of `stat_bin`.
    #[must_use]
    pub fn center(self, c: f64) -> Self {
        self.with_spec(|s| s.center = Some(c))
    }

    /// `trim =` of `stat_ydensity` (`geom_violin`).
    #[must_use]
    pub fn trim(self, trim: bool) -> Self {
        self.with_spec(|s| s.trim = trim)
    }

    /// `adjust =` of `stat_ydensity` (`geom_violin`): a multiplier on the bandwidth.
    #[must_use]
    pub fn adjust(self, adjust: f64) -> Self {
        self.with_spec(|s| s.adjust = adjust)
    }

    /// `scale =` of `stat_ydensity` (`geom_violin`).
    #[must_use]
    pub fn violin_scale(self, scale: ViolinScale) -> Self {
        self.with_spec(|s| s.violin_scale = scale)
    }
}

pub(super) fn refuse(msg: impl Into<String>) -> Error {
    Error::Rendering(msg.into())
}

/// `true` when this layer is a distribution layer.
pub(super) fn is_dist(layer: &Layer) -> bool {
    matches!(layer.geom.geom_type, GeomType::Dist(_))
}

pub(super) fn column<'a>(data: &'a DataFrame, name: &str) -> Result<&'a [DataValue]> {
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
pub(super) fn numeric(data: &DataFrame, name: &str) -> Result<Vec<f64>> {
    let v = numeric_lenient(data, name)?;
    if v.iter().any(|x| x.is_infinite()) {
        return Err(refuse(format!("column `{name}` holds an infinite value")));
    }
    Ok(v)
}

/// [`numeric`] that lets an infinite value through, for a stat whose ggplot2 original drops
/// non-finite rows (`remove_missing(finite = TRUE)`) and so can be matched by dropping them too.
pub(super) fn numeric_lenient(data: &DataFrame, name: &str) -> Result<Vec<f64>> {
    let cells = column(data, name)?;
    cells
        .iter()
        .map(|cell| match cell {
            DataValue::Text(_) => Err(refuse(format!(
                "column `{name}` is text: a discrete position scale is not supported here"
            ))),
            // `NaN` is the missing value (R's `NA`); a missing cell reads the same way.
            other => Ok(other.as_f32().map_or(f64::NAN, f64::from)),
        })
        .collect()
}

/// A grouping key `(class, number, text)`: class 0 is a number, 1 text, 2 missing, so values sort
/// before `NA` (R's `order()`), numbers by value and text byte-wise.
pub(super) type Key = (u8, f64, String);

pub(super) fn key_order(a: &Key, b: &Key) -> std::cmp::Ordering {
    a.0.cmp(&b.0).then(cmp_na_last(a.1, b.1)).then_with(|| a.2.as_bytes().cmp(b.2.as_bytes()))
}

pub(super) fn keys(data: &DataFrame, name: &str) -> Result<Vec<Key>> {
    let cells = column(data, name)?;
    Ok((0..data.nrow())
        .map(|i| match cells.get(i) {
            Some(DataValue::Text(s)) => (1, 0.0, s.clone()),
            Some(DataValue::Number(v)) if !v.is_nan() => (0, f64::from(*v), String::new()),
            _ => (2, 0.0, String::new()),
        })
        .collect())
}

pub(super) fn is_text(data: &DataFrame, name: &str) -> Result<bool> {
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
    // `distinct` is sorted by `key_order`, a total order, so each key's rank is found by binary
    // search: O(n log g), where a linear scan per row cost O(n * g) (reviewed on #6976, F16).
    Ok(ks
        .iter()
        .map(|k| {
            distinct
                .binary_search_by(|d| key_order(d, k))
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
        ("y", aes.y.is_some() && !matches!(kind, DistKind::Boxplot | DistKind::Violin)),
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

/// `trim`, `adjust` and `scale` belong to `stat_ydensity`. Set to anything but their default on a
/// layer that does not honour them they would be silently ignored, so they are refused by name.
/// (`geom_density(adjust = 0.5)` is valid in ggplot2 but not built here.)
fn check_violin_only(spec: &DistSpec) -> Result<()> {
    if spec.kind == DistKind::Violin {
        return Ok(());
    }
    let set = [
        ("adjust", spec.adjust != 1.0),
        ("trim", !spec.trim),
        ("violin_scale", spec.violin_scale != ViolinScale::Area),
    ];
    match set.iter().find(|(_, changed)| *changed) {
        Some((name, _)) => Err(refuse(format!(
            "`{name}` is a geom_violin parameter: this layer does not honour it, and it would be \
             silently ignored"
        ))),
        None => Ok(()),
    }
}

fn check_scope(coord: &Coord, facet: &Facet) -> Result<()> {
    let refused = match coord {
        Coord::Cartesian { xlim: None, ylim: None, flip: false } => None,
        Coord::Cartesian { flip: true, .. } => Some("coord_flip with distribution layers"),
        Coord::Cartesian { .. } => Some("coord_cartesian limits with distribution layers"),
        Coord::Polar { .. } => Some("coord_polar with distribution layers"),
        Coord::Fixed { .. } => Some("coord_fixed with distribution layers"),
    };
    if let Some(what) = refused {
        return Err(Error::UnsupportedCoord(what));
    }
    if !matches!(facet, Facet::None) {
        return Err(refuse("facets with distribution layers are not supported"));
    }
    Ok(())
}

/// What [`layout`] returns for input that passes every check: the stats are not built yet.
pub(super) const NOT_BUILT: &str =
    "distribution stats (stat_bin, stat_density, stat_boxplot, identity) are not built yet";

/// What `to_framebuffer`/`to_svg` return for a distribution layer, instead of an empty panel.
pub(super) const NOT_DRAWN: &str = "distribution layers are not drawn by the renderer yet";

/// Check a plot made only of distribution layers, as `ggplot_build()` would before its stats.
///
/// # Errors
///
/// [`Error::UnsupportedCoord`] for any coordinate system but plain Cartesian,
/// [`Error::UnsupportedStat`] for a stat other than the geom's own, [`Error::Rendering`] for
/// out-of-scope input (see the module docs), [`Error::UnknownColumn`], and
/// [`Error::UnsupportedLayer`] with [`NOT_BUILT`] when every check passes.
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
    let mut checked = Vec::with_capacity(layers.len());
    for layer in layers {
        let GeomType::Dist(spec) = &layer.geom.geom_type else {
            continue;
        };
        let data = layer.data.as_ref().unwrap_or(plot_data);
        let aes = plot_aes.merge(&layer.aes);
        check_inputs(spec.kind, &aes, layer.geom.stat.as_ref())?;
        check_violin_only(spec)?;
        let groups = group_ids(data, &aes)?;
        // A violin reads its own `x` (text is a discrete scale there) and `y`.
        if spec.kind != DistKind::Violin {
            for name in [&aes.x, &aes.y, &aes.ymin, &aes.ymax].into_iter().flatten() {
                numeric(data, name)?;
            }
        }
        checked.push((spec, data, aes, groups));
    }
    match checked.as_slice() {
        [(spec, data, aes, groups)] if spec.kind == DistKind::Histogram => {
            histogram(spec, data, aes, groups)
        }
        [(spec, data, aes, groups)] if spec.kind == DistKind::Density => density(data, aes, groups),
        [(spec, data, aes, groups)] if spec.kind == DistKind::Freqpoly => {
            freqpoly(spec, data, aes, groups)
        }
        [(spec, data, aes, _)] if spec.kind == DistKind::Violin => violin::violin(spec, data, aes),
        _ => Err(Error::UnsupportedLayer(NOT_BUILT)),
    }
}

/// The rows of `x` that are present, with their group ids (ggplot2 drops missing `x` before the
/// stat, after `add_group()` has numbered the groups).
fn present_rows(x: Vec<f64>, groups: &[i64]) -> (Vec<f64>, Vec<i64>) {
    x.into_iter().zip(groups).filter(|(v, _)| !v.is_nan()).map(|(v, g)| (v, *g)).unzip()
}

/// Every group's bins as one set of columns, in group order (the row order of `stat_bin`'s output).
fn concat_bins(parts: Vec<BuiltBins>) -> BuiltBins {
    let mut all = BuiltBins {
        x: vec![],
        xmin: vec![],
        xmax: vec![],
        width: vec![],
        count: vec![],
        density: vec![],
        ymin: vec![],
        ymax: vec![],
        group: vec![],
    };
    for p in parts {
        all.x.extend(p.x);
        all.xmin.extend(p.xmin);
        all.xmax.extend(p.xmax);
        all.width.extend(p.width);
        all.count.extend(p.count);
        all.density.extend(p.density);
        all.ymin.extend(p.ymin);
        all.ymax.extend(p.ymax);
        all.group.extend(p.group);
    }
    all
}

/// `position_stack()` on binned counts, through the bar rows' own stacking. Returns the stacked
/// `y` (the top of each bar) and writes `ymin`/`ymax` back. A layer with one group has no
/// repeated `x`, so the position leaves it as `stat_bin` made it.
fn stack_bins(bins: &mut BuiltBins) -> Vec<f64> {
    let mut ids: Vec<i64> = bins.group.clone();
    ids.sort_unstable();
    ids.dedup();
    let rank = |g: i64| u32::try_from(ids.partition_point(|i| *i < g) + 1).unwrap_or(u32::MAX);
    let mut rows: Vec<BarRow> = (0..bins.x.len())
        .map(|i| BarRow {
            x: bins.x[i],
            y: bins.count[i],
            ymin: 0.0,
            ymax: bins.count[i],
            xmin: bins.xmin[i],
            xmax: bins.xmax[i],
            width: bins.width[i],
            group: rank(bins.group[i]),
            count: Some(bins.count[i]),
            prop: None,
        })
        .collect();
    apply_position(&mut rows, Position::stack());
    bins.ymin = rows.iter().map(|r| r.ymin).collect();
    bins.ymax = rows.iter().map(|r| r.ymax).collect();
    rows.iter().map(|r| r.y).collect()
}

/// One `geom_histogram()`: `stat_bin` (`pad = FALSE`) with shared edges across groups, stacked
/// from 0 by `position_stack()`. The x scale is trained on the bin edges and the y scale on 0 and
/// the tallest stack, as ggplot2 retrains them on the stat's output.
fn histogram(spec: &DistSpec, data: &DataFrame, aes: &Aes, groups: &[i64]) -> Result<BuiltLayout> {
    let name = aes.x.as_deref().ok_or_else(|| refuse("geom_histogram needs an `x` aesthetic"))?;
    let (xs, ids) = present_rows(numeric(data, name)?, groups);
    let args = BinArgs {
        bins: spec.bins,
        binwidth: spec.binwidth,
        boundary: spec.boundary,
        center: spec.center,
    };
    let mut bins = concat_bins(stat_bin_groups(&xs, &ids, &args)?);
    let y = stack_bins(&mut bins);
    let top = bins.ymax.iter().copied().fold(0.0_f64, f64::max);
    let (lo, hi) = (
        bins.xmin.iter().copied().fold(f64::INFINITY, f64::min),
        bins.xmax.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
    let layer = BuiltLayer {
        x: bins.x.clone(),
        y,
        points: None,
        path: None,
        bins: Some(bins),
        density: None,
        violin: None,
    };
    Ok(BuiltLayout {
        layers: vec![layer],
        x: PositionScale::continuous((lo, hi)),
        y: PositionScale::continuous((0.0, top)),
    })
}

/// One ungrouped `geom_freqpoly()`: `stat_bin` with `pad = TRUE`, drawn as a path through the bin
/// centres. The scales are retrained on the padded edges and on 0 and the largest count.
fn freqpoly(spec: &DistSpec, data: &DataFrame, aes: &Aes, groups: &[i64]) -> Result<BuiltLayout> {
    if groups.iter().any(|g| *g != -1) {
        return Err(refuse("grouped frequency polygons are not built yet"));
    }
    let name = aes.x.as_deref().ok_or_else(|| refuse("geom_freqpoly needs an `x` aesthetic"))?;
    let xs: Vec<f64> = numeric(data, name)?.into_iter().filter(|v| !v.is_nan()).collect();
    let args = BinArgs {
        bins: spec.bins,
        binwidth: spec.binwidth,
        boundary: spec.boundary,
        center: spec.center,
    };
    let bins = stat_bin_padded(&xs, &args)?;
    let top = bins.count.iter().copied().fold(0.0_f64, f64::max);
    let (lo, hi) =
        (bins.xmin.first().copied().unwrap_or(0.0), bins.xmax.last().copied().unwrap_or(0.0));
    let layer = BuiltLayer {
        x: bins.x.clone(),
        y: bins.count.clone(),
        points: None,
        path: None,
        bins: Some(bins),
        density: None,
        violin: None,
    };
    Ok(BuiltLayout {
        layers: vec![layer],
        x: PositionScale::continuous((lo, hi)),
        y: PositionScale::continuous((0.0, top)),
    })
}

/// One ungrouped `geom_density()`: `stat_density` over the x range, then `GeomArea`'s `ymin = 0`
/// retrains the y scale, so it spans 0 to the largest density.
fn density(data: &DataFrame, aes: &Aes, groups: &[i64]) -> Result<BuiltLayout> {
    if groups.iter().any(|g| *g != -1) {
        return Err(refuse("grouped densities are not built yet"));
    }
    let name = aes.x.as_deref().ok_or_else(|| refuse("geom_density needs an `x` aesthetic"))?;
    // ggplot2 drops missing values (with a warning) before the stat.
    let xs: Vec<f64> = numeric(data, name)?.into_iter().filter(|v| !v.is_nan()).collect();
    let lo = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let cols = stat_density(&xs, lo, hi, DENSITY_POINTS)?;
    let top = cols.density.iter().copied().fold(0.0_f64, f64::max);
    let layer = BuiltLayer {
        x: cols.x.clone(),
        y: cols.density.clone(),
        points: None,
        path: None,
        bins: None,
        density: Some(BuiltDensity {
            scaled: cols.scaled,
            count: cols.count,
            wdensity: cols.wdensity,
            n: cols.n,
        }),
        violin: None,
    };
    Ok(BuiltLayout {
        layers: vec![layer],
        x: PositionScale::continuous((lo, hi)),
        y: PositionScale::continuous((0.0, top)),
    })
}

#[cfg(test)]
mod tests {
    use super::super::{scale_continuous, GGPlot};
    use super::*;

    fn small() -> DataFrame {
        DataFrame::from_xy(&[1.0, 2.0, 3.0, 4.0, 9.0], &[1.0, 2.0, 3.0, 4.0, 9.0])
    }

    fn all_geoms() -> Vec<(&'static str, Geom, Aes)> {
        vec![
            ("hist", Geom::gg_histogram(), Aes::new().x("x")),
            ("freq", Geom::gg_freqpoly(), Aes::new().x("x")),
            ("dens", Geom::gg_density(), Aes::new().x("x")),
            ("box", Geom::gg_boxplot(), Aes::new().y("x")),
            ("err", Geom::gg_errorbar(), Aes::new().x("x").ymin("x").ymax("x")),
        ]
    }

    fn plot(data: DataFrame, aes: Aes, geom: Geom) -> GGPlot {
        GGPlot::new().data(data).aes(aes).layer(Layer::new(geom))
    }

    fn run(p: GGPlot) -> Result<BuiltLayout> {
        p.build().and_then(|p| p.layout())
    }

    fn not_built(r: &Result<BuiltLayout>) -> bool {
        matches!(r, Err(Error::UnsupportedLayer(m)) if *m == NOT_BUILT)
    }

    /// `x = y = 1..5` with text `h` and numeric `n` (`NA` third), as in the R probe.
    fn probe() -> DataFrame {
        let mut d = small();
        d.add_column_str("h", &["b", "B", "a", "B", "b"]);
        d.add_column_str("f", &["b", "a", "b", "a", "a"]);
        d.add_column_f32("n", &[3.0, 1.0, f32::NAN, 3.0, 1.0]);
        d
    }

    /// `layer_data()$group` from R 4.6.2 / ggplot2 4.0.3 under `LC_ALL=C` (bytewise collation).
    #[test]
    fn group_ids_match_ggplot2() {
        let d = probe();
        let cases: [(Aes, [i64; 5]); 5] = [
            (Aes::new(), [-1; 5]),
            (Aes::new().color("h"), [3, 1, 2, 1, 3]),
            (Aes::new().fill("h").color("h"), [3, 1, 2, 1, 3]),
            (Aes::new().group("n"), [2, 1, 3, 2, 1]),
            (Aes::new().group("n").fill("h"), [2, 1, 3, 2, 1]),
        ];
        for (aes, want) in cases {
            assert_eq!(group_ids(&d, &aes).expect("groups"), want, "{aes:?}");
        }
        // R builds the fill:colour interaction (4 1 2 1 3); it is refused, not approximated.
        assert!(group_ids(&d, &Aes::new().fill("h").color("f")).is_err());
        assert!(group_ids(&d, &Aes::new().fill("n")).is_err(), "continuous fill");
    }

    /// F16: grouping is `O(n log n)`; a linear search per row takes minutes at 100k groups.
    #[test]
    fn one_hundred_thousand_groups_check_in_bounded_time() {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let k: Vec<f32> = (0..100_000_u32).map(|i| i as f32).collect();
            let mut d = DataFrame::from_xy(&k, &k);
            d.add_column_f32("k", &k);
            let a = Aes::new().x("x").ymin("y").ymax("y").group("k");
            let _ = tx.send(not_built(&run(plot(d, a, Geom::gg_errorbar()))));
        });
        let ok = rx.recv_timeout(std::time::Duration::from_secs(30)).expect("returned in time");
        assert!(ok);
    }

    /// Checked input is `NOT_BUILT`; F19: the renderers refuse it instead of an empty panel.
    #[test]
    fn checked_input_is_not_built_and_not_drawn() {
        let drawn = |r: Result<()>| matches!(r, Err(Error::UnsupportedLayer(m)) if m == NOT_DRAWN);
        for (name, g, a) in all_geoms() {
            let p = plot(small(), a, g).build().expect("builds");
            // The histogram, the frequency polygon and the density are built; the others are not.
            assert_eq!(not_built(&p.layout()), !matches!(name, "hist" | "freq" | "dens"), "{name}");
            assert!(drawn(p.to_framebuffer().map(drop)), "{name} framebuffer");
            assert!(drawn(p.to_svg().map(drop)), "{name} svg");
        }
    }

    #[test]
    fn unused_aesthetics_and_foreign_stats_are_refused() {
        let unused = [
            ("alpha", Geom::gg_histogram(), Aes::new().x("x").alpha("y")),
            ("size", Geom::gg_histogram(), Aes::new().x("x").size("y")),
            ("size_value", Geom::gg_density(), Aes::new().x("x").size_value(2.0)),
            ("shape", Geom::gg_histogram(), Aes::new().x("x").shape("y")),
            ("label", Geom::gg_histogram(), Aes::new().x("x").label("y")),
            ("y", Geom::gg_histogram(), Aes::new().x("x").y("y")),
            ("ymin", Geom::gg_boxplot(), Aes::new().y("y").ymin("x")),
            ("ymax", Geom::gg_boxplot(), Aes::new().y("y").ymax("x")),
            ("fill freq", Geom::gg_freqpoly(), Aes::new().x("x").fill("h")),
            ("fill err", Geom::gg_errorbar(), Aes::new().x("x").ymin("x").ymax("y").fill("h")),
        ];
        for (name, g, a) in unused {
            assert!(matches!(run(plot(probe(), a, g)), Err(Error::Rendering(_))), "{name}");
        }
        let stats = [
            (Geom::gg_errorbar().stat(Stat::Count), Aes::new().x("x").ymin("x").ymax("y")),
            (Geom::gg_boxplot().stat(Stat::Identity), Aes::new().y("y")),
            (Geom::gg_histogram().stat(Stat::Bin { bins: 30 }), Aes::new().x("x")),
            (Geom::gg_density().stat(Stat::Smooth), Aes::new().x("x")),
        ];
        for (g, a) in stats {
            assert!(matches!(run(plot(small(), a, g)), Err(Error::UnsupportedStat(_))));
        }
        let own = Geom::gg_density().stat(Stat::Density);
        assert!(run(plot(small(), Aes::new().x("x"), own)).is_ok());
    }

    /// `adjust`, `trim` and `violin_scale` are geom_violin parameters: on every other layer a
    /// non-default value is refused by name, and the default is accepted.
    #[test]
    fn violin_only_parameters_are_refused_on_the_other_distribution_layers() {
        let on = |g: Geom, a: Aes| run(plot(small(), a, g));
        let x = || Aes::new().x("x");
        for (name, geom, aes) in [
            ("adjust", Geom::gg_density().adjust(0.5), x()),
            ("trim", Geom::gg_density().trim(false), x()),
            ("adjust", Geom::gg_histogram().adjust(0.5), x()),
            ("trim", Geom::gg_histogram().trim(false), x()),
            ("violin_scale", Geom::gg_freqpoly().violin_scale(ViolinScale::Width), x()),
            ("adjust", Geom::gg_freqpoly().adjust(2.0), x()),
            (
                "violin_scale",
                Geom::gg_boxplot().violin_scale(ViolinScale::Count),
                Aes::new().y("x"),
            ),
        ] {
            let r = on(geom, aes);
            let want = format!("`{name}` is a geom_violin parameter");
            assert!(matches!(&r, Err(Error::Rendering(m)) if m.contains(&want)), "{name}: {r:?}");
        }
        assert!(on(Geom::gg_density(), x()).is_ok());
        assert!(on(Geom::gg_histogram().adjust(1.0).trim(true), x()).is_ok());
    }

    /// `geom_density` through `layout()`: missing values are dropped as ggplot2 does, and input
    /// with no bandwidth (fewer than two values) or with groups is refused with a typed error.
    #[test]
    fn density_drops_missing_and_refuses_degenerate_input() {
        let dens = |d: DataFrame| run(plot(d, Aes::new().x("x"), Geom::gg_density()));
        let clean = dens(DataFrame::from_xy(&[1.0, 2.0, 4.0, 8.0], &[0.0; 4])).expect("clean");
        let gappy = dens(DataFrame::from_xy(&[1.0, f32::NAN, 2.0, 4.0, 8.0], &[0.0; 5]));
        assert_eq!(gappy.expect("gappy").layers[0].y, clean.layers[0].y);
        // The bandwidth step is what refuses these: the message names it, so a `bw_nrd0` that let
        // a short sample through would change it (the range guard would otherwise also refuse).
        for xs in [&[3.0_f32][..], &[f32::NAN, f32::NAN]] {
            let r = dens(DataFrame::from_xy(xs, &vec![0.0; xs.len()]));
            assert!(
                matches!(&r, Err(Error::Rendering(m)) if m.contains("needs at least two values")),
                "{xs:?}: {r:?}"
            );
        }
        // A constant sample is built, as ggplot2 does (bandwidth falls back to |x[1]|).
        let flat = dens(DataFrame::from_xy(&[2.0, 2.0, 2.0], &[0.0; 3])).expect("constant builds");
        assert_eq!(flat.layers[0].x.len(), 512);
        let r = run(plot(probe(), Aes::new().x("x").color("h"), Geom::gg_density()));
        assert!(matches!(r, Err(Error::Rendering(m)) if m.contains("grouped")));
    }

    /// `geom_histogram` through `layout()`: missing values are dropped as ggplot2 does, and input
    /// with no bins (all missing) is refused with a typed error; a grouped histogram stacks.
    #[test]
    fn histogram_drops_missing_refuses_empty_and_stacks_groups() {
        let hist = |d: DataFrame| run(plot(d, Aes::new().x("x"), Geom::gg_histogram()));
        let clean = hist(DataFrame::from_xy(&[1.0, 2.0, 4.0, 8.0], &[0.0; 4])).expect("clean");
        let gappy = hist(DataFrame::from_xy(&[1.0, f32::NAN, 2.0, 4.0, 8.0], &[0.0; 5]));
        assert_eq!(gappy.expect("gappy").layers[0].y, clean.layers[0].y);
        let none = hist(DataFrame::from_xy(&[f32::NAN, f32::NAN], &[0.0; 2]));
        assert!(matches!(none, Err(Error::Rendering(_))));
        let grouped = run(plot(probe(), Aes::new().x("x").fill("h"), Geom::gg_histogram()));
        let layer = grouped.expect("grouped histogram builds").layers.remove(0);
        let b = layer.bins.expect("bins");
        assert_eq!(b.group.iter().copied().collect::<std::collections::BTreeSet<_>>().len(), 3);
        let total: f64 = b.count.iter().sum();
        assert_eq!(total, 5.0, "every row lands in exactly one bin");
        assert!(b.ymax.iter().zip(&b.ymin).all(|(hi, lo)| hi >= lo));
    }

    #[test]
    fn coords_facets_and_mixed_layers_are_refused() {
        let coords = [
            Coord::cartesian().xlim(0.0, 1.0),
            Coord::cartesian().ylim(0.0, 1.0),
            Coord::cartesian().flip(),
            Coord::polar(),
            Coord::fixed(1.0),
        ];
        for (name, g, a) in all_geoms() {
            for c in &coords {
                let r = run(plot(small(), a.clone(), g.clone()).coord(c.clone()));
                assert!(matches!(r, Err(Error::UnsupportedCoord(_))), "{name} {c:?}");
            }
            for f in [Facet::row("x"), Facet::col("x"), Facet::wrap("x", 2)] {
                let r = run(plot(small(), a.clone(), g.clone()).facet(f));
                assert!(matches!(r, Err(Error::Rendering(_))), "{name}");
            }
        }
        let mixed = plot(small(), Aes::new().x("x"), Geom::gg_histogram())
            .layer(Layer::new(Geom::point()).aes(Aes::new().x("x").y("y")));
        assert!(matches!(run(mixed), Err(Error::Rendering(m)) if m.contains("combined")));
    }

    #[test]
    fn ragged_text_infinite_and_unknown_columns_are_refused() {
        let mut d = probe();
        d.add_column_f32("z", &[1.0, 2.0, 3.0]);
        let inf = DataFrame::from_xy(&[1.0, f32::INFINITY, 3.0], &[1.0, 2.0, 3.0]);
        for (name, g, a) in all_geoms() {
            let r = run(plot(inf.clone(), a, g));
            assert!(matches!(r, Err(Error::Rendering(m)) if m.contains("infinite")), "{name}");
        }
        for col in ["z", "h"] {
            let r = run(plot(d.clone(), Aes::new().x(col), Geom::gg_histogram()));
            assert!(matches!(r, Err(Error::Rendering(_))), "{col}");
        }
        let r = run(plot(d, Aes::new().x("nope"), Geom::gg_density()));
        assert!(matches!(r, Err(Error::UnknownColumn(_))));
        let r = run(plot(probe(), Aes::new().x("x").fill("h").color("f"), Geom::gg_histogram()));
        assert!(matches!(r, Err(Error::Rendering(m)) if m.contains("interaction")));
    }

    /// F18: `ymin`/`ymax`, a layer's own data and a layer's own aesthetics are all checked.
    #[test]
    fn bounds_layer_data_and_layer_aes_are_checked() {
        let inf = DataFrame::from_xy(&[1.0, f32::INFINITY, 3.0], &[1.0, 2.0, 3.0]);
        let bar = |a: Aes, d: DataFrame| run(plot(d, a, Geom::gg_errorbar()));
        let r = bar(Aes::new().x("y").ymin("x").ymax("y"), inf.clone());
        assert!(matches!(r, Err(Error::Rendering(m)) if m.contains("`x` holds an infinite")));
        let r = bar(Aes::new().x("x").ymin("x").ymax("h"), probe());
        assert!(matches!(r, Err(Error::Rendering(m)) if m.contains("`h` is text")), "ymax");
        let own = GGPlot::new().data(probe()).aes(Aes::new().x("x"));
        let r = run(own.clone().layer(Layer::new(Geom::gg_histogram()).data(inf)));
        assert!(matches!(r, Err(Error::Rendering(m)) if m.contains("infinite")), "layer data");
        let r = run(own.layer(Layer::new(Geom::gg_freqpoly()).aes(Aes::new().fill("h"))));
        assert!(matches!(r, Err(Error::Rendering(m)) if m.contains("`fill`")), "layer aes");
    }

    /// F14: a `scale_x`/`scale_y` spec beside a distribution layer is refused by name.
    #[test]
    fn position_scale_specs_beside_dist_layers_are_refused() {
        let scaled = |m: &Result<BuiltLayout>| matches!(m, Err(Error::UnsupportedLayer(m)) if m.contains("scale_x/scale_y"));
        for (name, g, a) in all_geoms() {
            let p = plot(small(), a.clone(), g.clone());
            assert!(scaled(&run(p.scale_x(scale_continuous()))), "{name} x");
            let p = plot(small(), a, g);
            assert!(scaled(&run(p.scale_y(scale_continuous()))), "{name} y");
        }
        let p = GGPlot::new()
            .data(small())
            .layer(Layer::new(Geom::point()).aes(Aes::new().x("x").y("y")))
            .layer(Layer::new(Geom::gg_histogram()).aes(Aes::new().x("x")))
            .scale_x(scale_continuous());
        assert!(scaled(&run(p)), "a distribution layer after a point layer");
    }

    /// F15: `Geom::alpha`, `width` and `bins` reach the spec; alpha clamped, 0.3 kept as 0.3.
    #[test]
    fn geom_setters_reach_the_dist_spec() {
        let spec = |g: Geom| match g.geom_type {
            GeomType::Dist(s) => Some(s),
            _ => None,
        };
        let field = |g: Geom| spec(g).expect("a dist geom");
        for (given, want) in [(0.3_f32, 0.3_f64), (2.0, 1.0), (-1.0, 0.0)] {
            assert_eq!(field(Geom::gg_histogram().alpha(given)).alpha, Some(want), "{given}");
        }
        assert_eq!(field(Geom::gg_boxplot().width(0.3)).width, Some(0.3));
        assert_eq!(field(Geom::gg_freqpoly().bins(7)).bins, 7);
        assert_eq!(field(Geom::gg_density()).alpha, None);
        assert!(spec(Geom::point().alpha(0.5)).is_none(), "a point geom carries no dist spec");
        // A layer's `ymin`/`ymax` override the plot's; an unset one keeps the plot's.
        let m = Aes::new().ymin("a").ymax("b").merge(&Aes::new().ymin("c"));
        assert_eq!((m.ymin.as_deref(), m.ymax.as_deref()), (Some("c"), Some("b")));
        let m = Aes::new().ymin("a").merge(&Aes::new().ymax("d"));
        assert_eq!((m.ymin.as_deref(), m.ymax.as_deref()), (Some("a"), Some("d")));
    }
}
