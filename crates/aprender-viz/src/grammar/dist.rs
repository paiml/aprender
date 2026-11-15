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
use super::build::{BuiltDensity, BuiltLayer, BuiltLayout, PositionScale};
use super::coord::Coord;
use super::data::{DataFrame, DataValue};
use super::dist_math::{cmp_na_last, stat_density, DENSITY_POINTS};
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
}

impl DistSpec {
    fn new(kind: DistKind) -> Self {
        Self { kind, bins: 30, width: None, alpha: None }
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
}

fn refuse(msg: impl Into<String>) -> Error {
    Error::Rendering(msg.into())
}

/// `true` when this layer is a distribution layer.
pub(super) fn is_dist(layer: &Layer) -> bool {
    matches!(layer.geom.geom_type, GeomType::Dist(_))
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
        let GeomType::Dist(spec) = &layer.geom.geom_type else { continue };
        let data = layer.data.as_ref().unwrap_or(plot_data);
        let aes = plot_aes.merge(&layer.aes);
        check_inputs(spec.kind, &aes, layer.geom.stat.as_ref())?;
        let groups = group_ids(data, &aes)?;
        for name in [&aes.x, &aes.y, &aes.ymin, &aes.ymax].into_iter().flatten() {
            numeric(data, name)?;
        }
        checked.push((spec.kind, data, aes, groups));
    }
    match checked.as_slice() {
        [(DistKind::Density, data, aes, groups)] => density(data, aes, groups),
        _ => Err(Error::UnsupportedLayer(NOT_BUILT)),
    }
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
        density: Some(BuiltDensity {
            scaled: cols.scaled,
            count: cols.count,
            wdensity: cols.wdensity,
            n: cols.n,
        }),
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
            // The density is built; the other four stats are not.
            assert_eq!(not_built(&p.layout()), name != "dens", "{name}");
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

    /// `geom_density` through `layout()`: missing values are dropped as ggplot2 does, and input
    /// with no bandwidth (one value, a constant) or with groups is refused with a typed error.
    #[test]
    fn density_drops_missing_and_refuses_degenerate_input() {
        let dens = |d: DataFrame| run(plot(d, Aes::new().x("x"), Geom::gg_density()));
        let clean = dens(DataFrame::from_xy(&[1.0, 2.0, 4.0, 8.0], &[0.0; 4])).expect("clean");
        let gappy = dens(DataFrame::from_xy(&[1.0, f32::NAN, 2.0, 4.0, 8.0], &[0.0; 5]));
        assert_eq!(gappy.expect("gappy").layers[0].y, clean.layers[0].y);
        for xs in [&[3.0_f32][..], &[2.0, 2.0, 2.0], &[f32::NAN, f32::NAN]] {
            let r = dens(DataFrame::from_xy(xs, &vec![0.0; xs.len()]));
            assert!(matches!(r, Err(Error::Rendering(m)) if m.contains("stat_density")), "{xs:?}");
        }
        let r = run(plot(probe(), Aes::new().x("x").color("h"), Geom::gg_density()));
        assert!(matches!(r, Err(Error::Rendering(m)) if m.contains("grouped")));
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
