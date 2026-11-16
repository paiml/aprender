//! Filled marks and text: `geom_area`, `geom_ribbon`, `geom_tile`, `geom_rect`, `geom_text`
//! (PLOT-GG-001, paiml/aprender#6909). The "five" below are these; `geom_rect` has no
//! constructor yet, so only area, ribbon, tile and text can reach this module. Area and ribbon
//! are resolved by `filled`, but a `geom_point` beside them still gets the companion checks here.
//!
//! `layout()` does not resolve tile and text yet. It used to lay them out as points: raw `x`/`y`
//! rows, scales trained on them, tile bounds or labels. That is a plausible wrong answer, so
//! [`require_resolved`] refuses each of them with a typed error until its `ggplot_build()`
//! resolution lands.
//!
//! Companion layers. The x and y scales are shared, so a plot with one of these layers may hold
//! only these geoms and `geom_point` with the identity stat; every other geom is an
//! [`Error::UnsupportedLayer`] and every other stat on a `geom_point` an [`Error::UnsupportedStat`]
//! (see [`require_modelled_companions`]).

use super::aes::Aes;
use super::data::{DataFrame, DataValue};
use super::geom::GeomType;
use super::ggplot::Layer;
use super::stat::Stat;
use crate::error::{Error, Result};

/// Which of the marks geoms a layer is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarkKind {
    /// `geom_tile()`.
    Tile,
    /// `geom_text()`.
    Text,
}

impl MarkKind {
    fn of(geom: &GeomType) -> Option<Self> {
        match geom {
            GeomType::Tile => Some(MarkKind::Tile),
            GeomType::Text => Some(MarkKind::Text),
            _ => None,
        }
    }
}

/// Is any layer one of the marks geoms?
pub(super) fn any_marks(layers: &[Layer]) -> bool {
    layers.iter().any(|l| MarkKind::of(&l.geom.geom_type).is_some())
}

/// Refuse the first marks layer: `layout()` does not resolve it yet.
///
/// # Errors
///
/// [`Error::UnsupportedLayer`] naming the geom.
pub(super) fn require_resolved(layers: &[Layer]) -> Result<()> {
    match layers.iter().find_map(|l| MarkKind::of(&l.geom.geom_type)) {
        None => Ok(()),
        Some(MarkKind::Tile) => {
            Err(Error::UnsupportedLayer("geom_tile: layout() does not resolve it yet"))
        }
        Some(MarkKind::Text) => {
            Err(Error::UnsupportedLayer("geom_text: layout() does not resolve it yet"))
        }
    }
}

/// Refuse, while one of the five geoms (or an area or ribbon layer) is present, every layer `layout()` does not resolve beside
/// it. The scales are shared, so a companion whose ggplot2 stat, setup or position widens them
/// (measured on ggplot2 4.0.3 beside `geom_area`: `geom_bar` takes y to 0 and x to the bar edges,
/// `geom_histogram` x to the bin edges, `geom_point(stat = "count")` y to 0) would make `layout()`
/// answer with the wrong limits. Only the five geoms and `geom_point` with the identity stat may
/// share the plot; a `geom_point` beside them must map `x` and `y` to complete numeric columns of
/// the frame, with no infinite value (ggplot2 squishes those to the panel). Path, step and
/// reference layers are refused before this by `path`.
///
/// # Errors
///
/// [`Error::UnsupportedLayer`] for a companion geom, [`Error::UnsupportedStat`] for a stat on
/// `geom_point`, [`Error::UnknownColumn`] for a missing column, [`Error::Rendering`] for a missing
/// mapping or a text or short column, and [`Error::ScaleDomain`] for an infinite position.
pub(super) fn require_modelled_companions(
    layers: &[Layer],
    plot_aes: &Aes,
    plot_data: &DataFrame,
) -> Result<()> {
    let filled = |g: &GeomType| super::filled::is_filled(g) || super::smooth::is_smooth(g);
    if !any_marks(layers) && !layers.iter().any(|l| filled(&l.geom.geom_type)) {
        return Ok(());
    }
    for l in layers {
        let g = &l.geom.geom_type;
        if MarkKind::of(g).is_some() || filled(g) {
            continue;
        }
        if !matches!(g, GeomType::Point { .. }) {
            return Err(Error::UnsupportedLayer(unsupported_companion(g)));
        }
        if let Some(msg) = l.geom.stat.as_ref().and_then(unsupported_point_stat) {
            return Err(Error::UnsupportedStat(msg));
        }
        let data = l.data.as_ref().unwrap_or(plot_data);
        require_point_companion(data, &plot_aes.merge(&l.aes))?;
    }
    Ok(())
}

/// The refusal naming a companion geom that may not share a plot with the five.
fn unsupported_companion(g: &GeomType) -> &'static str {
    match g {
        GeomType::Bar { .. } => {
            "geom_bar beside area/ribbon/tile/rect/text: layout() does not resolve it there \
             (stat_count and the bar widths widen the shared scales)"
        }
        GeomType::Histogram { .. } => {
            "geom_histogram beside area/ribbon/tile/rect/text: layout() does not resolve it there \
             (stat_bin widens the shared scales)"
        }
        GeomType::Boxplot => {
            "geom_boxplot beside area/ribbon/tile/rect/text: layout() does not resolve it there \
             (stat_boxplot and the box widths widen the shared scales)"
        }
        GeomType::Violin => {
            "geom_violin beside area/ribbon/tile/rect/text: layout() does not resolve it there \
             (stat_ydensity and the violin widths widen the shared scales)"
        }
        GeomType::Smooth { .. } => {
            "geom_smooth beside area/ribbon/tile/rect/text: layout() does not resolve it there \
             (stat_smooth and its ribbon widen the shared scales)"
        }
        _ => {
            "a line, path, step or reference layer beside area/ribbon/tile/rect/text: layout() \
             does not resolve it there"
        }
    }
}

/// The refusal naming a stat on a `geom_point` beside the five; `None` for the identity stat.
fn unsupported_point_stat(s: &Stat) -> Option<&'static str> {
    let msg = match s {
        Stat::Identity => return None,
        Stat::Count => "stat_count on geom_point beside area/ribbon/tile/rect/text: layout() computes no stat",
        Stat::Bin { .. } => "stat_bin on geom_point beside area/ribbon/tile/rect/text: layout() computes no stat",
        Stat::Density => {
            "stat_density on geom_point beside area/ribbon/tile/rect/text: layout() computes no stat"
        }
        Stat::Boxplot => {
            "stat_boxplot on geom_point beside area/ribbon/tile/rect/text: layout() computes no stat"
        }
        Stat::Smooth => {
            "stat_smooth on geom_point beside area/ribbon/tile/rect/text: layout() computes no stat"
        }
        Stat::Bin2d { .. } => {
            "stat_bin_2d on geom_point beside area/ribbon/tile/rect/text: layout() computes no stat"
        }
        Stat::Sum => {
            "stat_summary(fun = sum) on geom_point beside area/ribbon/tile/rect/text: layout() computes no stat"
        }
        Stat::Mean => {
            "stat_summary(fun = mean) on geom_point beside area/ribbon/tile/rect/text: layout() computes no stat"
        }
    };
    Some(msg)
}

/// A `geom_point` beside the five trains the shared scales on its raw `x` and `y`: both must be
/// mapped, present, complete-length numeric columns without an infinite value.
fn require_point_companion(data: &DataFrame, aes: &Aes) -> Result<()> {
    for (axis, name) in [("x", &aes.x), ("y", &aes.y)] {
        let Some(name) = name.as_deref() else {
            return Err(Error::Rendering(format!(
                "geom_point beside area/ribbon/tile/rect/text needs a mapped {axis}"
            )));
        };
        require_point_column(data, name)?;
    }
    Ok(())
}

/// One position column of a `geom_point` beside the five: present, as long as the frame, and
/// numeric without an infinite value (an NA cell is kept: the scale trains on the other rows).
fn require_point_column(data: &DataFrame, name: &str) -> Result<()> {
    let cells = data.get(name).ok_or_else(|| Error::UnknownColumn(name.to_string()))?;
    if cells.len() < data.nrow() {
        return Err(Error::Rendering(format!(
            "geom_point beside area/ribbon/tile/rect/text: column {name} is shorter than the frame"
        )));
    }
    if cells.iter().any(|v| matches!(v, DataValue::Text(_))) {
        return Err(Error::Rendering(format!(
            "geom_point beside area/ribbon/tile/rect/text: column {name} is text, which needs a \
             discrete scale; not modelled"
        )));
    }
    if cells.iter().any(|v| matches!(v, DataValue::Number(n) if n.is_infinite())) {
        return Err(Error::ScaleDomain(format!(
            "geom_point beside area/ribbon/tile/rect/text: column {name}: infinite position is \
             not modelled"
        )));
    }
    Ok(())
}
