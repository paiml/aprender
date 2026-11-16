//! `geom_text` as `ggplot_build()` in ggplot2 4.0.3 resolves it (PLOT-GG-001,
//! paiml/aprender#6909). Proven against R by `ggplot2_text_oracle.rs`.
//!
//! # Scope (what this module promises, and what it refuses with a typed error)
//!
//! * Layer data: `x` and `y` after `position_nudge` (`nudge_x`, `nudge_y`; the position scales
//!   are retrained on the nudged values), the `label` column, `PANEL`, `group`, `angle`, `hjust`,
//!   `vjust`, `size` (`11 / .pt`), `alpha` and `colour`. Font family, fontface and lineheight are
//!   not layout and are not carried.
//! * Positions are continuous numbers; `NaN` rows are kept. A text or infinite position is an
//!   [`Error::Rendering`], a missing column an [`Error::UnknownColumn`].
//! * The label column must exist and be text (a numeric label column is refused: R formats
//!   numbers differently); a missing `label` aesthetic is an
//!   [`Error::Rendering`] (ggplot2 errors: "requires the following missing aesthetics: label").
//! * Groups and facets follow `path.rs` (explicit `group` only; `facet_wrap` over a text column).
//!   `label` never forms a group (ggplot2 excludes it from the implicit grouping).
//! * Non-finite `nudge_x`, `nudge_y`, `hjust`, `vjust` or `angle` are an [`Error::Rendering`].
//! * A data-mapped colour, size, shape, alpha or fill would train a scale and is refused.
//! * Only the default `coord_cartesian()`, the identity stat, and `geom_point` companions are
//!   supported with it, as in `path.rs`; `layout()` computes no text extents.

use crate::color::Rgba;
use crate::error::{Error, Result};

use super::aes::Aes;
use super::build::{BuiltLayer, PT_PER_MM};
use super::data::{DataFrame, DataValue};
use super::facet::Facet;
use super::geom::{GeomType, TextParams};
use super::ggplot::Layer;
use super::path::{global_panels, group_ids, read_column, row_panels, unsupported_stat};

/// What a `geom_text` layer adds to a [`BuiltLayer`]: one entry per data row.
#[derive(Debug, Clone, PartialEq)]
pub struct BuiltText {
    /// 0-based panel of each row.
    pub panel: Vec<usize>,
    /// Group id of each row (-1 when ungrouped).
    pub group: Vec<i64>,
    /// The label of each row; a missing cell is `None`.
    pub label: Vec<Option<String>>,
    /// `nudge_x` (one value, repeated per row by ggplot2).
    pub nudge_x: f64,
    /// `nudge_y`.
    pub nudge_y: f64,
    /// `angle` in degrees.
    pub angle: f64,
    /// `hjust`.
    pub hjust: f64,
    /// `vjust`.
    pub vjust: f64,
    /// `size` in millimetres: `11 / .pt` unless `size_value` is set.
    pub size: f64,
    /// `alpha`, `NA` unless `alpha_value` is set.
    pub alpha: Option<f64>,
    /// `colour`, black unless `color_value` is set.
    pub colour: Rgba,
}

/// `GeomText$default_aes$size`: `from_theme(fontsize)`, 11 points, in millimetres.
const DEFAULT_SIZE_MM: f64 = 11.0 / PT_PER_MM;

/// The nudge a `geom_text` layer applies to `(x, y)`; `(0, 0)` for any other geom.
pub(super) fn nudge(g: &GeomType) -> (f64, f64) {
    match g {
        GeomType::Text(p) => (f64::from(p.nudge_x), f64::from(p.nudge_y)),
        _ => (0.0, 0.0),
    }
}

fn refuse(msg: impl Into<String>) -> Error {
    Error::Rendering(msg.into())
}

fn labels(data: &DataFrame, aes: &Aes) -> Result<Vec<Option<String>>> {
    let name =
        aes.label.as_deref().ok_or_else(|| refuse("geom_text requires the label aesthetic"))?;
    let col =
        data.get(name).ok_or_else(|| Error::UnknownColumn(format!("no label column {name:?}")))?;
    // R keeps a numeric label numeric and formats it differently from Rust's `{v}` on an f32
    // (1e-05, 1e+05, widening digits); not modelled, so refused rather than approximated.
    if col.iter().any(|v| matches!(v, DataValue::Number(_))) {
        return Err(refuse(format!(
            "geom_text label column {name:?} is numeric: numeric label formatting is not modelled"
        )));
    }
    Ok((0..data.nrow())
        .map(|i| match col.get(i) {
            Some(DataValue::Text(s)) => Some(s.clone()),
            _ => None,
        })
        .collect())
}

fn refuse_non_finite(p: &TextParams) -> Result<()> {
    for (name, v) in [
        ("nudge_x", p.nudge_x),
        ("nudge_y", p.nudge_y),
        ("hjust", p.hjust),
        ("vjust", p.vjust),
        ("angle", p.angle),
    ] {
        if !v.is_finite() {
            return Err(refuse(format!("geom_text {name} must be finite, got {v}")));
        }
    }
    Ok(())
}

fn refuse_mapped(aes: &Aes) -> Result<()> {
    if aes.color.is_some()
        || aes.size.is_some()
        || aes.shape.is_some()
        || aes.alpha.is_some()
        || aes.fill.is_some()
    {
        return Err(refuse(
            "a data-mapped colour, size, shape, alpha or fill is not supported on a text layer",
        ));
    }
    Ok(())
}

/// Build one `geom_text` layer.
///
/// # Errors
///
/// See the module docs: a missing label or column, text or infinite positions, a mapped
/// aesthetic that would train a scale, an unsupported facet.
pub(super) fn build_layer(
    layer: &Layer,
    p: &TextParams,
    plot_aes: &Aes,
    plot_data: &DataFrame,
    facet: &Facet,
) -> Result<BuiltLayer> {
    if let Some(msg) = layer.geom.stat.as_ref().and_then(unsupported_stat) {
        return Err(Error::UnsupportedStat(msg));
    }
    refuse_non_finite(p)?;
    let aes = plot_aes.merge(&layer.aes);
    refuse_mapped(&aes)?;
    let global = global_panels(facet, plot_data)?;
    let own = layer.data.is_some();
    let data = layer.data.as_ref().unwrap_or(plot_data);
    let panel = row_panels(facet, &global, data, own)?;
    let group = group_ids(&aes, data)?;
    let label = labels(data, &aes)?;
    let (nx, ny) = (f64::from(p.nudge_x), f64::from(p.nudge_y));
    let x = read_column(data, aes.x.as_deref().unwrap_or("x"), "x")?;
    let y = read_column(data, aes.y.as_deref().unwrap_or("y"), "y")?;
    let text = BuiltText {
        panel,
        group,
        label,
        nudge_x: nx,
        nudge_y: ny,
        angle: f64::from(p.angle),
        hjust: f64::from(p.hjust),
        vjust: f64::from(p.vjust),
        size: aes.size_value.map_or(DEFAULT_SIZE_MM, f64::from),
        alpha: aes.alpha_value.map(f64::from),
        colour: aes.color_value.unwrap_or(Rgba::BLACK),
    };
    Ok(BuiltLayer {
        x: x.into_iter().map(|v| v + nx).collect(),
        y: y.into_iter().map(|v| v + ny).collect(),
        points: None,
        path: None,
        text: Some(text),
    })
}
