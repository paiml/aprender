//! The resolved plot: what ggplot2's `ggplot_build()` computes before anything is drawn.
//!
//! [`BuiltGGPlot::layout`](super::BuiltGGPlot::layout) returns a [`BuiltLayout`]: each layer's
//! positions and resolved aesthetics, and the trained, expanded x/y position scales with their
//! breaks and labels. The renderer draws from the same numbers, and the ggplot2 conformance
//! oracle (`ggplot2_oracle.rs`) compares them against fixtures produced by ggplot2 4.0.3.
//!
//! Every default in this file is ggplot2 4.0.3's, and each names where ggplot2 defines it.

pub use super::r_format::format_labels;
use super::scale_spec::{resolve_continuous, ContinuousSpec};
use crate::color::Rgba;

use super::path::BuiltPath;

/// Multiplicative expansion of a continuous position scale: `expansion(mult = 0.05)`, ggplot2's
/// `default_expansion()` for a continuous scale. Each side grows by 5% of the data span.
pub const CONTINUOUS_EXPAND_MULT: f64 = 0.05;

/// Additive expansion of a continuous position scale (`expansion(add = 0)`).
pub const CONTINUOUS_EXPAND_ADD: f64 = 0.0;

/// Number of breaks asked of `scales::extended_breaks()` (its `n = 5`), which hands it to
/// `labeling::extended(m = 5)`.
pub const DEFAULT_N_BREAKS: usize = 5;

/// `GeomPoint$default_aes` in ggplot2 4.0.3.
pub mod point_defaults {
    use crate::color::Rgba;

    /// `shape = 19`: R's solid circle (`pch = 19`).
    pub const SHAPE: u8 = 19;
    /// `colour = "black"`.
    pub const COLOUR: Rgba = Rgba::BLACK;
    /// `size = 1.5`, in millimetres.
    pub const SIZE_MM: f64 = 1.5;
    /// `stroke = 0.5`, in millimetres.
    pub const STROKE_MM: f64 = 0.5;
}

/// ggplot2's `.pt`: points per millimetre (72.27 / 25.4).
pub const PT_PER_MM: f64 = 72.27 / 25.4;
/// ggplot2's `.stroke`: the stroke unit (96 / 25.4).
pub const STROKE_PER_MM: f64 = 96.0 / 25.4;
/// Pixels per point at the 96 dpi this crate renders at.
pub const PX_PER_PT: f64 = 96.0 / 72.0;

/// Diameter in pixels of a solid point of ggplot2 `size` and `stroke` (both mm).
///
/// ggplot2 draws a point with `fontsize = size * .pt + stroke * .stroke / 2` (`GeomPoint$draw_panel`).
#[must_use]
pub fn point_diameter_px(size_mm: f64, stroke_mm: f64) -> f64 {
    (size_mm * PT_PER_MM + stroke_mm * STROKE_PER_MM / 2.0) * PX_PER_PT
}

/// The ggplot2 `size` (mm) whose default-stroke point is `px` pixels across: the inverse of
/// [`point_diameter_px`] at [`point_defaults::STROKE_MM`], floored at 0.
#[must_use]
pub fn point_size_mm_for_diameter_px(px: f64) -> f64 {
    let stroke = point_defaults::STROKE_MM * STROKE_PER_MM / 2.0;
    ((px / PX_PER_PT - stroke) / PT_PER_MM).max(0.0)
}

/// One resolved point: ggplot2's per-row aesthetic columns for `GeomPoint`.
#[derive(Debug, Clone, PartialEq)]
pub struct PointAes {
    /// R `pch` code.
    pub shape: u8,
    /// Outline/solid colour.
    pub colour: Rgba,
    /// Fill (`NA` in ggplot2 unless set).
    pub fill: Option<Rgba>,
    /// Size in millimetres.
    pub size: f64,
    /// Opacity (`NA` in ggplot2 unless set: the colour's own alpha applies).
    pub alpha: Option<f64>,
    /// Stroke width in millimetres.
    pub stroke: f64,
}

/// One layer's data after building: positions in data space (before the coordinate system), and
/// for a point layer, its resolved aesthetics.
///
/// Rows where `x` or `y` is missing or non-numeric are dropped here; ggplot2 keeps them in
/// `ggplot_build()` and drops them at draw time with a warning.
#[derive(Debug, Clone, PartialEq)]
pub struct BuiltLayer {
    /// x positions.
    pub x: Vec<f64>,
    /// y positions.
    pub y: Vec<f64>,
    /// Per-row point aesthetics; `None` for geoms whose aesthetics are not resolved yet.
    pub points: Option<Vec<PointAes>>,
    /// Resolved path, step or reference-line data; `None` for other geoms. For these layers `x`/`y`
    /// keep `NaN` rows, as `ggplot_build()` does (reference layers have no `x`/`y`).
    pub path: Option<BuiltPath>,
}

/// One trained continuous position scale, as a panel sees it (ggplot2's view scale).
#[derive(Debug, Clone, PartialEq)]
pub struct PositionScale {
    /// The trained limits: the range of the data over all layers.
    pub limits: (f64, f64),
    /// The limits after the default expansion: what the panel spans.
    pub continuous_range: (f64, f64),
    /// Major breaks inside `continuous_range`.
    pub breaks: Vec<f64>,
    /// One label per break, formatted as R's `format()` does.
    pub labels: Vec<String>,
    /// Breaks the algorithm proposed outside `continuous_range` (ggplot2 keeps these as `NA`).
    pub n_breaks_censored: usize,
    /// Minor breaks inside `continuous_range` (continuous axes; empty for discrete).
    pub minor_breaks: Vec<f64>,
    /// The level names of a discrete axis, in position order; `None` for a continuous axis.
    pub levels: Option<Vec<String>>,
}

/// Indices of the `proposed` breaks that lie inside the panel range `[lo, hi]`, ends included.
///
/// ggplot2 censors breaks with `oob_censor_any` on the continuous range; a break exactly on a
/// limit is kept.
#[must_use]
pub fn censor_breaks(proposed: &[f64], lo: f64, hi: f64) -> Vec<usize> {
    proposed.iter().enumerate().filter(|(_, b)| **b >= lo && **b <= hi).map(|(i, _)| i).collect()
}

impl PositionScale {
    /// Train a continuous scale on `limits` and apply ggplot2's defaults: 5% expansion,
    /// `extended_breaks(n = 5)` over the expanded range, censored to it, labelled by `format()`.
    ///
    /// A zero-width range is expanded with width 1 for the multiplicative part (`scales::expand_range`,
    /// `zero_width = 1`): a single value `v` spans `v +- 0.05`. Measured on ggplot2 4.0.3 and covered by
    /// the row 2-4 fixtures. An empty (untrained) range spans 0..1, as ggplot2's empty scale does.
    #[must_use]
    pub fn continuous(limits: (f64, f64)) -> Self {
        resolve_continuous(limits, &ContinuousSpec::new())
    }
}

/// The resolved plot: layers plus the shared x/y position scales (one panel's view; facets share
/// one domain, ggplot2's `scales = "fixed"`).
///
/// The scales are trained on the data, before the coordinate system: `coord_cartesian(xlim =)`
/// and `coord_flip()` are applied by the renderer, after this.
#[derive(Debug, Clone, PartialEq)]
pub struct BuiltLayout {
    /// One entry per layer, in layer order.
    pub layers: Vec<BuiltLayer>,
    /// The x position scale.
    pub x: PositionScale,
    /// The y position scale.
    pub y: PositionScale,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_width_range_spans_five_hundredths_each_side() {
        let s = PositionScale::continuous((5.0, 5.0));
        assert!((s.continuous_range.0 - 4.95).abs() < 1e-12);
        assert!((s.continuous_range.1 - 5.05).abs() < 1e-12);
        assert_eq!(s.labels, ["4.950", "4.975", "5.000", "5.025", "5.050"]);
    }

    #[test]
    fn empty_range_spans_zero_to_one() {
        let s = PositionScale::continuous((f64::MAX, f64::MIN));
        assert_eq!(s.limits, (0.0, 1.0));
        assert!((s.continuous_range.0 + 0.05).abs() < 1e-12);
    }

    #[test]
    fn pixel_size_round_trips() {
        let mm = point_size_mm_for_diameter_px(10.0);
        assert!((point_diameter_px(mm, point_defaults::STROKE_MM) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn default_point_is_about_seven_pixels() {
        let d = point_diameter_px(point_defaults::SIZE_MM, point_defaults::STROKE_MM);
        assert!((d - 6.95).abs() < 0.01, "{d}");
    }
}
