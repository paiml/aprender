//! The resolved plot: what ggplot2's `ggplot_build()` computes before anything is drawn.
//!
//! [`BuiltGGPlot::layout`](super::BuiltGGPlot::layout) returns a [`BuiltLayout`]: each layer's
//! positions and resolved aesthetics, and the trained, expanded x/y position scales with their
//! breaks and labels. The renderer draws from the same numbers, and the ggplot2 conformance
//! oracle (`ggplot2_oracle.rs`) compares them against fixtures produced by ggplot2 4.0.3.
//!
//! Every default in this file is ggplot2 4.0.3's, and each names where ggplot2 defines it.

use crate::breaks::{extended, Q_DEFAULT, W_DEFAULT};
use crate::color::Rgba;

use super::marks::BuiltMarks;

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
    /// The resolved columns of an area, ribbon, tile, rect or text layer; `None` for other geoms.
    pub marks: Option<BuiltMarks>,
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
}

impl PositionScale {
    /// Train a continuous scale on `limits` and apply ggplot2's defaults: 5% expansion,
    /// `extended_breaks(n = 5)` over the expanded range, censored to it, labelled by `format()`.
    ///
    /// A zero-width or empty range keeps this crate's earlier rule (widen by 1 each side, then
    /// expand). ggplot2 widens a zero-width range by 0.5 instead; that case has no oracle row yet.
    #[must_use]
    pub fn continuous(limits: (f64, f64)) -> Self {
        let continuous_range = expand_continuous(limits);
        let (lo, hi) = continuous_range;
        let proposed = extended(lo, hi, DEFAULT_N_BREAKS, &Q_DEFAULT, &W_DEFAULT);
        let breaks: Vec<f64> = proposed.iter().copied().filter(|b| *b >= lo && *b <= hi).collect();
        let n_breaks_censored = proposed.len() - breaks.len();
        let labels = format_labels(&breaks);
        Self { limits, continuous_range, breaks, labels, n_breaks_censored }
    }
}

/// `expand_range4(limits, expansion(mult = 0.05))`. A zero-width range (one distinct value)
/// pads by `mult * 1` each side, as ggplot2 4.0.3 does (measured: limits `[5, 5]` give
/// `[4.95, 5.05]`); a range that trained on nothing (`lo > hi`) widens by one on each side first.
fn expand_continuous((lo, hi): (f64, f64)) -> (f64, f64) {
    if lo > hi {
        let (lo, hi) = (lo - 1.0, hi + 1.0);
        let pad = (hi - lo) * CONTINUOUS_EXPAND_MULT + CONTINUOUS_EXPAND_ADD;
        return (lo - pad, hi + pad);
    }
    let width = if lo == hi { 1.0 } else { hi - lo };
    let pad = width * CONTINUOUS_EXPAND_MULT + CONTINUOUS_EXPAND_ADD;
    (lo - pad, hi + pad)
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

/// Significant digits R's `format()` uses by default (`getOption("digits")`).
const R_DIGITS: i32 = 7;

/// Digits needed right of the decimal point to print `v` to `R_DIGITS` significant digits with no
/// trailing zeros (the `rgt` of R's `formatReal`).
fn decimals_needed(v: f64) -> usize {
    if v == 0.0 || !v.is_finite() {
        return 0;
    }
    let a = v.abs();
    let mut e = libm::floor(libm::log10(a)) as i32;
    // Mantissa scaled to R_DIGITS integer digits, rounded as R rounds it.
    let mut m = libm::round(a / libm::pow(10.0, f64::from(e - (R_DIGITS - 1))));
    let full = libm::pow(10.0, f64::from(R_DIGITS));
    if m >= full {
        m /= 10.0;
        e += 1;
    }
    let mut sig = R_DIGITS;
    let mut mi = m as u64;
    while sig > 1 && mi % 10 == 0 {
        mi /= 10;
        sig -= 1;
    }
    usize::try_from(sig - e - 1).unwrap_or(0)
}

/// Label breaks as R's `format(x, trim = TRUE)` does: every label carries the same number of
/// decimals, the fewest that show each value to 7 significant digits.
///
/// Fixed notation only. R switches to scientific notation when it is narrower (e.g. `1e+06`);
/// that branch has no oracle row yet and is not implemented.
#[must_use]
pub fn format_labels(values: &[f64]) -> Vec<String> {
    let d = values.iter().map(|v| decimals_needed(*v)).max().unwrap_or(0);
    values
        .iter()
        .map(|v| {
            // R never prints a negative zero.
            let v = if *v == 0.0 { 0.0 } else { *v };
            format!("{v:.d$}")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_share_decimals() {
        assert_eq!(format_labels(&[-2.5, 0.0, 2.5, 5.0]), ["-2.5", "0.0", "2.5", "5.0"]);
        assert_eq!(format_labels(&[5.0, 10.0]), ["5", "10"]);
        assert_eq!(format_labels(&[0.25, 0.5]), ["0.25", "0.50"]);
        assert_eq!(format_labels(&[-0.0]), ["0"]);
        assert!(format_labels(&[]).is_empty());
    }

    #[test]
    fn decimals_round_to_seven_significant_digits() {
        // 1/3 prints as 0.3333333 in R: 7 significant digits.
        assert_eq!(decimals_needed(1.0 / 3.0), 7);
        assert_eq!(decimals_needed(1234.5), 1);
        assert_eq!(decimals_needed(9_999_999.6), 0);
    }

    #[test]
    fn zero_width_range_pads_by_five_percent_of_one() {
        // Measured against ggplot2 4.0.3: limits [5, 5] -> [4.95, 5.05] (geom_text single_row).
        let s = PositionScale::continuous((5.0, 5.0));
        assert!((s.continuous_range.0 - 4.95).abs() < 1e-12);
        assert!((s.continuous_range.1 - 5.05).abs() < 1e-12);
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
