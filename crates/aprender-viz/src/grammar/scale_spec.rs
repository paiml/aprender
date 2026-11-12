//! Position-scale arguments (`scale_x_continuous(limits, breaks, labels, expand)`,
//! `scale_x_log10()`, `scale_x_discrete()`) and the resolution of a trained axis into the panel
//! view ggplot2 reports.
//!
//! Everything is computed in the scale's own space, as ggplot2 does: for `scale_*_log10()` the
//! limits, the 5% expansion, the breaks and the layer positions are `log10` values; only the
//! labels come from data space (`format(10^break)`).
//!
//! Rows a scale cannot place stay in the layer with an `NA` position (`f64::NAN`), as
//! `ggplot_build()` returns them; ggplot2 drops them (with a warning) only when drawing. `oob =`
//! is fixed to `censor`, which (`only.finite = TRUE`) leaves infinite values alone. Exactly:
//!
//! - a negative value on `log10` is `NA`; `0` is `-Inf`, and `+-Inf` data stay `+-Inf` (except
//!   `-Inf` on `log10`, which is `NA`, as R's `log10(-Inf)`);
//! - a finite value outside explicit `limits` is `NA`; an infinite value is kept;
//! - text outside the levels of a discrete scale is `NA`.
//!
//! Training ignores non-finite values. Where ggplot2 errors, so does this: text in a continuous
//! scale. Deliberate divergences, each returning `Err(ScaleDomain)`: a continuous axis with no
//! limits and no finite value to train on (ggplot2 returns an infinite range); and explicit
//! `limits` that are not finite after the transform (`limits = c(0, 100)` on `log10` errors in
//! ggplot2; a negative limit, or an infinite one, makes it silently substitute the data range or
//! return unusable breaks).
//!
//! Not implemented: label functions, `n.breaks`, `minor_breaks =`, other `oob =` functions,
//! `position =`, secondary axes, and the `expand = c(a, b, c, d)` shorthand (use [`Expansion`]).
//!
//! Scope: these scales are resolved for `geom_point` layers with the identity stat, under the
//! default `coord_cartesian()` and no facet. Any other geom, stat, coord or a facet in a plot that
//! uses a position scale is refused by [`require_scale_scope`] with a typed error, never laid out
//! as if it were absent.

use super::build::{
    censor_breaks, PositionScale, CONTINUOUS_EXPAND_ADD, CONTINUOUS_EXPAND_MULT, DEFAULT_N_BREAKS,
};
use super::coord::Coord;
use super::data::DataValue;
use super::facet::Facet;
use super::geom::GeomType;
use super::ggplot::Layer;
use super::r_format::format_labels;
use super::scale_discrete::{position, resolve_discrete, train_levels, DiscreteSpec};
use super::scale_trans::{log_breaks, regular_minor_breaks, Transform};
use super::stat::Stat;
use crate::breaks::{extended, Q_DEFAULT, W_DEFAULT};
use crate::error::{Error, Result};

/// `ggplot2::expansion(mult, add)`: how far the panel extends past the limits, per side
/// (lower, upper). A side grows by `span * mult + add`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Expansion {
    /// Multiplicative expansion, (lower, upper).
    pub mult: (f64, f64),
    /// Additive expansion, (lower, upper).
    pub add: (f64, f64),
}

impl Expansion {
    /// `expansion(mult, add)` with both sides given.
    #[must_use]
    pub fn new(mult: (f64, f64), add: (f64, f64)) -> Self {
        Self { mult, add }
    }

    /// `expansion(0, 0)`: the panel is exactly the limits.
    #[must_use]
    pub fn none() -> Self {
        Self::new((0.0, 0.0), (0.0, 0.0))
    }

    /// `expansion(mult = m)`: the same on both sides.
    #[must_use]
    pub fn mult(m: f64) -> Self {
        Self::new((m, m), (0.0, 0.0))
    }

    /// `expansion(add = a)`: the same on both sides.
    #[must_use]
    pub fn add(a: f64) -> Self {
        Self::new((0.0, 0.0), (a, a))
    }

    /// `default_expansion()` for a continuous scale: 5% each side.
    #[must_use]
    pub fn continuous_default() -> Self {
        Self::new(
            (CONTINUOUS_EXPAND_MULT, CONTINUOUS_EXPAND_MULT),
            (CONTINUOUS_EXPAND_ADD, CONTINUOUS_EXPAND_ADD),
        )
    }

    /// `default_expansion(discrete = expansion(add = 0.6))` for a discrete scale.
    #[must_use]
    pub fn discrete_default() -> Self {
        Self::add(super::scale_discrete::DISCRETE_EXPAND_ADD)
    }

    /// `expand_range4(limits, expansion)` = `scales::expand_range(limits, mul, add, zero_width = 1)`:
    /// `c(lo - (w * mult_l + add_l), hi + (w * mult_u + add_u))`, where `w` is the span, or 1 when
    /// the range has zero width (`scales::zero_range`). A single value `v` on a default axis
    /// therefore spans `v +- 0.05`, measured on ggplot2 4.0.3.
    #[must_use]
    pub fn apply(self, (lo, hi): (f64, f64)) -> (f64, f64) {
        let width = if zero_range(lo, hi) { ZERO_RANGE_WIDTH } else { hi - lo };
        (lo - (width * self.mult.0 + self.add.0), hi + (width * self.mult.1 + self.add.1))
    }
}

/// `scales::expand_range(zero_width = 1)`: the width a zero-width range is given when the
/// multiplicative expansion is applied.
pub const ZERO_RANGE_WIDTH: f64 = 1.0;

/// `scales::zero_range(c(lo, hi))` with its default `tol = 1000 * .Machine$double.eps`: equal, or
/// closer than `tol` relative to the smaller magnitude.
#[must_use]
pub fn zero_range(lo: f64, hi: f64) -> bool {
    if lo == hi {
        return true;
    }
    if !lo.is_finite() || !hi.is_finite() {
        return false;
    }
    let m = lo.abs().min(hi.abs());
    m != 0.0 && ((lo - hi) / m).abs() < 1000.0 * f64::EPSILON
}

/// The arguments of `scale_x_continuous()` / `scale_y_continuous()` / `scale_*_log10()`.
///
/// `limits` and `breaks` are in data space (for a log scale, the untransformed values);
/// `labels` is positional against `breaks`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ContinuousSpec {
    pub(crate) log10: bool,
    pub(crate) limits: Option<(f64, f64)>,
    pub(crate) breaks: Option<Vec<f64>>,
    pub(crate) labels: Option<Vec<String>>,
    pub(crate) expand: Option<Expansion>,
}

impl ContinuousSpec {
    /// `scale_x_continuous()` with every default.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// `scale_x_log10()`: `transform_log10()`, log breaks.
    #[must_use]
    pub fn log10() -> Self {
        Self { log10: true, ..Self::default() }
    }

    /// `limits = c(lo, hi)`.
    #[must_use]
    pub fn limits(mut self, lo: f64, hi: f64) -> Self {
        self.limits = Some((lo.min(hi), lo.max(hi)));
        self
    }

    /// `breaks = c(...)`.
    #[must_use]
    pub fn breaks(mut self, breaks: impl IntoIterator<Item = f64>) -> Self {
        self.breaks = Some(breaks.into_iter().collect());
        self
    }

    /// `labels = c(...)`, one per break.
    #[must_use]
    pub fn labels<S: Into<String>>(mut self, labels: impl IntoIterator<Item = S>) -> Self {
        self.labels = Some(labels.into_iter().map(Into::into).collect());
        self
    }

    /// `expand = expansion(...)`.
    #[must_use]
    pub fn expand(mut self, expand: Expansion) -> Self {
        self.expand = Some(expand);
        self
    }

    fn transform(&self) -> Transform {
        if self.log10 {
            Transform::Log10
        } else {
            Transform::Identity
        }
    }
}

/// `scale_x_continuous()` / `scale_y_continuous()`.
#[must_use]
pub fn scale_continuous() -> ContinuousSpec {
    ContinuousSpec::new()
}

/// `scale_x_log10()` / `scale_y_log10()`.
#[must_use]
pub fn scale_log10() -> ContinuousSpec {
    ContinuousSpec::log10()
}

/// `scale_x_discrete()` / `scale_y_discrete()`.
#[must_use]
pub fn scale_discrete() -> DiscreteSpec {
    DiscreteSpec::new()
}

/// A position scale specification: what `scale_x_*()` / `scale_y_*()` was called with.
#[derive(Debug, Clone, PartialEq)]
pub enum PositionScaleSpec {
    /// Continuous (identity or log10 transform).
    Continuous(ContinuousSpec),
    /// Discrete.
    Discrete(DiscreteSpec),
}

impl From<ContinuousSpec> for PositionScaleSpec {
    fn from(s: ContinuousSpec) -> Self {
        Self::Continuous(s)
    }
}

impl From<DiscreteSpec> for PositionScaleSpec {
    fn from(s: DiscreteSpec) -> Self {
        Self::Discrete(s)
    }
}

/// The limits of an axis that was never trained: ggplot2 gives an empty continuous scale 0..1.
pub const EMPTY_AXIS_LIMITS: (f64, f64) = (0.0, 1.0);

/// Resolve a continuous axis. `trained` is the data range in scale space; an explicit
/// `spec.limits` replaces it.
#[must_use]
pub fn resolve_continuous(trained: (f64, f64), spec: &ContinuousSpec) -> PositionScale {
    let t = spec.transform();
    let limits = spec.limits.map_or(trained, |(a, b)| (t.forward(a), t.forward(b)));
    // A zero-width range is kept as is: `Expansion::apply` gives it ggplot2's width of 1. An
    // inverted range only arises from an untrained axis (no data at all); ggplot2's empty
    // continuous scale spans 0..1, and so does this.
    let (lo, hi) = if limits.0 <= limits.1 { limits } else { EMPTY_AXIS_LIMITS };
    let continuous_range = spec.expand.unwrap_or(Expansion::continuous_default()).apply((lo, hi));
    let (rlo, rhi) = continuous_range;

    // Breaks, in scale space.
    let proposed: Vec<f64> = match &spec.breaks {
        Some(b) => b.iter().map(|v| t.forward(*v)).collect(),
        None if spec.log10 => log_breaks(t.inverse(rlo), t.inverse(rhi), DEFAULT_N_BREAKS)
            .into_iter()
            .map(|v| t.forward(v))
            .collect(),
        None => extended(rlo, rhi, DEFAULT_N_BREAKS, &Q_DEFAULT, &W_DEFAULT),
    };
    let kept_idx = censor_breaks(&proposed, rlo, rhi);
    let breaks: Vec<f64> = kept_idx.iter().map(|i| proposed[*i]).collect();
    let n_breaks_censored = proposed.len() - breaks.len();

    let labels = match &spec.labels {
        Some(l) => kept_idx.iter().map(|i| l.get(*i).cloned().unwrap_or_default()).collect(),
        None => {
            let data_space: Vec<f64> = breaks.iter().map(|b| t.inverse(*b)).collect();
            format_labels(&data_space)
        }
    };

    // ggplot2 discards minor breaks outside the limits it was given (the panel range).
    let minor_breaks: Vec<f64> = regular_minor_breaks(&breaks, continuous_range, 2)
        .into_iter()
        .filter(|m| *m >= rlo && *m <= rhi)
        .collect();

    PositionScale {
        limits: (lo, hi),
        continuous_range,
        breaks,
        labels,
        n_breaks_censored,
        minor_breaks,
        levels: None,
    }
}

/// Refuse what a plot with a position scale (`scale_x`/`scale_y`, or a text position column) may
/// not contain: `layout()` resolves these scales for `geom_point` layers with the identity stat,
/// under the default `coord_cartesian()` and no facet, and applies no other geom's setup, no stat,
/// no coord and no facet. Every layer is checked, not just the first.
///
/// # Errors
///
/// [`Error::UnsupportedLayer`] for a geom other than `geom_point` or for any facet,
/// [`Error::UnsupportedStat`] for a stat other than identity, and [`Error::UnsupportedCoord`] for
/// `coord_flip()`, `coord_cartesian()` limits, `coord_polar()` or `coord_fixed()`.
pub(crate) fn require_scale_scope(layers: &[Layer], coord: &Coord, facet: &Facet) -> Result<()> {
    for l in layers {
        if !matches!(l.geom.geom_type, GeomType::Point { .. }) {
            return Err(Error::UnsupportedLayer(
                "position scales (scale_x/scale_y or a text position) are resolved for geom_point \
                 only: layout() does not lay out other geoms under them",
            ));
        }
        if !matches!(l.geom.stat, None | Some(Stat::Identity)) {
            return Err(Error::UnsupportedStat(
                "a stat other than identity with position scales: layout() computes none",
            ));
        }
    }
    if !matches!(facet, Facet::None) {
        return Err(Error::UnsupportedLayer(
            "a facet with position scales: layout() does not split the panels",
        ));
    }
    match coord {
        Coord::Cartesian { xlim: None, ylim: None, flip: false } => Ok(()),
        Coord::Cartesian { flip: true, .. } => {
            Err(Error::UnsupportedCoord("coord_flip with position scales: layout() does not flip"))
        }
        Coord::Cartesian { .. } => Err(Error::UnsupportedCoord(
            "coord_cartesian limits with position scales: layout() does not apply them",
        )),
        Coord::Polar { .. } => Err(Error::UnsupportedCoord("coord_polar with position scales")),
        Coord::Fixed { .. } => Err(Error::UnsupportedCoord("coord_fixed with position scales")),
    }
}

/// ggplot2's message when text reaches a continuous scale.
fn text_in_continuous() -> Error {
    Error::ScaleDomain("Discrete value supplied to a continuous scale".into())
}

/// How a data value becomes a position on one axis.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AxisMap {
    /// Numbers pass through (after the transform; a value outside the limits is `NA`).
    Continuous { transform: Transform, limits: Option<(f64, f64)> },
    /// Text becomes its level position; numbers pass through.
    Discrete { levels: Vec<String> },
}

/// A resolved axis: the panel view plus the data-to-position map.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedAxis {
    pub(crate) scale: PositionScale,
    map: AxisMap,
}

impl ResolvedAxis {
    /// Train on the columns that feed this axis (one per layer) and resolve with `spec`.
    ///
    /// With no spec, a column holding any text makes the axis discrete (ggplot2's default scale
    /// for character data); otherwise it is continuous.
    ///
    /// # Errors
    ///
    /// [`Error::ScaleDomain`] where ggplot2 errors (text in a continuous scale), and where it
    /// would return an infinite range: a continuous axis with no limits whose data has no finite
    /// value after the transform (all non-positive on `log10`, all missing). This crate refuses
    /// rather than return an `(inf, -inf)` panel. Explicit limits that are not finite after the transform
    /// (`limits = c(0, 100)` on `log10`, where R errors with "'from' must be a finite number") are the
    /// same error.
    pub(crate) fn resolve(
        spec: Option<&PositionScaleSpec>,
        columns: &[&[DataValue]],
    ) -> Result<Self> {
        let has_text = columns.iter().flat_map(|c| c.iter()).any(|v| v.as_str().is_some());
        match spec {
            Some(PositionScaleSpec::Discrete(d)) => Self::discrete(d, columns),
            None if has_text => Self::discrete(&DiscreteSpec::new(), columns),
            Some(PositionScaleSpec::Continuous(_)) if has_text => Err(text_in_continuous()),
            Some(PositionScaleSpec::Continuous(c)) => Self::continuous(c, columns),
            None => Self::continuous(&ContinuousSpec::new(), columns),
        }
    }

    fn discrete(spec: &DiscreteSpec, columns: &[&[DataValue]]) -> Result<Self> {
        let trained =
            train_levels(columns.iter().flat_map(|c| c.iter()).filter_map(DataValue::as_str));
        let scale = resolve_discrete(trained, spec);
        let levels = scale.levels.clone().unwrap_or_default();
        if levels.is_empty() {
            // R: `scale_x_discrete(limits = character(0))`, or no text to train on, gives the
            // panel `c(-Inf, Inf)`; refused like a continuous axis with nothing to train on.
            return Err(Error::ScaleDomain(
                "discrete scale has no levels (no text values and no limits)".into(),
            ));
        }
        Ok(Self { scale, map: AxisMap::Discrete { levels } })
    }

    fn continuous(spec: &ContinuousSpec, columns: &[&[DataValue]]) -> Result<Self> {
        let t = spec.transform();
        let trained = columns
            .iter()
            .flat_map(|c| c.iter())
            .filter_map(DataValue::as_f32)
            .map(|v| t.forward(f64::from(v)))
            .filter(|v| v.is_finite())
            .fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
        if spec.limits.is_none() && trained.0 > trained.1 {
            return Err(Error::ScaleDomain(format!(
                "continuous{} scale has no finite value to train on",
                if spec.log10 { " log10" } else { "" }
            )));
        }
        if let Some((a, b)) = spec.limits {
            if !(t.forward(a).is_finite() && t.forward(b).is_finite()) {
                return Err(Error::ScaleDomain(format!(
                    "scale limits ({a}, {b}) are not finite after the transform"
                )));
            }
        }
        let scale = resolve_continuous(trained, spec);
        let limits = spec.limits.map(|(a, b)| (t.forward(a), t.forward(b)));
        Ok(Self { scale, map: AxisMap::Continuous { transform: t, limits } })
    }

    /// A data value's position, `NaN` where ggplot2 would have `NA` (text outside the levels, a
    /// missing value on a continuous axis, a negative value on a log axis, a finite value outside
    /// explicit limits).
    ///
    /// # Errors
    ///
    /// [`Error::ScaleDomain`] for a number or a missing value on a discrete axis. ggplot2 places
    /// numbers on a continuous range beside the levels and turns a missing value into an `NA`
    /// level; neither is reproduced, so neither is returned as a position.
    pub(crate) fn position(&self, v: &DataValue) -> Result<f64> {
        match (&self.map, v) {
            (AxisMap::Discrete { levels }, DataValue::Text(s)) => {
                Ok(position(levels, s).unwrap_or(f64::NAN))
            }
            (AxisMap::Discrete { .. }, DataValue::Number(n)) => Err(Error::ScaleDomain(format!(
                "a discrete axis holds the number {n}: ggplot2 places numbers on a continuous \
                 range beside the levels, which is not supported"
            ))),
            (AxisMap::Discrete { .. }, DataValue::Null) => Err(Error::ScaleDomain(
                "a discrete axis holds a missing value: ggplot2 makes it an NA level, which is not \
                 supported"
                    .into(),
            )),
            (AxisMap::Continuous { transform, limits }, DataValue::Number(n)) => {
                let p = transform.forward(f64::from(*n));
                // ggplot2's `censor(only.finite = TRUE)`: an infinite value is never censored.
                Ok(match limits {
                    Some((lo, hi)) if p.is_finite() && (p < *lo || p > *hi) => f64::NAN,
                    _ => p,
                })
            }
            // A missing value is NA. Text cannot reach here: `resolve` refuses it.
            (AxisMap::Continuous { .. }, _) => Ok(f64::NAN),
        }
    }
}
