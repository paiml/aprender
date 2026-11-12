//! Position-scale arguments (`scale_x_continuous(limits, breaks, labels, expand)`,
//! `scale_x_log10()`, `scale_x_discrete()`) and the resolution of a trained axis into the panel
//! view ggplot2 reports.
//!
//! Everything is computed in the scale's own space, as ggplot2 does: for `scale_*_log10()` the
//! limits, the 5% expansion, the breaks and the layer positions are `log10` values; only the
//! labels come from data space (`format(10^break)`).
//!
//! Not implemented: label functions, `n.breaks`, `minor_breaks =`, `oob =` (rows outside explicit
//! limits are dropped from the layer rather than set to `NA`), `position =`, secondary axes, and
//! the `expand = c(a, b, c, d)` shorthand (use [`Expansion`]).

use super::build::{
    censor_breaks, PositionScale, CONTINUOUS_EXPAND_ADD, CONTINUOUS_EXPAND_MULT, DEFAULT_N_BREAKS,
};
use super::data::DataValue;
use super::r_format::format_labels;
use super::scale_discrete::{position, resolve_discrete, train_levels, DiscreteSpec};
use super::scale_trans::{log_breaks, regular_minor_breaks, Transform};
use crate::breaks::{extended, Q_DEFAULT, W_DEFAULT};

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

    /// `expand_range4(limits, expansion)`: `c(lo - (span * mult_l + add_l), hi + (span * mult_u + add_u))`.
    #[must_use]
    pub fn apply(self, (lo, hi): (f64, f64)) -> (f64, f64) {
        let span = hi - lo;
        (lo - (span * self.mult.0 + self.add.0), hi + (span * self.mult.1 + self.add.1))
    }
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

/// Resolve a continuous axis. `trained` is the data range in scale space; an explicit
/// `spec.limits` replaces it.
#[must_use]
pub fn resolve_continuous(trained: (f64, f64), spec: &ContinuousSpec) -> PositionScale {
    let t = spec.transform();
    let limits = spec.limits.map_or(trained, |(a, b)| (t.forward(a), t.forward(b)));
    let (lo, hi) = if limits.0 < limits.1 { limits } else { (limits.0 - 1.0, limits.1 + 1.0) };
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
        limits,
        continuous_range,
        breaks,
        labels,
        n_breaks_censored,
        minor_breaks,
        levels: None,
    }
}

/// How a data value becomes a position on one axis.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AxisMap {
    /// Numbers pass through (after the transform, and outside-limits rows are dropped).
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
    pub(crate) fn resolve(spec: Option<&PositionScaleSpec>, columns: &[&[DataValue]]) -> Self {
        let has_text = columns.iter().flat_map(|c| c.iter()).any(|v| v.as_str().is_some());
        match spec {
            Some(PositionScaleSpec::Discrete(d)) => Self::discrete(d, columns),
            None if has_text => Self::discrete(&DiscreteSpec::new(), columns),
            Some(PositionScaleSpec::Continuous(c)) => Self::continuous(c, columns),
            None => Self::continuous(&ContinuousSpec::new(), columns),
        }
    }

    fn discrete(spec: &DiscreteSpec, columns: &[&[DataValue]]) -> Self {
        let trained =
            train_levels(columns.iter().flat_map(|c| c.iter()).filter_map(DataValue::as_str));
        let scale = resolve_discrete(trained, spec);
        let levels = scale.levels.clone().unwrap_or_default();
        Self { scale, map: AxisMap::Discrete { levels } }
    }

    fn continuous(spec: &ContinuousSpec, columns: &[&[DataValue]]) -> Self {
        let t = spec.transform();
        let trained = columns
            .iter()
            .flat_map(|c| c.iter())
            .filter_map(DataValue::as_f32)
            .map(|v| t.forward(f64::from(v)))
            .filter(|v| v.is_finite())
            .fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
        let scale = resolve_continuous(trained, spec);
        let limits = spec.limits.map(|(a, b)| (t.forward(a), t.forward(b)));
        Self { scale, map: AxisMap::Continuous { transform: t, limits } }
    }

    /// A data value's position, or `None` where ggplot2 would have `NA` (text outside the levels,
    /// a non-positive value on a log axis, a value outside explicit limits).
    pub(crate) fn map_value(&self, v: &DataValue) -> Option<f64> {
        match (&self.map, v) {
            (AxisMap::Discrete { levels }, DataValue::Text(s)) => position(levels, s),
            (AxisMap::Discrete { .. }, DataValue::Number(n)) => Some(f64::from(*n)),
            (AxisMap::Continuous { transform, limits }, DataValue::Number(n)) => {
                let p = transform.forward(f64::from(*n));
                if p.is_nan() && *transform != Transform::Identity {
                    return None;
                }
                match limits {
                    Some((lo, hi)) if p < *lo || p > *hi => None,
                    _ => Some(p),
                }
            }
            _ => None,
        }
    }
}
