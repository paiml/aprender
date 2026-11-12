//! `scale_x_discrete()` / `scale_y_discrete()`: levels become integer positions `1..=n`.
//!
//! The API is deliberately small: [`DiscreteSpec`] (limits, breaks, labels, expand), [`train_levels`]
//! (the sorted distinct values of a column), [`position`] (level to its 1-based position) and
//! [`resolve_discrete`] (the panel view). Other layers (`stat_count`, `position_dodge`) place
//! categories through these.
//!
//! ggplot2 4.0.3 defaults: levels are the sorted distinct values (`scales::DiscreteRange`), the
//! panel spans `(1 - 0.6, n + 0.6)` (`default_expansion(discrete = expansion(add = 0.6))`), every
//! level is a break, and labels are the level names. A single level spans `(0.4, 1.6)`: the
//! same expansion applied to the zero-width range `(1, 1)` (measured, ggplot2 4.0.3).
//!
//! Levels sort by byte order. R sorts by locale collation, which agrees for ASCII lower-case names;
//! mixed-case names may order differently from R on a non-C locale.

use super::build::PositionScale;
use super::scale_spec::Expansion;

/// ggplot2's `default_expansion(discrete = expansion(add = 0.6))`: each side of a discrete axis
/// grows by 0.6 of a level step.
pub const DISCRETE_EXPAND_ADD: f64 = 0.6;

/// The arguments of `scale_x_discrete()` / `scale_y_discrete()` this crate understands.
///
/// Named labels (`labels = c(a = "Alpha")`) and label functions are not supported; `labels` is
/// positional against `breaks` (or against the limits when `breaks` is unset).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DiscreteSpec {
    pub(crate) limits: Option<Vec<String>>,
    pub(crate) breaks: Option<Vec<String>>,
    pub(crate) labels: Option<Vec<String>>,
    pub(crate) expand: Option<Expansion>,
}

impl DiscreteSpec {
    /// Defaults: levels from the data, every level a break, `expand = expansion(add = 0.6)`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// `limits =`: the levels, in order. A level absent from the data still takes a position.
    #[must_use]
    pub fn limits<S: Into<String>>(mut self, limits: impl IntoIterator<Item = S>) -> Self {
        self.limits = Some(limits.into_iter().map(Into::into).collect());
        self
    }

    /// `breaks =`: which levels get a tick, in this order.
    #[must_use]
    pub fn breaks<S: Into<String>>(mut self, breaks: impl IntoIterator<Item = S>) -> Self {
        self.breaks = Some(breaks.into_iter().map(Into::into).collect());
        self
    }

    /// `labels =`: one label per break, positionally.
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
}

/// The sorted distinct values: the levels ggplot2 trains on a character column.
#[must_use]
pub fn train_levels<'a>(values: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let mut levels: Vec<String> = values.into_iter().map(str::to_owned).collect();
    levels.sort();
    levels.dedup();
    levels
}

/// The 1-based position of `value` among `levels`; `None` for a value outside the limits.
#[must_use]
pub fn position(levels: &[String], value: &str) -> Option<f64> {
    levels.iter().position(|l| l == value).map(|i| (i + 1) as f64)
}

/// The panel range of `n` levels: `expand_range4(c(1, n), expand)`.
#[must_use]
pub fn discrete_range(n: usize, expand: Expansion) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    expand.apply((1.0, n as f64))
}

/// Resolve a discrete axis: `trained` levels (used unless the spec gives `limits`) into the panel
/// view. `limits` is `(1, n)`; the level names are in [`PositionScale::levels`], and `breaks` hold
/// the break positions with `labels` beside them.
#[must_use]
pub fn resolve_discrete(trained: Vec<String>, spec: &DiscreteSpec) -> PositionScale {
    let levels = spec.limits.clone().unwrap_or(trained);
    let n = levels.len();
    let expand = spec.expand.unwrap_or(Expansion::discrete_default());
    let continuous_range = discrete_range(n, expand);

    let requested: Vec<String> = spec.breaks.clone().unwrap_or_else(|| levels.clone());
    let mut breaks = Vec::new();
    let mut labels = Vec::new();
    for (i, name) in requested.iter().enumerate() {
        // `intersect(breaks, limits)`: breaks that are not levels are dropped.
        let Some(pos) = position(&levels, name) else { continue };
        breaks.push(pos);
        labels.push(
            spec.labels
                .as_ref()
                .map_or_else(|| name.clone(), |l| l.get(i).cloned().unwrap_or_default()),
        );
    }
    PositionScale {
        limits: (1.0, n.max(1) as f64),
        continuous_range,
        breaks,
        labels,
        n_breaks_censored: 0,
        minor_breaks: Vec::new(),
        levels: Some(levels),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_sorted_and_distinct() {
        assert_eq!(train_levels(["mid", "low", "high", "low"]), ["high", "low", "mid"]);
    }

    #[test]
    fn default_range_adds_point_six() {
        let s = resolve_discrete(train_levels(["a", "b", "c"]), &DiscreteSpec::new());
        assert!((s.continuous_range.0 - 0.4).abs() < 1e-12);
        assert!((s.continuous_range.1 - 3.6).abs() < 1e-12);
        assert_eq!(s.breaks, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn position_is_one_based() {
        let l = train_levels(["b", "a"]);
        assert_eq!(position(&l, "b"), Some(2.0));
        assert_eq!(position(&l, "z"), None);
    }
}
