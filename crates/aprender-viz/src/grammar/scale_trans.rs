//! Scale transformations and the break algorithms that go with them (`scales` 1.4.0).
//!
//! [`Transform::Log10`] is `scales::transform_log10()`; [`log_breaks`] is `scales::breaks_log()`
//! (with `log_sub_breaks` for narrow ranges); [`regular_minor_breaks`] is
//! `scales::regular_minor_breaks()`, the default minor breaks of every continuous scale.
//!
//! `log10` and `10^x` go through [`libm`], not the platform libm, so a tick never moves with the
//! host (APEX-001 EV-2a rule 5).

use crate::breaks::{extended, Q_DEFAULT, W_DEFAULT};

/// A continuous position scale's transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transform {
    /// `transform_identity()`.
    Identity,
    /// `transform_log10()`: `log10(x)` forward, `10^x` inverse.
    Log10,
}

impl Transform {
    /// Data space to scale space, as R's `log10()`: `0` gives `-Inf`, a negative (or `-Inf`, or
    /// `NaN`) gives `NaN`, `+Inf` stays `+Inf`.
    #[must_use]
    pub fn forward(self, x: f64) -> f64 {
        match self {
            Self::Identity => x,
            Self::Log10 => {
                if x > 0.0 {
                    libm::log10(x)
                } else if x == 0.0 {
                    f64::NEG_INFINITY
                } else {
                    f64::NAN
                }
            }
        }
    }

    /// Scale space back to data space.
    #[must_use]
    pub fn inverse(self, x: f64) -> f64 {
        match self {
            Self::Identity => x,
            Self::Log10 => libm::pow(10.0, x),
        }
    }
}

/// `scales::breaks_log(n = 5, base = 10)` applied to a range in data space.
///
/// Returns breaks in data space (powers of ten, or `1,2,5`-style sub-breaks for a narrow range);
/// the caller transforms and censors them.
#[must_use]
pub fn log_breaks(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    if !(lo.is_finite() && hi.is_finite()) || lo <= 0.0 || hi <= 0.0 {
        return Vec::new();
    }
    let (rlo, rhi) = (libm::log10(lo.min(hi)), libm::log10(lo.max(hi)));
    let min = libm::floor(rlo);
    let max = libm::ceil(rhi);
    #[allow(clippy::float_cmp)]
    if max == min {
        return vec![libm::pow(10.0, min)];
    }
    // R tries `by = floor((max - min) / n) + 1`, then `by - 1`, ... down to 1, and keeps the first
    // step whose powers put at least `n - 2` breaks inside the data range.
    let first_by = libm::floor((max - min) / n as f64) as usize + 1;
    let (rng_lo, rng_hi) = (libm::pow(10.0, rlo), libm::pow(10.0, rhi));
    let relevant =
        |breaks: &[f64]| breaks.iter().filter(|b| rng_lo <= **b && **b <= rng_hi).count();
    (1..=first_by)
        .rev()
        .map(|by| power_seq(min, max, by as f64))
        .find(|breaks| relevant(breaks) + 2 >= n)
        .unwrap_or_else(|| log_sub_breaks((rlo, rhi), n))
}

/// `base^seq(min, max, by = by)` for base 10 (R's `seq(by=)`: `floor((to-from)/by + 1e-10)` steps).
fn power_seq(min: f64, max: f64, by: f64) -> Vec<f64> {
    let steps = libm::floor((max - min) / by + 1e-10) as usize;
    (0..=steps).map(|i| libm::pow(10.0, min + i as f64 * by)).collect()
}

/// `scales:::log_sub_breaks(rng, n, base = 10)`: powers of ten times a growing set of multipliers
/// chosen to be as evenly spread on the log axis as possible.
fn log_sub_breaks((rlo, rhi): (f64, f64), n: usize) -> Vec<f64> {
    let min = libm::floor(rlo) as i32;
    let max = libm::ceil(rhi) as i32;
    let (rng_lo, rng_hi) = (libm::pow(10.0, rlo), libm::pow(10.0, rhi));
    let mut steps: Vec<f64> = vec![1.0];
    let mut candidate: Vec<f64> = (2..10).map(f64::from).collect();
    let delta = |x: f64, steps: &[f64]| -> f64 {
        let mut v: Vec<f64> = steps.to_vec();
        v.push(x);
        v.push(10.0);
        v.sort_by(f64::total_cmp);
        v.windows(2).map(|w| libm::log10(w[1]) - libm::log10(w[0])).fold(f64::INFINITY, f64::min)
    };
    let powers: Vec<f64> = (min..=max).map(|k| libm::pow(10.0, f64::from(k))).collect();
    let mut breaks: Vec<f64> = Vec::new();
    let mut n_relevant = 0;
    while !candidate.is_empty() {
        // `which.max`: the first index of the largest delta.
        let mut best = 0;
        let mut best_delta = f64::NEG_INFINITY;
        for (i, c) in candidate.iter().enumerate() {
            let d = delta(*c, &steps);
            if d > best_delta {
                best_delta = d;
                best = i;
            }
        }
        steps.push(candidate.remove(best));
        // `as.vector(outer(powers, steps))` is column-major: all powers for step 1, then step 2, ...
        breaks = steps.iter().flat_map(|s| powers.iter().map(move |p| p * s)).collect();
        n_relevant = breaks.iter().filter(|b| rng_lo <= **b && **b <= rng_hi).count();
        if n_relevant + 2 >= n {
            break;
        }
    }
    if n_relevant + 2 >= n {
        breaks.sort_by(f64::total_cmp);
        let first = breaks.iter().position(|b| rng_lo <= *b).unwrap_or(0);
        let last = breaks.iter().rposition(|b| *b <= rng_hi).unwrap_or(breaks.len() - 1);
        let lower = first.saturating_sub(1);
        let upper = (last + 1).min(breaks.len() - 1);
        breaks[lower..=upper].to_vec()
    } else {
        extended(rng_lo, rng_hi, n, &Q_DEFAULT, &W_DEFAULT)
    }
}

/// `scales::regular_minor_breaks()(b, limits, n = 2)`: the midpoints between major breaks, with one
/// extra major step added at each end when the limits reach beyond the majors. Returns the result
/// before ggplot2 discards the values outside the limits.
#[must_use]
pub fn regular_minor_breaks(major: &[f64], limits: (f64, f64), n: usize) -> Vec<f64> {
    let mut b: Vec<f64> = major.iter().copied().filter(|v| !v.is_nan()).collect();
    if b.len() < 2 {
        return Vec::new();
    }
    let bd = b[1] - b[0];
    let (lo, hi) = (limits.0.min(limits.1), limits.0.max(limits.1));
    let b_min = b.iter().copied().fold(f64::INFINITY, f64::min);
    if lo < b_min {
        b.insert(0, b[0] - bd);
    }
    // R tests `max(b)` after the first insertion: with unsorted majors (`c(5, 1, 3)`) the
    // inserted end can be the new maximum.
    let b_max = b.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if hi > b_max {
        b.push(b[b.len() - 1] + bd);
    }
    let mut out = Vec::new();
    for w in b.windows(2) {
        // `seq(a, b, length.out = n + 1)[-(n + 1)]`
        let step = (w[1] - w[0]) / n as f64;
        out.extend((0..n).map(|i| w[0] + i as f64 * step));
    }
    out.push(b[b.len() - 1]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log10_round_trips_powers_of_ten() {
        assert!((Transform::Log10.forward(1000.0) - 3.0).abs() < 1e-15);
        assert!((Transform::Log10.inverse(2.0) - 100.0).abs() < 1e-12);
        assert!(Transform::Log10.forward(-1.0).is_nan());
        assert_eq!(Transform::Log10.forward(0.0), f64::NEG_INFINITY);
        assert_eq!(Transform::Log10.forward(f64::INFINITY), f64::INFINITY);
        assert!(Transform::Log10.forward(f64::NEG_INFINITY).is_nan());
        assert!(Transform::Log10.forward(f64::NAN).is_nan());
    }

    #[test]
    fn wide_range_gives_decades() {
        assert_eq!(log_breaks(0.7, 2900.0, 5), [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]);
    }

    /// `scales::breaks_log()(c(3, 9.9e4))` in scales 1.4.0: `by = 2` leaves only 2 of 3 powers in
    /// range, so R steps down to `by = 1`.
    #[test]
    fn steps_down_to_by_one_like_r() {
        assert_eq!(log_breaks(3.0, 9.9e4, 5), [1.0, 10.0, 100.0, 1000.0, 10000.0, 100_000.0]);
    }

    /// `scales::breaks_log()(c(1.5, 120))` is `1 3 10 30 100 300`: no power step keeps 3 breaks
    /// in range, so R falls through to `log_sub_breaks`.
    #[test]
    fn falls_through_to_sub_breaks_like_r() {
        let got = log_breaks(1.5, 120.0, 5);
        let want = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0];
        assert_eq!(got.len(), want.len(), "{got:?}");
        assert!(got.iter().zip(want).all(|(g, w)| (g - w).abs() <= 1e-9 * w), "{got:?}");
    }

    /// A range with no positive finite end has no log breaks (`breaks_log` would see NaN).
    #[test]
    fn no_breaks_without_a_positive_finite_range() {
        assert!(log_breaks(0.0, 10.0, 5).is_empty());
        assert!(log_breaks(-1.0, 10.0, 5).is_empty());
        assert!(log_breaks(1.0, f64::INFINITY, 5).is_empty());
        assert!(log_breaks(f64::NAN, 10.0, 5).is_empty());
    }

    #[test]
    fn single_decade_is_one_break() {
        assert_eq!(log_breaks(10.0, 10.0, 5), [10.0]);
    }

    #[test]
    fn minor_breaks_extend_one_step_each_side() {
        let m = regular_minor_breaks(&[0.0, 2.0], (-0.5, 2.5), 2);
        assert_eq!(m, [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
        assert!(regular_minor_breaks(&[1.0], (0.0, 2.0), 2).is_empty());
    }
}

#[cfg(test)]
mod minor_break_order_tests {
    use super::*;

    /// `scales::regular_minor_breaks()(c(5, 1, 3), c(0.75, 6.25), 2)` in R: the low end is inserted
    /// first (`9 5 1 3`), and `max(b)` is then 9, so nothing is appended. Before the fix the stale
    /// maximum (5) appended `-1` and an extra `1` trailed the result.
    #[test]
    fn unsorted_majors_test_the_maximum_after_the_low_end_insert() {
        let got = regular_minor_breaks(&[5.0, 1.0, 3.0], (0.75, 6.25), 2);
        let inside: Vec<f64> = got.into_iter().filter(|m| (0.75..=6.25).contains(m)).collect();
        assert_eq!(inside, [5.0, 3.0, 1.0, 2.0, 3.0]);
    }
}
