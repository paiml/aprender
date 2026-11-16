//! Student-t quantiles for `stat_smooth`'s confidence band (PLOT-GG-001, paiml/aprender#6909).
//!
//! `predict.lm(interval = "confidence")` uses `qt((1 - level) / 2, df)`. This module computes the
//! upper-tail quantile `t` with `P(T > t) = q` to about 1e-14 relative, through the regularised
//! incomplete beta function (continued fraction, modified Lentz) and a bracketed Newton solve.
//! `lgamma` and `log1p` come from the pure-Rust `libm`, as the rest of the render path does.

use std::f64::consts::PI;

const TINY: f64 = 1e-300;
const CF_EPS: f64 = 1e-16;
const CF_MAX_ITER: u32 = 10_000;
const SOLVE_MAX_ITER: u32 = 400;
const SOLVE_REL_TOL: f64 = 4e-16;
const BRACKET_MAX_DOUBLINGS: u32 = 1_100;

fn ln_beta(a: f64, b: f64) -> f64 {
    libm::lgamma(a) + libm::lgamma(b) - libm::lgamma(a + b)
}

/// One Lentz update: `v <- 1 + coef * v`, kept away from zero.
fn guard(v: f64) -> f64 {
    if v.abs() < TINY {
        TINY
    } else {
        v
    }
}

/// The two terms of one continued-fraction iteration (the even and the odd coefficient).
fn cf_coefs(a: f64, b: f64, x: f64, m: f64) -> (f64, f64) {
    let m2 = 2.0 * m;
    let even = m * (b - m) * x / ((a - 1.0 + m2) * (a + m2));
    let odd = -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1.0 + m2));
    (even, odd)
}

/// Continued fraction of the incomplete beta function (Numerical Recipes `betacf`).
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let mut c = 1.0;
    let mut d = 1.0 / guard(1.0 - (a + b) * x / (a + 1.0));
    let mut h = d;
    for it in 1..=CF_MAX_ITER {
        let (even, odd) = cf_coefs(a, b, x, f64::from(it));
        d = 1.0 / guard(1.0 + even * d);
        c = guard(1.0 + even / c);
        h *= d * c;
        d = 1.0 / guard(1.0 + odd * d);
        c = guard(1.0 + odd / c);
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < CF_EPS {
            break;
        }
    }
    h
}

/// The regularised incomplete beta function `I_x(a, b)` for `x` in `(0, 1)`, given `y = 1 - x`
/// computed by the caller so neither end loses digits.
fn inc_beta(a: f64, b: f64, x: f64, y: f64) -> f64 {
    let front = libm::exp(a * libm::log(x) + b * libm::log(y) - ln_beta(a, b));
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_cf(a, b, x) / a
    } else {
        1.0 - front * beta_cf(b, a, y) / b
    }
}

/// `P(T > t)` for a Student-t variable with `df` degrees of freedom, `t >= 0`.
fn upper_tail(t: f64, df: f64) -> f64 {
    let t2 = t * t;
    let denom = df + t2;
    if t2 == 0.0 {
        return 0.5;
    }
    0.5 * inc_beta(df / 2.0, 0.5, df / denom, t2 / denom)
}

/// The Student-t density at `t`.
fn density(t: f64, df: f64) -> f64 {
    let ln_norm =
        libm::lgamma((df + 1.0) / 2.0) - libm::lgamma(df / 2.0) - 0.5 * libm::log(df * PI);
    libm::exp(ln_norm - (df + 1.0) / 2.0 * libm::log1p(t * t / df))
}

/// An upper bound `hi` with `upper_tail(hi) < q`, found by doubling; `None` if none is found.
fn bracket(q: f64, df: f64) -> Option<f64> {
    let mut hi = 1.0;
    for _ in 0..BRACKET_MAX_DOUBLINGS {
        if upper_tail(hi, df) < q {
            return Some(hi);
        }
        hi *= 2.0;
        if !hi.is_finite() {
            return None;
        }
    }
    None
}

/// One safeguarded Newton step inside `[lo, hi]`: the Newton point when it stays inside, else the
/// midpoint.
fn next_point(t: f64, f: f64, pdf: f64, lo: f64, hi: f64) -> f64 {
    let newton = t + f / pdf;
    if newton.is_finite() && newton > lo && newton < hi {
        newton
    } else {
        0.5 * (lo + hi)
    }
}

/// The upper-tail quantile of Student's t: `t >= 0` with `P(T > t) = q`, for `q` in `(0, 0.5]`
/// and `df > 0`. `NaN` for any other input (R's `qt` also gives `NaN` for `df = 0`).
pub(super) fn qt_upper(q: f64, df: f64) -> f64 {
    if !(q > 0.0 && q <= 0.5 && df > 0.0 && df.is_finite()) {
        return f64::NAN;
    }
    if q == 0.5 {
        return 0.0;
    }
    let Some(mut hi) = bracket(q, df) else { return f64::NAN };
    let mut lo = 0.0;
    let mut t = 0.5 * hi;
    for _ in 0..SOLVE_MAX_ITER {
        let f = upper_tail(t, df) - q;
        if f > 0.0 {
            lo = t;
        } else {
            hi = t;
        }
        let next = next_point(t, f, density(t, df), lo, hi);
        if (next - t).abs() <= SOLVE_REL_TOL * next.abs() {
            return next;
        }
        t = next;
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_median_has_quantile_zero() {
        assert_eq!(qt_upper(0.5, 7.0), 0.0);
    }

    #[test]
    fn inputs_without_a_quantile_are_nan() {
        for (q, df) in [(0.0, 3.0), (0.6, 3.0), (-0.1, 3.0), (0.025, 0.0), (f64::NAN, 3.0)] {
            assert!(qt_upper(q, df).is_nan(), "q {q} df {df}");
        }
        assert!(qt_upper(0.025, f64::INFINITY).is_nan());
    }

    #[test]
    fn one_degree_of_freedom_is_the_cauchy_quantile() {
        let q = 0.025;
        let want = 1.0 / (PI * q).tan();
        assert!((qt_upper(q, 1.0) - want).abs() <= 1e-12 * want);
    }
}
