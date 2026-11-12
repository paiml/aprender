//! Student's t upper quantile, for `geom_smooth(method = "lm", se = TRUE)`'s confidence band
//! (`qt((1 + level) / 2, df)` in R's `predict.lm`). Pure Rust through `libm`; proven against
//! R's `qt(0.975, df)` by `ggplot2_smooth_oracle.rs::qt_matches_r_table`.

/// `ln B`-free regularised incomplete beta by Lentz's continued fraction (Numerical Recipes
/// `betacf`), converged to machine precision.
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..2000 {
        let m = f64::from(m);
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h
}

/// Regularised incomplete beta `I_x(a, b)` given both `x` and `y = 1 - x` (each computed by the
/// caller without cancellation).
fn beta_inc(a: f64, b: f64, x: f64, y: f64) -> f64 {
    let ln_front = libm::lgamma(a + b) - libm::lgamma(a) - libm::lgamma(b)
        + a * libm::log(x)
        + b * libm::log(y);
    let front = libm::exp(ln_front);
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_cf(a, b, x) / a
    } else {
        1.0 - front * beta_cf(b, a, y) / b
    }
}

/// `P(T > t)` for Student's t with `df` degrees of freedom, `t >= 0`.
fn upper_tail(t: f64, df: f64) -> f64 {
    let t2 = t * t;
    let denom = df + t2;
    0.5 * beta_inc(df / 2.0, 0.5, df / denom, t2 / denom)
}

/// The `t` with `P(T > t) = tail`, for `0 < tail < 0.5` and `df >= 1`; `None` outside that.
/// Bisection to machine precision (the tail is strictly decreasing in `t`).
#[must_use]
pub(super) fn qt_upper(tail: f64, df: f64) -> Option<f64> {
    if tail.is_nan() || tail <= 0.0 || tail >= 0.5 || df.is_nan() || df < 1.0 || df.is_infinite() {
        return None;
    }
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    while upper_tail(hi, df) > tail {
        lo = hi;
        hi *= 2.0;
        if hi > 1e300 {
            return None;
        }
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid == lo || mid == hi {
            break;
        }
        if upper_tail(mid, df) > tail {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn df_one_is_the_cauchy_closed_form() {
        // qt(0.975, 1) = tan(pi * 0.475)
        let want = (std::f64::consts::PI * 0.475).tan();
        let got = qt_upper(0.025, 1.0).expect("a quantile");
        assert!((got - want).abs() <= 1e-12 * want, "{got} vs {want}");
    }

    #[test]
    fn df_two_is_the_closed_form() {
        // qt(p, 2) = (2p - 1) / sqrt(2 p (1 - p)), p = 0.975
        let p = 0.975_f64;
        let want = (2.0 * p - 1.0) / (2.0 * p * (1.0 - p)).sqrt();
        let got = qt_upper(0.025, 2.0).expect("a quantile");
        assert!((got - want).abs() <= 1e-12 * want, "{got} vs {want}");
    }

    #[test]
    fn outside_the_domain_is_none() {
        for (tail, df) in [(0.0, 3.0), (0.5, 3.0), (0.6, 3.0), (0.025, 0.0), (0.025, f64::NAN)] {
            assert!(qt_upper(tail, df).is_none(), "({tail}, {df})");
        }
    }
}
