//! Extended-Wilkinson tick placement — Talbot, Lin & Hanrahan 2010, §4.
//!
//! Two consumers: `aprender-viz`'s own axes, and `apex` (APEX-001 EV-14 lowers its Plot IR onto
//! this surface). Tracked as paiml/aprender#3233.
//!
//! # Which weights are normative
//!
//! The paper's prose (§3.2) writes the objective as
//! `0.2·simplicity + 0.25·coverage + 0.5·density + 0.05·legibility`. **Both of the first author's
//! own implementations disagree with that sentence**, and agree with each other:
//!
//! | source | simplicity | coverage |
//! |---|---|---|
//! | CRAN `labeling::extended` — `w[1]*s + w[2]*c`, `w = c(0.25, 0.2, 0.5, 0.05)` | 0.25 | 0.2 |
//! | `jtalbot/Labeling` `ExtendedAxisLabeler.cs` — `w[0]*s + w[1]*c`, `w = {0.25, 0.2, 0.5, 0.05}` | 0.25 | 0.2 |
//! | the paper's prose | 0.2 | 0.25 |
//!
//! The implementations are normative here, so [`W_DEFAULT`] is `[0.25, 0.2, 0.5, 0.05]` in the
//! order (simplicity, coverage, density, legibility). The golden fixtures in
//! `fixtures/breaks/manifest.json` were produced by running the CRAN reference, and it is exactly
//! this swap that they detect — a property suite cannot, because swapping the first two weights
//! leaves every property (in range, strictly increasing, count near `m`, step in `Q`) intact and
//! changes only which candidate labeling wins.
//!
//! # Numerics
//!
//! `log10` and `10^z` go through the pure-Rust [`libm`] crate rather than `std`, so the result does
//! not depend on the platform's libm. Everything else is `+ - * /` and comparison.

/// The preference-ordered "nice numbers". Paper §4.1 and both reference implementations.
pub const Q_DEFAULT: [f64; 6] = [1.0, 5.0, 2.0, 2.5, 4.0, 3.0];

/// Objective weights, in the order (simplicity, coverage, density, legibility).
///
/// See the module docs: this follows the reference implementations, not the paper's prose.
pub const W_DEFAULT: [f64; 4] = [0.25, 0.2, 0.5, 0.05];

/// `.Machine$double.eps * 100`, the reference's own tolerance.
const EPS: f64 = f64::EPSILON * 100.0;

fn simplicity(q: f64, qs: &[f64], j: f64, lmin: f64, lmax: f64, lstep: f64) -> f64 {
    let n = qs.len() as f64;
    let i = index_of(q, qs) as f64 + 1.0;
    let rem = lmin.rem_euclid(lstep);
    let v = if (rem < EPS || lstep - rem < EPS) && lmin <= 0.0 && lmax >= 0.0 { 1.0 } else { 0.0 };
    1.0 - (i - 1.0) / (n - 1.0) - j + v
}

fn simplicity_max(q: f64, qs: &[f64], j: f64) -> f64 {
    let n = qs.len() as f64;
    let i = index_of(q, qs) as f64 + 1.0;
    1.0 - (i - 1.0) / (n - 1.0) - j + 1.0
}

/// `x * x`. `f64::powi` is on the crate's `disallowed-methods` list (APEX-001 EV-2a rule 5:
/// multiply directly, so the ban needs no exceptions); `powi(2)` and `x * x` are bit-identical.
fn sq(x: f64) -> f64 {
    x * x
}

fn coverage(dmin: f64, dmax: f64, lmin: f64, lmax: f64) -> f64 {
    let range = dmax - dmin;
    1.0 - 0.5 * (sq(dmax - lmax) + sq(dmin - lmin)) / sq(0.1 * range)
}

fn coverage_max(dmin: f64, dmax: f64, span: f64) -> f64 {
    let range = dmax - dmin;
    if span > range {
        let half = (span - range) / 2.0;
        1.0 - 0.5 * (half * half + half * half) / sq(0.1 * range)
    } else {
        1.0
    }
}

fn density(k: f64, m: f64, dmin: f64, dmax: f64, lmin: f64, lmax: f64) -> f64 {
    let r = (k - 1.0) / (lmax - lmin);
    let rt = (m - 1.0) / (lmax.max(dmax) - dmin.min(lmin));
    2.0 - (r / rt).max(rt / r)
}

fn density_max(k: f64, m: f64) -> f64 {
    if k >= m {
        2.0 - (k - 1.0) / (m - 1.0)
    } else {
        1.0
    }
}

/// The paper's fourth term. The reference implementations stub it at 1: it only becomes
/// interesting once label *formatting* is being optimised too, which is out of scope here.
fn legibility(_lmin: f64, _lmax: f64, _lstep: f64) -> f64 {
    1.0
}

fn index_of(q: f64, qs: &[f64]) -> usize {
    qs.iter().position(|&x| x == q).unwrap_or(0)
}

/// Tick positions for the data range `[dmin, dmax]`, aiming for about `m` labels.
///
/// `q` is the preference-ordered nice-number list ([`Q_DEFAULT`]) and `w` the objective weights in
/// the order (simplicity, coverage, density, legibility) ([`W_DEFAULT`]).
///
/// The extreme labels may fall *inside* the data range — that is the extension over Wilkinson, and
/// it is why `[dmin, dmax]` is not necessarily covered. Use [`extended_loose`] when the labeling
/// must contain the data.
///
/// Degenerate inputs (`dmax - dmin` below the reference's epsilon, or a non-finite range) return
/// `m` evenly spaced values, as the reference does.
///
/// ```
/// use trueno_viz::breaks::{extended, Q_DEFAULT, W_DEFAULT};
/// assert_eq!(extended(0.0, 100.0, 5, &Q_DEFAULT, &W_DEFAULT), vec![0.0, 25.0, 50.0, 75.0, 100.0]);
/// ```
pub fn extended(dmin: f64, dmax: f64, m: usize, q: &[f64], w: &[f64; 4]) -> Vec<f64> {
    search(dmin, dmax, m, q, w, false)
}

/// As [`extended`], but the labeling is required to contain `[dmin, dmax]`.
pub fn extended_loose(dmin: f64, dmax: f64, m: usize, q: &[f64], w: &[f64; 4]) -> Vec<f64> {
    search(dmin, dmax, m, q, w, true)
}

fn degenerate(dmin: f64, dmax: f64, m: usize) -> Vec<f64> {
    if m <= 1 {
        return vec![dmin];
    }
    let gaps = m as f64 - 1.0;
    let mut step = (dmax - dmin) / gaps;
    if !step.is_finite() {
        // `dmax - dmin` overflowed (`c(-.Machine$double.xmax, .Machine$double.xmax)`): R's
        // `seq(length.out =)` divides each end first.
        step = dmax / gaps - dmin / gaps;
    }
    // `seq(from, to, length.out = m)` ends at `to` exactly (even when `step` underflows to 0, a
    // subnormal span); a value that would overflow counting up from `from` is counted down from
    // `to` instead.
    (0..m)
        .map(|i| {
            let up = dmin + i as f64 * step;
            if i + 1 == m {
                dmax
            } else if up.is_finite() {
                up
            } else {
                dmax - (m - 1 - i) as f64 * step
            }
        })
        .collect()
}

/// The best labeling seen so far.
struct Best {
    score: f64,
    lmin: f64,
    lmax: f64,
    lstep: f64,
}

/// Everything the inner loops need that does not change during a search.
struct Ctx<'a> {
    dmin: f64,
    dmax: f64,
    m: f64,
    qs: &'a [f64],
    w: &'a [f64; 4],
    only_loose: bool,
}

impl Best {
    /// Consider one candidate labeling, keeping it if it scores better.
    fn offer(&mut self, c: &Ctx<'_>, q: f64, j: f64, k: f64, lmin: f64, lstep: f64) {
        let lmax = lmin + lstep * (k - 1.0);
        if c.only_loose && !(lmin <= c.dmin && lmax >= c.dmax) {
            return;
        }
        let w = c.w;
        let score = w[0] * simplicity(q, c.qs, j, lmin, lmax, lstep)
            + w[1] * coverage(c.dmin, c.dmax, lmin, lmax)
            + w[2] * density(k, c.m, c.dmin, c.dmax, lmin, lmax)
            + w[3] * legibility(lmin, lmax, lstep);
        if score > self.score {
            *self = Best { score, lmin, lmax, lstep };
        }
    }
}

/// Every offset of a `k`-label sequence with this step. The reference walks the same range.
fn scan_starts(c: &Ctx<'_>, best: &mut Best, q: f64, j: f64, k: f64, step: f64) {
    let min_start = libm::floor(c.dmax / step) * j - (k - 1.0) * j;
    let max_start = libm::ceil(c.dmin / step) * j;
    if min_start > max_start {
        return;
    }
    // R's `min_start:max_start` has floor(max - min + 1 + FLT_EPSILON) elements, min_start + i.
    // Counting them keeps the loop finite: stepping an f64 by 1.0 stalls once |start| >= 2^53,
    // which a narrow range far from zero (a single value near 1e14) reaches.
    let n = libm::floor(max_start - min_start + 1.0 + f64::from(f32::EPSILON)) as i64;
    for i in 0..n {
        best.offer(c, q, j, k, (min_start + i as f64) * (step / j), step);
    }
}

/// Every power of ten for this `(j, q, k)`, stopping when even a perfect coverage score could not
/// beat what we already have.
fn scan_powers(c: &Ctx<'_>, best: &mut Best, q: f64, j: f64, k: f64, sm: f64, dm: f64) {
    let delta = (c.dmax - c.dmin) / (k + 1.0) / j / q;
    let mut z = libm::ceil(libm::log10(delta));
    let w = c.w;
    loop {
        let step = j * q * libm::pow(10.0, z);
        let cm = coverage_max(c.dmin, c.dmax, step * (k - 1.0));
        if w[0] * sm + w[1] * cm + w[2] * dm + w[3] < best.score {
            return;
        }
        scan_starts(c, best, q, j, k, step);
        z += 1.0;
    }
}

/// Every label count for this `(j, q)`, stopping when even a perfect density score could not beat
/// what we already have.
fn scan_counts(c: &Ctx<'_>, best: &mut Best, q: f64, j: f64, sm: f64) {
    let mut k = 2.0_f64;
    let w = c.w;
    loop {
        let dm = density_max(k, c.m);
        if w[0] * sm + w[1] + w[2] * dm + w[3] < best.score {
            return;
        }
        scan_powers(c, best, q, j, k, sm, dm);
        k += 1.0;
    }
}

/// R's `seq(from, to, by)` for `by > 0` (`seq.default`): a span below 100 machine epsilons of the
/// larger end is `from` alone; otherwise `from + (0:n) * by` with `n = floor((to - from) / by + 1e-10)`,
/// each value capped at `to`.
fn seq_by(from: f64, to: f64, by: f64) -> Vec<f64> {
    let del = to - from;
    if del.abs() / from.abs().max(to.abs()) < 100.0 * f64::EPSILON {
        return vec![from];
    }
    let n = libm::floor(del / by + 1e-10) as i64;
    (0..=n.max(0)).map(|i| (from + i as f64 * by).min(to)).collect()
}

fn search(dmin: f64, dmax: f64, m: usize, qs: &[f64], w: &[f64; 4], only_loose: bool) -> Vec<f64> {
    let (dmin, dmax) = if dmin > dmax { (dmax, dmin) } else { (dmin, dmax) };
    if m < 2 || qs.is_empty() || !dmin.is_finite() || !dmax.is_finite() || dmax - dmin < EPS {
        return degenerate(dmin, dmax, m.max(1));
    }
    // labeling::extended: a range wider than sqrt(.Machine$double.xmax) is returned as
    // seq(dmin, dmax, length.out = m). Without it the coverage bound squares the span, overflows
    // to inf/inf = NaN, never prunes, and the power loop runs forever.
    if dmax - dmin > libm::sqrt(f64::MAX) {
        return degenerate(dmin, dmax, m);
    }

    let c = Ctx { dmin, dmax, m: m as f64, qs, w, only_loose };
    let mut best = Best { score: -2.0, lmin: dmin, lmax: dmax, lstep: 0.0 };

    let mut j = 1.0_f64;
    'outer: loop {
        for &q in qs {
            let sm = simplicity_max(q, qs, j);
            // Even a perfect score at this skip level cannot beat the incumbent, and simplicity
            // only falls as j grows: nothing further can win.
            if w[0] * sm + w[1] + w[2] + w[3] < best.score {
                break 'outer;
            }
            scan_counts(&c, &mut best, q, j, sm);
        }
        j += 1.0;
    }

    if best.lstep > 0.0 {
        seq_by(best.lmin, best.lmax, best.lstep)
    } else {
        degenerate(dmin, dmax, m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `labeling::extended(0.89e200, 1.12e200, 5)` and `(-1e300, 1e300, 5)` in R: a range wider than
    /// `sqrt(.Machine$double.xmax)` is `seq(dmin, dmax, length.out = m)`. Before the guard the
    /// search never returned on these ranges.
    #[test]
    fn a_range_wider_than_sqrt_max_is_evenly_spaced_like_r() {
        let got = extended(0.89e200, 1.12e200, 5, &Q_DEFAULT, &W_DEFAULT);
        let want = [8.9e199, 9.475e199, 1.005e200, 1.0625e200, 1.12e200];
        assert_eq!(got.len(), want.len(), "{got:?}");
        assert!(got.iter().zip(want).all(|(g, w)| (g - w).abs() <= 1e-12 * w.abs()), "{got:?}");
        assert_eq!(
            extended(-1e300, 1e300, 5, &Q_DEFAULT, &W_DEFAULT),
            [-1e300, -5e299, 0.0, 5e299, 1e300]
        );
    }

    /// `labeling::extended(v - 0.05, v + 0.05, 5)` with `v = 131815660000000` returns the one break
    /// `v - 0.05` in R: the start offsets pass 2^53 (stepping them by 1.0 never ended), and R's
    /// `seq(lmin, lmax, by = 0.025)` returns `lmin` alone because the span is under 100 epsilons
    /// of `lmax`.
    #[test]
    fn a_narrow_range_far_from_zero_ends_like_r() {
        let v = 131_815_660_000_000.0_f64;
        assert_eq!(extended(v - 0.05, v + 0.05, 5, &Q_DEFAULT, &W_DEFAULT), [v - 0.05]);
    }
}

#[cfg(test)]
mod degenerate_tests {
    use super::*;

    /// `seq(-xmax, xmax, length.out = 5)` in R: the span overflows, so each end is divided first.
    #[test]
    fn a_span_that_overflows_keeps_finite_steps() {
        let m = f64::MAX;
        let got = degenerate(-m, m, 5);
        assert_eq!(got.len(), 5);
        assert!(got.iter().all(|v| v.is_finite()), "{got:?}");
        assert_eq!((got[0], got[2], got[4]), (-m, 0.0, m));
        assert!((got[1] + m / 2.0).abs() <= m * 1e-15, "{got:?}");
    }

    /// `seq(0, 5e-324, length.out = 5)`: the step underflows to 0, the last value is still `to`.
    #[test]
    fn the_last_value_is_the_upper_end_even_when_the_step_underflows() {
        assert_eq!(degenerate(0.0, 5e-324, 5), [0.0, 0.0, 0.0, 0.0, 5e-324]);
        assert_eq!(degenerate(1.0, 2.0, 1), [1.0]);
    }
}
