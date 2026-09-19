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
    let step = (dmax - dmin) / (m as f64 - 1.0);
    (0..m).map(|i| dmin + i as f64 * step).collect()
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
    let mut start = min_start;
    while start <= max_start {
        best.offer(c, q, j, k, start * (step / j), step);
        start += 1.0;
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

fn search(dmin: f64, dmax: f64, m: usize, qs: &[f64], w: &[f64; 4], only_loose: bool) -> Vec<f64> {
    let (dmin, dmax) = if dmin > dmax { (dmax, dmin) } else { (dmin, dmax) };
    if m < 2 || qs.is_empty() || !dmin.is_finite() || !dmax.is_finite() || dmax - dmin < EPS {
        return degenerate(dmin, dmax, m.max(1));
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
        let n = ((best.lmax - best.lmin) / best.lstep).round() as i64 + 1;
        (0..n.max(1)).map(|i| best.lmin + i as f64 * best.lstep).collect()
    } else {
        degenerate(dmin, dmax, m)
    }
}
