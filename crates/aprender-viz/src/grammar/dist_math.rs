//! The numeric kernels behind the distribution rows (PLOT-GG-001, paiml/aprender#6909), ported
//! from ggplot2 4.0.3 and R 4.6. This slice holds R's `order()` comparison, which the group ids
//! use; the stat kernels land with their rows.

use std::cmp::Ordering;

use crate::error::{Error, Result};

/// Ascending with `NaN` last (R's `order()`).
pub(super) fn cmp_na_last(a: f64, b: f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}

// ---- stat_density (PLOT-GG-001, paiml/aprender#6964) ----------------------------------------
//
// ggplot2 4.0.3's `compute_density()` calls R 4.6's `stats::density.default()` with
// `bw = "nrd0"`, `adjust = 1`, `kernel = "gaussian"`, `n = 512`, `cut = 3`, `from`/`to` the x
// scale's dimension and every weight `1 / n`. That function does not sum kernels: it bins the data
// linearly onto a `2n` grid over `[from - 4 bw, to + 4 bw]` (`BinDist`), convolves the bins with
// the kernel ordinates by FFT, and interpolates the first `n` results onto the output grid with
// `approx()`. The port below does the same steps; the circular convolution is summed directly,
// which is what the FFT computes up to rounding.

/// `stat_density`'s default number of points.
pub(super) const DENSITY_POINTS: usize = 512;

/// `ext` of `density.default`: the grid extends this many bandwidths past `from` and `to`.
const EXT: f64 = 4.0;

/// The columns `stat_density` adds, one entry per grid point (`x` is returned beside them).
#[derive(Debug, Clone, PartialEq)]
pub(super) struct DensityCols {
    /// The output grid, `from` to `to`.
    pub x: Vec<f64>,
    /// The density estimate.
    pub density: Vec<f64>,
    /// `density / max(density)` (ggplot2's `scaled`, equal to `ndensity`).
    pub scaled: Vec<f64>,
    /// `density * n`.
    pub count: Vec<f64>,
    /// `density * sum(weights)`, which is `n` for unweighted data.
    pub wdensity: Vec<f64>,
    /// Rows in the group.
    pub n: usize,
}

fn density_refusal(msg: &str) -> Error {
    Error::Rendering(format!("stat_density: {msg}"))
}

/// R's `mean()`: the sum over `n`, then one refinement pass.
fn r_mean(xs: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let m = xs.iter().sum::<f64>() / n;
    m + xs.iter().map(|x| x - m).sum::<f64>() / n
}

/// R's `sd()`: the square root of the sample variance (divisor `n - 1`).
fn r_sd(xs: &[f64]) -> f64 {
    let m = r_mean(xs);
    let ss: f64 = xs.iter().map(|x| (x - m) * (x - m)).sum();
    libm::sqrt(ss / (xs.len() as f64 - 1.0))
}

/// R's type 7 `quantile()` of an ascending slice.
fn quantile7(sorted: &[f64], p: f64) -> f64 {
    let index = 1.0 + (sorted.len() as f64 - 1.0) * p;
    let (lo, hi) = (index.floor(), index.ceil());
    let (a, b) = (sorted[lo as usize - 1], sorted[hi as usize - 1]);
    if index > lo && b != a {
        let h = index - lo;
        (1.0 - h) * a + h * b
    } else {
        a
    }
}

/// `stats::bw.nrd0()`: Silverman's rule of thumb, `0.9 * min(sd, IQR / 1.34) * n^-0.2`.
///
/// # Errors
///
/// [`Error::Rendering`] for fewer than two values, a non-finite value, or a constant sample
/// (zero spread: ggplot2 falls back to `|x[1]|`, which this crate refuses instead).
pub(super) fn bw_nrd0(xs: &[f64]) -> Result<f64> {
    if xs.len() < 2 {
        return Err(density_refusal("needs at least two values (ggplot2 drops the group)"));
    }
    if xs.iter().any(|v| !v.is_finite()) {
        return Err(density_refusal("non-finite value"));
    }
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| cmp_na_last(*a, *b));
    let hi = r_sd(xs);
    let iqr = quantile7(&sorted, 0.75) - quantile7(&sorted, 0.25);
    let lo = hi.min(iqr / 1.34);
    if lo <= 0.0 || !lo.is_finite() {
        return Err(density_refusal("constant data has no bandwidth (zero spread)"));
    }
    Ok(0.9 * lo * libm::pow(xs.len() as f64, -0.2))
}

/// R's `BinDist`: linear binning of `xs` (weight `w` each) onto `2n` cells from `lo` to `up`.
fn bin_dist(xs: &[f64], w: f64, lo: f64, up: f64, n: usize) -> Vec<f64> {
    let mut y = vec![0.0; 2 * n];
    let xdelta = (up - lo) / (n as f64 - 1.0);
    let ixmax = n as i64 - 2;
    for &x in xs {
        let pos = (x - lo) / xdelta;
        let ix = pos.floor();
        let fx = pos - ix;
        let ix = ix as i64;
        if (0..=ixmax).contains(&ix) {
            y[ix as usize] += w * (1.0 - fx);
            y[ix as usize + 1] += w * fx;
        } else if ix == -1 {
            y[0] += w * fx;
        } else if ix == ixmax + 1 {
            y[ix as usize] += w * (1.0 - fx);
        }
    }
    y
}

/// R's `dnorm(x, sd = sd)`, branch for branch (`dnorm4` in nmath).
fn dnorm(x: f64, sd: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    let x = (x / sd).abs();
    if x < 5.0 {
        return INV_SQRT_2PI * libm::exp(-0.5 * x * x) / sd;
    }
    // R's cut-off: exp(-x^2 / 2) is below the smallest subnormal past here.
    if x > libm::sqrt(2.0 * std::f64::consts::LN_2 * 1073.0) {
        return 0.0;
    }
    let x1 = (x * 65_536.0).round() / 65_536.0;
    let x2 = x - x1;
    INV_SQRT_2PI / sd * (libm::exp(-0.5 * x1 * x1) * libm::exp((-0.5 * x2 - x1) * x2))
}

/// The `2n` kernel ordinates of `density.default`: `dnorm(k * step)` for `k = 0..n`, mirrored.
fn kernel_ordinates(n: usize, span: f64, bw: f64) -> Vec<f64> {
    let nf = n as f64;
    let total = (2.0 * nf - 1.0) / (nf - 1.0) * span;
    let by = total / (2.0 * nf - 1.0);
    let at = |k: usize| dnorm(k as f64 * by, bw);
    (0..2 * n).map(|m| at(if m <= n { m } else { 2 * n - m })).collect()
}

/// The first `n` entries of the circular convolution of `y` and `k` (both length `2n`), floored at
/// zero as `pmax.int(0, ...)` does.
fn convolve_head(y: &[f64], k: &[f64], n: usize) -> Vec<f64> {
    let len = y.len();
    (0..n)
        .map(|j| {
            let sum: f64 = y.iter().enumerate().map(|(m, v)| v * k[(j + len - m) % len]).sum();
            sum.max(0.0)
        })
        .collect()
}

/// R's `seq.int(from, to, length.out = n)`.
fn seq_len_out(from: f64, to: f64, n: usize) -> Vec<f64> {
    let by = (to - from) / (n as f64 - 1.0);
    let mut v: Vec<f64> = (0..n).map(|i| from + i as f64 * by).collect();
    if let Some(last) = v.last_mut() {
        *last = to;
    }
    v
}

/// R's `approx(xs, ys, v)$y` for ascending `xs` (`approx1` in R's C code).
fn approx1(xs: &[f64], ys: &[f64], v: f64) -> f64 {
    let (mut i, mut j) = (0, xs.len() - 1);
    while i + 1 < j {
        let ij = (i + j) / 2;
        if v < xs[ij] {
            j = ij;
        } else {
            i = ij;
        }
    }
    if v == xs[j] {
        ys[j]
    } else if v == xs[i] {
        ys[i]
    } else {
        ys[i] + (ys[j] - ys[i]) * ((v - xs[i]) / (xs[j] - xs[i]))
    }
}

/// `compute_density(x, w = NULL, from, to, bw = "nrd0", n = 512)` of ggplot2 4.0.3.
///
/// # Errors
///
/// [`Error::Rendering`] for fewer than two values, a non-finite value or `from`/`to`, a constant
/// sample, or `from >= to`.
pub(super) fn stat_density(xs: &[f64], from: f64, to: f64, n: usize) -> Result<DensityCols> {
    let bw = bw_nrd0(xs)?;
    if !(from.is_finite() && to.is_finite() && from < to) || n < 2 {
        return Err(density_refusal("the output range must be finite with from < to"));
    }
    let size = n.max(512);
    let size = if size > 512 { size.next_power_of_two() } else { size };
    let nx = xs.len() as f64;
    let (lo, up) = (from - EXT * bw, to + EXT * bw);
    let y = bin_dist(xs, 1.0 / nx, lo, up, size);
    let conv = convolve_head(&y, &kernel_ordinates(size, up - lo, bw), size);
    let grid = seq_len_out(lo, up, size);
    let x = seq_len_out(from, to, n);
    let density: Vec<f64> = x.iter().map(|v| approx1(&grid, &conv, *v)).collect();
    let top = density.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok(DensityCols {
        scaled: density.iter().map(|d| d / top).collect(),
        count: density.iter().map(|d| d * nx).collect(),
        wdensity: density.iter().map(|d| d * nx).collect(),
        n: xs.len(),
        x,
        density,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn na_last_ordering() {
        assert_eq!(cmp_na_last(f64::NAN, f64::NAN), Ordering::Equal);
        assert_eq!(cmp_na_last(f64::NAN, 1.0), Ordering::Greater);
        assert_eq!(cmp_na_last(1.0, f64::NAN), Ordering::Less);
        assert_eq!(cmp_na_last(1.0, 2.0), Ordering::Less);
        assert_eq!(cmp_na_last(2.0, 1.0), Ordering::Greater);
    }
}

#[cfg(test)]
mod density_tests {
    use super::*;

    #[test]
    fn nrd0_of_one_to_ten() {
        // sd = 3.0276503540974917, IQR = 4.5: lo = min(sd, 3.358..) = sd; 0.9 * sd * 10^-0.2.
        let xs: Vec<f64> = (1..=10).map(f64::from).collect();
        let want = 0.9 * 3.027_650_354_097_491_7 * libm::pow(10.0, -0.2);
        assert!((bw_nrd0(&xs).expect("bw") - want).abs() < 1e-12);
    }

    #[test]
    fn refuses_short_constant_and_non_finite_input() {
        assert!(bw_nrd0(&[]).is_err());
        assert!(bw_nrd0(&[1.0]).is_err());
        assert!(bw_nrd0(&[2.0, 2.0, 2.0]).is_err());
        assert!(bw_nrd0(&[1.0, f64::NAN]).is_err());
        assert!(stat_density(&[1.0, 2.0], 2.0, 1.0, 512).is_err());
        assert!(stat_density(&[1.0, 2.0], f64::NAN, 1.0, 512).is_err());
    }

    #[test]
    fn density_integrates_to_about_one() {
        let xs = [1.0, 2.0, 2.5, 4.0, 7.0];
        let d = stat_density(&xs, 1.0, 7.0, 512).expect("density");
        let area: f64 =
            d.x.windows(2)
                .zip(d.density.windows(2))
                .map(|(x, y)| (x[1] - x[0]) * (y[0] + y[1]) / 2.0)
                .sum();
        // from/to are the data range with 3 bw of tail cut off, so slightly under one.
        assert!(area > 0.6 && area < 1.0, "{area}");
        assert_eq!(d.n, 5);
    }

    #[test]
    fn dnorm_branches_agree_with_the_closed_form() {
        for x in [0.0_f64, 1.0, 4.9, 5.0, 7.5, 30.0] {
            let want = libm::exp(-0.5 * x * x) / libm::sqrt(2.0 * std::f64::consts::PI);
            assert!((dnorm(x, 1.0) - want).abs() <= 1e-12 * want.max(1e-300), "{x}");
        }
        assert_eq!(dnorm(40.0, 1.0), 0.0);
    }
}
