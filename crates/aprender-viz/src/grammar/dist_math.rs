//! The numeric kernels behind the distribution rows (PLOT-GG-001, paiml/aprender#6909), ported
//! from ggplot2 4.0.3 and R 4.6. This slice holds R's `order()` comparison, which the group ids
//! use; the stat kernels land with their rows.

use std::cmp::Ordering;

use super::build::BuiltBins;
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

/// `stat_bin` arguments (`closed = "right"`, `pad = FALSE`).
#[derive(Debug, Clone, Copy)]
pub(super) struct BinArgs {
    pub bins: usize,
    pub binwidth: Option<f64>,
    pub boundary: Option<f64>,
    pub center: Option<f64>,
}

fn bad(msg: &str) -> Error {
    Error::Rendering(msg.to_string())
}

/// R's `%%`.
fn r_mod(a: f64, b: f64) -> f64 {
    a - (a / b).floor() * b
}

/// `scales::zero_range()` with its default tolerance.
fn zero_range(lo: f64, hi: f64) -> bool {
    if lo == hi {
        return true;
    }
    let m = lo.abs().min(hi.abs());
    m != 0.0 && ((lo - hi) / m).abs() < 1000.0 * f64::EPSILON
}

/// `seq(from, to, by)` for `by > 0`, as `seq.default` computes it.
fn seq_by(from: f64, to: f64, by: f64) -> Vec<f64> {
    let n = ((to - from) / by + 1e-10).floor().max(0.0) as usize;
    (0..=n).map(|i| (from + i as f64 * by).min(to)).collect()
}

/// `bin_breaks_width()`.
fn breaks_width(
    range: (f64, f64),
    width: f64,
    center: Option<f64>,
    boundary: Option<f64>,
) -> Result<Vec<f64>> {
    let boundary = boundary.unwrap_or_else(|| center.map_or(width / 2.0, |c| c - width / 2.0));
    let origin = boundary + ((range.0 - boundary) / width).floor() * width;
    let max_x = range.1 + (1.0 - 1e-8) * width;
    if (max_x - origin) / width > 1e6 {
        return Err(bad("more than 1,000,000 histogram bins: binwidth is too small"));
    }
    let breaks = seq_by(origin, max_x, width);
    Ok(if breaks.len() == 1 { vec![breaks[0], breaks[0] + width] } else { breaks })
}

/// `bin_breaks_bins()`'s `(width, boundary, center)` before `bin_breaks_width()`.
fn bins_params(
    range: (f64, f64),
    bins: usize,
    center: Option<f64>,
    boundary: Option<f64>,
) -> (f64, Option<f64>, Option<f64>) {
    if zero_range(range.0, range.1) {
        return (0.1, boundary, center);
    }
    if bins == 1 {
        return (range.1 - range.0, Some(range.0), None);
    }
    let mut width = (range.1 - range.0) / (bins - 1) as f64;
    let boundary =
        if center.is_none() { Some(boundary.unwrap_or(range.0 - width / 2.0)) } else { boundary };
    let on_edge = |b: f64| [range.0, range.1].iter().any(|x| r_mod(*x, width) == r_mod(b, width));
    if boundary.is_some_and(on_edge) {
        width = (range.1 - range.0) / bins as f64;
    }
    (width, boundary, center)
}

/// `bin_breaks_bins()`.
fn breaks_bins(
    range: (f64, f64),
    bins: usize,
    center: Option<f64>,
    boundary: Option<f64>,
) -> Result<Vec<f64>> {
    let (width, boundary, center) = bins_params(range, bins, center, boundary);
    breaks_width(range, width, center, boundary)
}

fn median_diff(breaks: &[f64]) -> f64 {
    let mut d: Vec<f64> = breaks.windows(2).map(|w| w[1] - w[0]).collect();
    d.sort_by(|a, b| cmp_na_last(*a, *b));
    let n = d.len();
    if n % 2 == 1 {
        d[n / 2]
    } else {
        (d[n / 2 - 1] + d[n / 2]) / 2.0
    }
}

/// Refuse input `stat_bin` cannot bin (see [`stat_bin`]).
fn check_args(x: &[f64], a: &BinArgs) -> Result<()> {
    if x.is_empty() || x.iter().any(|v| !v.is_finite()) {
        return Err(bad("stat_bin needs at least one value and only finite x values"));
    }
    if a.boundary.is_some() && a.center.is_some() {
        return Err(bad("only one of boundary and center may be specified"));
    }
    if [a.boundary, a.center].iter().flatten().any(|v| !v.is_finite()) {
        return Err(bad("boundary and center must be finite"));
    }
    match a.binwidth {
        Some(w) if !(w.is_finite() && w > 0.0) => Err(bad("binwidth must be positive and finite")),
        None if a.bins == 0 => Err(bad("bins must be at least 1")),
        _ => Ok(()),
    }
}

/// `bin_vector()`: how many values fall in each `cut(right = TRUE, include.lowest = TRUE)` bin of
/// the fuzzed breaks, (b[i], b[i+1]] with the first bin closed below.
fn count_bins(x: &[f64], breaks: &[f64]) -> Vec<f64> {
    let fuzz = 1e-8 * median_diff(breaks);
    let fuzz = if fuzz.is_finite() { fuzz } else { f64::EPSILON * 1000.0 };
    let fuzzy: Vec<f64> =
        breaks.iter().enumerate().map(|(i, b)| if i == 0 { b - fuzz } else { b + fuzz }).collect();
    let nbins = breaks.len() - 1;
    let mut count = vec![0.0; nbins];
    for v in x {
        let idx = fuzzy.partition_point(|b| b < v);
        let bin =
            if idx == 0 { (*v == fuzzy[0]).then_some(0) } else { (idx <= nbins).then(|| idx - 1) };
        if let Some(i) = bin {
            count[i] += 1.0;
        }
    }
    count
}

/// ggplot2 4.0.3's `compute_bins()`: the bin edges, from the range of all the layer's `x` (the
/// scale's dimension), so every group is cut at the same edges.
fn bin_breaks(x: &[f64], a: &BinArgs) -> Result<Vec<f64>> {
    check_args(x, a)?;
    let lo = x.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    match a.binwidth {
        Some(w) => breaks_width((lo, hi), w, a.center, a.boundary),
        None => breaks_bins((lo, hi), a.bins, a.center, a.boundary),
    }
}

/// `bin_out()`: one group's bins from its counts over `breaks`.
fn bins_from_counts(breaks: &[f64], count: Vec<f64>) -> BuiltBins {
    let width: Vec<f64> = breaks.windows(2).map(|w| w[1] - w[0]).collect();
    let centre: Vec<f64> = breaks.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();
    let total: f64 = count.iter().sum();
    BuiltBins {
        xmin: centre.iter().zip(&width).map(|(c, w)| c - w / 2.0).collect(),
        xmax: centre.iter().zip(&width).map(|(c, w)| c + w / 2.0).collect(),
        density: count.iter().zip(&width).map(|(n, w)| n / w / total).collect(),
        ymin: vec![0.0; count.len()],
        ymax: count.clone(),
        group: vec![-1; count.len()],
        x: centre,
        width,
        count,
    }
}

/// `stat_bin` over finite `x`: ggplot2 4.0.3's `compute_bins()` then `bin_vector()`. The layout
/// path goes through [`stat_bin_groups`]; this one-group form is what the kernel tests drive.
///
/// # Errors
///
/// [`Error::Rendering`] for no finite `x`, `bins == 0`, a non-positive or non-finite binwidth or
/// boundary/center, both `boundary` and `center`, or more than 1,000,000 bins.
#[cfg(test)]
pub(super) fn stat_bin(x: &[f64], a: &BinArgs) -> Result<BuiltBins> {
    let breaks = bin_breaks(x, a)?;
    Ok(bins_from_counts(&breaks, count_bins(x, &breaks)))
}

/// `stat_bin` per group: one set of edges from all of `x`, then each group's counts over them
/// (empty bins stay, `drop = "none"`). Groups come out in ascending id, each with its `group` id.
///
/// # Errors
///
/// As [`stat_bin`], judged on all of `x`.
pub(super) fn stat_bin_groups(x: &[f64], group: &[i64], a: &BinArgs) -> Result<Vec<BuiltBins>> {
    let breaks = bin_breaks(x, a)?;
    let mut ids: Vec<i64> = group.to_vec();
    ids.sort_unstable();
    ids.dedup();
    Ok(ids
        .into_iter()
        .map(|id| {
            let own: Vec<f64> =
                x.iter().zip(group).filter(|(_, g)| **g == id).map(|(v, _)| *v).collect();
            let mut bins = bins_from_counts(&breaks, count_bins(&own, &breaks));
            bins.group = vec![id; bins.count.len()];
            bins
        })
        .collect())
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

// ---- stat_bin with pad = TRUE (PLOT-GG-001, paiml/aprender#6963) ------------------------------

/// `bin_vector(pad = TRUE)`: one zero-count bin on each end, as wide as the bin next to it, so a
/// frequency polygon starts and ends on zero. `bin_out()` then rebuilds `xmin`, `xmax` and
/// `density` from the padded `x`, `width` and `count` (the counts' sum is unchanged).
fn pad_bins(b: &BuiltBins) -> Result<BuiltBins> {
    let (Some(w1), Some(wn), Some(x1), Some(xn)) =
        (b.width.first(), b.width.last(), b.x.first(), b.x.last())
    else {
        return Err(bad("stat_bin produced no bins to pad"));
    };
    let pad = |first: f64, mid: &[f64], last: f64| -> Vec<f64> {
        std::iter::once(first).chain(mid.iter().copied()).chain(std::iter::once(last)).collect()
    };
    let width = pad(*w1, &b.width, *wn);
    let x = pad(x1 - w1, &b.x, xn + wn);
    let count = pad(0.0, &b.count, 0.0);
    let total: f64 = count.iter().map(|c| c.abs()).sum();
    Ok(BuiltBins {
        xmin: x.iter().zip(&width).map(|(c, w)| c - w / 2.0).collect(),
        xmax: x.iter().zip(&width).map(|(c, w)| c + w / 2.0).collect(),
        density: count.iter().zip(&width).map(|(n, w)| n / w / total).collect(),
        x,
        width,
        count,
    })
}

/// `stat_bin(pad = TRUE)` over finite `x`, what `geom_freqpoly()` draws: [`stat_bin`] with a
/// zero-count bin added at each end.
///
/// # Errors
///
/// The errors of [`stat_bin`].
pub(super) fn stat_bin_padded(x: &[f64], a: &BinArgs) -> Result<BuiltBins> {
    pad_bins(&stat_bin(x, a)?)
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

    fn args(bins: usize) -> BinArgs {
        BinArgs { bins, binwidth: None, boundary: None, center: None }
    }

    #[test]
    fn bins_are_total_on_bad_input() {
        for xs in [&[][..], &[f64::NAN], &[1.0, f64::INFINITY]] {
            assert!(stat_bin(xs, &args(30)).is_err());
        }
        assert!(stat_bin(&[1.0, 2.0], &args(0)).is_err());
        let both = BinArgs { boundary: Some(0.0), center: Some(0.0), ..args(3) };
        assert!(stat_bin(&[1.0, 2.0], &both).is_err());
        for w in [0.0, -1.0, f64::NAN] {
            assert!(stat_bin(&[1.0, 2.0], &BinArgs { binwidth: Some(w), ..args(3) }).is_err());
        }
        assert!(stat_bin(&[0.0, 1e9], &BinArgs { binwidth: Some(1e-3), ..args(3) }).is_err());
    }

    #[test]
    fn constant_input_uses_width_one_tenth() {
        let b = stat_bin(&[3.0; 4], &args(30)).expect("constant x bins");
        assert_eq!(b.count.iter().sum::<f64>(), 4.0);
        assert!((b.width[0] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn right_closed_edges() {
        // 1, 2, 3 with binwidth 1: breaks 1, 2, 3 (R's origin is the floor of the minimum), so
        // the first bin is [1, 2] (`include.lowest`) and holds 1 and 2.
        let a = BinArgs { binwidth: Some(1.0), boundary: Some(0.0), ..args(30) };
        assert_eq!(stat_bin(&[1.0, 2.0, 3.0], &a).expect("bins").count, [2.0, 1.0]);
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

#[cfg(test)]
mod pad_tests {
    use super::*;

    #[test]
    fn padding_adds_a_zero_bin_each_side_with_the_neighbour_width() {
        let a = BinArgs { bins: 30, binwidth: Some(1.0), boundary: Some(0.0), center: None };
        let b = stat_bin_padded(&[1.0, 2.0, 3.0], &a).expect("bins");
        assert_eq!(b.count, [0.0, 2.0, 1.0, 0.0]);
        assert_eq!(b.x, [0.5, 1.5, 2.5, 3.5]);
        assert_eq!(b.width, [1.0; 4]);
        assert_eq!(b.xmin, [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(b.xmax, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(b.density, [0.0, 2.0 / 3.0, 1.0 / 3.0, 0.0]);
    }

    #[test]
    fn padding_refuses_what_stat_bin_refuses() {
        let a = BinArgs { bins: 30, binwidth: None, boundary: None, center: None };
        assert!(stat_bin_padded(&[], &a).is_err());
        assert!(stat_bin_padded(&[1.0, f64::INFINITY], &a).is_err());
        assert!(pad_bins(&BuiltBins {
            x: vec![],
            xmin: vec![],
            xmax: vec![],
            width: vec![],
            count: vec![],
            density: vec![],
        })
        .is_err());
    }
}
