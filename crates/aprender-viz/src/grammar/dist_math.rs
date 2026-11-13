//! The numeric kernels behind the distribution rows (PLOT-GG-001, paiml/aprender#6909):
//! `stat_bin`, `stat_density`, `stat_boxplot` and `position_dodge2`, ported from ggplot2 4.0.3
//! and R 4.6 (`stats::density`, `stats::quantile(type = 7)`, `seq.default`, `cut`, `%%`).
//!
//! Every function names the R or ggplot2 routine it follows. Nothing here panics on any finite
//! input: loops are bounded by [`MAX_BINS`] and [`MAX_OUTPUT_ROWS`], and slices are read with
//! `get`/iterators.

use std::cmp::Ordering;
use std::f64::consts::PI;

use crate::error::{Error, Result};

/// ggplot2's cap on histogram bins (`bin_breaks_width`: "must be less than 1,000,000").
pub(super) const MAX_BINS: f64 = 1e6;

/// Cap on the rows one layer may build (bins or grid points times groups). ggplot2 has no such
/// cap and would allocate; a layout that large is refused rather than attempted.
pub(super) const MAX_OUTPUT_ROWS: usize = 5_000_000;

/// Largest `n` for `stat_density` (a grid of 2^20 points; the FFT then runs on 2^21).
pub(super) const MAX_DENSITY_N: usize = 1 << 20;

/// `.Machine$double.eps`.
const EPS: f64 = f64::EPSILON;

fn refuse(msg: impl Into<String>) -> Error {
    Error::Rendering(msg.into())
}

/// Which side of a bin is closed (`closed = c("right", "left")`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Closed {
    /// Bins are `(a, b]`; ggplot2's default.
    #[default]
    Right,
    /// Bins are `[a, b)`.
    Left,
}

/// R's `x %% y` for doubles (`myfmod` in arithmetic.c).
pub(super) fn r_mod(x: f64, y: f64) -> f64 {
    if y == 0.0 {
        return f64::NAN;
    }
    let q = x / y;
    let tmp = x - q.floor() * y;
    let q2 = (tmp / y).floor();
    tmp - q2 * y
}

/// `scales::zero_range(c(lo, hi))` with its default tolerance `1000 * .Machine$double.eps`.
pub(super) fn zero_range(lo: f64, hi: f64) -> bool {
    if lo.is_nan() || hi.is_nan() {
        return false;
    }
    if lo == hi {
        return true;
    }
    if lo.is_infinite() || hi.is_infinite() {
        return false;
    }
    let m = lo.abs().min(hi.abs());
    if m == 0.0 {
        return false;
    }
    ((lo - hi) / m).abs() < 1000.0 * EPS
}

/// `seq.default(from, to, by)` for `by > 0`: `from + (0:n) * by`, capped at `to`.
fn seq_by(from: f64, to: f64, by: f64) -> Result<Vec<f64>> {
    let del = to - from;
    if del == 0.0 && to == 0.0 {
        return Ok(vec![to]);
    }
    let n = del / by;
    if !n.is_finite() || n < 0.0 || n > MAX_BINS * 2.0 {
        return Err(refuse("seq(): the bin sequence is not finite or is too long"));
    }
    let dd = del.abs() / to.abs().max(from.abs());
    if dd < 100.0 * EPS {
        return Ok(vec![from]);
    }
    let count = (n + 1e-10) as usize;
    Ok((0..=count).map(|i| (from + i as f64 * by).min(to)).collect())
}

/// ggplot2's `bins` object: sorted `breaks`, fuzzed `fuzzy` copy, and the closed side.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Bins {
    pub breaks: Vec<f64>,
    fuzzy: Vec<f64>,
    right_closed: bool,
}

fn median_of_sorted(v: &[f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let mid = n / 2;
    let hi = v.get(mid).copied().unwrap_or(f64::NAN);
    if n % 2 == 1 {
        hi
    } else {
        let lo = v.get(mid.wrapping_sub(1)).copied().unwrap_or(f64::NAN);
        (lo + hi) / 2.0
    }
}

/// `bins(breaks, closed)`: sort, then fuzz each break by `1e-8 * median(diff(breaks))` toward the
/// open side, so a value sitting on a break lands in the closed bin despite rounding.
pub(super) fn bin_breaks(mut breaks: Vec<f64>, closed: Closed) -> Result<Bins> {
    if breaks.iter().any(|b| !b.is_finite()) {
        return Err(refuse("bin breaks must all be finite"));
    }
    if breaks.len() as f64 > MAX_BINS + 1.0 {
        return Err(refuse("the number of histogram bins must be less than 1,000,000"));
    }
    breaks.sort_by(f64::total_cmp);
    let mut diffs: Vec<f64> = breaks.windows(2).map(|w| w[1] - w[0]).collect();
    diffs.sort_by(f64::total_cmp);
    let mut fuzz = 1e-8 * median_of_sorted(&diffs);
    if !fuzz.is_finite() {
        fuzz = EPS * 1000.0;
    }
    let last = breaks.len().saturating_sub(1);
    let fuzzy = breaks
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let f = match closed {
                Closed::Right if i == 0 => -fuzz,
                Closed::Right => fuzz,
                Closed::Left if i == last => fuzz,
                Closed::Left => -fuzz,
            };
            b + f
        })
        .collect();
    Ok(Bins { breaks, fuzzy, right_closed: closed == Closed::Right })
}

/// `bin_breaks_width(x_range, width, center, boundary, closed)`.
pub(super) fn bin_breaks_width(
    range: (f64, f64),
    width: f64,
    center: Option<f64>,
    boundary: Option<f64>,
    closed: Closed,
) -> Result<Bins> {
    if !width.is_finite() || width <= 0.0 {
        return Err(refuse("binwidth must be a positive finite number"));
    }
    let boundary = boundary.unwrap_or_else(|| center.map_or(width / 2.0, |c| c - width / 2.0));
    let shift = ((range.0 - boundary) / width).floor();
    let origin = boundary + shift * width;
    let max_x = range.1 + (1.0 - 1e-8) * width;
    let nbins = (max_x - origin) / width;
    if !nbins.is_finite() || nbins > MAX_BINS {
        return Err(refuse(
            "the number of histogram bins must be less than 1,000,000 (is binwidth too small?)",
        ));
    }
    let mut breaks = seq_by(origin, max_x, width)?;
    if breaks.len() == 1 {
        let b = breaks.first().copied().unwrap_or(origin);
        breaks.push(b + width);
    }
    bin_breaks(breaks, closed)
}

/// `bin_breaks_bins(x_range, bins, center, boundary, closed)`.
pub(super) fn bin_breaks_bins(
    range: (f64, f64),
    bins: usize,
    center: Option<f64>,
    boundary: Option<f64>,
    closed: Closed,
) -> Result<Bins> {
    if bins == 0 {
        return Err(refuse("bins must be at least 1"));
    }
    if bins as f64 > MAX_BINS {
        return Err(refuse("the number of histogram bins must be less than 1,000,000"));
    }
    let nb = bins as f64;
    let (mut center, mut boundary) = (center, boundary);
    let width;
    if zero_range(range.0, range.1) {
        width = 0.1;
    } else if bins == 1 {
        width = range.1 - range.0;
        boundary = Some(range.0);
        center = None;
    } else {
        let mut w = (range.1 - range.0) / (nb - 1.0);
        if center.is_none() {
            boundary = boundary.or(Some(range.0 - w / 2.0));
        }
        if let Some(b) = boundary {
            let bm = r_mod(b, w);
            if r_mod(range.0, w) == bm || r_mod(range.1, w) == bm {
                w = (range.1 - range.0) / nb;
            }
        }
        width = w;
    }
    bin_breaks_width(range, width, center, boundary, closed)
}

/// `cut(x, fuzzy, right, include.lowest = TRUE)` as a 0-based bin index (`.bincode`), or `None`
/// for a value outside every bin or `NaN`.
fn bin_cut(x: f64, bins: &Bins) -> Option<usize> {
    let b = &bins.fuzzy;
    let nb1 = b.len().checked_sub(1)?;
    let first = b.first().copied()?;
    let last = b.get(nb1).copied()?;
    if x.is_nan() || x < first || last < x {
        return None;
    }
    let (mut lo, mut hi) = (0usize, nb1);
    while hi - lo >= 2 {
        let mid = (lo + hi) / 2;
        let m = b.get(mid).copied()?;
        if x > m || (!bins.right_closed && x == m) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(lo)
}

/// One group's `bin_vector()` result: ggplot2's `bin_out()` columns.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct BinOut {
    pub count: Vec<f64>,
    pub x: Vec<f64>,
    pub xmin: Vec<f64>,
    pub xmax: Vec<f64>,
    pub width: Vec<f64>,
    pub density: Vec<f64>,
    pub ncount: Vec<f64>,
    pub ndensity: Vec<f64>,
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(f64::NEG_INFINITY, |m, x| m.max(x.abs()))
}

/// `bin_vector(x, bins, pad)` followed by `bin_out()`. A bin with no value counts 0.
pub(super) fn bin_vector(x: &[f64], bins: &Bins, pad: bool) -> BinOut {
    let nbin = bins.breaks.len().saturating_sub(1);
    let mut count = vec![0.0_f64; nbin];
    for v in x {
        if let Some(slot) = bin_cut(*v, bins).and_then(|i| count.get_mut(i)) {
            *slot += 1.0;
        }
    }
    let mut centres: Vec<f64> = bins.breaks.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();
    let mut widths: Vec<f64> = bins.breaks.windows(2).map(|w| w[1] - w[0]).collect();
    if pad {
        let w1 = widths.first().copied().unwrap_or(f64::NAN);
        let wn = widths.last().copied().unwrap_or(f64::NAN);
        let c1 = centres.first().copied().unwrap_or(f64::NAN);
        let cn = centres.last().copied().unwrap_or(f64::NAN);
        count.insert(0, 0.0);
        count.push(0.0);
        widths.insert(0, w1);
        widths.push(wn);
        centres.insert(0, c1 - w1);
        centres.push(cn + wn);
    }
    let total: f64 = count.iter().map(|c| c.abs()).sum();
    let density: Vec<f64> = count.iter().zip(&widths).map(|(c, w)| c / w / total).collect();
    let cmax = max_abs(&count);
    let dmax = max_abs(&density);
    BinOut {
        ncount: count.iter().map(|c| c / cmax).collect(),
        ndensity: density.iter().map(|d| d / dmax).collect(),
        xmin: centres.iter().zip(&widths).map(|(x, w)| x - w / 2.0).collect(),
        xmax: centres.iter().zip(&widths).map(|(x, w)| x + w / 2.0).collect(),
        count,
        x: centres,
        width: widths,
        density,
    }
}

/// `stats::quantile(sorted, p, type = 7)`. `sorted` must be non-empty and ascending.
pub(super) fn quantile7(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let index = 1.0 + (n as f64 - 1.0) * p;
    let lo = index.floor();
    let hi = index.ceil();
    let at = |k: f64| sorted.get((k as usize).saturating_sub(1)).copied().unwrap_or(f64::NAN);
    let (qlo, qhi) = (at(lo), at(hi));
    if index > lo && qhi != qlo {
        let h = index - lo;
        (1.0 - h) * qlo + h * qhi
    } else {
        qlo
    }
}

/// `sd(x)` (n - 1 denominator).
fn sd(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    if x.len() < 2 {
        return f64::NAN;
    }
    let mean = x.iter().sum::<f64>() / n;
    // R's mean() does a second pass to correct rounding.
    let mean = mean + x.iter().map(|v| v - mean).sum::<f64>() / n;
    (x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0)).sqrt()
}

/// `stats::bw.nrd0(x)`.
pub(super) fn bw_nrd0(x: &[f64]) -> f64 {
    let mut sorted = x.to_vec();
    sorted.sort_by(f64::total_cmp);
    let hi = sd(x);
    let iqr = quantile7(&sorted, 0.75) - quantile7(&sorted, 0.25);
    let mut lo = hi.min(iqr / 1.34);
    if lo == 0.0 || lo.is_nan() {
        lo = hi;
        if lo == 0.0 || lo.is_nan() {
            lo = match x.first().map(|v| v.abs()) {
                Some(v) if v != 0.0 => v,
                _ => 1.0,
            };
        }
    }
    0.9 * lo * libm::pow(x.len() as f64, -0.2)
}

/// `dnorm(x, mean = 0, sd = sigma)` as R's `dnorm4` computes it.
fn dnorm(x: f64, sigma: f64) -> f64 {
    const M_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    let x = (x / sigma).abs();
    if x >= 2.0 * f64::MAX.sqrt() {
        return 0.0;
    }
    if x < 5.0 {
        return M_1_SQRT_2PI * libm::exp(-0.5 * x * x) / sigma;
    }
    if x > 38.567_2 {
        return 0.0;
    }
    let x1 = libm::rint(x * 65536.0) / 65536.0;
    let x2 = x - x1;
    M_1_SQRT_2PI / sigma * (libm::exp(-0.5 * x1 * x1) * libm::exp((-0.5 * x2 - x1) * x2))
}

/// `seq.int(from, to, length.out = n)`: `from + i * by`, with both ends exact.
fn seq_len(from: f64, to: f64, n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![from],
        _ => {
            let by = (to - from) / (n as f64 - 1.0);
            (0..n)
                .map(|i| {
                    if i == 0 {
                        from
                    } else if i == n - 1 {
                        to
                    } else {
                        from + i as f64 * by
                    }
                })
                .collect()
        }
    }
}

/// In-place radix-2 FFT; `re.len()` must be a power of two. `inverse` flips the sign of the
/// exponent and leaves the result unscaled, as R's `fft(inverse = TRUE)` does.
fn fft(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    if n < 2 || im.len() != n || !n.is_power_of_two() {
        return;
    }
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let ang = sign * 2.0 * PI / len as f64;
        let half = len / 2;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let (s, c) = (ang * k as f64).sin_cos();
                let (a, b) = (start + k, start + k + half);
                let (tr, ti) = (re[b] * c - im[b] * s, re[b] * s + im[b] * c);
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
}

/// `C_BinDist`: linear binning of weighted `x` onto `2 * n` cells over `[lo, hi]`.
fn bin_dist(x: &[f64], w: f64, lo: f64, hi: f64, n: usize) -> Vec<f64> {
    let mut y = vec![0.0_f64; 2 * n];
    let ixmax = n as i64 - 2;
    let xdelta = (hi - lo) / (n as f64 - 1.0);
    for v in x.iter().filter(|v| v.is_finite()) {
        let xpos = (v - lo) / xdelta;
        let ix = xpos.floor();
        let fx = xpos - ix;
        let ix = ix as i64;
        if (0..=ixmax).contains(&ix) {
            let i = ix as usize;
            if let Some(c) = y.get_mut(i) {
                *c += w * (1.0 - fx);
            }
            if let Some(c) = y.get_mut(i + 1) {
                *c += w * fx;
            }
        } else if ix == -1 {
            if let Some(c) = y.first_mut() {
                *c += w * fx;
            }
        } else if ix == ixmax + 1 {
            if let Some(c) = usize::try_from(ix).ok().and_then(|i| y.get_mut(i)) {
                *c += w * (1.0 - fx);
            }
        }
    }
    y
}

/// `approx(xords, kords, x)$y` (linear, `rule = 1`) for an ascending `xords`.
fn approx(xords: &[f64], yords: &[f64], xs: &[f64]) -> Vec<f64> {
    xs.iter()
        .map(|&v| {
            let n = xords.len();
            if n == 0 || v.is_nan() || v < xords[0] || v > xords[n - 1] {
                return f64::NAN;
            }
            let (mut i, mut j) = (0usize, n - 1);
            while i + 1 < j {
                let ij = (i + j) / 2;
                if v < xords[ij] {
                    j = ij;
                } else {
                    i = ij;
                }
            }
            if v == xords[j] {
                return yords[j];
            }
            if v == xords[i] {
                return yords[i];
            }
            yords[i] + (yords[j] - yords[i]) * ((v - xords[i]) / (xords[j] - xords[i]))
        })
        .collect()
}

/// `stats::density(x, weights = rep(1/nx, nx), bw, kernel = "gaussian", n, from, to)$y`, with the
/// R >= 4.4 coordinates (`old.coords = FALSE`) and `ext = 4`. `bw` is the final bandwidth
/// (already multiplied by `adjust`). Returns `(grid, density)`.
pub(super) fn density(
    x: &[f64],
    bw: f64,
    from: f64,
    to: f64,
    n_user: usize,
) -> (Vec<f64>, Vec<f64>) {
    let nx = x.len();
    let n = {
        let n = n_user.max(512);
        if n > 512 {
            n.next_power_of_two()
        } else {
            n
        }
    };
    let lo = from - 4.0 * bw;
    let up = to + 4.0 * bw;
    let y = bin_dist(x, 1.0 / nx as f64, lo, up, n);
    let len = 2 * n;
    let total = (2.0 * n as f64 - 1.0) / (n as f64 - 1.0) * (up - lo);
    let mut kords = seq_len(0.0, total, len);
    for i in (n + 1)..len {
        let mirrored = kords.get(len - i).copied().unwrap_or(0.0);
        if let Some(k) = kords.get_mut(i) {
            *k = -mirrored;
        }
    }
    let mut kre: Vec<f64> = kords.iter().map(|k| dnorm(*k, bw)).collect();
    let mut kim = vec![0.0_f64; len];
    let (mut yre, mut yim) = (y, vec![0.0_f64; len]);
    fft(&mut yre, &mut yim, false);
    fft(&mut kre, &mut kim, false);
    // fft(y) * Conj(fft(kords))
    let mut pre: Vec<f64> = yre
        .iter()
        .zip(&yim)
        .zip(kre.iter().zip(&kim))
        .map(|((a, b), (c, d))| a * c + b * d)
        .collect();
    let mut pim: Vec<f64> = yre
        .iter()
        .zip(&yim)
        .zip(kre.iter().zip(&kim))
        .map(|((a, b), (c, d))| b * c - a * d)
        .collect();
    fft(&mut pre, &mut pim, true);
    let dens: Vec<f64> = pre.iter().take(n).map(|v| (v / len as f64).max(0.0)).collect();
    let xords = seq_len(lo, up, n);
    let grid = seq_len(from, to, n_user);
    let ys = approx(&xords, &dens, &grid);
    (grid, ys)
}

/// Ascending with `NaN` last (R's `order()`).
pub(super) fn cmp_na_last(a: f64, b: f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}

/// `resolution(x, zero, discrete = FALSE)` for a numeric, non-integer vector.
pub(super) fn resolution(x: &[f64], zero: bool) -> f64 {
    let finite: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let lo = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if finite.is_empty() || zero_range(lo, hi) {
        return 1.0;
    }
    let mut u = finite;
    if zero {
        u.push(0.0);
    }
    u.sort_by(f64::total_cmp);
    u.dedup();
    let tol = EPS.sqrt();
    let min_gap =
        u.windows(2).map(|w| w[1] - w[0]).filter(|d| *d > tol).fold(f64::INFINITY, f64::min);
    if min_gap.is_finite() {
        min_gap
    } else {
        1.0
    }
}

/// One box after `GeomBoxplot$setup_data`: the columns `position_dodge2` reads and writes.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Box2 {
    pub idx: usize,
    pub x: f64,
    pub xmin: f64,
    pub xmax: f64,
    pub group: i64,
    pub order: f64,
    pub xid: usize,
    pub newx: f64,
    pub new_width: f64,
}

/// `find_x_overlaps()` on rows already ordered by `(x, group)`: a 1-based cluster id per row.
fn find_x_overlaps(boxes: &[Box2]) -> Vec<usize> {
    let mut ids = Vec::with_capacity(boxes.len());
    let mut cluster = 1usize;
    let mut prev_max_end = f64::NEG_INFINITY;
    for (i, b) in boxes.iter().enumerate() {
        if i > 0 && (b.xmin > prev_max_end || (b.xmin == prev_max_end && b.xmax != b.xmin)) {
            cluster += 1;
        }
        prev_max_end = prev_max_end.max(b.xmax);
        ids.push(cluster);
    }
    ids
}

/// `position_dodge2(preserve = "total", padding = 0.1)` on one panel's boxes, as `collide2()` +
/// `pos_dodge2()` run it. Returns the boxes in `(x, group)` order and whether any cluster held
/// more than one box (ggplot2 pads only then, and drops its scratch columns only then).
///
/// # Errors
///
/// Refuses boxes whose centres would not follow cluster order (ggplot2 indexes `starts[i]` by
/// sorted centre and would misplace them).
pub(super) fn dodge2(mut boxes: Vec<Box2>, padding: f64) -> Result<(Vec<Box2>, bool)> {
    boxes.sort_by(|a, b| cmp_na_last(a.x, b.x).then(a.group.cmp(&b.group)));
    let mut rank = 0.0_f64;
    let mut prev_x = f64::NAN;
    for b in &mut boxes {
        rank = if b.x == prev_x { rank + 1.0 } else { 1.0 };
        prev_x = b.x;
        b.order = rank;
    }
    let ids = find_x_overlaps(&boxes);
    let nclusters = ids.iter().copied().max().unwrap_or(0);
    let mut lo = vec![f64::INFINITY; nclusters];
    let mut hi = vec![f64::NEG_INFINITY; nclusters];
    let mut size = vec![0usize; nclusters];
    for (b, id) in boxes.iter().zip(&ids) {
        let k = id - 1;
        if let (Some(l), Some(h), Some(s)) = (lo.get_mut(k), hi.get_mut(k), size.get_mut(k)) {
            *l = l.min(b.xmin);
            *h = h.max(b.xmax);
            *s += 1;
        }
    }
    let centres: Vec<f64> = lo.iter().zip(&hi).map(|(l, h)| (l + h) / 2.0).collect();
    for (b, id) in boxes.iter_mut().zip(&ids) {
        let k = id - 1;
        b.xid = *id;
        b.newx = centres.get(k).copied().unwrap_or(f64::NAN);
        b.new_width = (b.xmax - b.xmin) / size.get(k).copied().unwrap_or(1) as f64;
    }
    // aggregate(new_width ~ newx, sum): one start per distinct centre, ascending.
    let mut distinct = centres.clone();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup();
    if distinct.len() != nclusters || distinct.iter().zip(&centres).any(|(d, c)| d != c) {
        return Err(refuse(
            "geom_boxplot: boxes whose centres coincide or are out of cluster order are not supported",
        ));
    }
    // Each cluster's total width in one pass over the boxes (a scan per cluster was quadratic).
    let mut total = vec![0.0_f64; nclusters];
    for (b, id) in boxes.iter().zip(&ids) {
        if let Some(t) = total.get_mut(id - 1) {
            *t += b.new_width;
        }
    }
    let cursor_of = |(c, t): (&f64, &f64)| c - t / 2.0;
    let mut cursor: Vec<f64> = centres.iter().zip(&total).map(cursor_of).collect();
    for (b, id) in boxes.iter_mut().zip(&ids) {
        let k = id - 1;
        let start = cursor.get(k).copied().unwrap_or(f64::NAN);
        b.xmin = start;
        b.xmax = start + b.new_width;
        if let Some(s) = cursor.get_mut(k) {
            *s = b.xmax;
        }
        b.x = (b.xmin + b.xmax) / 2.0;
    }
    let duplicated = size.iter().any(|s| *s > 1);
    if duplicated {
        for b in &mut boxes {
            let pad = b.new_width * (1.0 - padding);
            b.xmin = b.x - pad / 2.0;
            b.xmax = b.x + pad / 2.0;
        }
    }
    Ok((boxes, duplicated))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r_mod_and_zero_range_edges() {
        assert!(r_mod(1.0, 0.0).is_nan());
        assert!((r_mod(-1.0, 3.0) - 2.0).abs() < 1e-15);
        assert!(!zero_range(f64::NAN, 1.0));
        assert!(zero_range(2.0, 2.0));
        assert!(!zero_range(1.0, f64::INFINITY));
        assert!(!zero_range(0.0, 1.0));
        assert!(zero_range(1.0, 1.0 + 1e-15));
    }

    #[test]
    fn seq_by_edges() {
        assert_eq!(seq_by(0.0, 0.0, 1.0).expect("ok"), vec![0.0]);
        assert!(seq_by(0.0, 1.0, 1e-12).is_err());
        assert!(seq_by(1.0, 0.0, 1.0).is_err());
        assert_eq!(seq_by(1.0, 1.0 + 1e-15, 1.0).expect("ok"), vec![1.0]);
        assert_eq!(seq_by(0.0, 2.0, 1.0).expect("ok"), vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn bin_breaks_edges() {
        assert!(median_of_sorted(&[]).is_nan());
        assert_eq!(median_of_sorted(&[1.0, 3.0]), 2.0);
        assert!(bin_breaks(vec![f64::NAN, 1.0], Closed::Right).is_err());
        let too_many: Vec<f64> = (0..1_000_002).map(f64::from).collect();
        assert!(bin_breaks(too_many, Closed::Right).is_err());
        // One break has no spacing to take a median of: the fuzz falls back to a constant.
        let b = bin_breaks(vec![1.0], Closed::Right).expect("ok");
        assert_eq!(b.breaks, vec![1.0]);
    }

    #[test]
    fn centred_bins_have_no_boundary_to_dodge() {
        let b = bin_breaks_bins((0.0, 10.0), 5, Some(0.0), None, Closed::Right).expect("ok");
        assert!(b.breaks.len() >= 6);
    }

    #[test]
    fn bins_move_off_a_boundary_that_lands_on_the_data() {
        // 11 bins over (0, 10) would give width 1 and a break on both ends: ggplot2 widens to 10/11.
        let b = bin_breaks_bins((0.0, 10.0), 11, None, Some(0.0), Closed::Right).expect("ok");
        assert_eq!(b.breaks.len(), 12);
        assert!((b.breaks[1] - 10.0 / 11.0).abs() < 1e-12, "{:?}", b.breaks);
        assert!(bin_breaks_bins((0.0, 10.0), 0, None, None, Closed::Right).is_err());
        assert!(bin_breaks_bins((0.0, 10.0), 2_000_000, None, None, Closed::Right).is_err());
    }

    #[test]
    fn quantile_sd_and_bandwidth_edges() {
        assert!(quantile7(&[], 0.5).is_nan());
        assert_eq!(quantile7(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.25), 2.0);
        assert!(sd(&[1.0]).is_nan());
        // A single zero has no spread to scale from: the bandwidth falls back to 0.9 * 1.
        assert!((bw_nrd0(&[0.0]) - 0.9).abs() < 1e-15);
        assert!((bw_nrd0(&[4.0]) - 0.9 * 4.0).abs() < 1e-15);
        assert!(bw_nrd0(&[1.0, 2.0, 3.0, 4.0]) > 0.0);
        // A zero IQR falls back to the standard deviation.
        assert!(bw_nrd0(&[1.0, 1.0, 1.0, 1.0, 2.0]) > 0.0);
    }

    #[test]
    fn dnorm_tails() {
        assert_eq!(dnorm(1e200, 1.0), 0.0);
        assert_eq!(dnorm(40.0, 1.0), 0.0);
        let mid = dnorm(6.0, 1.0);
        assert!(mid > 0.0 && mid < 1e-8);
        assert!((dnorm(0.0, 1.0) - 0.398_942_280_401_432_7).abs() < 1e-15);
    }

    #[test]
    fn seq_len_and_fft_edges() {
        assert!(seq_len(0.0, 1.0, 0).is_empty());
        assert_eq!(seq_len(3.0, 9.0, 1), vec![3.0]);
        let (mut re, mut im) = (vec![1.0], vec![0.0]);
        fft(&mut re, &mut im, false);
        assert_eq!(re, vec![1.0]);
        let (mut re, mut im) = (vec![1.0, 2.0, 3.0], vec![0.0; 3]);
        fft(&mut re, &mut im, false);
        assert_eq!(re, vec![1.0, 2.0, 3.0], "not a power of two: left alone");
    }

    #[test]
    fn bin_dist_spills_at_both_ends() {
        let y = bin_dist(&[-0.5], 1.0, 0.0, 10.0, 6);
        assert!((y[0] - 0.75).abs() < 1e-12, "{y:?}");
        let y = bin_dist(&[10.5], 1.0, 0.0, 10.0, 6);
        assert!((y[5] - 0.75).abs() < 1e-12, "{y:?}");
        assert_eq!(bin_dist(&[f64::NAN, 100.0], 1.0, 0.0, 10.0, 6).iter().sum::<f64>(), 0.0);
    }

    #[test]
    fn approx_outside_and_empty() {
        assert!(approx(&[], &[], &[1.0])[0].is_nan());
        assert!(approx(&[0.0, 1.0], &[0.0, 1.0], &[2.0])[0].is_nan());
        assert!(approx(&[0.0, 1.0], &[0.0, 1.0], &[f64::NAN])[0].is_nan());
        assert!((approx(&[0.0, 1.0], &[0.0, 2.0], &[0.25])[0] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn density_on_a_large_grid_integrates_to_one() {
        let x = [0.0, 1.0, 2.0, 3.5];
        let (grid, y) = density(&x, 0.8, 0.0, 3.5, 1000);
        assert_eq!((grid.len(), y.len()), (1000, 1000));
        let step = (grid[999] - grid[0]) / 999.0;
        let area: f64 = y.windows(2).map(|w| (w[0] + w[1]) / 2.0 * step).sum();
        assert!(area > 0.5 && area < 1.0, "area {area}");
    }

    #[test]
    fn na_last_ordering() {
        assert_eq!(cmp_na_last(f64::NAN, f64::NAN), Ordering::Equal);
        assert_eq!(cmp_na_last(f64::NAN, 1.0), Ordering::Greater);
        assert_eq!(cmp_na_last(1.0, f64::NAN), Ordering::Less);
        assert_eq!(cmp_na_last(1.0, 2.0), Ordering::Less);
    }

    #[test]
    fn dodge2_refuses_boxes_with_no_orderable_centre() {
        let b = Box2 {
            idx: 0,
            x: f64::NAN,
            xmin: f64::NAN,
            xmax: f64::NAN,
            group: 1,
            order: 0.0,
            xid: 0,
            newx: 0.0,
            new_width: 0.0,
        };
        assert!(dodge2(vec![b], 0.1).is_err());
    }
}
