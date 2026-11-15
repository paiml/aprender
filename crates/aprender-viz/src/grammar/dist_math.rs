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

/// `stat_bin` over finite `x`: ggplot2 4.0.3's `compute_bins()` then `bin_vector()`.
///
/// # Errors
///
/// [`Error::Rendering`] for no finite `x`, `bins == 0`, a non-positive or non-finite binwidth or
/// boundary/center, both `boundary` and `center`, or more than 1,000,000 bins.
pub(super) fn stat_bin(x: &[f64], a: &BinArgs) -> Result<BuiltBins> {
    check_args(x, a)?;
    let lo = x.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let breaks = match a.binwidth {
        Some(w) => breaks_width((lo, hi), w, a.center, a.boundary)?,
        None => breaks_bins((lo, hi), a.bins, a.center, a.boundary)?,
    };
    let count = count_bins(x, &breaks);
    let width: Vec<f64> = breaks.windows(2).map(|w| w[1] - w[0]).collect();
    let centre: Vec<f64> = breaks.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();
    let total: f64 = count.iter().sum();
    Ok(BuiltBins {
        xmin: centre.iter().zip(&width).map(|(c, w)| c - w / 2.0).collect(),
        xmax: centre.iter().zip(&width).map(|(c, w)| c + w / 2.0).collect(),
        density: count.iter().zip(&width).map(|(n, w)| n / w / total).collect(),
        x: centre,
        width,
        count,
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
