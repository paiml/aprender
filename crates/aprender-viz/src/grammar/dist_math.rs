//! The numeric kernels behind the distribution rows (PLOT-GG-001, paiml/aprender#6909), ported
//! from ggplot2 4.0.3 and R 4.6. This slice holds R's `order()` comparison, which the group ids
//! use; the stat kernels land with their rows.

use std::cmp::Ordering;

/// Ascending with `NaN` last (R's `order()`).
pub(super) fn cmp_na_last(a: f64, b: f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
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
