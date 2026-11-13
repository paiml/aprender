//! R's `format(x, trim = TRUE)` for a numeric vector at `digits = 7`: the fixed-or-scientific
//! decision of `formatReal`, which ggplot2 uses to print default axis labels.
//!
//! One shared format is chosen for the whole vector. R compares the width of the fixed form
//! (`mxsl + rgt + 1`) with the width of the scientific form and prints fixed when
//! `fixed_width <= sci_width` (`scipen = 0`). So `c(1, 10, 100)` prints `"1" "10" "100"` but
//! `c(100, 10000, 1e6)` prints `"1e+02" "1e+04" "1e+06"`, the form ggplot2 shows on a log axis.
//!
//! Not implemented: `roundingwidens` (a value such as 9.9999999 whose 7-digit rounding gains a
//! digit), `NA`/`Inf` elements, and a non-default `digits` or `scipen`.

/// Significant digits R's `format()` uses by default (`getOption("digits")`).
pub(crate) const R_DIGITS: i32 = 7;

/// Per-element facts R's `scientific()` computes: decimal exponent and significant digits after
/// rounding to [`R_DIGITS`] and dropping trailing zeros.
struct Sci {
    exponent: i32,
    nsig: i32,
}

/// R rounds the scaled mantissa with `nearbyintl`: ties go to the even integer, so 3000000.5 is
/// 3000000 and 3000001.5 is 3000002. (`libm::round` sends every tie away from zero.)
fn round_half_even(x: f64) -> f64 {
    libm::rint(x)
}

fn scientific(v: f64) -> Sci {
    if v == 0.0 || !v.is_finite() {
        return Sci { exponent: 0, nsig: 1 };
    }
    let a = v.abs();
    let mut e = libm::floor(libm::log10(a)) as i32;
    // Mantissa scaled to R_DIGITS integer digits, rounded as R rounds it.
    let k = e - (R_DIGITS - 1);
    // `10^-k` overflows below 1e-308 (subnormals): scale the value up first, exactly as R's long
    // double does implicitly.
    let mut m = if k < -300 {
        round_half_even(a * 1e30 / libm::pow(10.0, f64::from(k + 30)))
    } else {
        round_half_even(a / libm::pow(10.0, f64::from(k)))
    };
    let full = libm::pow(10.0, f64::from(R_DIGITS));
    if m >= full {
        m /= 10.0;
        e += 1;
    }
    let mut sig = R_DIGITS;
    let mut mi = m as u64;
    while sig > 1 && mi % 10 == 0 {
        mi /= 10;
        sig -= 1;
    }
    Sci { exponent: e, nsig: sig }
}

/// Digits needed right of the decimal point to print `v` to 7 significant digits with no trailing
/// zeros (the `rgt` of R's `formatReal`).
pub(crate) fn decimals_needed(v: f64) -> usize {
    let s = scientific(v);
    usize::try_from(s.nsig - s.exponent - 1).unwrap_or(0)
}

/// Label `values` as R's `format(x, trim = TRUE)` does.
#[must_use]
pub fn format_labels(values: &[f64]) -> Vec<String> {
    if values.is_empty() {
        return Vec::new();
    }
    let sci: Vec<Sci> = values.iter().map(|v| scientific(*v)).collect();
    let any_neg = values.iter().any(|v| *v < 0.0);
    let rgt = values.iter().map(|v| decimals_needed(*v)).max().unwrap_or(0);
    // Widest left part: the integer digits (at least 1), plus a sign if any element is negative.
    let mxsl = sci
        .iter()
        .zip(values)
        .map(|(s, v)| {
            let left = if s.exponent >= 0 { s.exponent + 1 } else { 1 };
            i32::from(*v < 0.0) + left
        })
        .max()
        .unwrap_or(1);
    let fixed_width = mxsl + rgt as i32 + i32::from(rgt != 0);

    let mxns = sci.iter().map(|s| s.nsig).max().unwrap_or(1);
    let mxe = sci.iter().map(|s| s.exponent).max().unwrap_or(0);
    let mne = sci.iter().map(|s| s.exponent).min().unwrap_or(0);
    let e_digits = if mxe >= 100 || mne <= -99 { 2 } else { 1 };
    let d = mxns - 1;
    let sci_width = i32::from(any_neg) + i32::from(d > 0) + d + 4 + e_digits;

    if fixed_width <= sci_width {
        values
            .iter()
            .map(|v| {
                // R never prints a negative zero.
                let v = if *v == 0.0 { 0.0 } else { *v };
                format!("{v:.rgt$}")
            })
            .collect()
    } else {
        let d = d as usize;
        values.iter().map(|v| sci_string(*v, d)).collect()
    }
}

/// `v` in R's scientific form with `d` mantissa decimals: `1.5e+03`, exponent at least two digits.
fn sci_string(v: f64, d: usize) -> String {
    let v = if v == 0.0 { 0.0 } else { v };
    let s = format!("{v:.d$e}");
    let (mantissa, exp) = s.split_once('e').unwrap_or((s.as_str(), "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let sign = if exp < 0 { '-' } else { '+' };
    format!("{mantissa}e{sign}{:02}", exp.abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_share_decimals() {
        assert_eq!(format_labels(&[-2.5, 0.0, 2.5, 5.0]), ["-2.5", "0.0", "2.5", "5.0"]);
        assert_eq!(format_labels(&[5.0, 10.0]), ["5", "10"]);
        assert_eq!(format_labels(&[0.25, 0.5]), ["0.25", "0.50"]);
        assert_eq!(format_labels(&[-0.0]), ["0"]);
        assert!(format_labels(&[]).is_empty());
    }

    #[test]
    fn decimals_round_to_seven_significant_digits() {
        // 1/3 prints as 0.3333333 in R: 7 significant digits.
        assert_eq!(decimals_needed(1.0 / 3.0), 7);
        assert_eq!(decimals_needed(1234.5), 1);
        assert_eq!(decimals_needed(9_999_999.6), 0);
    }

    #[test]
    fn scientific_when_narrower_than_fixed() {
        // R: format(c(1, 10, 100)) -> fixed; format(c(100, 1e4, 1e6)) -> scientific.
        assert_eq!(format_labels(&[1.0, 10.0, 100.0, 1000.0]), ["1", "10", "100", "1000"]);
        assert_eq!(format_labels(&[100.0, 1e4, 1e6]), ["1e+02", "1e+04", "1e+06"]);
        assert_eq!(format_labels(&[1e5]), ["1e+05"]);
        assert_eq!(format_labels(&[1.5e5, 2e5]), ["150000", "200000"]);
        assert_eq!(format_labels(&[1e-5]), ["1e-05"]);
    }
}

#[cfg(test)]
mod tie_tests {
    use super::*;

    /// R 4.6.2 `format(x, trim = TRUE)` (LC_COLLATE=C), the 7th significant digit a tie: R rounds
    /// ties to even, so 3000000.5 -> 3000000 (one significant digit) where half-away gives
    /// 3000001 (seven), and the fixed width then wins.
    #[test]
    fn a_tie_at_the_seventh_digit_rounds_to_even_as_r_does() {
        assert_eq!(format_labels(&[3e9, 3_000_000_500.0]), ["3e+09", "3e+09"]);
        let six: Vec<f64> = (0..6).map(|k| 3e9 + f64::from(k) * 100.0).collect();
        assert_eq!(format_labels(&six), ["3e+09"; 6]);
        // Not a tie at the 7th digit: all digits kept.
        assert_eq!(
            format_labels(&[3_000_000_001.0, 3_000_000_257.0, 3_000_000_513.0]),
            ["3000000001", "3000000257", "3000000513"]
        );
        assert_eq!(format_labels(&[3_000_001_500.0, 3e9]), ["3000001500", "3000000000"]);
        assert_eq!(format_labels(&[3_000_002_500.0, 3e9]), ["3000002500", "3000000000"]);
        assert_eq!(format_labels(&[30_000_005.0, 0.0]), ["3e+07", "0e+00"]);
        // Odd neighbour rounds up to the even one: 1234567.5 -> 1234568 (7 digits), tie at .5.
        assert_eq!(format_labels(&[2_500_000.5, 0.0]), ["2500000", "0"]);
        assert_eq!(format_labels(&[1_234_567.5, 0.0]), ["1234568", "0"]);
        assert_eq!(format_labels(&[12_345_675.0, 0.0]), ["12345675", "0"]);
        assert_eq!(format_labels(&[12_345_685.0, 0.0]), ["12345685", "0"]);
    }
}

#[cfg(test)]
mod subnormal_tests {
    use super::*;

    /// `format(c(0, 5e-324))` in R: scientific, seven significant digits, three-digit exponent.
    #[test]
    fn subnormals_print_in_scientific_form() {
        assert_eq!(format_labels(&[0.0, 5e-324]), ["0.000000e+00", "4.940656e-324"]);
        assert_eq!(decimals_needed(5e-324), 330);
        assert_eq!(format_labels(&[1e-310]), ["1e-310"]);
    }
}
