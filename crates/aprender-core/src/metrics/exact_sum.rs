//! Exactly-rounded f64 summation, bit-identical to Python's `math.fsum`.
//!
//! The Laya gate (contracts/laya-finetune-gate-v1.yaml `numeric_agreement`) computes every
//! quantity in f64 with exactly-rounded sums in BOTH languages, so the Rust verifier and the
//! Python trainer return the same bits on every input. Exact rounding is what makes that true
//! by construction: the correctly rounded value of a sum does not depend on the order of the
//! terms or on the summation tree, so it cannot differ between an iterator here and a numpy
//! array there. A plain left-to-right sum (or numpy's pairwise sum) is order-dependent and
//! is not a valid substitute.
//!
//! [`fsum`] is a safe port of CPython's `math_fsum` (Modules/mathmodule.c): Shewchuk's
//! non-overlapping partials ("Adaptive Precision Floating-Point Arithmetic and Fast Robust
//! Geometric Predicates", 1997) followed by CPython's final correct-rounding step, which
//! fixes the half-way case across several partials (`fsum([1e-16, 1, 1e16])` rounds up to
//! `1.0000000000000002e16`, not down to `1e16`).

/// The correctly rounded sum of `values`, equal to Python's `math.fsum` bit for bit on every
/// finite input whose partial sums do not overflow.
///
/// Non-finite handling follows `math.fsum`: an infinity sums to itself, `inf + -inf` and any
/// NaN give NaN. Where Python raises `OverflowError` (finite inputs whose exact sum overflows
/// an intermediate partial) this returns NaN. An empty input, or one whose terms cancel
/// exactly, returns `+0.0`.
#[must_use]
pub fn fsum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    let mut special_sum = 0.0_f64;
    let mut inf_sum = 0.0_f64;
    let mut overflow = false;

    for value in values {
        let mut x = value;
        let mut kept = 0;
        let mut j = 0;
        while j < partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let yr = hi - x;
            let lo = y - yr;
            if lo != 0.0 {
                partials[kept] = lo;
                kept += 1;
            }
            x = hi;
            j += 1;
        }
        partials.truncate(kept);
        if x != 0.0 {
            if x.is_finite() {
                partials.push(x);
            } else {
                // A non-finite x is either an intermediate overflow of finite terms or the
                // result of an inf / NaN summand (CPython keeps the specials apart).
                if value.is_finite() {
                    overflow = true;
                } else {
                    if value.is_infinite() {
                        inf_sum += value;
                    }
                    special_sum += value;
                }
                partials.clear();
            }
        }
    }

    if special_sum != 0.0 {
        // (NaN != 0.0 is true, so a NaN summand lands here too, as in CPython.)
        return if inf_sum.is_nan() {
            f64::NAN
        } else {
            special_sum
        };
    }
    if overflow {
        return f64::NAN;
    }

    let mut n = partials.len();
    if n == 0 {
        return 0.0;
    }
    n -= 1;
    let mut hi = partials[n];
    let mut lo = 0.0_f64;
    // Sum the partials from the top, stopping when the sum becomes inexact.
    while n > 0 {
        let x = hi;
        n -= 1;
        let y = partials[n];
        hi = x + y;
        let yr = hi - x;
        lo = y - yr;
        if lo != 0.0 {
            break;
        }
    }
    // Make half-even rounding work across multiple partials: if the remaining partials push
    // `lo` further in its own direction, `hi + 2 lo` is the correctly rounded result.
    if n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) || (lo > 0.0 && partials[n - 1] > 0.0)) {
        let y = lo * 2.0;
        let x = hi + y;
        let yr = x - hi;
        if y == yr {
            hi = x;
        }
    }
    hi
}

#[cfg(test)]
mod tests {
    use super::fsum;

    /// `(name, terms, math.fsum result as IEEE bits)`, frozen from CPython 3.13.7
    /// (`uv run --project scripts/laya_train --frozen python`).
    fn frozen() -> Vec<(&'static str, Vec<f64>, u64)> {
        vec![
            ("tenth_x10", vec![0.1; 10], 0x3ff0_0000_0000_0000),
            (
                "cancel",
                vec![1e100, 1.0, -1e100, 1e-100, 1e50, -1.0, -1e50],
                0x2b2b_ff2e_e48e_0530,
            ),
            ("halfway_up", vec![1e-16, 1.0, 1e16], 0x4341_c379_37e0_8001),
            (
                "p53_a",
                vec![2f64.powi(53), -0.5, -2f64.powi(-54)],
                0x433f_ffff_ffff_ffff,
            ),
            (
                "p53_b",
                vec![2f64.powi(53), 1.0, 2f64.powi(-100)],
                0x4340_0000_0000_0001,
            ),
            (
                "p53_c",
                vec![2f64.powi(53) + 10.0, 1.0, 2f64.powi(-100)],
                0x4340_0000_0000_0006,
            ),
            (
                "p53_d",
                vec![2f64.powi(53) - 4.0, 0.5, 2f64.powi(-54)],
                0x433f_ffff_ffff_fffd,
            ),
            ("empty", vec![], 0),
            ("neg_zero", vec![-0.0], 0),
            (
                "thirds",
                vec![
                    1.0 / 3.0,
                    1.0 / 3.0,
                    1.0 / 3.0,
                    2.0 / 3.0,
                    2.0 / 3.0,
                    2.0 / 3.0,
                ],
                0x4008_0000_0000_0000,
            ),
        ]
    }

    #[test]
    fn exact_sum_matches_frozen_math_fsum() {
        for (name, terms, want) in frozen() {
            let got = fsum(terms.iter().copied()).to_bits();
            assert_eq!(
                got,
                want,
                "fsum({name}) = {:e} ({got:#018x}), math.fsum = {:e} ({want:#018x})",
                f64::from_bits(got),
                f64::from_bits(want)
            );
        }
    }

    /// The order-dependence a plain sum has and fsum must not: ten 0.1s sum left to right to
    /// 0.9999999999999999, and the cancelling case loses the 1e-100 entirely.
    #[test]
    fn exact_sum_differs_from_a_sequential_sum_where_it_must() {
        let seq: f64 = [0.1_f64; 10].iter().fold(0.0, |a, b| a + b);
        assert_ne!(seq.to_bits(), fsum([0.1; 10]).to_bits());
        assert_eq!(fsum([0.1; 10]).to_bits(), 1.0_f64.to_bits());
        let terms = [1e100, 1.0, -1e100, 1e-100, 1e50, -1.0, -1e50];
        let mut reversed = terms;
        reversed.reverse();
        assert_eq!(fsum(terms).to_bits(), fsum(reversed).to_bits());
        assert_eq!(fsum(terms), 1e-100);
    }

    #[test]
    fn exact_sum_non_finite_follows_math_fsum() {
        assert_eq!(fsum([1.0, f64::INFINITY]), f64::INFINITY);
        assert_eq!(fsum([f64::NEG_INFINITY, 2.0]), f64::NEG_INFINITY);
        assert!(fsum([f64::INFINITY, f64::NEG_INFINITY]).is_nan());
        assert!(fsum([1.0, f64::NAN]).is_nan());
        // math.fsum raises OverflowError here; fsum returns NaN.
        assert!(fsum([f64::MAX, f64::MAX]).is_nan());
    }
}
