// =========================================================================
// FALSIFY-DT-010/011: regression split search (#3859)
//
// Contract: decision-tree-v1.yaml
// =========================================================================

use crate::primitives::{Matrix, Vector};
use crate::tree::helpers::{
    evaluate_split_gain, find_best_regression_split_for_feature,
    find_best_regression_split_for_feature_rescan, split_by_threshold, variance_f32,
};
use crate::tree::DecisionTreeRegressor;

/// The accuracy envelope `FALSIFY-DT-010` allows between the two searches.
///
/// The oracle computes each child's variance in `f32`, two passes, in row
/// order. For `m` values the standard two-pass analysis bounds that error by
/// about `(m + 3)·u·Var + m·δ²`, with `u = 2^-24` and `δ` the error of the
/// `f32` mean (`≤ m·u·max|y|`). The fast path uses `f64` running moments, whose
/// own error (`~m·2^-53`) is negligible next to it, and rounds the split MSE
/// to `f32` once (`≤ u·mean(y²)`). Both terms are dominated by
/// `(n + 4)·ε·mean(y²)` with `ε = 2u = f32::EPSILON`, since `Var ≤ mean(y²)`;
/// the envelope is 8x that, plus `f32::MIN_POSITIVE` so an all-zero `y` does
/// not demand exact equality of two different summations.
fn envelope(y: &[f32]) -> f32 {
    let n = y.len() as f32;
    let mean_sq = y.iter().map(|v| v * v).sum::<f32>() / n.max(1.0);
    8.0 * (n + 4.0) * f32::EPSILON * mean_sq + f32::MIN_POSITIVE
}

/// The oracle's own gain for splitting column 0 at `threshold`, or `None` when
/// the oracle would refuse it (an empty side or no gain).
fn oracle_gain_at(x: &Matrix<f32>, y: &[f32], threshold: f32, current: f32) -> Option<f32> {
    let (left, right) = split_by_threshold(x, y, 0, threshold);
    evaluate_split_gain(&left, &right, current)
}

/// FALSIFY-DT-010: the running-moment split search is within the stated
/// envelope of the rescan it replaced.
///
/// Bit equality is not available here the way it is for the classifier's
/// Gini (`FALSIFY-DT-008`): variance from running sums reassociates the
/// floating-point additions. So the claim is stated against the oracle, the
/// pre-#3859 rescan kept verbatim:
///
/// 1. both refuse, or
/// 2. both split, the oracle scores the fast threshold within `tol` of its own
///    best (so a different threshold is only ever a near-tie), and the fast
///    gain is within `tol` of the oracle's score for that threshold, or
/// 3. exactly one splits, and its gain is within `tol` of zero.
///
/// `tol` is [`envelope`].
#[test]
fn falsify_dt_010_regression_split_within_envelope_of_rescan() {
    fn check(xs: &[f32], y: &[f32], case: &str) {
        let n = xs.len();
        let x = Matrix::from_vec(n, 1, xs.to_vec()).expect("valid matrix");
        let current = variance_f32(y);
        let tol = envelope(y);
        let fast = find_best_regression_split_for_feature(&x, y, 0, n, current);
        let oracle = find_best_regression_split_for_feature_rescan(&x, y, 0, n, current);

        match (fast, oracle) {
            (None, None) => {}
            (Some((tf, gf)), Some((to, go))) => {
                let scored = oracle_gain_at(&x, y, tf, current).unwrap_or(0.0);
                assert!(
                    scored >= go - tol,
                    "FALSIFIED DT-010 [{case}]: fast chose {tf} (oracle scores it {scored}), \
                     oracle chose {to} (gain {go}); tol {tol}"
                );
                assert!(
                    (gf - scored).abs() <= tol,
                    "FALSIFIED DT-010 [{case}]: fast gain {gf} at {tf}, oracle scores it {scored}; tol {tol}"
                );
            }
            (Some((t, g)), None) | (None, Some((t, g))) => assert!(
                g <= tol,
                "FALSIFIED DT-010 [{case}]: fast {fast:?} vs rescan {oracle:?} — the split at {t} \
                 has gain {g}, above tol {tol}"
            ),
        }
    }

    // Structured edge cases.
    check(&[1.0], &[3.0], "single sample");
    check(&[2.0, 2.0, 2.0], &[1.0, 5.0, 9.0], "one distinct value");
    check(
        &[0.0, 1.0, 2.0, 3.0],
        &[4.0, 4.0, 4.0, 4.0],
        "constant targets",
    );
    check(
        &[0.0, 1.0, 10.0, 11.0],
        &[0.0, 0.0, 5.0, 5.0],
        "clean two-way split",
    );
    check(
        &[3.0, 0.0, 2.0, 1.0],
        &[9.0, 0.0, 9.0, 0.0],
        "rows out of order",
    );
    check(
        &[-0.0, 0.0, 1.0],
        &[1.0, 7.0, 7.0],
        "signed zeros are one value",
    );
    // Adjacent floats: `midpoint` rounds onto the upper value, so the `<=`
    // partition takes it left too — a `<=` → `<` mutant differs here.
    let a = 1.0_f32;
    let b = f32::from_bits(a.to_bits() + 1);
    check(&[a, b, 2.0, 2.0], &[0.0, 0.0, 8.0, 8.0], "adjacent floats");
    check(
        &[a, b, b, 2.0],
        &[0.0, 6.0, 6.0, 9.0],
        "adjacent floats, upper duplicated",
    );
    // NaN and infinities: a NaN row goes right at every threshold, and
    // `midpoint(-inf, inf)` is NaN, which the rescan cannot split on.
    check(&[f32::NAN, 0.0, 1.0, 2.0], &[9.0, 0.0, 0.0, 9.0], "NaN row");
    check(
        &[-f32::NAN, 0.0, 1.0, 2.0],
        &[9.0, 0.0, 0.0, 9.0],
        "negative NaN row",
    );
    check(
        &[f32::NEG_INFINITY, f32::INFINITY, f32::INFINITY],
        &[0.0, 5.0, 5.0],
        "only infinities",
    );
    check(
        &[f32::NEG_INFINITY, 0.0, f32::INFINITY, 1.0],
        &[0.0, 1.0, 5.0, 1.0],
        "infinities around finite values",
    );

    // Randomised cases. A small quantised value range forces duplicate feature
    // values, empty sides and gain ties, where an index-shifted rewrite would
    // diverge from the rescan.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as usize
    };
    for case in 0..500 {
        let n = 2 + next() % 64;
        let spread = 1 + next() % 10;
        let y_spread = 1 + next() % 1000;
        let offset = if case % 3 == 0 { 1.0e4 } else { 0.0 };
        let x: Vec<f32> = (0..n).map(|_| (next() % spread) as f32 * 0.25).collect();
        let y: Vec<f32> = (0..n)
            .map(|_| offset + (next() % y_spread) as f32 * 0.01)
            .collect();
        check(
            &x,
            &y,
            &format!("random case {case} (n={n}, offset={offset})"),
        );
    }

    // A continuous column, every value its own candidate — the shape that made
    // the rescan quadratic.
    let n = 400;
    let x: Vec<f32> = (0..n)
        .map(|_| next() as f32 / (1u64 << 31) as f32)
        .collect();
    let y: Vec<f32> = x
        .iter()
        .map(|&v| if v > 0.4 { 3.0 + v } else { v })
        .collect();
    check(&x, &y, "continuous column");

    // Exact ties: the envelope accepts either tied threshold, so the tie-break
    // is asserted separately. Both sides' moments here come from the same push
    // sequence, so the tied gains are bit-equal and the first threshold must win.
    for (xs, y, first) in [
        (&[0.0, 1.0, 2.0][..], &[0.0, 1.0, 0.0][..], 0.5_f32),
        (
            &[0.0, 1.0, 2.0, 3.0, 4.0][..],
            &[0.0, 0.0, 5.0, 0.0, 0.0][..],
            1.5,
        ),
    ] {
        let n = xs.len();
        let x = Matrix::from_vec(n, 1, xs.to_vec()).expect("valid matrix");
        let current = variance_f32(y);
        let fast = find_best_regression_split_for_feature(&x, y, 0, n, current);
        let oracle = find_best_regression_split_for_feature_rescan(&x, y, 0, n, current);
        assert_eq!(fast.map(|s| s.0), Some(first), "tie {y:?}: fast");
        assert_eq!(oracle.map(|s| s.0), Some(first), "tie {y:?}: oracle");
    }
}

/// FALSIFY-DT-011: regression root split search is sub-quadratic in row count
///
/// #3859. `max_depth = 1` on one continuous column, so exactly one split is
/// searched and the timing is the split search. Both sides of the budget are
/// measured in the same debug build; see the contract entry for the figures.
#[test]
fn falsify_dt_011_regression_root_split_search_is_subquadratic() {
    use std::time::{Duration, Instant};

    const N: usize = 200_000;
    const BUDGET: Duration = Duration::from_secs(60);

    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f32 / (1u64 << 53) as f32
    };

    let values: Vec<f32> = (0..N).map(|_| next()).collect();
    let y: Vec<f32> = values
        .iter()
        .map(|&v| if v > 0.6 { 2.0 + v } else { v })
        .collect();
    let x = Matrix::from_vec(N, 1, values).expect("valid matrix");
    let y = Vector::from_vec(y);

    let mut dt = DecisionTreeRegressor::new().with_max_depth(1);
    let started = Instant::now();
    dt.fit(&x, &y).expect("fit");
    let elapsed = started.elapsed();

    assert!(
        elapsed < BUDGET,
        "FALSIFIED DT-011: one root split over {N} rows took {elapsed:?} (budget {BUDGET:?}) \
         — regression split search is quadratic in row count again"
    );
}
