//! Tests for polyline simplification: hand-computed references, edge cases, properties.
//!
//! The reference polylines are derived by hand with the distances and areas written out in
//! comments. None is filled in by running the implementation under test.

use super::projection::GeoError;
use super::simplify::{douglas_peucker, visvalingam};
use proptest::prelude::*;

/// Committed fixture: seven points, the middle ones off the x-axis.
const FIXTURE: [(f64, f64); 7] =
    [(0.0, 0.0), (1.0, 0.1), (2.0, 0.0), (3.0, 2.0), (4.0, 0.0), (5.0, 0.4), (6.0, 0.0)];

/// Tolerance used for the fixture: a distance for Douglas-Peucker, an area for Visvalingam.
const FIXTURE_TOL: f64 = 0.5;

/// Hand-computed Douglas-Peucker result of `FIXTURE` at `FIXTURE_TOL`.
///
/// Chord p0-p6 is the x-axis, so a point's distance is `|y|`: p1 0.1, p2 0, p3 2, p4 0,
/// p5 0.4. The farthest, p3 (2 > 0.5), is kept.
///
/// Chord p0-p3 has `dx = 3, dy = 2, |chord| = sqrt(13) = 3.6056`. Distance is
/// `|(x - 0) * 2 - (y - 0) * 3| / sqrt(13)`: p1 gives `|2 - 0.3| / 3.6056 = 0.4714` (dropped),
/// p2 gives `|4 - 0| / 3.6056 = 1.1094` (kept, the farthest). Chord p0-p2 is the x-axis and
/// p1 is at distance 0.1 (dropped). Chord p2-p3 has no interior point.
///
/// Chord p3-p6 has `dx = 3, dy = -2, |chord| = sqrt(13)`. Distance is
/// `|(x - 3) * (-2) - (y - 2) * 3| / sqrt(13)`: p4 gives `|-2 + 6| / 3.6056 = 1.1094` (kept),
/// p5 gives `|-4 + 4.8| / 3.6056 = 0.2219` (dropped). Chord p3-p4 has no interior point.
/// Chord p4-p6 is the x-axis and p5 is at distance `0.4` (dropped).
///
/// Kept: p0, p2, p3, p4, p6.
const DP_REFERENCE: [(f64, f64); 5] = [(0.0, 0.0), (2.0, 0.0), (3.0, 2.0), (4.0, 0.0), (6.0, 0.0)];

/// Hand-computed Visvalingam-Whyatt result of `FIXTURE` at `FIXTURE_TOL`.
///
/// Triangle area is `|cross| / 2` of the point's two neighbours. Initial areas: p1 `|-0.2|/2
/// = 0.1`, p2 `|2.1|/2 = 1.05`, p3 `|-4|/2 = 2`, p4 `|2.4|/2 = 1.2`, p5 `|-0.8|/2 = 0.4`.
/// The smallest is p1 (0.1 < 0.5): removed.
///
/// Neighbours of p2 become p0 and p3: `|(2, 0) x (3, 2)| / 2 = 4 / 2 = 2`. Areas now: p2 2,
/// p3 2, p4 1.2, p5 0.4. The smallest is p5 (0.4 < 0.5): removed.
///
/// Neighbours of p4 become p3 and p6: `|(1, -2) x (3, -2)| / 2 = |-2 + 6| / 2 = 2`. Areas now:
/// p2 2, p3 2, p4 2, all at least 0.5: stop.
///
/// Kept: p0, p2, p3, p4, p6.
const VW_REFERENCE: [(f64, f64); 5] = [(0.0, 0.0), (2.0, 0.0), (3.0, 2.0), (4.0, 0.0), (6.0, 0.0)];

fn same_point(a: (f64, f64), b: (f64, f64)) -> bool {
    a.0.to_bits() == b.0.to_bits() && a.1.to_bits() == b.1.to_bits()
}

fn assert_same(got: &[(f64, f64)], want: &[(f64, f64)]) {
    assert_eq!(got.len(), want.len(), "length: got {got:?} want {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!(same_point(*g, *w), "got {got:?} want {want:?}");
    }
}

/// True when `out` is a subsequence of `input`: every point of `out` appears in `input`,
/// in the same order, with the same bits.
fn is_subsequence(out: &[(f64, f64)], input: &[(f64, f64)]) -> bool {
    let mut rest = input.iter();
    out.iter().all(|q| rest.any(|p| same_point(*p, *q)))
}

#[test]
fn douglas_peucker_matches_hand_computed_reference() {
    let got = douglas_peucker(&FIXTURE, FIXTURE_TOL).expect("valid input");
    assert_same(&got, &DP_REFERENCE);
}

#[test]
fn visvalingam_matches_hand_computed_reference() {
    let got = visvalingam(&FIXTURE, FIXTURE_TOL).expect("valid input");
    assert_same(&got, &VW_REFERENCE);
}

#[test]
fn visvalingam_stops_at_first_area_not_below_tolerance() {
    // Areas: p1 0.1, p2 1.05, p3 2, p4 1.2, p5 0.4. At tolerance 0.3 only p1 (0.1) is
    // removed. After that the smallest is p5 at 0.4, which is not below 0.3, so it stays.
    let got = visvalingam(&FIXTURE, 0.3).expect("valid input");
    assert_same(&got, &[(0.0, 0.0), (2.0, 0.0), (3.0, 2.0), (4.0, 0.0), (5.0, 0.4), (6.0, 0.0)]);
}

#[test]
fn tolerance_zero_returns_input_unchanged_including_collinear_points() {
    let collinear = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)];
    assert_same(&douglas_peucker(&FIXTURE, 0.0).expect("valid"), &FIXTURE);
    assert_same(&visvalingam(&FIXTURE, 0.0).expect("valid"), &FIXTURE);
    assert_same(&douglas_peucker(&collinear, 0.0).expect("valid"), &collinear);
    assert_same(&visvalingam(&collinear, 0.0).expect("valid"), &collinear);
}

#[test]
fn collinear_points_are_dropped_at_positive_tolerance() {
    // Every interior point is on the chord (distance 0, area 0), so all are removed.
    let collinear = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)];
    let want = [(0.0, 0.0), (3.0, 3.0)];
    assert_same(&douglas_peucker(&collinear, 0.1).expect("valid"), &want);
    assert_same(&visvalingam(&collinear, 0.1).expect("valid"), &want);
}

#[test]
fn negative_tolerance_is_an_invalid_parameter() {
    let want = GeoError::InvalidParameter { name: "tolerance", value: -1.0 };
    assert_eq!(douglas_peucker(&FIXTURE, -1.0), Err(want));
    assert_eq!(visvalingam(&FIXTURE, -1.0), Err(want));
}

#[test]
fn non_finite_tolerance_is_rejected() {
    for tol in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(douglas_peucker(&FIXTURE, tol), Err(GeoError::NonFiniteInput));
        assert_eq!(visvalingam(&FIXTURE, tol), Err(GeoError::NonFiniteInput));
    }
}

#[test]
fn non_finite_point_is_rejected() {
    let bad_x = [(0.0, 0.0), (f64::NAN, 1.0), (2.0, 0.0)];
    let bad_y = [(0.0, 0.0), (1.0, 1.0), (2.0, f64::INFINITY)];
    assert_eq!(douglas_peucker(&bad_x, 0.5), Err(GeoError::NonFiniteInput));
    assert_eq!(visvalingam(&bad_x, 0.5), Err(GeoError::NonFiniteInput));
    assert_eq!(douglas_peucker(&bad_y, 0.5), Err(GeoError::NonFiniteInput));
    assert_eq!(visvalingam(&bad_y, 0.5), Err(GeoError::NonFiniteInput));
}

#[test]
fn fewer_than_three_points_are_returned_unchanged() {
    let two = [(0.0, 0.0), (5.0, 5.0)];
    let one = [(1.0, 2.0)];
    let none: [(f64, f64); 0] = [];
    for pts in [&none[..], &one[..], &two[..]] {
        assert_same(&douglas_peucker(pts, 10.0).expect("valid"), pts);
        assert_same(&visvalingam(pts, 10.0).expect("valid"), pts);
    }
}

#[test]
fn douglas_peucker_zero_length_chord_measures_distance_to_the_point() {
    // Endpoints coincide at the origin. The interior point (3, 4) is 5 away from it.
    let spike = [(0.0, 0.0), (3.0, 4.0), (0.0, 0.0)];
    assert_same(&douglas_peucker(&spike, 1.0).expect("valid"), &spike);
    assert_same(&douglas_peucker(&spike, 6.0).expect("valid"), &[(0.0, 0.0), (0.0, 0.0)]);
}

fn polyline() -> impl Strategy<Value = Vec<(f64, f64)>> {
    prop::collection::vec((-1000.0f64..1000.0, -1000.0f64..1000.0), 0..64)
}

/// The three properties every simplifier must hold: end points kept, output is a
/// subsequence of the input (no point added or moved), and order preserved.
fn check_properties(pts: &[(f64, f64)], out: &[(f64, f64)]) {
    if let (Some(first), Some(last)) = (pts.first(), pts.last()) {
        assert!(out.first().is_some_and(|q| same_point(*q, *first)), "first end point lost");
        assert!(out.last().is_some_and(|q| same_point(*q, *last)), "last end point lost");
    }
    assert!(out.len() <= pts.len(), "output longer than input");
    assert!(is_subsequence(out, pts), "output is not a subsequence of input");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn douglas_peucker_keeps_ends_and_is_a_subsequence(pts in polyline(), tol in 0.0f64..500.0) {
        let out = douglas_peucker(&pts, tol).expect("finite input, valid tolerance");
        check_properties(&pts, &out);
    }

    #[test]
    fn visvalingam_keeps_ends_and_is_a_subsequence(pts in polyline(), tol in 0.0f64..500.0) {
        let out = visvalingam(&pts, tol).expect("finite input, valid tolerance");
        check_properties(&pts, &out);
    }

    #[test]
    fn zero_tolerance_is_the_identity_for_both(pts in polyline()) {
        let dp = douglas_peucker(&pts, 0.0).expect("valid");
        let vw = visvalingam(&pts, 0.0).expect("valid");
        prop_assert_eq!(dp.len(), pts.len());
        prop_assert_eq!(vw.len(), pts.len());
        prop_assert!(is_subsequence(&dp, &pts));
        prop_assert!(is_subsequence(&vw, &pts));
    }
}
