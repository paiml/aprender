//! Review-round regression tests for the map projections (MAP-1): polar radicand, sub-region
//! edges, polar band, and committed reference tables. Split out of `projection_tests.rs` to keep
//! each test file under the 500-line gate. The code is moved, not changed.

use super::projection::{
    AlbersEqualArea, AlbersUsa, Equirectangular, Projection, US_SUB_REGION_BOXES,
};
use super::projection_tests::{albers_snyder, assert_round_trip, mercator_snyder, REF_TOL};

// ---------------------------------------------------------------------------
// Review round 1 (MAP-1): polar radicand, sub-region edges, polar band, tables
// ---------------------------------------------------------------------------

#[test]
fn albers_polar_radicand_is_not_negative_for_southern_first_parallel() {
    // With std2 = 90, the radicand C - 2 n sin(90) is mathematically non-negative for every
    // std1, but the rounded difference can fall just below zero. A negative radicand is NaN
    // under sqrt, so the forward pass must still return a finite result.
    for std1 in -89..=0 {
        let proj = AlbersEqualArea::new(f64::from(std1), 90.0, 90.0, 0.0, 1.0)
            .expect("parallels in [-90, 90] with n > 0");
        assert!(proj.forward(0.0, 90.0).is_ok(), "std1 = {std1}");
    }
}

#[test]
fn albers_usa_sub_region_edges_and_corners_round_trip() {
    // Box edges and corners are valid lon/lat. A 25 x 25 grid per box includes all four edges
    // and all four corners; each forward image must invert back into the same box.
    let usa = AlbersUsa::new(1.0).expect("valid");
    for [lon_min, lon_max, lat_min, lat_max] in US_SUB_REGION_BOXES {
        for i in 0..25 {
            for j in 0..25 {
                let lon = lon_min + (lon_max - lon_min) * f64::from(i) / 24.0;
                let lat = lat_min + (lat_max - lat_min) * f64::from(j) / 24.0;
                assert_round_trip(&usa, lon, lat);
            }
        }
    }
}

#[test]
fn albers_polar_band_latitude_error_below_1e_minus_5() {
    // asin is ill-conditioned near the poles. Above |lat| = 89.9 the documented bound is
    // 1e-5 degrees of latitude error.
    let proj = albers_snyder();
    for lat in [89.9, 89.95, 89.999, 90.0, -89.9, -89.95, -89.999, -90.0] {
        for lon in [-180.0, -150.0, -96.0, -75.0, 0.0, 45.0, 120.0, 179.0] {
            let (x, y) = proj.forward(lon, lat).expect("in domain");
            let (_, lat2) = proj.inverse(x, y).expect("inverse of a forward image");
            assert!((lat - lat2).abs() < 1e-5, "({lon}, {lat}) -> lat {lat2}");
        }
    }
}

/// Reference points for `albers_snyder()` (R = 1, origin 23 N 96 W): `(lon, lat, x, y)`.
/// Computed with awk from Snyder's spherical Albers equations, not by this crate.
const ALBERS_SNYDER_REFERENCE: [(f64, f64, f64, f64); 3] = [
    (-75.0, 35.0, 0.2952720070, 0.2416774492),
    (-122.0, 48.0, -0.3023683454, 0.4786226223),
    (-80.0, 26.0, 0.2520936251, 0.0729069239),
];

/// Reference points for `mercator_snyder()` (R = 1, origin 180 W): `(lon, lat, x, y)`.
/// Computed with awk from the spherical Mercator equations, with lon - lon0 wrapped into
/// [-180, 180). Not by this crate.
const MERCATOR_SNYDER_REFERENCE: [(f64, f64, f64, f64); 3] = [
    (-75.0, 35.0, 1.8325957146, 0.6528365797),
    (10.0, -45.0, -2.9670597284, -0.8813735870),
    (170.0, 60.0, -0.1745329252, 1.3169578969),
];

/// Reference points for `Equirectangular::new(30.0, 0.0, 1.0)`: `(lon, lat, x, y)`.
/// Computed with awk from x = lon_rad * cos(30 deg), y = lat_rad. Not by this crate.
const EQUIRECT_30_REFERENCE: [(f64, f64, f64, f64); 3] = [
    (10.0, 20.0, 0.1511499470, 0.3490658504),
    (-120.0, -45.0, -1.8137993642, -0.7853981634),
    (90.0, 60.0, 1.3603495232, 1.0471975512),
];

fn assert_reference_table(proj: &dyn Projection, table: &[(f64, f64, f64, f64)], name: &str) {
    for &(lon, lat, want_x, want_y) in table {
        let (x, y) = proj.forward(lon, lat).expect("inside the domain");
        assert!((x - want_x).abs() < REF_TOL, "{name} ({lon}, {lat}) x {x} want {want_x}");
        assert!((y - want_y).abs() < REF_TOL, "{name} ({lon}, {lat}) y {y} want {want_y}");
    }
}

#[test]
fn albers_matches_reference_table() {
    assert_reference_table(&albers_snyder(), &ALBERS_SNYDER_REFERENCE, "albers");
}

#[test]
fn mercator_matches_reference_table() {
    assert_reference_table(&mercator_snyder(), &MERCATOR_SNYDER_REFERENCE, "mercator");
}

#[test]
fn equirectangular_matches_reference_table() {
    let proj = Equirectangular::new(30.0, 0.0, 1.0).expect("valid");
    assert_reference_table(&proj, &EQUIRECT_30_REFERENCE, "equirect");
}

/// `std1` near -90 with `std2 = 90` gives a cone constant near zero, where computing
/// `cos^2(std1)` as `1 - sin^2(std1)` cancels and breaks the documented polar bound.
#[test]
fn albers_polar_bound_holds_for_a_near_degenerate_cone() {
    for std1 in [-89.9, -89.0, -87.0, -80.0] {
        let proj = AlbersEqualArea::new(std1, 90.0, 0.0, 0.0, 1.0).expect("n is positive");
        for lat in [89.95, 90.0] {
            let (x, y) = proj.forward(0.0, lat).expect("in domain");
            let (_, got) = proj.inverse(x, y).expect("image of the domain");
            assert!((got - lat).abs() < 1e-5, "std1 {std1} lat {lat} got {got}");
        }
    }
}
