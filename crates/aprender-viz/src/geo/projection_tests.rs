//! Tests for the map projections: published reference values, round trips, domain errors.
//!
//! Expected values are committed constants (Snyder, USGS PP 1395) or hand-derived identities
//! written out in comments. None is filled in by running the implementation under test.

use super::projection::{
    AlbersEqualArea, AlbersUsa, Equirectangular, GeoError, Mercator, Projection,
    MERCATOR_MAX_LAT_DEG, US_SUB_REGION_BOXES,
};
use proptest::prelude::*;
use std::f64::consts::{FRAC_PI_2, PI};

/// Tolerance for the committed Snyder reference values.
const REF_TOL: f64 = 1e-6;
/// Tolerance for forward-then-inverse round trips, in degrees of longitude or latitude.
const ROUND_TRIP_DEG: f64 = 1e-7;
/// Tolerance for exact-by-construction identities (origin maps to the plane origin).
const EXACT_TOL: f64 = 1e-12;

fn albers_snyder() -> AlbersEqualArea {
    AlbersEqualArea::new(29.5, 45.5, 23.0, -96.0, 1.0).expect("Snyder Albers parameters are valid")
}

fn mercator_snyder() -> Mercator {
    Mercator::new(-180.0, 1.0).expect("Snyder Mercator parameters are valid")
}

/// Smallest angular gap between two longitudes in degrees, accounting for wrap at 180.
fn lon_gap(a: f64, b: f64) -> f64 {
    ((a - b + 180.0).rem_euclid(360.0) - 180.0).abs()
}

fn assert_pair_close(got: (f64, f64), want: (f64, f64), tol: f64) {
    assert!((got.0 - want.0).abs() < tol, "x: got {} want {}", got.0, want.0);
    assert!((got.1 - want.1).abs() < tol, "y: got {} want {}", got.1, want.1);
}

/// Asserts a round trip returns the original lon/lat within `ROUND_TRIP_DEG`.
fn assert_round_trip(proj: &dyn Projection, lon: f64, lat: f64) {
    let (x, y) = proj.forward(lon, lat).expect("forward inside the valid domain");
    let (lon2, lat2) = proj.inverse(x, y).expect("inverse of a forward image");
    assert!(lon_gap(lon, lon2) < ROUND_TRIP_DEG, "lon {lon} -> {lon2}");
    assert!((lat - lat2).abs() < ROUND_TRIP_DEG, "lat {lat} -> {lat2}");
}

// ---------------------------------------------------------------------------
// (a) Committed reference values
// ---------------------------------------------------------------------------

#[test]
fn albers_matches_snyder_reference() {
    // Snyder spherical example: R = 1, parallels 29.5 and 45.5, origin 23 N 96 W,
    // point 35 N 75 W.
    let got = albers_snyder().forward(-75.0, 35.0).expect("in domain");
    assert_pair_close(got, (0.295_272_0, 0.241_677_4), REF_TOL);
}

#[test]
fn mercator_matches_snyder_reference() {
    // Snyder spherical example: R = 1, origin longitude 180 W, point 35 N 75 W.
    // x = R * 105 deg in radians = 105 pi / 180 = 1.8325957...
    // y = R * ln tan(45 deg + 17.5 deg) = ln tan 62.5 deg = 0.6528366...
    let got = mercator_snyder().forward(-75.0, 35.0).expect("in domain");
    assert_pair_close(got, (1.832_595_7, 0.652_836_6), REF_TOL);
}

#[test]
fn equirectangular_matches_hand_identity() {
    // R = 1, standard parallel 30 deg, origin longitude 0.
    // x = R * (10 deg in rad) * cos(30 deg) = (pi/18) * (sqrt(3)/2) = pi sqrt(3) / 36.
    // y = R * (20 deg in rad) = pi / 9.
    let proj = Equirectangular::new(30.0, 0.0, 1.0).expect("valid parameters");
    let got = proj.forward(10.0, 20.0).expect("in domain");
    assert_pair_close(got, (PI * 3.0_f64.sqrt() / 36.0, PI / 9.0), EXACT_TOL);
}

#[test]
fn albers_origin_maps_to_plane_origin() {
    // At the origin longitude and latitude, theta = 0 and rho = rho0, so x = 0 and y = 0.
    let got = albers_snyder().forward(-96.0, 23.0).expect("in domain");
    assert_pair_close(got, (0.0, 0.0), EXACT_TOL);
}

#[test]
fn wrap_makes_lon_and_lon_plus_360_identical() {
    // Longitude differences are wrapped into [-180, 180) before projecting.
    let proj = Equirectangular::new(0.0, 0.0, 1.0).expect("valid parameters");
    assert_eq!(proj.forward(190.0, 0.0), proj.forward(-170.0, 0.0));
    assert_eq!(mercator_snyder().forward(105.0, 0.0), mercator_snyder().forward(-255.0, 0.0));
}

#[test]
fn equirectangular_pole_inverse_clamps_to_ninety() {
    let proj = Equirectangular::new(0.0, 0.0, 1.0).expect("valid parameters");
    let (_, lat) = proj.inverse(0.0, FRAC_PI_2).expect("y at the pole is in domain");
    assert!((lat - 90.0).abs() < EXACT_TOL, "lat {lat}");
}

// ---------------------------------------------------------------------------
// Constructor validation
// ---------------------------------------------------------------------------

#[test]
fn albers_rejects_invalid_radius() {
    for radius in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let err = AlbersEqualArea::new(29.5, 45.5, 23.0, -96.0, radius);
        assert!(matches!(err, Err(GeoError::InvalidParameter { name: "radius", .. })));
    }
}

#[test]
fn albers_rejects_zero_cone_constant() {
    // n = (sin 30 + sin -30) / 2 = 0: the cone degenerates.
    let err = AlbersEqualArea::new(30.0, -30.0, 0.0, 0.0, 1.0);
    assert!(matches!(err, Err(GeoError::InvalidParameter { name: "n", .. })));
}

#[test]
fn albers_rejects_southern_cone() {
    // Both parallels south of the equator give n < 0; this phase does not support it.
    let err = AlbersEqualArea::new(-30.0, -40.0, -35.0, 0.0, 1.0);
    assert!(matches!(err, Err(GeoError::InvalidParameter { name: "n", .. })));
}

#[test]
fn albers_rejects_invalid_angles() {
    assert!(matches!(
        AlbersEqualArea::new(95.0, 45.5, 23.0, -96.0, 1.0),
        Err(GeoError::InvalidParameter { name: "standard_parallel_1", .. })
    ));
    assert!(matches!(
        AlbersEqualArea::new(29.5, f64::NAN, 23.0, -96.0, 1.0),
        Err(GeoError::InvalidParameter { name: "standard_parallel_2", .. })
    ));
    assert!(matches!(
        AlbersEqualArea::new(29.5, 45.5, -91.0, -96.0, 1.0),
        Err(GeoError::InvalidParameter { name: "origin_lat", .. })
    ));
    assert!(matches!(
        AlbersEqualArea::new(29.5, 45.5, 23.0, f64::INFINITY, 1.0),
        Err(GeoError::InvalidParameter { name: "origin_lon", .. })
    ));
}

#[test]
fn mercator_and_equirectangular_reject_invalid_parameters() {
    assert!(matches!(
        Mercator::new(0.0, 0.0),
        Err(GeoError::InvalidParameter { name: "radius", .. })
    ));
    assert!(matches!(
        Mercator::new(f64::NAN, 1.0),
        Err(GeoError::InvalidParameter { name: "origin_lon", .. })
    ));
    assert!(matches!(
        Equirectangular::new(90.0, 0.0, 1.0),
        Err(GeoError::InvalidParameter { name: "standard_parallel", .. })
    ));
    assert!(matches!(
        Equirectangular::new(f64::NAN, 0.0, 1.0),
        Err(GeoError::InvalidParameter { name: "standard_parallel", .. })
    ));
    assert!(matches!(
        Equirectangular::new(0.0, 0.0, -2.0),
        Err(GeoError::InvalidParameter { name: "radius", .. })
    ));
    assert!(matches!(
        Equirectangular::new(0.0, f64::NAN, 1.0),
        Err(GeoError::InvalidParameter { name: "origin_lon", .. })
    ));
}

#[test]
fn albers_usa_rejects_invalid_radius() {
    assert!(matches!(AlbersUsa::new(0.0), Err(GeoError::InvalidParameter { name: "radius", .. })));
    assert!(AlbersUsa::new(f64::NAN).is_err());
}

// ---------------------------------------------------------------------------
// Non-finite input and result are errors
// ---------------------------------------------------------------------------

#[test]
fn non_finite_input_is_an_error_for_every_projection() {
    let projections: [Box<dyn Projection>; 4] = [
        Box::new(albers_snyder()),
        Box::new(mercator_snyder()),
        Box::new(Equirectangular::new(30.0, 0.0, 1.0).expect("valid")),
        Box::new(AlbersUsa::new(1.0).expect("valid")),
    ];
    for proj in &projections {
        assert_eq!(proj.forward(f64::NAN, 35.0), Err(GeoError::NonFiniteInput));
        assert_eq!(proj.forward(-96.0, f64::INFINITY), Err(GeoError::NonFiniteInput));
        assert_eq!(proj.inverse(f64::NAN, 0.0), Err(GeoError::NonFiniteInput));
        assert_eq!(proj.inverse(0.0, f64::NEG_INFINITY), Err(GeoError::NonFiniteInput));
    }
}

#[test]
fn overflowing_result_is_an_error_not_a_number() {
    // A radius near f64::MAX makes the plane coordinates overflow to infinity.
    let albers = AlbersEqualArea::new(29.5, 45.5, 23.0, -96.0, f64::MAX).expect("valid");
    assert_eq!(albers.forward(10.0, 10.0), Err(GeoError::NonFiniteResult));
    let mercator = Mercator::new(0.0, f64::MAX).expect("valid");
    assert_eq!(mercator.forward(170.0, 0.0), Err(GeoError::NonFiniteResult));
    let equirect = Equirectangular::new(0.0, 0.0, f64::MAX).expect("valid");
    assert_eq!(equirect.forward(170.0, 0.0), Err(GeoError::NonFiniteResult));
    let usa = AlbersUsa::new(f64::MAX).expect("valid");
    assert_eq!(usa.forward(-96.0, 37.0), Err(GeoError::NonFiniteResult));
}

// ---------------------------------------------------------------------------
// Domain errors, forward and inverse
// ---------------------------------------------------------------------------

#[test]
fn albers_domain_errors() {
    let albers = albers_snyder();
    assert!(matches!(
        albers.forward(0.0, 91.0),
        Err(GeoError::OutOfDomain { what: "latitude", .. })
    ));
    // Far outside the image: the sine of latitude falls below -1.
    assert!(matches!(
        albers.inverse(1000.0, 0.0),
        Err(GeoError::OutOfDomain { what: "sin(latitude)", .. })
    ));
    // Behind the cone: the sector angle exceeds n pi.
    assert!(matches!(
        albers.inverse(0.0, 1000.0),
        Err(GeoError::OutOfDomain { what: "sector angle", .. })
    ));
}

#[test]
fn mercator_domain_errors() {
    let mercator = mercator_snyder();
    assert!(mercator.forward(0.0, MERCATOR_MAX_LAT_DEG).is_ok());
    for lat in [85.5, 90.0, -90.0] {
        assert!(matches!(
            mercator.forward(0.0, lat),
            Err(GeoError::OutOfDomain { what: "latitude", .. })
        ));
    }
    assert!(matches!(
        mercator.inverse(4.0 * PI, 0.0),
        Err(GeoError::OutOfDomain { what: "x/R", .. })
    ));
    assert!(matches!(mercator.inverse(0.0, 4.0), Err(GeoError::OutOfDomain { what: "y/R", .. })));
}

#[test]
fn equirectangular_domain_errors() {
    let proj = Equirectangular::new(0.0, 0.0, 1.0).expect("valid");
    assert!(matches!(proj.forward(0.0, 91.0), Err(GeoError::OutOfDomain { what: "latitude", .. })));
    assert!(matches!(proj.inverse(0.0, 2.0), Err(GeoError::OutOfDomain { what: "y/R", .. })));
    assert!(matches!(
        proj.inverse(10.0, 0.0),
        Err(GeoError::OutOfDomain { what: "x/(R cos phi_s)", .. })
    ));
}

#[test]
fn albers_usa_domain_errors() {
    let usa = AlbersUsa::new(1.0).expect("valid");
    // Longitude 0 lies in no sub-region box.
    assert_eq!(usa.forward(0.0, 0.0), Err(GeoError::NoSubRegion { lon: 0.0, lat: 0.0 }));
    // A plane point far from every sub-region image.
    assert_eq!(usa.inverse(5.0, 5.0), Err(GeoError::NoPlacement { x: 5.0, y: 5.0 }));
    // A plane point inside one sub-projection's domain but outside its box.
    assert!(matches!(usa.inverse(0.0, 0.6), Err(GeoError::NoPlacement { .. })));
}

#[test]
fn albers_usa_places_each_sub_region() {
    let usa = AlbersUsa::new(1.0).expect("valid");
    // One point per box, checked for a finite, in-range placement and a clean round trip.
    for [lon_min, lon_max, lat_min, lat_max] in US_SUB_REGION_BOXES {
        let (lon, lat) = ((lon_min + lon_max) / 2.0, (lat_min + lat_max) / 2.0);
        assert_round_trip(&usa, lon, lat);
    }
}

// ---------------------------------------------------------------------------
// (b) Round trips inside each projection's valid domain
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Albers round trip for all longitudes and latitudes up to 89.9 degrees.
    /// Limit: `asin` is ill-conditioned at the pole, so the accuracy claim (1e-7 deg) holds
    /// to 89.9 degrees, the bound documented on `AlbersEqualArea`.
    #[test]
    fn albers_round_trip(lon in -180.0_f64..180.0, lat in -89.9_f64..89.9) {
        assert_round_trip(&albers_snyder(), lon, lat);
    }

    /// Mercator round trip inside the documented latitude cutoff.
    #[test]
    fn mercator_round_trip(lon in -180.0_f64..180.0, lat in -85.0_f64..85.0) {
        assert_round_trip(&mercator_snyder(), lon, lat);
    }

    /// Equirectangular round trip over the whole sphere, poles included.
    #[test]
    fn equirectangular_round_trip(lon in -180.0_f64..180.0, lat in -90.0_f64..90.0) {
        let proj = Equirectangular::new(30.0, -96.0, 6371.0).expect("valid");
        assert_round_trip(&proj, lon, lat);
    }

    /// AlbersUSA round trip for points drawn from each of the three sub-region boxes.
    #[test]
    fn albers_usa_round_trip(
        region in 0usize..3,
        u in 0.0_f64..=1.0,
        v in 0.0_f64..=1.0,
    ) {
        let [lon_min, lon_max, lat_min, lat_max] = US_SUB_REGION_BOXES[region];
        let lon = lon_min + u * (lon_max - lon_min);
        let lat = lat_min + v * (lat_max - lat_min);
        assert_round_trip(&AlbersUsa::new(1.0).expect("valid"), lon, lat);
    }

    // -----------------------------------------------------------------------
    // (c) Outside the domain returns Err
    // -----------------------------------------------------------------------

    #[test]
    fn albers_latitude_beyond_pole_is_err(lat in 90.000_001_f64..1000.0) {
        prop_assert!(albers_snyder().forward(0.0, lat).is_err());
    }

    #[test]
    fn mercator_latitude_beyond_cutoff_is_err(
        lat in prop_oneof![85.000_001_f64..=90.0, -90.0_f64..=-85.000_001],
    ) {
        prop_assert!(mercator_snyder().forward(0.0, lat).is_err());
    }

    #[test]
    fn mercator_y_beyond_cutoff_is_err(y in 3.2_f64..1000.0) {
        prop_assert!(mercator_snyder().inverse(0.0, y).is_err());
    }

    #[test]
    fn equirectangular_latitude_beyond_pole_is_err(lat in 90.000_001_f64..1000.0) {
        let proj = Equirectangular::new(0.0, 0.0, 1.0).expect("valid");
        prop_assert!(proj.forward(0.0, lat).is_err());
    }

    #[test]
    fn albers_usa_outside_every_box_is_err(lon in -60.0_f64..60.0, lat in -90.0_f64..90.0) {
        let usa = AlbersUsa::new(1.0).expect("valid");
        let outside = matches!(usa.forward(lon, lat), Err(GeoError::NoSubRegion { .. }));
        prop_assert!(outside);
    }
}

// ---------------------------------------------------------------------------
// AlbersUsa reference table and inset placement
// ---------------------------------------------------------------------------

/// One point per sub-region at unit radius. Computed outside this crate from Snyder's
/// spherical Albers equations with each sub-region's documented parallels and origin, then
/// its documented scale and shift: `(lon, lat, x, y)`.
const ALBERS_USA_REFERENCE: [(f64, f64, f64, f64); 3] = [
    (-75.0, 35.0, 0.295_272_007_0, -0.011_286_453_3),
    (-150.0, 61.0, -0.888_205_377_0, -0.382_477_789_2),
    (-156.0, 20.0, -0.583_537_258_5, -0.497_632_850_0),
];

#[test]
fn albers_usa_matches_the_reference_table() {
    let usa = AlbersUsa::new(1.0).expect("valid");
    for (lon, lat, want_x, want_y) in ALBERS_USA_REFERENCE {
        let (x, y) = usa.forward(lon, lat).expect("inside a sub-region");
        assert!((x - want_x).abs() < REF_TOL, "({lon}, {lat}) x {x}");
        assert!((y - want_y).abs() < REF_TOL, "({lon}, {lat}) y {y}");
    }
}

#[test]
fn albers_usa_inset_images_do_not_overlap() {
    let usa = AlbersUsa::new(1.0).expect("valid");
    // Plane bounding box [x_min, x_max, y_min, y_max] of a 101 x 101 sample of each box.
    let images: Vec<[f64; 4]> = US_SUB_REGION_BOXES
        .iter()
        .map(|&[lon_min, lon_max, lat_min, lat_max]| {
            let mut image = [f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY];
            for i in 0..=100 {
                for j in 0..=100 {
                    let lon = lon_min + (lon_max - lon_min) * f64::from(i) / 100.0;
                    let lat = lat_min + (lat_max - lat_min) * f64::from(j) / 100.0;
                    let (x, y) = usa.forward(lon, lat).expect("inside the box");
                    image = [image[0].min(x), image[1].max(x), image[2].min(y), image[3].max(y)];
                }
            }
            image
        })
        .collect();
    for (a, first) in images.iter().enumerate() {
        for second in &images[a + 1..] {
            let apart = first[1] < second[0]
                || second[1] < first[0]
                || first[3] < second[2]
                || second[3] < first[2];
            assert!(apart, "{first:?} overlaps {second:?}");
        }
    }
}

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

/// Claim: Mercator inverse accepts `y` up to `y_max + DOMAIN_TOL` but does not clamp the
/// latitude, so it can return a latitude that forward then rejects.
#[test]
fn mercator_inverse_slack_result_is_in_forward_domain() {
    let proj = Mercator::new(0.0, 1.0).expect("valid");
    let (_, y_cut) = proj.forward(0.0, MERCATOR_MAX_LAT_DEG).expect("cutoff is in domain");
    let y_slack = y_cut + 5e-13;
    let (lon, lat) = proj.inverse(0.0, y_slack).expect("within the inverse slack");
    println!("MERCATOR_SLACK y_cut={y_cut:?} lat={lat:?}");
    assert!(proj.forward(lon, lat).is_ok(), "inverse returned lat {lat:?} that forward rejects");
}

/// Claim: `wrap180` can return +180.0 for an input just below -180.0 (range should be [-180, 180)).
#[test]
fn equirectangular_inverse_longitude_stays_half_open() {
    let proj = Equirectangular::new(0.0, 0.0, 1.0).expect("valid");
    let x = -(PI + f64::EPSILON * PI);
    let (lon, _) = proj.inverse(x, 0.0).expect("within the inverse slack");
    println!("WRAP180 x={x:?} lon={lon:?}");
    assert!(lon < 180.0 && lon >= -180.0, "lon {lon:?} outside [-180, 180)");
}

#[test]
fn wrap180_stays_half_open_just_below_minus_180() {
    let below = f64::from_bits((-180.0_f64).to_bits() + 1);
    assert!(below < -180.0);
    let w = crate::geo::projection::wrap180(below);
    assert!((-180.0..180.0).contains(&w), "wrap180({below}) = {w}");
}
