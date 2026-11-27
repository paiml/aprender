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
