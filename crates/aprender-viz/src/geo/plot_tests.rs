//! Tests for map plots through the grammar API: a polygon with a hole, a projection refusal.
//!
//! Geometry is chosen so every pixel is derivable by hand. The test projection is the identity
//! (`x = lon`, `y = lat`), so the outer ring `[0, 10] x [0, 10]` and the hole `[4, 6] x [4, 6]`
//! stay in degrees. The plot is 180 x 180 with the grey theme's margin of 40, so the plot area is
//! 100 x 100 pixels at the origin `(40, 40)`. The fit scales the 10 x 10 box by 10 px per unit,
//! north up: pixel `(px, py)` maps to `u = (px - 40) / 10`, `v = 10 - (py - 40) / 10`.

use super::plot::POLYGON_FILL;
use super::projection::{GeoError, Mercator, Projection};
use crate::color::Rgba;
use crate::error::Error;
use crate::grammar::{Coord, GGPlot, Geom};

/// The identity projection, used only as a test double: `(lon, lat)` in, `(lon, lat)` out.
struct Identity;

impl Projection for Identity {
    fn forward(&self, lon_deg: f64, lat_deg: f64) -> Result<(f64, f64), GeoError> {
        Ok((lon_deg, lat_deg))
    }

    fn inverse(&self, x: f64, y: f64) -> Result<(f64, f64), GeoError> {
        Ok((x, y))
    }
}

fn square(lo: f64, hi: f64) -> Vec<(f64, f64)> {
    vec![(lo, lo), (hi, lo), (hi, hi), (lo, hi)]
}

/// One feature: the outer square `[0, 10]^2` with the hole `[4, 6]^2`.
fn square_with_hole() -> Vec<Vec<Vec<(f64, f64)>>> {
    vec![vec![square(0.0, 10.0), square(4.0, 6.0)]]
}

/// The plot with the square-with-hole under `coord`, sized so the plot area is 100 x 100.
fn map_plot(coord: Coord, features: Vec<Vec<Vec<(f64, f64)>>>) -> GGPlot {
    GGPlot::new().geom(Geom::polygon(features)).coord(coord).dimensions(180, 180)
}

#[test]
fn polygon_hole_is_unfilled_and_ring_is_filled_in_the_framebuffer() {
    let built = map_plot(Coord::geo(Identity), square_with_hole()).build().expect("builds");
    let fb = built.to_framebuffer().expect("draws");
    // Pixel (90, 90): centre (90.5, 90.5) maps to u = 5.05, v = 4.95, inside the hole.
    // The hole is not painted, so the panel background shows through.
    let panel = Rgba::rgb(235, 235, 235);
    assert_eq!(fb.get_pixel(90, 90), Some(panel), "hole pixel must stay unfilled");
    // Pixel (50, 90): centre (50.5, 90.5) maps to u = 1.05, v = 4.95, in the outer ring only.
    assert_eq!(fb.get_pixel(50, 90), Some(POLYGON_FILL), "ring pixel must be filled");
}

#[test]
fn polygon_emits_evenodd_svg_with_one_subpath_per_ring() {
    let built = map_plot(Coord::geo(Identity), square_with_hole()).build().expect("builds");
    let svg = built.to_svg().expect("renders");
    assert!(svg.contains(r#"fill-rule="evenodd""#), "SVG must carry the even-odd fill rule");
    // Two rings are two closed subpaths; the separator between them is "Z M" exactly once.
    assert_eq!(svg.matches("Z M").count(), 1, "outer ring and hole are separate subpaths");
}

#[test]
fn projection_domain_error_is_returned_not_skipped() {
    // Mercator refuses |lat| > 85. The vertex at lat 89 must fail the whole plot.
    let mercator = Mercator::new(0.0, 1.0).expect("valid Mercator parameters");
    let features = vec![vec![square(0.0, 10.0)], vec![vec![(0.0, 89.0), (1.0, 0.0), (2.0, 0.0)]]];
    let built = map_plot(Coord::geo(mercator), features).build().expect("builds");
    let err = built.to_framebuffer().expect_err("a refused vertex must fail the plot");
    assert!(matches!(err, Error::ScaleDomain(_)), "got {err:?}");
}

#[test]
fn non_polygon_layer_under_coord_geo_is_refused() {
    let built = GGPlot::new()
        .geom(Geom::point())
        .coord(Coord::geo(Identity))
        .dimensions(180, 180)
        .build()
        .expect("builds");
    let err = built.to_framebuffer().expect_err("only geom_polygon is drawn under coord_geo");
    assert!(matches!(err, Error::UnsupportedLayer(_)), "got {err:?}");
}
