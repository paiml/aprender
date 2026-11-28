//! Tests for map plots through the grammar API: a polygon with a hole, a projection refusal.
//!
//! Geometry is chosen so every pixel is derivable by hand. The test projection is the identity
//! (`x = lon`, `y = lat`), so the outer ring `[0, 10] x [0, 10]` and the hole `[4, 6] x [4, 6]`
//! stay in degrees. The plot is 180 x 180 with the grey theme's margin of 40, so the plot area is
//! 100 x 100 pixels at the origin `(40, 40)`. The fit scales the 10 x 10 box by 10 px per unit,
//! north up: pixel `(px, py)` maps to `u = (px - 40) / 10`, `v = 10 - (py - 40) / 10`.

use super::plot::{fit_to_area, GeoProjection, POLYGON_FILL};
use super::projection::{GeoError, Mercator, Projection};
use crate::color::Rgba;
use crate::error::Error;
use crate::grammar::{Coord, Facet, GGPlot, Geom};

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

#[test]
fn maps_are_drawn_north_up_and_east_right() {
    // A north-west square [0,2]x[8,10] and a south-east square [8,10]x[0,2]: no mirror or flip
    // of this pair looks like itself. Pixel (50, 50) is u = 1.05, v = 8.95.
    let features = vec![
        vec![vec![(0.0, 8.0), (2.0, 8.0), (2.0, 10.0), (0.0, 10.0)]],
        vec![vec![(8.0, 0.0), (10.0, 0.0), (10.0, 2.0), (8.0, 2.0)]],
    ];
    let built = map_plot(Coord::geo(Identity), features).build().expect("builds");
    let fb = built.to_framebuffer().expect("draws");
    assert_eq!(fb.get_pixel(50, 50), Some(POLYGON_FILL), "north-west is top left");
    assert_eq!(fb.get_pixel(130, 130), Some(POLYGON_FILL), "south-east is bottom right");
    assert_ne!(fb.get_pixel(50, 130), Some(POLYGON_FILL), "south-west is empty");
    assert_ne!(fb.get_pixel(130, 50), Some(POLYGON_FILL), "north-east is empty");
}

#[test]
fn coord_geo_with_facets_is_refused() {
    let built =
        map_plot(Coord::geo(Identity), square_with_hole()).facet(Facet::wrap("region", 2)).build();
    let err = match built {
        Ok(plot) => plot.to_framebuffer().expect_err("facets are not drawn under coord_geo"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedLayer("coord_geo with facets")), "got {err:?}");
}

#[test]
fn a_degenerate_ring_is_a_rendering_error() {
    let features = vec![vec![vec![(0.0, 0.0), (1.0, 1.0)]]];
    let built = map_plot(Coord::geo(Identity), features).build().expect("builds");
    let err = built.to_framebuffer().expect_err("two vertices do not bound an area");
    assert!(matches!(err, Error::Rendering(_)), "got {err:?}");
}

#[test]
fn fit_of_no_vertices_gives_no_rings() {
    assert!(fit_to_area(&Vec::new(), (0.0, 0.0, 100.0, 100.0)).is_empty());
}

/// Claim: `fit_to_area` silently drops a 0-vertex ring instead of reporting DegenerateRing.
#[test]
fn fit_keeps_a_zero_vertex_ring_as_an_empty_ring() {
    let plane = vec![vec![Vec::new(), square(0.0, 10.0)]];
    let fitted = fit_to_area(&plane, (0.0, 0.0, 100.0, 100.0));
    println!(
        "FIT_RINGS input={} output={:?}",
        plane[0].len(),
        fitted[0].iter().map(Vec::len).collect::<Vec<_>>()
    );
    assert_eq!(fitted.len(), 1, "feature dropped");
    assert_eq!(fitted[0].len(), 2, "ring dropped");
    assert!(fitted[0][0].is_empty());
}

#[test]
fn fit_of_a_flat_box_uses_the_other_axis_and_centres() {
    let plane = vec![vec![vec![(0.0, 3.0), (10.0, 3.0)]]];
    let fitted = fit_to_area(&plane, (0.0, 0.0, 100.0, 100.0));
    assert_eq!(fitted, vec![vec![vec![(0.0, 50.0), (100.0, 50.0)]]]);
}

#[test]
fn fit_of_a_single_point_collapses_to_the_centre() {
    let plane = vec![vec![vec![(7.0, 3.0), (7.0, 3.0)]]];
    let fitted = fit_to_area(&plane, (10.0, 20.0, 100.0, 60.0));
    assert_eq!(fitted, vec![vec![vec![(60.0, 50.0), (60.0, 50.0)]]]);
}

#[test]
fn geo_projection_debug_names_only_the_type() {
    assert_eq!(format!("{:?}", GeoProjection::new(Identity)), "GeoProjection");
    assert_eq!(GeoProjection::new(Identity).projection().inverse(1.0, 2.0), Ok((1.0, 2.0)));
}

// ---------------------------------------------------------------------------
// Review round 1 (MAP-1): spans that overflow f64 still fit finitely
// ---------------------------------------------------------------------------

#[test]
fn fit_of_an_extreme_span_stays_finite_and_inside_the_area() {
    // x1 - x0 and y1 - y0 are 2e308, which overflow to infinity. The fit works on halved values,
    // so every output point is finite and inside the 400 x 300 area at (10, 20). The scale is
    // 300 / 1e308 per halved unit, so the box is 300 px wide and centred at x = 60 .. 360.
    let plane = vec![vec![vec![(-1e308, -1e308), (1e308, 1e308), (-1e308, 1e308)]]];
    let out = fit_to_area(&plane, (10.0, 20.0, 400.0, 300.0));
    let points: Vec<(f64, f64)> = out.iter().flatten().flatten().copied().collect();
    assert_eq!(points.len(), 3);
    for (x, y) in points {
        assert!(x.is_finite() && y.is_finite(), "({x}, {y})");
        assert!((10.0..=410.0).contains(&x) && (20.0..=320.0).contains(&y), "({x}, {y})");
    }
}
