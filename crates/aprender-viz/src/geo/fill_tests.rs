//! Tests for even-odd polygon fill and its SVG path form.
//!
//! Every expected pixel and every golden byte is derived by hand in the comments. None is read
//! off the implementation under test. A pixel `(x, y)` is filled when its centre `(x + 0.5,
//! y + 0.5)` lies inside an odd number of rings.

use super::fill::{fill_polygon_even_odd, svg_path_d, FillError};
use crate::color::Rgba;
use crate::framebuffer::Framebuffer;
use crate::output::SvgEncoder;

/// The committed golden SVG for the three-shape fixture.
const GOLDEN: &str = include_str!("testdata/three_shapes.svg");

type Ring = Vec<(f64, f64)>;

/// A 10 by 10 white canvas.
fn canvas() -> Framebuffer {
    let mut fb = Framebuffer::new(10, 10).expect("10x10 is a valid size");
    fb.clear(Rgba::WHITE);
    fb
}

/// Corners of an axis-aligned square, counter-clockwise from the bottom-left.
fn square(x0: f64, y0: f64, x1: f64, y1: f64) -> Ring {
    vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
}

/// Whether a pixel holds the fill colour rather than the white background.
fn is_filled(fb: &Framebuffer, x: u32, y: u32) -> bool {
    fb.get_pixel(x, y) == Some(Rgba::RED)
}

/// Assert, for every pixel of the 10 by 10 canvas, that it is filled exactly when `expect`
/// says so.
fn assert_pixels(fb: &Framebuffer, expect: impl Fn(u32, u32) -> bool) {
    for y in 0..10 {
        for x in 0..10 {
            assert_eq!(is_filled(fb, x, y), expect(x, y), "pixel ({x},{y})");
        }
    }
}

/// The three-shape fixture. `apex_x` is the x of the triangle's third vertex.
fn three_shapes(apex_x: f64) -> String {
    let shapes: [(Vec<Ring>, Rgba); 3] = [
        (vec![square(10.0, 10.0, 40.0, 40.0), square(20.0, 20.0, 30.0, 30.0)], Rgba::RED),
        (vec![vec![(50.0, 10.0), (90.0, 10.0), (apex_x, 45.0)]], Rgba::GREEN),
        (vec![vec![(10.5, 60.0), (45.0, 55.25), (60.0, 90.0), (12.0, 95.0)]], Rgba::BLUE),
    ];
    let mut svg = SvgEncoder::new(100, 100).background(None);
    for (rings, color) in &shapes {
        let d = svg_path_d(rings).expect("fixture rings are valid");
        svg = svg.path_even_odd(&d, *color);
    }
    svg.render()
}

// ---- framebuffer: even-odd fill with a hole -------------------------------------------------

#[test]
fn hole_pixel_unfilled_and_ring_pixel_filled() {
    let mut fb = canvas();
    let rings = vec![square(1.0, 1.0, 9.0, 9.0), square(3.0, 3.0, 7.0, 7.0)];
    fill_polygon_even_odd(&mut fb, &rings, Rgba::RED).expect("valid rings");

    // Pixel (5,5) has centre (5.5,5.5), inside the hole [3,7)x[3,7): unfilled.
    assert!(!is_filled(&fb, 5, 5));
    // Pixel (2,5) has centre (2.5,5.5), inside the outer [1,9)^2 and outside the hole: filled.
    assert!(is_filled(&fb, 2, 5));
    assert!(is_filled(&fb, 8, 8));
    // Pixel (9,9) has centre (9.5,9.5), outside the outer square.
    assert!(!is_filled(&fb, 9, 9));
    assert_pixels(&fb, |x, y| {
        (1..9).contains(&x) && (1..9).contains(&y) && !((3..7).contains(&x) && (3..7).contains(&y))
    });
}

#[test]
fn filled_polygon_without_hole_fills_its_whole_square() {
    let mut fb = canvas();
    fill_polygon_even_odd(&mut fb, &[square(2.0, 2.0, 5.0, 4.0)], Rgba::RED).expect("valid");
    // Centres in [2,5) x [2,4): columns 2..=4, rows 2..=3.
    assert_pixels(&fb, |x, y| (2..5).contains(&x) && (2..4).contains(&y));
}

#[test]
fn clips_partly_outside_polygon_to_the_framebuffer() {
    let mut fb = canvas();
    // Centres below 3 are inside [-5,3)^2, so columns and rows 0..=2 fill.
    fill_polygon_even_odd(&mut fb, &[square(-5.0, -5.0, 3.0, 3.0)], Rgba::RED).expect("valid");
    assert_pixels(&fb, |x, y| x <= 2 && y <= 2);
}

#[test]
fn clips_polygon_running_off_the_bottom_right() {
    let mut fb = canvas();
    // Centres at or above 8 are inside [8,100)^2, so columns and rows 8..=9 fill.
    fill_polygon_even_odd(&mut fb, &[square(8.0, 8.0, 100.0, 100.0)], Rgba::RED).expect("valid");
    assert_pixels(&fb, |x, y| x >= 8 && y >= 8);
}

#[test]
fn polygon_entirely_off_canvas_fills_nothing() {
    let mut fb = canvas();
    fill_polygon_even_odd(&mut fb, &[square(20.0, 20.0, 30.0, 30.0)], Rgba::RED).expect("valid");
    // Rows above the canvas.
    fill_polygon_even_odd(&mut fb, &[square(0.0, -30.0, 10.0, -20.0)], Rgba::RED).expect("valid");
    // Off the right edge: rows are on the canvas but every span clips to nothing.
    fill_polygon_even_odd(&mut fb, &[square(20.0, 0.0, 30.0, 10.0)], Rgba::RED).expect("valid");
    assert_pixels(&fb, |_, _| false);
}

#[test]
fn huge_coordinates_clip_without_panicking() {
    let mut fb = canvas();
    fill_polygon_even_odd(&mut fb, &[square(-1e9, -1e9, 1e9, 1e9)], Rgba::RED).expect("valid");
    assert_pixels(&fb, |_, _| true);
}

#[test]
fn empty_input_is_a_no_op() {
    let mut fb = canvas();
    fill_polygon_even_odd(&mut fb, &[], Rgba::RED).expect("no rings is valid");
    assert_pixels(&fb, |_, _| false);
}

#[test]
fn degenerate_ring_is_refused_before_any_pixel_is_written() {
    let mut fb = canvas();
    let rings = vec![square(1.0, 1.0, 5.0, 5.0), vec![(0.0, 0.0), (5.0, 5.0)]];
    let err = fill_polygon_even_odd(&mut fb, &rings, Rgba::RED).expect_err("two vertices");
    assert_eq!(err, FillError::DegenerateRing { ring: 1, vertices: 2 });
    assert_pixels(&fb, |_, _| false);
}

#[test]
fn nan_coordinate_is_a_typed_error() {
    let mut fb = canvas();
    let ring = vec![(0.0, 0.0), (f64::NAN, 1.0), (5.0, 5.0)];
    let err = fill_polygon_even_odd(&mut fb, &[ring], Rgba::RED).expect_err("NaN x");
    assert_eq!(err, FillError::NonFinite { ring: 0, vertex: 1 });
    assert_pixels(&fb, |_, _| false);
}

#[test]
fn infinite_coordinate_is_a_typed_error() {
    let mut fb = canvas();
    let ring = vec![(0.0, 0.0), (1.0, f64::INFINITY), (5.0, 5.0)];
    let err = fill_polygon_even_odd(&mut fb, &[ring], Rgba::RED).expect_err("infinite y");
    assert_eq!(err, FillError::NonFinite { ring: 0, vertex: 1 });
}

#[test]
fn overflowing_crossing_is_a_typed_error() {
    let mut fb = canvas();
    // The first edge spans 2e308, which overflows to infinity, so its crossing at y = 0.5 is
    // not finite.
    let ring = vec![(-1e308, 0.0), (1e308, 1.0), (0.0, 5.0)];
    let err = fill_polygon_even_odd(&mut fb, &[ring], Rgba::RED).expect_err("overflow");
    assert_eq!(err, FillError::NonFiniteResult { ring: 0 });
}

// ---- SVG path data -------------------------------------------------------------------------

#[test]
fn svg_path_d_writes_closed_subpaths() {
    let d = svg_path_d(&[vec![(50.0, 10.0), (90.0, 10.0), (70.0, 45.0)]]).expect("valid");
    assert_eq!(d, "M50 10L90 10L70 45Z");
}

#[test]
fn svg_path_d_keeps_fractions_and_normalises_negative_zero() {
    let d = svg_path_d(&[vec![(10.5, 60.0), (45.0, 55.25), (60.0, 90.0), (12.0, 95.0)]])
        .expect("valid");
    assert_eq!(d, "M10.5 60L45 55.25L60 90L12 95Z");

    let zero = svg_path_d(&[vec![(-0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]]).expect("valid");
    assert_eq!(zero, "M0 0L1 0L0 1Z");
}

#[test]
fn svg_path_d_joins_rings_with_a_space() {
    let d = svg_path_d(&[square(1.0, 1.0, 9.0, 9.0), square(3.0, 3.0, 7.0, 7.0)]).expect("valid");
    assert_eq!(d, "M1 1L9 1L9 9L1 9Z M3 3L7 3L7 7L3 7Z");
}

#[test]
fn svg_path_d_of_no_rings_is_empty() {
    assert_eq!(svg_path_d(&[]).expect("valid"), "");
}

#[test]
fn svg_path_d_refuses_degenerate_and_non_finite_rings() {
    let short = svg_path_d(&[vec![(0.0, 0.0)]]).expect_err("one vertex");
    assert_eq!(short, FillError::DegenerateRing { ring: 0, vertices: 1 });
    let nan = svg_path_d(&[vec![(0.0, f64::NAN), (1.0, 0.0), (0.0, 1.0)]]).expect_err("NaN");
    assert_eq!(nan, FillError::NonFinite { ring: 0, vertex: 0 });
}

#[test]
fn svg_element_carries_both_subpaths_and_evenodd() {
    let d = svg_path_d(&[square(1.0, 1.0, 9.0, 9.0), square(3.0, 3.0, 7.0, 7.0)]).expect("valid");
    let svg = SvgEncoder::new(10, 10).background(None).path_even_odd(&d, Rgba::RED).render();
    assert!(
        svg.contains(
            r#"<path d="M1 1L9 1L9 9L1 9Z M3 3L7 3L7 7L3 7Z" fill="rgb(255,0,0)" fill-rule="evenodd"/>"#
        ),
        "{svg}"
    );
}

// ---- golden SVG ----------------------------------------------------------------------------

#[test]
fn three_shapes_match_the_golden_file_byte_for_byte() {
    assert_eq!(three_shapes(70.0), GOLDEN);
}

#[test]
fn changing_one_coordinate_changes_the_golden_output() {
    let changed = three_shapes(71.0);
    assert_ne!(changed, GOLDEN);
    assert!(changed.contains(r#"<path d="M50 10L90 10L71 45Z""#), "{changed}");
}

#[test]
fn fill_error_messages_name_the_ring_and_vertex() {
    let degenerate = FillError::DegenerateRing { ring: 2, vertices: 1 };
    assert_eq!(degenerate.to_string(), "ring 2 has 1 vertices; a ring needs at least 3");
    let non_finite = FillError::NonFinite { ring: 0, vertex: 4 };
    assert_eq!(non_finite.to_string(), "ring 0 vertex 4 is not finite");
    let overflow = FillError::NonFiniteResult { ring: 1 };
    assert_eq!(overflow.to_string(), "ring 1: an edge crossing is not finite");
}
