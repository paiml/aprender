//! Tests for `geom_hex` (`geo` feature): one pointy-top hexagon per cell, filled from its value.
//!
//! The layout is `HexLayout::new((0, 0), 10)`, so cell `(q, r)` is centred at
//! `(10 sqrt3 (q + r/2), 15 r)`. The fixture cells are `(0, 0) = 0`, `(1, 0) = 5` and `(0, 1) = 10`.
//! The plot is 180 x 180 with the grey theme's margin of 40, so the plot area is 100 x 100.

use crate::color::Rgba;
use crate::error::Error;
use crate::geo::hex::{Cell, HexLayout};
use crate::grammar::{GGPlot, Geom};
use crate::scale::{ColorScale, Scale};

/// The committed SVG of the fixture. Regenerate only with a reviewed change to the geometry.
const GOLDEN: &str = include_str!("../geo/testdata/hex_golden.svg");

fn fixture_cells() -> Vec<(Cell, f64)> {
    vec![(Cell { q: 0, r: 0 }, 0.0), (Cell { q: 1, r: 0 }, 5.0), (Cell { q: 0, r: 1 }, 10.0)]
}

fn hex_plot(cells: Vec<(Cell, f64)>) -> GGPlot {
    let layout = HexLayout::new((0.0, 0.0), 10.0).expect("valid layout");
    GGPlot::new().geom(Geom::hex(layout, cells)).dimensions(180, 180)
}

fn hex_svg(cells: Vec<(Cell, f64)>) -> String {
    hex_plot(cells).build().expect("builds").to_svg().expect("renders")
}

/// The `fill="..."` value of each `<path`, in document order.
fn path_fills(svg: &str) -> Vec<String> {
    svg.split("<path")
        .skip(1)
        .map(|p| {
            let start = p.find("fill=\"").expect("every hex path has a fill") + 6;
            let end = start + p[start..].find('"').expect("fill is quoted");
            p[start..end].to_string()
        })
        .collect()
}

/// The colour `ColorScale::viridis((0, 10))` gives value `v`, as the plot maps it.
fn expected_colour(v: f32) -> Rgba {
    ColorScale::viridis((0.0, 10.0)).expect("valid domain").scale(v)
}

#[test]
fn hex_svg_matches_committed_golden() {
    assert_eq!(hex_svg(fixture_cells()), GOLDEN);
}

#[test]
fn hex_svg_has_one_path_per_cell_and_one_fill_per_distinct_value() {
    let svg = hex_svg(fixture_cells());
    assert_eq!(svg.matches("<path").count(), 3, "three cells are three hexagon paths");
    let mut fills = path_fills(&svg);
    assert_eq!(fills.len(), 3);
    fills.sort();
    fills.dedup();
    assert_eq!(fills.len(), 3, "values 0, 5 and 10 are three distinct fills");
}

#[test]
fn hex_svg_fills_are_the_scale_colours_of_the_values() {
    let fills = path_fills(&hex_svg(fixture_cells()));
    assert_eq!(fills.len(), 3, "a blank SVG must not pass the colour check");
    let want = [expected_colour(0.0), expected_colour(5.0), expected_colour(10.0)];
    for (fill, colour) in fills.iter().zip(want) {
        let (r, g, b) = (colour.r, colour.g, colour.b);
        assert_eq!(fill, &format!("rgb({r},{g},{b})"));
    }
}

#[test]
fn one_cell_value_change_alters_the_svg() {
    let before = hex_svg(fixture_cells());
    let mut changed = fixture_cells();
    changed[1].1 = 6.0;
    let after = hex_svg(changed);
    assert_ne!(before, after, "a one-cell value change must change the output");
    assert_eq!(after.matches("<path").count(), 3, "the cell count does not change");
}

#[test]
fn hex_framebuffer_paints_each_value_colour_inside_the_plot_area() {
    let fb = hex_plot(fixture_cells()).build().expect("builds").to_framebuffer().expect("draws");
    let mut seen: Vec<Rgba> = Vec::new();
    for y in 40..140 {
        for x in 40..140 {
            if let Some(c) = fb.get_pixel(x, y) {
                if !seen.contains(&c) {
                    seen.push(c);
                }
            }
        }
    }
    for v in [0.0_f32, 5.0, 10.0] {
        assert!(seen.contains(&expected_colour(v)), "no pixel painted for value {v}");
    }
}

#[test]
fn hex_fit_keeps_the_hexagon_aspect_ratio() {
    let svg = hex_svg(fixture_cells());
    let start = svg.find(" d=\"").expect("a path") + 4;
    let d = &svg[start..start + svg[start..].find('"').expect("quoted d")];
    let nums: Vec<f64> = d
        .split(|c: char| c == 'M' || c == 'L' || c == 'Z' || c == ' ')
        .filter(|t| !t.is_empty())
        .map(|t| t.parse::<f64>().expect("numeric coordinate"))
        .collect();
    let xs: Vec<f64> = nums.iter().step_by(2).copied().collect();
    let ys: Vec<f64> = nums.iter().skip(1).step_by(2).copied().collect();
    let span = |v: &[f64]| {
        v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - v.iter().copied().fold(f64::INFINITY, f64::min)
    };
    // A pointy-top hexagon is sqrt3 * R wide and 2 * R tall.
    let ratio = span(&xs) / span(&ys);
    assert!((ratio - 3f64.sqrt() / 2.0).abs() < 0.01, "width/height {ratio}");
}

#[test]
fn all_equal_values_give_one_documented_colour_and_no_panic() {
    let cells = vec![(Cell { q: 0, r: 0 }, 2.0), (Cell { q: 1, r: 0 }, 2.0)];
    let fills = path_fills(&hex_svg(cells));
    assert_eq!(fills.len(), 2);
    assert_eq!(fills[0], fills[1], "equal values share the midpoint colour");
}

#[test]
fn empty_cell_list_is_a_typed_error() {
    let err = hex_plot(Vec::new()).build().expect("builds").to_framebuffer().expect_err("refused");
    assert!(matches!(err, Error::ScaleDomain(_)), "got {err:?}");
}

#[test]
fn non_finite_value_is_a_typed_error() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let cells = vec![(Cell { q: 0, r: 0 }, 1.0), (Cell { q: 1, r: 0 }, bad)];
        let err = hex_plot(cells).build().expect("builds").to_svg().expect_err("refused");
        assert!(matches!(err, Error::ScaleDomain(_)), "{bad}: got {err:?}");
    }
}
