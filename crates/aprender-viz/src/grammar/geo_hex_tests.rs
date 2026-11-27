//! Tests for `geom_hex` (`geo` feature): one pointy-top hexagon per cell, filled from its value.
//!
//! The layout is `HexLayout::new((0, 0), 10)`, so cell `(q, r)` is centred at
//! `(10 sqrt3 (q + r/2), 15 r)`. The fixture cells are `(0, 0) = 0`, `(1, 0) = 5` and `(0, 1) = 10`.
//! The plot is 180 x 180 with the grey theme's margin of 40, so the plot area is 100 x 100.

use crate::color::Rgba;
use crate::error::Error;
use crate::geo::hex::{rasterise, AxialWindow, Cell, HexLayout};
use crate::geo::projection::{GeoError, Projection};
use crate::grammar::{Coord, GGPlot, Geom};
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
    // One row of three cells on the square panel: the bounding box is about 52 x 20, so a fit that
    // scaled x and y independently would stretch the hexagons to about 0.33 instead of sqrt3 / 2.
    let row =
        vec![(Cell { q: 0, r: 0 }, 0.0), (Cell { q: 1, r: 0 }, 5.0), (Cell { q: 2, r: 0 }, 10.0)];
    let svg = hex_svg(row);
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
    assert!((ratio - 3f64.sqrt() / 2.0).abs() < 1e-3, "width/height {ratio}");
}

#[test]
fn all_equal_values_give_one_documented_colour_and_no_panic() {
    let cells = vec![(Cell { q: 0, r: 0 }, 2.0), (Cell { q: 1, r: 0 }, 2.0)];
    let fills = path_fills(&hex_svg(cells));
    assert_eq!(fills.len(), 2);
    // Documented: with no range, every hexagon takes the midpoint of the viridis scale, that is
    // its colour at t = 0.5. That colour is #21918c, which is (33, 145, 140).
    let mid = ColorScale::viridis((0.0, 1.0)).expect("valid domain").scale(0.5);
    assert_eq!((mid.r, mid.g, mid.b), (33, 145, 140), "viridis midpoint is #21918c");
    let want = "rgb(33,145,140)".to_string();
    assert_eq!(fills, vec![want.clone(), want], "every hexagon takes the midpoint colour");
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

/// The `d` attribute of each `<path`, in document order.
fn path_ds(svg: &str) -> Vec<String> {
    svg.split(" d=\"").skip(1).map(|p| p[..p.find('"').expect("quoted d")].to_string()).collect()
}

/// The vertices of a path `d`, in order, as `(x, y)` pairs.
fn vertices(d: &str) -> Vec<(f64, f64)> {
    let nums: Vec<f64> = d
        .split(|c: char| matches!(c, 'M' | 'L' | 'Z' | ' '))
        .filter(|t| !t.is_empty())
        .map(|t| t.parse::<f64>().expect("numeric coordinate"))
        .collect();
    nums.chunks_exact(2).map(|p| (p[0], p[1])).collect()
}

/// The mean pixel y of a path's vertices: the centre of a hexagon.
fn mean_y(d: &str) -> f64 {
    let v = vertices(d);
    v.iter().map(|p| p.1).sum::<f64>() / v.len() as f64
}

/// The identity projection, a test double: `(lon, lat)` in, `(lon, lat)` out.
struct Identity;

impl Projection for Identity {
    fn forward(&self, lon_deg: f64, lat_deg: f64) -> Result<(f64, f64), GeoError> {
        Ok((lon_deg, lat_deg))
    }
    fn inverse(&self, x: f64, y: f64) -> Result<(f64, f64), GeoError> {
        Ok((x, y))
    }
}

#[test]
fn larger_plane_y_is_drawn_higher() {
    let ds = path_ds(&hex_svg(fixture_cells()));
    assert_eq!(ds.len(), 3);
    // Cell (0, 0) sits at plane y 0 and cell (0, 1) at plane y 15, so (0, 1) is drawn higher.
    assert!(
        mean_y(&ds[2]) < mean_y(&ds[0]),
        "plane y 15 must land on a smaller pixel row than plane y 0"
    );
}

#[test]
fn hex_map_orients_like_polygon_map() {
    let layout = HexLayout::new((0.0, 0.0), 2.0).expect("valid layout");
    let window = AxialWindow::new(-10, 10, -10, 10).expect("valid window");
    // Apex (5, 10) is the highest plane point; base (0, 0) and (10, 0) the lowest.
    let tri = vec![(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)];

    let poly = GGPlot::new()
        .geom(Geom::polygon(vec![vec![tri.clone()]]))
        .coord(Coord::geo(Identity))
        .dimensions(180, 180)
        .build()
        .expect("builds");
    let poly_ds = path_ds(&poly.to_svg().expect("renders"));
    let pv = vertices(&poly_ds[0]);
    assert!(pv[2].1 < pv[0].1, "polygon map: apex must be drawn above the base, got {pv:?}");

    // The same triangle rasterised to cells, each valued by its plane centre y.
    let cells = rasterise(&layout, &window, &[tri]).expect("rasterises");
    let valued: Vec<(Cell, f64)> = cells.iter().map(|&(c, _)| (c, layout.center(c).1)).collect();
    let top =
        (0..valued.len()).max_by(|&a, &b| valued[a].1.total_cmp(&valued[b].1)).expect("cells");
    let bottom =
        (0..valued.len()).min_by(|&a, &b| valued[a].1.total_cmp(&valued[b].1)).expect("cells");
    let hex_ds = path_ds(&hex_svg(valued));
    assert!(
        mean_y(&hex_ds[top]) < mean_y(&hex_ds[bottom]),
        "hex map: the highest-plane cell must be drawn above the lowest, as in the polygon map"
    );
}

/// Builds the plot and asserts that rendering refuses it, returning the error.
fn refused(plot: GGPlot) -> Error {
    plot.build().expect("builds").to_svg().expect_err("refused")
}

/// The fixture hex layer, as a layer of `plot`.
fn with_hex(plot: GGPlot) -> GGPlot {
    let layout = HexLayout::new((0.0, 0.0), 10.0).expect("valid layout");
    plot.geom(Geom::hex(layout, fixture_cells()))
}

#[test]
fn flipped_cartesian_coord_is_refused() {
    let coord = Coord::Cartesian { xlim: None, ylim: None, flip: true };
    let err = refused(with_hex(GGPlot::new().coord(coord).dimensions(180, 180)));
    assert!(matches!(err, Error::UnsupportedLayer(_)), "flip: got {err:?}");
}

#[test]
fn cartesian_axis_limits_are_refused() {
    // Limits are not applied to hexagons, so drawing them would silently ignore the request.
    for coord in [
        Coord::Cartesian { xlim: Some((0.0, 5.0)), ylim: None, flip: false },
        Coord::Cartesian { xlim: None, ylim: Some((0.0, 5.0)), flip: false },
    ] {
        let err = refused(with_hex(GGPlot::new().coord(coord.clone()).dimensions(180, 180)));
        assert!(matches!(err, Error::UnsupportedLayer(_)), "{coord:?}: got {err:?}");
    }
}

#[test]
fn non_cartesian_coords_are_refused() {
    let coords = [
        Coord::geo(Identity),
        Coord::Polar { start: 0.0, direction: 1 },
        Coord::Fixed { ratio: 1.0 },
    ];
    for coord in coords {
        let err = refused(with_hex(GGPlot::new().coord(coord).dimensions(180, 180)));
        assert!(matches!(err, Error::UnsupportedLayer(_)), "got {err:?}");
    }
}

#[test]
fn faceted_hex_plot_is_refused() {
    let facet = crate::grammar::Facet::wrap("lineage", 2);
    let err = refused(with_hex(GGPlot::new().facet(facet).dimensions(180, 180)));
    assert!(matches!(err, Error::UnsupportedLayer(_)), "got {err:?}");
}

#[test]
fn hex_beside_another_layer_is_refused_in_either_order() {
    let tri = vec![(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)];
    let poly = || Geom::polygon(vec![vec![tri.clone()]]);
    let layout = HexLayout::new((0.0, 0.0), 10.0).expect("valid layout");
    let hex = || Geom::hex(layout, fixture_cells());
    let hex_first = GGPlot::new().geom(hex()).geom(poly()).dimensions(180, 180);
    let poly_first = GGPlot::new().geom(poly()).geom(hex()).dimensions(180, 180);
    for plot in [hex_first, poly_first] {
        let err = refused(plot);
        assert!(matches!(err, Error::UnsupportedLayer(_)), "got {err:?}");
    }
}

#[test]
fn two_hex_layers_are_refused() {
    let layout = HexLayout::new((0.0, 0.0), 10.0).expect("valid layout");
    let plot = GGPlot::new()
        .geom(Geom::hex(layout, fixture_cells()))
        .geom(Geom::hex(layout, fixture_cells()))
        .dimensions(180, 180);
    let err = refused(plot);
    assert!(matches!(err, Error::UnsupportedLayer(_)), "got {err:?}");
}

#[test]
fn values_beyond_the_f32_range_are_refused() {
    for big in [1e39, -1e39] {
        let cells = vec![(Cell { q: 0, r: 0 }, 0.0), (Cell { q: 1, r: 0 }, big)];
        let err = refused(hex_plot(cells));
        assert!(matches!(err, Error::ScaleDomain(_)), "{big}: got {err:?}");
    }
}

#[test]
fn values_too_close_to_colour_apart_are_refused() {
    // Distinct as f64, equal as f32: the scale cannot tell them apart.
    let cells = vec![(Cell { q: 0, r: 0 }, 1.0), (Cell { q: 1, r: 0 }, 1.0 + 1e-12)];
    let err = refused(hex_plot(cells));
    assert!(matches!(err, Error::ScaleDomain(_)), "got {err:?}");
}
