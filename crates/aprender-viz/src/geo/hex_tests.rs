//! Tests for the hex grid (MAP-3, contract F-MAP-HEX-001).
//!
//! Layout used throughout: pointy-top hexes, circumradius `R`, cell `(q, r)` centred at
//! `(ox + R sqrt3 (q + r/2), oy + 1.5 R r)`. A cell is the hexagon of points within `R` of its
//! centre, with flat vertical sides at `x = cx +- sqrt3/2 R` and corners at `(cx, cy +- R)` and
//! `(cx +- sqrt3/2 R, cy +- R/2)`. Every expected area below is derived by hand from that
//! geometry and stated in a comment; none is read off the implementation under test.

use std::collections::{BTreeMap, BTreeSet};

use proptest::prelude::*;

use super::hex::{
    neighbours, polygon_area, rasterise, spread, AxialWindow, Cell, CellColumn, HexError, HexLayout,
};

/// The committed partly-cut fixture (see the `description` field inside the file).
const FIXTURE: &str = include_str!("testdata/hex_cut_fixture.json");

/// A counter-clockwise axis-aligned rectangle as one ring.
fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<(f64, f64)> {
    vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
}

/// Midpoint of two pixel points.
fn midpoint(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0)
}

/// Euclidean distance between two pixel points.
fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

/// Sum of the areas in a rasterisation result.
fn total(cells: &[(Cell, f64)]) -> f64 {
    cells.iter().map(|(_, a)| a).sum()
}

/// The committed fixture, decoded.
struct Fixture {
    origin: (f64, f64),
    radius: f64,
    window: AxialWindow,
    rings: Vec<Vec<(f64, f64)>>,
    /// Reference cells with `k`, meaning area `k * sqrt3 / 8`.
    reference: Vec<(Cell, f64)>,
}

fn load_fixture() -> Fixture {
    let v: serde_json::Value = serde_json::from_str(FIXTURE).expect("fixture is valid JSON");
    let w = &v["window"];
    let window = AxialWindow::new(
        w[0].as_i64().expect("q_min"),
        w[1].as_i64().expect("q_max"),
        w[2].as_i64().expect("r_min"),
        w[3].as_i64().expect("r_max"),
    )
    .expect("fixture window is valid");
    let rings = v["rings"]
        .as_array()
        .expect("rings")
        .iter()
        .map(|ring| {
            ring.as_array()
                .expect("ring")
                .iter()
                .map(|p| (p[0].as_f64().expect("x"), p[1].as_f64().expect("y")))
                .collect()
        })
        .collect();
    let reference = v["cells"]
        .as_array()
        .expect("cells")
        .iter()
        .map(|e| {
            let cell = Cell { q: e["q"].as_i64().expect("q"), r: e["r"].as_i64().expect("r") };
            (cell, e["k"].as_f64().expect("k"))
        })
        .collect();
    Fixture {
        origin: (v["origin"][0].as_f64().expect("ox"), v["origin"][1].as_f64().expect("oy")),
        radius: v["radius"].as_f64().expect("radius"),
        window,
        rings,
        reference,
    }
}

/// FALSIFY-MAP-HEX-001: picking the centre of a cell returns that cell.
///
/// The window spans negative and positive q and r. Layouts cover unit radius at the origin,
/// an offset origin with a small radius, and a large radius far from the origin.
#[test]
fn falsify_map_hex_001_pick_centre_round_trip() {
    let layouts = [
        HexLayout::new((0.0, 0.0), 1.0).expect("valid layout"),
        HexLayout::new((1.5, -2.0), 0.7).expect("valid layout"),
        HexLayout::new((-30.0, 12.5), 250.0).expect("valid layout"),
    ];
    let window = AxialWindow::new(-6, 6, -6, 6).expect("valid window");
    for layout in layouts {
        for c in window.cells() {
            assert_eq!(layout.pick(layout.center(c)), Ok(c), "centre of {c:?} must pick {c:?}");
        }
    }
}

/// FALSIFY-MAP-HEX-002: overlap areas sum to the polygon area.
///
/// Hand-derived areas: the triangle with vertices (-2,-1), (3,-1), (0.5,2.5) has base 5 and
/// height 3.5, so area 8.75. The L shape is the 4 x 1 bottom bar plus the 1 x 2 upright,
/// area 4 + 2 = 6. The square [-3,3] x [-2,2] minus the hole [-1,1] x [-1,1] is 24 - 4 = 20.
#[test]
fn falsify_map_hex_002_area_is_conserved() {
    let layout = HexLayout::new((0.0, 0.0), 1.0).expect("valid layout");
    let window = AxialWindow::new(-6, 6, -6, 6).expect("valid window");
    let cases: [(&str, Vec<Vec<(f64, f64)>>, f64); 3] = [
        ("convex triangle", vec![vec![(-2.0, -1.0), (3.0, -1.0), (0.5, 2.5)]], 8.75),
        (
            "concave L",
            vec![vec![(0.0, 0.0), (4.0, 0.0), (4.0, 1.0), (1.0, 1.0), (1.0, 3.0), (0.0, 3.0)]],
            6.0,
        ),
        ("square with hole", vec![rect(-3.0, -2.0, 3.0, 2.0), rect(-1.0, -1.0, 1.0, 1.0)], 20.0),
    ];
    for (name, rings, area) in cases {
        let got = total(&rasterise(&layout, &window, &rings).expect("rasterises"));
        assert!(
            (got - area).abs() <= 1e-9 * area,
            "{name}: cell areas sum to {got}, expected {area}"
        );
        let a = polygon_area(&rings).expect("polygon has area");
        assert!((a - area).abs() <= 1e-9 * area, "{name}: polygon_area {a}");
    }
}

/// FALSIFY-MAP-HEX-003: spread values sum back to the value, including partly covered cells.
#[test]
fn falsify_map_hex_003_spread_sums_to_value() {
    let layout = HexLayout::new((0.0, 0.0), 1.0).expect("valid layout");
    let fx = load_fixture();
    let polygons = [fx.rings.clone(), vec![vec![(-2.0, -1.0), (3.0, -1.0), (0.5, 2.5)]]];
    for rings in polygons {
        let areas = rasterise(&layout, &fx.window, &rings).expect("rasterises");
        for value in [7.5, -3.25, 0.0, 1e12] {
            let spread_values = spread(value, &areas).expect("spread");
            let sum: f64 = spread_values.iter().map(|(_, v)| v).sum();
            assert!((sum - value).abs() <= 1e-9 * value.abs(), "spread of {value} sums to {sum}");
        }
    }
}

/// FALSIFY-MAP-HEX-004: each cut-cell area matches the committed reference.
///
/// The reference lives in `testdata/hex_cut_fixture.json`, derived by hand there. Every
/// reached cell appears in the reference, and every reference cell is reached with area
/// `k * sqrt3 / 8`.
#[test]
fn falsify_map_hex_004_cut_cells_match_reference() {
    let fx = load_fixture();
    let layout = HexLayout::new(fx.origin, fx.radius).expect("fixture layout is valid");
    let got = rasterise(&layout, &fx.window, &fx.rings).expect("fixture rasterises");
    let unit = 3f64.sqrt() / 8.0;
    let tol = 1e-9 * 21.0 * unit;
    let got: BTreeMap<Cell, f64> = got.into_iter().collect();
    assert_eq!(got.len(), fx.reference.len(), "reached cell set differs");
    for (c, k) in &fx.reference {
        let a = got.get(c).copied().expect("reference cell is reached");
        assert!((a - k * unit).abs() <= tol, "cell {c:?}: got {a}, reference {}", k * unit);
    }
    let sum: f64 = got.values().sum();
    assert!((sum - 21.0 * unit).abs() <= tol, "polygon area {sum}");
}

/// FALSIFY-MAP-HEX-005: a point on a shared edge picks one of the two cells, the same each time.
///
/// The midpoint of two neighbouring centres lies on the edge they share, at the middle of it.
#[test]
fn falsify_map_hex_005_shared_edge_is_deterministic() {
    let layouts = [
        HexLayout::new((0.0, 0.0), 1.0).expect("valid layout"),
        HexLayout::new((1.5, -2.0), 0.7).expect("valid layout"),
    ];
    let pairs = [
        (Cell { q: 0, r: 0 }, Cell { q: 1, r: 0 }),
        (Cell { q: 0, r: 0 }, Cell { q: 0, r: 1 }),
        (Cell { q: 0, r: 0 }, Cell { q: -1, r: 1 }),
        (Cell { q: 2, r: -1 }, Cell { q: 2, r: 0 }),
        (Cell { q: -3, r: 2 }, Cell { q: -2, r: 1 }),
    ];
    for layout in layouts {
        for (a, b) in pairs {
            let p = midpoint(layout.center(a), layout.center(b));
            let first = layout.pick(p).expect("edge point is finite");
            assert!(first == a || first == b, "{p:?} picked {first:?}");
            for _ in 0..8 {
                assert_eq!(layout.pick(p), Ok(first), "repeated pick at {p:?}");
            }
        }
    }
}

/// Neighbour offsets, corner geometry and cell area.
#[test]
fn hex_neighbours_corners_and_area() {
    let layout = HexLayout::new((0.0, 0.0), 1.0).expect("valid layout");
    let s3 = 3f64.sqrt();
    let c = Cell { q: -2, r: 3 };
    let ns = neighbours(c);
    assert_eq!(BTreeSet::from(ns).len(), 6, "six distinct neighbours");
    for n in ns {
        let (dq, dr) = (n.q - c.q, n.r - c.r);
        let ds = -dq - dr;
        assert_eq!(dq.abs().max(dr.abs()).max(ds.abs()), 1, "{n:?} is adjacent");
        let d = dist(layout.center(c), layout.center(n));
        assert!((d - s3).abs() < 1e-12, "centre distance {d}");
    }
    // Hand: the corner at -30 degrees of the centre (0, 0) is (sqrt3/2, -1/2).
    let corners = layout.corners(Cell { q: 0, r: 0 });
    assert!((corners[0].0 - s3 / 2.0).abs() < 1e-12);
    assert!((corners[0].1 + 0.5).abs() < 1e-12);
    // Hand: the hexagon of circumradius 1 has area 3 sqrt3 / 2.
    assert!((layout.cell_area() - 1.5 * s3).abs() < 1e-12);
}

/// Layout validation: a radius that is not finite and positive, or a non-finite origin.
#[test]
fn hex_layout_rejects_bad_radius_and_origin() {
    for r in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(HexLayout::new((0.0, 0.0), r), Err(HexError::InvalidRadius(_))),
            "radius {r}"
        );
    }
    assert!(matches!(HexLayout::new((f64::NAN, 0.0), 1.0), Err(HexError::NonFinite { .. })));
    let layout = HexLayout::new((0.0, 0.0), 1.0).expect("valid layout");
    assert!(matches!(layout.pick((f64::NAN, 0.0)), Err(HexError::NonFinite { .. })));
    assert!(matches!(layout.pick((1e300, 0.0)), Err(HexError::OutOfRange { .. })));
}

/// Polygon validation: too few vertices, non-finite vertices, zero area, no rings.
#[test]
fn hex_rasterise_rejects_bad_polygons() {
    let layout = HexLayout::new((0.0, 0.0), 1.0).expect("valid layout");
    let window = AxialWindow::new(-6, 6, -6, 6).expect("valid window");
    assert_eq!(
        rasterise(&layout, &window, &[vec![(0.0, 0.0), (1.0, 0.0)]]),
        Err(HexError::RingTooShort { ring: 0, vertices: 2 })
    );
    let nan_ring = vec![vec![(0.0, 0.0), (f64::NAN, 0.0), (1.0, 1.0)]];
    assert_eq!(
        rasterise(&layout, &window, &nan_ring),
        Err(HexError::NonFiniteVertex { ring: 0, vertex: 1 })
    );
    let collinear = vec![vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]];
    assert_eq!(rasterise(&layout, &window, &collinear), Err(HexError::ZeroArea));
    assert_eq!(rasterise(&layout, &window, &[]), Err(HexError::ZeroArea));
}

/// A polygon reaching outside the window, and a polygon that needs too many candidate cells.
#[test]
fn hex_rasterise_rejects_window_and_size() {
    let layout = HexLayout::new((0.0, 0.0), 1.0).expect("valid layout");
    let small = AxialWindow::new(0, 1, 0, 1).expect("valid window");
    let big = vec![rect(-3.0, -3.0, 3.0, 3.0)];
    assert!(matches!(rasterise(&layout, &small, &big), Err(HexError::OutsideWindow { .. })));
    // Hand: a 10 x 10 square at R = 1e-3 spans about 5774 x 6667 axial cells, about 3.8e7,
    // which is above the 4_000_000 candidate limit.
    let tiny = HexLayout::new((0.0, 0.0), 1e-3).expect("valid layout");
    let window = AxialWindow::new(-1, 1, -1, 1).expect("valid window");
    let square = vec![rect(0.0, 0.0, 10.0, 10.0)];
    assert!(matches!(rasterise(&tiny, &window, &square), Err(HexError::TooManyCells { .. })));
}

/// Spread and window/column validation.
#[test]
fn hex_spread_and_columns_reject_bad_inputs() {
    let c = Cell { q: 0, r: 0 };
    assert_eq!(spread(1.0, &[(c, 0.0)]), Err(HexError::ZeroArea));
    assert!(matches!(spread(f64::NAN, &[(c, 1.0)]), Err(HexError::NonFinite { .. })));
    assert_eq!(spread(1e13, &[(c, 1.0)]), Err(HexError::ValueOutOfRange { value: 1e13 }));
    assert_eq!(spread(1.0, &[(c, f64::NAN)]), Err(HexError::InvalidArea { index: 0 }));
    assert!(matches!(AxialWindow::new(1, 0, 0, 0), Err(HexError::InvalidWindow { .. })));

    let w = AxialWindow::new(-2, 2, -3, 1).expect("valid window");
    assert_eq!(w.len(), 25);
    let mut seen = vec![false; w.len()];
    for cell in w.cells() {
        let i = w.index_of(cell).expect("window cell has an index");
        assert!(!seen[i], "index {i} used twice");
        seen[i] = true;
    }
    assert!(seen.iter().all(|&s| s));
    assert_eq!(w.index_of(Cell { q: 3, r: 0 }), None);

    let mut col = CellColumn::zeros(w);
    col.add(Cell { q: -2, r: -3 }, 1.5).expect("in window");
    col.add(Cell { q: -2, r: -3 }, 0.5).expect("in window");
    assert_eq!(col.get(Cell { q: -2, r: -3 }), Some(2.0));
    assert_eq!(col.get(Cell { q: 0, r: 0 }), Some(0.0));
    assert_eq!(col.get(Cell { q: 3, r: 0 }), None);
    assert_eq!(col.values().len(), 25);
    assert_eq!(col.add(Cell { q: 0, r: 2 }, 1.0), Err(HexError::OutsideWindow { q: 0, r: 2 }));
}

proptest! {
    /// `pick` returns a cell whose centre is no farther from `p` than any neighbour's centre.
    #[test]
    fn pick_returns_nearest_centre(
        px in -25.0f64..25.0,
        py in -25.0f64..25.0,
        ox in -5.0f64..5.0,
        oy in -5.0f64..5.0,
        radius in 0.25f64..8.0,
    ) {
        let layout = HexLayout::new((ox, oy), radius).expect("valid layout");
        let p = (px, py);
        let c = layout.pick(p).expect("finite point");
        let d = dist(p, layout.center(c));
        for n in neighbours(c) {
            let dn = dist(p, layout.center(n));
            prop_assert!(d <= dn + 1e-9 * radius, "picked {c:?} at {d}, {n:?} at {dn}");
        }
    }

    /// Random triangles: cell areas sum to the triangle area and spread values sum to the value.
    #[test]
    fn random_triangles_conserve_area_and_value(
        ax in -3.0f64..3.0, ay in -3.0f64..3.0,
        bx in -3.0f64..3.0, by in -3.0f64..3.0,
        cx in -3.0f64..3.0, cy in -3.0f64..3.0,
        value in -1e6f64..1e6,
    ) {
        let area = 0.5 * ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)).abs();
        prop_assume!(area > 1e-3);
        let layout = HexLayout::new((0.0, 0.0), 1.0).expect("valid layout");
        let window = AxialWindow::new(-8, 8, -8, 8).expect("valid window");
        let rings = vec![vec![(ax, ay), (bx, by), (cx, cy)]];
        let cells = rasterise(&layout, &window, &rings).expect("triangle rasterises");
        prop_assert!((total(&cells) - area).abs() <= 1e-9 * area);
        let spread_values = spread(value, &cells).expect("spread");
        let sum: f64 = spread_values.iter().map(|(_, v)| v).sum();
        prop_assert!((sum - value).abs() <= 1e-9 * value.abs());
    }
}
