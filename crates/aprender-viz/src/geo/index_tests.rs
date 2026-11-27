//! Tests for the spatial index (MAP-3 phase 2).
//!
//! The oracles are brute-force scans over every polygon or point. The polygon oracle is
//! `oracle_contains`, a predicate written independently of `index::rings_contain`: it casts its
//! ray along +y where the index casts along +x, and it has its own closed-edge test. A fault in
//! the index predicate therefore shows up as a disagreement, not as agreement with itself.
//! Generated coordinates are multiples of 1/8, and edge midpoints are multiples of 1/16, so every
//! point placed on an edge is exact.

use proptest::prelude::*;

use super::index::{IndexError, PointIndex, PolygonIndex, Rings};

/// Counter-clockwise axis-aligned rectangle as one ring.
fn square(x0: f64, y0: f64, x1: f64, y1: f64) -> Rings {
    vec![vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]
}

/// Independent point-in-polygon oracle over all rings of one polygon.
///
/// A point on any closed edge is inside. Otherwise the answer is the even-odd parity of edges
/// crossed by a ray from `p` towards +y, using a half-open rule on the x-span of each edge.
fn oracle_contains(rings: &[Vec<(f64, f64)>], p: (f64, f64)) -> bool {
    let on_edge = rings.iter().any(|ring| {
        (0..ring.len()).any(|i| on_closed_edge(ring[i], ring[(i + 1) % ring.len()], p))
    });
    if on_edge {
        return true;
    }
    let mut crossings = 0_usize;
    for ring in rings {
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            let straddles = (a.0 <= p.0 && p.0 < b.0) || (b.0 <= p.0 && p.0 < a.0);
            if straddles && a.1 + (p.0 - a.0) * (b.1 - a.1) / (b.0 - a.0) > p.1 {
                crossings += 1;
            }
        }
    }
    crossings % 2 == 1
}

/// Closed segment test by collinearity (`(b-a) x (p-a) == 0`) and the segment's own bounds.
fn on_closed_edge(a: (f64, f64), b: (f64, f64), p: (f64, f64)) -> bool {
    let collinear = (b.0 - a.0) * (p.1 - a.1) == (b.1 - a.1) * (p.0 - a.0);
    let in_x = a.0.min(b.0) <= p.0 && p.0 <= a.0.max(b.0);
    let in_y = a.1.min(b.1) <= p.1 && p.1 <= a.1.max(b.1);
    collinear && in_x && in_y
}

/// Brute-force pick: the lowest id of a polygon that contains `p`.
fn brute_pick(polys: &[Rings], p: (f64, f64)) -> Option<usize> {
    polys.iter().position(|rings| oracle_contains(rings, p))
}

/// Brute-force nearest: the lowest id at minimum squared distance.
fn brute_nearest(points: &[(f64, f64)], q: (f64, f64)) -> Option<usize> {
    let mut best: Option<(f64, usize)> = None;
    for (i, &pt) in points.iter().enumerate() {
        let dx = pt.0 - q.0;
        let dy = pt.1 - q.1;
        let d2 = dx * dx + dy * dy;
        match best {
            Some((b, _)) if d2 >= b => {}
            _ => best = Some((d2, i)),
        }
    }
    best.map(|(_, i)| i)
}

#[test]
fn pick_inside_and_outside_square() {
    let index = PolygonIndex::build(&[square(0.0, 0.0, 2.0, 2.0)]).expect("valid");
    assert_eq!(index.pick((1.0, 1.0)), Ok(Some(0)));
    assert_eq!(index.pick((3.0, 1.0)), Ok(None));
}

#[test]
fn pick_edge_and_vertex_count_as_inside() {
    let index = PolygonIndex::build(&[square(0.0, 0.0, 2.0, 2.0)]).expect("valid");
    for p in [(0.0, 1.0), (1.0, 0.0), (2.0, 2.0), (0.0, 0.0), (2.0, 1.5)] {
        assert_eq!(index.pick(p), Ok(Some(0)), "boundary point {p:?}");
    }
    assert_eq!(index.pick((2.0, 2.5)), Ok(None));
}

#[test]
fn pick_hole_is_outside_and_its_edge_is_inside() {
    let mut holed = square(0.0, 0.0, 4.0, 4.0);
    holed.push(vec![(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]);
    let index = PolygonIndex::build(&[holed]).expect("valid");
    assert_eq!(index.pick((0.5, 0.5)), Ok(Some(0)));
    assert_eq!(index.pick((2.0, 2.0)), Ok(None), "hole interior");
    assert_eq!(index.pick((1.0, 2.0)), Ok(Some(0)), "hole edge counts as inside");
}

#[test]
fn pick_concave_l_shape_notch_is_outside() {
    // L shape: the notch x in (1,4], y in (1,4] is outside.
    let l = vec![vec![(0.0, 0.0), (4.0, 0.0), (4.0, 1.0), (1.0, 1.0), (1.0, 4.0), (0.0, 4.0)]];
    let index = PolygonIndex::build(&[l]).expect("valid");
    assert_eq!(index.pick((3.0, 3.0)), Ok(None), "notch");
    assert_eq!(index.pick((0.5, 3.0)), Ok(Some(0)), "left arm");
    assert_eq!(index.pick((3.0, 0.5)), Ok(Some(0)), "bottom arm");
}

#[test]
fn pick_lowest_id_on_overlap_and_shared_edge() {
    let polys =
        [square(0.0, 0.0, 2.0, 2.0), square(1.0, 0.0, 3.0, 2.0), square(2.0, 0.0, 4.0, 2.0)];
    let index = PolygonIndex::build(&polys).expect("valid");
    assert_eq!(index.pick((1.5, 1.0)), Ok(Some(0)), "overlap of 0 and 1");
    assert_eq!(index.pick((2.0, 1.0)), Ok(Some(0)), "shared edge of 0, 1, 2");
    assert_eq!(index.pick((2.5, 1.0)), Ok(Some(1)), "overlap of 1 and 2");
}

#[test]
fn pick_far_outside_extent_is_none() {
    let index = PolygonIndex::build(&[square(0.0, 0.0, 1.0, 1.0)]).expect("valid");
    assert_eq!(index.pick((1e9, 1e9)), Ok(None));
    assert_eq!(index.pick((-1e9, 0.5)), Ok(None));
}

#[test]
fn zero_width_polygon_is_degenerate_but_works() {
    let line = vec![vec![(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)]];
    let index = PolygonIndex::build(&[line]).expect("valid");
    assert_eq!(index.pick((0.0, 1.0)), Ok(Some(0)), "on the line");
    assert_eq!(index.pick((1.0, 1.0)), Ok(None));
}

#[test]
fn empty_indexes_return_none() {
    assert_eq!(PolygonIndex::build(&[]).expect("valid").pick((0.0, 0.0)), Ok(None));
    assert_eq!(PointIndex::build(&[]).expect("valid").nearest((0.0, 0.0)), Ok(None));
}

#[test]
fn non_finite_polygon_coordinate_is_build_error() {
    let nan = vec![vec![(0.0, 0.0), (1.0, 0.0), (f64::NAN, 1.0)]];
    assert!(matches!(PolygonIndex::build(&[nan]), Err(IndexError::NonFinite { .. })));
    let inf = vec![vec![(0.0, 0.0), (f64::INFINITY, 0.0), (1.0, 1.0)]];
    assert!(matches!(PolygonIndex::build(&[inf]), Err(IndexError::NonFinite { .. })));
}

#[test]
fn short_ring_is_build_error_naming_the_polygon() {
    let short = vec![vec![(0.0, 0.0), (1.0, 1.0)]];
    let err = PolygonIndex::build(&[square(0.0, 0.0, 1.0, 1.0), short])
        .expect_err("a two-vertex ring must be refused");
    assert_eq!(err, IndexError::RingTooShort { polygon: 1, ring: 0, vertices: 2 });
}

#[test]
fn non_finite_query_is_typed_error_not_panic() {
    let index = PolygonIndex::build(&[square(0.0, 0.0, 1.0, 1.0)]).expect("valid");
    assert!(matches!(index.pick((f64::NAN, 0.0)), Err(IndexError::NonFinite { .. })));
    let points = PointIndex::build(&[(0.0, 0.0), (1.0, 1.0)]).expect("valid");
    assert!(matches!(points.nearest((0.0, f64::INFINITY)), Err(IndexError::NonFinite { .. })));
}

#[test]
fn non_finite_point_is_build_error() {
    assert!(matches!(
        PointIndex::build(&[(0.0, 0.0), (f64::NAN, 1.0)]),
        Err(IndexError::NonFinite { .. })
    ));
}

#[test]
fn nearest_hand_case() {
    let index = PointIndex::build(&[(0.0, 0.0), (3.0, 4.0), (6.0, 8.0)]).expect("valid");
    assert_eq!(index.nearest((3.0, 3.0)), Ok(Some(1)));
}

#[test]
fn nearest_ties_return_lowest_id() {
    let index = PointIndex::build(&[(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0)]).expect("valid");
    assert_eq!(index.nearest((0.0, 0.0)), Ok(Some(0)));
}

#[test]
fn nearest_all_identical_points() {
    let index = PointIndex::build(&[(2.0, 2.0); 5]).expect("valid");
    assert_eq!(index.nearest((0.0, 0.0)), Ok(Some(0)));
    assert_eq!(index.nearest((2.0, 2.0)), Ok(Some(0)));
}

#[test]
fn nearest_zero_width_extent() {
    let index = PointIndex::build(&[(3.0, 0.0), (3.0, 7.0), (3.0, 2.0)]).expect("valid");
    // Squared distances from (0,5): 34, 13, 18.
    assert_eq!(index.nearest((0.0, 5.0)), Ok(Some(1)));
}

#[test]
fn nearest_far_outside_extent() {
    let index = PointIndex::build(&[(0.0, 0.0), (10.0, 0.0)]).expect("valid");
    assert_eq!(index.nearest((1e9, 0.0)), Ok(Some(1)));
    assert_eq!(index.nearest((-1e9, 0.0)), Ok(Some(0)));
}

/// A dyadic coordinate: a multiple of 1/16 in `[-10, 10]`.
fn dyadic16(k: i32) -> f64 {
    f64::from(k) / 16.0
}

/// Snap to the nearest multiple of 1/8.
fn snap8(v: f64) -> f64 {
    (v * 8.0).round() / 8.0
}

/// A star-shaped ring about `(cx, cy)`: vertices at increasing angles, so concave shapes occur.
fn star(c: (f64, f64), radii: &[u8]) -> Vec<(f64, f64)> {
    let n = radii.len();
    (0..n)
        .map(|k| {
            let t = std::f64::consts::TAU * k as f64 / n as f64;
            let r = f64::from(radii[k]) / 8.0;
            (snap8(c.0 + r * t.cos()), snap8(c.1 + r * t.sin()))
        })
        .collect()
}

/// A random star polygon, optionally with a hole star at half the radii.
fn polygon() -> impl Strategy<Value = Rings> {
    (-32i32..=32, -32i32..=32, prop::collection::vec(1u8..=64, 3..=8), any::<bool>()).prop_map(
        |(cx, cy, radii, holed)| {
            let c = (f64::from(cx) / 8.0, f64::from(cy) / 8.0);
            let mut rings = vec![star(c, &radii)];
            if holed {
                let inner: Vec<u8> = radii.iter().map(|r| (r / 2).max(1)).collect();
                rings.push(star(c, &inner));
            }
            rings
        },
    )
}

/// Vertices and edge midpoints of every ring: points exactly on the boundary.
fn boundary_points(polys: &[Rings]) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    for rings in polys {
        for ring in rings {
            for (i, &a) in ring.iter().enumerate() {
                let b = ring[(i + 1) % ring.len()];
                out.push(a);
                out.push(((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0));
            }
        }
    }
    out
}

/// Far-outside query points.
fn far_points() -> Vec<(f64, f64)> {
    vec![(1e9, 1e9), (-1e9, 0.5), (0.0, -1e7), (1e12, -1e12)]
}

proptest! {
    #[test]
    fn pick_matches_brute_force(
        polys in prop::collection::vec(polygon(), 1..=6),
        raw in prop::collection::vec((-160i32..=160, -160i32..=160), 1..=30),
    ) {
        let index = PolygonIndex::build(&polys).expect("generated rings are valid");
        let mut queries: Vec<(f64, f64)> =
            raw.iter().map(|&(x, y)| (dyadic16(x), dyadic16(y))).collect();
        queries.extend(boundary_points(&polys));
        queries.extend(far_points());
        for p in queries {
            prop_assert_eq!(index.pick(p), Ok(brute_pick(&polys, p)), "point {:?}", p);
            prop_assert_eq!(index.pick(p), index.pick(p), "repeat call on {:?}", p);
        }
    }

    #[test]
    fn nearest_matches_brute_force(
        raw in prop::collection::vec((-160i32..=160, -160i32..=160), 1..=60),
        qraw in prop::collection::vec((-160i32..=160, -160i32..=160), 1..=30),
    ) {
        let pts: Vec<(f64, f64)> = raw.iter().map(|&(x, y)| (dyadic16(x), dyadic16(y))).collect();
        let index = PointIndex::build(&pts).expect("finite points");
        let mut queries: Vec<(f64, f64)> =
            qraw.iter().map(|&(x, y)| (dyadic16(x), dyadic16(y))).collect();
        queries.extend(pts.iter().copied());
        queries.extend(far_points());
        for q in queries {
            prop_assert_eq!(index.nearest(q), Ok(brute_nearest(&pts, q)), "query {:?}", q);
        }
    }

    #[test]
    fn nearest_ties_match_brute_force(
        raw in prop::collection::vec((-3i32..=3, -3i32..=3), 1..=40),
    ) {
        // A tiny coordinate range gives many duplicate points and exact distance ties.
        let pts: Vec<(f64, f64)> = raw.iter().map(|&(x, y)| (f64::from(x), f64::from(y))).collect();
        let index = PointIndex::build(&pts).expect("finite points");
        for x in -4..=4 {
            for y in -4..=4 {
                let q = (f64::from(x) / 2.0, f64::from(y) / 2.0);
                prop_assert_eq!(index.nearest(q), Ok(brute_nearest(&pts, q)), "query {:?}", q);
            }
        }
    }
}
