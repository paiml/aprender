//! Tests for the hex grid. The oracle derives centres from the layout written in hex.rs and does
//! not call `HexGrid::centre` or `HexGrid::cell_at_world`, so it shares no code with them.

use super::*;

const DELTA: f64 = 1e-6;
const EPS: f64 = 1e-9;

/// Neighbour offsets `(dcol, drow)` for an even row, in the order E, NE, NW, W, SW, SE.
const EVEN_ROW_NEIGHBOURS: [(i32, i32); 6] = [(1, 0), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)];
/// Neighbour offsets `(dcol, drow)` for an odd row, in the same order.
const ODD_ROW_NEIGHBOURS: [(i32, i32); 6] = [(1, 0), (1, -1), (0, -1), (-1, 0), (0, 1), (1, 1)];

fn neighbours_of(row: i32) -> [(i32, i32); 6] {
    if row.rem_euclid(2) == 0 {
        EVEN_ROW_NEIGHBOURS
    } else {
        ODD_ROW_NEIGHBOURS
    }
}

/// Centre of `(col, row)` in the layout of hex.rs, with signed coordinates so the oracle can
/// address the lattice outside the grid.
fn lattice_centre(col: i32, row: i32, r: f64) -> (f64, f64) {
    let odd = f64::from(row.rem_euclid(2));
    (
        3f64.sqrt() * r * (f64::from(col) + 0.5 * odd),
        1.5 * r * f64::from(row),
    )
}

fn grid_id(grid: &HexGrid, col: i32, row: i32) -> Option<u32> {
    let c = u32::try_from(col).ok()?;
    let r = u32::try_from(row).ok()?;
    grid.id_of(c, r)
}

fn signed(grid: &HexGrid, id: u32) -> (i32, i32) {
    let (c, r) = grid.coords_of(id).expect("id is on the grid");
    (
        i32::try_from(c).expect("col fits i32"),
        i32::try_from(r).expect("row fits i32"),
    )
}

fn close(a: (f64, f64), b: (f64, f64)) -> bool {
    (a.0 - b.0).abs() <= EPS && (a.1 - b.1).abs() <= EPS
}

/// The two points either side of the edge shared by a cell at `c` and its neighbour at `n`:
/// `inside` on the cell's side, `outside` on the neighbour's side, `DELTA * r` from the edge
/// midpoint along the centre-to-centre direction.
fn edge_points(c: (f64, f64), n: (f64, f64), r: f64) -> ((f64, f64), (f64, f64)) {
    let (dx, dy) = (n.0 - c.0, n.1 - c.1);
    let len = dx.hypot(dy);
    let (ux, uy) = (dx / len, dy / len);
    let mid = ((c.0 + n.0) / 2.0, (c.1 + n.1) / 2.0);
    let d = DELTA * r;
    (
        (mid.0 - d * ux, mid.1 - d * uy),
        (mid.0 + d * ux, mid.1 + d * uy),
    )
}

/// Brute force: nearest centre over an extended lattice two rings wider than the grid, first
/// minimum in row-major order. The point is on the grid iff that centre is a cell of the grid.
fn oracle(grid: &HexGrid, p: (f64, f64)) -> Option<u32> {
    let cols = i32::try_from(grid.cols()).expect("cols fit i32");
    let rows = i32::try_from(grid.rows()).expect("rows fit i32");
    let r = grid.radius();
    let mut best: Option<(f64, i32, i32)> = None;
    for row in -2..rows + 2 {
        for col in -2..cols + 2 {
            let c = lattice_centre(col, row, r);
            let d2 = (c.0 - p.0).powi(2) + (c.1 - p.1).powi(2);
            if best.is_none_or(|(b, _, _)| d2 < b) {
                best = Some((d2, col, row));
            }
        }
    }
    let (_, col, row) = best?;
    grid_id(grid, col, row)
}

/// Deterministic xorshift-free LCG in `[0, 1)`, so the random sample is the same on every run.
fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let top = u32::try_from(*state >> 40).expect("24 bits fit in u32");
    f64::from(top) / 16_777_216.0
}

#[test]
fn map_constants_name_the_117_by_90_grid() {
    assert_eq!(MAP_COLS, 117);
    assert_eq!(MAP_ROWS, 90);
    let grid = HexGrid::map_default(1.0).expect("map grid");
    assert_eq!(grid.cols(), 117);
    assert_eq!(grid.rows(), 90);
    assert_eq!(grid.cell_count(), 10_530);
    assert_eq!(grid.radius(), 1.0);
}

#[test]
fn new_rejects_empty_grid_bad_radius_and_overflowing_cell_count() {
    assert_eq!(HexGrid::new(0, 5, 1.0), Err(MapError::OutOfRange("grid")));
    assert_eq!(HexGrid::new(5, 0, 1.0), Err(MapError::OutOfRange("grid")));
    assert_eq!(
        HexGrid::new(u32::MAX, 2, 1.0),
        Err(MapError::OutOfRange("grid"))
    );
    assert_eq!(
        HexGrid::new(5, 5, 0.0),
        Err(MapError::NonPositive("radius"))
    );
    assert_eq!(
        HexGrid::new(5, 5, f64::NAN),
        Err(MapError::NonFinite("radius"))
    );
    assert_eq!(
        HexGrid::new(5, 5, 1e13),
        Err(MapError::OutOfRange("radius"))
    );
    assert!(HexGrid::new(5, 5, 1.0).is_ok());
}

#[test]
fn id_round_trips_through_coords_for_every_cell_and_rejects_outside() {
    let grid = HexGrid::map_default(1.0).expect("map grid");
    for id in 0..grid.cell_count() {
        let (c, r) = grid.coords_of(id).expect("on grid");
        assert_eq!(grid.id_of(c, r), Some(id));
    }
    assert_eq!(grid.id_of(1, 2), Some(2 * 117 + 1), "row-major ids");
    assert_eq!(grid.coords_of(grid.cell_count()), None);
    assert_eq!(grid.id_of(MAP_COLS, 0), None);
    assert_eq!(grid.id_of(0, MAP_ROWS), None);
}

#[test]
fn centre_follows_pointy_top_layout_and_odd_row_offset() {
    let grid = HexGrid::new(10, 10, 2.0).expect("grid");
    let s3 = 3f64.sqrt();
    let at = |c: u32, r: u32| {
        grid.centre(grid.id_of(c, r).expect("on grid"))
            .expect("centre")
    };
    assert!(close(at(0, 0), (0.0, 0.0)));
    assert!(close(at(1, 0), (2.0 * s3, 0.0)), "column pitch sqrt(3) R");
    assert!(
        close(at(0, 1), (s3, 3.0)),
        "odd row offset half a column, row pitch 1.5 R"
    );
    assert!(close(at(1, 1), (3.0 * s3, 3.0)));
    assert!(close(at(0, 2), (0.0, 6.0)));
}

#[test]
fn centre_outside_the_grid_is_none() {
    let grid = HexGrid::new(4, 3, 1.0).expect("grid");
    assert_eq!(grid.centre(12), None);
    assert!(grid.centre(11).is_some());
}

#[test]
fn centre_picks_back_to_its_own_cell_for_every_cell_of_the_map() {
    for radius in [1.0, 2.5] {
        let grid = HexGrid::map_default(radius).expect("map grid");
        for id in 0..grid.cell_count() {
            let centre = grid.centre(id).expect("centre");
            assert_eq!(
                grid.cell_at_world(centre),
                Some(id),
                "cell {id} at R={radius}"
            );
        }
    }
}

#[test]
fn points_far_outside_or_not_finite_are_none() {
    let grid = HexGrid::map_default(1.0).expect("map grid");
    assert_eq!(grid.cell_at_world((1e11, 0.0)), None);
    assert_eq!(grid.cell_at_world((-1e11, -1e11)), None);
    assert_eq!(grid.cell_at_world((f64::NAN, 0.0)), None);
    assert_eq!(grid.cell_at_world((0.0, f64::INFINITY)), None);
}

#[test]
fn oracle_agrees_on_every_cell_each_edge_and_random_points_of_a_small_grid() {
    let grid = HexGrid::new(12, 9, 1.0).expect("grid");
    let r = grid.radius();
    let mut points = Vec::new();
    for id in 0..grid.cell_count() {
        let (col, row) = signed(&grid, id);
        let c = lattice_centre(col, row, r);
        points.push(c);
        for (dc, dr) in neighbours_of(row) {
            let n = lattice_centre(col + dc, row + dr, r);
            let dist = (n.0 - c.0).hypot(n.1 - c.1);
            assert!((dist - 3f64.sqrt() * r).abs() < EPS, "neighbour table");
            let (inside, outside) = edge_points(c, n, r);
            points.push(inside);
            points.push(outside);
        }
    }
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let (w, h) = (
        f64::from(grid.cols()) * 3f64.sqrt(),
        1.5 * f64::from(grid.rows()),
    );
    for _ in 0..400 {
        let x = -2.0 + unit(&mut state) * (w + 4.0);
        let y = -2.0 + unit(&mut state) * (h + 4.0);
        points.push((x, y));
    }
    for p in points {
        assert_eq!(grid.cell_at_world(p), oracle(&grid, p), "point {p:?}");
    }
}

#[test]
fn every_edge_of_every_cell_of_the_map_picks_the_cell_inside_and_the_neighbour_outside() {
    let grid = HexGrid::map_default(1.0).expect("map grid");
    for id in 0..grid.cell_count() {
        let (col, row) = signed(&grid, id);
        let c = lattice_centre(col, row, 1.0);
        for (dc, dr) in neighbours_of(row) {
            let n = lattice_centre(col + dc, row + dr, 1.0);
            let (inside, outside) = edge_points(c, n, 1.0);
            assert_eq!(grid.cell_at_world(inside), Some(id), "inside, cell {id}");
            assert_eq!(
                grid.cell_at_world(outside),
                grid_id(&grid, col + dc, row + dr),
                "outside, cell {id} toward {dc},{dr}"
            );
        }
    }
}

#[test]
fn points_near_each_vertex_agree_with_the_oracle() {
    // Vertices sit at circumradius R in the directions 30 + 60k degrees. A plain per-axis round
    // goes wrong near them; the cube fix-up must not. Probe at 0.98 R, inside the hexagon.
    let grid = HexGrid::new(12, 9, 1.0).expect("grid");
    for id in 0..grid.cell_count() {
        let (col, row) = signed(&grid, id);
        let c = lattice_centre(col, row, 1.0);
        for k in 0..6 {
            let angle = std::f64::consts::FRAC_PI_6 + f64::from(k) * std::f64::consts::FRAC_PI_3;
            let p = (c.0 + 0.98 * angle.cos(), c.1 + 0.98 * angle.sin());
            assert_eq!(
                grid.cell_at_world(p),
                oracle(&grid, p),
                "vertex probe {id} k={k}"
            );
        }
    }
}
