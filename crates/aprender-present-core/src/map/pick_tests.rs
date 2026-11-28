//! Tests for screen picking through the camera. Edge probes use the same neighbour table as
//! hex_tests.rs, restricted to interior cells so every neighbour lies on the grid.

use super::*;
use crate::map::hex::{HexGrid, MAP_COLS, MAP_ROWS};

const SCALES: [f64; 4] = [0.5, 3.0, 17.0, 120.0];
const PANS: [(f64, f64); 3] = [(0.0, 0.0), (-37.3, 12.9), (1234.5, -987.25)];
const DELTA: f64 = 1e-6;

const EVEN_ROW_NEIGHBOURS: [(i32, i32); 6] = [(1, 0), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)];
const ODD_ROW_NEIGHBOURS: [(i32, i32); 6] = [(1, 0), (1, -1), (0, -1), (-1, 0), (0, 1), (1, 1)];

fn camera_for(scale: f64, pan: (f64, f64)) -> Camera2D {
    let mut cam = Camera2D::new((0.0, 0.0), scale, (800.0, 600.0)).expect("camera");
    cam.pan(pan.0, pan.1).expect("pan");
    cam
}

fn map_grid() -> HexGrid {
    HexGrid::map_default(1.0).expect("map grid")
}

/// Id of the cell at signed `(col, row)`, which must be on the grid.
fn id_at(grid: &HexGrid, col: i32, row: i32) -> u32 {
    let c = u32::try_from(col).expect("col on grid");
    let r = u32::try_from(row).expect("row on grid");
    grid.id_of(c, r).expect("on grid")
}

#[test]
fn pick_inverts_screen_at_for_every_cell_of_the_map_at_several_scales_and_pans() {
    let grid = map_grid();
    for scale in SCALES {
        for pan in PANS {
            let cam = camera_for(scale, pan);
            for id in 0..grid.cell_count() {
                let screen = cam.screen_at(grid.centre(id).expect("centre"));
                assert_eq!(
                    pick(&grid, &cam, screen),
                    Ok(Some(id)),
                    "cell {id} at scale {scale} pan {pan:?}"
                );
            }
        }
    }
}

/// Neighbour offsets for the row parity of `row`.
fn neighbour_offsets(row: i32) -> [(i32, i32); 6] {
    if row % 2 == 0 {
        EVEN_ROW_NEIGHBOURS
    } else {
        ODD_ROW_NEIGHBOURS
    }
}

/// Points a fixed distance either side of the edge between the centres `c` and `n`, on the cell
/// side (`inside`) and the neighbour side (`outside`).
fn edge_probes(c: (f64, f64), n: (f64, f64)) -> ((f64, f64), (f64, f64)) {
    let (dx, dy) = (n.0 - c.0, n.1 - c.1);
    let len = dx.hypot(dy);
    let (ux, uy) = (dx / len, dy / len);
    let mid = ((c.0 + n.0) / 2.0, (c.1 + n.1) / 2.0);
    (
        (mid.0 - DELTA * ux, mid.1 - DELTA * uy),
        (mid.0 + DELTA * ux, mid.1 + DELTA * uy),
    )
}

/// Checks both probes of every edge of one interior cell under `cam`.
fn check_cell_edges(grid: &HexGrid, cam: &Camera2D, col: u32, row: u32) {
    let id = grid.id_of(col, row).expect("on grid");
    let c = grid.centre(id).expect("centre");
    let (ci, ri) = (
        i32::try_from(col).expect("fits"),
        i32::try_from(row).expect("fits"),
    );
    for (dc, dr) in neighbour_offsets(ri) {
        let nid = id_at(grid, ci + dc, ri + dr);
        let (inside, outside) = edge_probes(c, grid.centre(nid).expect("centre"));
        let at = |p: (f64, f64)| cam.screen_at(p);
        assert_eq!(
            pick(grid, cam, at(inside)),
            Ok(Some(id)),
            "inside edge of {id} toward {dc},{dr} at scale {}",
            cam.scale()
        );
        assert_eq!(
            pick(grid, cam, at(outside)),
            Ok(Some(nid)),
            "outside edge of {id} toward {dc},{dr} at scale {}",
            cam.scale()
        );
    }
}

#[test]
fn a_point_just_inside_each_edge_picks_the_cell_and_just_outside_picks_the_neighbour() {
    let grid = map_grid();
    for scale in SCALES {
        for pan in PANS {
            let cam = camera_for(scale, pan);
            for row in (1..MAP_ROWS - 1).step_by(7) {
                for col in (1..MAP_COLS - 1).step_by(7) {
                    check_cell_edges(&grid, &cam, col, row);
                }
            }
        }
    }
}

#[test]
fn the_camera_centre_cell_is_picked_at_the_screen_centre() {
    let grid = map_grid();
    for id in [0, 1, 117, 5_000, grid.cell_count() - 1] {
        let centre = grid.centre(id).expect("centre");
        let cam = Camera2D::new(centre, 40.0, (800.0, 600.0)).expect("camera");
        assert_eq!(pick(&grid, &cam, cam.screen_centre()), Ok(Some(id)));
    }
}

#[test]
fn a_pixel_off_the_map_picks_nothing() {
    let grid = map_grid();
    let cam = camera_for(10.0, (0.0, 0.0));
    let far = cam.screen_at((1e6, 1e6));
    assert_eq!(pick(&grid, &cam, far), Ok(None));
}

#[test]
fn a_pixel_that_is_not_finite_is_refused() {
    let grid = map_grid();
    let cam = camera_for(10.0, (0.0, 0.0));
    assert_eq!(
        pick(&grid, &cam, (f64::NAN, 0.0)),
        Err(MapError::NonFinite("screen"))
    );
    assert_eq!(
        pick(&grid, &cam, (0.0, f64::INFINITY)),
        Err(MapError::NonFinite("screen"))
    );
    assert_eq!(
        pick(&grid, &cam, (1e13, 0.0)),
        Err(MapError::OutOfRange("screen"))
    );
}
