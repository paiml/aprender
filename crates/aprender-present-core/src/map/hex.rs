//! Pointy-top hex grid for the map runtime (APR-MAP-001 MAP-6 slice 2).
//!
//! Layout, world units, circumradius `R`:
//! - column pitch `sqrt(3) * R`, row pitch `1.5 * R`;
//! - odd rows are offset by half a column: centre `(col, row)` is
//!   `(sqrt(3) * R * (col + 0.5 * (row & 1)), 1.5 * R * row)`;
//! - cell id `row * cols + col`, row-major.
//!
//! Picking converts a world point to fractional axial coordinates and rounds in cube space.
//! Target-neutral plain Rust.

use super::{check_scale, MapError};

/// Columns of the map grid.
pub const MAP_COLS: u32 = 117;
/// Rows of the map grid.
pub const MAP_ROWS: u32 = 90;

/// A pointy-top hex grid of `cols` by `rows` cells with circumradius `radius` world units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HexGrid {
    cols: u32,
    rows: u32,
    radius: f64,
}

impl HexGrid {
    /// Builds a grid. Rejects an empty grid or one whose cell count overflows `u32`
    /// (`OutOfRange("grid")`), and a radius that is not finite, not positive or out of range.
    pub fn new(cols: u32, rows: u32, radius: f64) -> Result<Self, MapError> {
        unimplemented!("MAP-6 slice 2 RED: HexGrid::new {cols} {rows} {radius}")
    }

    /// The 117 by 90 map grid with the given circumradius.
    pub fn map_default(radius: f64) -> Result<Self, MapError> {
        Self::new(MAP_COLS, MAP_ROWS, radius)
    }

    /// Number of columns.
    pub fn cols(&self) -> u32 {
        self.cols
    }

    /// Number of rows.
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Circumradius in world units.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Number of cells, `cols * rows`.
    pub fn cell_count(&self) -> u32 {
        unimplemented!("MAP-6 slice 2 RED: cell_count")
    }

    /// Id of the cell at `(col, row)`, or `None` outside the grid.
    pub fn id_of(&self, col: u32, row: u32) -> Option<u32> {
        unimplemented!("MAP-6 slice 2 RED: id_of {col} {row}")
    }

    /// `(col, row)` of a cell id, or `None` outside the grid.
    pub fn coords_of(&self, id: u32) -> Option<(u32, u32)> {
        unimplemented!("MAP-6 slice 2 RED: coords_of {id}")
    }

    /// World centre of a cell, or `None` outside the grid.
    pub fn centre(&self, id: u32) -> Option<(f64, f64)> {
        unimplemented!("MAP-6 slice 2 RED: centre {id}")
    }

    /// The cell whose hexagon contains a world point, or `None` when that hexagon is not on the
    /// grid or the point is not finite.
    pub fn cell_at_world(&self, p: (f64, f64)) -> Option<u32> {
        unimplemented!("MAP-6 slice 2 RED: cell_at_world {p:?}")
    }
}

#[cfg(test)]
#[path = "hex_tests.rs"]
mod tests;
