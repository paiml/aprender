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

use super::{check_coord, check_scale, MapError};

/// Pitch factor of the columns: `sqrt(3)` in units of the circumradius.
fn sqrt3() -> f64 {
    3f64.sqrt()
}

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
        if cols == 0 || rows == 0 || cols.checked_mul(rows).is_none() {
            return Err(MapError::OutOfRange("grid"));
        }
        let radius = check_scale(radius, "radius")?;
        check_coord(radius, "radius")?;
        Ok(Self { cols, rows, radius })
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
        self.cols * self.rows
    }

    /// Id of the cell at `(col, row)`, or `None` outside the grid.
    pub fn id_of(&self, col: u32, row: u32) -> Option<u32> {
        (col < self.cols && row < self.rows).then(|| row * self.cols + col)
    }

    /// `(col, row)` of a cell id, or `None` outside the grid.
    pub fn coords_of(&self, id: u32) -> Option<(u32, u32)> {
        (id < self.cell_count()).then(|| (id % self.cols, id / self.cols))
    }

    /// World centre of a cell, or `None` outside the grid.
    pub fn centre(&self, id: u32) -> Option<(f64, f64)> {
        let (col, row) = self.coords_of(id)?;
        let odd = f64::from(row % 2);
        Some((
            sqrt3() * self.radius * (f64::from(col) + 0.5 * odd),
            1.5 * self.radius * f64::from(row),
        ))
    }

    /// The cell whose hexagon contains a world point, or `None` when that hexagon is not on the
    /// grid or the point is not finite.
    pub fn cell_at_world(&self, p: (f64, f64)) -> Option<u32> {
        if !p.0.is_finite() || !p.1.is_finite() {
            return None;
        }
        // Fractional axial coordinates of the point (pointy-top, odd-r offset convention).
        let q = (sqrt3() / 3.0 * p.0 - p.1 / 3.0) / self.radius;
        let r = (2.0 / 3.0 * p.1) / self.radius;
        // Beyond this every rounded axial value lies off the grid; the bound also keeps the
        // integer conversion exact.
        let limit = f64::from(self.cols) + f64::from(self.rows) + 4.0;
        if q.abs() > limit || r.abs() > limit {
            return None;
        }
        let (aq, ar) = cube_round(q, r);
        // Back from axial to odd-r offset: col = q + floor(row / 2), row = r.
        let col = aq + ar.div_euclid(2);
        let col = u32::try_from(col).ok()?;
        let row = u32::try_from(ar).ok()?;
        self.id_of(col, row)
    }
}

/// Rounds fractional axial `(q, r)` to the containing hex in cube space: round each cube
/// component, then reset the one with the largest rounding error so the three sum to zero.
/// The caller bounds the inputs, so the integer conversion is exact.
#[allow(clippy::cast_possible_truncation)]
fn cube_round(q: f64, r: f64) -> (i64, i64) {
    let s = -q - r;
    let (mut rq, mut rr) = (q.round(), r.round());
    let rs = s.round();
    let (dq, dr, ds) = ((rq - q).abs(), (rr - r).abs(), (rs - s).abs());
    if dq > dr && dq > ds {
        rq = -rr - rs;
    } else if dr > ds {
        rr = -rq - rs;
    }
    (rq as i64, rr as i64)
}

#[cfg(test)]
#[path = "hex_tests.rs"]
mod tests;
