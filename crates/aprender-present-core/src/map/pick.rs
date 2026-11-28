//! Screen-pixel picking for the map runtime (APR-MAP-001 MAP-6 slice 2).
//!
//! A screen pixel goes to a world point through [`Camera2D::world_at`], then to the hex cell
//! containing it through [`HexGrid::cell_at_world`].

use super::{camera::Camera2D, hex::HexGrid, MapError};

/// The cell under a screen pixel, or `None` when that point is off the grid.
///
/// Fails with `NonFinite("screen")` or `OutOfRange("screen")` for a pixel that is not finite or
/// not within the coordinate bound.
pub fn pick(
    grid: &HexGrid,
    camera: &Camera2D,
    screen: (f64, f64),
) -> Result<Option<u32>, MapError> {
    unimplemented!("MAP-6 slice 2 RED: pick {grid:?} {camera:?} {screen:?}")
}

#[cfg(test)]
#[path = "pick_tests.rs"]
mod tests;
