//! Boundary Condition Tests (20 points)
//!
//! Tests for edge cases and boundary conditions.

/// Boundary condition category marker
#[derive(Debug, Clone, Copy)]
pub struct BoundaryCategory;

impl BoundaryCategory {
    pub const ID_PREFIX: &'static str = "BC";
    pub const TOTAL_POINTS: u32 = 20;
}
