//! Invariant Violation Tests (20 points)
//!
//! Tests for mathematical invariants.

/// Invariant category marker
#[derive(Debug, Clone, Copy)]
pub struct InvariantCategory;

impl InvariantCategory {
    pub const ID_PREFIX: &'static str = "INV";
    pub const TOTAL_POINTS: u32 = 20;
}
