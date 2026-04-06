//! Numerical Stability Tests (20 points)
//!
//! Tests for floating-point and numerical correctness.

/// Numerical stability category marker
#[derive(Debug, Clone, Copy)]
pub struct NumericalCategory;

impl NumericalCategory {
    pub const ID_PREFIX: &'static str = "NUM";
    pub const TOTAL_POINTS: u32 = 20;
}
