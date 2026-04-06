//! Cross-Implementation Parity Tests (10 points)
//!
//! Tests for consistency between implementations.

/// Parity category marker
#[derive(Debug, Clone, Copy)]
pub struct ParityCategory;

impl ParityCategory {
    pub const ID_PREFIX: &'static str = "PAR";
    pub const TOTAL_POINTS: u32 = 10;
}
