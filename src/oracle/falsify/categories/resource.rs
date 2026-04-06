//! Resource Exhaustion Tests (15 points)
//!
//! Tests for memory, file descriptor, and stack limits.

/// Resource exhaustion category marker
#[derive(Debug, Clone, Copy)]
pub struct ResourceCategory;

impl ResourceCategory {
    pub const ID_PREFIX: &'static str = "RES";
    pub const TOTAL_POINTS: u32 = 15;
}
