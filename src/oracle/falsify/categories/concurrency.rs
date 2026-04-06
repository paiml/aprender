//! Concurrency Tests (15 points)
//!
//! Tests for race conditions and deadlocks.

/// Concurrency category marker
#[derive(Debug, Clone, Copy)]
pub struct ConcurrencyCategory;

impl ConcurrencyCategory {
    pub const ID_PREFIX: &'static str = "CONC";
    pub const TOTAL_POINTS: u32 = 15;
}
