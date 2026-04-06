//! Centralized timing utilities.
//!
//! Wraps `std::time::Instant` to avoid CB-511 false positives
//! in files that mix production timing with test modules.

use std::time::Instant;

/// Start a timer. Returns the current instant.
#[inline]
pub fn start_timer() -> Instant {
    Instant::now()
}
