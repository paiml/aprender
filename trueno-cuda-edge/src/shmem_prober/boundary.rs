//! Shared memory boundary probe with sentinel detection.
//!
//! Places sentinel values at the edges of shared memory allocations
//! to detect out-of-bounds writes.

use serde::{Deserialize, Serialize};

/// Sentinel value placed before the allocation (underflow guard).
pub const SENTINEL_BEFORE: u32 = 0xDEAD_BEEF;

/// Sentinel value placed after the allocation (overflow guard).
pub const SENTINEL_AFTER: u32 = 0xCAFE_BABE;

/// Sentinel size in bytes (one u32).
pub const SENTINEL_SIZE: u64 = 4;

/// A region of shared memory with sentinel guards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedMemoryRegion {
    /// Base offset within shared memory (bytes).
    pub base_offset: u64,
    /// Size of the user allocation (bytes, excluding sentinels).
    pub size: u64,
}

impl SharedMemoryRegion {
    /// Create a new shared memory region.
    #[must_use]
    pub fn new(base_offset: u64, size: u64) -> Self {
        Self { base_offset, size }
    }

    /// Total size including sentinel guards.
    #[must_use]
    pub fn total_size(&self) -> u64 {
        SENTINEL_SIZE + self.size + SENTINEL_SIZE
    }

    /// Offset of the underflow sentinel.
    #[must_use]
    pub fn sentinel_before_offset(&self) -> u64 {
        self.base_offset
    }

    /// Offset of the overflow sentinel.
    #[must_use]
    pub fn sentinel_after_offset(&self) -> u64 {
        self.base_offset + SENTINEL_SIZE + self.size
    }
}

/// A detected boundary violation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryViolation {
    /// The underflow sentinel was corrupted.
    UnderflowCorrupted {
        /// Expected sentinel value.
        expected: u32,
        /// Actual value found.
        actual: u32,
    },
    /// The overflow sentinel was corrupted.
    OverflowCorrupted {
        /// Expected sentinel value.
        expected: u32,
        /// Actual value found.
        actual: u32,
    },
}

/// Report from a boundary probe.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BoundaryProbeReport {
    /// Detected violations.
    pub violations: Vec<BoundaryViolation>,
    /// Number of regions probed.
    pub regions_probed: u32,
}

impl BoundaryProbeReport {
    /// Returns true if any violations were detected.
    #[must_use]
    pub fn has_violations(&self) -> bool {
        !self.violations.is_empty()
    }
}

/// Compute the byte offsets where sentinels should be placed for a
/// given set of shared memory regions.
///
/// Returns pairs of (`before_offset`, `after_offset`) for each region.
#[must_use]
pub fn compute_sentinel_offsets(regions: &[SharedMemoryRegion]) -> Vec<(u64, u64)> {
    regions.iter().map(|r| (r.sentinel_before_offset(), r.sentinel_after_offset())).collect()
}

/// Check sentinel values against expected constants.
#[must_use]
pub fn check_sentinels(before_value: u32, after_value: u32) -> Vec<BoundaryViolation> {
    let mut violations = Vec::new();
    if before_value != SENTINEL_BEFORE {
        violations.push(BoundaryViolation::UnderflowCorrupted {
            expected: SENTINEL_BEFORE,
            actual: before_value,
        });
    }
    if after_value != SENTINEL_AFTER {
        violations.push(BoundaryViolation::OverflowCorrupted {
            expected: SENTINEL_AFTER,
            actual: after_value,
        });
    }
    violations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentinel_offsets_computed_correctly() {
        let region = SharedMemoryRegion::new(0, 1024);
        assert_eq!(region.sentinel_before_offset(), 0);
        assert_eq!(region.sentinel_after_offset(), SENTINEL_SIZE + 1024);
    }

    #[test]
    fn total_size_includes_sentinels() {
        let region = SharedMemoryRegion::new(0, 1024);
        assert_eq!(region.total_size(), 1024 + 2 * SENTINEL_SIZE);
    }

    #[test]
    fn compute_sentinel_offsets_multiple_regions() {
        let regions = vec![
            SharedMemoryRegion::new(0, 256),
            SharedMemoryRegion::new(264, 512), // 256 + 2*4 = 264
        ];
        let offsets = compute_sentinel_offsets(&regions);
        assert_eq!(offsets.len(), 2);
        assert_eq!(offsets[0], (0, SENTINEL_SIZE + 256));
        assert_eq!(offsets[1], (264, 264 + SENTINEL_SIZE + 512));
    }

    #[test]
    fn check_sentinels_clean() {
        let violations = check_sentinels(SENTINEL_BEFORE, SENTINEL_AFTER);
        assert!(violations.is_empty());
    }

    #[test]
    fn check_sentinels_underflow_corrupted() {
        let violations = check_sentinels(0xBAAD_F00D, SENTINEL_AFTER);
        assert_eq!(violations.len(), 1);
        assert!(matches!(violations[0], BoundaryViolation::UnderflowCorrupted { .. }));
    }

    #[test]
    fn check_sentinels_both_corrupted() {
        let violations = check_sentinels(0, 0);
        assert_eq!(violations.len(), 2);
    }
}
