//! Shared memory overflow detection for CUDA compute capabilities.
//!
//! Each GPU Streaming Multiprocessor (SM) has a fixed shared memory limit
//! that depends on the compute capability. This module validates that
//! allocations stay within hardware limits and models access patterns for
//! bank conflict analysis.

use serde::{Deserialize, Serialize};

/// CUDA compute capability (major.minor).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ComputeCapability {
    /// Major version (e.g., 7 for Volta, 8 for Ampere, 9 for Hopper).
    pub major: u32,
    /// Minor version.
    pub minor: u32,
}

impl ComputeCapability {
    /// Create a new compute capability.
    #[must_use]
    pub fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }
}

impl std::fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sm_{}{}", self.major, self.minor)
    }
}

/// Returns the shared memory limit per SM in bytes for a given compute capability.
///
/// Sources: NVIDIA CUDA Programming Guide, Table 15.
#[must_use]
pub fn shared_memory_limit(cc: ComputeCapability) -> u64 {
    match cc.major {
        // Volta / Turing
        7 => 96 * 1024,
        // Ampere
        8 => 164 * 1024,
        // Hopper
        9 => 228 * 1024,
        // Kepler (3), Maxwell (5), Pascal (6), Unknown — conservative fallback
        _ => 48 * 1024,
    }
}

/// Check whether a shared memory allocation fits within hardware limits.
///
/// Returns `Ok(())` if the allocation fits, or an [`EdgeError::SharedMemoryOverflow`]
/// if it exceeds the limit.
///
/// # Errors
///
/// Returns an error if `requested_bytes` exceeds the shared memory limit
/// for the given compute capability.
pub fn check_allocation(cc: ComputeCapability, requested_bytes: u64) -> crate::error::Result<()> {
    let limit = shared_memory_limit(cc);
    if requested_bytes > limit {
        return Err(crate::error::EdgeError::SharedMemoryOverflow {
            requested: requested_bytes,
            limit,
        });
    }
    Ok(())
}

/// Memory access pattern for shared memory bank conflict analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessPattern {
    /// Sequential access — each thread accesses consecutive 4-byte words.
    Sequential,
    /// Full conflict — all 32 threads access the same bank.
    FullConflict,
    /// Stride-2 access — threads access every other word.
    Stride2,
    /// Stride-32 access — threads access every 32nd word (broadcast).
    Stride32,
    /// Padded access — adds 1-word padding per row to avoid conflicts.
    Padded,
}

impl AccessPattern {
    /// Returns the expected serialization factor for this pattern.
    ///
    /// A factor of 1 means no bank conflicts; 32 means full serialization.
    #[must_use]
    pub fn serialization_factor(&self) -> u32 {
        match self {
            Self::FullConflict => 32,
            Self::Stride2 => 2,
            // Sequential, Stride32 (broadcast), Padded — no conflicts
            Self::Sequential | Self::Stride32 | Self::Padded => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn volta_shared_memory_is_96k() {
        let cc = ComputeCapability::new(7, 0);
        assert_eq!(shared_memory_limit(cc), 96 * 1024);
    }

    #[test]
    fn ampere_shared_memory_is_164k() {
        let cc = ComputeCapability::new(8, 0);
        assert_eq!(shared_memory_limit(cc), 164 * 1024);
    }

    #[test]
    fn hopper_shared_memory_is_228k() {
        let cc = ComputeCapability::new(9, 0);
        assert_eq!(shared_memory_limit(cc), 228 * 1024);
    }

    #[test]
    fn check_allocation_within_limit() {
        let cc = ComputeCapability::new(7, 0);
        assert!(check_allocation(cc, 48 * 1024).is_ok());
    }

    #[test]
    fn check_allocation_exceeds_limit() {
        let cc = ComputeCapability::new(3, 5);
        assert!(check_allocation(cc, 49 * 1024).is_err());
    }

    #[test]
    fn compute_capability_display() {
        let cc = ComputeCapability::new(8, 6);
        assert_eq!(cc.to_string(), "sm_86");
    }

    #[test]
    fn serialization_factors() {
        assert_eq!(AccessPattern::Sequential.serialization_factor(), 1);
        assert_eq!(AccessPattern::FullConflict.serialization_factor(), 32);
        assert_eq!(AccessPattern::Stride2.serialization_factor(), 2);
        assert_eq!(AccessPattern::Stride32.serialization_factor(), 1);
        assert_eq!(AccessPattern::Padded.serialization_factor(), 1);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn allocation_at_limit_succeeds(major in 3u32..=9) {
            let cc = ComputeCapability::new(major, 0);
            let limit = shared_memory_limit(cc);
            prop_assert!(check_allocation(cc, limit).is_ok());
        }

        #[test]
        fn allocation_above_limit_fails(major in 3u32..=9, extra in 1u64..10000) {
            let cc = ComputeCapability::new(major, 0);
            let limit = shared_memory_limit(cc);
            prop_assert!(check_allocation(cc, limit + extra).is_err());
        }
    }
}
