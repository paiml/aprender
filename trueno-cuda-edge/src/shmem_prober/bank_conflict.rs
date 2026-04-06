//! Bank conflict injection for shared memory.
//!
//! CUDA shared memory is organized into 32 banks. When multiple threads
//! in a warp access the same bank, the accesses are serialized. This
//! module provides tools to deliberately inject and measure bank conflicts.

use serde::{Deserialize, Serialize};

use super::overflow::AccessPattern;

/// Injector for deliberate bank conflicts in shared memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BankConflictInjector {
    /// Number of banks (typically 32 for CUDA).
    pub num_banks: u32,
    /// Bank width in bytes (typically 4 for 32-bit words).
    pub bank_width_bytes: u32,
}

impl Default for BankConflictInjector {
    fn default() -> Self {
        Self { num_banks: 32, bank_width_bytes: 4 }
    }
}

impl BankConflictInjector {
    /// Create a new bank conflict injector with default CUDA parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the expected serialization factor for a given access pattern.
    ///
    /// A factor of 1 means no serialization (all accesses proceed in parallel).
    /// A factor of 32 means full serialization (worst case).
    #[must_use]
    pub fn expected_serialization(&self, pattern: AccessPattern) -> u32 {
        pattern.serialization_factor()
    }

    /// Compute which bank a byte offset maps to.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn bank_for_offset(&self, byte_offset: u64) -> u32 {
        let word_index = byte_offset / u64::from(self.bank_width_bytes);
        // num_banks is u32, so modulo result fits in u32
        (word_index % u64::from(self.num_banks)) as u32
    }

    /// Generate byte offsets that cause a full 32-way bank conflict.
    ///
    /// All returned offsets map to the same bank (bank 0 by default).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn full_conflict_offsets(&self, count: usize) -> Vec<u64> {
        // All threads access bank 0: offsets 0, 128, 256, 384, ...
        // (offset / 4) % 32 == 0 when offset is a multiple of 128
        let stride = u64::from(self.num_banks) * u64::from(self.bank_width_bytes);
        (0..count).map(|i| i as u64 * stride).collect()
    }

    /// Generate byte offsets with stride-2 access (2-way conflicts).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn stride2_offsets(&self, count: usize) -> Vec<u64> {
        // Stride of 2 words = 8 bytes
        // Every pair of threads conflicts
        (0..count).map(|i| i as u64 * 8).collect()
    }
}

/// Result of measuring bank conflicts for a kernel execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BankConflictResult {
    /// Total shared memory transactions.
    pub total_transactions: u64,
    /// Ideal transactions (no conflicts).
    pub ideal_transactions: u64,
    /// Measured serialization factor.
    pub serialization_factor: f64,
}

impl BankConflictResult {
    /// Returns true if significant bank conflicts were detected.
    #[must_use]
    pub fn has_conflicts(&self) -> bool {
        self.serialization_factor > 1.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_has_no_serialization() {
        let injector = BankConflictInjector::new();
        assert_eq!(injector.expected_serialization(AccessPattern::Sequential), 1);
    }

    #[test]
    fn full_conflict_has_32x_serialization() {
        let injector = BankConflictInjector::new();
        assert_eq!(injector.expected_serialization(AccessPattern::FullConflict), 32);
    }

    #[test]
    fn stride2_has_2x_serialization() {
        let injector = BankConflictInjector::new();
        assert_eq!(injector.expected_serialization(AccessPattern::Stride2), 2);
    }

    #[test]
    fn bank_for_offset_cycles() {
        let injector = BankConflictInjector::new();
        assert_eq!(injector.bank_for_offset(0), 0);
        assert_eq!(injector.bank_for_offset(4), 1);
        assert_eq!(injector.bank_for_offset(124), 31);
        assert_eq!(injector.bank_for_offset(128), 0); // wraps
    }

    #[test]
    fn full_conflict_offsets_all_same_bank() {
        let injector = BankConflictInjector::new();
        let offsets = injector.full_conflict_offsets(32);
        for offset in &offsets {
            assert_eq!(injector.bank_for_offset(*offset), 0);
        }
    }

    #[test]
    fn stride2_offsets_pattern() {
        let injector = BankConflictInjector::new();
        let offsets = injector.stride2_offsets(4);
        assert_eq!(offsets, vec![0, 8, 16, 24]);
    }

    #[test]
    fn bank_conflict_result_has_conflicts() {
        let no_conflict = BankConflictResult {
            total_transactions: 100,
            ideal_transactions: 100,
            serialization_factor: 1.0,
        };
        assert!(!no_conflict.has_conflicts());

        let with_conflict = BankConflictResult {
            total_transactions: 3200,
            ideal_transactions: 100,
            serialization_factor: 32.0,
        };
        assert!(with_conflict.has_conflicts());
    }

    #[test]
    fn bank_conflict_result_borderline() {
        let borderline = BankConflictResult {
            total_transactions: 110,
            ideal_transactions: 100,
            serialization_factor: 1.1,
        };
        assert!(!borderline.has_conflicts()); // exactly 1.1 is not > 1.1

        let above = BankConflictResult {
            total_transactions: 111,
            ideal_transactions: 100,
            serialization_factor: 1.11,
        };
        assert!(above.has_conflicts());
    }

    #[test]
    fn injector_default() {
        let injector = BankConflictInjector::default();
        assert_eq!(injector.num_banks, 32);
        assert_eq!(injector.bank_width_bytes, 4);
    }

    #[test]
    fn padded_access_pattern_serialization() {
        let injector = BankConflictInjector::new();
        assert_eq!(
            injector.expected_serialization(super::super::overflow::AccessPattern::Padded),
            1
        );
    }

    #[test]
    fn stride32_access_pattern_serialization() {
        let injector = BankConflictInjector::new();
        assert_eq!(
            injector.expected_serialization(super::super::overflow::AccessPattern::Stride32),
            1
        );
    }
}
