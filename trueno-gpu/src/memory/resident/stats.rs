//! GPU Transfer Statistics (WAPR-PERF-004)
//!
//! Global counters and snapshot structs for tracking host↔device data movement.

use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================================
// Global Transfer Counters
// ============================================================================

/// Global transfer counters for debugging
static TOTAL_H2D_TRANSFERS: AtomicU64 = AtomicU64::new(0);
static TOTAL_D2H_TRANSFERS: AtomicU64 = AtomicU64::new(0);
static TOTAL_H2D_BYTES: AtomicU64 = AtomicU64::new(0);
static TOTAL_D2H_BYTES: AtomicU64 = AtomicU64::new(0);

/// Get total host-to-device transfers since last reset
#[must_use]
pub fn total_h2d_transfers() -> u64 {
    TOTAL_H2D_TRANSFERS.load(Ordering::Relaxed)
}

/// Get total device-to-host transfers since last reset
#[must_use]
pub fn total_d2h_transfers() -> u64 {
    TOTAL_D2H_TRANSFERS.load(Ordering::Relaxed)
}

/// Get total bytes transferred host-to-device since last reset
#[must_use]
pub fn total_h2d_bytes() -> u64 {
    TOTAL_H2D_BYTES.load(Ordering::Relaxed)
}

/// Get total bytes transferred device-to-host since last reset
#[must_use]
pub fn total_d2h_bytes() -> u64 {
    TOTAL_D2H_BYTES.load(Ordering::Relaxed)
}

/// Reset all transfer counters to zero
pub fn reset_transfer_counters() {
    TOTAL_H2D_TRANSFERS.store(0, Ordering::Relaxed);
    TOTAL_D2H_TRANSFERS.store(0, Ordering::Relaxed);
    TOTAL_H2D_BYTES.store(0, Ordering::Relaxed);
    TOTAL_D2H_BYTES.store(0, Ordering::Relaxed);
}

/// Increment H2D transfer counter (used by GpuResidentTensor)
pub(crate) fn record_h2d_transfer(bytes: u64) {
    TOTAL_H2D_TRANSFERS.fetch_add(1, Ordering::Relaxed);
    TOTAL_H2D_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

/// Increment D2H transfer counter (used by GpuResidentTensor)
pub(crate) fn record_d2h_transfer(bytes: u64) {
    TOTAL_D2H_TRANSFERS.fetch_add(1, Ordering::Relaxed);
    TOTAL_D2H_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

// ============================================================================
// Transfer Statistics Summary
// ============================================================================

/// Summary of GPU transfer statistics
#[derive(Debug, Clone, Default)]
pub struct TransferStats {
    /// Total host-to-device transfers
    pub h2d_transfers: u64,
    /// Total device-to-host transfers
    pub d2h_transfers: u64,
    /// Total bytes transferred host-to-device
    pub h2d_bytes: u64,
    /// Total bytes transferred device-to-host
    pub d2h_bytes: u64,
}

impl TransferStats {
    /// Capture current transfer statistics
    #[must_use]
    pub fn capture() -> Self {
        Self {
            h2d_transfers: total_h2d_transfers(),
            d2h_transfers: total_d2h_transfers(),
            h2d_bytes: total_h2d_bytes(),
            d2h_bytes: total_d2h_bytes(),
        }
    }

    /// Calculate delta from a previous snapshot
    #[must_use]
    pub fn delta_from(&self, prev: &Self) -> Self {
        Self {
            h2d_transfers: self.h2d_transfers.saturating_sub(prev.h2d_transfers),
            d2h_transfers: self.d2h_transfers.saturating_sub(prev.d2h_transfers),
            h2d_bytes: self.h2d_bytes.saturating_sub(prev.h2d_bytes),
            d2h_bytes: self.d2h_bytes.saturating_sub(prev.d2h_bytes),
        }
    }

    /// Total transfers (H2D + D2H)
    #[must_use]
    pub const fn total_transfers(&self) -> u64 {
        self.h2d_transfers + self.d2h_transfers
    }

    /// Total bytes transferred
    #[must_use]
    pub const fn total_bytes(&self) -> u64 {
        self.h2d_bytes + self.d2h_bytes
    }
}

impl std::fmt::Display for TransferStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "H2D: {} ({:.2} MB), D2H: {} ({:.2} MB)",
            self.h2d_transfers,
            self.h2d_bytes as f64 / (1024.0 * 1024.0),
            self.d2h_transfers,
            self.d2h_bytes as f64 / (1024.0 * 1024.0)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_counter_reset() {
        reset_transfer_counters();
        assert_eq!(total_h2d_transfers(), 0);
        assert_eq!(total_d2h_transfers(), 0);
        assert_eq!(total_h2d_bytes(), 0);
        assert_eq!(total_d2h_bytes(), 0);
    }

    #[test]
    fn test_transfer_counter_increment() {
        reset_transfer_counters();
        record_h2d_transfer(1024);
        record_h2d_transfer(2048);
        record_d2h_transfer(512);

        assert_eq!(total_h2d_transfers(), 2);
        assert_eq!(total_d2h_transfers(), 1);
        assert_eq!(total_h2d_bytes(), 3072);
        assert_eq!(total_d2h_bytes(), 512);
    }

    #[test]
    fn test_transfer_stats_capture() {
        reset_transfer_counters();
        record_h2d_transfer(100);
        record_d2h_transfer(200);

        let stats = TransferStats::capture();
        assert_eq!(stats.h2d_transfers, 1);
        assert_eq!(stats.d2h_transfers, 1);
        assert_eq!(stats.h2d_bytes, 100);
        assert_eq!(stats.d2h_bytes, 200);
    }

    #[test]
    fn test_transfer_stats_delta() {
        let prev =
            TransferStats { h2d_transfers: 10, d2h_transfers: 5, h2d_bytes: 1000, d2h_bytes: 500 };

        let curr =
            TransferStats { h2d_transfers: 15, d2h_transfers: 8, h2d_bytes: 2500, d2h_bytes: 1200 };

        let delta = curr.delta_from(&prev);
        assert_eq!(delta.h2d_transfers, 5);
        assert_eq!(delta.d2h_transfers, 3);
        assert_eq!(delta.h2d_bytes, 1500);
        assert_eq!(delta.d2h_bytes, 700);
    }

    #[test]
    fn test_transfer_stats_totals() {
        let stats =
            TransferStats { h2d_transfers: 10, d2h_transfers: 5, h2d_bytes: 1000, d2h_bytes: 500 };

        assert_eq!(stats.total_transfers(), 15);
        assert_eq!(stats.total_bytes(), 1500);
    }

    #[test]
    fn test_transfer_stats_display() {
        let stats = TransferStats {
            h2d_transfers: 100,
            d2h_transfers: 50,
            h2d_bytes: 1024 * 1024, // 1 MB
            d2h_bytes: 512 * 1024,  // 0.5 MB
        };

        let display = format!("{}", stats);
        assert!(display.contains("H2D: 100"));
        assert!(display.contains("D2H: 50"));
        assert!(display.contains("1.00 MB"));
        assert!(display.contains("0.50 MB"));
    }
}
