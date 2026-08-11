//! GPU Transfer Statistics (WAPR-PERF-004)
//!
//! Per-thread counters and snapshot structs for tracking host↔device data
//! movement.
//!
//! # Why per-thread and not process-global (GPU-ORD-1 / GPU-ORD-3)
//!
//! These counters exist to answer "how much did *this* piece of work move
//! across the bus", and every caller uses them the same way: reset, do some
//! work, read. That reading is only meaningful if nothing else contributed in
//! between.
//!
//! They used to be `static AtomicU64`, shared by the whole process. 87 call
//! sites reset-then-asserted them, so any two of those running concurrently
//! measured each other. In `cargo test --workspace --lib` that showed up as
//! `test_gpu_resident_tensor_lifecycle` asserting `total_h2d_transfers() == 1`
//! and reading 7, 12, 22 or 30 — the magnitude tracking how many neighbours
//! happened to transfer inside its measurement window — with the failing set
//! changing every run.
//!
//! Binding the counters to the thread that performs the transfer makes that
//! **unrepresentable** rather than merely unlikely: a reset-then-read pair on
//! one thread cannot observe a transfer performed on another, so no amount of
//! test parallelism or ordering can perturb it. It also matches what the
//! numbers were always meant to describe, since a transfer is attributed to
//! whoever issued it.
//!
//! No aggregate-across-threads consumer exists; the only non-test callers are
//! re-exports in `memory::resident`.

use std::cell::Cell;

// ============================================================================
// Per-Thread Transfer Counters
// ============================================================================

thread_local! {
    static H2D_TRANSFERS: Cell<u64> = const { Cell::new(0) };
    static D2H_TRANSFERS: Cell<u64> = const { Cell::new(0) };
    static H2D_BYTES: Cell<u64> = const { Cell::new(0) };
    static D2H_BYTES: Cell<u64> = const { Cell::new(0) };
}

/// Get host-to-device transfers issued by the current thread since last reset
#[must_use]
pub fn total_h2d_transfers() -> u64 {
    H2D_TRANSFERS.with(Cell::get)
}

/// Get device-to-host transfers issued by the current thread since last reset
#[must_use]
pub fn total_d2h_transfers() -> u64 {
    D2H_TRANSFERS.with(Cell::get)
}

/// Get bytes transferred host-to-device by the current thread since last reset
#[must_use]
pub fn total_h2d_bytes() -> u64 {
    H2D_BYTES.with(Cell::get)
}

/// Get bytes transferred device-to-host by the current thread since last reset
#[must_use]
pub fn total_d2h_bytes() -> u64 {
    D2H_BYTES.with(Cell::get)
}

/// Reset the current thread's transfer counters to zero
pub fn reset_transfer_counters() {
    H2D_TRANSFERS.with(|c| c.set(0));
    D2H_TRANSFERS.with(|c| c.set(0));
    H2D_BYTES.with(|c| c.set(0));
    D2H_BYTES.with(|c| c.set(0));
}

/// Increment H2D transfer counter (used by `GpuResidentTensor`)
pub(crate) fn record_h2d_transfer(bytes: u64) {
    H2D_TRANSFERS.with(|c| c.set(c.get() + 1));
    H2D_BYTES.with(|c| c.set(c.get() + bytes));
}

/// Increment D2H transfer counter (used by `GpuResidentTensor`)
pub(crate) fn record_d2h_transfer(bytes: u64) {
    D2H_TRANSFERS.with(|c| c.set(c.get() + 1));
    D2H_BYTES.with(|c| c.set(c.get() + bytes));
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

    /// GPU-ORD-1 / GPU-ORD-3 falsifier: transfers performed by another thread
    /// must not land in this thread's measurement window.
    ///
    /// This is the failure as it actually presented — a test resets the
    /// counters, does one transfer, asserts `1`, and reads 7/12/22/30 because
    /// neighbouring tests transferred inside the window. Black box: the
    /// assertion is only about what this thread did, and says nothing about
    /// how the counters are stored.
    #[test]
    fn test_counters_are_not_polluted_by_other_threads() {
        reset_transfer_counters();
        record_h2d_transfer(32);

        // A neighbour doing far more work than us, concurrently.
        let neighbour = std::thread::spawn(|| {
            reset_transfer_counters();
            for _ in 0..1000 {
                record_h2d_transfer(4096);
                record_d2h_transfer(4096);
            }
            // The neighbour's own window is likewise its own.
            assert_eq!(total_h2d_transfers(), 1000);
            assert_eq!(total_d2h_transfers(), 1000);
        });
        neighbour.join().expect("neighbour thread must not panic");

        assert_eq!(
            total_h2d_transfers(),
            1,
            "another thread's H2D transfers were attributed to this thread"
        );
        assert_eq!(
            total_h2d_bytes(),
            32,
            "another thread's H2D bytes were attributed to this thread"
        );
        assert_eq!(
            total_d2h_transfers(),
            0,
            "another thread's D2H transfers were attributed to this thread"
        );
        assert_eq!(
            total_d2h_bytes(),
            0,
            "another thread's D2H bytes were attributed to this thread"
        );
    }

    /// A neighbour calling `reset_transfer_counters()` must not zero ours
    /// either — the pre-fix reset was a process-wide `store(0)`.
    #[test]
    fn test_reset_on_another_thread_does_not_zero_ours() {
        reset_transfer_counters();
        record_h2d_transfer(64);
        record_d2h_transfer(64);

        std::thread::spawn(|| {
            reset_transfer_counters();
        })
        .join()
        .expect("neighbour thread must not panic");

        assert_eq!(
            (total_h2d_transfers(), total_d2h_transfers()),
            (1, 1),
            "another thread's reset_transfer_counters() wiped this thread's window"
        );
    }

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
        let prev = TransferStats {
            h2d_transfers: 10,
            d2h_transfers: 5,
            h2d_bytes: 1000,
            d2h_bytes: 500,
        };

        let curr = TransferStats {
            h2d_transfers: 15,
            d2h_transfers: 8,
            h2d_bytes: 2500,
            d2h_bytes: 1200,
        };

        let delta = curr.delta_from(&prev);
        assert_eq!(delta.h2d_transfers, 5);
        assert_eq!(delta.d2h_transfers, 3);
        assert_eq!(delta.h2d_bytes, 1500);
        assert_eq!(delta.d2h_bytes, 700);
    }

    #[test]
    fn test_transfer_stats_totals() {
        let stats = TransferStats {
            h2d_transfers: 10,
            d2h_transfers: 5,
            h2d_bytes: 1000,
            d2h_bytes: 500,
        };

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
