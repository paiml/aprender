//! Cleanup module - Orphan device detection and graceful shutdown
//!
//! Per DT-003: Implements signal handlers and startup orphan cleanup to prevent
//! stale ublk device state that causes I/O errors (see F083 root cause analysis).
//!
//! ## Features
//! - Detects orphaned ublk character devices on startup (via duende-ublk)
//! - Cleans up orphaned devices before creating new ones
//! - Provides graceful shutdown via SIGTERM/SIGINT handlers

#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Result type for cleanup operations
pub type CleanupResult<T> = Result<T, CleanupError>;

/// Errors that can occur during cleanup
#[derive(Debug, thiserror::Error)]
pub enum CleanupError {
    #[error("Failed to read /dev directory: {0}")]
    ReadDevDir(#[from] std::io::Error),

    #[error("Failed to reset device {dev_id}: {message}")]
    ResetDevice { dev_id: u32, message: String },

    #[error("Cleanup timed out after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    #[error("duende-ublk error: {0}")]
    DuendeUblk(#[from] duende_ublk::Error),
}

/// Detect orphaned ublk character devices.
///
/// Returns a list of device IDs that have character devices (/dev/ublkcN)
/// but no corresponding block devices (/dev/ublkbN) or running daemon.
///
/// Delegates to duende-ublk for the actual detection.
pub fn detect_orphaned_devices() -> CleanupResult<Vec<u32>> {
    match duende_ublk::detect_orphaned_devices() {
        Ok(orphans) => {
            if !orphans.is_empty() {
                info!(count = orphans.len(), "Detected orphaned ublk devices: {:?}", orphans);
            }
            Ok(orphans)
        }
        Err(e) => {
            debug!(error = %e, "Failed to detect orphaned devices");
            Ok(Vec::new()) // Return empty on error - don't fail startup
        }
    }
}

/// Clean up orphaned ublk devices.
///
/// This function attempts to reset any orphaned devices found by `detect_orphaned_devices`.
/// It's designed to be called at startup before creating new devices.
///
/// Delegates to duende-ublk for the actual cleanup.
pub fn cleanup_orphaned_devices() -> CleanupResult<usize> {
    match duende_ublk::cleanup_orphaned_devices() {
        Ok(cleaned) => {
            if cleaned > 0 {
                info!(cleaned, "Orphan cleanup completed");
            }
            Ok(cleaned)
        }
        Err(e) => {
            warn!(error = %e, "Failed to clean up orphaned devices");
            Ok(0) // Don't fail startup due to cleanup errors
        }
    }
}

/// Set up graceful shutdown signal handlers.
///
/// Installs handlers for SIGTERM and SIGINT that:
/// 1. Set the stop flag to signal the daemon to exit
/// 2. Log the shutdown request
///
/// Returns an Arc<AtomicBool> that will be set to true when shutdown is requested.
pub fn setup_signal_handlers() -> Arc<AtomicBool> {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();

    // SIGINT (Ctrl+C) and SIGTERM handler
    ctrlc::set_handler(move || {
        if stop_clone.load(Ordering::Relaxed) {
            // Second signal - force exit
            warn!("Received second interrupt, forcing exit");
            std::process::exit(1);
        }
        info!("Received shutdown signal, initiating graceful shutdown...");
        stop_clone.store(true, Ordering::Relaxed);
    })
    .expect("Failed to set signal handler");

    stop
}

/// Perform startup cleanup and return stop signal.
///
/// This is the main entry point for cleanup, called from main() before
/// any device operations. It:
/// 1. Detects and cleans orphaned devices
/// 2. Sets up signal handlers
///
/// Returns the stop signal for graceful shutdown.
pub fn startup_cleanup() -> Arc<AtomicBool> {
    info!("Performing startup cleanup...");

    // Clean up any orphaned devices from previous runs
    match cleanup_orphaned_devices() {
        Ok(cleaned) if cleaned > 0 => {
            info!(cleaned, "Startup cleanup: removed orphaned devices");
        }
        Ok(_) => {
            debug!("Startup cleanup: no orphaned devices found");
        }
        Err(e) => {
            warn!(error = %e, "Startup cleanup: failed to clean orphaned devices");
        }
    }

    // Set up signal handlers for graceful shutdown
    setup_signal_handlers()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_orphaned_devices_empty() {
        // Should not panic on systems without ublk
        let result = detect_orphaned_devices();
        assert!(result.is_ok());
    }

    #[test]
    fn test_signal_handler_setup() {
        let stop = setup_signal_handlers();
        assert!(!stop.load(Ordering::Relaxed));
    }
}
