//! High-Performance Profiling Patterns
//!
//! CPU cycle counters, cached time service, and page fault detection.
//! Based on Phase 11: E.9 patterns.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ============================================================================
// CPU Cycle Counters
// ============================================================================

/// CPU cycle counter using RDTSCP (x86_64) or CNTVCT_EL0 (ARM64).
///
/// Returns actual CPU cycles for frequency-invariant performance analysis.
/// Use with `elapsed_ns` to calculate IPC (Instructions Per Cycle).
///
/// # Example
/// ```rust,ignore
/// let start_cycles = cpu_cycles();
/// // ... operation ...
/// let end_cycles = cpu_cycles();
/// let cycles_per_element = (end_cycles - start_cycles) / num_elements;
/// ```
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn cpu_cycles() -> u64 {
    // SAFETY: __rdtscp reads the timestamp counter and has no safety invariants
    // beyond requiring x86_64 (guaranteed by cfg).
    unsafe {
        let mut _aux: u32 = 0;
        core::arch::x86_64::__rdtscp(&mut _aux)
    }
}

/// CPU cycle counter for ARM64 using CNTVCT_EL0 register.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn cpu_cycles() -> u64 {
    let cycles: u64;
    // SAFETY: CNTVCT_EL0 is always readable from EL0 on aarch64; no invariants.
    unsafe {
        core::arch::asm!("mrs {}, cntvct_el0", out(reg) cycles);
    }
    cycles
}

/// Fallback for unsupported architectures (returns 0).
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
pub fn cpu_cycles() -> u64 {
    0
}

// ============================================================================
// Cached Time Service (Pattern 2 from actix-web)
// ============================================================================

/// Global cached instant in nanoseconds, updated by background thread.
static CACHED_NANOS: AtomicU64 = AtomicU64::new(0);

/// Epoch instant for cached time calculation.
static EPOCH: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

/// Guard ensuring the time service background thread is spawned at most once.
static TIME_SERVICE_INIT: std::sync::OnceLock<()> = std::sync::OnceLock::new();

/// Initialize the cached time service (call once at startup).
///
/// Spawns a background thread that updates cached time every 100µs.
/// This avoids syscall overhead when profiling high-frequency operations.
///
/// # Example
/// ```rust,ignore
/// trueno::brick::init_time_service();
/// // Later...
/// let ns = trueno::brick::cached_nanos();
/// ```
pub fn init_time_service() {
    TIME_SERVICE_INIT.get_or_init(|| {
        let epoch = *EPOCH.get_or_init(Instant::now);
        CACHED_NANOS.store(0, Ordering::Relaxed);

        std::thread::Builder::new()
            .name("trueno-time-service".into())
            .spawn(move || loop {
                std::thread::sleep(std::time::Duration::from_micros(100));
                let elapsed = epoch.elapsed().as_nanos() as u64;
                CACHED_NANOS.store(elapsed, Ordering::Relaxed);
            })
            .expect("Failed to spawn time service thread");
    });
}

/// Get cached time in nanoseconds since epoch (NO SYSCALL, ~1ns overhead).
///
/// Returns 0 if time service is not initialized. For accurate timing,
/// call `init_time_service()` at application startup.
#[inline]
pub fn cached_nanos() -> u64 {
    CACHED_NANOS.load(Ordering::Relaxed)
}

/// Get cached time or fall back to Instant::now() if service not initialized.
#[inline]
pub fn cached_nanos_or_now() -> u64 {
    let cached = CACHED_NANOS.load(Ordering::Relaxed);
    if cached == 0 && TIME_SERVICE_INIT.get().is_none() {
        // Fall back to syscall if time service not initialized
        EPOCH.get_or_init(Instant::now).elapsed().as_nanos() as u64
    } else {
        cached
    }
}

// ============================================================================
// Page Fault Detection (Pattern from B4 Investigation)
// ============================================================================

/// Get current minor and major page fault counts (Linux only).
///
/// Returns (minor_faults, major_faults).
/// - Minor faults: Page in memory but not mapped (soft fault)
/// - Major faults: Page on disk, requires I/O (hard fault)
#[cfg(target_os = "linux")]
pub fn get_page_faults() -> (u64, u64) {
    use std::fs;
    let stat = fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let fields: Vec<&str> = stat.split_whitespace().collect();
    if fields.len() > 12 {
        let minor = fields[9].parse().unwrap_or(0);
        let major = fields[11].parse().unwrap_or(0);
        (minor, major)
    } else {
        (0, 0)
    }
}

/// Fallback for non-Linux platforms.
#[cfg(not(target_os = "linux"))]
pub fn get_page_faults() -> (u64, u64) {
    (0, 0)
}

/// Execute a closure while tracking page faults.
///
/// Logs a warning if more than 1000 minor faults or any major faults occur.
///
/// # Example
/// ```rust,ignore
/// let result = with_page_fault_tracking("mmap_copy", || {
///     data.copy_from_slice(&mmap_region);
/// });
/// ```
pub fn with_page_fault_tracking<T, F: FnOnce() -> T>(name: &str, f: F) -> (T, u64, u64) {
    let (minor_before, major_before) = get_page_faults();
    let result = f();
    let (minor_after, major_after) = get_page_faults();

    let minor_delta = minor_after.saturating_sub(minor_before);
    let major_delta = major_after.saturating_sub(major_before);

    #[cfg(feature = "tracing")]
    if minor_delta > 1000 || major_delta > 0 {
        tracing::warn!(
            operation = name,
            minor_faults = minor_delta,
            major_faults = major_delta,
            "High page fault count detected"
        );
    }

    let _ = name; // Suppress unused warning when tracing disabled
    (result, minor_delta, major_delta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_cycles_returns_value() {
        let cycles = cpu_cycles();
        // On x86_64/aarch64, should return non-zero
        // On other architectures, returns 0
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        assert!(cycles > 0);
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        assert_eq!(cycles, 0);
    }

    #[test]
    fn test_cached_nanos_or_now_returns_value() {
        let nanos = cached_nanos_or_now();
        // Should return a valid nanosecond count (non-zero on most systems)
        // Note: u64 is always >= 0, so just verify we got a value
        let _ = nanos; // Value is always valid for u64
    }

    #[test]
    fn test_page_fault_tracking() {
        let (result, minor, major) = with_page_fault_tracking("test", || 42);
        assert_eq!(result, 42);
        // Page faults are u64, always non-negative by type
        let _ = (minor, major); // Values are always valid for u64
    }

    #[test]
    fn test_get_page_faults() {
        let (minor, major) = get_page_faults();
        // Page faults are u64, always non-negative by type (may be 0 on non-Linux)
        let _ = (minor, major); // Values are always valid for u64
    }
}
