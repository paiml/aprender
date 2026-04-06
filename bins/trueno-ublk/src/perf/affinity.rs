//! CPU affinity management for worker threads
//!
//! Pins ublk worker threads to dedicated CPU cores for:
//!
//! Benefits:
//! - Reduced cache misses (data stays in L1/L2)
//! - Predictable latency (no core migration)
//! - Better NUMA locality when combined with numa.rs
//!
//! ## Core Selection Strategy
//!
//! 1. **Avoid core 0**: Often handles interrupts and kernel tasks
//! 2. **Prefer physical cores**: Avoid SMT siblings for latency-sensitive work
//! 3. **Same NUMA node**: Keep threads on same node as memory
//!
//! ## Usage
//!
//! ```ignore
//! let affinity = CpuAffinity::auto_select(4)?;
//! affinity.pin_current_thread()?;
//!
//! // Or specify cores explicitly
//! let affinity = CpuAffinity::new(vec![1, 2, 3, 4]);
//! ```

use std::io;
use thiserror::Error;

/// Errors from CPU affinity operations
#[derive(Error, Debug)]
pub enum AffinityError {
    #[error("Failed to set CPU affinity: {0}")]
    SetAffinity(io::Error),

    #[error("Failed to get CPU affinity: {0}")]
    GetAffinity(io::Error),

    #[error("No CPUs available")]
    NoCpus,

    #[error("Invalid core ID: {0}")]
    InvalidCore(usize),

    #[error("Core {0} is not online")]
    CoreOffline(usize),
}

/// CPU affinity configuration
#[derive(Debug, Clone, Default)]
pub struct CpuAffinity {
    /// CPU cores to use
    cores: Vec<usize>,
}

impl CpuAffinity {
    /// Create affinity with specific cores
    pub fn new(cores: Vec<usize>) -> Self {
        Self { cores }
    }

    /// Get the cores
    pub fn cores(&self) -> &[usize] {
        &self.cores
    }

    /// Number of cores
    pub fn len(&self) -> usize {
        self.cores.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.cores.is_empty()
    }

    /// Auto-select optimal cores for given thread count
    ///
    /// Strategy:
    /// 1. Skip core 0 (often busy with interrupts)
    /// 2. Prefer cores 1..num_threads+1
    /// 3. Wrap around if not enough cores
    pub fn auto_select(num_threads: usize) -> Result<Self, AffinityError> {
        let num_cpus = Self::get_num_cpus();
        if num_cpus == 0 {
            return Err(AffinityError::NoCpus);
        }

        let mut cores = Vec::with_capacity(num_threads);

        // Start from core 1, skip core 0
        for i in 0..num_threads {
            let core = if num_cpus > 1 {
                1 + (i % (num_cpus - 1))
            } else {
                0 // Single CPU system
            };
            cores.push(core);
        }

        Ok(Self { cores })
    }

    /// Auto-select avoiding specific cores
    pub fn auto_select_avoiding(
        num_threads: usize,
        avoid: &[usize],
    ) -> Result<Self, AffinityError> {
        let num_cpus = Self::get_num_cpus();
        if num_cpus == 0 {
            return Err(AffinityError::NoCpus);
        }

        let available: Vec<usize> = (0..num_cpus).filter(|c| !avoid.contains(c)).collect();

        if available.is_empty() {
            return Err(AffinityError::NoCpus);
        }

        let mut cores = Vec::with_capacity(num_threads);
        for i in 0..num_threads {
            cores.push(available[i % available.len()]);
        }

        Ok(Self { cores })
    }

    /// Pin current thread to the configured cores
    #[cfg(target_os = "linux")]
    pub fn pin_current_thread(&self) -> Result<(), AffinityError> {
        use nix::sched::{sched_setaffinity, CpuSet};
        use nix::unistd::Pid;

        if self.cores.is_empty() {
            return Ok(()); // No-op if no cores specified
        }

        let mut cpu_set = CpuSet::new();
        for &core in &self.cores {
            cpu_set.set(core).map_err(|_| AffinityError::InvalidCore(core))?;
        }

        sched_setaffinity(Pid::from_raw(0), &cpu_set)
            .map_err(|e| AffinityError::SetAffinity(io::Error::from_raw_os_error(e as i32)))?;

        Ok(())
    }

    /// Pin current thread (non-Linux stub)
    #[cfg(not(target_os = "linux"))]
    pub fn pin_current_thread(&self) -> Result<(), AffinityError> {
        // No-op on non-Linux
        Ok(())
    }

    /// Pin a specific thread by thread ID
    #[cfg(target_os = "linux")]
    pub fn pin_thread(&self, tid: i32) -> Result<(), AffinityError> {
        use nix::sched::{sched_setaffinity, CpuSet};
        use nix::unistd::Pid;

        if self.cores.is_empty() {
            return Ok(());
        }

        let mut cpu_set = CpuSet::new();
        for &core in &self.cores {
            cpu_set.set(core).map_err(|_| AffinityError::InvalidCore(core))?;
        }

        sched_setaffinity(Pid::from_raw(tid), &cpu_set)
            .map_err(|e| AffinityError::SetAffinity(io::Error::from_raw_os_error(e as i32)))?;

        Ok(())
    }

    /// Pin a specific thread (non-Linux stub)
    #[cfg(not(target_os = "linux"))]
    pub fn pin_thread(&self, _tid: i32) -> Result<(), AffinityError> {
        Ok(())
    }

    /// Get current thread's affinity
    #[cfg(target_os = "linux")]
    pub fn get_current_affinity() -> Result<Vec<usize>, AffinityError> {
        use nix::sched::{sched_getaffinity, CpuSet};
        use nix::unistd::Pid;

        let cpu_set = sched_getaffinity(Pid::from_raw(0))
            .map_err(|e| AffinityError::GetAffinity(io::Error::from_raw_os_error(e as i32)))?;

        let mut cores = Vec::new();
        for i in 0..CpuSet::count() {
            if cpu_set.is_set(i).unwrap_or(false) {
                cores.push(i);
            }
        }

        Ok(cores)
    }

    /// Get current thread's affinity (non-Linux stub)
    #[cfg(not(target_os = "linux"))]
    pub fn get_current_affinity() -> Result<Vec<usize>, AffinityError> {
        let num_cpus = Self::get_num_cpus();
        Ok((0..num_cpus).collect())
    }

    /// Get number of available CPUs
    pub fn get_num_cpus() -> usize {
        std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1)
    }

    /// Check if a core is online
    #[cfg(target_os = "linux")]
    pub fn is_core_online(core: usize) -> bool {
        let path = format!("/sys/devices/system/cpu/cpu{}/online", core);
        std::fs::read_to_string(&path).map(|s| s.trim() == "1").unwrap_or(core == 0)
        // Core 0 is always online
    }

    /// Check if a core is online (non-Linux stub)
    #[cfg(not(target_os = "linux"))]
    pub fn is_core_online(core: usize) -> bool {
        core < Self::get_num_cpus()
    }

    /// Validate that all cores are valid and online
    pub fn validate(&self) -> Result<(), AffinityError> {
        let num_cpus = Self::get_num_cpus();

        for &core in &self.cores {
            if core >= num_cpus {
                return Err(AffinityError::InvalidCore(core));
            }
            if !Self::is_core_online(core) {
                return Err(AffinityError::CoreOffline(core));
            }
        }

        Ok(())
    }

    /// Get core for thread index (round-robin)
    pub fn core_for_thread(&self, thread_idx: usize) -> Option<usize> {
        if self.cores.is_empty() {
            None
        } else {
            Some(self.cores[thread_idx % self.cores.len()])
        }
    }

    /// Create affinity that spreads threads across all available cores
    pub fn spread(num_threads: usize) -> Result<Self, AffinityError> {
        let num_cpus = Self::get_num_cpus();
        if num_cpus == 0 {
            return Err(AffinityError::NoCpus);
        }

        let mut cores = Vec::with_capacity(num_threads);
        for i in 0..num_threads {
            cores.push(i % num_cpus);
        }

        Ok(Self { cores })
    }
}

impl From<Vec<usize>> for CpuAffinity {
    fn from(cores: Vec<usize>) -> Self {
        Self::new(cores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // CpuAffinity Construction Tests
    // ============================================================================

    #[test]
    fn test_cpu_affinity_new() {
        let affinity = CpuAffinity::new(vec![1, 2, 3]);
        assert_eq!(affinity.cores(), &[1, 2, 3]);
        assert_eq!(affinity.len(), 3);
        assert!(!affinity.is_empty());
    }

    #[test]
    fn test_cpu_affinity_default() {
        let affinity = CpuAffinity::default();
        assert!(affinity.is_empty());
        assert_eq!(affinity.len(), 0);
    }

    #[test]
    fn test_cpu_affinity_from_vec() {
        let affinity: CpuAffinity = vec![1, 2].into();
        assert_eq!(affinity.cores(), &[1, 2]);
    }

    // ============================================================================
    // Auto-selection Tests
    // ============================================================================

    #[test]
    fn test_cpu_affinity_auto_select() {
        let affinity = CpuAffinity::auto_select(4).unwrap();
        assert_eq!(affinity.len(), 4);

        // Should skip core 0 if more than 1 CPU
        let num_cpus = CpuAffinity::get_num_cpus();
        if num_cpus > 1 {
            assert!(!affinity.cores().contains(&0));
        }
    }

    #[test]
    fn test_cpu_affinity_auto_select_more_threads_than_cores() {
        let num_cpus = CpuAffinity::get_num_cpus();
        let affinity = CpuAffinity::auto_select(num_cpus * 2).unwrap();
        assert_eq!(affinity.len(), num_cpus * 2);
    }

    #[test]
    fn test_cpu_affinity_auto_select_avoiding() {
        let affinity = CpuAffinity::auto_select_avoiding(4, &[0, 1]).unwrap();
        assert_eq!(affinity.len(), 4);
        assert!(!affinity.cores().contains(&0));
        assert!(!affinity.cores().contains(&1));
    }

    #[test]
    fn test_cpu_affinity_spread() {
        let affinity = CpuAffinity::spread(8).unwrap();
        assert_eq!(affinity.len(), 8);
    }

    // ============================================================================
    // Core Selection Tests
    // ============================================================================

    #[test]
    fn test_cpu_affinity_core_for_thread() {
        let affinity = CpuAffinity::new(vec![2, 4, 6]);

        assert_eq!(affinity.core_for_thread(0), Some(2));
        assert_eq!(affinity.core_for_thread(1), Some(4));
        assert_eq!(affinity.core_for_thread(2), Some(6));
        assert_eq!(affinity.core_for_thread(3), Some(2)); // Wraps around
    }

    #[test]
    fn test_cpu_affinity_core_for_thread_empty() {
        let affinity = CpuAffinity::default();
        assert_eq!(affinity.core_for_thread(0), None);
    }

    // ============================================================================
    // Validation Tests
    // ============================================================================

    #[test]
    fn test_cpu_affinity_validate_success() {
        let num_cpus = CpuAffinity::get_num_cpus();
        if num_cpus > 0 {
            let affinity = CpuAffinity::new(vec![0]);
            assert!(affinity.validate().is_ok());
        }
    }

    #[test]
    fn test_cpu_affinity_validate_invalid_core() {
        let affinity = CpuAffinity::new(vec![99999]);
        let result = affinity.validate();
        assert!(matches!(result, Err(AffinityError::InvalidCore(99999))));
    }

    #[test]
    fn test_cpu_affinity_validate_empty() {
        let affinity = CpuAffinity::default();
        assert!(affinity.validate().is_ok()); // Empty is valid (no-op)
    }

    // ============================================================================
    // System Info Tests
    // ============================================================================

    #[test]
    fn test_get_num_cpus() {
        let num = CpuAffinity::get_num_cpus();
        assert!(num > 0);
    }

    #[test]
    fn test_is_core_online() {
        // Core 0 should always be online
        assert!(CpuAffinity::is_core_online(0));
    }

    #[test]
    fn test_get_current_affinity() {
        let result = CpuAffinity::get_current_affinity();
        assert!(result.is_ok());
        let cores = result.unwrap();
        assert!(!cores.is_empty());
    }

    // ============================================================================
    // Pin Tests (integration - may require privileges)
    // ============================================================================

    #[test]
    fn test_pin_current_thread_empty() {
        // Empty affinity should be no-op
        let affinity = CpuAffinity::default();
        assert!(affinity.pin_current_thread().is_ok());
    }

    #[test]
    fn test_pin_thread_empty() {
        let affinity = CpuAffinity::default();
        assert!(affinity.pin_thread(0).is_ok());
    }

    // ============================================================================
    // Error Tests
    // ============================================================================

    #[test]
    fn test_affinity_error_display() {
        let err = AffinityError::InvalidCore(42);
        let msg = format!("{}", err);
        assert!(msg.contains("42"));
    }

    #[test]
    fn test_affinity_error_debug() {
        let err = AffinityError::NoCpus;
        let debug = format!("{:?}", err);
        assert!(debug.contains("NoCpus"));
    }

    #[test]
    fn test_affinity_error_variants() {
        // Test all error variants can be constructed
        let _ = AffinityError::SetAffinity(io::Error::from_raw_os_error(1));
        let _ = AffinityError::GetAffinity(io::Error::from_raw_os_error(1));
        let _ = AffinityError::NoCpus;
        let _ = AffinityError::InvalidCore(0);
        let _ = AffinityError::CoreOffline(0);
    }

    // ============================================================================
    // Clone Tests
    // ============================================================================

    #[test]
    fn test_cpu_affinity_clone() {
        let affinity = CpuAffinity::new(vec![1, 2, 3]);
        let cloned = affinity.clone();
        assert_eq!(affinity.cores(), cloned.cores());
    }

    // ============================================================================
    // Integration-style Tests
    // ============================================================================

    #[test]
    fn test_worker_thread_pattern() {
        // Simulate worker thread core assignment
        let num_workers = 4;
        let affinity = CpuAffinity::auto_select(num_workers).unwrap();

        for worker_id in 0..num_workers {
            let core = affinity.core_for_thread(worker_id);
            assert!(core.is_some());
        }
    }

    #[test]
    fn test_avoiding_interrupt_core() {
        // Common pattern: avoid core 0 which handles interrupts
        let affinity = CpuAffinity::auto_select_avoiding(4, &[0]);
        if let Ok(a) = affinity {
            let num_cpus = CpuAffinity::get_num_cpus();
            if num_cpus > 1 {
                assert!(!a.cores().contains(&0));
            }
        }
    }
}
