//! CPU Frequency Control Backend (PMAT-038)
//!
//! Interface with Linux cpufreq to lock CPU frequency for deterministic benchmarks.
//!
//! # Features
//!
//! - Read current CPU frequency
//! - Detect CPU governor (performance/powersave/ondemand)
//! - Frequency lock with RAII guard
//! - Measure variance before/after locking
//!
//! # Falsification Criteria (F1301-F1310)
//!
//! See `tests/frequency_control_f1301.rs` for falsification tests.

mod controller;
mod governor;
mod info;
mod lock;
mod reading;
mod sysfs;
mod variance;

pub use controller::FrequencyController;
pub use governor::CpuGovernor;
pub use info::CpuFrequencyInfo;
pub use lock::FrequencyLock;
pub use reading::FrequencyReading;
pub use variance::FrequencyVariance;

/// Get number of CPUs
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests;
