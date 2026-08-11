//! CPU information and detection.

use serde::{Deserialize, Serialize};

use super::simd::SimdCapability;

/// CPU information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CpuInfo {
    /// Number of physical cores
    pub cores: u32,
    /// Number of logical threads (with hyperthreading)
    pub threads: u32,
    /// SIMD capability
    pub simd: SimdCapability,
    /// CPU model name
    pub model: String,
    /// Cache size in bytes (L3 or total)
    pub cache_bytes: u64,
}

impl CpuInfo {
    /// Create new CPU info
    pub fn new(cores: u32, threads: u32, simd: SimdCapability, model: impl Into<String>) -> Self {
        Self { cores, threads, simd, model: model.into(), cache_bytes: 0 }
    }

    /// Set cache size
    pub fn with_cache(mut self, cache_bytes: u64) -> Self {
        self.cache_bytes = cache_bytes;
        self
    }

    /// Reconcile a machine-wide physical core count against the parallelism this
    /// process is actually allowed to use.
    ///
    /// `available_parallelism` is cgroup- and affinity-aware: it reports what THIS
    /// process may run on. `/proc/cpuinfo` is not — it describes the whole machine
    /// regardless of any restriction. Comparing the two directly mixes denominators,
    /// and under any CPU restriction the machine-wide figure is both larger than the
    /// usable one and useless for sizing work.
    ///
    /// Clamping keeps `cores <= threads` true by construction and makes `cores` mean
    /// "physical cores this process can actually use", which is what every consumer
    /// of the field (thread-pool sizing, efficiency estimates) needs.
    pub(super) fn usable_cores(detected_physical: Option<u32>, threads: u32) -> u32 {
        let threads = threads.max(1);
        detected_physical.map_or(threads, |physical| physical.clamp(1, threads))
    }

    /// Detect current CPU information
    pub fn detect() -> Self {
        // Get logical CPU count using standard library. This is cgroup/affinity aware.
        let threads = std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(1);

        // Physical cores come from /proc/cpuinfo, which is NOT restriction-aware, so
        // the result is reconciled against `threads` — see `usable_cores`.
        let cores = Self::usable_cores(Self::detect_physical_cores(), threads);
        let simd = SimdCapability::detect();

        // Try to get CPU model name
        let model = Self::detect_model();

        Self {
            cores,
            threads,
            simd,
            model,
            cache_bytes: 0, // Would need platform-specific APIs
        }
    }

    /// Detect physical core count (Linux-specific)
    #[cfg(target_os = "linux")]
    fn detect_physical_cores() -> Option<u32> {
        std::fs::read_to_string("/proc/cpuinfo").ok().map(|info| {
            // Count unique core IDs
            let mut core_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut current_physical_id = String::new();

            for line in info.lines() {
                if line.starts_with("physical id") {
                    current_physical_id =
                        line.split(':').nth(1).map(|s| s.trim().to_string()).unwrap_or_default();
                } else if line.starts_with("core id") {
                    let core_id =
                        line.split(':').nth(1).map(|s| s.trim().to_string()).unwrap_or_default();
                    core_ids.insert(format!("{current_physical_id}-{core_id}"));
                }
            }

            if core_ids.is_empty() {
                // Fallback: count processor entries
                info.lines().filter(|line| line.starts_with("processor")).count() as u32
            } else {
                core_ids.len() as u32
            }
        })
    }

    /// Detect physical core count (macOS-specific)
    #[cfg(target_os = "macos")]
    fn detect_physical_cores() -> Option<u32> {
        std::process::Command::new("sysctl")
            .args(["-n", "hw.physicalcpu"])
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|s| s.trim().parse().ok())
    }

    /// Detect physical core count (fallback)
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn detect_physical_cores() -> Option<u32> {
        None
    }

    /// Detect CPU model name
    #[cfg(target_os = "linux")]
    fn detect_model() -> String {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|info| {
                info.lines()
                    .find(|line| line.starts_with("model name"))
                    .and_then(|line| line.split(':').nth(1))
                    .map(|s| s.trim().to_string())
            })
            .unwrap_or_else(|| "Unknown CPU".to_string())
    }

    #[cfg(target_os = "macos")]
    fn detect_model() -> String {
        std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "Unknown CPU".to_string())
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn detect_model() -> String {
        "Unknown CPU".to_string()
    }

    /// Estimate memory bandwidth based on core count (rough approximation)
    pub fn estimated_memory_bandwidth_gbps(&self) -> f64 {
        // Rough estimate: ~20 GB/s per channel, assume 2 channels for desktop
        40.0 * (f64::from(self.cores) / 8.0).min(2.0)
    }
}
