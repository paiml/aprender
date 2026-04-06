//! CPU detection for accurate theoretical peak calculation.

/// Detected CPU capabilities for theoretical peak calculation
#[derive(Debug, Clone)]
pub struct CpuCapabilities {
    /// Number of physical cores
    pub cores: usize,
    /// Max frequency in MHz
    pub max_freq_mhz: u32,
    /// AVX-512 support
    pub has_avx512: bool,
    /// AVX2 support
    pub has_avx2: bool,
    /// L1 data cache size in bytes
    pub l1d_cache: usize,
    /// L2 cache size in bytes
    pub l2_cache: usize,
    /// L3 cache size in bytes
    pub l3_cache: usize,
    /// Memory bandwidth estimate in GB/s
    pub mem_bandwidth_gbs: f64,
}

impl Default for CpuCapabilities {
    fn default() -> Self {
        Self::detect()
    }
}

impl CpuCapabilities {
    /// Detect CPU capabilities at runtime
    pub fn detect() -> Self {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        // Use CPUID to detect features
        #[cfg(target_arch = "x86_64")]
        let (has_avx512, has_avx2) = {
            (
                is_x86_feature_detected!("avx512f"),
                is_x86_feature_detected!("avx2"),
            )
        };

        #[cfg(not(target_arch = "x86_64"))]
        let (has_avx512, has_avx2) = (false, false);

        // Estimate max frequency (conservative default, can be improved with sysfs)
        let max_freq_mhz = Self::detect_max_freq().unwrap_or(3500);

        // Estimate cache sizes (conservative defaults for desktop CPUs)
        // These could be read from /sys/devices/system/cpu/cpu0/cache on Linux
        let (l1d_cache, l2_cache, l3_cache) = Self::detect_cache_sizes();

        // Estimate memory bandwidth based on core count
        // Conservative: ~4 GB/s per core for DDR4, ~6 GB/s for DDR5
        let mem_bandwidth_gbs = (cores as f64) * 4.0;

        Self {
            cores,
            max_freq_mhz,
            has_avx512,
            has_avx2,
            l1d_cache,
            l2_cache,
            l3_cache,
            mem_bandwidth_gbs,
        }
    }

    /// Detect maximum CPU frequency from sysfs
    fn detect_max_freq() -> Option<u32> {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) =
                std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
            {
                // cpuinfo_max_freq is in kHz
                return content.trim().parse::<u32>().ok().map(|khz| khz / 1000);
            }
        }
        None
    }

    /// Detect cache sizes from sysfs
    fn detect_cache_sizes() -> (usize, usize, usize) {
        #[cfg(target_os = "linux")]
        {
            let l1d = Self::read_cache_size(0, 0).unwrap_or(32 * 1024); // 32 KB default
            let l2 = Self::read_cache_size(0, 2).unwrap_or(512 * 1024); // 512 KB default
            let l3 = Self::read_cache_size(0, 3).unwrap_or(32 * 1024 * 1024); // 32 MB default
            (l1d, l2, l3)
        }

        #[cfg(not(target_os = "linux"))]
        {
            (32 * 1024, 512 * 1024, 32 * 1024 * 1024)
        }
    }

    #[cfg(target_os = "linux")]
    fn read_cache_size(cpu: u32, index: u32) -> Option<usize> {
        let path = format!(
            "/sys/devices/system/cpu/cpu{}/cache/index{}/size",
            cpu, index
        );
        if let Ok(content) = std::fs::read_to_string(&path) {
            let s = content.trim();
            if let Some(kb_str) = s.strip_suffix('K') {
                return kb_str.parse::<usize>().ok().map(|kb| kb * 1024);
            } else if let Some(mb_str) = s.strip_suffix('M') {
                return mb_str.parse::<usize>().ok().map(|mb| mb * 1024 * 1024);
            }
        }
        None
    }

    /// Calculate theoretical peak GFLOP/s for compute-bound operations
    pub fn compute_peak_gflops(&self) -> f64 {
        let freq_ghz = self.max_freq_mhz as f64 / 1000.0;

        // f32 FLOPs per cycle per core
        let flops_per_cycle = if self.has_avx512 {
            // AVX-512: 2 × 512-bit FMA units = 2 × 16 × 2 = 64 FLOPs/cycle (theoretical)
            // Most CPUs have 2 AVX-512 units, but frequency drops, so use conservative 32
            32.0
        } else if self.has_avx2 {
            // AVX2: 2 × 256-bit FMA units = 2 × 8 × 2 = 32 FLOPs/cycle (theoretical)
            // Conservative: single FMA port = 16
            16.0
        } else {
            // SSE: 4 FLOPs/cycle
            4.0
        };

        self.cores as f64 * freq_ghz * flops_per_cycle
    }

    /// Calculate theoretical peak GFLOP/s for memory-bound operations
    /// bytes_per_flop: number of bytes that must be transferred per FLOP
    pub fn memory_peak_gflops(&self, bytes_per_flop: f64) -> f64 {
        self.mem_bandwidth_gbs / bytes_per_flop
    }

    /// Calculate theoretical peak for a given size (cache vs memory bound)
    /// Uses bytes_per_flop to estimate total working set (includes all arrays)
    pub fn theoretical_peak_for_size(
        &self,
        size: usize,
        _bytes_per_element: usize,
        bytes_per_flop: f64,
    ) -> f64 {
        // Calculate working set using bytes_per_flop which accounts for all arrays
        // e.g., elementwise_mul: 12 bytes/FLOP = 3 arrays × 4 bytes
        // This gives accurate cache behavior estimation
        let working_set_bytes = (size as f64 * bytes_per_flop) as usize;

        // Determine which cache level (if any) the data fits in
        // Use 80% of cache as threshold to account for other data
        if working_set_bytes < (self.l1d_cache * 80 / 100) {
            // L1 cache: effectively compute-bound
            self.compute_peak_gflops()
        } else if working_set_bytes < (self.l2_cache * 80 / 100) {
            // L2 cache: ~50% of compute peak
            self.compute_peak_gflops() * 0.5
        } else if working_set_bytes < (self.l3_cache * 80 / 100) {
            // L3 cache: ~25% of compute peak
            self.compute_peak_gflops() * 0.25
        } else {
            // Main memory: memory-bound
            self.memory_peak_gflops(bytes_per_flop)
        }
    }
}
