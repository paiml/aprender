//! PTX Registry and Statistics Types
//!
//! Aggregated statistics for brick profiling and PTX kernel tracking.

use std::collections::HashMap;

use super::BrickBottleneck;

/// PTX kernel registry for execution graph correlation.
///
/// PAR-201: Maps PTX hashes to source code for debugging and analysis.
#[derive(Debug, Default)]
pub struct PtxRegistry {
    /// Hash → (kernel_name, ptx_source, file_path)
    kernels: HashMap<u64, (String, String, Option<std::path::PathBuf>)>,
}

impl PtxRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register PTX source code.
    ///
    /// # Arguments
    /// - `name`: Kernel name (e.g., "batched_q4k_gemv")
    /// - `ptx`: PTX source code
    /// - `path`: Optional file path for source correlation
    pub fn register(&mut self, name: &str, ptx: &str, path: Option<&std::path::Path>) {
        debug_assert!(!name.is_empty(), "CB-BUDGET: kernel name must not be empty");
        debug_assert!(!ptx.is_empty(), "CB-BUDGET: PTX source must not be empty");
        let hash = Self::hash_ptx(ptx);
        self.kernels
            .insert(hash, (name.to_string(), ptx.to_string(), path.map(|p| p.to_path_buf())));
    }

    /// Compute FNV-1a hash of PTX source.
    #[inline]
    pub fn hash_ptx(ptx: &str) -> u64 {
        // FNV-1a hash
        let mut hash: u64 = 0xcbf29ce484222325;
        for byte in ptx.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    /// Lookup PTX source by hash.
    pub fn lookup(&self, hash: u64) -> Option<&str> {
        self.kernels.get(&hash).map(|(_, ptx, _)| ptx.as_str())
    }

    /// Lookup kernel name by hash.
    pub fn lookup_name(&self, hash: u64) -> Option<&str> {
        self.kernels.get(&hash).map(|(name, _, _)| name.as_str())
    }

    /// Lookup file path by hash.
    pub fn lookup_path(&self, hash: u64) -> Option<&std::path::Path> {
        self.kernels.get(&hash).and_then(|(_, _, path)| path.as_deref())
    }

    /// Get all registered hashes.
    pub fn hashes(&self) -> impl Iterator<Item = u64> + '_ {
        self.kernels.keys().copied()
    }

    /// Number of registered kernels.
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }
}

/// Aggregated statistics for a brick category.
#[derive(Debug, Clone, Copy, Default)]
pub struct CategoryStats {
    /// Total elapsed time (nanoseconds)
    pub total_ns: u64,
    /// Total elements processed
    pub total_elements: u64,
    /// Total samples
    pub count: u64,
}

impl CategoryStats {
    /// Average time per sample in microseconds.
    #[inline]
    pub fn avg_us(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_ns as f64 / self.count as f64 / 1000.0
        }
    }

    /// Throughput in elements per second.
    #[inline]
    pub fn throughput(&self) -> f64 {
        if self.total_ns == 0 {
            0.0
        } else {
            self.total_elements as f64 / (self.total_ns as f64 / 1_000_000_000.0)
        }
    }

    /// Percentage of total time (given total_ns across all categories).
    #[inline]
    pub fn percentage(&self, total: u64) -> f64 {
        if total == 0 {
            0.0
        } else {
            100.0 * self.total_ns as f64 / total as f64
        }
    }
}

/// Accumulated per-brick statistics.
#[derive(Debug, Clone, Default)]
pub struct BrickStats {
    /// Brick name
    pub name: String,
    /// Total samples collected
    pub count: u64,
    /// Total elapsed time (nanoseconds)
    pub total_ns: u64,
    /// Min elapsed time (nanoseconds)
    pub min_ns: u64,
    /// Max elapsed time (nanoseconds)
    pub max_ns: u64,
    /// Total elements processed
    pub total_elements: u64,
    /// PMAT-451: Total bytes processed (for throughput calculation)
    pub total_bytes: u64,
    /// PMAT-451: Total compressed bytes (for compression ratio)
    pub total_compressed_bytes: u64,
    /// PMAT-451: Bottleneck classification
    pub bottleneck: BrickBottleneck,
    /// Phase 11 (E.9.2): Total CPU cycles (from RDTSCP/CNTVCT)
    pub total_cycles: u64,
    /// Phase 11: Minimum CPU cycles observed
    pub min_cycles: u64,
    /// Phase 11: Maximum CPU cycles observed
    pub max_cycles: u64,
}

impl BrickStats {
    /// Create new stats for a brick.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            count: 0,
            total_ns: 0,
            min_ns: u64::MAX,
            max_ns: 0,
            total_elements: 0,
            total_bytes: 0,
            total_compressed_bytes: 0,
            bottleneck: BrickBottleneck::Unknown,
            total_cycles: 0,
            min_cycles: u64::MAX,
            max_cycles: 0,
        }
    }

    /// Add a sample to statistics.
    pub fn add_sample(&mut self, elapsed_ns: u64, elements: u64) {
        debug_assert!(elements > 0, "CB-BUDGET: elements must be > 0");
        self.count += 1;
        self.total_ns += elapsed_ns;
        self.min_ns = self.min_ns.min(elapsed_ns);
        self.max_ns = self.max_ns.max(elapsed_ns);
        self.total_elements += elements;
    }

    /// Phase 11 (E.9.2): Add a sample with CPU cycle count.
    ///
    /// Use this for frequency-invariant performance analysis.
    /// Cycles are immune to CPU frequency scaling (turbo boost).
    pub fn add_sample_with_cycles(&mut self, elapsed_ns: u64, elements: u64, cycles: u64) {
        self.add_sample(elapsed_ns, elements);
        self.total_cycles += cycles;
        self.min_cycles = self.min_cycles.min(cycles);
        self.max_cycles = self.max_cycles.max(cycles);
    }

    /// Phase 11: Cycles per element (frequency-invariant throughput metric).
    ///
    /// Lower is better. This metric is immune to CPU frequency scaling.
    #[must_use]
    pub fn cycles_per_element(&self) -> f64 {
        if self.total_elements == 0 {
            0.0
        } else {
            self.total_cycles as f64 / self.total_elements as f64
        }
    }

    /// Phase 11: Average cycles per sample.
    #[must_use]
    pub fn avg_cycles(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_cycles as f64 / self.count as f64
        }
    }

    /// Phase 11: Estimated IPC (Instructions Per Cycle).
    ///
    /// Approximation assuming ~1 instruction per element for simple ops.
    /// - Low IPC (<1.0): Memory stalls (cache misses, memory latency)
    /// - High IPC (>2.0): Compute bound (efficient execution)
    #[must_use]
    pub fn estimated_ipc(&self) -> f64 {
        if self.total_cycles == 0 {
            0.0
        } else {
            // Rough approximation: assume 1 instruction per element
            self.total_elements as f64 / self.total_cycles as f64
        }
    }

    /// Phase 11: Diagnose bottleneck based on cycles vs time ratio.
    ///
    /// High cycles + low time = likely cache misses
    /// Low cycles + high time = likely CPU throttling or context switches
    #[must_use]
    pub fn diagnose_from_cycles(&self) -> &'static str {
        if self.total_cycles == 0 || self.total_ns == 0 {
            return "insufficient data";
        }

        let ipc = self.estimated_ipc();
        let ns_per_cycle = self.total_ns as f64 / self.total_cycles as f64;

        // Typical CPU runs at ~3GHz, so 1 cycle ≈ 0.33ns
        // If ns_per_cycle >> 0.33, we're seeing stalls or throttling
        if ipc < 0.5 {
            "memory-bound (low IPC, likely cache misses)"
        } else if ipc > 2.0 {
            "compute-bound (efficient)"
        } else if ns_per_cycle > 1.0 {
            "throttled or context-switched"
        } else {
            "balanced"
        }
    }

    /// PMAT-451: Add a sample with byte metrics for compression workloads.
    ///
    /// # Arguments
    /// - `elapsed_ns`: Time taken in nanoseconds
    /// - `elements`: Number of elements processed (e.g., pages)
    /// - `input_bytes`: Original uncompressed size
    /// - `output_bytes`: Compressed output size
    pub fn add_sample_with_bytes(
        &mut self,
        elapsed_ns: u64,
        elements: u64,
        input_bytes: u64,
        output_bytes: u64,
    ) {
        self.add_sample(elapsed_ns, elements);
        self.total_bytes += input_bytes;
        self.total_compressed_bytes += output_bytes;
    }

    /// PMAT-451: Calculate compression ratio (input_size / output_size).
    /// Returns 1.0 if no compression data available.
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.total_compressed_bytes == 0 {
            1.0
        } else {
            self.total_bytes as f64 / self.total_compressed_bytes as f64
        }
    }

    /// PMAT-451: Calculate throughput in GB/s.
    /// Based on total input bytes processed.
    #[must_use]
    pub fn throughput_gbps(&self) -> f64 {
        if self.total_ns == 0 {
            0.0
        } else {
            let bytes_per_ns = self.total_bytes as f64 / self.total_ns as f64;
            bytes_per_ns * 1e9 / 1e9 // Convert to GB/s (ns to sec, bytes to GB)
        }
    }

    /// PMAT-451: Set bottleneck classification.
    pub fn set_bottleneck(&mut self, bottleneck: BrickBottleneck) {
        self.bottleneck = bottleneck;
    }

    /// PMAT-451: Get bottleneck classification.
    #[must_use]
    pub fn get_bottleneck(&self) -> BrickBottleneck {
        self.bottleneck
    }

    /// Average time in microseconds.
    #[must_use]
    pub fn avg_us(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_ns as f64 / self.count as f64 / 1000.0
        }
    }

    /// Throughput in elements/second.
    #[must_use]
    pub fn throughput(&self) -> f64 {
        if self.total_ns == 0 {
            0.0
        } else {
            self.total_elements as f64 / (self.total_ns as f64 / 1_000_000_000.0)
        }
    }

    /// Throughput in tokens/second (alias for throughput).
    #[must_use]
    pub fn tokens_per_sec(&self) -> f64 {
        self.throughput()
    }

    /// Minimum time in microseconds.
    #[must_use]
    pub fn min_us(&self) -> f64 {
        if self.min_ns == u64::MAX {
            0.0
        } else {
            self.min_ns as f64 / 1000.0
        }
    }

    /// Maximum time in microseconds.
    #[must_use]
    pub fn max_us(&self) -> f64 {
        self.max_ns as f64 / 1000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // PtxRegistry Tests
    // =========================================================================

    #[test]
    fn test_ptx_registry_new_is_empty() {
        let reg = PtxRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_ptx_registry_register_and_lookup() {
        let mut reg = PtxRegistry::new();
        let ptx = ".version 8.0\n.entry gemm_tiled {}";
        reg.register("gemm_tiled", ptx, None);

        assert_eq!(reg.len(), 1);
        assert!(!reg.is_empty());

        let hash = PtxRegistry::hash_ptx(ptx);
        assert_eq!(reg.lookup(hash), Some(ptx));
        assert_eq!(reg.lookup_name(hash), Some("gemm_tiled"));
        assert_eq!(reg.lookup_path(hash), None);
    }

    #[test]
    fn test_ptx_registry_register_with_path() {
        let mut reg = PtxRegistry::new();
        let ptx = ".version 8.0\n.entry softmax {}";
        let path = std::path::Path::new("/src/kernels/softmax.ptx");
        reg.register("softmax", ptx, Some(path));

        let hash = PtxRegistry::hash_ptx(ptx);
        assert_eq!(reg.lookup_path(hash), Some(path));
    }

    #[test]
    fn test_ptx_registry_lookup_missing() {
        let reg = PtxRegistry::new();
        assert_eq!(reg.lookup(12345), None);
        assert_eq!(reg.lookup_name(12345), None);
        assert_eq!(reg.lookup_path(12345), None);
    }

    #[test]
    fn test_ptx_registry_hashes() {
        let mut reg = PtxRegistry::new();
        reg.register("k1", "ptx_source_1", None);
        reg.register("k2", "ptx_source_2", None);

        let hashes: Vec<u64> = reg.hashes().collect();
        assert_eq!(hashes.len(), 2);
    }

    #[test]
    fn test_ptx_registry_hash_deterministic() {
        let ptx = "some ptx source code";
        let h1 = PtxRegistry::hash_ptx(ptx);
        let h2 = PtxRegistry::hash_ptx(ptx);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_ptx_registry_hash_different_inputs() {
        let h1 = PtxRegistry::hash_ptx("kernel_a");
        let h2 = PtxRegistry::hash_ptx("kernel_b");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_ptx_registry_overwrite_same_hash() {
        let mut reg = PtxRegistry::new();
        let ptx = "same_source";
        reg.register("name1", ptx, None);
        reg.register("name2", ptx, None);
        // Same PTX hash overwrites
        assert_eq!(reg.len(), 1);
        let hash = PtxRegistry::hash_ptx(ptx);
        assert_eq!(reg.lookup_name(hash), Some("name2"));
    }

    // =========================================================================
    // CategoryStats Tests
    // =========================================================================

    #[test]
    fn test_category_stats_default() {
        let stats = CategoryStats::default();
        assert_eq!(stats.total_ns, 0);
        assert_eq!(stats.total_elements, 0);
        assert_eq!(stats.count, 0);
    }

    #[test]
    fn test_category_stats_avg_us_zero_count() {
        let stats = CategoryStats::default();
        assert_eq!(stats.avg_us(), 0.0);
    }

    #[test]
    fn test_category_stats_avg_us() {
        let stats = CategoryStats { total_ns: 10_000, total_elements: 0, count: 2 };
        // 10_000 ns / 2 / 1000 = 5.0 us
        assert!((stats.avg_us() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_category_stats_throughput_zero_ns() {
        let stats = CategoryStats::default();
        assert_eq!(stats.throughput(), 0.0);
    }

    #[test]
    fn test_category_stats_throughput() {
        let stats = CategoryStats {
            total_ns: 1_000_000_000, // 1 second
            total_elements: 1_000,
            count: 1,
        };
        // 1000 elements / 1s = 1000 elem/s
        assert!((stats.throughput() - 1_000.0).abs() < 1e-5);
    }

    #[test]
    fn test_category_stats_percentage_zero_total() {
        let stats = CategoryStats { total_ns: 500, total_elements: 0, count: 1 };
        assert_eq!(stats.percentage(0), 0.0);
    }

    #[test]
    fn test_category_stats_percentage() {
        let stats = CategoryStats { total_ns: 250, total_elements: 0, count: 1 };
        // 100.0 * 250 / 1000 = 25.0%
        assert!((stats.percentage(1000) - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_category_stats_percentage_full() {
        let stats = CategoryStats { total_ns: 1000, total_elements: 0, count: 1 };
        assert!((stats.percentage(1000) - 100.0).abs() < 1e-10);
    }

    // =========================================================================
    // BrickStats Tests
    // =========================================================================

    #[test]
    fn test_brick_stats_new() {
        let stats = BrickStats::new("test_brick");
        assert_eq!(stats.name, "test_brick");
        assert_eq!(stats.count, 0);
        assert_eq!(stats.total_ns, 0);
        assert_eq!(stats.min_ns, u64::MAX);
        assert_eq!(stats.max_ns, 0);
        assert_eq!(stats.total_elements, 0);
        assert_eq!(stats.total_bytes, 0);
        assert_eq!(stats.total_compressed_bytes, 0);
        assert_eq!(stats.bottleneck, BrickBottleneck::Unknown);
        assert_eq!(stats.total_cycles, 0);
        assert_eq!(stats.min_cycles, u64::MAX);
        assert_eq!(stats.max_cycles, 0);
    }

    #[test]
    fn test_brick_stats_add_sample() {
        let mut stats = BrickStats::new("op");
        stats.add_sample(1000, 50);
        assert_eq!(stats.count, 1);
        assert_eq!(stats.total_ns, 1000);
        assert_eq!(stats.min_ns, 1000);
        assert_eq!(stats.max_ns, 1000);
        assert_eq!(stats.total_elements, 50);

        stats.add_sample(500, 25);
        assert_eq!(stats.count, 2);
        assert_eq!(stats.total_ns, 1500);
        assert_eq!(stats.min_ns, 500);
        assert_eq!(stats.max_ns, 1000);
        assert_eq!(stats.total_elements, 75);

        stats.add_sample(2000, 100);
        assert_eq!(stats.count, 3);
        assert_eq!(stats.min_ns, 500);
        assert_eq!(stats.max_ns, 2000);
    }

    #[test]
    fn test_brick_stats_add_sample_with_cycles() {
        let mut stats = BrickStats::new("op");
        stats.add_sample_with_cycles(1000, 50, 3000);
        assert_eq!(stats.count, 1);
        assert_eq!(stats.total_cycles, 3000);
        assert_eq!(stats.min_cycles, 3000);
        assert_eq!(stats.max_cycles, 3000);

        stats.add_sample_with_cycles(500, 25, 1500);
        assert_eq!(stats.total_cycles, 4500);
        assert_eq!(stats.min_cycles, 1500);
        assert_eq!(stats.max_cycles, 3000);
    }

    #[test]
    fn test_brick_stats_cycles_per_element_zero() {
        let stats = BrickStats::new("op");
        assert_eq!(stats.cycles_per_element(), 0.0);
    }

    #[test]
    fn test_brick_stats_cycles_per_element() {
        let mut stats = BrickStats::new("op");
        stats.add_sample_with_cycles(1000, 100, 500);
        // 500 cycles / 100 elements = 5.0
        assert!((stats.cycles_per_element() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_brick_stats_avg_cycles_zero() {
        let stats = BrickStats::new("op");
        assert_eq!(stats.avg_cycles(), 0.0);
    }

    #[test]
    fn test_brick_stats_avg_cycles() {
        let mut stats = BrickStats::new("op");
        stats.add_sample_with_cycles(1000, 50, 300);
        stats.add_sample_with_cycles(1000, 50, 500);
        // (300 + 500) / 2 = 400.0
        assert!((stats.avg_cycles() - 400.0).abs() < 1e-10);
    }

    #[test]
    fn test_brick_stats_estimated_ipc_zero() {
        let stats = BrickStats::new("op");
        assert_eq!(stats.estimated_ipc(), 0.0);
    }

    #[test]
    fn test_brick_stats_estimated_ipc() {
        let mut stats = BrickStats::new("op");
        stats.add_sample_with_cycles(1000, 200, 100);
        // IPC = elements / cycles = 200 / 100 = 2.0
        assert!((stats.estimated_ipc() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_brick_stats_diagnose_insufficient_data() {
        let stats = BrickStats::new("op");
        assert_eq!(stats.diagnose_from_cycles(), "insufficient data");
    }

    #[test]
    fn test_brick_stats_diagnose_insufficient_data_zero_cycles() {
        let mut stats = BrickStats::new("op");
        stats.add_sample(1000, 50);
        // total_cycles = 0
        assert_eq!(stats.diagnose_from_cycles(), "insufficient data");
    }

    #[test]
    fn test_brick_stats_diagnose_insufficient_data_zero_ns() {
        let mut stats = BrickStats::new("op");
        stats.total_cycles = 100;
        // total_ns = 0
        assert_eq!(stats.diagnose_from_cycles(), "insufficient data");
    }

    #[test]
    fn test_brick_stats_diagnose_memory_bound() {
        let mut stats = BrickStats::new("op");
        // IPC < 0.5 means memory-bound
        // IPC = elements / cycles, so set elements=10, cycles=100 => IPC = 0.1
        stats.total_elements = 10;
        stats.total_cycles = 100;
        stats.total_ns = 50; // ns_per_cycle = 0.5 (doesn't matter, ipc<0.5 catches first)
        assert_eq!(stats.diagnose_from_cycles(), "memory-bound (low IPC, likely cache misses)");
    }

    #[test]
    fn test_brick_stats_diagnose_compute_bound() {
        let mut stats = BrickStats::new("op");
        // IPC > 2.0 means compute-bound
        // elements=300, cycles=100 => IPC = 3.0
        stats.total_elements = 300;
        stats.total_cycles = 100;
        stats.total_ns = 33; // ns_per_cycle = 0.33
        assert_eq!(stats.diagnose_from_cycles(), "compute-bound (efficient)");
    }

    #[test]
    fn test_brick_stats_diagnose_throttled() {
        let mut stats = BrickStats::new("op");
        // IPC between 0.5 and 2.0, ns_per_cycle > 1.0 => throttled
        // elements=100, cycles=100 => IPC=1.0
        // ns=200, cycles=100 => ns_per_cycle=2.0
        stats.total_elements = 100;
        stats.total_cycles = 100;
        stats.total_ns = 200;
        assert_eq!(stats.diagnose_from_cycles(), "throttled or context-switched");
    }

    #[test]
    fn test_brick_stats_diagnose_balanced() {
        let mut stats = BrickStats::new("op");
        // IPC between 0.5 and 2.0, ns_per_cycle <= 1.0 => balanced
        // elements=100, cycles=100 => IPC=1.0
        // ns=50, cycles=100 => ns_per_cycle=0.5
        stats.total_elements = 100;
        stats.total_cycles = 100;
        stats.total_ns = 50;
        assert_eq!(stats.diagnose_from_cycles(), "balanced");
    }

    #[test]
    fn test_brick_stats_add_sample_with_bytes() {
        let mut stats = BrickStats::new("compress");
        stats.add_sample_with_bytes(1000, 1, 4096, 1024);
        assert_eq!(stats.count, 1);
        assert_eq!(stats.total_bytes, 4096);
        assert_eq!(stats.total_compressed_bytes, 1024);
        assert_eq!(stats.total_elements, 1);

        stats.add_sample_with_bytes(2000, 1, 8192, 2048);
        assert_eq!(stats.total_bytes, 12288);
        assert_eq!(stats.total_compressed_bytes, 3072);
    }

    #[test]
    fn test_brick_stats_compression_ratio_no_data() {
        let stats = BrickStats::new("op");
        assert!((stats.compression_ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_brick_stats_compression_ratio() {
        let mut stats = BrickStats::new("compress");
        stats.add_sample_with_bytes(1000, 1, 4096, 1024);
        // 4096 / 1024 = 4.0
        assert!((stats.compression_ratio() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_brick_stats_throughput_gbps_zero_ns() {
        let stats = BrickStats::new("op");
        assert_eq!(stats.throughput_gbps(), 0.0);
    }

    #[test]
    fn test_brick_stats_throughput_gbps() {
        let mut stats = BrickStats::new("op");
        stats.total_bytes = 1_000_000_000; // 1 GB
        stats.total_ns = 1_000_000_000; // 1 second
                                        // 1 GB / 1s = 1.0 GB/s
        assert!((stats.throughput_gbps() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_brick_stats_set_get_bottleneck() {
        let mut stats = BrickStats::new("op");
        assert_eq!(stats.get_bottleneck(), BrickBottleneck::Unknown);

        stats.set_bottleneck(BrickBottleneck::Memory);
        assert_eq!(stats.get_bottleneck(), BrickBottleneck::Memory);

        stats.set_bottleneck(BrickBottleneck::Compute);
        assert_eq!(stats.get_bottleneck(), BrickBottleneck::Compute);
    }

    #[test]
    fn test_brick_stats_avg_us_zero_count() {
        let stats = BrickStats::new("op");
        assert_eq!(stats.avg_us(), 0.0);
    }

    #[test]
    fn test_brick_stats_avg_us() {
        let mut stats = BrickStats::new("op");
        stats.add_sample(2000, 10);
        stats.add_sample(4000, 10);
        // total_ns=6000, count=2 => 6000/2/1000 = 3.0 us
        assert!((stats.avg_us() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_brick_stats_throughput_zero_ns() {
        let stats = BrickStats::new("op");
        assert_eq!(stats.throughput(), 0.0);
    }

    #[test]
    fn test_brick_stats_throughput() {
        let mut stats = BrickStats::new("op");
        stats.add_sample(1_000_000_000, 500); // 1 second, 500 elements
                                              // 500 / 1.0 = 500.0 elem/s
        assert!((stats.throughput() - 500.0).abs() < 1e-5);
    }

    #[test]
    fn test_brick_stats_tokens_per_sec() {
        let mut stats = BrickStats::new("op");
        stats.add_sample(1_000_000_000, 42);
        // tokens_per_sec is alias for throughput
        assert!((stats.tokens_per_sec() - stats.throughput()).abs() < 1e-10);
    }

    #[test]
    fn test_brick_stats_min_us_no_samples() {
        let stats = BrickStats::new("op");
        // min_ns = u64::MAX → returns 0.0
        assert_eq!(stats.min_us(), 0.0);
    }

    #[test]
    fn test_brick_stats_min_us() {
        let mut stats = BrickStats::new("op");
        stats.add_sample(5000, 1);
        stats.add_sample(3000, 1);
        // min_ns = 3000 => 3.0 us
        assert!((stats.min_us() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_brick_stats_max_us() {
        let mut stats = BrickStats::new("op");
        stats.add_sample(5000, 1);
        stats.add_sample(3000, 1);
        // max_ns = 5000 => 5.0 us
        assert!((stats.max_us() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_brick_stats_max_us_no_samples() {
        let stats = BrickStats::new("op");
        // max_ns = 0 => 0.0 us
        assert_eq!(stats.max_us(), 0.0);
    }

    #[test]
    fn test_brick_stats_default() {
        let stats = BrickStats::default();
        assert!(stats.name.is_empty());
        assert_eq!(stats.count, 0);
        assert_eq!(stats.total_ns, 0);
        assert_eq!(stats.bottleneck, BrickBottleneck::Unknown);
    }
}
