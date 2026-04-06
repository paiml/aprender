//! Optimization suite for baseline collection (OPT-001).

use crate::config::{ComputeBackend, WorkloadType};
use crate::error::CbtopError;
use crate::headless::Benchmark;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

use super::cpu_detect::CpuCapabilities;

/// Configuration for a specific workload in the benchmark suite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadConfig {
    /// Workload type
    pub workload: WorkloadType,
    /// Human-readable name
    pub name: String,
    /// Legacy: static theoretical peak GFLOP/s (deprecated, use bytes_per_flop instead)
    pub theoretical_peak_gflops: f64,
    /// Whether this workload is memory-bound
    pub memory_bound: bool,
    /// Bytes transferred per FLOP (for memory-bound analysis)
    /// - dot_product: read 2 floats (8 bytes) per 2 FLOPs (mul+add) = 4 bytes/FLOP
    /// - elementwise: read 2, write 1 float (12 bytes) per 1 FLOP = 12 bytes/FLOP
    /// - reduction: read 1 float (4 bytes) per 1 FLOP = 4 bytes/FLOP
    #[serde(default = "default_bytes_per_flop")]
    pub bytes_per_flop: f64,
}

fn default_bytes_per_flop() -> f64 {
    8.0 // Conservative default
}

impl WorkloadConfig {
    /// Calculate size-aware theoretical peak using detected CPU capabilities
    pub fn theoretical_peak_for_size(&self, size: usize, cpu: &CpuCapabilities) -> f64 {
        // Each element is 4 bytes (f32)
        let bytes_per_element = 4;
        cpu.theoretical_peak_for_size(size, bytes_per_element, self.bytes_per_flop)
    }
}

/// Entry in the baseline report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineEntry {
    /// Workload name
    pub workload: String,
    /// Problem size
    pub size: usize,
    /// Backend used
    pub backend: String,
    /// Achieved GFLOP/s
    pub gflops: f64,
    /// Efficiency (achieved / theoretical)
    pub efficiency: f64,
    /// Coefficient of variation (%)
    pub cv_percent: f64,
    /// Quality score (0-100)
    pub score: u8,
}

/// Complete baseline report with all measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineReport {
    /// Version of cbtop that generated this report
    pub version: String,
    /// Timestamp when collected
    pub timestamp: String,
    /// All baseline entries
    pub entries: Vec<BaselineEntry>,
    /// System information
    pub system: String,
}

impl BaselineReport {
    /// Save baseline to JSON file
    pub fn save(&self, path: &std::path::Path) -> Result<(), CbtopError> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| CbtopError::Config(format!("JSON serialization failed: {}", e)))?;
        std::fs::write(path, json)
            .map_err(|e| CbtopError::Config(format!("Failed to write file: {}", e)))?;
        Ok(())
    }

    /// Load baseline from JSON file
    pub fn load(path: &std::path::Path) -> Result<Self, CbtopError> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| CbtopError::Config(format!("Failed to read file: {}", e)))?;
        serde_json::from_str(&json)
            .map_err(|e| CbtopError::Config(format!("JSON parsing failed: {}", e)))
    }
}

/// Comprehensive benchmark suite for optimization identification
pub struct OptimizationSuite {
    /// Workloads to benchmark
    pub workloads: Vec<WorkloadConfig>,
    /// Backends to test
    pub backends: Vec<ComputeBackend>,
    /// Problem sizes to test
    pub sizes: Vec<usize>,
    /// Duration per benchmark
    pub duration: Duration,
    /// Output file for baseline
    pub baseline_file: PathBuf,
}

impl Default for OptimizationSuite {
    fn default() -> Self {
        Self::standard()
    }
}

impl OptimizationSuite {
    /// Create standard optimization suite with recommended configurations
    pub fn standard() -> Self {
        Self {
            workloads: vec![
                WorkloadConfig {
                    workload: WorkloadType::Gemm,
                    name: "dot_product".to_string(),
                    theoretical_peak_gflops: 100.0, // Legacy, use bytes_per_flop
                    memory_bound: false,
                    // dot_product: read 2 floats per 2 FLOPs = 4 bytes/FLOP
                    bytes_per_flop: 4.0,
                },
                WorkloadConfig {
                    workload: WorkloadType::Elementwise,
                    name: "elementwise_mul".to_string(),
                    theoretical_peak_gflops: 50.0, // Legacy
                    memory_bound: true,
                    // elementwise: read 2, write 1 float per 1 FLOP = 12 bytes/FLOP
                    bytes_per_flop: 12.0,
                },
                WorkloadConfig {
                    workload: WorkloadType::Reduction,
                    name: "sum_reduction".to_string(),
                    theoretical_peak_gflops: 50.0, // Legacy
                    memory_bound: true,
                    // reduction: read 1 float per 1 FLOP = 4 bytes/FLOP
                    bytes_per_flop: 4.0,
                },
                WorkloadConfig {
                    workload: WorkloadType::Bandwidth,
                    name: "memory_bandwidth".to_string(),
                    theoretical_peak_gflops: 30.0, // Legacy
                    memory_bound: true,
                    // bandwidth: read + write = 8 bytes per "FLOP" (copy)
                    bytes_per_flop: 8.0,
                },
            ],
            backends: vec![ComputeBackend::Simd],
            sizes: vec![
                1_000,      // L1 cache (~4 KB for 1000 f32)
                10_000,     // L2 cache (~40 KB)
                100_000,    // L3 cache (~400 KB)
                1_000_000,  // Main memory (~4 MB)
                4_000_000,  // Large (tiling threshold, ~16 MB)
                16_000_000, // Very large (~64 MB)
            ],
            duration: Duration::from_secs(3),
            baseline_file: PathBuf::from("benchmarks/baseline.json"),
        }
    }

    /// Create a quick suite for CI (fewer configurations, shorter duration)
    pub fn quick() -> Self {
        Self {
            workloads: vec![
                WorkloadConfig {
                    workload: WorkloadType::Gemm,
                    name: "dot_product".to_string(),
                    theoretical_peak_gflops: 100.0,
                    memory_bound: false,
                    bytes_per_flop: 4.0,
                },
                WorkloadConfig {
                    workload: WorkloadType::Elementwise,
                    name: "elementwise_mul".to_string(),
                    theoretical_peak_gflops: 50.0,
                    memory_bound: true,
                    bytes_per_flop: 12.0,
                },
            ],
            backends: vec![ComputeBackend::Simd],
            sizes: vec![10_000, 1_000_000],
            duration: Duration::from_secs(1),
            baseline_file: PathBuf::from("benchmarks/baseline-quick.json"),
        }
    }

    /// Collect baseline measurements for all configurations
    pub fn collect_baseline(&self) -> Result<BaselineReport, CbtopError> {
        let mut entries = Vec::new();
        let cpu = CpuCapabilities::detect();

        let mut prev_working_set_mb: usize = 0;

        for workload in &self.workloads {
            for &size in &self.sizes {
                for &backend in &self.backends {
                    // OPT-011: Adaptive cooldown based on working set size
                    // Scale cooldown: 100ms base + 10ms per MB of previous working set (max 500ms)
                    // This allows memory subsystem to stabilize for large workloads
                    if !entries.is_empty() {
                        let cooldown_ms = 100 + (prev_working_set_mb * 10).min(400);
                        std::thread::sleep(Duration::from_millis(cooldown_ms as u64));

                        // OPT-012: Memory barrier to ensure previous benchmark's
                        // writes are visible and memory allocator state is stable
                        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
                    }

                    // Calculate working set for this benchmark (used for next cooldown)
                    // Working set = size * bytes_per_flop (accounts for all arrays)
                    prev_working_set_mb =
                        ((size as f64 * workload.bytes_per_flop) / (1024.0 * 1024.0)) as usize;

                    let result = Benchmark::builder()
                        .workload_type(workload.workload)
                        .size(size)
                        .backend(backend)
                        .duration(self.duration)
                        .build()?
                        .run()?;

                    // Use size-aware theoretical peak
                    let theoretical_peak = workload.theoretical_peak_for_size(size, &cpu);
                    let efficiency = if theoretical_peak > 0.0 {
                        // Cap efficiency at 1.0 (100%) - values > 100% indicate
                        // measurement noise or overly conservative theoretical peak
                        (result.results.gflops / theoretical_peak).min(1.0)
                    } else {
                        0.0
                    };

                    entries.push(BaselineEntry {
                        workload: workload.name.clone(),
                        size,
                        backend: format!("{:?}", backend),
                        gflops: result.results.gflops,
                        efficiency,
                        cv_percent: result.results.latency_ms.cv_percent,
                        score: result.score.total,
                    });
                }
            }
        }

        let timestamp = chrono::Utc::now().to_rfc3339();

        Ok(BaselineReport {
            version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp,
            entries,
            system: Self::get_system_info(&cpu),
        })
    }

    pub(crate) fn get_system_info(cpu: &CpuCapabilities) -> String {
        format!(
            "{} cores @ {} MHz, AVX2={}, AVX512={}, L3={}MB, mem_bw={:.0} GB/s",
            cpu.cores,
            cpu.max_freq_mhz,
            cpu.has_avx2,
            cpu.has_avx512,
            cpu.l3_cache / (1024 * 1024),
            cpu.mem_bandwidth_gbs
        )
    }
}
