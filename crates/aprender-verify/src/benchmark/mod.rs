//! Scientific benchmarking infrastructure for certeza
//!
//! This module provides data structures and utilities for reproducible
//! performance measurement following the scientific reporting specification.
//!
//! # Examples
//!
//! See the `metadata` module for metadata collection functions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete benchmark report with metadata and results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// Schema version for backward compatibility
    pub schema_version: String,

    /// Complete environmental metadata
    pub metadata: BenchmarkMetadata,

    /// Individual benchmark results
    pub benchmarks: Vec<BenchmarkResult>,

    /// Summary statistics across all benchmarks
    pub summary: BenchmarkSummary,
}

/// Environmental metadata for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetadata {
    /// Benchmark suite name
    pub benchmark_suite: String,

    /// ISO 8601 timestamp
    pub timestamp: String,

    /// Git commit hash (short)
    pub git_commit: String,

    /// Git branch name
    pub git_branch: String,

    /// Operator (human or automated-ci)
    pub operator: String,

    /// Hardware specifications
    pub hardware: HardwareInfo,

    /// Software environment
    pub software: SoftwareInfo,

    /// Runtime configuration
    pub environment: EnvironmentConfig,
}

/// Hardware specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    /// CPU information
    pub cpu: CpuInfo,

    /// Memory information
    pub memory: MemoryInfo,

    /// Storage information
    pub storage: Option<StorageInfo>,
}

/// CPU details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    /// CPU model name
    pub model: String,

    /// Number of physical cores
    pub cores: usize,

    /// Number of logical threads
    pub threads: usize,

    /// Base frequency in `MHz`
    pub frequency_mhz: u64,

    /// L1 cache size
    pub cache_l1: String,

    /// L2 cache size
    pub cache_l2: String,

    /// L3 cache size
    pub cache_l3: String,
}

/// Memory details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    /// Total memory in GB
    pub total_gb: f64,

    /// Memory type (DDR4, DDR5, etc.)
    #[serde(rename = "type")]
    pub memory_type: String,

    /// Memory frequency in `MHz`
    pub frequency_mhz: Option<u64>,
}

/// Storage details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageInfo {
    /// Storage type (SSD, `NVMe`, HDD)
    #[serde(rename = "type")]
    pub storage_type: String,

    /// Device model
    pub model: String,

    /// Capacity in GB
    pub capacity_gb: u64,
}

/// Software environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftwareInfo {
    /// Operating system
    pub os: String,

    /// Kernel version
    pub kernel: String,

    /// Rust compiler version
    pub rustc: String,

    /// Cargo version
    pub cargo: String,

    /// LLVM version
    pub llvm: String,
}

/// Runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentConfig {
    /// CPU frequency governor
    pub cpu_governor: String,

    /// Turbo boost enabled/disabled
    pub turbo_boost: String,

    /// Swap enabled/disabled
    pub swap: String,

    /// CPU isolation configuration
    pub isolation: Option<String>,
}

/// Individual benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Benchmark name
    pub name: String,

    /// Category (cpu-bound, memory-intensive, io-bound)
    pub category: String,

    /// Scope (component, system)
    pub scope: String,

    /// Binary path
    pub binary: String,

    /// Optimization profile used
    pub optimization_profile: String,

    /// Measurement data
    pub measurements: Measurements,

    /// Comparison against baseline (if available)
    pub comparison: Option<Comparison>,
}

/// Measurement data and statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measurements {
    /// Number of warmup runs
    pub warmup_runs: usize,

    /// Number of measured runs
    pub measured_runs: usize,

    /// Raw measurement values in milliseconds
    pub raw_values_ms: Vec<f64>,

    /// Outliers removed (if any)
    pub outliers_removed: Vec<f64>,

    /// Statistical summary
    pub statistics: Statistics,

    /// Distribution analysis
    pub distribution: Distribution,
}

/// Statistical summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Statistics {
    /// Arithmetic mean (ms)
    pub mean_ms: f64,

    /// Median (ms)
    pub median_ms: f64,

    /// Standard deviation (ms)
    pub std_dev_ms: f64,

    /// Minimum value (ms)
    pub min_ms: f64,

    /// Maximum value (ms)
    pub max_ms: f64,

    /// Coefficient of variation (`std_dev` / mean)
    pub coefficient_of_variation: f64,

    /// 95% confidence interval [lower, upper]
    pub confidence_interval_95: (f64, f64),
}

/// Distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Distribution {
    /// Normality test used (shapiro, ks)
    pub normality_test: String,

    /// P-value from normality test
    pub normality_p_value: f64,

    /// Whether distribution is normal (p > 0.05)
    pub is_normal: bool,
}

/// Comparison against baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comparison {
    /// Baseline commit hash
    pub baseline_commit: String,

    /// Baseline mean runtime (ms)
    pub baseline_mean_ms: f64,

    /// Speedup ratio (baseline / current)
    pub speedup_ratio: f64,

    /// 95% confidence interval on speedup [lower, upper]
    pub speedup_ci_95: (f64, f64),

    /// T-test p-value
    pub t_test_p_value: f64,

    /// Effect size (Cohen's d)
    pub effect_size_cohens_d: f64,

    /// Whether improvement is statistically significant
    pub significant_improvement: bool,

    /// Whether regression is statistically significant
    pub significant_regression: bool,
}

/// Summary across all benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    /// Total number of benchmarks
    pub total_benchmarks: usize,

    /// Successfully completed benchmarks
    pub successful: usize,

    /// Failed benchmarks
    pub failed: usize,

    /// Total runtime in seconds
    pub total_runtime_seconds: f64,

    /// Number of significant improvements
    pub significant_improvements: usize,

    /// Number of significant regressions
    pub significant_regressions: usize,

    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

impl BenchmarkReport {
    /// Create a new benchmark report with default values
    #[must_use]
    pub fn new(suite_name: &str, metadata: BenchmarkMetadata) -> Self {
        // GH-18: Use suite_name to override metadata.benchmark_suite if it differs
        let mut metadata = metadata;
        if metadata.benchmark_suite.is_empty() && !suite_name.is_empty() {
            metadata.benchmark_suite = suite_name.to_string();
        }
        Self {
            schema_version: "1.0".to_string(),
            metadata,
            benchmarks: Vec::new(),
            summary: BenchmarkSummary {
                total_benchmarks: 0,
                successful: 0,
                failed: 0,
                total_runtime_seconds: 0.0,
                significant_improvements: 0,
                significant_regressions: 0,
                metrics: HashMap::new(),
            },
        }
    }

    /// Add a benchmark result
    pub fn add_benchmark(&mut self, result: BenchmarkResult) {
        self.benchmarks.push(result);
        self.update_summary();
    }

    /// Update summary statistics
    fn update_summary(&mut self) {
        self.summary.total_benchmarks = self.benchmarks.len();

        // Count significant changes and failed benchmarks
        let mut improvements = 0;
        let mut regressions = 0;

        for bench in &self.benchmarks {
            if let Some(comp) = &bench.comparison {
                if comp.significant_improvement {
                    improvements += 1;
                }
                if comp.significant_regression {
                    regressions += 1;
                }
            }
        }

        // GH-18: Compute failed count (benchmarks with significant regressions)
        self.summary.failed = regressions;
        self.summary.successful = self.benchmarks.len().saturating_sub(regressions);

        // GH-18: Compute total runtime from individual benchmark mean times
        self.summary.total_runtime_seconds =
            self.benchmarks.iter().map(|b| b.measurements.statistics.mean_ms / 1000.0).sum();

        self.summary.significant_improvements = improvements;
        self.summary.significant_regressions = regressions;
    }

    /// Serialize to JSON string
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Serialize to JSON file
    ///
    /// # Errors
    ///
    /// Returns an error if file creation or writing fails
    pub fn to_json_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = self.to_json().map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    /// Deserialize from JSON string
    ///
    /// # Errors
    ///
    /// Returns an error if deserialization fails or JSON is invalid
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Deserialize from JSON file
    ///
    /// # Errors
    ///
    /// Returns an error if file reading fails or JSON is invalid
    pub fn from_json_file(path: &std::path::Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_report_serialization() {
        let metadata = BenchmarkMetadata {
            benchmark_suite: "test-suite".to_string(),
            timestamp: "2025-11-18T10:30:00Z".to_string(),
            git_commit: "abc123".to_string(),
            git_branch: "main".to_string(),
            operator: "test".to_string(),
            hardware: HardwareInfo {
                cpu: CpuInfo {
                    model: "Test CPU".to_string(),
                    cores: 4,
                    threads: 8,
                    frequency_mhz: 3000,
                    cache_l1: "32KB".to_string(),
                    cache_l2: "256KB".to_string(),
                    cache_l3: "8MB".to_string(),
                },
                memory: MemoryInfo {
                    total_gb: 16.0,
                    memory_type: "DDR4".to_string(),
                    frequency_mhz: Some(2400),
                },
                storage: None,
            },
            software: SoftwareInfo {
                os: "Linux".to_string(),
                kernel: "5.15.0".to_string(),
                rustc: "1.75.0".to_string(),
                cargo: "1.75.0".to_string(),
                llvm: "17.0".to_string(),
            },
            environment: EnvironmentConfig {
                cpu_governor: "performance".to_string(),
                turbo_boost: "disabled".to_string(),
                swap: "disabled".to_string(),
                isolation: None,
            },
        };

        let report = BenchmarkReport::new("test-suite", metadata);

        // Test JSON serialization
        let json = report.to_json().expect("Failed to serialize");
        assert!(json.contains("\"schema_version\": \"1.0\""));

        // Test round-trip
        let deserialized = BenchmarkReport::from_json(&json).expect("Failed to deserialize");
        assert_eq!(deserialized.schema_version, "1.0");
        assert_eq!(deserialized.metadata.benchmark_suite, "test-suite");
    }
}
