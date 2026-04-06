//! Backend benchmark matrix for comparing compute implementations
//!
//! Extracted from bench/mod.rs (PMAT-802) to reduce module size.
//! Contains:
//! - ComputeBackendType enum (CPU, wgpu, CUDA)
//! - BenchmarkMatrix for cross-backend comparison
//! - MatrixBenchmarkConfig and related types

#![allow(clippy::cast_precision_loss)]

use std::fmt::Write;

use serde::{Deserialize, Serialize};

use super::{chrono_timestamp, compute_cv, percentile, HardwareSpec, RuntimeType};

// ============================================================================
// Backend Benchmark Matrix (per Hoefler & Belli SC'15)
// ============================================================================

/// Compute backend type for benchmark matrix
///
/// Represents the different compute backends that can be benchmarked:
/// - CPU: Scalar/SIMD operations via trueno CPU backend
/// - Wgpu: Cross-platform GPU via trueno wgpu backend
/// - Cuda: NVIDIA GPU via trueno-gpu PTX execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComputeBackendType {
    /// CPU backend (scalar/SIMD via trueno)
    Cpu,
    /// wgpu GPU backend (cross-platform via trueno)
    Wgpu,
    /// CUDA GPU backend (NVIDIA via trueno-gpu)
    Cuda,
}

impl std::fmt::Display for ComputeBackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Wgpu => write!(f, "wgpu"),
            Self::Cuda => write!(f, "cuda"),
        }
    }
}

impl ComputeBackendType {
    /// Parse from string
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cpu" => Some(Self::Cpu),
            "wgpu" | "gpu" => Some(Self::Wgpu),
            "cuda" | "nvidia" => Some(Self::Cuda),
            _ => None,
        }
    }

    /// All available backend types
    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![Self::Cpu, Self::Wgpu, Self::Cuda]
    }
}

/// Single entry in the benchmark matrix
///
/// Represents results for one (runtime, backend) combination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixBenchmarkEntry {
    /// Runtime type (realizar, llama-cpp, ollama, vllm)
    pub runtime: RuntimeType,
    /// Compute backend (cpu, wgpu, cuda)
    pub backend: ComputeBackendType,
    /// Model name/identifier
    pub model: String,
    /// Whether this configuration is available
    pub available: bool,
    /// p50 latency in milliseconds
    pub p50_latency_ms: f64,
    /// p99 latency in milliseconds
    pub p99_latency_ms: f64,
    /// Throughput in tokens per second
    pub throughput_tps: f64,
    /// Cold start time in milliseconds
    pub cold_start_ms: f64,
    /// Number of samples collected
    pub samples: usize,
    /// Final CV at stop
    pub cv_at_stop: f64,
    /// Additional notes (e.g., "GPU layers: 99")
    pub notes: String,
}

impl Default for MatrixBenchmarkEntry {
    fn default() -> Self {
        Self {
            runtime: RuntimeType::Realizar,
            backend: ComputeBackendType::Cpu,
            model: String::new(),
            available: false,
            p50_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            throughput_tps: 0.0,
            cold_start_ms: 0.0,
            samples: 0,
            cv_at_stop: 0.0,
            notes: String::new(),
        }
    }
}

impl MatrixBenchmarkEntry {
    /// Create a new unavailable entry (placeholder)
    #[must_use]
    pub fn unavailable(runtime: RuntimeType, backend: ComputeBackendType) -> Self {
        Self {
            runtime,
            backend,
            available: false,
            notes: "Backend not available".to_string(),
            ..Default::default()
        }
    }

    /// Create entry from raw latency samples
    #[must_use]
    pub fn from_samples(
        runtime: RuntimeType,
        backend: ComputeBackendType,
        model: &str,
        latencies_ms: &[f64],
        throughputs_tps: &[f64],
        cold_start_ms: f64,
    ) -> Self {
        let samples = latencies_ms.len();
        if samples == 0 {
            return Self::unavailable(runtime, backend);
        }

        let p50_latency = percentile(latencies_ms, 50.0);
        let p99_latency = percentile(latencies_ms, 99.0);
        let throughput = if throughputs_tps.is_empty() {
            0.0
        } else {
            throughputs_tps.iter().sum::<f64>() / throughputs_tps.len() as f64
        };
        let cv = compute_cv(latencies_ms);

        Self {
            runtime,
            backend,
            model: model.to_string(),
            available: true,
            p50_latency_ms: p50_latency,
            p99_latency_ms: p99_latency,
            throughput_tps: throughput,
            cold_start_ms,
            samples,
            cv_at_stop: cv,
            notes: String::new(),
        }
    }

    /// Add notes to the entry
    #[must_use]
    pub fn with_notes(mut self, notes: &str) -> Self {
        self.notes = notes.to_string();
        self
    }
}

/// Complete benchmark matrix comparing runtimes across backends
///
/// Per Hoefler & Belli SC'15, this matrix enables:
/// - Reproducible comparisons across configurations
/// - Statistical validity via CV-based stopping
/// - Clear identification of performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMatrix {
    /// Schema version
    pub version: String,
    /// ISO 8601 timestamp
    pub timestamp: String,
    /// Model used for benchmarking
    pub model: String,
    /// Hardware specification
    pub hardware: HardwareSpec,
    /// Benchmark methodology
    pub methodology: String,
    /// CV threshold used
    pub cv_threshold: f64,
    /// Matrix entries indexed by (runtime, backend)
    pub entries: Vec<MatrixBenchmarkEntry>,
}

impl BenchmarkMatrix {
    /// Create a new empty matrix
    #[must_use]
    pub fn new(model: &str, hardware: HardwareSpec) -> Self {
        Self {
            version: "1.1".to_string(),
            timestamp: chrono_timestamp(),
            model: model.to_string(),
            hardware,
            methodology: "CV-based stopping (Hoefler & Belli SC'15)".to_string(),
            cv_threshold: 0.05,
            entries: Vec::new(),
        }
    }

    /// Add an entry to the matrix
    pub fn add_entry(&mut self, entry: MatrixBenchmarkEntry) {
        // Remove existing entry for same (runtime, backend) if present
        self.entries
            .retain(|e| e.runtime != entry.runtime || e.backend != entry.backend);
        self.entries.push(entry);
    }

    /// Get entry for specific (runtime, backend) combination
    #[must_use]
    pub fn get_entry(
        &self,
        runtime: RuntimeType,
        backend: ComputeBackendType,
    ) -> Option<&MatrixBenchmarkEntry> {
        self.entries
            .iter()
            .find(|e| e.runtime == runtime && e.backend == backend)
    }

    /// Get all entries for a specific runtime
    #[must_use]
    pub fn entries_for_runtime(&self, runtime: RuntimeType) -> Vec<&MatrixBenchmarkEntry> {
        self.entries
            .iter()
            .filter(|e| e.runtime == runtime)
            .collect()
    }

    /// Get all entries for a specific backend
    #[must_use]
    pub fn entries_for_backend(&self, backend: ComputeBackendType) -> Vec<&MatrixBenchmarkEntry> {
        self.entries
            .iter()
            .filter(|e| e.backend == backend)
            .collect()
    }

    /// Find the fastest runtime for a given backend (by p50 latency)
    #[must_use]
    pub fn fastest_for_backend(
        &self,
        backend: ComputeBackendType,
    ) -> Option<&MatrixBenchmarkEntry> {
        self.entries_for_backend(backend)
            .into_iter()
            .filter(|e| e.available)
            .min_by(|a, b| {
                a.p50_latency_ms
                    .partial_cmp(&b.p50_latency_ms)
                    .expect("test")
            })
    }

    /// Find the highest throughput runtime for a given backend
    #[must_use]
    pub fn highest_throughput_for_backend(
        &self,
        backend: ComputeBackendType,
    ) -> Option<&MatrixBenchmarkEntry> {
        self.entries_for_backend(backend)
            .into_iter()
            .filter(|e| e.available)
            .max_by(|a, b| {
                a.throughput_tps
                    .partial_cmp(&b.throughput_tps)
                    .expect("test")
            })
    }

    /// Generate markdown table for README
    #[must_use]
    pub fn to_markdown_table(&self) -> String {
        let mut table = String::new();

        // Header
        table.push_str("| Runtime | Backend | p50 Latency | p99 Latency | Throughput | Cold Start | Samples | CV |\n");
        table.push_str("|---------|---------|-------------|-------------|------------|------------|---------|----|\n");

        // Sort entries by runtime, then backend
        let mut sorted_entries = self.entries.clone();
        sorted_entries.sort_by(|a, b| {
            let runtime_cmp = format!("{:?}", a.runtime).cmp(&format!("{:?}", b.runtime));
            if runtime_cmp == std::cmp::Ordering::Equal {
                format!("{}", a.backend).cmp(&format!("{}", b.backend))
            } else {
                runtime_cmp
            }
        });

        for entry in &sorted_entries {
            if entry.available {
                let _ = writeln!(
                    table,
                    "| **{}** | {} | {:.1}ms | {:.1}ms | {:.1} tok/s | {:.0}ms | {} | {:.3} |",
                    format!("{:?}", entry.runtime).to_lowercase(),
                    entry.backend,
                    entry.p50_latency_ms,
                    entry.p99_latency_ms,
                    entry.throughput_tps,
                    entry.cold_start_ms,
                    entry.samples,
                    entry.cv_at_stop,
                );
            } else {
                let _ = writeln!(
                    table,
                    "| {} | {} | - | - | - | - | - | - |",
                    format!("{:?}", entry.runtime).to_lowercase(),
                    entry.backend,
                );
            }
        }

        table
    }

    /// Serialize to JSON
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON
    ///
    /// # Errors
    ///
    /// Returns error if JSON is invalid.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Matrix benchmark runner configuration
#[derive(Debug, Clone)]
pub struct MatrixBenchmarkConfig {
    /// Runtimes to benchmark
    pub runtimes: Vec<RuntimeType>,
    /// Backends to benchmark
    pub backends: Vec<ComputeBackendType>,
    /// Model path
    pub model_path: String,
    /// Prompt for benchmarking
    pub prompt: String,
    /// Max tokens to generate
    pub max_tokens: usize,
    /// CV threshold for stopping
    pub cv_threshold: f64,
    /// Minimum samples
    pub min_samples: usize,
    /// Maximum samples (failsafe)
    pub max_samples: usize,
    /// Warmup iterations
    pub warmup_iterations: usize,
}

impl Default for MatrixBenchmarkConfig {
    fn default() -> Self {
        Self {
            runtimes: vec![
                RuntimeType::Realizar,
                RuntimeType::LlamaCpp,
                RuntimeType::Ollama,
            ],
            backends: vec![ComputeBackendType::Cpu, ComputeBackendType::Wgpu],
            model_path: String::new(),
            prompt: "Explain machine learning in one sentence.".to_string(),
            max_tokens: 50,
            cv_threshold: 0.05,
            min_samples: 30,
            max_samples: 200,
            warmup_iterations: 5,
        }
    }
}

/// Summary statistics for a single matrix column (backend)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendSummary {
    /// Backend type
    pub backend: ComputeBackendType,
    /// Number of available runtimes
    pub available_runtimes: usize,
    /// Fastest runtime (by p50 latency)
    pub fastest_runtime: Option<String>,
    /// Fastest p50 latency
    pub fastest_p50_ms: f64,
    /// Highest throughput runtime
    pub highest_throughput_runtime: Option<String>,
    /// Highest throughput (tok/s)
    pub highest_throughput_tps: f64,
}

/// Summary of the entire benchmark matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixSummary {
    /// Total entries in matrix
    pub total_entries: usize,
    /// Number of available entries
    pub available_entries: usize,
    /// Per-backend summaries
    pub backend_summaries: Vec<BackendSummary>,
    /// Overall fastest (runtime, backend) combination
    pub overall_fastest: Option<(String, String)>,
    /// Overall highest throughput (runtime, backend)
    pub overall_highest_throughput: Option<(String, String)>,
}

include!("matrix_summary.rs");
include!("matrix_benchmark.rs");
