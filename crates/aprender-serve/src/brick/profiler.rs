//! BrickProfiler: Real-time telemetry for transformer inference (PMAT-112)
//!
//! Captures actual timing data for every major operation (Brick) during inference:
//! - token_embed
//! - attention_qkv
//! - attention_score
//! - mlp_gate_up
//! - mlp_down
//! - rms_norm
//!
//! # Purpose
//!
//! "If you cannot measure it, you cannot improve it. If you fake the measurement,
//! you are not improving it; you are lying to yourself." - PMAT-112
//!
//! # Usage
//!
//! ```rust,ignore
//! use realizar::brick::BrickProfiler;
//!
//! let mut profiler = BrickProfiler::new();
//!
//! // Profile actual inference operations
//! profiler.start("token_embed");
//! let embeddings = model.embed(&tokens);
//! profiler.stop("token_embed");
//!
//! // Get aggregated stats
//! let report = profiler.report();
//! for (name, stats) in &report.operations {
//!     println!("{}: min={:.2}µs, max={:.2}µs, avg={:.2}µs",
//!              name, stats.min_us, stats.max_us, stats.avg_us);
//! }
//! ```
//!
//! # References
//!
//! - PMAT-112: "End the Observability Theatre"
//! - Williams et al. (2009): Roofline Model
//! - Graham et al. (1982): Call Graph Profiling

use std::collections::HashMap;
use std::time::Instant;

/// Statistics for a single operation type
#[derive(Debug, Clone, Default)]
pub struct OpStats {
    /// Minimum time in microseconds
    pub min_us: f64,
    /// Maximum time in microseconds
    pub max_us: f64,
    /// Average time in microseconds
    pub avg_us: f64,
    /// Total time in microseconds
    pub total_us: f64,
    /// Number of calls
    pub count: usize,
    /// Per-layer breakdown (layer_idx -> time_us)
    pub per_layer: Vec<f64>,
}

impl OpStats {
    /// Create new stats with initial measurement
    fn new(time_us: f64) -> Self {
        Self {
            min_us: time_us,
            max_us: time_us,
            avg_us: time_us,
            total_us: time_us,
            count: 1,
            per_layer: vec![time_us],
        }
    }

    /// Add a new measurement
    fn add(&mut self, time_us: f64) {
        self.min_us = self.min_us.min(time_us);
        self.max_us = self.max_us.max(time_us);
        self.total_us += time_us;
        self.count += 1;
        self.avg_us = self.total_us / self.count as f64;
        self.per_layer.push(time_us);
    }
}

/// Profile report containing all operation statistics
#[derive(Debug, Clone)]
pub struct ProfileReport {
    /// Per-operation statistics
    pub operations: HashMap<String, OpStats>,
    /// Total inference time in microseconds
    pub total_inference_us: f64,
    /// Number of tokens processed
    pub tokens_processed: usize,
    /// Number of layers in the model
    pub num_layers: usize,
    /// Throughput in tokens/second
    pub throughput_tok_s: f64,
    /// Whether the profiler captured real data (vs being disabled)
    pub is_real_data: bool,
}

impl ProfileReport {
    /// Get the hottest operation by total time
    pub fn hottest(&self) -> Option<(&str, &OpStats)> {
        self.operations
            .iter()
            .max_by(|a, b| {
                a.1.total_us
                    .partial_cmp(&b.1.total_us)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(name, stats)| (name.as_str(), stats))
    }

    /// Get operations sorted by total time (descending)
    pub fn sorted_by_time(&self) -> Vec<(&str, &OpStats)> {
        let mut sorted: Vec<_> = self
            .operations
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect();
        sorted.sort_by(|a, b| {
            b.1.total_us
                .partial_cmp(&a.1.total_us)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// Get percentage breakdown
    pub fn percentage_breakdown(&self) -> HashMap<String, f64> {
        let total = self.total_inference_us;
        if total <= 0.0 {
            return HashMap::new();
        }
        self.operations
            .iter()
            .map(|(name, stats)| (name.clone(), (stats.total_us / total) * 100.0))
            .collect()
    }
}

include!("profiler_contracts.rs");

/// Severity levels for contract violations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractSeverity {
    /// Contract violation that indicates a measurement bug
    Error,
    /// Contract violation that indicates a fidelity concern
    Warning,
}

/// Active timing record
#[derive(Debug)]
struct ActiveTimer {
    start: Instant,
}

/// BrickProfiler: Captures real timing data for inference operations
///
/// Thread-safe profiler that records start/stop times for each brick operation.
#[derive(Debug)]
pub struct BrickProfiler {
    /// Per-operation accumulated stats
    stats: HashMap<String, OpStats>,
    /// Currently active timers (operation -> start time)
    active: HashMap<String, ActiveTimer>,
    /// Total inference start time
    inference_start: Option<Instant>,
    /// Total inference end time
    inference_end: Option<Instant>,
    /// Number of tokens being processed
    tokens_count: usize,
    /// Number of layers (for per-layer breakdown)
    num_layers: usize,
    /// Current layer being profiled
    current_layer: usize,
    /// Whether profiling is enabled
    enabled: bool,
}

impl Default for BrickProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl BrickProfiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
            active: HashMap::new(),
            inference_start: None,
            inference_end: None,
            tokens_count: 0,
            num_layers: 0,
            current_layer: 0,
            enabled: true,
        }
    }

    /// Create a disabled profiler (no-op for production)
    pub fn disabled() -> Self {
        let mut p = Self::new();
        p.enabled = false;
        p
    }

    /// Check if profiler is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Set the number of tokens being processed
    pub fn set_tokens(&mut self, count: usize) {
        self.tokens_count = count;
    }

    /// Set the number of layers in the model
    pub fn set_num_layers(&mut self, num_layers: usize) {
        self.num_layers = num_layers;
    }

    /// Set the current layer being profiled
    pub fn set_current_layer(&mut self, layer: usize) {
        self.current_layer = layer;
    }

    /// Start timing an operation
    pub fn start(&mut self, operation: &str) {
        if !self.enabled {
            return;
        }
        self.active.insert(
            operation.to_string(),
            ActiveTimer {
                start: Instant::now(),
            },
        );
    }

    /// Start timing the overall inference
    pub fn start_inference(&mut self) {
        if !self.enabled {
            return;
        }
        self.inference_start = Some(Instant::now());
    }

    /// Stop timing the overall inference
    pub fn stop_inference(&mut self) {
        if !self.enabled {
            return;
        }
        self.inference_end = Some(Instant::now());
    }

    /// Stop timing an operation and record the measurement
    pub fn stop(&mut self, operation: &str) {
        if !self.enabled {
            return;
        }

        if let Some(timer) = self.active.remove(operation) {
            let elapsed_us = timer.start.elapsed().as_secs_f64() * 1_000_000.0;

            if let Some(stats) = self.stats.get_mut(operation) {
                stats.add(elapsed_us);
            } else {
                self.stats
                    .insert(operation.to_string(), OpStats::new(elapsed_us));
            }
        }
    }

    /// Record a measurement directly (for external timing)
    pub fn record(&mut self, operation: &str, time_us: f64) {
        if !self.enabled {
            return;
        }

        if let Some(stats) = self.stats.get_mut(operation) {
            stats.add(time_us);
        } else {
            self.stats
                .insert(operation.to_string(), OpStats::new(time_us));
        }
    }

    /// Convenience method to time a closure
    pub fn measure<F, T>(&mut self, operation: &str, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        if !self.enabled {
            return f();
        }

        self.start(operation);
        let result = f();
        self.stop(operation);
        result
    }

    /// Clear all recorded stats
    pub fn clear(&mut self) {
        self.stats.clear();
        self.active.clear();
        self.inference_start = None;
        self.inference_end = None;
        self.tokens_count = 0;
        self.current_layer = 0;
    }

    /// Generate a profile report with contract validation.
    ///
    /// Automatically checks gpu-decode-profiling-v1 invariants
    /// and prints violations to stderr (realizr#202, P15-06).
    pub fn report(&self) -> ProfileReport {
        let total_inference_us = match (self.inference_start, self.inference_end) {
            (Some(start), Some(end)) => end.duration_since(start).as_secs_f64() * 1_000_000.0,
            _ => self.stats.values().map(|s| s.total_us).sum(),
        };

        let throughput_tok_s = if total_inference_us > 0.0 && self.tokens_count > 0 {
            (self.tokens_count as f64 / total_inference_us) * 1_000_000.0
        } else {
            0.0
        };

        let report = ProfileReport {
            operations: self.stats.clone(),
            total_inference_us,
            tokens_processed: self.tokens_count,
            num_layers: self.num_layers,
            throughput_tok_s,
            is_real_data: self.enabled && !self.stats.is_empty(),
        };

        // P15-06: Contract enforcement — validate on every report generation
        let violations = report.validate_contracts();
        for (severity, msg) in &violations {
            match severity {
                ContractSeverity::Error => eprintln!("[CONTRACT ERROR] {}", msg),
                ContractSeverity::Warning => eprintln!("[CONTRACT WARN] {}", msg),
            }
        }

        report
    }

    /// Get raw stats reference (for inspection)
    pub fn stats(&self) -> &HashMap<String, OpStats> {
        &self.stats
    }
}

// Thread-local profiler for automatic instrumentation
thread_local! {
    /// Thread-local profiler instance
    pub static PROFILER: std::cell::RefCell<BrickProfiler> = std::cell::RefCell::new(BrickProfiler::new());
}

/// Start profiling an operation using the thread-local profiler
#[macro_export]
macro_rules! profile_start {
    ($op:expr) => {
        $crate::brick::profiler::PROFILER.with(|p| {
            p.borrow_mut().start($op);
        });
    };
}

/// Stop profiling an operation using the thread-local profiler
#[macro_export]
macro_rules! profile_stop {
    ($op:expr) => {
        $crate::brick::profiler::PROFILER.with(|p| {
            p.borrow_mut().stop($op);
        });
    };
}

/// Get a report from the thread-local profiler
#[macro_export]
macro_rules! profile_report {
    () => {
        $crate::brick::profiler::PROFILER.with(|p| p.borrow().report())
    };
}

/// Clear the thread-local profiler
#[macro_export]
macro_rules! profile_clear {
    () => {
        $crate::brick::profiler::PROFILER.with(|p| p.borrow_mut().clear());
    };
}

include!("profiler_basic_profiling.rs");
