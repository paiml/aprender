//! CORRECTNESS-011: BrickProfiler for CPU/GPU Divergence Detection

use super::types::{DivergenceReport, KernelTrace};

/// BrickProfiler collects per-kernel traces for automated divergence detection.
///
/// Five-Whys Root Cause: Hours of manual "let me check X in Y" debugging
/// → No automated tool identified which kernel diverged
/// → BrickProfiler only captured timing, not checksums
/// → Missing feature: per-kernel checksum capture
/// → ROOT CAUSE: Brick Profiling lacked correctness instrumentation
///
/// # Usage
///
/// ```rust,ignore
/// use cbtop::{BrickProfiler, KernelTrace};
///
/// // CPU execution
/// let mut cpu_profiler = BrickProfiler::new("cpu_baseline");
/// cpu_profiler.add_trace(KernelTrace::new("rope_neox", 0, pos, "CPU")
///     .with_input_checksum(&input)
///     .with_output_checksum(&output));
///
/// // GPU execution
/// let mut gpu_profiler = BrickProfiler::new("cuda_test");
/// gpu_profiler.add_trace(KernelTrace::new("rope_neox", 0, pos, "CUDA")
///     .with_input_checksum(&input)
///     .with_output_checksum(&output));
///
/// // Automated divergence detection
/// let report = cpu_profiler.compare(&gpu_profiler);
/// if !report.matched {
///     eprintln!("FIVE-WHYS ALERT: {}", report.diagnosis);
/// }
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BrickProfiler {
    /// Run identifier (e.g., "cpu_baseline", "cuda_test")
    pub run_id: String,
    /// Collected kernel traces
    pub traces: Vec<KernelTrace>,
    /// Total execution time in microseconds
    pub total_time_us: f64,
    /// Whether any divergence was detected
    pub diverged: bool,
    /// Divergence diagnosis (if any)
    pub divergence_diagnosis: String,
}

impl BrickProfiler {
    /// Create a new profiler for a run
    pub fn new(run_id: &str) -> Self {
        Self {
            run_id: run_id.to_string(),
            traces: Vec::new(),
            total_time_us: 0.0,
            diverged: false,
            divergence_diagnosis: String::new(),
        }
    }

    /// Add a kernel trace
    pub fn add_trace(&mut self, trace: KernelTrace) {
        self.total_time_us += trace.time_us;
        self.traces.push(trace);
    }

    /// Check if divergence was detected
    pub fn is_diverged(&self) -> bool {
        self.diverged
    }

    /// Compare this profiler's traces against a reference (e.g., CPU vs GPU)
    ///
    /// Returns a DivergenceReport identifying the first divergent kernel.
    /// Matching is done by (kernel_name, layer_idx, position) triple.
    pub fn compare(&self, reference: &BrickProfiler) -> DivergenceReport {
        // Build index from reference traces
        let ref_index: std::collections::HashMap<(&str, usize, u32), &KernelTrace> = reference
            .traces
            .iter()
            .map(|t| ((t.kernel_name.as_str(), t.layer_idx, t.position), t))
            .collect();

        let mut kernels_compared = 0;

        for actual_trace in &self.traces {
            let key = (
                actual_trace.kernel_name.as_str(),
                actual_trace.layer_idx,
                actual_trace.position,
            );

            if let Some(expected_trace) = ref_index.get(&key) {
                kernels_compared += 1;

                // Compare output checksums
                if actual_trace.output_checksum != expected_trace.output_checksum {
                    return DivergenceReport::diverged(
                        (*expected_trace).clone(),
                        actual_trace.clone(),
                        kernels_compared,
                    );
                }
            }
        }

        DivergenceReport::matched(kernels_compared)
    }

    /// Compare and set internal divergence state
    pub fn compare_and_mark(&mut self, reference: &BrickProfiler) -> DivergenceReport {
        let report = self.compare(reference);
        self.diverged = !report.matched;
        self.divergence_diagnosis = report.diagnosis.clone();
        report
    }

    /// Get traces for a specific kernel name
    pub fn traces_for_kernel(&self, kernel_name: &str) -> Vec<&KernelTrace> {
        self.traces
            .iter()
            .filter(|t| t.kernel_name == kernel_name)
            .collect()
    }

    /// Get traces for a specific layer
    pub fn traces_for_layer(&self, layer_idx: usize) -> Vec<&KernelTrace> {
        self.traces
            .iter()
            .filter(|t| t.layer_idx == layer_idx)
            .collect()
    }

    /// Clear all traces (for reuse)
    pub fn clear(&mut self) {
        self.traces.clear();
        self.total_time_us = 0.0;
        self.diverged = false;
        self.divergence_diagnosis.clear();
    }

    /// Serialize to JSON for pmat brick-score consumption
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}
