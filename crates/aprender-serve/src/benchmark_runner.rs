
// ============================================================================
// Benchmark Runner
// ============================================================================

/// Benchmark runner with profiling
#[derive(Debug)]
pub struct BenchmarkRunner {
    /// Results grid
    pub grid: BenchmarkGrid,
    /// Profiling start time
    start_time: Option<Instant>,
    /// Component timings
    component_times: Vec<(String, Duration, u64)>,
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchmarkRunner {
    /// Create new benchmark runner
    pub fn new() -> Self {
        Self {
            grid: BenchmarkGrid::new(),
            start_time: None,
            component_times: Vec::new(),
        }
    }

    /// Start profiling
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Record component timing
    pub fn record_component(&mut self, name: &str, duration: Duration, calls: u64) {
        self.component_times
            .push((name.to_string(), duration, calls));
    }

    /// Measure a component
    pub fn measure<F, R>(&mut self, name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        self.record_component(name, start.elapsed(), 1);
        result
    }

    /// Finalize and compute hotspots
    pub fn finalize(&mut self) {
        let total_time: Duration = self.component_times.iter().map(|(_, d, _)| *d).sum();
        let total_nanos = total_time.as_nanos() as f64;

        if total_nanos == 0.0 {
            return;
        }

        for (name, duration, calls) in &self.component_times {
            let percentage = (duration.as_nanos() as f64 / total_nanos) * 100.0;

            if percentage > 5.0 {
                let avg_per_call = if *calls > 0 {
                    Duration::from_nanos((duration.as_nanos() / *calls as u128) as u64)
                } else {
                    Duration::ZERO
                };

                let (explanation, is_expected) = explain_inference_hotspot(name, percentage);

                self.grid.add_hotspot(ProfilingHotspot {
                    component: name.clone(),
                    time: *duration,
                    percentage,
                    call_count: *calls,
                    avg_per_call,
                    explanation,
                    is_expected,
                });
            }
        }

        // Sort by percentage descending
        self.grid.hotspots.sort_by(|a, b| {
            b.percentage
                .partial_cmp(&a.percentage)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Render ASCII bar
fn render_bar(value: f64, max: f64, width: usize) -> String {
    let ratio = if max > 0.0 { value / max } else { 0.0 };
    let filled = ((ratio * width as f64) as usize).min(width);
    let empty = width - filled;

    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}

/// Truncate string to max length
fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        &s[..max_len]
    }
}

/// Explain inference hotspot
fn explain_inference_hotspot(component: &str, percentage: f64) -> (String, bool) {
    match component {
        "Q4K_GEMV" | "MatMul" | "GEMM" => (
            format!(
                "Matrix ops dominate ({:.1}%) - expected for transformer inference",
                percentage
            ),
            true,
        ),
        "Attention" | "FlashAttention" => (
            format!(
                "Attention at {:.1}% - normal for autoregressive decoding",
                percentage
            ),
            true,
        ),
        "KV_Cache" | "KVCache" => {
            if percentage > 20.0 {
                (
                    "KV cache overhead high - consider FP16 cache or graph capture".to_string(),
                    false,
                )
            } else {
                ("KV cache within normal range".to_string(), true)
            }
        },
        "Softmax" => {
            if percentage > 10.0 {
                (
                    "Softmax unusually high - check for redundant computations".to_string(),
                    false,
                )
            } else {
                ("Softmax within normal range".to_string(), true)
            }
        },
        "RMSNorm" | "LayerNorm" => {
            if percentage > 15.0 {
                (
                    "Normalization overhead high - consider fused kernels".to_string(),
                    false,
                )
            } else {
                ("Normalization within normal range".to_string(), true)
            }
        },
        "MemcpyH2D" | "MemcpyD2H" | "Transfer" => (
            "Memory transfer - consider persistent GPU buffers".to_string(),
            false,
        ),
        "KernelLaunch" => (
            "Kernel launch overhead - consider CUDA graphs or megakernels".to_string(),
            false,
        ),
        "Embedding" => (
            "Embedding lookup - expected at start of inference".to_string(),
            true,
        ),
        "Sampling" | "TopK" | "TopP" => (
            "Sampling overhead - expected for token generation".to_string(),
            true,
        ),
        _ => {
            if percentage > 20.0 {
                (
                    format!("Unknown component at {:.1}% - investigate", percentage),
                    false,
                )
            } else {
                (String::new(), true)
            }
        },
    }
}
