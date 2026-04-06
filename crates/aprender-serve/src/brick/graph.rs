
/// Benchmark report with statistical analysis.
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// Brick name
    pub brick_name: String,
    /// Mean latency (µs)
    pub mean_us: f64,
    /// Standard deviation (µs)
    pub std_us: f64,
    /// Coefficient of variation
    pub cv: f64,
    /// 50th percentile (µs)
    pub p50_us: f64,
    /// 99th percentile (µs)
    pub p99_us: f64,
    /// Throughput (tokens/sec)
    pub tokens_per_sec: f64,
    /// Budget target (µs)
    pub budget_us: f64,
    /// Budget met?
    pub budget_met: bool,
    /// Statistical validity (CV < max_cv)
    pub statistically_valid: bool,
}

impl fmt::Display for BenchmarkReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.budget_met { "PASS" } else { "FAIL" };
        write!(
            f,
            "{}: {:.1}µs ± {:.1}µs (CV={:.1}%) | {:.0} tok/s | budget: {} ({})",
            self.brick_name,
            self.mean_us,
            self.std_us,
            self.cv * 100.0,
            self.tokens_per_sec,
            self.budget_us,
            status
        )
    }
}

/// Calculate percentile from sorted samples.
fn percentile(samples: &[f64], p: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let idx = ((samples.len() as f64) * p).floor() as usize;
    samples[idx.min(samples.len() - 1)]
}

/// Run benchmark on a brick with statistical rigor.
pub fn benchmark_brick<B: ComputeBrick>(
    brick: &B,
    run_fn: impl Fn() -> f64,
    config: &BenchmarkConfig,
) -> BenchmarkReport {
    // Warmup (Jidoka: ensure stable state)
    for _ in 0..config.warmup {
        let _ = run_fn();
    }

    // Collect samples
    let mut samples: Vec<f64> = Vec::with_capacity(config.samples);
    for _ in 0..config.samples {
        samples.push(run_fn());
    }

    // Sort for percentiles
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Statistical analysis
    let n = samples.len().max(1) as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let std =
        (samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
    let cv = if mean.abs() > f64::EPSILON { std / mean } else { 0.0 };

    let budget = brick.budget();

    BenchmarkReport {
        brick_name: brick.name().to_string(),
        mean_us: mean,
        std_us: std,
        cv,
        p50_us: percentile(&samples, 0.50),
        p99_us: percentile(&samples, 0.99),
        tokens_per_sec: if mean > 0.0 { 1_000_000.0 / mean } else { 0.0 },
        budget_us: budget.us_per_token,
        budget_met: mean <= budget.us_per_token,
        statistically_valid: cv <= config.max_cv,
    }
}

// ============================================================================
// CUDA Graph Brick (Section 5.2 - P0)
// ============================================================================

/// CUDA Graph Brick for eliminating kernel launch overhead.
///
/// Per spec: docs/specifications/qwen2.5-coder-showcase-demo.md §5.2
///
/// Uses CUDA graph capture to reduce ~280 kernel launches to single graph replay.
/// Expected impact: 5.6ms overhead → 0.02ms = 280x overhead reduction.
///
/// # Implementation
///
/// Wraps `CudaExecutor::decode_graph` and `try_graph_capture()` from cuda.rs.
/// Uses indirect kernels (KvCacheScatterIndirect, RopeIndirect) for graph compatibility.
#[derive(Debug, Clone)]
pub struct CudaGraphBrick {
    /// Number of layers captured in graph
    pub num_layers: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Whether graph is currently captured
    pub captured: bool,
    /// Token budget (target: 10µs launch overhead vs 5600µs eager)
    budget: TokenBudget,
}

impl CudaGraphBrick {
    /// Create new CUDA Graph brick for model configuration.
    #[must_use]
    pub fn new(num_layers: usize, hidden_dim: usize) -> Self {
        // Graph overhead should be < 100µs (vs ~5.6ms for 280 launches)
        let budget_us = 20.0; // Conservative: 20µs for graph replay
        Self {
            num_layers,
            hidden_dim,
            captured: false,
            budget: TokenBudget::from_latency(budget_us),
        }
    }

    /// Set custom budget.
    #[must_use]
    pub fn with_budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Mark graph as captured.
    pub fn set_captured(&mut self, captured: bool) {
        self.captured = captured;
    }

    /// Check if graph can be used (captured and valid).
    #[must_use]
    pub fn can_replay(&self) -> bool {
        self.captured
    }

    /// Replay the captured graph (stub - actual execution via CudaExecutor).
    pub fn replay(&self) -> Result<(), BrickError> {
        if !self.captured {
            return Err(BrickError::ComputeError(
                "CUDA graph not captured yet".to_string(),
            ));
        }
        // Actual replay would be done via CudaExecutor::forward_graphed()
        Ok(())
    }
}

impl ComputeBrick for CudaGraphBrick {
    type Output = ();

    fn name(&self) -> &'static str {
        "cuda_graph"
    }

    fn budget(&self) -> TokenBudget {
        self.budget
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::budget_met(),
            BrickAssertion {
                name: "graph_speedup".to_string(),
                description: "Graph replay faster than eager execution".to_string(),
                kind: AssertionKind::Custom {
                    check_name: "graph_speedup".to_string(),
                },
            },
        ]
    }

    fn can_run(&self) -> bool {
        self.num_layers > 0 && self.hidden_dim > 0
    }
}

// ============================================================================
// Tests (F001-F020)
// ============================================================================

// Tests extracted to tests.rs (PMAT-802)
#[cfg(test)]
#[path = "tests.rs"]
mod brick_tests;

// Additional tests in tests_part_02.rs
#[cfg(test)]
#[path = "tests_token_budget.rs"]
mod brick_tests_part_02;

// Additional tests in tests_part_03.rs
#[cfg(test)]
#[path = "tests_flash_attention.rs"]
mod brick_tests_part_03;

// tests_part_04 through tests_part_08 are now include!() fragments inside tests.rs

// BrickProfiler tests (PMAT-112)
#[cfg(test)]
#[path = "profiler_tests.rs"]
mod profiler_tests;

// Fused ops tests (cuda feature)
#[cfg(all(test, feature = "cuda"))]
mod fused_tests;
