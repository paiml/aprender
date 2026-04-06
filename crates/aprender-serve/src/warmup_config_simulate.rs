
impl WarmupExecutor {
    /// Create a new warm-up executor
    #[must_use]
    pub fn new(config: WarmupConfig) -> Self {
        Self { config }
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &WarmupConfig {
        &self.config
    }

    /// Simulate warm-up (for testing without actual model)
    ///
    /// This runs the warm-up process with test inference delays.
    #[must_use]
    pub fn simulate_warmup(&self) -> WarmupResult {
        let start = Instant::now();
        let mut latencies = Vec::with_capacity(self.config.warmup_iterations);

        // Simulate decreasing latency as model warms up
        for i in 0..self.config.warmup_iterations {
            // First iteration is "cold" (slower)
            let base_latency_us = if i == 0 { 1000 } else { 100 };
            let jitter = (i * 10) as u64;
            let latency = Duration::from_micros(base_latency_us - jitter.min(50));
            latencies.push(latency);
        }

        WarmupResult::success(self.config.warmup_iterations, start.elapsed(), &latencies)
    }

    /// Check if timeout has been exceeded
    #[allow(dead_code)]
    fn check_timeout(&self, start: Instant, iterations: usize) -> Option<WarmupResult> {
        if start.elapsed() > self.config.timeout {
            Some(WarmupResult::timed_out(iterations, start.elapsed()))
        } else {
            None
        }
    }
}

impl Default for WarmupExecutor {
    fn default() -> Self {
        Self::new(WarmupConfig::default())
    }
}

// ============================================================================
// WARM-006: Pre-load Configuration
// ============================================================================

/// Configuration for model pre-loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreloadConfig {
    /// Models to pre-load on startup
    pub models: Vec<PreloadModelConfig>,
    /// Load models in parallel
    pub parallel_loading: bool,
    /// Maximum concurrent model loads
    pub max_concurrent: usize,
    /// Fail startup if any model fails to load
    pub fail_fast: bool,
}

impl Default for PreloadConfig {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            parallel_loading: true,
            max_concurrent: 4,
            fail_fast: false,
        }
    }
}

impl PreloadConfig {
    /// Create new pre-load configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a model to pre-load
    #[must_use]
    pub fn with_model(mut self, model: PreloadModelConfig) -> Self {
        self.models.push(model);
        self
    }

    /// Set parallel loading
    #[must_use]
    pub fn with_parallel_loading(mut self, parallel: bool) -> Self {
        self.parallel_loading = parallel;
        self
    }

    /// Set maximum concurrent loads
    #[must_use]
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max.max(1);
        self
    }

    /// Set fail-fast behavior
    #[must_use]
    pub fn with_fail_fast(mut self, fail_fast: bool) -> Self {
        self.fail_fast = fail_fast;
        self
    }
}

/// Configuration for a single model to pre-load
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreloadModelConfig {
    /// Model identifier
    pub model_id: String,
    /// Model URI (pacha://, file://, hf://)
    pub uri: String,
    /// Priority (lower = load first)
    pub priority: u32,
    /// Run warm-up after loading
    pub warmup: bool,
    /// Warm-up configuration
    pub warmup_config: Option<WarmupConfig>,
}

impl PreloadModelConfig {
    /// Create new model pre-load config
    #[must_use]
    pub fn new(model_id: impl Into<String>, uri: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            uri: uri.into(),
            priority: 100,
            warmup: true,
            warmup_config: None,
        }
    }

    /// Set priority
    #[must_use]
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Enable/disable warm-up
    #[must_use]
    pub fn with_warmup(mut self, warmup: bool) -> Self {
        self.warmup = warmup;
        self
    }

    /// Set warm-up configuration
    #[must_use]
    pub fn with_warmup_config(mut self, config: WarmupConfig) -> Self {
        self.warmup_config = Some(config);
        self.warmup = true;
        self
    }
}
