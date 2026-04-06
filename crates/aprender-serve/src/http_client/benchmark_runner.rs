
impl HttpBenchmarkRunner {
    /// Create a new benchmark runner
    #[must_use]
    pub fn new(config: HttpBenchmarkConfig) -> Self {
        let outlier_detector = OutlierDetector::new(config.outlier_k_factor);
        Self {
            client: ModelHttpClient::with_timeout(120), // 2 minute timeout for benchmarks
            config,
            preflight_runner: None,
            outlier_detector,
        }
    }

    /// Create with default configuration (5% CV, preflight enabled, outlier filtering)
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(HttpBenchmarkConfig::default())
    }

    /// Create with relaxed configuration (20% CV, no preflight, no outlier filtering)
    #[must_use]
    pub fn with_relaxed() -> Self {
        Self::new(HttpBenchmarkConfig::relaxed())
    }

    /// Create with reproducible configuration (3% CV, strict preflight, outlier filtering)
    #[must_use]
    pub fn with_reproducible() -> Self {
        Self::new(HttpBenchmarkConfig::reproducible())
    }

    /// Shared preflight validation logic for any server type
    fn run_preflight_impl(
        &mut self,
        base_url: &str,
        default_port: u16,
        make_check: fn(u16) -> ServerAvailabilityCheck,
        health_check: impl FnOnce(&ModelHttpClient, &str) -> Result<bool>,
    ) -> Result<Vec<String>> {
        let mut runner = PreflightRunner::new();

        let url_parts: Vec<&str> = base_url.trim_end_matches('/').split(':').collect();
        let port = url_parts
            .last()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(default_port);

        let mut server_check = make_check(port);

        match health_check(&self.client, base_url) {
            Ok(true) => server_check.set_health_status(200),
            Ok(false) => server_check.set_health_status(500),
            Err(_) => server_check.set_health_status(0),
        }

        runner.add_check(Box::new(server_check));

        let passed = runner
            .run()
            .map_err(|e| RealizarError::ConnectionError(format!("Preflight failed: {}", e)))?;

        self.preflight_runner = Some(runner);
        Ok(passed)
    }

    /// Run preflight validation for llama.cpp server
    ///
    /// # Errors
    /// Returns error if preflight validation fails
    pub fn run_preflight_llamacpp(&mut self, base_url: &str) -> Result<Vec<String>> {
        self.run_preflight_impl(
            base_url,
            8080,
            ServerAvailabilityCheck::llama_cpp,
            |client, url| client.health_check_openai(url),
        )
    }

    /// Run preflight validation for Ollama server
    ///
    /// # Errors
    /// Returns error if preflight validation fails
    pub fn run_preflight_ollama(&mut self, base_url: &str) -> Result<Vec<String>> {
        self.run_preflight_impl(
            base_url,
            11434,
            ServerAvailabilityCheck::ollama,
            |client, url| client.health_check_ollama(url),
        )
    }

    /// Get list of passed preflight checks
    #[must_use]
    pub fn preflight_checks_passed(&self) -> Vec<String> {
        self.preflight_runner
            .as_ref()
            .map(|r| r.passed_checks().to_vec())
            .unwrap_or_default()
    }

    /// Shared measurement loop: warmup, CV-based stopping, outlier detection
    ///
    /// The `send_request` closure is called for both warmup (errors ignored) and
    /// measurement iterations. This avoids duplicating the entire loop structure.
    fn benchmark_measure_loop(
        &mut self,
        mut send_request: impl FnMut(&ModelHttpClient) -> Result<InferenceTiming>,
    ) -> Result<HttpBenchmarkResult> {
        let mut latencies = Vec::with_capacity(self.config.max_samples());
        let mut throughputs = Vec::with_capacity(self.config.max_samples());
        let mut cold_start_ms = 0.0;

        // Warmup -- CB-121: errors expected and harmless before server is ready
        for _ in 0..self.config.warmup_iterations {
            drop(send_request(&self.client));
        }

        // Measurement loop with CV-based stopping (per Hoefler & Belli SC'15)
        for i in 0..self.config.max_samples() {
            let timing = send_request(&self.client)?;

            latencies.push(timing.total_time_ms);
            if timing.tokens_generated > 0 {
                let tps = (timing.tokens_generated as f64) / (timing.total_time_ms / 1000.0);
                throughputs.push(tps);
            }

            if i == 0 {
                cold_start_ms = timing.total_time_ms;
            }

            if let StopDecision::Stop(_) = self.config.cv_criterion.should_stop(&latencies) {
                break;
            }
        }

        // Apply outlier detection if configured
        let (filtered_latencies, outliers_detected, outliers_excluded) =
            if self.config.filter_outliers {
                let outliers = self.outlier_detector.detect(&latencies);
                let outlier_count = outliers.iter().filter(|&&x| x).count();
                let filtered = self.outlier_detector.filter(&latencies);
                (filtered, outlier_count, outlier_count)
            } else {
                (latencies.clone(), 0, 0)
            };

        self.compute_results_with_quality(
            &latencies,
            &filtered_latencies,
            &throughputs,
            cold_start_ms,
            outliers_detected,
            outliers_excluded,
        )
    }

    /// Run benchmark against llama.cpp server
    ///
    /// Per spec v1.0.1:
    /// 1. Runs optional preflight validation (Jidoka)
    /// 2. Uses CV-based stopping criterion (Hoefler & Belli SC'15)
    /// 3. Applies MAD-based outlier detection
    /// 4. Returns quality metrics for reproducibility assessment
    ///
    /// # Errors
    /// Returns error if preflight fails or server is unreachable
    pub fn benchmark_llamacpp(&mut self, base_url: &str) -> Result<HttpBenchmarkResult> {
        if self.config.run_preflight {
            self.run_preflight_llamacpp(base_url)?;
        }

        let prompt = self.config.prompt.clone();
        let max_tokens = self.config.max_tokens;
        let temperature = self.config.temperature;
        let base = base_url.to_owned();

        self.benchmark_measure_loop(move |client| {
            let request = CompletionRequest {
                model: "default".to_string(),
                prompt: prompt.clone(),
                max_tokens,
                temperature: Some(temperature),
                stream: false,
            };
            client.llamacpp_completion(&base, &request)
        })
    }

    /// Run benchmark against Ollama server
    ///
    /// Per spec v1.0.1:
    /// 1. Runs optional preflight validation (Jidoka)
    /// 2. Uses CV-based stopping criterion (Hoefler & Belli SC'15)
    /// 3. Applies MAD-based outlier detection
    /// 4. Returns quality metrics for reproducibility assessment
    ///
    /// # Errors
    /// Returns error if preflight fails or server is unreachable
    pub fn benchmark_ollama(&mut self, base_url: &str, model: &str) -> Result<HttpBenchmarkResult> {
        if self.config.run_preflight {
            self.run_preflight_ollama(base_url)?;
        }

        let prompt = self.config.prompt.clone();
        let max_tokens = self.config.max_tokens;
        let temperature = self.config.temperature;
        let base = base_url.to_owned();
        let model_name = model.to_owned();

        self.benchmark_measure_loop(move |client| {
            let request = OllamaRequest {
                model: model_name.clone(),
                prompt: prompt.clone(),
                stream: false,
                options: Some(OllamaOptions {
                    num_predict: Some(max_tokens),
                    temperature: Some(temperature),
                }),
            };
            client.ollama_generate(&base, &request)
        })
    }

    /// Calculate coefficient of variation (kept for backward compatibility in tests)
    #[cfg(test)]
    fn calculate_cv(samples: &[f64]) -> f64 {
        if samples.len() < 2 {
            return f64::MAX;
        }

        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;

        if mean.abs() < f64::EPSILON {
            return f64::MAX;
        }

        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);

        let std_dev = variance.sqrt();
        std_dev / mean
    }

    /// Compute final benchmark results with quality metrics
    ///
    /// Per spec v1.0.1: Uses filtered samples for statistics but reports both raw and filtered
    fn compute_results_with_quality(
        &self,
        raw_latencies: &[f64],
        filtered_latencies: &[f64],
        throughputs: &[f64],
        cold_start_ms: f64,
        outliers_detected: usize,
        outliers_excluded: usize,
    ) -> Result<HttpBenchmarkResult> {
        // Use filtered samples for statistics
        let latencies = if filtered_latencies.is_empty() {
            raw_latencies
        } else {
            filtered_latencies
        };

        if latencies.is_empty() {
            return Err(RealizarError::InferenceError(
                "No valid samples collected".to_string(),
            ));
        }

        let (mean_latency_ms, std_dev_ms, cv, p50_latency_ms, p99_latency_ms) =
            Self::latency_stats(latencies);
        let throughput_tps = Self::mean_throughput(throughputs);
        let cv_converged = cv < self.config.cv_threshold();

        Ok(HttpBenchmarkResult {
            latency_samples: raw_latencies.to_vec(),
            latency_samples_filtered: filtered_latencies.to_vec(),
            mean_latency_ms,
            p50_latency_ms,
            p99_latency_ms,
            std_dev_ms,
            cv_at_stop: cv,
            throughput_tps,
            cold_start_ms,
            sample_count: raw_latencies.len(),
            filtered_sample_count: filtered_latencies.len(),
            cv_converged,
            quality_metrics: QualityMetrics {
                cv_at_stop: cv,
                cv_converged,
                outliers_detected,
                outliers_excluded,
                preflight_checks_passed: self.preflight_checks_passed(),
            },
        })
    }

    /// Compute latency statistics: (mean, std_dev, cv, p50, p99).
    fn latency_stats(latencies: &[f64]) -> (f64, f64, f64, f64, f64) {
        let n = latencies.len() as f64;
        let mean = if n > 0.0 {
            latencies.iter().sum::<f64>() / n
        } else {
            0.0
        };
        let variance = if n > 1.0 {
            latencies.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
        } else {
            0.0
        };
        let std_dev = variance.sqrt();
        let cv = if mean.abs() > f64::EPSILON {
            std_dev / mean
        } else {
            f64::MAX
        };
        let mut sorted = latencies.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p50 = sorted.get(latencies.len() / 2).copied().unwrap_or(0.0);
        let p99_idx = (latencies.len() * 99 / 100).min(latencies.len().saturating_sub(1));
        let p99 = sorted.get(p99_idx).copied().unwrap_or(0.0);
        (mean, std_dev, cv, p50, p99)
    }

    /// Compute mean throughput (tokens per second).
    fn mean_throughput(throughputs: &[f64]) -> f64 {
        if throughputs.is_empty() {
            0.0
        } else {
            throughputs.iter().sum::<f64>() / throughputs.len() as f64
        }
    }

    /// Backward-compatible compute_results (for tests)
    #[cfg(test)]
    fn compute_results(
        latencies: &[f64],
        throughputs: &[f64],
        cold_start_ms: f64,
        cv_threshold: f64,
    ) -> HttpBenchmarkResult {
        let (mean_latency_ms, std_dev_ms, cv, p50_latency_ms, p99_latency_ms) =
            Self::latency_stats(latencies);
        let throughput_tps = Self::mean_throughput(throughputs);
        let cv_converged = cv < cv_threshold;

        HttpBenchmarkResult {
            latency_samples: latencies.to_vec(),
            latency_samples_filtered: latencies.to_vec(),
            mean_latency_ms,
            p50_latency_ms,
            p99_latency_ms,
            std_dev_ms,
            cv_at_stop: cv,
            throughput_tps,
            cold_start_ms,
            sample_count: latencies.len(),
            filtered_sample_count: latencies.len(),
            cv_converged,
            quality_metrics: QualityMetrics {
                cv_at_stop: cv,
                cv_converged,
                outliers_detected: 0,
                outliers_excluded: 0,
                preflight_checks_passed: Vec::new(),
            },
        }
    }
}
