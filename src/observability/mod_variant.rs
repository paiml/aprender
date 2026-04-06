
impl ABTest {
    /// Create a new A/B test
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            variants: Vec::new(),
            active: true,
            start_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    /// Add a variant
    #[must_use]
    pub fn with_variant(
        mut self,
        name: impl Into<String>,
        model: impl Into<String>,
        weight: f64,
    ) -> Self {
        self.variants.push(ABVariant {
            name: name.into(),
            model: model.into(),
            weight: weight.clamp(0.0, 1.0),
        });
        self
    }

    /// Select a variant for a user (deterministic based on user ID)
    #[must_use]
    pub fn select(&self, user_id: &str) -> Option<&ABVariant> {
        if !self.active || self.variants.is_empty() {
            return None;
        }

        // Normalize weights
        let total_weight: f64 = self.variants.iter().map(|v| v.weight).sum();
        if total_weight <= 0.0 {
            return self.variants.first();
        }

        // Deterministic hash-based selection
        let hash = simple_hash(user_id);
        let selection = (hash as f64 / u64::MAX as f64) * total_weight;

        let mut cumulative = 0.0;
        for variant in &self.variants {
            cumulative += variant.weight;
            if selection < cumulative {
                return Some(variant);
            }
        }

        self.variants.last()
    }

    /// Check if test is valid (weights sum to ~1.0)
    #[must_use]
    pub fn is_valid(&self) -> bool {
        if self.variants.is_empty() {
            return false;
        }
        let total: f64 = self.variants.iter().map(|v| v.weight).sum();
        (total - 1.0).abs() < 0.001
    }
}

/// A/B test result tracking
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ABTestResult {
    /// Test name
    pub test_name: String,
    /// Results per variant
    pub variants: HashMap<String, VariantResult>,
}

/// Results for a single variant
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VariantResult {
    /// Number of requests
    pub requests: u64,
    /// Number of successes
    pub successes: u64,
    /// Total latency (ms)
    pub total_latency_ms: u64,
    /// Total tokens generated
    pub total_tokens: u64,
}

impl VariantResult {
    /// Calculate success rate
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.requests == 0 {
            0.0
        } else {
            self.successes as f64 / self.requests as f64
        }
    }

    /// Calculate average latency
    #[must_use]
    pub fn avg_latency_ms(&self) -> f64 {
        if self.requests == 0 {
            0.0
        } else {
            self.total_latency_ms as f64 / self.requests as f64
        }
    }

    /// Calculate tokens per request
    #[must_use]
    pub fn tokens_per_request(&self) -> f64 {
        if self.requests == 0 {
            0.0
        } else {
            self.total_tokens as f64 / self.requests as f64
        }
    }
}

// ============================================================================
// OBS-005: Observer (Main Interface)
// ============================================================================

/// Central observability interface
#[derive(Debug)]
pub struct Observer {
    /// Configuration
    config: ObservabilityConfig,
    /// Metrics buffer
    metrics_buffer: Arc<RwLock<Vec<MetricPoint>>>,
    /// Spans buffer
    spans_buffer: Arc<RwLock<Vec<Span>>>,
    /// A/B test results
    ab_results: Arc<RwLock<HashMap<String, ABTestResult>>>,
    /// Request counter
    request_counter: AtomicU64,
}

impl Observer {
    /// Create a new observer
    #[must_use]
    pub fn new(config: ObservabilityConfig) -> Self {
        Self {
            config,
            metrics_buffer: Arc::new(RwLock::new(Vec::new())),
            spans_buffer: Arc::new(RwLock::new(Vec::new())),
            ab_results: Arc::new(RwLock::new(HashMap::new())),
            request_counter: AtomicU64::new(0),
        }
    }

    /// Create with default configuration
    #[must_use]
    pub fn default_observer() -> Self {
        Self::new(ObservabilityConfig::new())
    }

    /// Record an inference metric
    pub fn record_inference(&self, model: &str, tokens: usize, latency_ms: u64) {
        // Tokens metric
        self.record_metric(
            MetricPoint::new("realizar_tokens_total", tokens as f64).with_label("model", model),
        );

        // Latency metric
        self.record_metric(
            MetricPoint::new("realizar_latency_ms", latency_ms as f64).with_label("model", model),
        );

        // Request count
        self.record_metric(
            MetricPoint::new("realizar_requests_total", 1.0).with_label("model", model),
        );
    }

    /// Record a custom metric
    pub fn record_metric(&self, metric: MetricPoint) {
        if let Ok(mut buffer) = self.metrics_buffer.write() {
            buffer.push(metric);
        }
    }

    /// Create a new trace
    #[must_use]
    pub fn start_trace(&self, operation: &str) -> Span {
        let trace_id = generate_id();
        Span::new(operation, trace_id)
    }

    /// Record a completed span
    pub fn record_span(&self, span: Span) {
        if !self.config.tracing_enabled {
            return;
        }

        // Sample based on rate
        if self.config.trace_sample_rate < 1.0 {
            let sample = simple_hash(&span.span_id) as f64 / u64::MAX as f64;
            if sample > self.config.trace_sample_rate {
                return;
            }
        }

        if let Ok(mut buffer) = self.spans_buffer.write() {
            buffer.push(span);
        }
    }

    /// Record A/B test result
    pub fn record_ab_result(
        &self,
        test_name: &str,
        variant_name: &str,
        success: bool,
        latency_ms: u64,
        tokens: u64,
    ) {
        if !self.config.ab_testing_enabled {
            return;
        }

        if let Ok(mut results) = self.ab_results.write() {
            let test_result =
                results
                    .entry(test_name.to_string())
                    .or_insert_with(|| ABTestResult {
                        test_name: test_name.to_string(),
                        variants: HashMap::new(),
                    });

            let variant_result = test_result
                .variants
                .entry(variant_name.to_string())
                .or_default();

            variant_result.requests += 1;
            if success {
                variant_result.successes += 1;
            }
            variant_result.total_latency_ms += latency_ms;
            variant_result.total_tokens += tokens;
        }
    }

    /// Get A/B test results
    #[must_use]
    pub fn get_ab_results(&self, test_name: &str) -> Option<ABTestResult> {
        self.ab_results
            .read()
            .ok()
            .and_then(|r| r.get(test_name).cloned())
    }

    /// Flush metrics to Trueno-DB (returns line protocol)
    pub fn flush_metrics(&self) -> Vec<String> {
        let metrics = if let Ok(mut buffer) = self.metrics_buffer.write() {
            std::mem::take(&mut *buffer)
        } else {
            Vec::new()
        };

        metrics.iter().map(MetricPoint::to_line_protocol).collect()
    }

    /// Flush spans
    pub fn flush_spans(&self) -> Vec<Span> {
        if let Ok(mut buffer) = self.spans_buffer.write() {
            std::mem::take(&mut *buffer)
        } else {
            Vec::new()
        }
    }

    /// Get next request ID
    pub fn next_request_id(&self) -> u64 {
        self.request_counter.fetch_add(1, Ordering::SeqCst)
    }

    /// Generate Prometheus-format metrics
    #[must_use]
    pub fn prometheus_metrics(&self) -> String {
        use std::fmt::Write;
        let metrics = if let Ok(buffer) = self.metrics_buffer.read() {
            buffer.clone()
        } else {
            Vec::new()
        };

        let mut output = String::new();
        let mut by_name: HashMap<String, Vec<&MetricPoint>> = HashMap::new();

        for metric in &metrics {
            by_name.entry(metric.name.clone()).or_default().push(metric);
        }

        for (name, points) in by_name {
            writeln!(output, "# TYPE {name} gauge").expect("fmt::Write for String is infallible");
            for point in points {
                let labels = if point.labels.is_empty() {
                    String::new()
                } else {
                    let pairs: Vec<String> = point
                        .labels
                        .iter()
                        .map(|(k, v)| format!("{k}=\"{v}\""))
                        .collect();
                    format!("{{{}}}", pairs.join(","))
                };
                writeln!(output, "{name}{labels} {}", point.value)
                    .expect("fmt::Write for String is infallible");
            }
        }

        output
    }
}

impl Default for Observer {
    fn default() -> Self {
        Self::default_observer()
    }
}

// ============================================================================
// OBS-006: Helper Functions
// ============================================================================

/// Generate a unique ID (16 hex chars / 8 bytes for span ID)
fn generate_id() -> String {
    use std::{
        collections::hash_map::RandomState,
        hash::{BuildHasher, Hasher},
    };

    let hasher_state = RandomState::new();
    let mut hasher = hasher_state.build_hasher();
    hasher.write_u64(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0),
    );
    format!("{:016x}", hasher.finish())
}

/// Generate a trace ID (32 hex chars / 16 bytes per W3C spec)
fn generate_trace_id() -> String {
    use std::{
        collections::hash_map::RandomState,
        hash::{BuildHasher, Hasher},
    };

    let hasher_state = RandomState::new();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);

    let mut hasher1 = hasher_state.build_hasher();
    hasher1.write_u64(now);
    let high = hasher1.finish();

    let mut hasher2 = hasher_state.build_hasher();
    hasher2.write_u64(now.wrapping_add(1));
    let low = hasher2.finish();

    format!("{high:016x}{low:016x}")
}

/// Simple hash function for deterministic selection (FNV-1a variant)
fn simple_hash(input: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;

    let mut hash: u64 = FNV_OFFSET;
    for byte in input.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
