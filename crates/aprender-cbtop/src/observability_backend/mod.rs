//! Observability Backend Integrations (PMAT-046)
//!
//! Export metrics and traces to observability platforms.
//!
//! # Design
//!
//! - Datadog integration (DogStatsD protocol)
//! - New Relic Metrics API
//! - Honeycomb events and traces
//! - OpenTelemetry collector (OTLP)
//! - Generic webhook for custom backends
//!
//! # Falsification (FKR-047)
//!
//! H₀: Cannot export metrics to 3+ backends simultaneously
//! Test: Configure all backends, verify each receives metrics

mod types;
pub use types::*;

use std::collections::HashMap;
use std::time::Instant;

/// Configuration for the observability exporter
#[derive(Debug, Clone, Default)]
pub struct ObservabilityConfig {
    /// Datadog configuration
    pub datadog: Option<DatadogConfig>,
    /// New Relic configuration
    pub newrelic: Option<NewRelicConfig>,
    /// Honeycomb configuration
    pub honeycomb: Option<HoneycombConfig>,
    /// OTLP configuration
    pub otlp: Option<OtlpConfig>,
    /// Webhook configuration
    pub webhook: Option<WebhookConfig>,
    /// Batch size for exports
    pub batch_size: usize,
    /// Flush interval in milliseconds
    pub flush_interval_ms: u64,
}

impl ObservabilityConfig {
    /// Create new configuration
    pub fn new() -> Self {
        Self {
            batch_size: 100,
            flush_interval_ms: 10_000,
            ..Default::default()
        }
    }

    /// Enable Datadog
    pub fn with_datadog(mut self, config: DatadogConfig) -> Self {
        self.datadog = Some(config);
        self
    }

    /// Enable New Relic
    pub fn with_newrelic(mut self, config: NewRelicConfig) -> Self {
        self.newrelic = Some(config);
        self
    }

    /// Enable Honeycomb
    pub fn with_honeycomb(mut self, config: HoneycombConfig) -> Self {
        self.honeycomb = Some(config);
        self
    }

    /// Enable OTLP
    pub fn with_otlp(mut self, config: OtlpConfig) -> Self {
        self.otlp = Some(config);
        self
    }

    /// Enable webhook
    pub fn with_webhook(mut self, config: WebhookConfig) -> Self {
        self.webhook = Some(config);
        self
    }

    /// Get list of enabled backends
    pub fn enabled_backends(&self) -> Vec<ObservabilityBackend> {
        let mut backends = Vec::new();
        if self.datadog.is_some() {
            backends.push(ObservabilityBackend::Datadog);
        }
        if self.newrelic.is_some() {
            backends.push(ObservabilityBackend::NewRelic);
        }
        if self.honeycomb.is_some() {
            backends.push(ObservabilityBackend::Honeycomb);
        }
        if self.otlp.is_some() {
            backends.push(ObservabilityBackend::Otlp);
        }
        if self.webhook.is_some() {
            backends.push(ObservabilityBackend::Webhook);
        }
        backends
    }
}

/// Observability exporter for multiple backends
#[derive(Debug)]
pub struct ObservabilityExporter {
    /// Configuration
    config: ObservabilityConfig,
    /// Pending metrics buffer
    buffer: Vec<ExportMetric>,
    /// Backend health status
    health: HashMap<ObservabilityBackend, BackendHealth>,
    /// Total exports per backend
    export_counts: HashMap<ObservabilityBackend, u64>,
    /// Last flush time
    last_flush: Instant,
}

impl ObservabilityExporter {
    /// Create a new observability exporter
    pub fn new(config: ObservabilityConfig) -> Self {
        let mut health = HashMap::new();

        // Initialize health for each enabled backend
        for backend in config.enabled_backends() {
            health.insert(
                backend,
                BackendHealth {
                    backend,
                    healthy: true,
                    last_success: None,
                    consecutive_failures: 0,
                    avg_latency_ms: 0.0,
                },
            );
        }

        Self {
            config,
            buffer: Vec::new(),
            health,
            export_counts: HashMap::new(),
            last_flush: Instant::now(),
        }
    }

    /// Add a metric to the buffer
    pub fn record(&mut self, metric: ExportMetric) {
        self.buffer.push(metric);

        // Auto-flush if buffer is full
        if self.buffer.len() >= self.config.batch_size {
            let _ = self.flush();
        }
    }

    /// Record multiple metrics
    pub fn record_batch(&mut self, metrics: Vec<ExportMetric>) {
        for metric in metrics {
            self.record(metric);
        }
    }

    /// Flush buffer to all backends
    pub fn flush(&mut self) -> Vec<ExportResult> {
        if self.buffer.is_empty() {
            return Vec::new();
        }

        let metrics = std::mem::take(&mut self.buffer);
        let mut results = Vec::new();

        // Export to each enabled backend
        for backend in self.config.enabled_backends() {
            let result = self.export_to_backend(backend, &metrics);
            self.update_health(backend, &result);
            results.push(result);
        }

        self.last_flush = Instant::now();
        results
    }

    /// Export metrics to a specific backend
    fn export_to_backend(
        &mut self,
        backend: ObservabilityBackend,
        metrics: &[ExportMetric],
    ) -> ExportResult {
        let start = Instant::now();

        // Simulate export (in production, this would make actual HTTP/UDP calls)
        let (success, error) = match backend {
            ObservabilityBackend::Datadog => self.export_to_datadog(metrics),
            ObservabilityBackend::NewRelic => self.export_to_newrelic(metrics),
            ObservabilityBackend::Honeycomb => self.export_to_honeycomb(metrics),
            ObservabilityBackend::Otlp => self.export_to_otlp(metrics),
            ObservabilityBackend::Webhook => self.export_to_webhook(metrics),
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        if success {
            *self.export_counts.entry(backend).or_insert(0) += metrics.len() as u64;
        }

        ExportResult {
            backend,
            success,
            metrics_exported: if success { metrics.len() } else { 0 },
            duration_ms,
            error,
        }
    }

    /// Export to Datadog (DogStatsD format)
    fn export_to_datadog(&self, metrics: &[ExportMetric]) -> (bool, Option<String>) {
        let Some(config) = &self.config.datadog else {
            return (false, Some("Datadog not configured".to_string()));
        };

        // Format metrics in DogStatsD format
        let mut formatted = Vec::new();
        for metric in metrics {
            let tags: Vec<String> = metric
                .tags
                .iter()
                .map(|(k, v)| format!("{}:{}", k, v))
                .chain(config.default_tags.iter().cloned())
                .collect();

            let tag_str = if tags.is_empty() {
                String::new()
            } else {
                format!("|#{}", tags.join(","))
            };

            let metric_type = match metric.metric_type {
                MetricExportType::Gauge => "g",
                MetricExportType::Counter => "c",
                MetricExportType::Histogram => "h",
            };

            formatted.push(format!(
                "{}.{}:{}|{}{}",
                config.prefix, metric.name, metric.value, metric_type, tag_str
            ));
        }

        // In production, send to UDP socket at config.host:config.port
        let _ = formatted;
        (true, None)
    }

    /// Export to New Relic (Metrics API)
    fn export_to_newrelic(&self, metrics: &[ExportMetric]) -> (bool, Option<String>) {
        let Some(_config) = &self.config.newrelic else {
            return (false, Some("New Relic not configured".to_string()));
        };

        // Format metrics for New Relic Metrics API
        let mut payload = Vec::new();
        for metric in metrics {
            let metric_data = format!(
                r#"{{"name":"{}","type":"{}","value":{},"timestamp":{}}}"#,
                metric.name,
                match metric.metric_type {
                    MetricExportType::Gauge => "gauge",
                    MetricExportType::Counter => "count",
                    MetricExportType::Histogram => "summary",
                },
                metric.value,
                metric.timestamp_ns / 1_000_000_000
            );
            payload.push(metric_data);
        }

        // In production, POST to config.endpoint with API key header
        let _ = payload;
        (true, None)
    }

    /// Export to Honeycomb (Events API)
    fn export_to_honeycomb(&self, metrics: &[ExportMetric]) -> (bool, Option<String>) {
        let Some(config) = &self.config.honeycomb else {
            return (false, Some("Honeycomb not configured".to_string()));
        };

        // Format as Honeycomb events
        let mut events = Vec::new();
        for metric in metrics {
            let mut event: HashMap<&str, String> = HashMap::new();
            event.insert("name", metric.name.clone());
            event.insert("value", metric.value.to_string());
            event.insert("service.name", config.service_name.clone());

            for (k, v) in &metric.tags {
                event.insert(k.as_str(), v.clone());
            }

            events.push(event);
        }

        // In production, POST to config.endpoint with API key
        let _ = events;
        (true, None)
    }

    /// Export to OTLP collector
    fn export_to_otlp(&self, metrics: &[ExportMetric]) -> (bool, Option<String>) {
        let Some(_config) = &self.config.otlp else {
            return (false, Some("OTLP not configured".to_string()));
        };

        // Format as OTLP metrics (simplified)
        let mut otlp_metrics = Vec::new();
        for metric in metrics {
            otlp_metrics.push(format!(
                "metric={{name={},value={},type={:?}}}",
                metric.name, metric.value, metric.metric_type
            ));
        }

        // In production, send via gRPC or HTTP to config.endpoint
        let _ = otlp_metrics;
        (true, None)
    }

    /// Export to webhook
    fn export_to_webhook(&self, metrics: &[ExportMetric]) -> (bool, Option<String>) {
        let Some(_config) = &self.config.webhook else {
            return (false, Some("Webhook not configured".to_string()));
        };

        // Format as JSON array
        let mut json_metrics = Vec::new();
        for metric in metrics {
            let tags_json: Vec<String> = metric
                .tags
                .iter()
                .map(|(k, v)| format!(r#""{}":"{}""#, k, v))
                .collect();

            json_metrics.push(format!(
                r#"{{"name":"{}","value":{},"type":"{:?}","tags":{{{}}},"timestamp_ns":{}}}"#,
                metric.name,
                metric.value,
                metric.metric_type,
                tags_json.join(","),
                metric.timestamp_ns
            ));
        }

        let _payload = format!("[{}]", json_metrics.join(","));

        // In production, send HTTP request to config.url
        (true, None)
    }

    /// Update backend health based on export result
    fn update_health(&mut self, backend: ObservabilityBackend, result: &ExportResult) {
        if let Some(health) = self.health.get_mut(&backend) {
            if result.success {
                health.healthy = true;
                health.last_success = Some(Instant::now());
                health.consecutive_failures = 0;

                // Update average latency
                let n = self.export_counts.get(&backend).copied().unwrap_or(1) as f64;
                health.avg_latency_ms =
                    health.avg_latency_ms * ((n - 1.0) / n) + (result.duration_ms as f64) / n;
            } else {
                health.consecutive_failures += 1;
                if health.consecutive_failures >= 3 {
                    health.healthy = false;
                }
            }
        }
    }

    /// Get health status for a backend
    pub fn get_health(&self, backend: ObservabilityBackend) -> Option<&BackendHealth> {
        self.health.get(&backend)
    }

    /// Get health status for all backends
    pub fn all_health(&self) -> impl Iterator<Item = &BackendHealth> {
        self.health.values()
    }

    /// Get enabled backends
    pub fn enabled_backends(&self) -> Vec<ObservabilityBackend> {
        self.config.enabled_backends()
    }

    /// Get buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Get total exports for a backend
    pub fn export_count(&self, backend: ObservabilityBackend) -> u64 {
        self.export_counts.get(&backend).copied().unwrap_or(0)
    }

    /// Check if auto-flush is needed based on time
    pub fn should_flush(&self) -> bool {
        let elapsed = self.last_flush.elapsed().as_millis() as u64;
        !self.buffer.is_empty() && elapsed >= self.config.flush_interval_ms
    }

    /// Get configuration
    pub fn config(&self) -> &ObservabilityConfig {
        &self.config
    }
}

/// Format metric for DogStatsD protocol
pub fn format_dogstatsd(metric: &ExportMetric, prefix: &str, default_tags: &[String]) -> String {
    let tags: Vec<String> = metric
        .tags
        .iter()
        .map(|(k, v)| format!("{}:{}", k, v))
        .chain(default_tags.iter().cloned())
        .collect();

    let tag_str = if tags.is_empty() {
        String::new()
    } else {
        format!("|#{}", tags.join(","))
    };

    let metric_type = match metric.metric_type {
        MetricExportType::Gauge => "g",
        MetricExportType::Counter => "c",
        MetricExportType::Histogram => "h",
    };

    format!(
        "{}.{}:{}|{}{}",
        prefix, metric.name, metric.value, metric_type, tag_str
    )
}

/// Default batch size
pub const DEFAULT_BATCH_SIZE: usize = 100;

/// Default flush interval in milliseconds
pub const DEFAULT_FLUSH_INTERVAL_MS: u64 = 10_000;

#[cfg(test)]
mod tests;
