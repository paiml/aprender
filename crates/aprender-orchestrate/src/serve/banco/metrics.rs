//! Prometheus-compatible metrics endpoint.
//!
//! Tracks request counts, latency, and system state.
//! GET /api/v1/metrics returns text/plain in Prometheus exposition format.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Server-wide metrics collector.
pub struct MetricsCollector {
    /// Total requests served.
    pub total_requests: AtomicU64,
    /// Total chat completion requests.
    pub chat_requests: AtomicU64,
    /// Total inference tokens generated.
    pub tokens_generated: AtomicU64,
    /// Total errors (4xx + 5xx).
    pub errors: AtomicU64,
    /// Server start time.
    start: Instant,
}

impl MetricsCollector {
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            chat_requests: AtomicU64::new(0),
            tokens_generated: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            start: Instant::now(),
        }
    }

    /// Increment total request counter.
    pub fn inc_requests(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment chat request counter.
    pub fn inc_chat(&self) {
        self.chat_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Add generated tokens.
    pub fn add_tokens(&self, n: u64) {
        self.tokens_generated.fetch_add(n, Ordering::Relaxed);
    }

    /// Increment error counter.
    pub fn inc_errors(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Render metrics in Prometheus exposition format.
    #[must_use]
    pub fn render(&self, model_loaded: bool, endpoint_count: u64) -> String {
        let uptime = self.start.elapsed().as_secs();
        let total = self.total_requests.load(Ordering::Relaxed);
        let chat = self.chat_requests.load(Ordering::Relaxed);
        let tokens = self.tokens_generated.load(Ordering::Relaxed);
        let errors = self.errors.load(Ordering::Relaxed);

        format!(
            "# HELP banco_requests_total Total HTTP requests served.\n\
             # TYPE banco_requests_total counter\n\
             banco_requests_total {total}\n\
             # HELP banco_chat_requests_total Total chat completion requests.\n\
             # TYPE banco_chat_requests_total counter\n\
             banco_chat_requests_total {chat}\n\
             # HELP banco_tokens_generated_total Total tokens generated.\n\
             # TYPE banco_tokens_generated_total counter\n\
             banco_tokens_generated_total {tokens}\n\
             # HELP banco_errors_total Total error responses.\n\
             # TYPE banco_errors_total counter\n\
             banco_errors_total {errors}\n\
             # HELP banco_uptime_seconds Server uptime in seconds.\n\
             # TYPE banco_uptime_seconds gauge\n\
             banco_uptime_seconds {uptime}\n\
             # HELP banco_model_loaded Whether a model is loaded (1=yes, 0=no).\n\
             # TYPE banco_model_loaded gauge\n\
             banco_model_loaded {}\n\
             # HELP banco_endpoints_total Number of registered API endpoints.\n\
             # TYPE banco_endpoints_total gauge\n\
             banco_endpoints_total {endpoint_count}\n",
            if model_loaded { 1 } else { 0 }
        )
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}
