//! Metrics collection and reporting for production monitoring
//!
//! This module provides comprehensive metrics for tracking:
//! - Request latency (p50, p95, p99)
//! - Throughput (requests/sec, tokens/sec)
//! - Error rates and categorization
//! - Model performance (inference time, token generation)
//!
//! Metrics are exposed in Prometheus format for easy integration with monitoring systems.

use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

/// How many recent request latencies are kept for percentile reporting.
///
/// A bounded window: memory is constant regardless of uptime, and the reported
/// percentiles describe recent behaviour rather than the whole life of the
/// process — which is what a monitor graphing p95 wants.
const LATENCY_WINDOW: usize = 1024;

/// Measured request-latency percentiles, in milliseconds.
///
/// aprender#2375(7): `/v1/metrics` reported `latency_p50/p95/p99 = 0.0` while
/// `/metrics` on the same process at the same instant reported
/// `avg_latency_ms 626.79` over the same 27 requests — the collector kept only
/// running totals, so there was no distribution to take a percentile of. The
/// non-GPU variant of the handler filled the gap by *deriving* p95 as
/// `avg * 1.5` and p99 as `avg * 2.0`, which are not measurements of anything.
/// These are order statistics over the real samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatencyPercentiles {
    /// Median request latency (ms).
    pub p50_ms: f64,
    /// 95th-percentile request latency (ms).
    pub p95_ms: f64,
    /// 99th-percentile request latency (ms).
    pub p99_ms: f64,
}

/// Central metrics collector for tracking system performance
#[derive(Debug, Clone)]
pub struct MetricsCollector {
    /// Total number of requests processed
    total_requests: Arc<AtomicUsize>,
    /// Total number of successful requests
    successful_requests: Arc<AtomicUsize>,
    /// Total number of failed requests
    failed_requests: Arc<AtomicUsize>,
    /// Total number of tokens generated
    total_tokens: Arc<AtomicUsize>,
    /// Total inference time in microseconds
    total_inference_time_us: Arc<AtomicU64>,
    /// Most recent per-request latencies in microseconds (oldest first), capped
    /// at [`LATENCY_WINDOW`]. This is the sample set the percentiles come from.
    recent_latencies_us: Arc<Mutex<VecDeque<u64>>>,
    /// Start time for rate calculations
    start_time: Instant,
}

impl MetricsCollector {
    /// Create a new metrics collector
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_requests: Arc::new(AtomicUsize::new(0)),
            successful_requests: Arc::new(AtomicUsize::new(0)),
            failed_requests: Arc::new(AtomicUsize::new(0)),
            total_tokens: Arc::new(AtomicUsize::new(0)),
            total_inference_time_us: Arc::new(AtomicU64::new(0)),
            recent_latencies_us: Arc::new(Mutex::new(VecDeque::with_capacity(LATENCY_WINDOW))),
            start_time: Instant::now(),
        }
    }

    /// Record a successful request
    #[allow(clippy::cast_possible_truncation)]
    pub fn record_success(&self, tokens: usize, duration: Duration) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.successful_requests.fetch_add(1, Ordering::Relaxed);
        self.total_tokens.fetch_add(tokens, Ordering::Relaxed);
        let elapsed_us = duration.as_micros() as u64;
        self.total_inference_time_us
            .fetch_add(elapsed_us, Ordering::Relaxed);
        // aprender#2375(7): keep the SAMPLE, not just the sum. Without it there
        // is no distribution, and the percentile fields could only be zeros or
        // multiples of the mean.
        if let Ok(mut samples) = self.recent_latencies_us.lock() {
            if samples.len() == LATENCY_WINDOW {
                samples.pop_front();
            }
            samples.push_back(elapsed_us);
        }
    }

    /// Percentiles over the recent request-latency window.
    ///
    /// `None` when no successful request has been recorded — an honest "not
    /// measured yet", which the caller must not render as a measured `0.0`.
    ///
    /// Nearest-rank order statistics (no interpolation): with `n` samples the
    /// q-th percentile is the `ceil(q*n)`-th smallest, so `p95` of 100 samples
    /// is the 95th smallest and can never be a scaled mean.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    pub fn latency_percentiles(&self) -> Option<LatencyPercentiles> {
        let mut samples: Vec<u64> = {
            let guard = self.recent_latencies_us.lock().ok()?;
            if guard.is_empty() {
                return None;
            }
            guard.iter().copied().collect()
        };
        samples.sort_unstable();

        let nth = |quantile: f64| -> f64 {
            let n = samples.len();
            let rank = (quantile * n as f64).ceil().max(1.0) as usize;
            let index = rank.min(n) - 1;
            samples[index] as f64 / 1000.0
        };
        Some(LatencyPercentiles {
            p50_ms: nth(0.50),
            p95_ms: nth(0.95),
            p99_ms: nth(0.99),
        })
    }

    /// Record a failed request
    pub fn record_failure(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current snapshot of metrics
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn snapshot(&self) -> MetricsSnapshot {
        let total_requests = self.total_requests.load(Ordering::Relaxed);
        let successful = self.successful_requests.load(Ordering::Relaxed);
        let failed = self.failed_requests.load(Ordering::Relaxed);
        let total_tokens = self.total_tokens.load(Ordering::Relaxed);
        let total_time_us = self.total_inference_time_us.load(Ordering::Relaxed);
        let uptime = self.start_time.elapsed();

        MetricsSnapshot {
            total_requests,
            successful_requests: successful,
            failed_requests: failed,
            total_tokens,
            total_inference_time_us: total_time_us,
            uptime_secs: uptime.as_secs(),
            requests_per_sec: if uptime.as_secs() > 0 {
                total_requests as f64 / uptime.as_secs_f64()
            } else {
                0.0
            },
            tokens_per_sec: if uptime.as_secs() > 0 {
                total_tokens as f64 / uptime.as_secs_f64()
            } else {
                0.0
            },
            avg_latency_ms: if successful > 0 {
                (total_time_us as f64 / 1000.0) / successful as f64
            } else {
                0.0
            },
            error_rate: if total_requests > 0 {
                failed as f64 / total_requests as f64
            } else {
                0.0
            },
        }
    }

    /// Export metrics in Prometheus format
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_prometheus(&self) -> String {
        let snapshot = self.snapshot();
        format!(
            "# HELP realizar_requests_total Total number of requests\n\
             # TYPE realizar_requests_total counter\n\
             realizar_requests_total {}\n\
             # HELP realizar_requests_successful Successful requests\n\
             # TYPE realizar_requests_successful counter\n\
             realizar_requests_successful {}\n\
             # HELP realizar_requests_failed Failed requests\n\
             # TYPE realizar_requests_failed counter\n\
             realizar_requests_failed {}\n\
             # HELP realizar_tokens_generated Total tokens generated\n\
             # TYPE realizar_tokens_generated counter\n\
             realizar_tokens_generated {}\n\
             # HELP realizar_inference_time_seconds Total inference time\n\
             # TYPE realizar_inference_time_seconds counter\n\
             realizar_inference_time_seconds {:.6}\n\
             # HELP realizar_requests_per_second Request rate\n\
             # TYPE realizar_requests_per_second gauge\n\
             realizar_requests_per_second {:.2}\n\
             # HELP realizar_tokens_per_second Token generation rate\n\
             # TYPE realizar_tokens_per_second gauge\n\
             realizar_tokens_per_second {:.2}\n\
             # HELP realizar_avg_latency_ms Average latency in milliseconds\n\
             # TYPE realizar_avg_latency_ms gauge\n\
             realizar_avg_latency_ms {:.2}\n\
             # HELP realizar_error_rate Error rate (0.0-1.0)\n\
             # TYPE realizar_error_rate gauge\n\
             realizar_error_rate {:.4}\n\
             # HELP realizar_uptime_seconds Uptime in seconds\n\
             # TYPE realizar_uptime_seconds counter\n\
             realizar_uptime_seconds {}\n",
            snapshot.total_requests,
            snapshot.successful_requests,
            snapshot.failed_requests,
            snapshot.total_tokens,
            snapshot.total_inference_time_us as f64 / 1_000_000.0,
            snapshot.requests_per_sec,
            snapshot.tokens_per_sec,
            snapshot.avg_latency_ms,
            snapshot.error_rate,
            snapshot.uptime_secs
        )
    }

    /// Reset all metrics (useful for testing)
    pub fn reset(&self) {
        self.total_requests.store(0, Ordering::Relaxed);
        self.successful_requests.store(0, Ordering::Relaxed);
        self.failed_requests.store(0, Ordering::Relaxed);
        self.total_tokens.store(0, Ordering::Relaxed);
        self.total_inference_time_us.store(0, Ordering::Relaxed);
        // The latency window is part of "all metrics": leaving it behind would
        // let a reset collector report percentiles for traffic it no longer
        // counts.
        if let Ok(mut samples) = self.recent_latencies_us.lock() {
            samples.clear();
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of current metrics
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    /// Total number of requests processed
    pub total_requests: usize,
    /// Number of successful requests
    pub successful_requests: usize,
    /// Number of failed requests
    pub failed_requests: usize,
    /// Total tokens generated across all requests
    pub total_tokens: usize,
    /// Total inference time in microseconds
    pub total_inference_time_us: u64,
    /// System uptime in seconds
    pub uptime_secs: u64,
    /// Request rate (requests per second)
    pub requests_per_sec: f64,
    /// Token generation rate (tokens per second)
    pub tokens_per_sec: f64,
    /// Average request latency in milliseconds
    pub avg_latency_ms: f64,
    /// Error rate as a fraction (0.0 to 1.0)
    pub error_rate: f64,
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    #[test]
    fn test_metrics_collector_creation() {
        let metrics = MetricsCollector::new();
        let snapshot = metrics.snapshot();

        assert_eq!(snapshot.total_requests, 0);
        assert_eq!(snapshot.successful_requests, 0);
        assert_eq!(snapshot.failed_requests, 0);
        assert_eq!(snapshot.total_tokens, 0);
        assert_eq!(snapshot.total_inference_time_us, 0);
    }

    #[test]
    fn test_record_success() {
        let metrics = MetricsCollector::new();
        metrics.record_success(10, Duration::from_millis(100));

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_requests, 1);
        assert_eq!(snapshot.successful_requests, 1);
        assert_eq!(snapshot.failed_requests, 0);
        assert_eq!(snapshot.total_tokens, 10);
        assert!(snapshot.total_inference_time_us >= 100_000);
    }

    #[test]
    fn test_record_failure() {
        let metrics = MetricsCollector::new();
        metrics.record_failure();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_requests, 1);
        assert_eq!(snapshot.successful_requests, 0);
        assert_eq!(snapshot.failed_requests, 1);
        approx::assert_relative_eq!(snapshot.error_rate, 1.0);
    }

    #[test]
    fn test_multiple_requests() {
        let metrics = MetricsCollector::new();

        metrics.record_success(5, Duration::from_millis(50));
        metrics.record_success(10, Duration::from_millis(100));
        metrics.record_failure();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_requests, 3);
        assert_eq!(snapshot.successful_requests, 2);
        assert_eq!(snapshot.failed_requests, 1);
        assert_eq!(snapshot.total_tokens, 15);
        approx::assert_relative_eq!(snapshot.error_rate, 1.0 / 3.0);
    }

    #[test]
    fn test_avg_latency_calculation() {
        let metrics = MetricsCollector::new();

        // Record 100ms and 200ms requests
        metrics.record_success(1, Duration::from_millis(100));
        metrics.record_success(1, Duration::from_millis(200));

        let snapshot = metrics.snapshot();
        // Average should be 150ms
        assert!((snapshot.avg_latency_ms - 150.0).abs() < 1.0);
    }

    #[test]
    fn test_tokens_per_second() {
        let metrics = MetricsCollector::new();

        // Wait to ensure at least 1 second has passed for rate calculation
        thread::sleep(Duration::from_secs(1));

        // Record some tokens
        metrics.record_success(100, Duration::from_millis(10));

        let snapshot = metrics.snapshot();
        // Should have positive rate after 1+ seconds
        assert!(snapshot.tokens_per_sec > 0.0);
        assert!(snapshot.tokens_per_sec <= 100.0); // Can't exceed total tokens
    }

    #[test]
    fn test_prometheus_format() {
        let metrics = MetricsCollector::new();
        metrics.record_success(10, Duration::from_millis(100));
        metrics.record_failure();

        let prom = metrics.to_prometheus();

        // Check that all required metrics are present
        assert!(prom.contains("realizar_requests_total 2"));
        assert!(prom.contains("realizar_requests_successful 1"));
        assert!(prom.contains("realizar_requests_failed 1"));
        assert!(prom.contains("realizar_tokens_generated 10"));
        assert!(prom.contains("realizar_error_rate 0.5000"));
    }

    #[test]
    fn test_reset_metrics() {
        let metrics = MetricsCollector::new();
        metrics.record_success(10, Duration::from_millis(100));
        metrics.record_failure();

        metrics.reset();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_requests, 0);
        assert_eq!(snapshot.successful_requests, 0);
        assert_eq!(snapshot.failed_requests, 0);
        assert_eq!(snapshot.total_tokens, 0);
        assert_eq!(snapshot.total_inference_time_us, 0);
    }

    #[test]
    fn test_concurrent_updates() {
        let metrics = MetricsCollector::new();
        let metrics_clone = metrics.clone();

        let handle = thread::spawn(move || {
            for _ in 0..100 {
                metrics_clone.record_success(1, Duration::from_micros(100));
            }
        });

        for _ in 0..100 {
            metrics.record_success(1, Duration::from_micros(100));
        }

        handle.join().expect("test");

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_requests, 200);
        assert_eq!(snapshot.successful_requests, 200);
        assert_eq!(snapshot.total_tokens, 200);
    }

    #[test]
    fn test_zero_division_safety() {
        let metrics = MetricsCollector::new();
        let snapshot = metrics.snapshot();

        // Should not panic with zero values
        approx::assert_relative_eq!(snapshot.requests_per_sec, 0.0);
        approx::assert_relative_eq!(snapshot.tokens_per_sec, 0.0);
        approx::assert_relative_eq!(snapshot.avg_latency_ms, 0.0);
        approx::assert_relative_eq!(snapshot.error_rate, 0.0);
    }

    // -----------------------------------------------------------------------
    // aprender#2375(7): the percentiles must be order statistics of the real
    // samples — not zeros, and not multiples of the mean.
    // -----------------------------------------------------------------------

    /// The falsifier. 100 requests at 1..=100 ms have mean 50.5 ms, so the
    /// shipped formulas produce p50 = 50.5, p95 = 75.75 (`avg * 1.5`) and
    /// p99 = 101.0 (`avg * 2.0`). The measured order statistics are 50, 95 and
    /// 99 — a value `avg * k` cannot equal for both p95 and p99.
    #[test]
    fn percentiles_are_order_statistics_not_multiples_of_the_mean() {
        let metrics = MetricsCollector::new();
        for ms in 1..=100u64 {
            metrics.record_success(1, Duration::from_millis(ms));
        }

        let p = metrics
            .latency_percentiles()
            .expect("100 recorded requests must produce percentiles");

        approx::assert_relative_eq!(p.p50_ms, 50.0, epsilon = 1e-6);
        approx::assert_relative_eq!(p.p95_ms, 95.0, epsilon = 1e-6);
        approx::assert_relative_eq!(p.p99_ms, 99.0, epsilon = 1e-6);

        // Spell the shipped fabrication out so a regression names itself.
        let avg = metrics.snapshot().avg_latency_ms;
        assert!(
            (p.p95_ms - avg * 1.5).abs() > 1.0,
            "p95 must be measured, not derived as avg*1.5 ({avg} -> {})",
            avg * 1.5
        );
        assert!(
            (p.p99_ms - avg * 2.0).abs() > 1.0,
            "p99 must be measured, not derived as avg*2.0 ({avg} -> {})",
            avg * 2.0
        );
    }

    /// Order is a property of percentiles, whatever the distribution.
    #[test]
    fn percentiles_are_monotonic() {
        let metrics = MetricsCollector::new();
        for ms in [5u64, 900, 12, 7, 350, 8, 9, 11, 6, 10] {
            metrics.record_success(1, Duration::from_millis(ms));
        }
        let p = metrics.latency_percentiles().expect("samples recorded");
        assert!(
            p.p50_ms <= p.p95_ms && p.p95_ms <= p.p99_ms,
            "percentiles must be non-decreasing: {p:?}"
        );
        assert!(
            p.p99_ms >= 350.0,
            "the tail must reach the slow requests, or the window is dropping them: {p:?}"
        );
    }

    /// No traffic is reported as "no measurement", never as a measured 0 ms.
    #[test]
    fn no_samples_reports_absence_not_zero() {
        let metrics = MetricsCollector::new();
        assert!(
            metrics.latency_percentiles().is_none(),
            "a collector with no successful request has nothing to take a percentile of"
        );
        metrics.record_failure();
        assert!(
            metrics.latency_percentiles().is_none(),
            "a failed request carries no latency sample"
        );
    }

    /// The window is bounded, and keeps the RECENT samples.
    #[test]
    fn window_is_bounded_and_keeps_the_newest_samples() {
        let metrics = MetricsCollector::new();
        // Fill past capacity with slow requests, then push a full window of fast
        // ones. The slow ones must have aged out entirely.
        for _ in 0..LATENCY_WINDOW {
            metrics.record_success(1, Duration::from_millis(500));
        }
        for _ in 0..LATENCY_WINDOW {
            metrics.record_success(1, Duration::from_millis(1));
        }
        let p = metrics.latency_percentiles().expect("samples recorded");
        approx::assert_relative_eq!(p.p99_ms, 1.0, epsilon = 1e-6);

        let held = metrics
            .recent_latencies_us
            .lock()
            .expect("window lock")
            .len();
        assert_eq!(
            held, LATENCY_WINDOW,
            "the window must stay bounded regardless of uptime"
        );
    }

    #[test]
    fn reset_clears_the_latency_window() {
        let metrics = MetricsCollector::new();
        metrics.record_success(1, Duration::from_millis(42));
        assert!(metrics.latency_percentiles().is_some());
        metrics.reset();
        assert!(
            metrics.latency_percentiles().is_none(),
            "reset must drop the samples too, or percentiles describe traffic the \
             counters no longer count"
        );
    }
}
