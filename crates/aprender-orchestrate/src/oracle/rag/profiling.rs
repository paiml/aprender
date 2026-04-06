//! RAG Profiling
//!
//! Tracing spans and histogram metrics for RAG query performance.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Histogram bucket for latency measurements
#[derive(Debug, Clone, Copy)]
pub struct HistogramBucket {
    /// Upper bound in milliseconds
    pub le: f64,
    /// Count of observations
    pub count: u64,
}

/// A simple histogram for latency measurements
#[derive(Debug)]
pub struct Histogram {
    /// Bucket boundaries in milliseconds
    buckets: Vec<f64>,
    /// Counts per bucket
    counts: Vec<AtomicU64>,
    /// Sum of all observations
    sum: AtomicU64,
    /// Total count
    total: AtomicU64,
}

impl Histogram {
    /// Create a new histogram with default latency buckets (in ms)
    pub fn new() -> Self {
        // Standard latency buckets: 1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s
        let buckets =
            vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0];
        let counts = buckets.iter().map(|_| AtomicU64::new(0)).collect();

        Self { buckets, counts, sum: AtomicU64::new(0), total: AtomicU64::new(0) }
    }

    /// Create a histogram with custom buckets
    pub fn with_buckets(buckets: Vec<f64>) -> Self {
        let counts = buckets.iter().map(|_| AtomicU64::new(0)).collect();

        Self { buckets, counts, sum: AtomicU64::new(0), total: AtomicU64::new(0) }
    }

    /// Observe a duration
    pub fn observe(&self, duration: Duration) {
        let ms = duration.as_secs_f64() * 1000.0;

        // Update sum (storing as microseconds for precision)
        let us = (ms * 1000.0) as u64;
        self.sum.fetch_add(us, Ordering::Relaxed);
        self.total.fetch_add(1, Ordering::Relaxed);

        // Update bucket counts
        for (i, &le) in self.buckets.iter().enumerate() {
            if ms <= le {
                self.counts[i].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Get the current bucket counts
    pub fn get_buckets(&self) -> Vec<HistogramBucket> {
        self.buckets
            .iter()
            .zip(self.counts.iter())
            .map(|(&le, count)| HistogramBucket { le, count: count.load(Ordering::Relaxed) })
            .collect()
    }

    /// Get the total count
    pub fn count(&self) -> u64 {
        self.total.load(Ordering::Relaxed)
    }

    /// Get the sum in milliseconds
    pub fn sum_ms(&self) -> f64 {
        let us = self.sum.load(Ordering::Relaxed);
        us as f64 / 1000.0
    }

    /// Calculate approximate percentile (p50, p90, p99, etc.)
    pub fn percentile(&self, p: f64) -> f64 {
        let total = self.count();
        if total == 0 {
            return 0.0;
        }

        let target = (total as f64 * p / 100.0).ceil() as u64;
        let buckets = self.get_buckets();

        for bucket in &buckets {
            if bucket.count >= target {
                return bucket.le;
            }
        }

        // Return the largest bucket boundary
        self.buckets.last().copied().unwrap_or(0.0)
    }

    /// Get p50 latency
    pub fn p50(&self) -> f64 {
        self.percentile(50.0)
    }

    /// Get p90 latency
    pub fn p90(&self) -> f64 {
        self.percentile(90.0)
    }

    /// Get p99 latency
    pub fn p99(&self) -> f64 {
        self.percentile(99.0)
    }

    /// Get mean latency in milliseconds
    pub fn mean(&self) -> f64 {
        let count = self.count();
        if count == 0 {
            return 0.0;
        }
        self.sum_ms() / count as f64
    }

    /// Reset all counters
    pub fn reset(&self) {
        self.sum.store(0, Ordering::Relaxed);
        self.total.store(0, Ordering::Relaxed);
        for count in &self.counts {
            count.store(0, Ordering::Relaxed);
        }
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

/// A simple counter metric
#[derive(Debug, Default)]
pub struct Counter {
    value: AtomicU64,
}

impl Counter {
    /// Create a new counter
    pub fn new() -> Self {
        Self { value: AtomicU64::new(0) }
    }

    /// Increment by 1
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment by a specific amount
    pub fn inc_by(&self, n: u64) {
        self.value.fetch_add(n, Ordering::Relaxed);
    }

    /// Get the current value
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Reset the counter
    pub fn reset(&self) {
        self.value.store(0, Ordering::Relaxed);
    }
}

/// RAG metrics collector
#[derive(Debug)]
pub struct RagMetrics {
    /// Query latency histogram
    pub query_latency: Histogram,
    /// Index load latency histogram
    pub index_load_latency: Histogram,
    /// Cache hit counter
    pub cache_hits: Counter,
    /// Cache miss counter
    pub cache_misses: Counter,
    /// Total queries counter
    pub total_queries: Counter,
    /// Documents retrieved counter
    pub docs_retrieved: Counter,
    /// Custom spans
    spans: Mutex<HashMap<String, SpanStats>>,
}

/// Statistics for a named span
#[derive(Debug, Clone, Default)]
pub struct SpanStats {
    /// Number of invocations
    pub count: u64,
    /// Total duration in microseconds
    pub total_us: u64,
    /// Min duration in microseconds
    pub min_us: u64,
    /// Max duration in microseconds
    pub max_us: u64,
}

impl RagMetrics {
    /// Create new metrics collector
    pub fn new() -> Self {
        Self {
            query_latency: Histogram::new(),
            index_load_latency: Histogram::new(),
            cache_hits: Counter::new(),
            cache_misses: Counter::new(),
            total_queries: Counter::new(),
            docs_retrieved: Counter::new(),
            spans: Mutex::new(HashMap::new()),
        }
    }

    /// Record a span's duration
    pub fn record_span(&self, name: &str, duration: Duration) {
        let us = duration.as_micros() as u64;

        let mut spans = self.spans.lock().unwrap_or_else(|e| e.into_inner());
        let stats = spans.entry(name.to_string()).or_default();

        stats.count += 1;
        stats.total_us += us;

        if stats.min_us == 0 || us < stats.min_us {
            stats.min_us = us;
        }
        if us > stats.max_us {
            stats.max_us = us;
        }
    }

    /// Get span statistics
    pub fn get_span_stats(&self, name: &str) -> Option<SpanStats> {
        let spans = self.spans.lock().unwrap_or_else(|e| e.into_inner());
        spans.get(name).cloned()
    }

    /// Get all span statistics
    pub fn all_span_stats(&self) -> HashMap<String, SpanStats> {
        let spans = self.spans.lock().unwrap_or_else(|e| e.into_inner());
        spans.clone()
    }

    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.get();
        let misses = self.cache_misses.get();
        let total = hits + misses;
        if total == 0 {
            return 0.0;
        }
        hits as f64 / total as f64
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.query_latency.reset();
        self.index_load_latency.reset();
        self.cache_hits.reset();
        self.cache_misses.reset();
        self.total_queries.reset();
        self.docs_retrieved.reset();
        self.spans.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /// Generate a summary report
    pub fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            total_queries: self.total_queries.get(),
            query_latency_p50_ms: self.query_latency.p50(),
            query_latency_p90_ms: self.query_latency.p90(),
            query_latency_p99_ms: self.query_latency.p99(),
            query_latency_mean_ms: self.query_latency.mean(),
            cache_hit_rate: self.cache_hit_rate(),
            cache_hits: self.cache_hits.get(),
            cache_misses: self.cache_misses.get(),
            docs_retrieved: self.docs_retrieved.get(),
            spans: self.all_span_stats(),
        }
    }
}

impl Default for RagMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of RAG metrics
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    /// Total queries executed
    pub total_queries: u64,
    /// Query latency p50 in milliseconds
    pub query_latency_p50_ms: f64,
    /// Query latency p90 in milliseconds
    pub query_latency_p90_ms: f64,
    /// Query latency p99 in milliseconds
    pub query_latency_p99_ms: f64,
    /// Query latency mean in milliseconds
    pub query_latency_mean_ms: f64,
    /// Cache hit rate (0.0 - 1.0)
    pub cache_hit_rate: f64,
    /// Total cache hits
    pub cache_hits: u64,
    /// Total cache misses
    pub cache_misses: u64,
    /// Total documents retrieved
    pub docs_retrieved: u64,
    /// Span statistics
    pub spans: HashMap<String, SpanStats>,
}

impl std::fmt::Display for MetricsSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "RAG Metrics Summary")?;
        writeln!(f, "===================")?;
        writeln!(f, "Total Queries: {}", self.total_queries)?;
        writeln!(f)?;
        writeln!(f, "Query Latency:")?;
        writeln!(f, "  p50:  {:.2}ms", self.query_latency_p50_ms)?;
        writeln!(f, "  p90:  {:.2}ms", self.query_latency_p90_ms)?;
        writeln!(f, "  p99:  {:.2}ms", self.query_latency_p99_ms)?;
        writeln!(f, "  mean: {:.2}ms", self.query_latency_mean_ms)?;
        writeln!(f)?;
        writeln!(f, "Cache:")?;
        writeln!(f, "  Hit Rate: {:.1}%", self.cache_hit_rate * 100.0)?;
        writeln!(f, "  Hits:     {}", self.cache_hits)?;
        writeln!(f, "  Misses:   {}", self.cache_misses)?;
        writeln!(f)?;
        writeln!(f, "Documents Retrieved: {}", self.docs_retrieved)?;

        if !self.spans.is_empty() {
            writeln!(f)?;
            writeln!(f, "Spans:")?;
            for (name, stats) in &self.spans {
                let avg_us = if stats.count > 0 { stats.total_us / stats.count } else { 0 };
                writeln!(
                    f,
                    "  {}: count={}, avg={:.2}ms, min={:.2}ms, max={:.2}ms",
                    name,
                    stats.count,
                    avg_us as f64 / 1000.0,
                    stats.min_us as f64 / 1000.0,
                    stats.max_us as f64 / 1000.0
                )?;
            }
        }

        Ok(())
    }
}

/// A timed span that records duration on drop
pub struct TimedSpan<'a> {
    name: String,
    start: Instant,
    metrics: &'a RagMetrics,
}

impl<'a> TimedSpan<'a> {
    /// Create a new timed span
    pub fn new(name: &str, metrics: &'a RagMetrics) -> Self {
        Self { name: name.to_string(), start: crate::timing::start_timer(), metrics }
    }

    /// Get elapsed time without finishing
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

impl Drop for TimedSpan<'_> {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        self.metrics.record_span(&self.name, duration);
    }
}

/// Global metrics instance (thread-safe)
pub static GLOBAL_METRICS: std::sync::LazyLock<RagMetrics> =
    std::sync::LazyLock::new(RagMetrics::new);

/// Start a timed span using global metrics
pub fn span(name: &str) -> TimedSpan<'static> {
    TimedSpan::new(name, &GLOBAL_METRICS)
}

/// Record a query latency using global metrics
pub fn record_query_latency(duration: Duration) {
    GLOBAL_METRICS.query_latency.observe(duration);
    GLOBAL_METRICS.total_queries.inc();
}

/// Record a cache hit using global metrics
pub fn record_cache_hit() {
    GLOBAL_METRICS.cache_hits.inc();
}

/// Record a cache miss using global metrics
pub fn record_cache_miss() {
    GLOBAL_METRICS.cache_misses.inc();
}

/// Get global metrics summary
pub fn get_summary() -> MetricsSummary {
    GLOBAL_METRICS.summary()
}

/// Reset global metrics
pub fn reset_metrics() {
    GLOBAL_METRICS.reset();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_creation() {
        let hist = Histogram::new();
        assert_eq!(hist.count(), 0);
        assert_eq!(hist.sum_ms(), 0.0);
    }

    #[test]
    fn test_histogram_observe() {
        let hist = Histogram::new();
        hist.observe(Duration::from_millis(5));
        hist.observe(Duration::from_millis(10));
        hist.observe(Duration::from_millis(50));

        assert_eq!(hist.count(), 3);
        // Sum should be approximately 65ms
        assert!((hist.sum_ms() - 65.0).abs() < 1.0);
    }

    #[test]
    fn test_histogram_percentiles() {
        let hist = Histogram::new();

        // Add 100 observations spread across buckets
        for i in 1..=100 {
            hist.observe(Duration::from_millis(i));
        }

        // p50 should be around 50ms bucket
        let p50 = hist.p50();
        assert!(p50 >= 50.0, "p50 should be >= 50ms, got {}", p50);

        // p99 should be higher
        let p99 = hist.p99();
        assert!(p99 >= p50, "p99 should be >= p50");
    }

    #[test]
    fn test_histogram_mean() {
        let hist = Histogram::new();
        hist.observe(Duration::from_millis(10));
        hist.observe(Duration::from_millis(20));
        hist.observe(Duration::from_millis(30));

        let mean = hist.mean();
        assert!((mean - 20.0).abs() < 1.0, "mean should be ~20ms, got {}", mean);
    }

    #[test]
    fn test_histogram_reset() {
        let hist = Histogram::new();
        hist.observe(Duration::from_millis(10));
        assert_eq!(hist.count(), 1);

        hist.reset();
        assert_eq!(hist.count(), 0);
        assert_eq!(hist.sum_ms(), 0.0);
    }

    #[test]
    fn test_counter_basic() {
        let counter = Counter::new();
        assert_eq!(counter.get(), 0);

        counter.inc();
        assert_eq!(counter.get(), 1);

        counter.inc_by(5);
        assert_eq!(counter.get(), 6);
    }

    #[test]
    fn test_counter_reset() {
        let counter = Counter::new();
        counter.inc_by(100);
        assert_eq!(counter.get(), 100);

        counter.reset();
        assert_eq!(counter.get(), 0);
    }

    #[test]
    fn test_rag_metrics_creation() {
        let metrics = RagMetrics::new();
        assert_eq!(metrics.total_queries.get(), 0);
        assert_eq!(metrics.cache_hits.get(), 0);
    }

    #[test]
    fn test_rag_metrics_record_span() {
        let metrics = RagMetrics::new();

        metrics.record_span("test_span", Duration::from_millis(10));
        metrics.record_span("test_span", Duration::from_millis(20));

        let stats = metrics.get_span_stats("test_span").expect("unexpected failure");
        assert_eq!(stats.count, 2);
        assert_eq!(stats.total_us, 30_000);
        assert_eq!(stats.min_us, 10_000);
        assert_eq!(stats.max_us, 20_000);
    }

    #[test]
    fn test_rag_metrics_cache_hit_rate() {
        let metrics = RagMetrics::new();

        // No hits or misses
        assert_eq!(metrics.cache_hit_rate(), 0.0);

        // 3 hits, 2 misses = 60% hit rate
        metrics.cache_hits.inc_by(3);
        metrics.cache_misses.inc_by(2);
        assert!((metrics.cache_hit_rate() - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_rag_metrics_summary() {
        let metrics = RagMetrics::new();

        metrics.total_queries.inc_by(100);
        metrics.cache_hits.inc_by(80);
        metrics.cache_misses.inc_by(20);
        metrics.docs_retrieved.inc_by(500);

        // Add some query latencies
        for _ in 0..50 {
            metrics.query_latency.observe(Duration::from_millis(15));
        }
        for _ in 0..50 {
            metrics.query_latency.observe(Duration::from_millis(25));
        }

        let summary = metrics.summary();
        assert_eq!(summary.total_queries, 100);
        assert_eq!(summary.cache_hits, 80);
        assert_eq!(summary.cache_misses, 20);
        assert!((summary.cache_hit_rate - 0.8).abs() < 0.001);
        assert_eq!(summary.docs_retrieved, 500);
    }

    #[test]
    fn test_rag_metrics_reset() {
        let metrics = RagMetrics::new();

        metrics.total_queries.inc_by(100);
        metrics.cache_hits.inc_by(50);
        metrics.record_span("span1", Duration::from_millis(10));

        metrics.reset();

        assert_eq!(metrics.total_queries.get(), 0);
        assert_eq!(metrics.cache_hits.get(), 0);
        assert!(metrics.all_span_stats().is_empty());
    }

    #[test]
    fn test_timed_span() {
        let metrics = RagMetrics::new();

        {
            let _span = TimedSpan::new("test", &metrics);
            std::thread::sleep(Duration::from_millis(5));
        }

        let stats = metrics.get_span_stats("test").expect("unexpected failure");
        assert_eq!(stats.count, 1);
        assert!(stats.total_us >= 5_000, "should be at least 5ms");
    }

    #[test]
    fn test_metrics_summary_display() {
        let metrics = RagMetrics::new();
        metrics.total_queries.inc_by(10);
        metrics.cache_hits.inc_by(8);
        metrics.cache_misses.inc_by(2);

        let summary = metrics.summary();
        let display = format!("{}", summary);

        assert!(display.contains("RAG Metrics Summary"));
        assert!(display.contains("Total Queries: 10"));
        assert!(display.contains("Hit Rate: 80.0%"));
    }

    #[test]
    fn test_histogram_custom_buckets() {
        let hist = Histogram::with_buckets(vec![1.0, 10.0, 100.0]);
        hist.observe(Duration::from_millis(5));

        let buckets = hist.get_buckets();
        assert_eq!(buckets.len(), 3);
        assert_eq!(buckets[0].le, 1.0);
        assert_eq!(buckets[1].le, 10.0);
        assert_eq!(buckets[2].le, 100.0);
    }

    #[test]
    fn test_global_metrics() {
        // Reset first to ensure clean state
        reset_metrics();

        record_cache_hit();
        record_cache_hit();
        record_cache_miss();

        let summary = get_summary();
        assert_eq!(summary.cache_hits, 2);
        assert_eq!(summary.cache_misses, 1);

        reset_metrics();
        let summary = get_summary();
        assert_eq!(summary.cache_hits, 0);
    }

    #[test]
    fn test_span_helper() {
        reset_metrics();

        {
            let _s = span("helper_test");
            std::thread::sleep(Duration::from_millis(1));
        }

        let stats = GLOBAL_METRICS.get_span_stats("helper_test");
        assert!(stats.is_some());
        assert_eq!(stats.expect("unexpected failure").count, 1);

        reset_metrics();
    }

    #[test]
    fn test_histogram_percentile() {
        let hist = Histogram::new();

        // Add values so we know the distribution
        for _ in 0..10 {
            hist.observe(Duration::from_millis(5));
        }
        for _ in 0..90 {
            hist.observe(Duration::from_millis(50));
        }

        // p10 should be in the low bucket
        let p10 = hist.percentile(0.10);
        assert!(p10 <= 10.0, "p10 should be <= 10ms, got {}", p10);
    }

    #[test]
    fn test_histogram_p90() {
        let hist = Histogram::new();
        for i in 1..=100 {
            hist.observe(Duration::from_millis(i));
        }

        let p90 = hist.p90();
        assert!(p90 >= 90.0, "p90 should be >= 90ms, got {}", p90);
    }

    #[test]
    fn test_timed_span_elapsed() {
        let metrics = RagMetrics::new();
        let span = TimedSpan::new("elapsed_test", &metrics);
        let elapsed = span.elapsed();
        // Elapsed is always non-negative; avoid wall-clock assertions
        assert!(elapsed >= Duration::ZERO);
    }

    #[test]
    fn test_record_query_latency() {
        reset_metrics();
        record_query_latency(Duration::from_millis(10));
        record_query_latency(Duration::from_millis(20));

        let summary = get_summary();
        assert_eq!(summary.total_queries, 2);
        assert!(summary.query_latency_p50_ms >= 10.0);

        reset_metrics();
    }

    #[test]
    fn test_histogram_default() {
        let hist = Histogram::default();
        assert_eq!(hist.count(), 0);
    }

    #[test]
    fn test_counter_default() {
        let counter = Counter::default();
        assert_eq!(counter.get(), 0);
    }

    #[test]
    fn test_span_stats_default() {
        let stats = SpanStats::default();
        assert_eq!(stats.count, 0);
        assert_eq!(stats.total_us, 0);
        // Default uses 0 for all fields
        assert_eq!(stats.min_us, 0);
        assert_eq!(stats.max_us, 0);
    }

    #[test]
    fn test_metrics_summary_fields() {
        let metrics = RagMetrics::new();
        let summary = metrics.summary();
        assert_eq!(summary.total_queries, 0);
        assert_eq!(summary.cache_hits, 0);
        assert_eq!(summary.query_latency_p50_ms, 0.0);
        assert_eq!(summary.query_latency_p99_ms, 0.0);
    }

    #[test]
    fn test_all_span_stats() {
        let metrics = RagMetrics::new();
        metrics.record_span("span_a", Duration::from_millis(10));
        metrics.record_span("span_b", Duration::from_millis(20));

        let all = metrics.all_span_stats();
        assert!(all.contains_key("span_a"));
        assert!(all.contains_key("span_b"));
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_histogram_empty_percentile() {
        let hist = Histogram::new();
        // Empty histogram should return 0
        assert_eq!(hist.p50(), 0.0);
    }

    #[test]
    fn test_histogram_empty_mean() {
        let hist = Histogram::new();
        // Empty histogram should return 0
        assert_eq!(hist.mean(), 0.0);
    }

    #[test]
    fn test_histogram_bucket_fields() {
        let bucket = HistogramBucket { le: 100.0, count: 42 };
        assert_eq!(bucket.le, 100.0);
        assert_eq!(bucket.count, 42);
    }

    #[test]
    fn test_histogram_bucket_copy() {
        let bucket = HistogramBucket { le: 50.0, count: 10 };
        let copied = bucket;
        assert_eq!(copied.le, bucket.le);
        assert_eq!(copied.count, bucket.count);
    }

    #[test]
    fn test_span_stats_clone() {
        let stats = SpanStats { count: 5, total_us: 5000, min_us: 100, max_us: 2000 };
        let cloned = stats.clone();
        assert_eq!(cloned.count, 5);
        assert_eq!(cloned.total_us, 5000);
    }

    #[test]
    fn test_get_span_stats_none() {
        let metrics = RagMetrics::new();
        assert!(metrics.get_span_stats("nonexistent").is_none());
    }

    #[test]
    fn test_metrics_summary_display_with_spans() {
        let metrics = RagMetrics::new();
        metrics.record_span("tokenize", Duration::from_millis(10));
        metrics.record_span("retrieve", Duration::from_millis(50));

        let summary = metrics.summary();
        let display = format!("{}", summary);

        assert!(display.contains("Spans:"));
        assert!(display.contains("tokenize"));
        assert!(display.contains("retrieve"));
    }

    #[test]
    fn test_metrics_summary_clone() {
        let metrics = RagMetrics::new();
        metrics.total_queries.inc_by(5);
        let summary = metrics.summary();
        let cloned = summary.clone();
        assert_eq!(cloned.total_queries, 5);
    }

    #[test]
    fn test_histogram_percentile_returns_first_matching_bucket() {
        let hist = Histogram::with_buckets(vec![1.0, 2.0, 3.0]);
        // Add observation that fits in first bucket (0.5ms <= 1.0)
        hist.observe(Duration::from_micros(500)); // 0.5ms

        // p50 with 1 observation: target = ceil(1 * 50/100) = 1
        // First bucket with count >= 1 is bucket 1.0
        let p50 = hist.percentile(50.0);
        assert_eq!(p50, 1.0);
    }

    #[test]
    fn test_rag_metrics_default() {
        let metrics = RagMetrics::default();
        assert_eq!(metrics.total_queries.get(), 0);
    }

    #[test]
    fn test_histogram_observe_large_values() {
        let hist = Histogram::new();
        hist.observe(Duration::from_secs(15)); // 15000ms, beyond last bucket

        assert_eq!(hist.count(), 1);
        // p99 for single large value should be the largest bucket
        let p99 = hist.p99();
        assert_eq!(p99, 10000.0); // Last bucket is 10s
    }

    #[test]
    fn test_histogram_debug() {
        let hist = Histogram::new();
        let debug = format!("{:?}", hist);
        assert!(debug.contains("Histogram"));
    }

    #[test]
    fn test_counter_debug() {
        let counter = Counter::new();
        let debug = format!("{:?}", counter);
        assert!(debug.contains("Counter"));
    }

    #[test]
    fn test_rag_metrics_debug() {
        let metrics = RagMetrics::new();
        let debug = format!("{:?}", metrics);
        assert!(debug.contains("RagMetrics"));
    }

    #[test]
    fn test_span_stats_debug() {
        let stats = SpanStats::default();
        let debug = format!("{:?}", stats);
        assert!(debug.contains("SpanStats"));
    }

    #[test]
    fn test_metrics_summary_debug() {
        let metrics = RagMetrics::new();
        let summary = metrics.summary();
        let debug = format!("{:?}", summary);
        assert!(debug.contains("MetricsSummary"));
    }
}
