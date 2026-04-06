//! TruenoCollector: Renacer Collector trait implementation for trueno-ublk.
//!
//! VIZ-001: Feeds metrics to renacer visualization framework.

use crate::daemon::{TieredPageStore, TieredPageStoreStats};
use anyhow::Result;
use renacer::visualize::collectors::{Collector, MetricValue, Metrics};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Collector for trueno-ublk metrics → renacer visualization.
///
/// Implements the renacer `Collector` trait to feed real-time metrics
/// to TUI dashboards, HTML reports, and OTLP exporters.
///
/// # Metrics Provided
///
/// - `throughput_gbps`: Current I/O throughput (Gauge)
/// - `iops`: Operations per second (Rate)
/// - `compression_ratio`: Original / compressed size (Gauge)
/// - `same_fill_pages`: Count of same-fill pages (Counter)
/// - `tier_kernel_zram_pct`: Percentage routed to kernel ZRAM (Gauge)
/// - `tier_simd_pct`: Percentage routed to SIMD compression (Gauge)
/// - `tier_skip_pct`: Percentage with compression skipped (Gauge)
/// - `pages_total`: Total pages processed (Counter)
pub struct TruenoCollector {
    /// Reference to tiered page store for stats
    store: Arc<TieredPageStore>,
    /// Previous stats snapshot for rate calculation
    prev_stats: Option<TieredPageStoreStats>,
    /// Previous snapshot timestamp
    prev_time: Option<Instant>,
    /// Cumulative bytes for throughput calculation
    prev_bytes: u64,
    /// Cumulative pages for IOPS calculation
    prev_pages: u64,
}

impl TruenoCollector {
    /// Create a new collector for the given tiered page store.
    pub fn new(store: Arc<TieredPageStore>) -> Self {
        Self { store, prev_stats: None, prev_time: None, prev_bytes: 0, prev_pages: 0 }
    }

    /// Calculate tier distribution percentages.
    fn tier_percentages(stats: &TieredPageStoreStats) -> (f64, f64, f64, f64) {
        let total =
            stats.kernel_pages + stats.trueno_pages + stats.skipped_pages + stats.samefill_pages;
        if total == 0 {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let total_f = total as f64;
        (
            (stats.kernel_pages as f64 / total_f) * 100.0,
            (stats.trueno_pages as f64 / total_f) * 100.0,
            (stats.skipped_pages as f64 / total_f) * 100.0,
            (stats.samefill_pages as f64 / total_f) * 100.0,
        )
    }

    /// Calculate compression ratio from stats.
    fn compression_ratio(stats: &TieredPageStoreStats) -> f64 {
        let inner = &stats.inner_stats;
        if inner.bytes_compressed == 0 {
            return 1.0;
        }
        inner.bytes_stored as f64 / inner.bytes_compressed as f64
    }
}

impl Collector for TruenoCollector {
    fn collect(&mut self) -> Result<Metrics> {
        let stats = self.store.stats();
        let now = Instant::now();

        let mut values = HashMap::new();

        // Total pages processed
        let total_pages =
            stats.kernel_pages + stats.trueno_pages + stats.skipped_pages + stats.samefill_pages;
        values.insert("pages_total".into(), MetricValue::Counter(total_pages));

        // Same-fill pages (no storage needed)
        values.insert("same_fill_pages".into(), MetricValue::Counter(stats.samefill_pages));

        // Compression ratio
        let ratio = Self::compression_ratio(&stats);
        values.insert("compression_ratio".into(), MetricValue::Gauge(ratio));

        // Tier distribution percentages
        let (kernel_pct, simd_pct, skip_pct, samefill_pct) = Self::tier_percentages(&stats);
        values.insert("tier_kernel_zram_pct".into(), MetricValue::Gauge(kernel_pct));
        values.insert("tier_simd_pct".into(), MetricValue::Gauge(simd_pct));
        values.insert("tier_skip_pct".into(), MetricValue::Gauge(skip_pct));
        values.insert("tier_samefill_pct".into(), MetricValue::Gauge(samefill_pct));

        // Batched store stats
        values.insert("pages_stored".into(), MetricValue::Counter(stats.inner_stats.pages_stored));
        values.insert(
            "pending_pages".into(),
            MetricValue::Gauge(stats.inner_stats.pending_pages as f64),
        );
        values
            .insert("batch_flushes".into(), MetricValue::Counter(stats.inner_stats.batch_flushes));

        // Rate-based metrics (need previous snapshot)
        if let (Some(prev_time), Some(_prev_stats)) = (self.prev_time, &self.prev_stats) {
            let elapsed = now.duration_since(prev_time).as_secs_f64();
            if elapsed > 0.0 {
                // IOPS: pages delta / time
                let pages_delta = total_pages.saturating_sub(self.prev_pages);
                let iops = pages_delta as f64 / elapsed;
                values.insert("iops".into(), MetricValue::Rate(iops));

                // Throughput: bytes delta / time (convert to GB/s)
                let bytes_stored = stats.inner_stats.bytes_stored;
                let bytes_delta = bytes_stored.saturating_sub(self.prev_bytes);
                let throughput_gbps = (bytes_delta as f64 / elapsed) / 1_000_000_000.0;
                values.insert("throughput_gbps".into(), MetricValue::Gauge(throughput_gbps));

                self.prev_bytes = bytes_stored;
            }
        } else {
            // First collection - no rates available yet
            values.insert("iops".into(), MetricValue::Rate(0.0));
            values.insert("throughput_gbps".into(), MetricValue::Gauge(0.0));
            self.prev_bytes = stats.inner_stats.bytes_stored;
        }

        // Update previous snapshot
        self.prev_stats = Some(stats);
        self.prev_time = Some(now);
        self.prev_pages = total_pages;

        Ok(Metrics::new(values))
    }

    fn is_available(&self) -> bool {
        // Always available as long as we have a store reference
        true
    }

    fn name(&self) -> &'static str {
        "trueno-ublk"
    }

    fn reset(&mut self) {
        self.prev_stats = None;
        self.prev_time = None;
        self.prev_bytes = 0;
        self.prev_pages = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::daemon::{BatchConfig, BatchedPageStore, BatchedPageStoreStats, TieredConfig};
    use renacer::visualize::collectors::Collector;
    use std::time::Duration;
    use trueno_zram_core::Algorithm;

    fn make_test_stats(
        kernel: u64,
        trueno: u64,
        skipped: u64,
        samefill: u64,
    ) -> TieredPageStoreStats {
        TieredPageStoreStats {
            kernel_pages: kernel,
            trueno_pages: trueno,
            skipped_pages: skipped,
            samefill_pages: samefill,
            inner_stats: BatchedPageStoreStats {
                pages_stored: kernel + trueno,
                pending_pages: 0,
                bytes_stored: (kernel + trueno) * 4096,
                bytes_compressed: (kernel + trueno) * 2048, // 2:1 ratio
                zero_pages: samefill,
                gpu_pages: 0,
                simd_pages: trueno,
                batch_flushes: 10,
            },
        }
    }

    fn create_test_store() -> Arc<TieredPageStore> {
        let inner = Arc::new(BatchedPageStore::with_config(
            Algorithm::Lz4,
            BatchConfig {
                batch_threshold: 100,
                flush_timeout: Duration::from_millis(10),
                gpu_batch_size: 1000,
            },
        ));
        Arc::new(TieredPageStore::new(inner, TieredConfig::default()).unwrap())
    }

    #[test]
    fn test_tier_percentages_empty() {
        let stats = make_test_stats(0, 0, 0, 0);
        let (kernel, simd, skip, samefill) = TruenoCollector::tier_percentages(&stats);
        assert_eq!(kernel, 0.0);
        assert_eq!(simd, 0.0);
        assert_eq!(skip, 0.0);
        assert_eq!(samefill, 0.0);
    }

    #[test]
    fn test_tier_percentages_distribution() {
        let stats = make_test_stats(50, 30, 10, 10);
        let (kernel, simd, skip, samefill) = TruenoCollector::tier_percentages(&stats);
        assert!((kernel - 50.0).abs() < 0.01);
        assert!((simd - 30.0).abs() < 0.01);
        assert!((skip - 10.0).abs() < 0.01);
        assert!((samefill - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_compression_ratio() {
        let stats = make_test_stats(100, 100, 0, 0);
        let ratio = TruenoCollector::compression_ratio(&stats);
        // bytes_stored = 200 * 4096, bytes_compressed = 200 * 2048 = 2:1 ratio
        assert!((ratio - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_compression_ratio_zero() {
        let mut stats = make_test_stats(0, 0, 0, 0);
        stats.inner_stats.bytes_compressed = 0;
        let ratio = TruenoCollector::compression_ratio(&stats);
        assert_eq!(ratio, 1.0); // Default to 1:1 when no data
    }

    #[test]
    fn test_collector_new() {
        let store = create_test_store();
        let collector = TruenoCollector::new(store);
        assert!(collector.prev_stats.is_none());
        assert!(collector.prev_time.is_none());
        assert_eq!(collector.prev_bytes, 0);
        assert_eq!(collector.prev_pages, 0);
    }

    #[test]
    fn test_collector_name() {
        let store = create_test_store();
        let collector = TruenoCollector::new(store);
        assert_eq!(collector.name(), "trueno-ublk");
    }

    #[test]
    fn test_collector_is_available() {
        let store = create_test_store();
        let collector = TruenoCollector::new(store);
        assert!(collector.is_available());
    }

    #[test]
    fn test_collector_reset() {
        let store = create_test_store();
        let mut collector = TruenoCollector::new(store);

        // Simulate some state
        collector.prev_bytes = 1000;
        collector.prev_pages = 100;
        collector.prev_time = Some(Instant::now());

        // Reset should clear all state
        collector.reset();
        assert!(collector.prev_stats.is_none());
        assert!(collector.prev_time.is_none());
        assert_eq!(collector.prev_bytes, 0);
        assert_eq!(collector.prev_pages, 0);
    }

    #[test]
    fn test_collector_collect_first_call() {
        let store = create_test_store();
        let mut collector = TruenoCollector::new(store);

        let metrics = collector.collect().unwrap();

        // First call should have zero rates
        assert!(metrics.values.get("iops").is_some());
        assert!(metrics.values.get("throughput_gbps").is_some());
        assert!(metrics.values.get("pages_total").is_some());
        assert!(metrics.values.get("compression_ratio").is_some());
        assert!(metrics.values.get("tier_kernel_zram_pct").is_some());
        assert!(metrics.values.get("tier_simd_pct").is_some());
        assert!(metrics.values.get("tier_skip_pct").is_some());
        assert!(metrics.values.get("tier_samefill_pct").is_some());
    }

    #[test]
    fn test_collector_collect_second_call() {
        let store = create_test_store();
        let mut collector = TruenoCollector::new(store);

        // First collection
        let _ = collector.collect().unwrap();

        // Small delay to get measurable elapsed time
        std::thread::sleep(Duration::from_millis(10));

        // Second collection should have rate calculations
        let metrics = collector.collect().unwrap();

        // Verify all metrics are present
        assert!(metrics.values.get("iops").is_some());
        assert!(metrics.values.get("throughput_gbps").is_some());
        assert!(metrics.values.get("pages_stored").is_some());
        assert!(metrics.values.get("pending_pages").is_some());
        assert!(metrics.values.get("batch_flushes").is_some());
    }

    #[test]
    fn test_collector_with_data() {
        let store = create_test_store();

        // Write some pages to the store
        let page = [0xAA_u8; 4096];
        for i in 0..10 {
            store.store(i, &page).unwrap();
        }

        let mut collector = TruenoCollector::new(store);
        let metrics = collector.collect().unwrap();

        // Should have pages_total > 0
        if let Some(MetricValue::Counter(count)) = metrics.values.get("pages_total") {
            assert!(count >= &10, "Expected at least 10 pages, got {}", count);
        }
    }

    #[test]
    fn test_tier_percentages_all_kernel() {
        let stats = make_test_stats(100, 0, 0, 0);
        let (kernel, simd, skip, samefill) = TruenoCollector::tier_percentages(&stats);
        assert!((kernel - 100.0).abs() < 0.01);
        assert_eq!(simd, 0.0);
        assert_eq!(skip, 0.0);
        assert_eq!(samefill, 0.0);
    }

    #[test]
    fn test_tier_percentages_all_samefill() {
        let stats = make_test_stats(0, 0, 0, 100);
        let (kernel, simd, skip, samefill) = TruenoCollector::tier_percentages(&stats);
        assert_eq!(kernel, 0.0);
        assert_eq!(simd, 0.0);
        assert_eq!(skip, 0.0);
        assert!((samefill - 100.0).abs() < 0.01);
    }
}
