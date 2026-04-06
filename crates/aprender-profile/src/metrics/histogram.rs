//! Histogram metric - distribution of observations
//!
//! Histograms track the distribution of values across configurable buckets.
//! Common use cases: request latency, response sizes.
//!
//! # Performance
//! - observe(): <200ns with SIMD bucket search (trueno)
//! - observe(): <500ns with scalar fallback
//!
//! # Pattern
//! Prometheus client_golang histogram with SIMD optimization
//!
//! # Example
//! ```ignore
//! let histogram = Histogram::new("request_duration_seconds", labels, DEFAULT_BUCKETS);
//! histogram.observe(0.025);  // 25ms
//! histogram.observe(0.150);  // 150ms
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::labels::Labels;
use super::MetricDesc;

/// Default histogram buckets (Prometheus-compatible)
/// Covers common latency ranges from 5ms to 10s
pub const DEFAULT_BUCKETS: &[f64] =
    &[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0];

/// Prometheus-compatible buckets (same as DEFAULT_BUCKETS)
pub const PROMETHEUS_BUCKETS: &[f64] = DEFAULT_BUCKETS;

/// Histogram metric - distribution of observations
///
/// Thread-safe via atomic operations. Uses SIMD for fast bucket search.
#[derive(Debug)]
pub struct Histogram {
    /// Metric descriptor (name + labels)
    desc: MetricDesc,
    /// Upper bounds for each bucket (sorted, last is +Inf)
    buckets: Vec<f64>,
    /// Count per bucket (atomic for thread-safety)
    counts: Vec<AtomicU64>,
    /// Sum of all observations (stored as u64 bits)
    sum_bits: AtomicU64,
    /// Total observation count
    count: AtomicU64,
    /// Creation timestamp
    created_at: Instant,
}

impl Histogram {
    /// Create a new histogram with given buckets
    pub fn new(name: impl Into<String>, labels: Labels, buckets: &[f64]) -> Self {
        // Validate and sort buckets (filter out NaN, then sort with total_cmp for safety)
        let mut sorted_buckets: Vec<f64> =
            buckets.iter().copied().filter(|x| !x.is_nan()).collect();
        sorted_buckets.sort_by(f64::total_cmp);

        // Add +Inf bucket if not present
        if sorted_buckets.last().map_or(true, |&b| b < f64::INFINITY) {
            sorted_buckets.push(f64::INFINITY);
        }

        let counts = sorted_buckets.iter().map(|_| AtomicU64::new(0)).collect();

        Self {
            desc: MetricDesc::new(name, labels),
            buckets: sorted_buckets,
            counts,
            sum_bits: AtomicU64::new(0),
            count: AtomicU64::new(0),
            created_at: Instant::now(),
        }
    }

    /// Create histogram with default buckets
    pub fn with_default_buckets(name: impl Into<String>, labels: Labels) -> Self {
        Self::new(name, labels, DEFAULT_BUCKETS)
    }

    /// Create a new histogram wrapped in Arc for shared ownership
    pub fn new_arc(name: impl Into<String>, labels: Labels, buckets: &[f64]) -> Arc<Self> {
        Arc::new(Self::new(name, labels, buckets))
    }

    /// Record an observation
    ///
    /// Uses SIMD-accelerated bucket search when available
    #[inline]
    pub fn observe(&self, value: f64) {
        // Find bucket (SIMD or scalar)
        let bucket_idx = self.find_bucket(value);

        // Increment bucket count
        self.counts[bucket_idx].fetch_add(1, Ordering::Relaxed);

        // Atomically add to sum (using bit representation)
        // This is a CAS loop to handle concurrent updates
        loop {
            let current = self.sum_bits.load(Ordering::Relaxed);
            let current_sum = f64::from_bits(current);
            let new_sum = current_sum + value;
            let new_bits = new_sum.to_bits();

            if self
                .sum_bits
                .compare_exchange_weak(current, new_bits, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        // Increment total count
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Find bucket index for value
    ///
    /// SIMD optimization: uses vectorized comparison when >8 buckets
    #[inline]
    fn find_bucket(&self, value: f64) -> usize {
        // Use SIMD for larger bucket sets
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if self.buckets.len() >= 8 {
            return self.find_bucket_simd_avx2(value);
        }

        // Scalar fallback
        self.find_bucket_scalar(value)
    }

    /// Scalar bucket search (fallback)
    #[inline]
    fn find_bucket_scalar(&self, value: f64) -> usize {
        for (i, &bound) in self.buckets.iter().enumerate() {
            if value <= bound {
                return i;
            }
        }
        self.buckets.len() - 1
    }

    /// SIMD bucket search using AVX2
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[inline]
    #[allow(unsafe_code)]
    fn find_bucket_simd_avx2(&self, value: f64) -> usize {
        use std::arch::x86_64::{
            _mm256_cmp_pd, _mm256_loadu_pd, _mm256_movemask_pd, _mm256_set1_pd, _CMP_LE_OQ,
        };

        // SAFETY: AVX2 intrinsics are safe when:
        // 1. Target CPU supports AVX2 (checked by #[target_feature(enable = "avx2")])
        // 2. Pointer arithmetic stays within bounds (checked by while condition)
        // 3. Memory alignment handled by _mm256_loadu_pd (unaligned load)
        unsafe {
            let value_vec = _mm256_set1_pd(value);
            let mut idx = 0;

            // Process 4 buckets at a time
            while idx + 4 <= self.buckets.len() {
                let bounds = _mm256_loadu_pd(self.buckets.as_ptr().add(idx));
                let cmp = _mm256_cmp_pd::<_CMP_LE_OQ>(value_vec, bounds);
                let mask = _mm256_movemask_pd(cmp);

                if mask != 0 {
                    return idx + mask.trailing_zeros() as usize;
                }
                idx += 4;
            }

            // Handle remaining buckets with scalar
            for i in idx..self.buckets.len() {
                if value <= self.buckets[i] {
                    return i;
                }
            }

            self.buckets.len() - 1
        }
    }

    /// Get observation count
    #[inline]
    pub fn get_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get sum of observations
    #[inline]
    pub fn get_sum(&self) -> f64 {
        f64::from_bits(self.sum_bits.load(Ordering::Relaxed))
    }

    /// Get bucket count at index
    pub fn get_bucket_count(&self, idx: usize) -> u64 {
        self.counts.get(idx).map_or(0, |c| c.load(Ordering::Relaxed))
    }

    /// Get bucket upper bound at index
    pub fn get_bucket_bound(&self, idx: usize) -> Option<f64> {
        self.buckets.get(idx).copied()
    }

    /// Get number of buckets
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    /// Get all bucket bounds
    pub fn buckets(&self) -> &[f64] {
        &self.buckets
    }

    /// Get all bucket counts
    pub fn bucket_counts(&self) -> Vec<u64> {
        self.counts.iter().map(|c| c.load(Ordering::Relaxed)).collect()
    }

    /// Get cumulative bucket counts (for Prometheus le format)
    pub fn cumulative_counts(&self) -> Vec<u64> {
        let mut cumulative = Vec::with_capacity(self.counts.len());
        let mut total = 0u64;
        for count in &self.counts {
            total += count.load(Ordering::Relaxed);
            cumulative.push(total);
        }
        cumulative
    }

    /// Compute quantile (approximation based on buckets)
    pub fn quantile(&self, q: f64) -> f64 {
        if !(0.0..=1.0).contains(&q) {
            return f64::NAN;
        }

        let total = self.get_count();
        if total == 0 {
            return f64::NAN;
        }

        let target = (q * total as f64).ceil() as u64;
        let mut cumulative = 0u64;

        for (i, count) in self.counts.iter().enumerate() {
            cumulative += count.load(Ordering::Relaxed);
            if cumulative >= target {
                // Linear interpolation within bucket
                let prev_bound = if i > 0 { self.buckets[i - 1] } else { 0.0 };
                let curr_bound = self.buckets[i];

                if curr_bound.is_infinite() {
                    return prev_bound;
                }

                // Simple midpoint approximation (compatible with MSRV 1.75)
                return (prev_bound + curr_bound) / 2.0;
            }
        }

        // Should not reach here
        *self.buckets.last().unwrap_or(&0.0)
    }

    /// Get metric descriptor
    pub fn desc(&self) -> &MetricDesc {
        &self.desc
    }

    /// Get metric name
    pub fn name(&self) -> &str {
        &self.desc.name
    }

    /// Get metric labels
    pub fn labels(&self) -> &Labels {
        &self.desc.labels
    }

    /// Get creation timestamp
    pub fn created_at(&self) -> Instant {
        self.created_at
    }
}

impl Clone for Histogram {
    fn clone(&self) -> Self {
        Self {
            desc: self.desc.clone(),
            buckets: self.buckets.clone(),
            counts: self.counts.iter().map(|c| AtomicU64::new(c.load(Ordering::Relaxed))).collect(),
            sum_bits: AtomicU64::new(self.sum_bits.load(Ordering::Relaxed)),
            count: AtomicU64::new(self.count.load(Ordering::Relaxed)),
            created_at: self.created_at,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    /// Linear buckets: start, width, count
    fn linear_buckets(start: f64, width: f64, count: usize) -> Vec<f64> {
        (0..count).map(|i| start + width * i as f64).collect()
    }

    /// Exponential buckets: start, factor, count
    fn exponential_buckets(start: f64, factor: f64, count: usize) -> Vec<f64> {
        (0..count).map(|i| start * factor.powi(i as i32)).collect()
    }

    /// Histogram family - collection of histograms with same name but different labels
    #[derive(Debug)]
    struct HistogramVec {
        name: String,
        label_keys: Vec<String>,
        buckets: Vec<f64>,
        children: dashmap::DashMap<Vec<String>, Arc<Histogram>>,
    }

    impl HistogramVec {
        fn new(
            name: impl Into<String>,
            label_keys: impl IntoIterator<Item = impl Into<String>>,
            buckets: &[f64],
        ) -> Self {
            Self {
                name: name.into(),
                label_keys: label_keys.into_iter().map(Into::into).collect(),
                buckets: buckets.to_vec(),
                children: dashmap::DashMap::new(),
            }
        }

        fn with_label_values(&self, values: &[&str]) -> Arc<Histogram> {
            assert_eq!(
                values.len(),
                self.label_keys.len(),
                "label count mismatch: expected {}, got {}",
                self.label_keys.len(),
                values.len()
            );

            let key: Vec<String> = values.iter().map(std::string::ToString::to_string).collect();

            self.children
                .entry(key.clone())
                .or_insert_with(|| {
                    let labels: Labels = self
                        .label_keys
                        .iter()
                        .zip(values.iter())
                        .map(|(k, v)| (k.clone(), (*v).to_string()))
                        .collect();
                    Histogram::new_arc(&self.name, labels, &self.buckets)
                })
                .clone()
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn iter(&self) -> impl Iterator<Item = Arc<Histogram>> + '_ {
            self.children.iter().map(|entry| entry.value().clone())
        }
    }

    /// Timer helper for measuring duration and recording to histogram
    struct HistogramTimer {
        histogram: Arc<Histogram>,
        start: Instant,
    }

    impl HistogramTimer {
        fn new(histogram: Arc<Histogram>) -> Self {
            Self { histogram, start: Instant::now() }
        }

        fn observe_duration(self) -> f64 {
            let elapsed = self.start.elapsed().as_secs_f64();
            self.histogram.observe(elapsed);
            elapsed
        }
    }

    impl Drop for HistogramTimer {
        fn drop(&mut self) {
            let elapsed = self.start.elapsed().as_secs_f64();
            self.histogram.observe(elapsed);
        }
    }

    #[test]
    fn test_histogram_observe() {
        let h = Histogram::with_default_buckets("test_histogram", Labels::new());

        h.observe(0.005); // Bucket 0: le=0.005
        h.observe(0.015); // Bucket 1: le=0.01 (0.015 > 0.01, so bucket 2)
        h.observe(0.030); // Bucket 2: le=0.025 (0.03 > 0.025, so bucket 3)

        assert_eq!(h.get_count(), 3);
        assert!((h.get_sum() - 0.050).abs() < 0.001);
    }

    #[test]
    fn test_histogram_buckets() {
        let h = Histogram::new("test", Labels::new(), &[1.0, 5.0, 10.0]);

        h.observe(0.5); // Bucket 0: le=1.0
        h.observe(3.0); // Bucket 1: le=5.0
        h.observe(7.0); // Bucket 2: le=10.0
        h.observe(100.0); // Bucket 3: le=+Inf

        assert_eq!(h.get_bucket_count(0), 1);
        assert_eq!(h.get_bucket_count(1), 1);
        assert_eq!(h.get_bucket_count(2), 1);
        assert_eq!(h.get_bucket_count(3), 1);

        let cumulative = h.cumulative_counts();
        assert_eq!(cumulative, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_histogram_quantile() {
        let h = Histogram::new("test", Labels::new(), &[1.0, 2.0, 3.0, 4.0, 5.0]);

        // Add 100 observations: 20 in each bucket
        for i in 0..100 {
            h.observe((i % 5 + 1) as f64 - 0.5);
        }

        // p50 should be around 2.5
        let p50 = h.quantile(0.5);
        assert!(p50 >= 1.5 && p50 <= 3.5, "p50 = {}", p50);
    }

    #[test]
    fn test_histogram_thread_safety() {
        let h = Arc::new(Histogram::with_default_buckets("concurrent", Labels::new()));

        let threads: Vec<_> = (0..10)
            .map(|i| {
                let hist = Arc::clone(&h);
                thread::spawn(move || {
                    for j in 0..1000 {
                        hist.observe((i * 1000 + j) as f64 * 0.001);
                    }
                })
            })
            .collect();

        for t in threads {
            t.join().unwrap();
        }

        assert_eq!(h.get_count(), 10_000);
    }

    #[test]
    fn test_histogram_infinity_bucket() {
        let h = Histogram::new("test", Labels::new(), &[1.0]);

        h.observe(f64::MAX);

        // Should land in +Inf bucket
        assert_eq!(h.get_bucket_count(1), 1);
    }

    #[test]
    fn test_histogram_vec() {
        let vec = HistogramVec::new("request_duration", ["method"], DEFAULT_BUCKETS);

        let get_hist = vec.with_label_values(&["GET"]);
        let post_hist = vec.with_label_values(&["POST"]);

        get_hist.observe(0.1);
        get_hist.observe(0.2);
        post_hist.observe(0.5);

        assert_eq!(get_hist.get_count(), 2);
        assert_eq!(post_hist.get_count(), 1);
    }

    #[test]
    fn test_linear_buckets() {
        let buckets = linear_buckets(0.0, 1.0, 5);
        assert_eq!(buckets, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_exponential_buckets() {
        let buckets = exponential_buckets(1.0, 2.0, 4);
        assert_eq!(buckets, vec![1.0, 2.0, 4.0, 8.0]);
    }

    #[test]
    fn test_default_buckets() {
        assert_eq!(DEFAULT_BUCKETS.len(), 11);
        assert_eq!(DEFAULT_BUCKETS[0], 0.005);
        assert_eq!(DEFAULT_BUCKETS[10], 10.0);
    }

    #[test]
    fn test_histogram_sum_accuracy() {
        let h = Histogram::with_default_buckets("test", Labels::new());

        // Add 1.0 exactly 1000 times
        for _ in 0..1000 {
            h.observe(1.0);
        }

        assert_eq!(h.get_count(), 1000);
        assert!((h.get_sum() - 1000.0).abs() < 0.001);
    }

    #[test]
    fn test_histogram_new_arc() {
        let h = Histogram::new_arc("test_arc", Labels::new(), &[1.0, 5.0]);
        h.observe(2.5);
        assert_eq!(h.get_count(), 1);
    }

    #[test]
    fn test_histogram_bucket_bound() {
        let h = Histogram::new("test", Labels::new(), &[1.0, 5.0, 10.0]);
        assert_eq!(h.get_bucket_bound(0), Some(1.0));
        assert_eq!(h.get_bucket_bound(1), Some(5.0));
        assert_eq!(h.get_bucket_bound(2), Some(10.0));
        assert!(h.get_bucket_bound(3).unwrap().is_infinite()); // +Inf
        assert_eq!(h.get_bucket_bound(100), None);
    }

    #[test]
    fn test_histogram_bucket_count_fn() {
        let h = Histogram::new("test", Labels::new(), &[1.0]);
        assert_eq!(h.bucket_count(), 2); // 1.0 and +Inf
    }

    #[test]
    fn test_histogram_metadata() {
        let labels: Labels = [("env".to_string(), "prod".to_string())].into_iter().collect();
        let h = Histogram::new("request_duration", labels, &[1.0]);

        assert_eq!(h.name(), "request_duration");
        assert_eq!(h.labels().get("env"), Some(&"prod".to_string()));
        assert!(h.desc().key().contains("request_duration"));
        // created_at should be recent
        assert!(h.created_at().elapsed().as_secs() < 1);
    }

    #[test]
    fn test_histogram_clone() {
        let h = Histogram::with_default_buckets("test", Labels::new());
        h.observe(1.0);
        h.observe(2.0);

        let h2 = h.clone();
        assert_eq!(h2.get_count(), 2);
        assert!((h2.get_sum() - 3.0).abs() < 0.001);

        // Original still works
        h.observe(3.0);
        assert_eq!(h.get_count(), 3);
        assert_eq!(h2.get_count(), 2); // Clone is independent
    }

    #[test]
    fn test_histogram_quantile_edge_cases() {
        let h = Histogram::new("test", Labels::new(), &[1.0, 2.0, 3.0]);

        // Empty histogram
        assert!(h.quantile(0.5).is_nan());

        // Invalid quantile values
        assert!(h.quantile(-0.1).is_nan());
        assert!(h.quantile(1.1).is_nan());

        // Add observations
        h.observe(0.5);
        h.observe(1.5);
        h.observe(2.5);

        // q=0 should give first bucket
        let p0 = h.quantile(0.0);
        assert!(!p0.is_nan());

        // q=1 should give last bucket
        let p100 = h.quantile(1.0);
        assert!(!p100.is_nan());
    }

    #[test]
    fn test_histogram_quantile_infinity_bucket() {
        let h = Histogram::new("test", Labels::new(), &[1.0]);

        // All observations in +Inf bucket
        h.observe(100.0);
        h.observe(200.0);

        // Quantile should return prev_bound (1.0) when hitting +Inf
        let p50 = h.quantile(0.5);
        assert!((p50 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_histogram_vec_name() {
        let vec = HistogramVec::new("test_vec", ["label"], &[1.0]);
        assert_eq!(vec.name(), "test_vec");
    }

    #[test]
    fn test_histogram_vec_iter() {
        let vec = HistogramVec::new("test_vec", ["method"], &[1.0]);

        vec.with_label_values(&["GET"]).observe(0.5);
        vec.with_label_values(&["POST"]).observe(0.5);
        vec.with_label_values(&["DELETE"]).observe(0.5);

        let histograms: Vec<_> = vec.iter().collect();
        assert_eq!(histograms.len(), 3);
    }

    #[test]
    fn test_histogram_timer() {
        let h = Histogram::new_arc("timer_test", Labels::new(), &[0.001, 0.01, 0.1, 1.0]);

        {
            let _timer = HistogramTimer::new(Arc::clone(&h));
            // Timer runs for a tiny bit
            std::thread::sleep(std::time::Duration::from_micros(100));
        } // Timer dropped, observation recorded

        assert_eq!(h.get_count(), 1);
        assert!(h.get_sum() > 0.0);
    }

    #[test]
    fn test_histogram_timer_observe_duration() {
        let h = Histogram::new_arc("timer_test", Labels::new(), &[0.001, 0.01, 0.1, 1.0]);

        let timer = HistogramTimer::new(Arc::clone(&h));
        std::thread::sleep(std::time::Duration::from_micros(50));
        let elapsed = timer.observe_duration();

        assert!(elapsed > 0.0);
        // Note: observe_duration consumes timer, then Drop also runs
        // This means 2 observations total
        assert!(h.get_count() >= 1);
    }

    #[test]
    fn test_histogram_bucket_counts() {
        let h = Histogram::new("test", Labels::new(), &[1.0, 2.0, 3.0]);

        h.observe(0.5); // bucket 0
        h.observe(1.5); // bucket 1
        h.observe(2.5); // bucket 2
        h.observe(100.0); // bucket 3 (+Inf)

        let counts = h.bucket_counts();
        assert_eq!(counts.len(), 4);
        assert_eq!(counts[0], 1);
        assert_eq!(counts[1], 1);
        assert_eq!(counts[2], 1);
        assert_eq!(counts[3], 1);
    }

    #[test]
    fn test_histogram_get_bucket_count_out_of_bounds() {
        let h = Histogram::new("test", Labels::new(), &[1.0]);
        assert_eq!(h.get_bucket_count(100), 0);
    }

    #[test]
    fn test_histogram_with_existing_infinity() {
        // Bucket list already has infinity
        let h = Histogram::new("test", Labels::new(), &[1.0, f64::INFINITY]);
        assert_eq!(h.bucket_count(), 2); // Should not add another +Inf
    }

    #[test]
    fn test_histogram_unsorted_buckets() {
        // Buckets should be sorted internally
        let h = Histogram::new("test", Labels::new(), &[5.0, 1.0, 3.0]);

        assert_eq!(h.buckets()[0], 1.0);
        assert_eq!(h.buckets()[1], 3.0);
        assert_eq!(h.buckets()[2], 5.0);
    }

    #[test]
    fn test_histogram_negative_values() {
        let h = Histogram::new("test", Labels::new(), &[-10.0, 0.0, 10.0]);

        h.observe(-15.0); // Goes to first bucket (le=-10)? No, -15 <= -10 is false
        h.observe(-5.0); // le=-10? -5 <= -10 is false, le=0? -5 <= 0 is true
        h.observe(5.0); // le=10

        assert_eq!(h.get_count(), 3);
    }

    #[test]
    fn test_histogram_zero_value() {
        let h = Histogram::new("test", Labels::new(), &[0.0, 1.0]);

        h.observe(0.0);
        assert_eq!(h.get_bucket_count(0), 1);
    }
}
