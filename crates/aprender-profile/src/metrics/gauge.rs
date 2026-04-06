//! Gauge metric - arbitrary value that can go up or down
//!
//! Gauges track point-in-time values that may increase or decrease.
//! Common use cases: temperature, queue size, active connections.
//!
//! # Performance
//! - set(): <30ns (single atomic store)
//! - inc(): <30ns (single atomic add)
//! - dec(): <30ns (single atomic sub)
//!
//! # Pattern
//! Linux kernel's `/proc/meminfo` style metrics
//!
//! # Example
//! ```ignore
//! let gauge = Gauge::new("active_connections", labels);
//! gauge.set(100);
//! gauge.inc();
//! gauge.dec();
//! assert_eq!(gauge.get(), 100);
//! ```

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::labels::Labels;
use super::MetricDesc;

/// Gauge metric - arbitrary value that can go up or down
///
/// Thread-safe via atomic operations. Supports negative values.
#[derive(Debug)]
pub struct Gauge {
    /// Metric descriptor (name + labels)
    desc: MetricDesc,
    /// Current value (atomic for thread-safety)
    value: AtomicI64,
    /// Creation timestamp
    created_at: Instant,
}

impl Gauge {
    /// Create a new gauge
    pub fn new(name: impl Into<String>, labels: Labels) -> Self {
        Self {
            desc: MetricDesc::new(name, labels),
            value: AtomicI64::new(0),
            created_at: Instant::now(),
        }
    }

    /// Create a new gauge wrapped in Arc for shared ownership
    pub fn new_arc(name: impl Into<String>, labels: Labels) -> Arc<Self> {
        Arc::new(Self::new(name, labels))
    }

    /// Set gauge to specific value
    ///
    /// Hot path: single atomic store (~5-10ns)
    #[inline]
    pub fn set(&self, value: i64) {
        self.value.store(value, Ordering::Relaxed);
    }

    /// Increment gauge by 1
    ///
    /// Hot path: single atomic add (~10-20ns)
    #[inline]
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement gauge by 1
    ///
    /// Hot path: single atomic sub (~10-20ns)
    #[inline]
    pub fn dec(&self) {
        self.value.fetch_sub(1, Ordering::Relaxed);
    }

    /// Add arbitrary value to gauge
    #[inline]
    pub fn add(&self, delta: i64) {
        self.value.fetch_add(delta, Ordering::Relaxed);
    }

    /// Subtract arbitrary value from gauge
    #[inline]
    pub fn sub(&self, delta: i64) {
        self.value.fetch_sub(delta, Ordering::Relaxed);
    }

    /// Get current gauge value
    #[inline]
    pub fn get(&self) -> i64 {
        self.value.load(Ordering::Relaxed)
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

    /// Set gauge to current Unix timestamp
    pub fn set_to_current_time(&self) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        self.set(timestamp);
    }
}

impl Clone for Gauge {
    fn clone(&self) -> Self {
        Self {
            desc: self.desc.clone(),
            value: AtomicI64::new(self.value.load(Ordering::Relaxed)),
            created_at: self.created_at,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    /// Gauge family - collection of gauges with same name but different labels
    #[derive(Debug)]
    struct GaugeVec {
        name: String,
        label_keys: Vec<String>,
        children: dashmap::DashMap<Vec<String>, Arc<Gauge>>,
    }

    impl GaugeVec {
        fn new(
            name: impl Into<String>,
            label_keys: impl IntoIterator<Item = impl Into<String>>,
        ) -> Self {
            Self {
                name: name.into(),
                label_keys: label_keys.into_iter().map(Into::into).collect(),
                children: dashmap::DashMap::new(),
            }
        }

        fn with_label_values(&self, values: &[&str]) -> Arc<Gauge> {
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
                    Gauge::new_arc(&self.name, labels)
                })
                .clone()
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn label_keys(&self) -> &[String] {
            &self.label_keys
        }

        fn iter(&self) -> impl Iterator<Item = Arc<Gauge>> + '_ {
            self.children.iter().map(|entry| entry.value().clone())
        }
    }

    #[test]
    fn test_gauge_set() {
        let gauge = Gauge::new("test_gauge", Labels::new());
        assert_eq!(gauge.get(), 0);
        gauge.set(42);
        assert_eq!(gauge.get(), 42);
        gauge.set(-10);
        assert_eq!(gauge.get(), -10);
    }

    #[test]
    fn test_gauge_inc_dec() {
        let gauge = Gauge::new("test_gauge", Labels::new());
        gauge.inc();
        assert_eq!(gauge.get(), 1);
        gauge.inc();
        assert_eq!(gauge.get(), 2);
        gauge.dec();
        assert_eq!(gauge.get(), 1);
    }

    #[test]
    fn test_gauge_add_sub() {
        let gauge = Gauge::new("test_gauge", Labels::new());
        gauge.add(10);
        assert_eq!(gauge.get(), 10);
        gauge.sub(5);
        assert_eq!(gauge.get(), 5);
        gauge.add(-15);
        assert_eq!(gauge.get(), -10);
    }

    #[test]
    fn test_gauge_negative() {
        let gauge = Gauge::new("test_gauge", Labels::new());
        gauge.set(-100);
        assert_eq!(gauge.get(), -100);
    }

    #[test]
    fn test_gauge_thread_safety() {
        let gauge = Arc::new(Gauge::new("concurrent_gauge", Labels::new()));
        let threads: Vec<_> = (0..10)
            .map(|i| {
                let g = Arc::clone(&gauge);
                thread::spawn(move || {
                    for _ in 0..1000 {
                        if i % 2 == 0 {
                            g.inc();
                        } else {
                            g.dec();
                        }
                    }
                })
            })
            .collect();

        for t in threads {
            t.join().unwrap();
        }

        // 5 threads inc, 5 threads dec, 1000 each = should be 0
        assert_eq!(gauge.get(), 0);
    }

    #[test]
    fn test_gauge_i64_min() {
        let gauge = Gauge::new("test_gauge", Labels::new());
        gauge.set(i64::MIN);
        assert_eq!(gauge.get(), i64::MIN);
    }

    #[test]
    fn test_gauge_i64_max() {
        let gauge = Gauge::new("test_gauge", Labels::new());
        gauge.set(i64::MAX);
        assert_eq!(gauge.get(), i64::MAX);
    }

    #[test]
    fn test_gauge_vec() {
        let vec = GaugeVec::new("temperature", ["location"]);

        let kitchen = vec.with_label_values(&["kitchen"]);
        let bedroom = vec.with_label_values(&["bedroom"]);

        kitchen.set(22);
        bedroom.set(18);

        assert_eq!(kitchen.get(), 22);
        assert_eq!(bedroom.get(), 18);

        // Same labels should return same gauge
        let kitchen_again = vec.with_label_values(&["kitchen"]);
        assert_eq!(kitchen_again.get(), 22);
    }

    #[test]
    fn test_gauge_with_labels() {
        let labels: Labels = [("host", "server1")]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        let gauge = Gauge::new("cpu_usage", labels);
        assert_eq!(gauge.name(), "cpu_usage");
        assert_eq!(gauge.labels().get("host"), Some(&"server1".to_string()));
    }

    #[test]
    fn test_gauge_new_arc() {
        let gauge = Gauge::new_arc("arc_gauge", Labels::new());
        gauge.set(100);
        gauge.inc();
        assert_eq!(gauge.get(), 101);
    }

    #[test]
    fn test_gauge_created_at() {
        let before = std::time::Instant::now();
        let gauge = Gauge::new("time_test", Labels::new());
        let after = std::time::Instant::now();

        assert!(gauge.created_at() >= before);
        assert!(gauge.created_at() <= after);
    }

    #[test]
    fn test_gauge_set_to_current_time() {
        let gauge = Gauge::new("timestamp", Labels::new());

        let before =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()
                as i64;

        gauge.set_to_current_time();

        let after =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()
                as i64;

        let timestamp = gauge.get();
        assert!(timestamp >= before, "timestamp {} < before {}", timestamp, before);
        assert!(timestamp <= after, "timestamp {} > after {}", timestamp, after);
    }

    #[test]
    fn test_gauge_clone() {
        let gauge = Gauge::new("clone_test", Labels::new());
        gauge.set(42);

        let cloned = gauge.clone();
        assert_eq!(cloned.get(), 42);
        assert_eq!(cloned.name(), gauge.name());

        // Cloned gauge should be independent
        gauge.inc();
        assert_eq!(gauge.get(), 43);
        assert_eq!(cloned.get(), 42);
    }

    #[test]
    fn test_gauge_desc_accessor() {
        let labels: Labels =
            [("env", "staging")].into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();

        let gauge = Gauge::new("memory_usage", labels);
        let desc = gauge.desc();
        assert_eq!(desc.name, "memory_usage");
        assert_eq!(desc.labels.get("env"), Some(&"staging".to_string()));
    }

    #[test]
    fn test_gauge_vec_name() {
        let vec = GaugeVec::new("my_gauge", ["label1", "label2"]);
        assert_eq!(vec.name(), "my_gauge");
    }

    #[test]
    fn test_gauge_vec_label_keys() {
        let vec = GaugeVec::new("test", ["host", "region", "az"]);
        assert_eq!(vec.label_keys(), &["host", "region", "az"]);
    }

    #[test]
    fn test_gauge_vec_iter() {
        let vec = GaugeVec::new("connections", ["service"]);

        vec.with_label_values(&["api"]).set(100);
        vec.with_label_values(&["web"]).set(200);
        vec.with_label_values(&["worker"]).set(50);

        let total: i64 = vec.iter().map(|g| g.get()).sum();
        assert_eq!(total, 350);
    }

    #[test]
    #[should_panic(expected = "label count mismatch")]
    fn test_gauge_vec_wrong_label_count() {
        let vec = GaugeVec::new("test", ["a", "b"]);
        vec.with_label_values(&["only_one"]);
    }

    #[test]
    fn test_gauge_vec_thread_safety() {
        let vec = Arc::new(GaugeVec::new("concurrent", ["id"]));
        let threads: Vec<_> = (0..10)
            .map(|i| {
                let v = Arc::clone(&vec);
                thread::spawn(move || {
                    let gauge = v.with_label_values(&[&i.to_string()]);
                    for j in 0..100 {
                        gauge.set(j);
                    }
                })
            })
            .collect();

        for t in threads {
            t.join().unwrap();
        }

        // All gauges should be at 99 (last set value)
        for g in vec.iter() {
            assert_eq!(g.get(), 99);
        }
    }

    #[test]
    fn test_gauge_sub_negative() {
        let gauge = Gauge::new("test", Labels::new());
        gauge.set(10);
        gauge.sub(-5); // Subtracting negative = adding
        assert_eq!(gauge.get(), 15);
    }
}
