//! Counter metric - monotonically increasing value
//!
//! Counters track cumulative values that only increase (or reset to zero on restart).
//! Common use cases: request counts, bytes transferred, errors.
//!
//! # Performance
//! - inc(): <50ns (single atomic add)
//! - add(): <50ns (single atomic add)
//!
//! # Pattern
//! Linux kernel's `perf_event_attr.type = PERF_TYPE_HARDWARE`
//!
//! # Example
//! ```ignore
//! let counter = Counter::new("requests_total", labels);
//! counter.inc();
//! counter.add(5);
//! assert_eq!(counter.get(), 6);
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::labels::Labels;
use super::MetricDesc;

/// Counter metric - monotonically increasing value
///
/// Thread-safe via atomic operations. Never decreases except on reset.
#[derive(Debug)]
pub struct Counter {
    /// Metric descriptor (name + labels)
    desc: MetricDesc,
    /// Current value (atomic for thread-safety)
    value: AtomicU64,
    /// Creation timestamp
    created_at: Instant,
}

impl Counter {
    /// Create a new counter
    pub fn new(name: impl Into<String>, labels: Labels) -> Self {
        Self {
            desc: MetricDesc::new(name, labels),
            value: AtomicU64::new(0),
            created_at: Instant::now(),
        }
    }

    /// Create a new counter wrapped in Arc for shared ownership
    pub fn new_arc(name: impl Into<String>, labels: Labels) -> Arc<Self> {
        Arc::new(Self::new(name, labels))
    }

    /// Increment counter by 1
    ///
    /// Hot path: single atomic add (~10-20ns)
    #[inline]
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment counter by arbitrary value
    ///
    /// Hot path: single atomic add (~10-20ns)
    #[inline]
    pub fn add(&self, delta: u64) {
        self.value.fetch_add(delta, Ordering::Relaxed);
    }

    /// Get current counter value
    #[inline]
    pub fn get(&self) -> u64 {
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

    /// Reset counter to zero (only for testing)
    #[cfg(test)]
    pub fn reset(&self) {
        self.value.store(0, Ordering::Relaxed);
    }
}

impl Clone for Counter {
    fn clone(&self) -> Self {
        Self {
            desc: self.desc.clone(),
            value: AtomicU64::new(self.value.load(Ordering::Relaxed)),
            created_at: self.created_at,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    /// Counter family - collection of counters with same name but different labels
    #[derive(Debug)]
    struct CounterVec {
        name: String,
        label_keys: Vec<String>,
        children: dashmap::DashMap<Vec<String>, Arc<Counter>>,
    }

    impl CounterVec {
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

        fn with_label_values(&self, values: &[&str]) -> Arc<Counter> {
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
                    Counter::new_arc(&self.name, labels)
                })
                .clone()
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn label_keys(&self) -> &[String] {
            &self.label_keys
        }

        fn iter(&self) -> impl Iterator<Item = Arc<Counter>> + '_ {
            self.children.iter().map(|entry| entry.value().clone())
        }
    }

    #[test]
    fn test_counter_inc() {
        let counter = Counter::new("test_counter", Labels::new());
        assert_eq!(counter.get(), 0);
        counter.inc();
        assert_eq!(counter.get(), 1);
        counter.inc();
        assert_eq!(counter.get(), 2);
    }

    #[test]
    fn test_counter_add() {
        let counter = Counter::new("test_counter", Labels::new());
        counter.add(5);
        assert_eq!(counter.get(), 5);
        counter.add(10);
        assert_eq!(counter.get(), 15);
    }

    #[test]
    fn test_counter_with_labels() {
        let labels: Labels = [("method", "GET"), ("path", "/api")]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        let counter = Counter::new("http_requests", labels.clone());
        assert_eq!(counter.name(), "http_requests");
        assert_eq!(counter.labels().get("method"), Some(&"GET".to_string()));
    }

    #[test]
    fn test_counter_thread_safety() {
        let counter = Arc::new(Counter::new("concurrent_counter", Labels::new()));
        let threads: Vec<_> = (0..10)
            .map(|_| {
                let c = Arc::clone(&counter);
                thread::spawn(move || {
                    for _ in 0..1000 {
                        c.inc();
                    }
                })
            })
            .collect();

        for t in threads {
            t.join().unwrap();
        }

        assert_eq!(counter.get(), 10_000);
    }

    #[test]
    fn test_counter_overflow() {
        let counter = Counter::new("overflow_test", Labels::new());
        // Set to MAX-1
        counter.add(u64::MAX - 1);
        assert_eq!(counter.get(), u64::MAX - 1);

        // Increment twice - should wrap to 0
        counter.inc();
        assert_eq!(counter.get(), u64::MAX);
        counter.inc();
        assert_eq!(counter.get(), 0); // Wrapped
    }

    #[test]
    fn test_counter_vec() {
        let vec = CounterVec::new("http_requests_total", ["method", "status"]);

        let get_200 = vec.with_label_values(&["GET", "200"]);
        let post_201 = vec.with_label_values(&["POST", "201"]);

        get_200.inc();
        get_200.inc();
        post_201.inc();

        assert_eq!(get_200.get(), 2);
        assert_eq!(post_201.get(), 1);

        // Same labels should return same counter
        let get_200_again = vec.with_label_values(&["GET", "200"]);
        assert_eq!(get_200_again.get(), 2);
    }

    #[test]
    #[should_panic(expected = "label count mismatch")]
    fn test_counter_vec_wrong_label_count() {
        let vec = CounterVec::new("test", ["a", "b"]);
        vec.with_label_values(&["only_one"]);
    }

    #[test]
    fn test_counter_desc_key() {
        let labels: Labels = [("a", "1"), ("b", "2")]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        let counter = Counter::new("test", labels);
        assert_eq!(counter.desc().key(), "test|a=1|b=2");
    }

    #[test]
    fn test_counter_new_arc() {
        let counter = Counter::new_arc("arc_counter", Labels::new());
        counter.inc();
        counter.inc();
        assert_eq!(counter.get(), 2);
    }

    #[test]
    fn test_counter_created_at() {
        let before = std::time::Instant::now();
        let counter = Counter::new("time_test", Labels::new());
        let after = std::time::Instant::now();

        // created_at should be between before and after
        assert!(counter.created_at() >= before);
        assert!(counter.created_at() <= after);
    }

    #[test]
    fn test_counter_reset() {
        let counter = Counter::new("reset_test", Labels::new());
        counter.add(100);
        assert_eq!(counter.get(), 100);
        counter.reset();
        assert_eq!(counter.get(), 0);
    }

    #[test]
    fn test_counter_clone() {
        let counter = Counter::new("clone_test", Labels::new());
        counter.add(42);

        let cloned = counter.clone();
        assert_eq!(cloned.get(), 42);
        assert_eq!(cloned.name(), counter.name());

        // Cloned counter should be independent
        counter.inc();
        assert_eq!(counter.get(), 43);
        assert_eq!(cloned.get(), 42);
    }

    #[test]
    fn test_counter_vec_name() {
        let vec = CounterVec::new("my_counter", ["label1", "label2"]);
        assert_eq!(vec.name(), "my_counter");
    }

    #[test]
    fn test_counter_vec_label_keys() {
        let vec = CounterVec::new("test", ["method", "status", "path"]);
        assert_eq!(vec.label_keys(), &["method", "status", "path"]);
    }

    #[test]
    fn test_counter_vec_iter() {
        let vec = CounterVec::new("http_requests", ["method"]);

        vec.with_label_values(&["GET"]).add(10);
        vec.with_label_values(&["POST"]).add(20);
        vec.with_label_values(&["PUT"]).add(5);

        let total: u64 = vec.iter().map(|c| c.get()).sum();
        assert_eq!(total, 35);
    }

    #[test]
    fn test_counter_vec_thread_safety() {
        let vec = Arc::new(CounterVec::new("concurrent", ["id"]));
        let threads: Vec<_> = (0..10)
            .map(|i| {
                let v = Arc::clone(&vec);
                thread::spawn(move || {
                    let counter = v.with_label_values(&[&i.to_string()]);
                    for _ in 0..100 {
                        counter.inc();
                    }
                })
            })
            .collect();

        for t in threads {
            t.join().unwrap();
        }

        let total: u64 = vec.iter().map(|c| c.get()).sum();
        assert_eq!(total, 1000);
    }

    #[test]
    fn test_counter_desc_accessor() {
        let labels: Labels =
            [("env", "prod")].into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();

        let counter = Counter::new("api_calls", labels);
        let desc = counter.desc();
        assert_eq!(desc.name, "api_calls");
        assert_eq!(desc.labels.get("env"), Some(&"prod".to_string()));
    }
}
