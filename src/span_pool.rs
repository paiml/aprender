/// Sprint 36: Memory Pool for Span Data
///
/// Provides efficient object pooling for span allocations to reduce
/// allocator pressure and improve performance.
///
/// Zero-copy optimizations: Uses Cow<'static, str> for strings that
/// are often known at compile time (syscall names, operation types).
use std::borrow::Cow;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Configuration for the span pool
#[derive(Debug, Clone)]
pub struct SpanPoolConfig {
    /// Initial capacity of the pool
    pub capacity: usize,
    /// Whether pooling is enabled
    pub enabled: bool,
}

impl Default for SpanPoolConfig {
    fn default() -> Self {
        SpanPoolConfig { capacity: 1024, enabled: true }
    }
}

impl SpanPoolConfig {
    /// Create a new pool configuration
    pub fn new(capacity: usize) -> Self {
        SpanPoolConfig { capacity, enabled: true }
    }

    /// Disable pooling (for debugging)
    pub fn disabled() -> Self {
        SpanPoolConfig { capacity: 0, enabled: false }
    }
}

/// Pooled span data (Sprint 36: zero-copy with Cow)
#[derive(Debug, Clone)]
pub struct PooledSpan {
    pub trace_id: String,
    pub span_id: String,
    /// Operation name - often static (e.g., "syscall:open", "`compute_block:mean`")
    /// Uses Cow for zero-copy when static strings are used
    pub name: Cow<'static, str>,
    /// Attributes with static keys (zero-copy optimization)
    pub attributes: Vec<(Cow<'static, str>, String)>,
    pub timestamp_nanos: u64,
    pub duration_nanos: u64,
    pub status_code: i32, // 0 = OK, 1 = ERROR
}

impl PooledSpan {
    /// Create a new empty span
    fn new() -> Self {
        PooledSpan {
            trace_id: String::new(),
            span_id: String::new(),
            name: Cow::Borrowed(""),
            attributes: Vec::new(),
            timestamp_nanos: 0,
            duration_nanos: 0,
            status_code: 0,
        }
    }

    /// Reset span data for reuse
    fn reset(&mut self) {
        self.trace_id.clear();
        self.span_id.clear();
        self.name = Cow::Borrowed("");
        self.attributes.clear();
        self.timestamp_nanos = 0;
        self.duration_nanos = 0;
        self.status_code = 0;
    }

    /// Set span name from static string (zero-copy)
    pub fn set_name_static(&mut self, name: &'static str) {
        self.name = Cow::Borrowed(name);
    }

    /// Set span name from owned string
    pub fn set_name_owned(&mut self, name: String) {
        self.name = Cow::Owned(name);
    }

    /// Add attribute with static key (zero-copy for key)
    pub fn add_attribute_static(&mut self, key: &'static str, value: String) {
        self.attributes.push((Cow::Borrowed(key), value));
    }

    /// Add attribute with owned key
    pub fn add_attribute_owned(&mut self, key: String, value: String) {
        self.attributes.push((Cow::Owned(key), value));
    }
}

/// Memory pool for span allocations
pub struct SpanPool {
    pool: Vec<PooledSpan>,
    config: SpanPoolConfig,
    allocated: AtomicUsize,
    acquired: AtomicUsize,
}

impl SpanPool {
    /// Create a new span pool with the given configuration
    pub fn new(config: SpanPoolConfig) -> Self {
        let capacity = config.capacity;
        let mut pool = Vec::with_capacity(capacity);

        if config.enabled {
            // Pre-allocate pool
            for _ in 0..capacity {
                pool.push(PooledSpan::new());
            }
        }

        SpanPool {
            pool,
            config,
            allocated: AtomicUsize::new(capacity),
            acquired: AtomicUsize::new(0),
        }
    }

    /// Acquire a span from the pool
    ///
    /// If the pool is empty, allocates a new span (automatic growth).
    /// If pooling is disabled, always allocates a new span.
    pub fn acquire(&mut self) -> PooledSpan {
        if !self.config.enabled {
            // Pooling disabled, always allocate
            self.allocated.fetch_add(1, Ordering::Relaxed);
            return PooledSpan::new();
        }

        self.acquired.fetch_add(1, Ordering::Relaxed);

        if let Some(mut span) = self.pool.pop() {
            span.reset();
            span
        } else {
            // Pool exhausted, allocate new
            self.allocated.fetch_add(1, Ordering::Relaxed);
            PooledSpan::new()
        }
    }

    /// Release a span back to the pool
    ///
    /// If pooling is disabled, the span is simply dropped.
    pub fn release(&mut self, span: PooledSpan) {
        if !self.config.enabled {
            // Pooling disabled, just drop
            return;
        }

        // Only pool if we have capacity
        if self.pool.len() < self.pool.capacity() {
            self.pool.push(span);
        }
        // Otherwise drop (prevents unbounded growth)
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            capacity: self.config.capacity,
            available: self.pool.len(),
            allocated: self.allocated.load(Ordering::Relaxed),
            acquired: self.acquired.load(Ordering::Relaxed),
            enabled: self.config.enabled,
        }
    }

    /// Check if the pool is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the current pool size (available spans)
    pub fn available(&self) -> usize {
        self.pool.len()
    }

    /// Get the pool capacity
    pub fn capacity(&self) -> usize {
        self.config.capacity
    }
}

/// Pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Pool capacity
    pub capacity: usize,
    /// Available spans in pool
    pub available: usize,
    /// Total spans allocated (including auto-growth)
    pub allocated: usize,
    /// Total acquire operations
    pub acquired: usize,
    /// Whether pooling is enabled
    pub enabled: bool,
}

impl PoolStats {
    /// Calculate hit rate (percentage of acquires served from pool)
    pub fn hit_rate(&self) -> f64 {
        if self.acquired == 0 {
            return 0.0;
        }
        let hits = self.acquired.saturating_sub(self.allocated - self.capacity);
        (hits as f64 / self.acquired as f64) * 100.0
    }

    /// Calculate pool utilization (percentage of capacity used)
    pub fn utilization(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        let used = self.capacity - self.available;
        (used as f64 / self.capacity as f64) * 100.0
    }
}

// Compile-time thread-safety verification (Sprint 59)
static_assertions::assert_impl_all!(SpanPoolConfig: Send, Sync);
static_assertions::assert_impl_all!(PooledSpan: Send, Sync);
static_assertions::assert_impl_all!(PoolStats: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_acquire_release() {
        let mut pool = SpanPool::new(SpanPoolConfig::new(10));

        // Acquire a span
        let span = pool.acquire();
        assert_eq!(pool.available(), 9);

        // Release it back
        pool.release(span);
        assert_eq!(pool.available(), 10);
    }

    #[test]
    fn test_pool_exhaustion() {
        let mut pool = SpanPool::new(SpanPoolConfig::new(2));

        // Exhaust the pool
        let span1 = pool.acquire();
        let span2 = pool.acquire();
        assert_eq!(pool.available(), 0);

        // Acquiring more should auto-grow
        let span3 = pool.acquire();
        assert_eq!(pool.available(), 0);

        // Release all
        pool.release(span1);
        pool.release(span2);
        pool.release(span3);
        assert_eq!(pool.available(), 2); // Only capacity worth retained
    }

    #[test]
    fn test_pool_reset() {
        let mut pool = SpanPool::new(SpanPoolConfig::new(10));

        // Acquire and modify a span
        let mut span = pool.acquire();
        span.set_name_owned("test".to_string());
        span.trace_id = "abc123".to_string();
        span.timestamp_nanos = 12345;

        // Release and re-acquire
        pool.release(span);
        let span2 = pool.acquire();

        // Should be reset
        assert_eq!(span2.name.as_ref(), "");
        assert_eq!(span2.trace_id, "");
        assert_eq!(span2.timestamp_nanos, 0);
    }

    #[test]
    fn test_zero_copy_static_strings() {
        let mut pool = SpanPool::new(SpanPoolConfig::new(10));

        // Acquire span and use static string (zero-copy)
        let mut span = pool.acquire();
        span.set_name_static("syscall:open");
        span.add_attribute_static("syscall.name", "open".to_string());
        span.add_attribute_static("syscall.result", "3".to_string());

        // Verify static borrowing (no allocation for keys)
        assert_eq!(span.name.as_ref(), "syscall:open");
        assert!(matches!(span.name, Cow::Borrowed(_)));
        assert_eq!(span.attributes.len(), 2);
        assert!(matches!(span.attributes[0].0, Cow::Borrowed(_)));
        assert!(matches!(span.attributes[1].0, Cow::Borrowed(_)));

        // Compare with owned version
        let mut span2 = pool.acquire();
        span2.set_name_owned("syscall:open".to_string());
        span2.add_attribute_owned("syscall.name".to_string(), "open".to_string());

        assert_eq!(span2.name.as_ref(), "syscall:open");
        assert!(matches!(span2.name, Cow::Owned(_)));
        assert!(matches!(span2.attributes[0].0, Cow::Owned(_)));

        pool.release(span);
        pool.release(span2);
    }

    #[test]
    fn test_pool_disabled() {
        let mut pool = SpanPool::new(SpanPoolConfig::disabled());
        assert!(!pool.is_enabled());
        assert_eq!(pool.capacity(), 0);

        // Acquire should always allocate
        let span1 = pool.acquire();
        let span2 = pool.acquire();

        // Release should be no-op
        pool.release(span1);
        pool.release(span2);
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn test_pool_stats() {
        let mut pool = SpanPool::new(SpanPoolConfig::new(10));

        // Initial stats
        let stats = pool.stats();
        assert_eq!(stats.capacity, 10);
        assert_eq!(stats.available, 10);
        assert_eq!(stats.allocated, 10);
        assert_eq!(stats.acquired, 0);

        // Acquire some spans
        let span1 = pool.acquire();
        let span2 = pool.acquire();

        let stats = pool.stats();
        assert_eq!(stats.available, 8);
        assert_eq!(stats.acquired, 2);

        // Release one
        pool.release(span1);
        let stats = pool.stats();
        assert_eq!(stats.available, 9);

        // Keep one alive
        drop(span2);
    }

    #[test]
    fn test_pool_hit_rate() {
        let mut pool = SpanPool::new(SpanPoolConfig::new(2));

        // All hits (from pool)
        let span1 = pool.acquire(); // hit
        let span2 = pool.acquire(); // hit
        pool.release(span1);
        pool.release(span2);

        let stats = pool.stats();
        assert!(stats.hit_rate() >= 99.0); // Should be ~100%

        // Cause a miss (pool exhaustion)
        let span1 = pool.acquire();
        let span2 = pool.acquire();
        let _span3 = pool.acquire(); // miss (allocate)

        let stats = pool.stats();
        assert!(stats.hit_rate() < 100.0); // Should be ~66%

        drop((span1, span2));
    }

    #[test]
    fn test_pool_utilization() {
        let mut pool = SpanPool::new(SpanPoolConfig::new(10));

        // 0% utilization (all available)
        let stats = pool.stats();
        assert_eq!(stats.utilization(), 0.0);

        // Acquire 5 (50% utilization)
        let mut spans = Vec::new();
        for _ in 0..5 {
            spans.push(pool.acquire());
        }

        let stats = pool.stats();
        assert_eq!(stats.utilization(), 50.0);

        // Acquire all (100% utilization)
        for _ in 0..5 {
            spans.push(pool.acquire());
        }

        let stats = pool.stats();
        assert_eq!(stats.utilization(), 100.0);

        // Release all
        for span in spans {
            pool.release(span);
        }
    }

    #[test]
    fn test_pool_concurrent_usage() {
        let mut pool = SpanPool::new(SpanPoolConfig::new(100));

        // Simulate concurrent acquire/release pattern
        for _ in 0..1000 {
            let span = pool.acquire();
            pool.release(span);
        }

        let stats = pool.stats();
        assert_eq!(stats.acquired, 1000);
        assert_eq!(stats.available, 100); // All released
    }

    #[test]
    fn test_pool_growth() {
        let mut pool = SpanPool::new(SpanPoolConfig::new(10));

        // Acquire more than capacity
        let mut spans = Vec::new();
        for _ in 0..20 {
            spans.push(pool.acquire());
        }

        let stats = pool.stats();
        assert_eq!(stats.acquired, 20);
        assert!(stats.allocated >= 20); // Had to allocate more

        // Release all
        for span in spans {
            pool.release(span);
        }

        // Pool should retain only capacity worth
        assert_eq!(pool.available(), 10);
    }
}
