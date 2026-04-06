//! LCP-08: Graph Reuse Counter

/// Counter for tracking graph reuse in inference optimization.
///
/// Tracks how many times a computation graph has been reused,
/// enabling optimization decisions like caching or recompilation.
#[derive(Debug, Clone, Default)]
pub struct GraphReuseCounter {
    /// Number of times this graph has been executed
    reuse_count: u64,
    /// Threshold for considering graph "hot"
    hot_threshold: u64,
    /// Whether to enable caching
    cache_enabled: bool,
}

impl GraphReuseCounter {
    /// Create a new counter with hot threshold.
    pub fn new(hot_threshold: u64) -> Self {
        Self { reuse_count: 0, hot_threshold, cache_enabled: false }
    }

    /// Record a graph execution.
    pub fn record_use(&mut self) {
        self.reuse_count += 1;
        if self.reuse_count >= self.hot_threshold {
            self.cache_enabled = true;
        }
    }

    /// Check if graph is considered "hot" (heavily reused).
    #[must_use]
    pub fn is_hot(&self) -> bool {
        self.reuse_count >= self.hot_threshold
    }

    /// Check if caching should be enabled.
    #[must_use]
    pub fn should_cache(&self) -> bool {
        self.cache_enabled
    }

    /// Get the current reuse count.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.reuse_count
    }

    /// Reset the counter.
    pub fn reset(&mut self) {
        self.reuse_count = 0;
        self.cache_enabled = false;
    }
}
