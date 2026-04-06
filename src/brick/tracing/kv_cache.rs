// ============================================================================
// E.11.6: KvCacheStateTrace (MLT-05)
// ============================================================================

/// KV cache state at a single generation step.
#[derive(Debug, Clone, Default)]
pub struct KvCacheStateTrace {
    /// Generation step (0-indexed)
    pub step: usize,
    /// Total cache size in bytes
    pub cache_size_bytes: usize,
    /// Number of valid (filled) positions in cache
    pub valid_positions: usize,
    /// Maximum positions (context window size)
    pub max_positions: usize,
    /// Evictions performed this step
    pub evictions_this_step: usize,
    /// Cache hit rate (reused positions / total lookups)
    pub cache_hit_rate: f32,
    /// Oldest position still in cache
    pub oldest_position: usize,
    /// Memory fragmentation (0.0 = compact, 1.0 = fully scattered)
    pub fragmentation: f32,
    /// Positions accessed this step (for locality analysis)
    pub accessed_positions: Vec<usize>,
}

impl KvCacheStateTrace {
    /// Create a new trace for a step.
    pub fn new(step: usize, max_positions: usize) -> Self {
        Self { step, max_positions, ..Default::default() }
    }

    /// Check if context window is exhausted.
    pub fn is_window_exhausted(&self) -> bool {
        self.valid_positions >= self.max_positions
    }

    /// Get cache utilization ratio.
    pub fn utilization(&self) -> f32 {
        if self.max_positions == 0 {
            return 0.0;
        }
        self.valid_positions as f32 / self.max_positions as f32
    }
}

/// Full KV cache trace for a generation session.
#[derive(Debug, Clone, Default)]
pub struct KvCacheSessionTrace {
    /// Per-step traces
    pub steps: Vec<KvCacheStateTrace>,
    /// Total evictions across the session
    pub total_evictions: usize,
    /// Average cache hit rate
    pub avg_hit_rate: f32,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
}

impl KvCacheSessionTrace {
    /// Add a step trace.
    pub fn add_step(&mut self, trace: KvCacheStateTrace) {
        self.total_evictions += trace.evictions_this_step;
        self.peak_memory_bytes = self.peak_memory_bytes.max(trace.cache_size_bytes);

        // Update rolling average
        let n = self.steps.len() as f32 + 1.0;
        self.avg_hit_rate = (self.avg_hit_rate * (n - 1.0) + trace.cache_hit_rate) / n;

        self.steps.push(trace);
    }

    /// Check if eviction rate is concerning (>10% of steps).
    pub fn has_high_eviction_rate(&self) -> bool {
        if self.steps.is_empty() {
            return false;
        }
        let eviction_steps = self.steps.iter().filter(|s| s.evictions_this_step > 0).count();
        eviction_steps as f32 / self.steps.len() as f32 > 0.1
    }

    /// Check if KV cache is thrashing (high evictions + low hit rate).
    ///
    /// Returns true if the recent window shows both high eviction rate and low hit rate.
    /// Uses all available steps if fewer than `window` steps exist.
    ///
    /// # Arguments
    /// - `window`: Number of recent steps to consider (uses available if fewer)
    /// - `min_hit_rate`: Minimum acceptable hit rate (0.0-1.0)
    pub fn has_thrashing(&self, window: usize, min_hit_rate: f32) -> bool {
        if self.steps.is_empty() {
            return false;
        }

        // Use all steps if fewer than window
        let actual_window = std::cmp::min(window, self.steps.len());
        let recent_steps = &self.steps[self.steps.len() - actual_window..];
        let recent_evictions: usize = recent_steps.iter().map(|s| s.evictions_this_step).sum();
        let recent_hit_rate: f32 =
            recent_steps.iter().map(|s| s.cache_hit_rate).sum::<f32>() / actual_window as f32;

        // Thrashing: more than half the steps have evictions AND hit rate below threshold
        recent_evictions > actual_window / 2 && recent_hit_rate < min_hit_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_state_trace() {
        let trace = KvCacheStateTrace::new(0, 2048);
        assert_eq!(trace.step, 0);
        assert_eq!(trace.max_positions, 2048);
        assert!(!trace.is_window_exhausted());
    }

    #[test]
    fn test_kv_cache_state_utilization() {
        let mut trace = KvCacheStateTrace::new(0, 1000);
        trace.valid_positions = 500;
        assert!((trace.utilization() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_kv_cache_session_trace() {
        let mut session = KvCacheSessionTrace::default();
        session.add_step(KvCacheStateTrace {
            step: 0,
            cache_hit_rate: 0.9,
            evictions_this_step: 0,
            cache_size_bytes: 1000,
            ..Default::default()
        });
        session.add_step(KvCacheStateTrace {
            step: 1,
            cache_hit_rate: 0.8,
            evictions_this_step: 5,
            cache_size_bytes: 2000,
            ..Default::default()
        });

        assert_eq!(session.steps.len(), 2);
        assert_eq!(session.total_evictions, 5);
        assert_eq!(session.peak_memory_bytes, 2000);
        assert!((session.avg_hit_rate - 0.85).abs() < 0.01);
    }

    // ================================================================
    // Coverage tests for has_high_eviction_rate (0% → 100%)
    // ================================================================

    #[test]
    fn test_high_eviction_rate_empty_session() {
        let session = KvCacheSessionTrace::default();
        // Empty steps → early return false
        assert!(!session.has_high_eviction_rate());
    }

    #[test]
    fn test_high_eviction_rate_no_evictions() {
        let mut session = KvCacheSessionTrace::default();
        for i in 0..10 {
            session.add_step(KvCacheStateTrace {
                step: i,
                evictions_this_step: 0,
                ..Default::default()
            });
        }
        // 0/10 = 0% eviction rate → false
        assert!(!session.has_high_eviction_rate());
    }

    #[test]
    fn test_high_eviction_rate_at_boundary() {
        // Exactly 10% eviction rate: 1 out of 10 steps has evictions.
        // 1/10 = 0.1, but the check is strictly > 0.1, so this is false.
        let mut session = KvCacheSessionTrace::default();
        for i in 0..10 {
            session.add_step(KvCacheStateTrace {
                step: i,
                evictions_this_step: if i == 0 { 1 } else { 0 },
                ..Default::default()
            });
        }
        assert!(!session.has_high_eviction_rate());
    }

    #[test]
    fn test_high_eviction_rate_just_above_boundary() {
        // 2 out of 10 steps have evictions = 20% > 10% → true
        let mut session = KvCacheSessionTrace::default();
        for i in 0..10 {
            session.add_step(KvCacheStateTrace {
                step: i,
                evictions_this_step: if i < 2 { 1 } else { 0 },
                ..Default::default()
            });
        }
        assert!(session.has_high_eviction_rate());
    }

    #[test]
    fn test_high_eviction_rate_all_evictions() {
        // Every step has evictions → 100% > 10% → true
        let mut session = KvCacheSessionTrace::default();
        for i in 0..5 {
            session.add_step(KvCacheStateTrace {
                step: i,
                evictions_this_step: 3,
                ..Default::default()
            });
        }
        assert!(session.has_high_eviction_rate());
    }

    #[test]
    fn test_high_eviction_rate_single_step_with_eviction() {
        // 1 step with evictions out of 1 total = 100% → true
        let mut session = KvCacheSessionTrace::default();
        session.add_step(KvCacheStateTrace {
            step: 0,
            evictions_this_step: 1,
            ..Default::default()
        });
        assert!(session.has_high_eviction_rate());
    }

    #[test]
    fn test_high_eviction_rate_single_step_without_eviction() {
        // 1 step with 0 evictions out of 1 total = 0% → false
        let mut session = KvCacheSessionTrace::default();
        session.add_step(KvCacheStateTrace {
            step: 0,
            evictions_this_step: 0,
            ..Default::default()
        });
        assert!(!session.has_high_eviction_rate());
    }
}
