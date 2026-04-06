//! Refresh Policy - Adaptive refresh rate control (ENT-060)
//!
//! Rate-limiting for terminal updates to balance responsiveness and performance.

use std::time::{Duration, Instant};

/// Adaptive refresh rate policy.
#[derive(Debug, Clone)]
pub struct RefreshPolicy {
    /// Minimum interval between refreshes
    pub min_interval: Duration,
    /// Maximum interval (force refresh)
    pub max_interval: Duration,
    /// Refresh every N steps
    pub step_interval: usize,
    /// Last refresh time
    last_refresh: Instant,
    /// Last refresh step
    last_step: usize,
}

impl Default for RefreshPolicy {
    fn default() -> Self {
        Self {
            min_interval: Duration::from_millis(50),
            max_interval: Duration::from_millis(1000),
            step_interval: 10,
            last_refresh: Instant::now(),
            last_step: 0,
        }
    }
}

impl RefreshPolicy {
    /// Create a new refresh policy.
    pub fn new(min_ms: u64, max_ms: u64, step_interval: usize) -> Self {
        Self {
            min_interval: Duration::from_millis(min_ms),
            max_interval: Duration::from_millis(max_ms),
            step_interval,
            last_refresh: Instant::now(),
            last_step: 0,
        }
    }

    /// Check if a refresh should occur.
    pub fn should_refresh(&mut self, global_step: usize) -> bool {
        let elapsed = self.last_refresh.elapsed();

        // Force refresh after max interval
        if elapsed >= self.max_interval {
            self.last_refresh = Instant::now();
            self.last_step = global_step;
            return true;
        }

        // Rate-limit to min interval
        if elapsed < self.min_interval {
            return false;
        }

        // Step-based refresh
        if global_step.saturating_sub(self.last_step) >= self.step_interval {
            self.last_refresh = Instant::now();
            self.last_step = global_step;
            return true;
        }

        false
    }

    /// Force a refresh (resets timer).
    pub fn force_refresh(&mut self, global_step: usize) {
        self.last_refresh = Instant::now();
        self.last_step = global_step;
    }

    /// Simulate time passage for deterministic testing.
    #[cfg(test)]
    fn advance_time(&mut self, duration: Duration) {
        self.last_refresh -= duration;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refresh_policy_default() {
        let policy = RefreshPolicy::default();
        assert_eq!(policy.min_interval, Duration::from_millis(50));
        assert_eq!(policy.max_interval, Duration::from_millis(1000));
        assert_eq!(policy.step_interval, 10);
    }

    #[test]
    fn test_refresh_policy_new() {
        let policy = RefreshPolicy::new(100, 500, 5);
        assert_eq!(policy.min_interval, Duration::from_millis(100));
        assert_eq!(policy.max_interval, Duration::from_millis(500));
        assert_eq!(policy.step_interval, 5);
    }

    #[test]
    fn test_refresh_policy_rate_limiting() {
        // Use a large min_interval (10s) so wall-clock jitter cannot cause
        // the "immediate call blocked" assertion to flake under CI load.
        let mut policy = RefreshPolicy::new(10_000, 60_000, 1);
        policy.force_refresh(0);

        // Immediate call should be blocked (min_interval not elapsed)
        let blocked = !policy.should_refresh(1);
        assert!(blocked, "Immediate refresh should be blocked");

        // Simulate 15s passing (deterministic, no thread::sleep) — past min_interval
        policy.advance_time(Duration::from_millis(15_000));
        let allowed = policy.should_refresh(2);
        assert!(allowed, "Refresh should be allowed after min_interval");
    }

    #[test]
    fn test_refresh_policy_step_interval() {
        let mut policy = RefreshPolicy::new(0, 10000, 10);
        policy.force_refresh(0);

        // Simulate time past min_interval (deterministic)
        policy.advance_time(Duration::from_millis(20));

        // Step 5 should not trigger (need 10 steps)
        assert!(!policy.should_refresh(5));
        // Step 10 should trigger
        assert!(policy.should_refresh(10));
    }

    #[test]
    fn test_refresh_policy_force_refresh() {
        let mut policy = RefreshPolicy::default();
        policy.force_refresh(100);
        assert_eq!(policy.last_step, 100);
    }

    #[test]
    fn test_refresh_policy_max_interval_triggers() {
        let mut policy = RefreshPolicy::new(10, 50, 1000);
        policy.force_refresh(0);

        // Simulate 500ms passing (deterministic)
        policy.advance_time(Duration::from_millis(500));

        // Should trigger due to max_interval
        assert!(policy.should_refresh(1));
    }

    #[test]
    fn test_refresh_policy_clone() {
        let policy = RefreshPolicy::new(100, 500, 5);
        let cloned = policy.clone();
        assert_eq!(policy.min_interval, cloned.min_interval);
        assert_eq!(policy.max_interval, cloned.max_interval);
        assert_eq!(policy.step_interval, cloned.step_interval);
    }

    #[test]
    fn test_refresh_policy_debug() {
        let policy = RefreshPolicy::default();
        let debug_str = format!("{policy:?}");
        assert!(debug_str.contains("RefreshPolicy"));
    }

    #[test]
    fn test_refresh_policy_no_refresh_below_step_interval() {
        let mut policy = RefreshPolicy::new(0, 10000, 100);
        policy.force_refresh(0);

        // Simulate time past min_interval (deterministic)
        policy.advance_time(Duration::from_millis(20));

        // Steps below interval should not trigger
        assert!(!policy.should_refresh(50));
        assert!(!policy.should_refresh(99));
        // But at 100 it should trigger
        assert!(policy.should_refresh(100));
    }
}
