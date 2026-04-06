
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    #[test]
    fn test_token_rate_limiter_capacity_limit() {
        let mut limiter = TokenRateLimiter::new(1000.0, 10);
        // Already full
        std::thread::sleep(Duration::from_millis(50));
        limiter.refill();
        // Should not exceed capacity
        assert!(limiter.tokens_available() <= 10);
    }

    // ==================== ResourceTracker Tests ====================

    #[test]
    fn test_resource_tracker_new() {
        let tracker = ResourceTracker::new(1024, 100);
        assert_eq!(tracker.memory_usage(), 0);
        assert_eq!(tracker.compute_usage(), 0);
    }

    #[test]
    fn test_resource_tracker_default() {
        let tracker = ResourceTracker::default();
        assert_eq!(tracker.memory_usage(), 0);
    }

    #[test]
    fn test_resource_tracker_can_allocate() {
        let tracker = ResourceTracker::new(1000, 100);
        assert!(tracker.can_allocate(500, 50));
        assert!(!tracker.can_allocate(2000, 50));
        assert!(!tracker.can_allocate(500, 150));
    }

    #[test]
    fn test_resource_tracker_allocate() {
        let mut tracker = ResourceTracker::new(1000, 100);
        let id = tracker.allocate(300, 30);
        assert!(id.is_some());
        assert_eq!(tracker.memory_usage(), 300);
        assert_eq!(tracker.compute_usage(), 30);
    }

    #[test]
    fn test_resource_tracker_allocate_failure() {
        let mut tracker = ResourceTracker::new(100, 100);
        let id = tracker.allocate(200, 50);
        assert!(id.is_none());
    }

    #[test]
    fn test_resource_tracker_release() {
        let mut tracker = ResourceTracker::new(1000, 100);
        let id = tracker.allocate(500, 50).expect("id");
        tracker.release(id);
        assert_eq!(tracker.memory_usage(), 0);
        assert_eq!(tracker.compute_usage(), 0);
    }

    #[test]
    fn test_resource_tracker_multiple_allocations() {
        let mut tracker = ResourceTracker::new(1000, 100);
        let id1 = tracker.allocate(200, 20).expect("id1");
        let id2 = tracker.allocate(300, 30).expect("id2");

        assert_eq!(tracker.memory_usage(), 500);
        assert_eq!(tracker.compute_usage(), 50);

        tracker.release(id1);
        assert_eq!(tracker.memory_usage(), 300);
        assert_eq!(tracker.compute_usage(), 30);

        tracker.release(id2);
        assert_eq!(tracker.memory_usage(), 0);
    }

    #[test]
    fn test_resource_tracker_usage_percentage() {
        let mut tracker = ResourceTracker::new(1000, 100);
        tracker.allocate(500, 25).expect("allocate");

        let (mem_pct, compute_pct) = tracker.usage_percentage();
        assert!((mem_pct - 50.0).abs() < 0.1);
        assert!((compute_pct - 25.0).abs() < 0.1);
    }

    #[test]
    fn test_resource_tracker_usage_percentage_zero_capacity() {
        let tracker = ResourceTracker::new(0, 0);
        let (mem_pct, compute_pct) = tracker.usage_percentage();
        assert!((mem_pct - 0.0).abs() < 0.1);
        assert!((compute_pct - 0.0).abs() < 0.1);
    }
include!("batch_scheduling_token.rs");
}
