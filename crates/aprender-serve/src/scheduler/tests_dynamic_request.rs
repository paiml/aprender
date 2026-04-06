//! Scheduler Tests Part 02 - Coverage Improvements
//!
//! Additional tests for scheduler request handling, batch management, and error cases.
//! Extracted to comply with pmat file health rules (<400 lines).

#[cfg(test)]
mod tests {
    use crate::paged_kv::{PagedKvCache, SeqId};
    use crate::scheduler::*;
    use std::collections::HashMap;

    #[test]
    fn test_dynamic_request_urgency_calculation() {
        let request =
            DynamicRequest::new(0, vec![1], 10).with_deadline(Deadline::with_target(1000));

        // Urgency should be close to 0 for fresh request
        let urgency = request.urgency_score();
        assert!(urgency < 0.1);
    }

    #[test]
    fn test_dynamic_scheduler_waiting_vs_running() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);

        scheduler.add_simple_request(vec![1], 5);
        scheduler.add_simple_request(vec![2], 5);
        scheduler.add_simple_request(vec![3], 5);

        assert_eq!(scheduler.waiting_count(), 3);
        assert_eq!(scheduler.running_count(), 0);

        // Schedule 2
        scheduler.schedule(2);

        assert_eq!(scheduler.waiting_count(), 1);
        assert_eq!(scheduler.running_count(), 2);
    }

    // ========================================================================
    // Priority Tests
    // ========================================================================

    #[test]
    fn test_priority_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Priority::Low);
        set.insert(Priority::Normal);
        set.insert(Priority::High);
        set.insert(Priority::Critical);
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn test_priority_clone_copy() {
        let p1 = Priority::High;
        let p2 = p1; // Copy
        let p3 = p1;
        assert_eq!(p1, p2);
        assert_eq!(p2, p3);
    }

    // ========================================================================
    // Serialization Round-Trip Tests
    // ========================================================================

    #[test]
    fn test_slot_state_serde_roundtrip() {
        for state in [
            SlotState::Idle,
            SlotState::Processing,
            SlotState::Generating,
        ] {
            let json = serde_json::to_string(&state).unwrap();
            let parsed: SlotState = serde_json::from_str(&json).unwrap();
            assert_eq!(state, parsed);
        }
    }

    #[test]
    fn test_batch_type_serde_roundtrip() {
        for bt in [BatchType::Prefill, BatchType::Decode, BatchType::Mixed] {
            let json = serde_json::to_string(&bt).unwrap();
            let parsed: BatchType = serde_json::from_str(&json).unwrap();
            assert_eq!(bt, parsed);
        }
    }

    #[test]
    fn test_scheduler_stats_default_values() {
        let stats = SchedulerStats::default();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.completed_requests, 0);
        assert_eq!(stats.preemptions, 0);
        assert_eq!(stats.avg_wait_time_ms, 0.0);
        assert_eq!(stats.queue_depth, 0);
    }
include!("tests_request_remaining.rs");
}
