//! Scheduler Core Types
//!
//! Common types for the continuous batching scheduler.
//! Extracted from scheduler/mod.rs (PMAT-802).

use serde::{Deserialize, Serialize};

/// Request priority level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Priority {
    /// Low priority (background tasks)
    Low = 0,
    /// Normal priority (standard requests)
    Normal = 1,
    /// High priority (interactive requests)
    High = 2,
    /// Critical priority (system requests)
    Critical = 3,
}

impl Default for Priority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Sequence state in the scheduler
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SequenceState {
    /// Waiting in queue
    Waiting,
    /// Currently running
    Running,
    /// Preempted (swapped out)
    Preempted,
    /// Completed (EOS or max_tokens)
    Completed,
    /// Failed (error during generation)
    Failed,
}

/// Scheduler statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchedulerStats {
    /// Total requests received
    pub total_requests: u64,
    /// Total requests completed
    pub completed_requests: u64,
    /// Total preemptions
    pub preemptions: u64,
    /// Average queue wait time (ms)
    pub avg_wait_time_ms: f64,
    /// Average time-to-first-token (ms)
    pub avg_ttft_ms: f64,
    /// Current queue depth
    pub queue_depth: usize,
    /// Current running count
    pub running_count: usize,
}

// ============================================================================
// Tests (T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Priority Tests
    // =========================================================================

    #[test]
    fn test_priority_default() {
        let p: Priority = Priority::default();
        assert_eq!(p, Priority::Normal);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Low < Priority::Normal);
        assert!(Priority::Normal < Priority::High);
        assert!(Priority::High < Priority::Critical);
    }

    #[test]
    fn test_priority_ordering_reverse() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn test_priority_equality() {
        assert_eq!(Priority::Low, Priority::Low);
        assert_eq!(Priority::Normal, Priority::Normal);
        assert_eq!(Priority::High, Priority::High);
        assert_eq!(Priority::Critical, Priority::Critical);
    }

    #[test]
    fn test_priority_not_equal() {
        assert_ne!(Priority::Low, Priority::Normal);
        assert_ne!(Priority::Normal, Priority::High);
        assert_ne!(Priority::High, Priority::Critical);
    }

    #[test]
    fn test_priority_clone() {
        let p = Priority::High;
        let cloned = p;
        assert_eq!(p, cloned);
    }

    #[test]
    fn test_priority_copy() {
        let p = Priority::Critical;
        let copied: Priority = p;
        assert_eq!(p, copied);
    }

    #[test]
    fn test_priority_debug() {
        let debug_str = format!("{:?}", Priority::Normal);
        assert!(debug_str.contains("Normal"));
    }

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
    fn test_priority_serde_roundtrip() {
        for priority in [
            Priority::Low,
            Priority::Normal,
            Priority::High,
            Priority::Critical,
        ] {
            let json = serde_json::to_string(&priority).expect("serialize");
            let restored: Priority = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(priority, restored);
        }
    }

    // =========================================================================
    // SequenceState Tests
    // =========================================================================

    #[test]
    fn test_sequence_state_equality() {
        assert_eq!(SequenceState::Waiting, SequenceState::Waiting);
        assert_eq!(SequenceState::Running, SequenceState::Running);
        assert_eq!(SequenceState::Preempted, SequenceState::Preempted);
        assert_eq!(SequenceState::Completed, SequenceState::Completed);
        assert_eq!(SequenceState::Failed, SequenceState::Failed);
    }

    #[test]
    fn test_sequence_state_not_equal() {
        assert_ne!(SequenceState::Waiting, SequenceState::Running);
        assert_ne!(SequenceState::Running, SequenceState::Completed);
        assert_ne!(SequenceState::Preempted, SequenceState::Failed);
    }

    #[test]
    fn test_sequence_state_clone() {
        let state = SequenceState::Running;
        let cloned = state;
        assert_eq!(state, cloned);
    }

    #[test]
    fn test_sequence_state_copy() {
        let state = SequenceState::Completed;
        let copied: SequenceState = state;
        assert_eq!(state, copied);
    }

    #[test]
    fn test_sequence_state_debug() {
        let debug_str = format!("{:?}", SequenceState::Preempted);
        assert!(debug_str.contains("Preempted"));
    }

    #[test]
    fn test_sequence_state_serde_roundtrip() {
        for state in [
            SequenceState::Waiting,
            SequenceState::Running,
            SequenceState::Preempted,
            SequenceState::Completed,
            SequenceState::Failed,
        ] {
            let json = serde_json::to_string(&state).expect("serialize");
            let restored: SequenceState = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(state, restored);
        }
    }

    // =========================================================================
    // SchedulerStats Tests
    // =========================================================================

    #[test]
    fn test_scheduler_stats_default() {
        let stats = SchedulerStats::default();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.completed_requests, 0);
        assert_eq!(stats.preemptions, 0);
        assert!((stats.avg_wait_time_ms - 0.0).abs() < f64::EPSILON);
        assert!((stats.avg_ttft_ms - 0.0).abs() < f64::EPSILON);
        assert_eq!(stats.queue_depth, 0);
        assert_eq!(stats.running_count, 0);
    }

    #[test]
    fn test_scheduler_stats_clone() {
        let stats = SchedulerStats {
            total_requests: 100,
            completed_requests: 90,
            preemptions: 5,
            avg_wait_time_ms: 10.5,
            avg_ttft_ms: 50.0,
            queue_depth: 10,
            running_count: 5,
        };
        let cloned = stats.clone();
        assert_eq!(stats.total_requests, cloned.total_requests);
        assert_eq!(stats.completed_requests, cloned.completed_requests);
        assert_eq!(stats.preemptions, cloned.preemptions);
        assert!((stats.avg_wait_time_ms - cloned.avg_wait_time_ms).abs() < f64::EPSILON);
        assert!((stats.avg_ttft_ms - cloned.avg_ttft_ms).abs() < f64::EPSILON);
        assert_eq!(stats.queue_depth, cloned.queue_depth);
        assert_eq!(stats.running_count, cloned.running_count);
    }

    #[test]
    fn test_scheduler_stats_debug() {
        let stats = SchedulerStats {
            total_requests: 42,
            ..Default::default()
        };
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("total_requests: 42"));
    }

    #[test]
    fn test_scheduler_stats_serde_roundtrip() {
        let stats = SchedulerStats {
            total_requests: 1000,
            completed_requests: 950,
            preemptions: 25,
            avg_wait_time_ms: 12.5,
            avg_ttft_ms: 75.3,
            queue_depth: 15,
            running_count: 8,
        };
        let json = serde_json::to_string(&stats).expect("serialize");
        let restored: SchedulerStats = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(stats.total_requests, restored.total_requests);
        assert_eq!(stats.completed_requests, restored.completed_requests);
        assert_eq!(stats.preemptions, restored.preemptions);
        assert!((stats.avg_wait_time_ms - restored.avg_wait_time_ms).abs() < f64::EPSILON);
        assert!((stats.avg_ttft_ms - restored.avg_ttft_ms).abs() < f64::EPSILON);
        assert_eq!(stats.queue_depth, restored.queue_depth);
        assert_eq!(stats.running_count, restored.running_count);
    }

    #[test]
    fn test_scheduler_stats_json_fields() {
        let stats = SchedulerStats {
            total_requests: 1,
            completed_requests: 2,
            preemptions: 3,
            avg_wait_time_ms: 4.0,
            avg_ttft_ms: 5.0,
            queue_depth: 6,
            running_count: 7,
        };
        let json = serde_json::to_string(&stats).expect("serialize");
        assert!(json.contains("total_requests"));
        assert!(json.contains("completed_requests"));
        assert!(json.contains("preemptions"));
        assert!(json.contains("avg_wait_time_ms"));
        assert!(json.contains("avg_ttft_ms"));
        assert!(json.contains("queue_depth"));
        assert!(json.contains("running_count"));
    }
}
