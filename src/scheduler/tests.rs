#[cfg(test)]
mod tests {
    use crate::scheduler::*;

    // --- DynamicRequest with No Hard Deadline ---

    #[test]
    fn test_cov_dynamic_request_no_hard_deadline_not_expired() {
        let mut request = DynamicRequest::new(0, vec![1], 10);
        request.deadline = Some(Deadline::with_target(0)); // Target only, no hard deadline

        std::thread::sleep(std::time::Duration::from_millis(1));

        // Should not be expired (no hard deadline)
        assert!(!request.is_expired());
    }

    // --- BatchScheduler Recording Stats ---

    #[test]
    fn test_cov_batch_scheduler_record_ubatch_stats() {
        let mut scheduler = BatchScheduler::new();

        // Add sequence and create multiple ubatches
        scheduler.add_sequence(0, 1, vec![10, 20, 30]);
        let _ = scheduler.create_ubatch();

        let seq_idx = scheduler.add_sequence(1, 2, vec![40, 50]).expect("add 2");
        scheduler.start_decode(seq_idx, 2);
        let _ = scheduler.create_ubatch();

        let stats = scheduler.stats();
        assert!(stats.ubatches_created >= 2);
        assert!(stats.avg_ubatch_size > 0.0);
    }

    // --- Priority Types Serialization ---

    #[test]
    fn test_cov_priority_serialization() {
        let priorities = [
            Priority::Low,
            Priority::Normal,
            Priority::High,
            Priority::Critical,
        ];
        for p in &priorities {
            let json = serde_json::to_string(p).expect("serialize");
            let parsed: Priority = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(*p, parsed);
        }
    }

    // --- SequenceState Serialization ---

    #[test]
    fn test_cov_sequence_state_serialization() {
        let states = [
            SequenceState::Waiting,
            SequenceState::Running,
            SequenceState::Preempted,
            SequenceState::Completed,
            SequenceState::Failed,
        ];
        for s in &states {
            let json = serde_json::to_string(s).expect("serialize");
            let parsed: SequenceState = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(*s, parsed);
        }
    }
include!("tests_priority_ordering_default.rs");
include!("tests_slot_manager_batch.rs");
include!("tests_dynamic_scheduler.rs");
include!("tests_chunked_prefill.rs");
include!("tests_deep_scov.rs");
include!("tests_cov_batch.rs");
}
