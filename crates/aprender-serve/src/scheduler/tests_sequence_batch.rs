//! Scheduler Tests Part 03 - Deep Coverage
//!
//! Tests for DynamicPriorityScheduler, BatchConfig builders, Deadline handling,
//! and preemption logic. Targets uncovered branches in scheduler/mod.rs.
//! Refs PMAT-802: Protocol T-COV-95

#[cfg(test)]
mod tests {
    use crate::paged_kv::PagedKvCache;
    use crate::scheduler::*;
    use std::collections::HashMap;

    #[test]
    fn test_sequence_batch_num_prefill_decode() {
        let mut batch = SequenceBatch::new(4);
        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1)); // prefill
        batch.add_sequence(SequenceBatchEntry::new(1, 1, 2).decoding()); // decode
        batch.add_sequence(SequenceBatchEntry::new(2, 2, 3)); // prefill

        assert_eq!(batch.num_prefill(), 2);
        assert_eq!(batch.num_decode(), 1);
    }

    #[test]
    fn test_sequence_batch_clear() {
        let mut batch = SequenceBatch::new(4);
        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1));
        batch.add_sequence(SequenceBatchEntry::new(1, 1, 2));

        batch.clear();
        assert!(batch.is_empty());
        assert!((batch.utilization - 0.0).abs() < 0.001);
    }

    // ========================================================================
    // BatchScheduler Tests
    // ========================================================================

    #[test]
    fn test_batch_scheduler_default() {
        let scheduler = BatchScheduler::default();
        assert_eq!(scheduler.num_sequences(), 0);
    }

    #[test]
    fn test_batch_scheduler_complete_sequence() {
        let mut scheduler = BatchScheduler::new();
        let seq_idx = scheduler.add_sequence(0, 1, vec![1, 2, 3]).unwrap();

        assert_eq!(scheduler.num_sequences(), 1);

        let entry = scheduler.complete_sequence(seq_idx);
        assert!(entry.is_some());
        assert_eq!(scheduler.num_sequences(), 0);
    }

    #[test]
    fn test_batch_scheduler_complete_nonexistent() {
        let mut scheduler = BatchScheduler::new();
        let entry = scheduler.complete_sequence(999);
        assert!(entry.is_none());
    }

    #[test]
    fn test_batch_scheduler_start_decode() {
        let mut scheduler = BatchScheduler::new();
        let seq_idx = scheduler.add_sequence(0, 1, vec![1, 2, 3]).unwrap();

        assert!(scheduler.start_decode(seq_idx, 3));

        let sbatch = scheduler.sbatch();
        let entry = sbatch.get(seq_idx).unwrap();
        assert!(!entry.is_prefill);
        assert_eq!(entry.position, 3);
        assert!(entry.tokens.is_empty()); // cleared after transitioning
    }

    #[test]
    fn test_batch_scheduler_start_decode_nonexistent() {
        let mut scheduler = BatchScheduler::new();
        assert!(!scheduler.start_decode(999, 0));
    }

    #[test]
    fn test_batch_scheduler_add_when_full() {
        let config = BatchConfig::default().with_max_sequences(2);
        let mut scheduler = BatchScheduler::with_config(config);

        assert!(scheduler.add_sequence(0, 1, vec![1]).is_some());
        assert!(scheduler.add_sequence(1, 2, vec![2]).is_some());
        assert!(scheduler.add_sequence(2, 3, vec![3]).is_none()); // Full
    }

    #[test]
    fn test_batch_scheduler_has_capacity() {
        let config = BatchConfig::default().with_max_sequences(2);
        let mut scheduler = BatchScheduler::with_config(config);

        assert!(scheduler.has_capacity());
        scheduler.add_sequence(0, 1, vec![1]);
        assert!(scheduler.has_capacity());
        scheduler.add_sequence(1, 2, vec![2]);
        assert!(!scheduler.has_capacity());
    }

    #[test]
    fn test_batch_scheduler_utilization() {
        let config = BatchConfig::default().with_max_sequences(4);
        let mut scheduler = BatchScheduler::with_config(config);

        assert!((scheduler.utilization() - 0.0).abs() < 0.001);

        scheduler.add_sequence(0, 1, vec![1]);
        assert!((scheduler.utilization() - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_batch_scheduler_create_empty_ubatch() {
        let mut scheduler = BatchScheduler::new();
        let ubatch = scheduler.create_ubatch();
        assert!(ubatch.is_empty());
    }

    #[test]
    fn test_batch_scheduler_decode_ubatch() {
        let mut scheduler = BatchScheduler::new();
        let seq1 = scheduler.add_sequence(0, 1, vec![1, 2, 3]).unwrap();
        let seq2 = scheduler.add_sequence(1, 2, vec![4, 5]).unwrap();

        // Transition both to decode
        scheduler.start_decode(seq1, 3);
        scheduler.start_decode(seq2, 2);

        let ubatch = scheduler.create_ubatch();
        assert!(ubatch.is_decode());
        assert_eq!(ubatch.n_decode_tokens, 2);
        assert_eq!(ubatch.n_prompt_tokens, 0);
    }

    #[test]
    fn test_batch_scheduler_config_accessor() {
        let config = BatchConfig::default().with_max_tokens(256);
        let scheduler = BatchScheduler::with_config(config);
        assert_eq!(scheduler.config().max_ubatch_tokens, 256);
    }

    // ========================================================================
    // BatchToken Tests
    // ========================================================================

    #[test]
    fn test_batch_token_new() {
        let token = BatchToken::new(42, 1, 5, true);
        assert_eq!(token.token_id, 42);
        assert_eq!(token.seq_idx, 1);
        assert_eq!(token.seq_pos, 5);
        assert!(token.is_prompt);
    }

    #[test]
    fn test_batch_token_clone() {
        let token = BatchToken::new(100, 2, 10, false);
        let cloned = token;
        assert_eq!(cloned.token_id, 100);
        assert_eq!(cloned.seq_idx, 2);
        assert_eq!(cloned.seq_pos, 10);
        assert!(!cloned.is_prompt);
    }

    // ========================================================================
    // MicroBatch Tests
    // ========================================================================

    #[test]
    fn test_micro_batch_token_ids() {
        let mut batch = MicroBatch::new();
        batch.add_token(BatchToken::new(10, 0, 0, true));
        batch.add_token(BatchToken::new(20, 0, 1, true));
        batch.add_token(BatchToken::new(30, 0, 2, true));

        assert_eq!(batch.token_ids(), vec![10, 20, 30]);
    }

    #[test]
    fn test_micro_batch_positions() {
        let mut batch = MicroBatch::new();
        batch.add_token(BatchToken::new(1, 0, 0, true));
        batch.add_token(BatchToken::new(2, 0, 1, true));
        batch.add_token(BatchToken::new(3, 1, 5, false)); // Different sequence

        assert_eq!(batch.positions(), vec![0, 1, 5]);
    }

    #[test]
    fn test_micro_batch_max_seq_len_tracking() {
        let mut batch = MicroBatch::new();
        batch.add_token(BatchToken::new(1, 0, 5, true));
        assert_eq!(batch.max_seq_len, 6); // pos 5 + 1

        batch.add_token(BatchToken::new(2, 0, 10, true));
        assert_eq!(batch.max_seq_len, 11);

        batch.add_token(BatchToken::new(3, 0, 3, true)); // Lower pos
        assert_eq!(batch.max_seq_len, 11); // Still 11
    }

    // ========================================================================
    // BatchStats Tests
    // ========================================================================

    #[test]
    fn test_batch_stats_default() {
        let stats = BatchStats::default();
        assert_eq!(stats.ubatches_created, 0);
        assert_eq!(stats.sbatches_created, 0);
        assert_eq!(stats.tokens_processed, 0);
        assert_eq!(stats.prefill_tokens, 0);
        assert_eq!(stats.decode_tokens, 0);
        assert!((stats.avg_ubatch_size - 0.0).abs() < 0.001);
        assert!((stats.avg_sbatch_size - 0.0).abs() < 0.001);
    }

    // ========================================================================
    // SlotManager Additional Tests
    // ========================================================================

    #[test]
    fn test_slot_manager_batch_slots() {
        let mut manager = SlotManager::new(4, 2048);

        // Assign and start generation on some slots
        manager.assign_request(vec![1], 5);
        manager.assign_request(vec![2], 5);
        manager.assign_request(vec![3], 5);

        // Start generation on slots 0 and 2
        manager.get_slot_mut(0).unwrap().start_generation(1.0);
        manager.get_slot_mut(2).unwrap().start_generation(1.0);

        let batch_slots = manager.batch_slots();
        assert_eq!(batch_slots.len(), 2);
        assert!(batch_slots.contains(&0));
        assert!(batch_slots.contains(&2));
    }

    #[test]
    fn test_slot_manager_active_slots() {
        let mut manager = SlotManager::new(3, 2048);

        // All idle initially
        assert_eq!(manager.active_slots().count(), 0);

        manager.assign_request(vec![1], 5);
        manager.assign_request(vec![2], 5);

        assert_eq!(manager.active_slots().count(), 2);
    }

    #[test]
    fn test_slot_manager_empty_utilization() {
        let manager = SlotManager::new(0, 2048);
        assert!((manager.utilization() - 0.0).abs() < 0.001);
    }

    // ========================================================================
    // Slot Additional Tests
    // ========================================================================

    #[test]
    fn test_slot_tokens_per_second_no_generation_time() {
        let slot = Slot::new(0);
        assert!((slot.tokens_per_second() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_slot_debug() {
        let slot = Slot::new(42);
        let debug_str = format!("{:?}", slot);
        assert!(debug_str.contains("Slot"));
        assert!(debug_str.contains("42"));
    }

    #[test]
    fn test_slot_clone() {
        let mut slot = Slot::new(5);
        slot.assign(100, vec![1, 2, 3], 10);

        let cloned = slot.clone();
        assert_eq!(cloned.id, 5);
        assert_eq!(cloned.request_id, Some(100));
        assert_eq!(cloned.input_tokens, vec![1, 2, 3]);
    }

    // ========================================================================
    // Scheduler Preemption Tests
    // ========================================================================

    #[test]
    fn test_scheduler_with_max_tokens() {
        let scheduler = Scheduler::new(32, 1000).with_max_tokens(4096);
        // Just verify it compiles and doesn't panic
        assert!(scheduler.running_count() == 0);
    }

    #[test]
    fn test_scheduler_get_nonexistent_request() {
        let scheduler = Scheduler::new(32, 1000);
        assert!(scheduler.get_request(999).is_none());
    }

    #[test]
    fn test_scheduler_preempted_count() {
        let scheduler = Scheduler::new(32, 1000);
        assert_eq!(scheduler.preempted_count(), 0);
    }

    #[test]
    fn test_scheduler_update_after_iteration_nonexistent() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut gen = HashMap::new();
        gen.insert(99999u64, 42u32);

        // Should not panic with nonexistent request
        scheduler.update_after_iteration(&gen);
    }

    #[test]
    fn test_scheduler_complete_nonexistent_request() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        // Should not panic with nonexistent request
        scheduler.complete_request(99999, &mut kv_cache);
    }
include!("tests_deadline_default.rs");
}
