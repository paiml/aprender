
    // ========================================================================
    // Request Handling Tests
    // ========================================================================

    #[test]
    fn test_request_remaining_tokens_overflow_protection() {
        let mut request = SchedulerRequest::new(1, vec![1, 2, 3], 2);
        // Generated more than max_tokens (edge case)
        request.generated_tokens = vec![4, 5, 6, 7, 8];
        // Should not underflow, should return 0
        assert_eq!(request.remaining_tokens(), 0);
    }

    #[test]
    fn test_request_is_complete_with_empty_generated() {
        let request = SchedulerRequest::new(1, vec![1, 2, 3], 0);
        // max_tokens is 0, so complete immediately
        assert!(request.is_complete(999));
    }

    #[test]
    fn test_request_eos_at_first_position() {
        let mut request = SchedulerRequest::new(1, vec![1, 2, 3], 100);
        request.generated_tokens = vec![42]; // EOS token
        assert!(request.is_complete(42));
    }

    #[test]
    fn test_scheduler_request_clone() {
        let request = SchedulerRequest::new(1, vec![1, 2, 3], 10).with_priority(Priority::High);
        let cloned = request.clone();
        assert_eq!(cloned.request_id, 1);
        assert_eq!(cloned.priority, Priority::High);
        assert_eq!(cloned.input_ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_scheduler_request_debug() {
        let request = SchedulerRequest::new(1, vec![1], 10);
        let debug_str = format!("{:?}", request);
        assert!(debug_str.contains("SchedulerRequest"));
        assert!(debug_str.contains("request_id: 1"));
    }

    // ========================================================================
    // Scheduler Batch Management Tests
    // ========================================================================

    #[test]
    fn test_scheduler_empty_batch_schedule() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        // Schedule with empty queue
        let output = scheduler.schedule(&mut kv_cache, 0).expect("should work");
        assert!(output.is_empty());
        assert_eq!(output.total_tokens(), 0);
        assert!(output.scheduled_seq_ids.is_empty());
    }

    #[test]
    fn test_scheduler_batch_prefill_vs_decode_tokens() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        // Add request with 5 input tokens
        let request_id = scheduler
            .add_request(vec![1, 2, 3, 4, 5], 10)
            .expect("test");

        // First schedule - prefill phase
        let output1 = scheduler.schedule(&mut kv_cache, 0).expect("test");
        assert_eq!(output1.num_prefill_tokens, 5);
        assert_eq!(output1.num_decode_tokens, 0);

        // Simulate iteration
        let mut gen = HashMap::new();
        gen.insert(request_id, 100u32);
        scheduler.update_after_iteration(&gen);

        // Second schedule - decode phase
        let output2 = scheduler.schedule(&mut kv_cache, 0).expect("test");
        assert_eq!(output2.num_prefill_tokens, 0);
        assert_eq!(output2.num_decode_tokens, 1);
    }

    #[test]
    fn test_scheduler_multiple_requests_batch() {
        let mut scheduler = Scheduler::new(4, 1000);
        let mut kv_cache = PagedKvCache::new(200, 16, 8, 64);

        // Add 4 requests
        let ids: Vec<_> = (0..4)
            .map(|i| scheduler.add_request(vec![i as u32], 5).expect("test"))
            .collect();

        let output = scheduler.schedule(&mut kv_cache, 0).expect("test");
        assert_eq!(output.scheduled_request_ids.len(), 4);

        for id in &ids {
            assert!(output.scheduled_request_ids.contains(id));
        }
    }

    #[test]
    fn test_scheduler_completion_removes_from_running() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        let id1 = scheduler.add_request(vec![1], 5).expect("test");
        let id2 = scheduler.add_request(vec![2], 5).expect("test");
        let _ = scheduler.schedule(&mut kv_cache, 0);

        assert_eq!(scheduler.running_count(), 2);

        scheduler.complete_request(id1, &mut kv_cache);
        assert_eq!(scheduler.running_count(), 1);

        scheduler.complete_request(id2, &mut kv_cache);
        assert_eq!(scheduler.running_count(), 0);
    }

    #[test]
    fn test_scheduler_output_accessors() {
        let mut output = SchedulerOutput::default();
        output.scheduled_seq_ids = vec![SeqId::new(), SeqId::new()];
        output.scheduled_request_ids = vec![10, 20];
        output.preempted_seq_ids = vec![SeqId::new()];
        output.completed_request_ids = vec![5];
        output.num_prefill_tokens = 50;
        output.num_decode_tokens = 10;

        assert!(!output.is_empty());
        assert_eq!(output.total_tokens(), 60);
        assert_eq!(output.scheduled_seq_ids.len(), 2);
    }

    // ========================================================================
    // Error Handling Tests
    // ========================================================================

    #[test]
    fn test_scheduler_error_queue_full_message() {
        let err = SchedulerError::QueueFull { capacity: 256 };
        let msg = err.to_string();
        assert!(msg.contains("256"));
        assert!(msg.contains("full"));
    }

    #[test]
    fn test_scheduler_error_request_not_found_message() {
        let err = SchedulerError::RequestNotFound(12345);
        let msg = err.to_string();
        assert!(msg.contains("12345"));
        assert!(msg.contains("not found"));
    }

    #[test]
    fn test_scheduler_error_invalid_state_message() {
        let err = SchedulerError::InvalidState("corrupted scheduler".to_string());
        let msg = err.to_string();
        assert!(msg.contains("corrupted scheduler"));
    }

    #[test]
    fn test_scheduler_error_debug_format() {
        let err = SchedulerError::QueueFull { capacity: 100 };
        let debug = format!("{:?}", err);
        assert!(debug.contains("QueueFull"));
    }

    #[test]
    fn test_scheduler_queue_boundary_conditions() {
        let mut scheduler = Scheduler::new(32, 3);

        // Fill queue to exactly capacity
        let _ = scheduler.add_request(vec![1], 5).expect("first");
        let _ = scheduler.add_request(vec![2], 5).expect("second");
        let _ = scheduler.add_request(vec![3], 5).expect("third");

        // Next should fail
        let result = scheduler.add_request(vec![4], 5);
        assert!(matches!(
            result,
            Err(SchedulerError::QueueFull { capacity: 3 })
        ));
    }

    // ========================================================================
    // Slot State Machine Tests
    // ========================================================================

    #[test]
    fn test_slot_state_transitions() {
        let mut slot = Slot::new(0);

        // IDLE -> PROCESSING
        assert!(slot.is_idle());
        slot.assign(1, vec![10, 20], 5);
        assert_eq!(slot.state, SlotState::Processing);
        assert!(!slot.is_idle());

        // PROCESSING -> GENERATING
        slot.start_generation(10.5);
        assert_eq!(slot.state, SlotState::Generating);
        assert!(slot.is_generating());

        // GENERATING -> IDLE
        slot.finish();
        assert!(slot.is_idle());
    }

    #[test]
    fn test_slot_reuse_after_finish() {
        let mut slot = Slot::new(0);

        // First use
        slot.assign(1, vec![1, 2, 3], 10);
        slot.start_generation(5.0);
        slot.add_token(100);
        slot.add_token(200);
        slot.finish();

        // Verify clean state
        assert!(slot.is_idle());
        assert!(slot.request_id.is_none());

        // Second use
        slot.assign(2, vec![4, 5], 5);
        assert_eq!(slot.request_id, Some(2));
        assert_eq!(slot.input_tokens, vec![4, 5]);
        assert!(slot.generated_tokens.is_empty());
    }

    #[test]
    fn test_slot_complete_exactly_at_max_tokens() {
        let mut slot = Slot::new(0);
        slot.assign(1, vec![1], 3);
        slot.start_generation(1.0);

        slot.add_token(10);
        assert!(!slot.is_complete(999));
        slot.add_token(20);
        assert!(!slot.is_complete(999));
        slot.add_token(30);
        assert!(slot.is_complete(999)); // Exactly at max_tokens
    }

    // ========================================================================
    // SlotManager Tests
    // ========================================================================

    #[test]
    fn test_slot_manager_find_idle_slot_none() {
        let mut manager = SlotManager::new(2, 2048);

        // Fill all slots
        manager.assign_request(vec![1], 5);
        manager.assign_request(vec![2], 5);

        assert!(manager.find_idle_slot().is_none());
    }

    #[test]
    fn test_slot_manager_generating_slots_iterator() {
        let mut manager = SlotManager::new(4, 2048);

        manager.assign_request(vec![1], 5);
        manager.assign_request(vec![2], 5);

        // Start generation on slot 0 only
        manager.get_slot_mut(0).unwrap().start_generation(1.0);

        let generating: Vec<_> = manager.generating_slots().collect();
        assert_eq!(generating.len(), 1);
        assert_eq!(generating[0].id, 0);
    }

    #[test]
    fn test_slot_manager_empty_aggregate_tokens() {
        let manager = SlotManager::new(4, 2048);
        assert_eq!(manager.aggregate_tokens_per_second(), 0.0);
    }

    // ========================================================================
    // MicroBatch Edge Cases
    // ========================================================================

    #[test]
    fn test_micro_batch_empty_after_clear() {
        let mut batch = MicroBatch::with_capacity(100);
        batch.add_token(BatchToken::new(1, 0, 0, true));
        batch.add_token(BatchToken::new(2, 0, 1, true));

        assert!(!batch.is_empty());
        batch.clear();
        assert!(batch.is_empty());
        assert!(batch.is_decode()); // Default type after clear
    }

    #[test]
    fn test_micro_batch_decode_only() {
        let mut batch = MicroBatch::new();
        batch.add_token(BatchToken::new(1, 0, 5, false));
        batch.add_token(BatchToken::new(2, 1, 10, false));

        assert!(batch.is_decode());
        assert!(!batch.is_prefill());
        assert!(!batch.is_mixed());
    }

    #[test]
    fn test_micro_batch_multi_sequence_tracking() {
        let mut batch = MicroBatch::new();
        batch.add_token(BatchToken::new(1, 0, 0, true));
        batch.add_token(BatchToken::new(2, 0, 1, true));
        batch.add_token(BatchToken::new(3, 1, 0, true));
        batch.add_token(BatchToken::new(4, 2, 0, true));

        assert_eq!(batch.num_sequences(), 3);
        assert!(batch.seq_indices.contains(&0));
        assert!(batch.seq_indices.contains(&1));
        assert!(batch.seq_indices.contains(&2));
    }

    // ========================================================================
    // SequenceBatch Tests
    // ========================================================================

    #[test]
    fn test_sequence_batch_iterators() {
        let mut batch = SequenceBatch::new(8);

        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1));
        batch.add_sequence(SequenceBatchEntry::new(1, 1, 2).decoding());
        batch.add_sequence(SequenceBatchEntry::new(2, 2, 3));
        batch.add_sequence(SequenceBatchEntry::new(3, 3, 4).decoding());

        let prefill: Vec<_> = batch.prefill_sequences().collect();
        assert_eq!(prefill.len(), 2);

        let decode: Vec<_> = batch.decode_sequences().collect();
        assert_eq!(decode.len(), 2);
    }

    #[test]
    fn test_sequence_batch_get_and_get_mut() {
        let mut batch = SequenceBatch::new(8);
        batch.add_sequence(SequenceBatchEntry::new(5, 0, 100));

        // Test get
        let entry = batch.get(5);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().request_id, 100);

        // Test get_mut
        if let Some(e) = batch.get_mut(5) {
            e.position = 42;
        }
        assert_eq!(batch.get(5).unwrap().position, 42);

        // Test nonexistent
        assert!(batch.get(999).is_none());
        assert!(batch.get_mut(999).is_none());
    }

    // ========================================================================
    // BatchScheduler Tests
    // ========================================================================

    #[test]
    fn test_batch_scheduler_ubatch_max_tokens_limit() {
        let config = BatchConfig::default().with_max_tokens(5);
        let mut scheduler = BatchScheduler::with_config(config);

        // Add sequence with more tokens than limit
        scheduler.add_sequence(0, 1, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let ubatch = scheduler.create_ubatch();
        // Should be limited to max_ubatch_tokens
        assert!(ubatch.len() <= 5);
    }

    #[test]
    fn test_batch_scheduler_mixed_prefill_decode_pure_decode_preference() {
        let config = BatchConfig {
            prefer_pure_decode: true,
            ..Default::default()
        };
        let mut scheduler = BatchScheduler::with_config(config);

        // Add prefill sequence
        let seq1 = scheduler.add_sequence(0, 1, vec![1, 2, 3]).unwrap();

        // Add decode sequence
        let seq2 = scheduler.add_sequence(1, 2, vec![10, 20]).unwrap();
        scheduler.start_decode(seq2, 2);

        // First ubatch should be pure prefill (due to prefer_pure_decode)
        let ubatch1 = scheduler.create_ubatch();
        assert!(ubatch1.is_prefill());

        // Transition seq1 to decode
        scheduler.start_decode(seq1, 3);

        // Now should be pure decode
        let ubatch2 = scheduler.create_ubatch();
        assert!(ubatch2.is_decode());
    }

    #[test]
    fn test_batch_scheduler_stats_accumulation() {
        let mut scheduler = BatchScheduler::new();

        scheduler.add_sequence(0, 1, vec![1, 2, 3]);
        scheduler.create_ubatch();

        scheduler.add_sequence(1, 2, vec![4, 5]);
        scheduler.create_ubatch();

        let stats = scheduler.stats();
        assert_eq!(stats.ubatches_created, 2);
        assert!(stats.tokens_processed > 0);
    }

    // ========================================================================
    // DynamicPriorityScheduler Edge Cases
    // ========================================================================

    #[test]
    fn test_dynamic_scheduler_empty_schedule() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);
        let batch = scheduler.schedule(10);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_dynamic_scheduler_complete_nonexistent() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);
        let result = scheduler.complete_request(99999);
        assert!(result.is_none());
    }

    #[test]
    fn test_dynamic_scheduler_drop_expired_empty() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);
        let dropped = scheduler.drop_expired();
        assert!(dropped.is_empty());
    }
