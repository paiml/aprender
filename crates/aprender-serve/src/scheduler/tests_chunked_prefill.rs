
    #[test]
    fn test_chunked_prefill_scheduler_prefix_cache_hit() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());

        scheduler.record_prefix_cache_hit(100);
        assert_eq!(scheduler.stats().prefix_cache_hits, 100);

        scheduler.record_prefix_cache_hit(50);
        assert_eq!(scheduler.stats().prefix_cache_hits, 150);
    }

    #[test]
    fn test_chunked_prefill_scheduler_disabled() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::disabled());

        // Even long prompts bypass chunking when disabled
        let (_, use_chunking) = scheduler.submit(10000);
        assert!(!use_chunking);
        assert_eq!(scheduler.stats().bypassed_sequences, 1);
        assert_eq!(scheduler.stats().chunked_sequences, 0);
    }

    #[test]
    fn test_chunked_prefill_scheduler_default() {
        let scheduler = ChunkedPrefillScheduler::default();
        assert!(scheduler.config().enabled);
    }

    // ========================================================================
    // Deep Coverage Tests (_deep_scov_ prefix)
    // ========================================================================

    // --- Error handling and edge cases ---

    #[test]
    fn test_deep_scov_scheduler_error_cache_error_conversion() {
        // Test SchedulerError::CacheError conversion from PagedCacheError
        let cache_err = PagedCacheError::OutOfMemory {
            needed: 10,
            available: 5,
        };
        let scheduler_err: SchedulerError = cache_err.into();
        let msg = scheduler_err.to_string();
        assert!(msg.contains("KV cache error"));
    }

    #[test]
    fn test_deep_scov_priority_entry_equality() {
        // Test PriorityEntry PartialEq implementation
        let now = Instant::now();
        let entry1 = PriorityEntry {
            priority: Priority::High,
            arrival_time: now,
            request_id: 42,
        };
        let entry2 = PriorityEntry {
            priority: Priority::Low, // Different priority
            arrival_time: now,
            request_id: 42, // Same request_id
        };
        let entry3 = PriorityEntry {
            priority: Priority::High,
            arrival_time: now,
            request_id: 43, // Different request_id
        };
        // PartialEq compares only request_id
        assert_eq!(entry1, entry2);
        assert_ne!(entry1, entry3);
    }

    #[test]
    fn test_deep_scov_priority_entry_ordering() {
        // Test PriorityEntry Ord implementation - higher priority first
        let now = Instant::now();
        let high = PriorityEntry {
            priority: Priority::High,
            arrival_time: now,
            request_id: 1,
        };
        let low = PriorityEntry {
            priority: Priority::Low,
            arrival_time: now,
            request_id: 2,
        };
        // High priority should be "greater" (higher in heap)
        assert!(high > low);
    }

    #[test]
    fn test_deep_scov_priority_entry_ordering_same_priority_earlier_first() {
        // Test that earlier arrival time has higher priority when priorities are equal
        use std::thread::sleep;
        use std::time::Duration;

        let earlier = Instant::now();
        sleep(Duration::from_millis(1));
        let later = Instant::now();

        let entry_earlier = PriorityEntry {
            priority: Priority::Normal,
            arrival_time: earlier,
            request_id: 1,
        };
        let entry_later = PriorityEntry {
            priority: Priority::Normal,
            arrival_time: later,
            request_id: 2,
        };
        // Earlier arrival should have higher priority
        assert!(entry_earlier > entry_later);
    }

    #[test]
    fn test_deep_scov_scheduler_with_max_tokens() {
        let scheduler = Scheduler::new(32, 1000).with_max_tokens(4096);
        assert_eq!(scheduler.max_tokens_per_batch, 4096);
    }

    #[test]
    fn test_deep_scov_request_wait_time() {
        let request = SchedulerRequest::new(1, vec![1], 10);
        // Wait time should be very small immediately after creation
        let wait = request.wait_time();
        assert!(wait.as_millis() < 100);
    }

    #[test]
    fn test_deep_scov_scheduler_complete_nonexistent_request() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);
        // Completing a non-existent request should not panic
        scheduler.complete_request(999, &mut kv_cache);
        assert_eq!(scheduler.stats().completed_requests, 0);
    }

    #[test]
    fn test_deep_scov_scheduler_get_nonexistent_request() {
        let scheduler = Scheduler::new(32, 1000);
        assert!(scheduler.get_request(999).is_none());
    }

    #[test]
    fn test_deep_scov_scheduler_preemption_flow() {
        // Test preemption when high priority request arrives while low priority is running
        let mut scheduler = Scheduler::new(1, 1000); // Single slot
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        // Schedule low priority request
        let low_id = scheduler
            .add_request_with_priority(vec![1, 2, 3], 10, Priority::Low)
            .expect("add low");
        let _ = scheduler.schedule(&mut kv_cache, 0).expect("schedule");
        assert_eq!(scheduler.running_count(), 1);

        // Add high priority request
        let _high_id = scheduler
            .add_request_with_priority(vec![4, 5], 10, Priority::Critical)
            .expect("add high");

        // Schedule again - should preempt low priority
        let output = scheduler.schedule(&mut kv_cache, 0).expect("schedule 2");

        // Low priority request should be preempted
        let low_req = scheduler.get_request(low_id).expect("get low");
        assert!(
            low_req.state == SequenceState::Preempted
                || low_req.state == SequenceState::Waiting
                || !output.preempted_seq_ids.is_empty()
        );
    }

    #[test]
    fn test_deep_scov_scheduler_check_completions_eos() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        let request_id = scheduler.add_request(vec![1], 100).expect("add");
        let _ = scheduler.schedule(&mut kv_cache, 0).expect("schedule");

        // Add EOS token
        let mut generated = HashMap::new();
        generated.insert(request_id, 42u32); // EOS = 42
        scheduler.update_after_iteration(&generated);

        // Schedule again to trigger completion check
        let output = scheduler.schedule(&mut kv_cache, 42).expect("schedule 2");
        assert!(output.completed_request_ids.contains(&request_id));
    }

    #[test]
    fn test_deep_scov_scheduler_output_default_values() {
        let output = SchedulerOutput::default();
        assert!(output.scheduled_seq_ids.is_empty());
        assert!(output.scheduled_request_ids.is_empty());
        assert!(output.preempted_seq_ids.is_empty());
        assert!(output.completed_request_ids.is_empty());
        assert_eq!(output.num_prefill_tokens, 0);
        assert_eq!(output.num_decode_tokens, 0);
        assert_eq!(output.total_tokens(), 0);
        assert!(output.is_empty());
    }

    #[test]
    fn test_deep_scov_scheduler_multiple_iterations() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        let request_id = scheduler.add_request(vec![1, 2, 3], 5).expect("add");

        // First schedule (prefill)
        let output1 = scheduler.schedule(&mut kv_cache, 0).expect("schedule 1");
        assert_eq!(output1.num_prefill_tokens, 3);

        // Update after iteration
        let mut generated = HashMap::new();
        generated.insert(request_id, 10u32);
        scheduler.update_after_iteration(&generated);

        // Second schedule (decode)
        let output2 = scheduler.schedule(&mut kv_cache, 0).expect("schedule 2");
        assert_eq!(output2.num_decode_tokens, 1);
    }

    // --- Slot edge cases ---

    #[test]
    fn test_deep_scov_slot_tokens_per_second_zero_time() {
        let slot = Slot::new(0);
        // No generation time means 0 tok/s
        assert_eq!(slot.tokens_per_second(), 0.0);
    }

    #[test]
    fn test_deep_scov_slot_is_complete_empty_generated() {
        let mut slot = Slot::new(0);
        slot.assign(1, vec![1], 10);
        slot.start_generation(1.0);
        // No tokens generated yet
        assert!(!slot.is_complete(999));
    }

    #[test]
    fn test_deep_scov_slot_manager_empty_utilization() {
        let manager = SlotManager::new(0, 2048);
        assert_eq!(manager.utilization(), 0.0);
    }

    #[test]
    fn test_deep_scov_slot_manager_generating_slots() {
        let mut manager = SlotManager::new(4, 2048);

        manager.assign_request(vec![1], 10);
        manager.assign_request(vec![2], 10);

        // Start generation on slot 0
        manager
            .get_slot_mut(0)
            .expect("operation failed")
            .start_generation(1.0);
        // Start generation on slot 1
        manager
            .get_slot_mut(1)
            .expect("operation failed")
            .start_generation(2.0);

        let generating: Vec<_> = manager.generating_slots().collect();
        assert_eq!(generating.len(), 2);
    }

    #[test]
    fn test_deep_scov_slot_manager_aggregate_tokens_per_second() {
        let manager = SlotManager::new(4, 2048);
        // All slots idle, no tokens generated
        let tps = manager.aggregate_tokens_per_second();
        assert_eq!(tps, 0.0);
    }

    // --- MicroBatch edge cases ---

    #[test]
    fn test_deep_scov_micro_batch_with_capacity() {
        let batch = MicroBatch::with_capacity(128);
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    #[test]
    fn test_deep_scov_micro_batch_multiple_sequences() {
        let mut batch = MicroBatch::new();
        batch.add_token(BatchToken::new(1, 0, 0, true));
        batch.add_token(BatchToken::new(2, 1, 0, true));
        batch.add_token(BatchToken::new(3, 2, 0, false));

        assert_eq!(batch.num_sequences(), 3);
        assert!(batch.is_mixed());
    }

    #[test]
    fn test_deep_scov_micro_batch_pure_decode() {
        let mut batch = MicroBatch::new();
        batch.add_token(BatchToken::new(1, 0, 5, false));
        batch.add_token(BatchToken::new(2, 1, 10, false));

        assert!(batch.is_decode());
        assert_eq!(batch.n_decode_tokens, 2);
        assert_eq!(batch.n_prompt_tokens, 0);
    }

    // --- SequenceBatch edge cases ---

    #[test]
    fn test_deep_scov_sequence_batch_remove_nonexistent() {
        let mut batch = SequenceBatch::new(4);
        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1));
        let removed = batch.remove_sequence(999);
        assert!(removed.is_none());
    }

    #[test]
    fn test_deep_scov_sequence_batch_get_nonexistent() {
        let batch = SequenceBatch::new(4);
        assert!(batch.get(999).is_none());
    }

    #[test]
    fn test_deep_scov_sequence_batch_get_mut() {
        let mut batch = SequenceBatch::new(4);
        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1));

        let entry = batch.get_mut(0);
        assert!(entry.is_some());
        entry.expect("operation failed").position = 42;

        let entry = batch.get(0).expect("index out of bounds");
        assert_eq!(entry.position, 42);
    }

    #[test]
    fn test_deep_scov_sequence_batch_clear() {
        let mut batch = SequenceBatch::new(4);
        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1));
        batch.add_sequence(SequenceBatchEntry::new(1, 1, 2));
        batch.clear();

        assert!(batch.is_empty());
        assert_eq!(batch.utilization, 0.0);
    }

    #[test]
    fn test_deep_scov_sequence_batch_utilization_zero_max() {
        let mut batch = SequenceBatch::new(0);
        batch.update_utilization();
        assert_eq!(batch.utilization, 0.0);
    }

    #[test]
    fn test_deep_scov_sequence_batch_prefill_decode_iterators() {
        let mut batch = SequenceBatch::new(4);
        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1)); // prefill
        batch.add_sequence(SequenceBatchEntry::new(1, 1, 2).decoding()); // decode

        let prefill_count = batch.prefill_sequences().count();
        let decode_count = batch.decode_sequences().count();
        assert_eq!(prefill_count, 1);
        assert_eq!(decode_count, 1);
    }

    // --- BatchScheduler edge cases ---

    #[test]
    fn test_deep_scov_batch_scheduler_default() {
        let scheduler = BatchScheduler::default();
        assert!(scheduler.has_capacity());
        assert_eq!(scheduler.num_sequences(), 0);
    }

    #[test]
    fn test_deep_scov_batch_scheduler_start_decode_nonexistent() {
        let mut scheduler = BatchScheduler::new();
        assert!(!scheduler.start_decode(999, 10));
    }

    #[test]
    fn test_deep_scov_batch_scheduler_complete_nonexistent() {
        let mut scheduler = BatchScheduler::new();
        let result = scheduler.complete_sequence(999);
        assert!(result.is_none());
    }

    #[test]
    fn test_deep_scov_batch_scheduler_create_empty_ubatch() {
        let mut scheduler = BatchScheduler::new();
        let ubatch = scheduler.create_ubatch();
        assert!(ubatch.is_empty());
    }

    #[test]
    fn test_deep_scov_batch_scheduler_ubatch_limit() {
        let config = BatchConfig::default().with_max_tokens(2);
        let mut scheduler = BatchScheduler::with_config(config);

        scheduler.add_sequence(0, 1, vec![10, 20, 30, 40, 50]);

        let ubatch = scheduler.create_ubatch();
        assert_eq!(ubatch.len(), 2); // Limited by max_ubatch_tokens
    }

    #[test]
    fn test_deep_scov_batch_scheduler_mixed_prefill_decode() {
        let config = BatchConfig {
            prefer_pure_decode: false,
            ..Default::default()
        };
        let mut scheduler = BatchScheduler::with_config(config);

        // Add prefill sequence
        scheduler.add_sequence(0, 1, vec![10, 20]);

        // Add decode sequence
        let seq_idx = scheduler
            .add_sequence(1, 2, vec![30])
            .expect("index out of bounds");
        scheduler.start_decode(seq_idx, 1);

        let ubatch = scheduler.create_ubatch();
        assert!(ubatch.is_mixed());
    }

    // --- DynamicPriorityScheduler edge cases ---

    #[test]
    fn test_deep_scov_dynamic_request_is_expired_no_deadline() {
        let request = DynamicRequest::new(0, vec![1], 10);
        assert!(!request.is_expired());
    }

    #[test]
    fn test_deep_scov_dynamic_request_urgency_zero_target() {
        let mut request = DynamicRequest::new(0, vec![1], 10);
        request.deadline = Some(Deadline {
            target_latency_ms: 0,
            hard_deadline_ms: None,
            sla_target: 0.99,
        });
        assert_eq!(request.urgency_score(), 0.0);
    }
