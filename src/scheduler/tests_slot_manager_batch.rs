
    #[test]
    fn test_slot_manager_active_slots() {
        let mut manager = SlotManager::new(4, 2048);

        manager.assign_request(vec![1], 10);
        manager.assign_request(vec![2], 10);

        let active: Vec<_> = manager.active_slots().collect();
        assert_eq!(active.len(), 2);
    }

    // === Continuous Batching Tests (ubatch/sbatch) ===

    #[test]
    fn test_batch_type_default() {
        assert_eq!(BatchType::default(), BatchType::Decode);
    }

    #[test]
    fn test_batch_token_new() {
        let token = BatchToken::new(42, 0, 5, true);
        assert_eq!(token.token_id, 42);
        assert_eq!(token.seq_idx, 0);
        assert_eq!(token.seq_pos, 5);
        assert!(token.is_prompt);
    }

    #[test]
    fn test_micro_batch_new() {
        let batch = MicroBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
        assert_eq!(batch.num_sequences(), 0);
        assert!(batch.is_decode()); // Default type
    }

    #[test]
    fn test_micro_batch_add_tokens() {
        let mut batch = MicroBatch::new();

        batch.add_token(BatchToken::new(1, 0, 0, true));
        batch.add_token(BatchToken::new(2, 0, 1, true));

        assert_eq!(batch.len(), 2);
        assert_eq!(batch.num_sequences(), 1);
        assert!(batch.is_prefill());
        assert_eq!(batch.n_prompt_tokens, 2);
        assert_eq!(batch.n_decode_tokens, 0);
    }

    #[test]
    fn test_micro_batch_mixed_type() {
        let mut batch = MicroBatch::new();

        batch.add_token(BatchToken::new(1, 0, 0, true)); // Prefill
        batch.add_token(BatchToken::new(2, 1, 5, false)); // Decode

        assert!(batch.is_mixed());
        assert_eq!(batch.n_prompt_tokens, 1);
        assert_eq!(batch.n_decode_tokens, 1);
    }

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
        batch.add_token(BatchToken::new(10, 0, 0, true));
        batch.add_token(BatchToken::new(20, 0, 1, true));
        batch.add_token(BatchToken::new(30, 1, 5, false));

        assert_eq!(batch.positions(), vec![0, 1, 5]);
    }

    #[test]
    fn test_micro_batch_clear() {
        let mut batch = MicroBatch::new();
        batch.add_token(BatchToken::new(1, 0, 0, true));
        batch.add_token(BatchToken::new(2, 1, 0, false));

        batch.clear();

        assert!(batch.is_empty());
        assert_eq!(batch.n_prompt_tokens, 0);
        assert_eq!(batch.n_decode_tokens, 0);
        assert_eq!(batch.max_seq_len, 0);
    }

    #[test]
    fn test_micro_batch_max_seq_len() {
        let mut batch = MicroBatch::new();
        batch.add_token(BatchToken::new(1, 0, 0, true));
        batch.add_token(BatchToken::new(2, 0, 10, true));

        assert_eq!(batch.max_seq_len, 11); // Position 10 + 1
    }

    #[test]
    fn test_sequence_batch_entry_new() {
        let entry = SequenceBatchEntry::new(0, 1, 100);
        assert_eq!(entry.seq_idx, 0);
        assert_eq!(entry.slot_id, 1);
        assert_eq!(entry.request_id, 100);
        assert!(entry.is_prefill);
        assert_eq!(entry.position, 0);
    }

    #[test]
    fn test_sequence_batch_entry_builder() {
        let entry = SequenceBatchEntry::new(0, 1, 100)
            .with_tokens(vec![1, 2, 3])
            .at_position(5)
            .decoding();

        assert_eq!(entry.tokens, vec![1, 2, 3]);
        assert_eq!(entry.position, 5);
        assert!(!entry.is_prefill);
    }

    #[test]
    fn test_sequence_batch_new() {
        let batch = SequenceBatch::new(8);
        assert!(batch.is_empty());
        assert!(!batch.is_full());
        assert_eq!(batch.max_batch_size, 8);
    }

    #[test]
    fn test_sequence_batch_add_remove() {
        let mut batch = SequenceBatch::new(4);

        let entry = SequenceBatchEntry::new(0, 0, 1);
        assert!(batch.add_sequence(entry));
        assert_eq!(batch.len(), 1);

        let removed = batch.remove_sequence(0);
        assert!(removed.is_some());
        assert!(batch.is_empty());
    }

    #[test]
    fn test_sequence_batch_full() {
        let mut batch = SequenceBatch::new(2);

        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1));
        batch.add_sequence(SequenceBatchEntry::new(1, 1, 2));

        assert!(batch.is_full());
        assert!(!batch.add_sequence(SequenceBatchEntry::new(2, 2, 3)));
    }

    #[test]
    fn test_sequence_batch_prefill_decode_counts() {
        let mut batch = SequenceBatch::new(4);

        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1)); // Prefill
        batch.add_sequence(SequenceBatchEntry::new(1, 1, 2).decoding()); // Decode
        batch.add_sequence(SequenceBatchEntry::new(2, 2, 3).decoding()); // Decode

        assert_eq!(batch.num_prefill(), 1);
        assert_eq!(batch.num_decode(), 2);
    }

    #[test]
    fn test_sequence_batch_utilization() {
        let mut batch = SequenceBatch::new(4);

        assert_eq!(batch.utilization, 0.0);

        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1));
        assert!((batch.utilization - 0.25).abs() < 0.01);

        batch.add_sequence(SequenceBatchEntry::new(1, 1, 2));
        assert!((batch.utilization - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.max_ubatch_tokens, 512);
        assert_eq!(config.max_sbatch_sequences, 8);
        assert!(config.prefer_pure_decode);
        assert!(config.dynamic_batching);
    }

    #[test]
    fn test_batch_config_builder() {
        let config = BatchConfig::default()
            .with_max_tokens(1024)
            .with_max_sequences(16);

        assert_eq!(config.max_ubatch_tokens, 1024);
        assert_eq!(config.max_sbatch_sequences, 16);
    }

    #[test]
    fn test_batch_scheduler_new() {
        let scheduler = BatchScheduler::new();
        assert!(scheduler.has_capacity());
        assert_eq!(scheduler.num_sequences(), 0);
        assert_eq!(scheduler.utilization(), 0.0);
    }

    #[test]
    fn test_batch_scheduler_add_sequence() {
        let mut scheduler = BatchScheduler::new();

        let seq_idx = scheduler.add_sequence(0, 1, vec![10, 20, 30]);
        assert!(seq_idx.is_some());
        assert_eq!(seq_idx.expect("test"), 0);
        assert_eq!(scheduler.num_sequences(), 1);
    }

    #[test]
    fn test_batch_scheduler_complete_sequence() {
        let mut scheduler = BatchScheduler::new();

        let seq_idx = scheduler.add_sequence(0, 1, vec![10, 20]).expect("test");
        assert_eq!(scheduler.num_sequences(), 1);

        let completed = scheduler.complete_sequence(seq_idx);
        assert!(completed.is_some());
        assert_eq!(scheduler.num_sequences(), 0);
    }

    #[test]
    fn test_batch_scheduler_start_decode() {
        let mut scheduler = BatchScheduler::new();

        let seq_idx = scheduler
            .add_sequence(0, 1, vec![10, 20, 30])
            .expect("test");

        // Initially in prefill
        assert!(scheduler.sbatch().get(seq_idx).expect("test").is_prefill);

        // Transition to decode
        assert!(scheduler.start_decode(seq_idx, 3));

        let entry = scheduler.sbatch().get(seq_idx).expect("test");
        assert!(!entry.is_prefill);
        assert_eq!(entry.position, 3);
        assert!(entry.tokens.is_empty()); // Cleared after prefill
    }

    #[test]
    fn test_batch_scheduler_create_ubatch_prefill() {
        let mut scheduler = BatchScheduler::new();

        scheduler.add_sequence(0, 1, vec![10, 20, 30]);

        let ubatch = scheduler.create_ubatch();

        assert!(ubatch.is_prefill());
        assert_eq!(ubatch.len(), 3);
        assert_eq!(ubatch.token_ids(), vec![10, 20, 30]);
    }

    #[test]
    fn test_batch_scheduler_create_ubatch_decode() {
        let mut scheduler = BatchScheduler::new();

        let seq_idx = scheduler
            .add_sequence(0, 1, vec![10, 20, 30])
            .expect("test");
        scheduler.start_decode(seq_idx, 3);

        let ubatch = scheduler.create_ubatch();

        assert!(ubatch.is_decode());
        assert_eq!(ubatch.len(), 1);
    }

    #[test]
    fn test_batch_scheduler_stats() {
        let mut scheduler = BatchScheduler::new();

        scheduler.add_sequence(0, 1, vec![10, 20, 30]);
        scheduler.create_ubatch();

        let stats = scheduler.stats();
        assert_eq!(stats.ubatches_created, 1);
        assert_eq!(stats.tokens_processed, 3);
        assert_eq!(stats.prefill_tokens, 3);
    }

    #[test]
    fn test_batch_scheduler_capacity() {
        let config = BatchConfig::default().with_max_sequences(2);
        let mut scheduler = BatchScheduler::with_config(config);

        scheduler.add_sequence(0, 1, vec![1]);
        scheduler.add_sequence(1, 2, vec![2]);

        assert!(!scheduler.has_capacity());
        assert!(scheduler.add_sequence(2, 3, vec![3]).is_none());
    }

    #[test]
    fn test_batch_stats_default() {
        let stats = BatchStats::default();
        assert_eq!(stats.ubatches_created, 0);
        assert_eq!(stats.tokens_processed, 0);
        assert_eq!(stats.avg_ubatch_size, 0.0);
    }

    // ========================================================================
    // Dynamic Priority Scheduling Tests
    // ========================================================================

    #[test]
    fn test_deadline_default() {
        let deadline = Deadline::default();
        assert_eq!(deadline.target_latency_ms, 1000);
        assert!(deadline.hard_deadline_ms.is_none());
        assert!((deadline.sla_target - 0.99).abs() < 0.001);
    }

    #[test]
    fn test_deadline_with_target() {
        let deadline = Deadline::with_target(500);
        assert_eq!(deadline.target_latency_ms, 500);
    }

    #[test]
    fn test_deadline_strict() {
        let deadline = Deadline::strict(100, 200);
        assert_eq!(deadline.target_latency_ms, 100);
        assert_eq!(deadline.hard_deadline_ms, Some(200));
        assert!((deadline.sla_target - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_dynamic_priority_config_default() {
        let config = DynamicPriorityConfig::default();
        assert!(config.enable_age_promotion);
        assert_eq!(config.promotion_interval_ms, 5000);
        assert_eq!(config.max_promoted_priority, Priority::High);
        assert!(config.enable_deadline_scheduling);
        assert!(config.enable_fair_share);
    }

    #[test]
    fn test_dynamic_priority_config_builder() {
        let config = DynamicPriorityConfig::with_budgets([0.1, 0.2, 0.3, 0.4])
            .no_promotion()
            .with_promotion_interval(1000);

        assert!(!config.enable_age_promotion);
        assert_eq!(config.promotion_interval_ms, 1000);
        assert!((config.priority_budgets[0] - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_dynamic_request_new() {
        let request = DynamicRequest::new(0, vec![1, 2, 3], 10);
        assert_eq!(request.request_id, 0);
        assert_eq!(request.input_ids.len(), 3);
        assert_eq!(request.max_tokens, 10);
        assert_eq!(request.original_priority, Priority::Normal);
        assert_eq!(request.effective_priority, Priority::Normal);
        assert_eq!(request.promotions, 0);
    }

    #[test]
    fn test_dynamic_request_with_priority() {
        let request = DynamicRequest::new(0, vec![1], 10).with_priority(Priority::High);
        assert_eq!(request.original_priority, Priority::High);
        assert_eq!(request.effective_priority, Priority::High);
    }

    #[test]
    fn test_dynamic_request_with_deadline() {
        let request = DynamicRequest::new(0, vec![1], 10).with_deadline(Deadline::with_target(500));
        assert!(request.deadline.is_some());
        assert_eq!(request.deadline.expect("test").target_latency_ms, 500);
    }

    #[test]
    fn test_dynamic_request_urgency_no_deadline() {
        let request = DynamicRequest::new(0, vec![1], 10);
        assert_eq!(request.urgency_score(), 0.0);
        assert!(!request.is_urgent());
    }

    #[test]
    fn test_dynamic_request_remaining_tokens() {
        let mut request = DynamicRequest::new(0, vec![1], 10);
        assert_eq!(request.remaining_tokens(), 10);
        request.generated_tokens = vec![2, 3, 4];
        assert_eq!(request.remaining_tokens(), 7);
    }

    #[test]
    fn test_dynamic_request_total_tokens() {
        let mut request = DynamicRequest::new(0, vec![1, 2, 3], 10);
        assert_eq!(request.total_tokens(), 3);
        request.generated_tokens = vec![4, 5];
        assert_eq!(request.total_tokens(), 5);
    }

    #[test]
    fn test_dynamic_scheduler_new() {
        let scheduler = DynamicPriorityScheduler::new(1024);
        assert_eq!(scheduler.waiting_count(), 0);
        assert_eq!(scheduler.running_count(), 0);
        assert_eq!(scheduler.batch_token_budget, 1024);
    }

    #[test]
    fn test_dynamic_scheduler_add_request() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);

        let id1 = scheduler.add_request(vec![1, 2, 3], 10, Priority::Normal, None);
        assert_eq!(id1, 0);
        assert_eq!(scheduler.waiting_count(), 1);
        assert_eq!(scheduler.queue_depth(Priority::Normal), 1);

        let id2 = scheduler.add_request(vec![4, 5], 5, Priority::High, None);
        assert_eq!(id2, 1);
        assert_eq!(scheduler.waiting_count(), 2);
        assert_eq!(scheduler.queue_depth(Priority::High), 1);
    }

    #[test]
    fn test_dynamic_scheduler_add_simple_request() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);
        let id = scheduler.add_simple_request(vec![1, 2], 5);
        assert_eq!(id, 0);
        assert_eq!(scheduler.queue_depth(Priority::Normal), 1);
    }
