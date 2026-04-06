
    // ========================================================================
    // Deadline Tests
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
        assert!(deadline.hard_deadline_ms.is_none());
    }

    #[test]
    fn test_deadline_strict() {
        let deadline = Deadline::strict(200, 1000);
        assert_eq!(deadline.target_latency_ms, 200);
        assert_eq!(deadline.hard_deadline_ms, Some(1000));
        assert!((deadline.sla_target - 1.0).abs() < 0.001);
    }

    // ========================================================================
    // DynamicPriorityConfig Tests
    // ========================================================================

    #[test]
    fn test_dynamic_priority_config_default() {
        let config = DynamicPriorityConfig::default();
        assert!(config.enable_age_promotion);
        assert_eq!(config.promotion_interval_ms, 5000);
        assert_eq!(config.max_promoted_priority, Priority::High);
        assert!(config.enable_deadline_scheduling);
        assert!((config.urgency_factor - 2.0).abs() < 0.001);
        assert_eq!(config.min_tokens_per_request, 1);
        assert!(config.enable_fair_share);
    }

    #[test]
    fn test_dynamic_priority_config_with_budgets() {
        let budgets = [0.1, 0.2, 0.3, 0.4];
        let config = DynamicPriorityConfig::with_budgets(budgets);
        assert_eq!(config.priority_budgets, budgets);
    }

    #[test]
    fn test_dynamic_priority_config_no_promotion() {
        let config = DynamicPriorityConfig::default().no_promotion();
        assert!(!config.enable_age_promotion);
    }

    #[test]
    fn test_dynamic_priority_config_with_promotion_interval() {
        let config = DynamicPriorityConfig::default().with_promotion_interval(10000);
        assert_eq!(config.promotion_interval_ms, 10000);
    }

    // ========================================================================
    // BatchConfig Tests
    // ========================================================================

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.max_ubatch_tokens, 512);
        assert_eq!(config.max_sbatch_sequences, 8);
        assert!(config.prefer_pure_decode);
        assert_eq!(config.max_context_length, 2048);
        assert!(config.dynamic_batching);
    }

    #[test]
    fn test_batch_config_with_max_tokens() {
        let config = BatchConfig::default().with_max_tokens(1024);
        assert_eq!(config.max_ubatch_tokens, 1024);
    }

    #[test]
    fn test_batch_config_with_max_sequences() {
        let config = BatchConfig::default().with_max_sequences(16);
        assert_eq!(config.max_sbatch_sequences, 16);
    }

    // ========================================================================
    // DynamicRequest Tests
    // ========================================================================

    #[test]
    fn test_dynamic_request_new() {
        let request = DynamicRequest::new(42, vec![1, 2, 3], 10);
        assert_eq!(request.request_id, 42);
        assert_eq!(request.input_ids, vec![1, 2, 3]);
        assert_eq!(request.max_tokens, 10);
        assert_eq!(request.original_priority, Priority::Normal);
        assert_eq!(request.effective_priority, Priority::Normal);
        assert!(request.deadline.is_none());
        assert_eq!(request.promotions, 0);
        assert_eq!(request.state, SequenceState::Waiting);
        assert!(request.generated_tokens.is_empty());
        assert!(request.seq_id.is_none());
        assert!(request.ttft_ms.is_none());
    }

    #[test]
    fn test_dynamic_request_with_priority() {
        let request = DynamicRequest::new(1, vec![1], 5).with_priority(Priority::Critical);
        assert_eq!(request.original_priority, Priority::Critical);
        assert_eq!(request.effective_priority, Priority::Critical);
    }

    #[test]
    fn test_dynamic_request_with_deadline() {
        let request = DynamicRequest::new(1, vec![1], 5).with_deadline(Deadline::with_target(500));
        assert!(request.deadline.is_some());
        assert_eq!(request.deadline.as_ref().unwrap().target_latency_ms, 500);
    }

    #[test]
    fn test_dynamic_request_wait_time() {
        let request = DynamicRequest::new(1, vec![1], 5);
        // Wait time should be very small (just created)
        assert!(request.wait_time_ms() < 100);
    }

    #[test]
    fn test_dynamic_request_is_urgent_false() {
        let request =
            DynamicRequest::new(1, vec![1], 5).with_deadline(Deadline::with_target(10000));
        // Fresh request is not urgent
        assert!(!request.is_urgent());
    }

    #[test]
    fn test_dynamic_request_is_urgent_no_deadline() {
        let request = DynamicRequest::new(1, vec![1], 5);
        // No deadline means never urgent
        assert!(!request.is_urgent());
    }

    #[test]
    fn test_dynamic_request_is_expired_no_hard_deadline() {
        let request = DynamicRequest::new(1, vec![1], 5).with_deadline(Deadline::with_target(500));
        // No hard deadline, so never expired
        assert!(!request.is_expired());
    }

    #[test]
    fn test_dynamic_request_urgency_score_no_deadline() {
        let request = DynamicRequest::new(1, vec![1], 5);
        assert!((request.urgency_score() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_dynamic_request_urgency_score_zero_target() {
        let deadline = Deadline {
            target_latency_ms: 0,
            hard_deadline_ms: None,
            sla_target: 0.99,
        };
        let request = DynamicRequest::new(1, vec![1], 5).with_deadline(deadline);
        assert!((request.urgency_score() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_dynamic_request_remaining_tokens() {
        let mut request = DynamicRequest::new(1, vec![1, 2, 3], 10);
        assert_eq!(request.remaining_tokens(), 10);
        request.generated_tokens = vec![4, 5, 6];
        assert_eq!(request.remaining_tokens(), 7);
    }

    #[test]
    fn test_dynamic_request_remaining_tokens_overflow() {
        let mut request = DynamicRequest::new(1, vec![1], 5);
        request.generated_tokens = vec![1, 2, 3, 4, 5, 6, 7]; // More than max
        assert_eq!(request.remaining_tokens(), 0); // saturating_sub protects
    }

    #[test]
    fn test_dynamic_request_total_tokens() {
        let mut request = DynamicRequest::new(1, vec![1, 2, 3], 10);
        request.generated_tokens = vec![4, 5];
        assert_eq!(request.total_tokens(), 5);
    }

    // ========================================================================
    // DynamicSchedulerStats Tests
    // ========================================================================

    #[test]
    fn test_dynamic_scheduler_stats_default() {
        let stats = DynamicSchedulerStats::default();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.completed_requests, 0);
        assert_eq!(stats.sla_met, 0);
        assert_eq!(stats.sla_missed, 0);
        assert_eq!(stats.dropped_requests, 0);
        assert_eq!(stats.promotions, 0);
        assert!((stats.avg_ttft_ms - 0.0).abs() < 0.001);
        assert!((stats.p99_ttft_ms - 0.0).abs() < 0.001);
        assert_eq!(stats.tokens_by_priority, [0, 0, 0, 0]);
        assert_eq!(stats.queue_depth_by_priority, [0, 0, 0, 0]);
    }

    // ========================================================================
    // DynamicPriorityScheduler Full Coverage Tests
    // ========================================================================

    #[test]
    fn test_dynamic_scheduler_with_config() {
        let config = DynamicPriorityConfig::default().no_promotion();
        let scheduler = DynamicPriorityScheduler::with_config(1024, config);
        assert!(!scheduler.config().enable_age_promotion);
    }

    #[test]
    fn test_dynamic_scheduler_add_request_with_deadline() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);
        let id = scheduler.add_request(
            vec![1, 2, 3],
            10,
            Priority::High,
            Some(Deadline::strict(100, 500)),
        );
        assert_eq!(id, 0);
        let request = scheduler.get_request(id).unwrap();
        assert_eq!(request.original_priority, Priority::High);
        assert!(request.deadline.is_some());
    }

    #[test]
    fn test_dynamic_scheduler_queue_depth() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);
        scheduler.add_request(vec![1], 5, Priority::Low, None);
        scheduler.add_request(vec![2], 5, Priority::Normal, None);
        scheduler.add_request(vec![3], 5, Priority::High, None);
        scheduler.add_request(vec![4], 5, Priority::Critical, None);

        assert_eq!(scheduler.queue_depth(Priority::Low), 1);
        assert_eq!(scheduler.queue_depth(Priority::Normal), 1);
        assert_eq!(scheduler.queue_depth(Priority::High), 1);
        assert_eq!(scheduler.queue_depth(Priority::Critical), 1);
    }

    #[test]
    fn test_dynamic_scheduler_stats_accessor() {
        let scheduler = DynamicPriorityScheduler::new(1024);
        let stats = scheduler.stats();
        assert_eq!(stats.total_requests, 0);
    }

    #[test]
    fn test_dynamic_scheduler_sla_compliance_empty() {
        let scheduler = DynamicPriorityScheduler::new(1024);
        // No requests = 100% compliance
        assert!((scheduler.sla_compliance_rate() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_dynamic_scheduler_schedule_with_fair_share() {
        let config = DynamicPriorityConfig::default();
        let mut scheduler = DynamicPriorityScheduler::with_config(100, config);

        // Add requests at different priorities
        scheduler.add_request(vec![1], 5, Priority::Low, None);
        scheduler.add_request(vec![2], 5, Priority::Normal, None);
        scheduler.add_request(vec![3], 5, Priority::High, None);

        let scheduled = scheduler.schedule(3);
        // Higher priority should be scheduled first
        assert!(!scheduled.is_empty());
    }

    #[test]
    fn test_dynamic_scheduler_schedule_critical_first() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);

        // Add low priority first
        scheduler.add_request(vec![1], 5, Priority::Low, None);
        // Then critical
        scheduler.add_request(vec![2], 5, Priority::Critical, None);

        let scheduled = scheduler.schedule(1);
        assert_eq!(scheduled.len(), 1);
        // Critical should be scheduled first
        let (id, _tokens) = scheduled[0];
        let request = scheduler.get_request(id).unwrap();
        assert_eq!(request.original_priority, Priority::Critical);
    }

    #[test]
    fn test_dynamic_scheduler_complete_with_sla_met() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);

        // Add request with long target latency (will definitely be met)
        let id = scheduler.add_request(
            vec![1],
            5,
            Priority::Normal,
            Some(Deadline::with_target(60000)),
        );
        scheduler.schedule(1);

        scheduler.complete_request(id);
        assert_eq!(scheduler.stats().sla_met, 1);
        assert_eq!(scheduler.stats().sla_missed, 0);
    }

    #[test]
    fn test_dynamic_scheduler_promote_disabled() {
        let config = DynamicPriorityConfig::default().no_promotion();
        let mut scheduler = DynamicPriorityScheduler::with_config(1024, config);

        scheduler.add_request(vec![1], 5, Priority::Low, None);

        // Promotion is disabled, so this should be a no-op
        scheduler.promote_aged_requests();

        assert_eq!(scheduler.stats().promotions, 0);
    }

    #[test]
    fn test_dynamic_scheduler_schedule_no_slots() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);
        scheduler.add_request(vec![1], 5, Priority::Normal, None);

        // Schedule with 0 available slots
        let scheduled = scheduler.schedule(0);
        assert!(scheduled.is_empty());
    }

    #[test]
    fn test_dynamic_scheduler_schedule_no_budget() {
        let mut scheduler =
            DynamicPriorityScheduler::with_config(0, DynamicPriorityConfig::default());
        scheduler.add_request(vec![1], 5, Priority::Normal, None);

        // Zero budget
        let scheduled = scheduler.schedule(10);
        // With 0 budget and min_tokens_per_request=1, nothing should be scheduled
        assert!(scheduled.is_empty());
    }

    // ========================================================================
    // SequenceBatchEntry Tests
    // ========================================================================

    #[test]
    fn test_sequence_batch_entry_new() {
        let entry = SequenceBatchEntry::new(0, 1, 100);
        assert_eq!(entry.seq_idx, 0);
        assert_eq!(entry.slot_id, 1);
        assert_eq!(entry.request_id, 100);
        assert_eq!(entry.position, 0);
        assert!(entry.tokens.is_empty());
        assert!(entry.is_prefill);
    }

    #[test]
    fn test_sequence_batch_entry_with_tokens() {
        let entry = SequenceBatchEntry::new(0, 1, 100).with_tokens(vec![1, 2, 3]);
        assert_eq!(entry.tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_sequence_batch_entry_at_position() {
        let entry = SequenceBatchEntry::new(0, 1, 100).at_position(42);
        assert_eq!(entry.position, 42);
    }

    #[test]
    fn test_sequence_batch_entry_decoding() {
        let entry = SequenceBatchEntry::new(0, 1, 100).decoding();
        assert!(!entry.is_prefill);
    }

    #[test]
    fn test_sequence_batch_entry_builder_chain() {
        let entry = SequenceBatchEntry::new(0, 1, 100)
            .with_tokens(vec![10, 20])
            .at_position(5)
            .decoding();
        assert_eq!(entry.tokens, vec![10, 20]);
        assert_eq!(entry.position, 5);
        assert!(!entry.is_prefill);
    }

    // ========================================================================
    // SequenceBatch Tests
    // ========================================================================

    #[test]
    fn test_sequence_batch_full() {
        let mut batch = SequenceBatch::new(2);
        assert!(!batch.is_full());

        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1));
        assert!(!batch.is_full());

        batch.add_sequence(SequenceBatchEntry::new(1, 1, 2));
        assert!(batch.is_full());
    }

    #[test]
    fn test_sequence_batch_add_when_full() {
        let mut batch = SequenceBatch::new(1);
        assert!(batch.add_sequence(SequenceBatchEntry::new(0, 0, 1)));
        assert!(!batch.add_sequence(SequenceBatchEntry::new(1, 1, 2)));
    }

    #[test]
    fn test_sequence_batch_remove_nonexistent() {
        let mut batch = SequenceBatch::new(8);
        batch.add_sequence(SequenceBatchEntry::new(5, 0, 1));
        assert!(batch.remove_sequence(999).is_none());
    }

    #[test]
    fn test_sequence_batch_utilization() {
        let mut batch = SequenceBatch::new(4);
        assert!((batch.utilization - 0.0).abs() < 0.001);

        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1));
        assert!((batch.utilization - 0.25).abs() < 0.001);

        batch.add_sequence(SequenceBatchEntry::new(1, 1, 2));
        assert!((batch.utilization - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_sequence_batch_utilization_zero_max() {
        let batch = SequenceBatch::new(0);
        assert!((batch.utilization - 0.0).abs() < 0.001);
    }
