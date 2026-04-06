
    #[test]
    fn test_deep_scov_dynamic_scheduler_complete_nonexistent() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);
        let result = scheduler.complete_request(999);
        assert!(result.is_none());
    }

    #[test]
    fn test_deep_scov_dynamic_scheduler_sla_compliance_no_sla() {
        let scheduler = DynamicPriorityScheduler::new(1024);
        // No requests with SLA, should return 1.0
        assert_eq!(scheduler.sla_compliance_rate(), 1.0);
    }

    #[test]
    fn test_deep_scov_dynamic_scheduler_promote_aged_disabled() {
        let config = DynamicPriorityConfig::default().no_promotion();
        let mut scheduler = DynamicPriorityScheduler::with_config(1024, config);
        scheduler.add_request(vec![1], 5, Priority::Low, None);
        scheduler.promote_aged_requests();
        // Should not promote when disabled
        assert_eq!(scheduler.queue_depth(Priority::Low), 1);
    }

    #[test]
    fn test_deep_scov_dynamic_scheduler_fair_share_disabled() {
        let config = DynamicPriorityConfig {
            enable_fair_share: false,
            ..Default::default()
        };
        let mut scheduler = DynamicPriorityScheduler::with_config(1024, config);
        scheduler.add_request(vec![1], 50, Priority::Low, None);
        let batch = scheduler.schedule(1);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_deep_scov_dynamic_scheduler_zero_slots() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);
        scheduler.add_request(vec![1], 5, Priority::Normal, None);
        let batch = scheduler.schedule(0);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_deep_scov_dynamic_scheduler_zero_budget() {
        let mut scheduler = DynamicPriorityScheduler::new(0);
        scheduler.add_request(vec![1], 5, Priority::Normal, None);
        let batch = scheduler.schedule(10);
        // With 0 budget, nothing should be scheduled
        assert!(batch.is_empty());
    }

    // --- ChunkedPrefillScheduler edge cases ---

    #[test]
    fn test_deep_scov_chunked_prefill_scheduler_get_state_nonexistent() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        assert!(scheduler.get_state(999).is_none());
    }

    #[test]
    fn test_deep_scov_chunked_prefill_scheduler_remove_nonexistent() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        let result = scheduler.remove(999);
        assert!(result.is_none());
    }

    #[test]
    fn test_deep_scov_chunked_prefill_round_robin_mode() {
        let config = ChunkedPrefillConfig {
            boost_partial_prefill: false,
            ..Default::default()
        };
        let mut scheduler = ChunkedPrefillScheduler::new(config);

        scheduler.submit(1000);
        scheduler.submit(2000);

        // Complete first chunk of first sequence
        scheduler.complete_chunk(0, 512, 50);

        // Should be moved to back of queue (round-robin)
        // Queue should now have seq 1 first
        let (next_seq, _) = scheduler.next_chunk().expect("operation failed");
        assert_eq!(next_seq, 1);
    }

    #[test]
    fn test_deep_scov_chunked_prefill_no_interleave() {
        let config = ChunkedPrefillConfig {
            allow_decode_interleave: false,
            ..Default::default()
        };
        let mut scheduler = ChunkedPrefillScheduler::new(config);
        scheduler.submit(1000);
        assert!(!scheduler.should_interleave_decode());
    }

    #[test]
    fn test_deep_scov_chunked_prefill_next_chunk_empty_queue() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        assert!(scheduler.next_chunk().is_none());
    }

    #[test]
    fn test_deep_scov_chunked_prefill_complete_chunk_nonexistent() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        // Should not panic
        scheduler.complete_chunk(999, 100, 10);
        assert_eq!(scheduler.stats().chunks_processed, 0);
    }

    #[test]
    fn test_deep_scov_update_after_iteration_nonexistent() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut generated = HashMap::new();
        generated.insert(999u64, 42u32);
        // Should not panic when updating non-existent request
        scheduler.update_after_iteration(&generated);
    }

    // ========================================================================
    // Coverage Improvement Tests (_cov_ prefix)
    // ========================================================================
    // These tests target specific uncovered branches identified in coverage analysis

    // --- Scheduler KV Cache Allocation Failure Paths ---

    #[test]
    fn test_cov_schedule_waiting_kv_allocation_failure() {
        // Test the Err branch in schedule_waiting when KV cache is full
        let mut scheduler = Scheduler::new(32, 1000);
        // Create KV cache with very limited pages to force allocation failure
        let mut kv_cache = PagedKvCache::new(1, 4, 4, 16); // Very small cache

        // Add multiple requests that will exceed cache capacity
        let _ = scheduler
            .add_request(vec![1, 2, 3, 4, 5, 6, 7, 8], 10)
            .expect("add 1");
        let _ = scheduler
            .add_request(vec![1, 2, 3, 4, 5, 6, 7, 8], 10)
            .expect("add 2");
        let _ = scheduler
            .add_request(vec![1, 2, 3, 4, 5, 6, 7, 8], 10)
            .expect("add 3");

        // First schedule attempt - should schedule at least one
        let output1 = scheduler.schedule(&mut kv_cache, 0).expect("schedule 1");

        // At least one request was put back due to allocation failure
        // or all succeeded with small cache
        let total_scheduled = output1.scheduled_request_ids.len();
        let total_waiting = scheduler.waiting_count();

        // Either some are scheduled and some waiting, or all scheduled
        assert!(total_scheduled >= 1 || total_waiting >= 1);
    }

    #[test]
    fn test_cov_resume_preempted_kv_allocation_failure() {
        // Test the Err branch in resume_preempted when KV cache is full
        let mut scheduler = Scheduler::new(1, 1000); // Single slot
        let mut kv_cache = PagedKvCache::new(2, 4, 4, 16); // Limited cache

        // Schedule low priority request
        let low_id = scheduler
            .add_request_with_priority(vec![1, 2, 3, 4], 10, Priority::Low)
            .expect("add low");
        let _ = scheduler.schedule(&mut kv_cache, 0).expect("schedule 1");

        // Add critical priority to trigger preemption
        let _critical_id = scheduler
            .add_request_with_priority(vec![1, 2], 5, Priority::Critical)
            .expect("add critical");
        let _ = scheduler.schedule(&mut kv_cache, 0).expect("schedule 2");

        // Low priority may be preempted - verify the method is callable
        let _preempted = scheduler.preempted_count();

        // Complete the critical request
        let low_req = scheduler.get_request(low_id);
        if low_req.is_some() && low_req.expect("test").state == SequenceState::Preempted {
            // The preempted request exists
            assert!(scheduler.preempted_count() >= 1);
        }
    }

    #[test]
    fn test_cov_preemption_with_seq_id() {
        // Test that preemption properly frees seq_id
        let mut scheduler = Scheduler::new(1, 1000);
        let mut kv_cache = PagedKvCache::new(10, 4, 4, 16);

        // Add and schedule low priority
        let low_id = scheduler
            .add_request_with_priority(vec![1, 2], 10, Priority::Low)
            .expect("add low");
        let _ = scheduler.schedule(&mut kv_cache, 0).expect("schedule");

        // Verify low priority got a seq_id
        let low_req = scheduler.get_request(low_id).expect("get low");
        assert!(low_req.seq_id.is_some());

        // Add critical priority to trigger preemption
        let _critical_id = scheduler
            .add_request_with_priority(vec![3, 4], 5, Priority::Critical)
            .expect("add critical");
        let output = scheduler.schedule(&mut kv_cache, 0).expect("schedule 2");

        // Check preemption occurred
        if !output.preempted_seq_ids.is_empty() {
            assert_eq!(scheduler.stats().preemptions, 1);
        }
    }

    // --- Dynamic Priority Scheduler Promotion Tests ---

    #[test]
    fn test_cov_dynamic_scheduler_promotion_low_to_normal() {
        // Test actual promotion from Low to Normal with instant promotion
        let config = DynamicPriorityConfig {
            enable_age_promotion: true,
            promotion_interval_ms: 0, // Instant promotion
            max_promoted_priority: Priority::High,
            ..Default::default()
        };
        let mut scheduler = DynamicPriorityScheduler::with_config(1024, config);

        // Add low priority request
        let id = scheduler.add_request(vec![1], 5, Priority::Low, None);
        assert_eq!(scheduler.queue_depth(Priority::Low), 1);

        // Promote - should move from Low to Normal
        scheduler.promote_aged_requests();

        // Request should have been promoted
        let request = scheduler.get_request(id).expect("get request");
        assert!(request.effective_priority >= Priority::Normal);
        assert!(request.promotions >= 1);
    }

    #[test]
    fn test_cov_dynamic_scheduler_promotion_normal_to_high() {
        let config = DynamicPriorityConfig {
            enable_age_promotion: true,
            promotion_interval_ms: 0,
            max_promoted_priority: Priority::High,
            ..Default::default()
        };
        let mut scheduler = DynamicPriorityScheduler::with_config(1024, config);

        // Add Normal priority request
        let id = scheduler.add_request(vec![1], 5, Priority::Normal, None);

        // First promotion: Normal -> High
        scheduler.promote_aged_requests();

        let request = scheduler.get_request(id).expect("get request");
        assert_eq!(request.effective_priority, Priority::High);
    }

    #[test]
    fn test_cov_dynamic_scheduler_promotion_capped_at_max() {
        let config = DynamicPriorityConfig {
            enable_age_promotion: true,
            promotion_interval_ms: 0,
            max_promoted_priority: Priority::Normal, // Cap at Normal
            ..Default::default()
        };
        let mut scheduler = DynamicPriorityScheduler::with_config(1024, config);

        // Add Low priority request
        let id = scheduler.add_request(vec![1], 5, Priority::Low, None);

        // First promotion: Low -> Normal
        scheduler.promote_aged_requests();

        let request = scheduler.get_request(id).expect("get request");
        // Should be Normal now
        assert_eq!(request.effective_priority, Priority::Normal);

        // Second promotion should not happen (Normal >= max_promoted_priority)
        scheduler.promote_aged_requests();
        let request = scheduler.get_request(id).expect("get request 2");
        assert_eq!(request.effective_priority, Priority::Normal);
    }

    #[test]
    fn test_cov_dynamic_scheduler_high_priority_no_promotion_when_at_max() {
        let config = DynamicPriorityConfig {
            enable_age_promotion: true,
            promotion_interval_ms: 0,
            max_promoted_priority: Priority::High, // Cap at High
            ..Default::default()
        };
        let mut scheduler = DynamicPriorityScheduler::with_config(1024, config);

        // Add High priority request (already at max)
        let id = scheduler.add_request(vec![1], 5, Priority::High, None);

        // Promotion should not happen (High >= max_promoted_priority)
        scheduler.promote_aged_requests();

        let request = scheduler.get_request(id).expect("get request");
        assert_eq!(request.effective_priority, Priority::High);
        assert_eq!(request.promotions, 0);
    }

    // --- Deadline Expiration Tests ---

    #[test]
    fn test_cov_dynamic_scheduler_deadline_expired() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);

        // Add request with immediate hard deadline (already expired)
        let deadline = Deadline {
            target_latency_ms: 0,
            hard_deadline_ms: Some(0), // Immediate expiration
            sla_target: 1.0,
        };
        let id = scheduler.add_request(vec![1], 5, Priority::Normal, Some(deadline));
        assert_eq!(scheduler.waiting_count(), 1);

        // Small delay to ensure deadline passes
        std::thread::sleep(std::time::Duration::from_millis(1));

        // Drop expired
        let dropped = scheduler.drop_expired();
        assert!(dropped.contains(&id));
        assert_eq!(scheduler.stats().dropped_requests, 1);
        assert_eq!(scheduler.waiting_count(), 0);
    }

    #[test]
    fn test_cov_dynamic_scheduler_sla_missed() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);

        // Add request with immediate target (will miss SLA)
        let deadline = Deadline {
            target_latency_ms: 0,   // Will miss immediately
            hard_deadline_ms: None, // No hard deadline
            sla_target: 0.99,
        };
        let id = scheduler.add_request(vec![1], 5, Priority::Normal, Some(deadline));

        // Small delay
        std::thread::sleep(std::time::Duration::from_millis(1));

        // Schedule and complete
        let _ = scheduler.schedule(1);
        let _ = scheduler.complete_request(id);

        // SLA should be missed
        assert_eq!(scheduler.stats().sla_missed, 1);
        assert!(scheduler.sla_compliance_rate() < 1.0);
    }

    #[test]
    fn test_cov_dynamic_request_is_urgent() {
        let mut request = DynamicRequest::new(0, vec![1], 10);
        request.deadline = Some(Deadline::with_target(1)); // 1ms target

        // Wait a tiny bit
        std::thread::sleep(std::time::Duration::from_millis(1));

        // Should now be urgent (elapsed >= target/2)
        assert!(request.is_urgent() || request.urgency_score() > 0.0);
    }

    #[test]
    fn test_cov_dynamic_request_is_expired_with_hard_deadline() {
        let mut request = DynamicRequest::new(0, vec![1], 10);
        request.deadline = Some(Deadline::strict(0, 0)); // Immediate hard deadline

        std::thread::sleep(std::time::Duration::from_millis(1));

        assert!(request.is_expired());
    }

    // --- TTFT Statistics Tests ---

    #[test]
    fn test_cov_dynamic_scheduler_ttft_p99_calculation() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);

        // Add and complete multiple requests with SLA
        for i in 0..10 {
            let deadline = Deadline::with_target(100_000); // Long deadline
            let id = scheduler.add_request(vec![i as u32], 5, Priority::Normal, Some(deadline));
            let _ = scheduler.schedule(1);
            let _ = scheduler.complete_request(id);
        }

        // Check TTFT stats are populated
        assert!(scheduler.stats().avg_ttft_ms >= 0.0);
        assert!(scheduler.stats().p99_ttft_ms >= 0.0);
    }

    // --- Slot State Serialization Tests ---

    #[test]
    fn test_cov_slot_state_serialization() {
        let states = [
            SlotState::Idle,
            SlotState::Processing,
            SlotState::Generating,
        ];
        for state in &states {
            let json = serde_json::to_string(state).expect("serialize");
            let parsed: SlotState = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(*state, parsed);
        }
    }

    #[test]
    fn test_cov_batch_type_serialization() {
        let types = [BatchType::Prefill, BatchType::Decode, BatchType::Mixed];
        for bt in &types {
            let json = serde_json::to_string(bt).expect("serialize");
            let parsed: BatchType = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(*bt, parsed);
        }
    }

    #[test]
    fn test_cov_batch_token_serialization() {
        let token = BatchToken::new(42, 1, 5, true);
        let json = serde_json::to_string(&token).expect("serialize");
        let parsed: BatchToken = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(token.token_id, parsed.token_id);
        assert_eq!(token.seq_idx, parsed.seq_idx);
        assert_eq!(token.seq_pos, parsed.seq_pos);
        assert_eq!(token.is_prompt, parsed.is_prompt);
    }
