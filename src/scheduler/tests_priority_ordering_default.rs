
    // === Priority Tests ===

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn test_priority_default() {
        assert_eq!(Priority::default(), Priority::Normal);
    }

    // === SchedulerRequest Tests ===

    #[test]
    fn test_request_new() {
        let request = SchedulerRequest::new(1, vec![1, 2, 3], 10);
        assert_eq!(request.request_id, 1);
        assert_eq!(request.input_ids.len(), 3);
        assert_eq!(request.max_tokens, 10);
        assert_eq!(request.state, SequenceState::Waiting);
    }

    #[test]
    fn test_request_with_priority() {
        let request = SchedulerRequest::new(1, vec![1], 10).with_priority(Priority::High);
        assert_eq!(request.priority, Priority::High);
    }

    #[test]
    fn test_request_total_tokens() {
        let mut request = SchedulerRequest::new(1, vec![1, 2, 3], 10);
        assert_eq!(request.total_tokens(), 3);
        request.generated_tokens = vec![4, 5];
        assert_eq!(request.total_tokens(), 5);
    }

    #[test]
    fn test_request_remaining_tokens() {
        let mut request = SchedulerRequest::new(1, vec![1, 2, 3], 10);
        assert_eq!(request.remaining_tokens(), 10);
        request.generated_tokens = vec![4, 5, 6];
        assert_eq!(request.remaining_tokens(), 7);
    }

    #[test]
    fn test_request_is_complete() {
        let mut request = SchedulerRequest::new(1, vec![1], 3);

        // Not complete initially
        assert!(!request.is_complete(0));

        // Complete by max_tokens
        request.generated_tokens = vec![2, 3, 4];
        assert!(request.is_complete(0));

        // Complete by EOS
        let mut request2 = SchedulerRequest::new(2, vec![1], 10);
        request2.generated_tokens = vec![2, 0]; // 0 is EOS
        assert!(request2.is_complete(0));
    }

    // === SchedulerOutput Tests ===

    #[test]
    fn test_scheduler_output_total_tokens() {
        let output = SchedulerOutput {
            num_prefill_tokens: 100,
            num_decode_tokens: 10,
            ..Default::default()
        };
        assert_eq!(output.total_tokens(), 110);
    }

    #[test]
    fn test_scheduler_output_is_empty() {
        let output = SchedulerOutput::default();
        assert!(output.is_empty());
    }

    // === Scheduler Tests ===

    #[test]
    fn test_scheduler_new() {
        let scheduler = Scheduler::new(32, 1000);
        assert_eq!(scheduler.max_batch_size, 32);
        assert_eq!(scheduler.max_queue_size, 1000);
        assert_eq!(scheduler.waiting_count(), 0);
    }

    #[test]
    fn test_scheduler_add_request() {
        let mut scheduler = Scheduler::new(32, 1000);
        let request_id = scheduler.add_request(vec![1, 2, 3], 10).expect("test");

        assert_eq!(request_id, 0);
        assert_eq!(scheduler.waiting_count(), 1);
        assert_eq!(scheduler.stats().total_requests, 1);
    }

    #[test]
    fn test_scheduler_add_request_queue_full() {
        let mut scheduler = Scheduler::new(32, 1);
        let _ = scheduler.add_request(vec![1], 10).expect("test");

        let result = scheduler.add_request(vec![2], 10);
        assert!(matches!(result, Err(SchedulerError::QueueFull { .. })));
    }

    #[test]
    fn test_scheduler_schedule() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        let _ = scheduler.add_request(vec![1, 2, 3], 10).expect("test");
        let output = scheduler.schedule(&mut kv_cache, 0).expect("test");

        assert_eq!(output.scheduled_request_ids.len(), 1);
        assert_eq!(scheduler.running_count(), 1);
        assert_eq!(scheduler.waiting_count(), 0);
    }

    #[test]
    fn test_scheduler_update_after_iteration() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        let request_id = scheduler.add_request(vec![1], 10).expect("test");
        let _ = scheduler.schedule(&mut kv_cache, 0).expect("test");

        let mut generated = HashMap::new();
        generated.insert(request_id, 42u32);
        scheduler.update_after_iteration(&generated);

        let request = scheduler.get_request(request_id).expect("test");
        assert_eq!(request.generated_tokens, vec![42]);
        assert_eq!(request.iterations, 1);
    }

    #[test]
    fn test_scheduler_complete_request() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        let request_id = scheduler.add_request(vec![1], 10).expect("test");
        let _ = scheduler.schedule(&mut kv_cache, 0).expect("test");

        scheduler.complete_request(request_id, &mut kv_cache);

        assert_eq!(scheduler.running_count(), 0);
        assert_eq!(scheduler.stats().completed_requests, 1);
    }

    #[test]
    fn test_scheduler_priority_ordering() {
        let mut scheduler = Scheduler::new(1, 1000); // Only 1 slot
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        // Add low priority first
        let low_id = scheduler
            .add_request_with_priority(vec![1], 10, Priority::Low)
            .expect("test");
        // Add high priority second
        let _high_id = scheduler
            .add_request_with_priority(vec![2], 10, Priority::High)
            .expect("test");

        // Schedule - should pick high priority
        let output = scheduler.schedule(&mut kv_cache, 0).expect("test");

        // With only 1 slot, we should see preemption or priority selection
        assert_eq!(output.scheduled_request_ids.len(), 1);

        // Low priority should still be waiting or preempted
        let low_request = scheduler.get_request(low_id).expect("test");
        assert!(
            low_request.state == SequenceState::Waiting
                || low_request.state == SequenceState::Preempted
        );
    }

    #[test]
    fn test_scheduler_max_batch_size() {
        let mut scheduler = Scheduler::new(2, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        // Add 3 requests
        let _ = scheduler.add_request(vec![1], 10).expect("test");
        let _ = scheduler.add_request(vec![2], 10).expect("test");
        let _ = scheduler.add_request(vec![3], 10).expect("test");

        let output = scheduler.schedule(&mut kv_cache, 0).expect("test");

        // Should only schedule 2 (max batch size)
        assert_eq!(output.scheduled_request_ids.len(), 2);
        assert_eq!(scheduler.running_count(), 2);
        assert_eq!(scheduler.waiting_count(), 1);
    }

    #[test]
    fn test_scheduler_stats() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        let request_id = scheduler.add_request(vec![1], 10).expect("test");
        let _ = scheduler.schedule(&mut kv_cache, 0).expect("test");
        scheduler.complete_request(request_id, &mut kv_cache);

        let stats = scheduler.stats();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.completed_requests, 1);
    }

    // === Error Display Tests ===

    #[test]
    fn test_scheduler_error_display() {
        let err = SchedulerError::QueueFull { capacity: 100 };
        assert!(err.to_string().contains("100"));

        let err = SchedulerError::RequestNotFound(42);
        assert!(err.to_string().contains("42"));

        let err = SchedulerError::InvalidState("test".to_string());
        assert!(err.to_string().contains("test"));
    }

    // === SequenceState Tests ===

    #[test]
    fn test_sequence_state_variants() {
        let states = [
            SequenceState::Waiting,
            SequenceState::Running,
            SequenceState::Preempted,
            SequenceState::Completed,
            SequenceState::Failed,
        ];
        // Just ensure all variants exist and are distinct
        for (i, s1) in states.iter().enumerate() {
            for (j, s2) in states.iter().enumerate() {
                if i == j {
                    assert_eq!(s1, s2);
                } else {
                    assert_ne!(s1, s2);
                }
            }
        }
    }

    // === Stats Serialization ===

    #[test]
    fn test_scheduler_stats_serialization() {
        let stats = SchedulerStats {
            total_requests: 100,
            completed_requests: 90,
            preemptions: 5,
            avg_wait_time_ms: 10.5,
            avg_ttft_ms: 50.0,
            queue_depth: 10,
            running_count: 8,
        };

        let json = serde_json::to_string(&stats).expect("test");
        let parsed: SchedulerStats = serde_json::from_str(&json).expect("test");

        assert_eq!(parsed.total_requests, stats.total_requests);
        assert_eq!(parsed.preemptions, stats.preemptions);
    }

    // ========================================================================
    // Slot-Based Server Tests
    // ========================================================================

    #[test]
    fn test_slot_state_default() {
        assert_eq!(SlotState::default(), SlotState::Idle);
    }

    #[test]
    fn test_slot_new() {
        let slot = Slot::new(0);
        assert_eq!(slot.id, 0);
        assert!(slot.is_idle());
        assert!(!slot.is_generating());
        assert!(slot.request_id.is_none());
    }

    #[test]
    fn test_slot_assign() {
        let mut slot = Slot::new(0);
        slot.assign(42, vec![1, 2, 3], 10);

        assert_eq!(slot.state, SlotState::Processing);
        assert_eq!(slot.request_id, Some(42));
        assert_eq!(slot.input_tokens, vec![1, 2, 3]);
        assert_eq!(slot.max_tokens, 10);
        assert!(!slot.is_idle());
    }

    #[test]
    fn test_slot_start_generation() {
        let mut slot = Slot::new(0);
        slot.assign(1, vec![1], 10);
        slot.start_generation(5.0);

        assert_eq!(slot.state, SlotState::Generating);
        assert!(slot.is_generating());
        assert_eq!(slot.prompt_time_ms, 5.0);
        assert!(slot.generation_start.is_some());
    }

    #[test]
    fn test_slot_add_token() {
        let mut slot = Slot::new(0);
        slot.assign(1, vec![1], 10);
        slot.start_generation(1.0);

        slot.add_token(100);
        slot.add_token(200);

        assert_eq!(slot.generated_tokens, vec![100, 200]);
    }

    #[test]
    fn test_slot_is_complete_max_tokens() {
        let mut slot = Slot::new(0);
        slot.assign(1, vec![1], 3);
        slot.start_generation(1.0);

        slot.add_token(100);
        assert!(!slot.is_complete(999)); // EOS token

        slot.add_token(200);
        slot.add_token(300);
        assert!(slot.is_complete(999)); // Max tokens reached
    }

    #[test]
    fn test_slot_is_complete_eos() {
        let mut slot = Slot::new(0);
        slot.assign(1, vec![1], 100);
        slot.start_generation(1.0);

        slot.add_token(100);
        assert!(!slot.is_complete(999));

        slot.add_token(999); // EOS token
        assert!(slot.is_complete(999));
    }

    #[test]
    fn test_slot_finish() {
        let mut slot = Slot::new(0);
        slot.assign(1, vec![1], 10);
        slot.start_generation(1.0);
        slot.add_token(100);

        slot.finish();

        assert!(slot.is_idle());
        assert!(slot.request_id.is_none());
        assert!(slot.seq_id.is_none());
    }

    #[test]
    fn test_slot_manager_new() {
        let manager = SlotManager::new(4, 2048);

        assert_eq!(manager.num_slots(), 4);
        assert_eq!(manager.num_idle_slots(), 4);
        assert_eq!(manager.num_active_slots(), 0);
        assert_eq!(manager.max_context_length, 2048);
    }

    #[test]
    fn test_slot_manager_assign_request() {
        let mut manager = SlotManager::new(4, 2048);

        let result = manager.assign_request(vec![1, 2, 3], 10);
        assert!(result.is_some());

        let (slot_id, request_id) = result.expect("test");
        assert_eq!(slot_id, 0);
        assert_eq!(request_id, 0);

        assert_eq!(manager.num_idle_slots(), 3);
        assert_eq!(manager.num_active_slots(), 1);
    }

    #[test]
    fn test_slot_manager_no_slots_available() {
        let mut manager = SlotManager::new(2, 2048);

        // Fill all slots
        manager.assign_request(vec![1], 10).expect("test");
        manager.assign_request(vec![2], 10).expect("test");

        // Third assignment should fail
        let result = manager.assign_request(vec![3], 10);
        assert!(result.is_none());
    }

    #[test]
    fn test_slot_manager_utilization() {
        let mut manager = SlotManager::new(4, 2048);

        assert_eq!(manager.utilization(), 0.0);

        manager.assign_request(vec![1], 10);
        assert!((manager.utilization() - 0.25).abs() < 0.01);

        manager.assign_request(vec![2], 10);
        assert!((manager.utilization() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_slot_manager_batch_slots() {
        let mut manager = SlotManager::new(4, 2048);

        // Assign and start generating on some slots
        manager.assign_request(vec![1], 10);
        manager.assign_request(vec![2], 10);

        manager.get_slot_mut(0).expect("test").start_generation(1.0);

        let batch = manager.batch_slots();
        assert_eq!(batch, vec![0]); // Only slot 0 is generating
    }

    #[test]
    fn test_slot_manager_get_slot() {
        let manager = SlotManager::new(4, 2048);

        assert!(manager.get_slot(0).is_some());
        assert!(manager.get_slot(3).is_some());
        assert!(manager.get_slot(4).is_none()); // Out of bounds
    }
