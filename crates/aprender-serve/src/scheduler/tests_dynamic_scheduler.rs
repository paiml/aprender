
    #[test]
    fn test_dynamic_scheduler_schedule_priority_order() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);

        // Add requests at different priorities
        let low_id = scheduler.add_request(vec![1], 5, Priority::Low, None);
        let normal_id = scheduler.add_request(vec![2], 5, Priority::Normal, None);
        let high_id = scheduler.add_request(vec![3], 5, Priority::High, None);

        // Schedule with 2 slots
        let batch = scheduler.schedule(2);

        // Should schedule high and normal first
        assert_eq!(batch.len(), 2);
        let scheduled_ids: Vec<_> = batch.iter().map(|(id, _)| *id).collect();
        assert!(scheduled_ids.contains(&high_id));
        assert!(scheduled_ids.contains(&normal_id));
        assert!(!scheduled_ids.contains(&low_id));
    }

    #[test]
    fn test_dynamic_scheduler_complete_request() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);

        let id = scheduler.add_simple_request(vec![1], 5);
        let _ = scheduler.schedule(1);

        assert_eq!(scheduler.running_count(), 1);

        let completed = scheduler.complete_request(id);
        assert!(completed.is_some());
        assert_eq!(scheduler.running_count(), 0);
        assert_eq!(scheduler.stats().completed_requests, 1);
    }

    #[test]
    fn test_dynamic_scheduler_sla_compliance() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);

        // Add request with very long deadline (will be met)
        let id = scheduler.add_request(
            vec![1],
            5,
            Priority::Normal,
            Some(Deadline::with_target(100_000)), // 100 seconds
        );

        let _ = scheduler.schedule(1);
        let _ = scheduler.complete_request(id);

        assert_eq!(scheduler.stats().sla_met, 1);
        assert_eq!(scheduler.stats().sla_missed, 0);
        assert!((scheduler.sla_compliance_rate() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_dynamic_scheduler_stats() {
        let scheduler = DynamicPriorityScheduler::new(1024);
        let stats = scheduler.stats();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.completed_requests, 0);
        assert_eq!(stats.promotions, 0);
        assert_eq!(stats.dropped_requests, 0);
    }

    #[test]
    fn test_dynamic_scheduler_stats_serialization() {
        let stats = DynamicSchedulerStats {
            total_requests: 100,
            completed_requests: 90,
            sla_met: 85,
            sla_missed: 5,
            dropped_requests: 10,
            promotions: 20,
            avg_ttft_ms: 50.5,
            p99_ttft_ms: 200.0,
            tokens_by_priority: [100, 500, 300, 100],
            queue_depth_by_priority: [5, 10, 3, 1],
        };

        let json = serde_json::to_string(&stats).expect("test");
        let parsed: DynamicSchedulerStats = serde_json::from_str(&json).expect("test");

        assert_eq!(parsed.total_requests, 100);
        assert_eq!(parsed.sla_met, 85);
    }

    #[test]
    fn test_dynamic_scheduler_get_request() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);
        let id = scheduler.add_simple_request(vec![1, 2, 3], 10);

        let request = scheduler.get_request(id);
        assert!(request.is_some());
        assert_eq!(request.expect("test").input_ids, vec![1, 2, 3]);

        assert!(scheduler.get_request(999).is_none());
    }

    #[test]
    fn test_dynamic_scheduler_config() {
        let config = DynamicPriorityConfig::default().no_promotion();
        let scheduler = DynamicPriorityScheduler::with_config(1024, config);

        assert!(!scheduler.config().enable_age_promotion);
    }

    #[test]
    fn test_dynamic_scheduler_queue_depths() {
        let mut scheduler = DynamicPriorityScheduler::new(1024);

        scheduler.add_request(vec![1], 5, Priority::Low, None);
        scheduler.add_request(vec![2], 5, Priority::Low, None);
        scheduler.add_request(vec![3], 5, Priority::Normal, None);
        scheduler.add_request(vec![4], 5, Priority::High, None);
        scheduler.add_request(vec![5], 5, Priority::Critical, None);

        assert_eq!(scheduler.queue_depth(Priority::Low), 2);
        assert_eq!(scheduler.queue_depth(Priority::Normal), 1);
        assert_eq!(scheduler.queue_depth(Priority::High), 1);
        assert_eq!(scheduler.queue_depth(Priority::Critical), 1);
        assert_eq!(scheduler.waiting_count(), 5);
    }

    #[test]
    fn test_dynamic_scheduler_token_budget_allocation() {
        let mut scheduler = DynamicPriorityScheduler::new(100);

        // Add one request per priority
        scheduler.add_request(vec![1], 50, Priority::Low, None);
        scheduler.add_request(vec![2], 50, Priority::Normal, None);
        scheduler.add_request(vec![3], 50, Priority::High, None);
        scheduler.add_request(vec![4], 50, Priority::Critical, None);

        // Schedule with enough slots
        let batch = scheduler.schedule(4);

        // All should be scheduled, token allocation based on budgets
        assert_eq!(batch.len(), 4);

        // Higher priority should get more tokens
        let stats = scheduler.stats();
        assert!(stats.tokens_by_priority[3] > 0); // Critical
        assert!(stats.tokens_by_priority[2] > 0); // High
    }

    // === Chunked Prefill Tests ===

    #[test]
    fn test_chunked_prefill_config_default() {
        let config = ChunkedPrefillConfig::default();
        assert!(config.enabled);
        assert_eq!(config.chunk_size, 512);
        assert_eq!(config.min_prompt_length, 256);
        assert!(config.allow_decode_interleave);
        assert!(config.boost_partial_prefill);
        assert_eq!(config.max_chunks, 16);
    }

    #[test]
    fn test_chunked_prefill_config_disabled() {
        let config = ChunkedPrefillConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_chunked_prefill_config_low_latency() {
        let config = ChunkedPrefillConfig::low_latency();
        assert!(config.enabled);
        assert_eq!(config.chunk_size, 128);
        assert_eq!(config.min_prompt_length, 64);
    }

    #[test]
    fn test_chunked_prefill_config_high_throughput() {
        let config = ChunkedPrefillConfig::high_throughput();
        assert!(config.enabled);
        assert_eq!(config.chunk_size, 1024);
        assert!(!config.allow_decode_interleave);
    }

    #[test]
    fn test_chunked_prefill_config_with_chunk_size() {
        let config = ChunkedPrefillConfig::default().with_chunk_size(256);
        assert_eq!(config.chunk_size, 256);
    }

    #[test]
    fn test_chunked_prefill_state_new() {
        let state = ChunkedPrefillState::new(1, 1000, 512);
        assert_eq!(state.seq_id, 1);
        assert_eq!(state.total_tokens, 1000);
        assert_eq!(state.processed_tokens, 0);
        assert_eq!(state.current_chunk, 0);
        assert_eq!(state.total_chunks, 2); // 1000 / 512 = 2 (ceiling)
        assert!(!state.is_complete());
    }

    #[test]
    fn test_chunked_prefill_state_next_chunk() {
        let state = ChunkedPrefillState::new(1, 1000, 512);
        let range = state.next_chunk(512);
        assert_eq!(range, 0..512);
    }

    #[test]
    fn test_chunked_prefill_state_advance() {
        let mut state = ChunkedPrefillState::new(1, 1000, 512);
        state.advance(512, 50);
        assert_eq!(state.processed_tokens, 512);
        assert_eq!(state.current_chunk, 1);
        assert_eq!(state.chunk_latencies.len(), 1);
        assert_eq!(state.chunk_latencies[0], 50);

        // Next chunk
        let range = state.next_chunk(512);
        assert_eq!(range, 512..1000);
    }

    #[test]
    fn test_chunked_prefill_state_completion() {
        let mut state = ChunkedPrefillState::new(1, 1000, 512);
        assert!(!state.is_complete());

        state.advance(512, 50);
        assert!(!state.is_complete());

        state.advance(488, 40);
        assert!(state.is_complete());
    }

    #[test]
    fn test_chunked_prefill_state_progress() {
        let mut state = ChunkedPrefillState::new(1, 1000, 512);
        assert!((state.progress() - 0.0).abs() < 0.01);

        state.advance(500, 50);
        assert!((state.progress() - 50.0).abs() < 0.01);

        state.advance(500, 50);
        assert!((state.progress() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_chunked_prefill_state_remaining_tokens() {
        let mut state = ChunkedPrefillState::new(1, 1000, 512);
        assert_eq!(state.remaining_tokens(), 1000);

        state.advance(600, 50);
        assert_eq!(state.remaining_tokens(), 400);
    }

    #[test]
    fn test_chunked_prefill_state_avg_latency() {
        let mut state = ChunkedPrefillState::new(1, 1000, 512);
        assert_eq!(state.avg_chunk_latency_ms(), 0.0);

        state.advance(512, 50);
        assert_eq!(state.avg_chunk_latency_ms(), 50.0);

        state.advance(488, 30);
        assert_eq!(state.avg_chunk_latency_ms(), 40.0);
    }

    #[test]
    fn test_chunked_prefill_state_zero_tokens() {
        let state = ChunkedPrefillState::new(1, 0, 512);
        assert!(state.is_complete());
        assert_eq!(state.progress(), 100.0);
    }

    #[test]
    fn test_chunked_prefill_stats_default() {
        let stats = ChunkedPrefillStats::default();
        assert_eq!(stats.chunked_sequences, 0);
        assert_eq!(stats.bypassed_sequences, 0);
        assert_eq!(stats.chunks_processed, 0);
        assert_eq!(stats.avg_chunk_latency_ms(), 0.0);
        assert_eq!(stats.chunking_rate(), 0.0);
    }

    #[test]
    fn test_chunked_prefill_stats_avg_latency() {
        let stats = ChunkedPrefillStats {
            chunks_processed: 4,
            total_chunk_latency_ms: 200,
            ..Default::default()
        };
        assert_eq!(stats.avg_chunk_latency_ms(), 50.0);
    }

    #[test]
    fn test_chunked_prefill_stats_chunking_rate() {
        let stats = ChunkedPrefillStats {
            chunked_sequences: 3,
            bypassed_sequences: 7,
            ..Default::default()
        };
        assert!((stats.chunking_rate() - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_chunked_prefill_scheduler_new() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        assert_eq!(scheduler.queue_len(), 0);
        assert_eq!(scheduler.pending_count(), 0);
    }

    #[test]
    fn test_chunked_prefill_scheduler_submit_short() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());

        // Short prompt bypasses chunking
        let (seq_id, use_chunking) = scheduler.submit(100); // < min_prompt_length (256)
        assert_eq!(seq_id, 0);
        assert!(!use_chunking);
        assert_eq!(scheduler.stats().bypassed_sequences, 1);
        assert_eq!(scheduler.stats().chunked_sequences, 0);
    }

    #[test]
    fn test_chunked_prefill_scheduler_submit_long() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());

        // Long prompt uses chunking
        let (seq_id, use_chunking) = scheduler.submit(1000); // >= min_prompt_length
        assert_eq!(seq_id, 0);
        assert!(use_chunking);
        assert_eq!(scheduler.stats().chunked_sequences, 1);
        assert_eq!(scheduler.queue_len(), 1);
    }

    #[test]
    fn test_chunked_prefill_scheduler_next_chunk() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());

        scheduler.submit(1000);

        let chunk = scheduler.next_chunk();
        assert!(chunk.is_some());
        let (seq_id, range) = chunk.expect("test");
        assert_eq!(seq_id, 0);
        assert_eq!(range, 0..512);
    }

    #[test]
    fn test_chunked_prefill_scheduler_complete_chunk() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());

        scheduler.submit(1000);
        scheduler.complete_chunk(0, 512, 50);

        assert_eq!(scheduler.stats().chunks_processed, 1);
        assert_eq!(scheduler.stats().total_chunk_latency_ms, 50);
        assert_eq!(scheduler.stats().max_chunk_latency_ms, 50);

        // State should be updated
        let state = scheduler.get_state(0).expect("test");
        assert_eq!(state.processed_tokens, 512);
    }

    #[test]
    fn test_chunked_prefill_scheduler_full_prefill() {
        let mut scheduler =
            ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default().with_chunk_size(512));

        scheduler.submit(1000);

        // First chunk
        let (seq_id, range) = scheduler.next_chunk().expect("test");
        assert_eq!(range, 0..512);
        scheduler.complete_chunk(seq_id, 512, 50);

        // Second chunk
        let (seq_id, range) = scheduler.next_chunk().expect("test");
        assert_eq!(range, 512..1000);
        scheduler.complete_chunk(seq_id, 488, 40);

        // No more chunks
        assert!(scheduler.next_chunk().is_none());
        assert!(!scheduler.has_pending_prefill(0));
    }

    #[test]
    fn test_chunked_prefill_scheduler_has_pending_prefill() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());

        // Non-existent sequence
        assert!(!scheduler.has_pending_prefill(999));

        // New sequence has pending prefill
        scheduler.submit(1000);
        assert!(scheduler.has_pending_prefill(0));

        // Complete prefill
        scheduler.complete_chunk(0, 1000, 100);
        assert!(!scheduler.has_pending_prefill(0));
    }

    #[test]
    fn test_chunked_prefill_scheduler_remove() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());

        scheduler.submit(1000);
        scheduler.submit(2000);
        assert_eq!(scheduler.queue_len(), 2);

        let removed = scheduler.remove(0);
        assert!(removed.is_some());
        assert_eq!(scheduler.queue_len(), 1);
    }

    #[test]
    fn test_chunked_prefill_scheduler_clear() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());

        scheduler.submit(1000);
        scheduler.submit(2000);
        scheduler.clear();

        assert_eq!(scheduler.queue_len(), 0);
        assert_eq!(scheduler.pending_count(), 0);
    }

    #[test]
    fn test_chunked_prefill_scheduler_decode_interleave() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());

        // No interleave when queue is empty
        assert!(!scheduler.should_interleave_decode());

        scheduler.submit(1000);

        // Should interleave when queue has items
        assert!(scheduler.should_interleave_decode());

        scheduler.record_decode_interleave();
        assert_eq!(scheduler.stats().decode_interleaves, 1);
    }
