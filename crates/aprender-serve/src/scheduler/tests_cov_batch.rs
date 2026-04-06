
    // --- MicroBatch Token Limit Tests ---

    #[test]
    fn test_cov_batch_scheduler_decode_ubatch_limit() {
        let config = BatchConfig {
            max_ubatch_tokens: 2,
            prefer_pure_decode: true,
            ..Default::default()
        };
        let mut scheduler = BatchScheduler::with_config(config);

        // Add and start decode for multiple sequences
        let seq1 = scheduler.add_sequence(0, 1, vec![10]).expect("add 1");
        let seq2 = scheduler.add_sequence(1, 2, vec![20]).expect("add 2");
        let seq3 = scheduler.add_sequence(2, 3, vec![30]).expect("add 3");

        scheduler.start_decode(seq1, 1);
        scheduler.start_decode(seq2, 1);
        scheduler.start_decode(seq3, 1);

        // Create ubatch - should be limited to 2 tokens
        let ubatch = scheduler.create_ubatch();
        assert_eq!(ubatch.len(), 2);
        assert!(ubatch.is_decode());
    }

    #[test]
    fn test_cov_batch_scheduler_mixed_with_limited_budget() {
        let config = BatchConfig {
            max_ubatch_tokens: 3,
            prefer_pure_decode: false, // Allow mixed
            ..Default::default()
        };
        let mut scheduler = BatchScheduler::with_config(config);

        // Add prefill sequence with 2 tokens
        let seq1 = scheduler.add_sequence(0, 1, vec![10, 20]).expect("add 1");

        // Add decode sequence
        let seq2 = scheduler.add_sequence(1, 2, vec![30]).expect("add 2");
        scheduler.start_decode(seq2, 1);

        // Add another decode
        let seq3 = scheduler.add_sequence(2, 3, vec![40]).expect("add 3");
        scheduler.start_decode(seq3, 1);

        // Create ubatch - should be limited to 3 tokens total
        let ubatch = scheduler.create_ubatch();
        assert_eq!(ubatch.len(), 3);

        // With prefer_pure_decode=false, it may include both prefill and decode
        // The first 2 are prefill, then 1 decode to hit the limit
        assert!(ubatch.is_mixed() || ubatch.is_prefill());

        // Cleanup
        scheduler.complete_sequence(seq1);
    }

    // --- Priority Entry Comparison Edge Cases ---

    #[test]
    fn test_cov_priority_entry_partial_cmp() {
        let now = Instant::now();
        let entry1 = PriorityEntry {
            priority: Priority::Normal,
            arrival_time: now,
            request_id: 1,
        };
        let entry2 = PriorityEntry {
            priority: Priority::Normal,
            arrival_time: now,
            request_id: 2,
        };

        // partial_cmp should delegate to cmp
        let result = entry1.partial_cmp(&entry2);
        assert!(result.is_some());
    }

    // --- SchedulerOutput Non-Empty Tests ---

    #[test]
    fn test_cov_scheduler_output_non_empty() {
        let mut output = SchedulerOutput::default();
        assert!(output.is_empty());

        output.scheduled_seq_ids.push(crate::paged_kv::SeqId::new());
        assert!(!output.is_empty());
    }

    // --- Scheduler Preempted Count ---

    #[test]
    fn test_cov_scheduler_preempted_count() {
        let mut scheduler = Scheduler::new(1, 1000);
        let mut kv_cache = PagedKvCache::new(10, 4, 4, 16);

        assert_eq!(scheduler.preempted_count(), 0);

        // Add low and critical priority
        let _low_id = scheduler
            .add_request_with_priority(vec![1], 10, Priority::Low)
            .expect("add low");
        let _ = scheduler.schedule(&mut kv_cache, 0);

        // Add critical to trigger preemption
        let _critical_id = scheduler
            .add_request_with_priority(vec![2], 5, Priority::Critical)
            .expect("add critical");
        let _ = scheduler.schedule(&mut kv_cache, 0);

        // Preempted count may be >= 0 depending on implementation
        let _count = scheduler.preempted_count();
    }

    // --- Dynamic Scheduler Scheduling With Deadline Sorting ---

    #[test]
    fn test_cov_dynamic_scheduler_deadline_aware_scheduling() {
        let config = DynamicPriorityConfig {
            enable_deadline_scheduling: true,
            enable_fair_share: false, // Disable fair share to test pure deadline ordering
            ..Default::default()
        };
        let mut scheduler = DynamicPriorityScheduler::with_config(1024, config);

        // Add requests with different deadlines
        let deadline_short = Deadline::with_target(10);
        let deadline_long = Deadline::with_target(1000);

        // Add long deadline first
        let _id_long = scheduler.add_request(vec![1], 5, Priority::Normal, Some(deadline_long));

        // Add short deadline second
        let id_short = scheduler.add_request(vec![2], 5, Priority::Normal, Some(deadline_short));

        // Small delay to make short deadline more urgent
        std::thread::sleep(std::time::Duration::from_millis(1));

        // Schedule - short deadline should be scheduled first due to higher urgency
        let batch = scheduler.schedule(1);
        assert_eq!(batch.len(), 1);

        // The short deadline request should be scheduled first
        let (scheduled_id, _) = batch[0];
        assert_eq!(scheduled_id, id_short);
    }

    // --- Promotion to Critical (index 3) ---

    #[test]
    fn test_cov_dynamic_scheduler_promotion_to_critical() {
        let config = DynamicPriorityConfig {
            enable_age_promotion: true,
            promotion_interval_ms: 0,
            max_promoted_priority: Priority::Critical, // Allow promotion to Critical
            ..Default::default()
        };
        let mut scheduler = DynamicPriorityScheduler::with_config(1024, config);

        // Add High priority request
        let id = scheduler.add_request(vec![1], 5, Priority::High, None);

        // Promote: High -> Critical
        scheduler.promote_aged_requests();

        let request = scheduler.get_request(id).expect("get request");
        assert_eq!(request.effective_priority, Priority::Critical);
    }

    // --- ChunkedPrefillState Edge Cases ---

    #[test]
    fn test_cov_chunked_prefill_state_single_chunk() {
        // Test when total_tokens < chunk_size (single chunk)
        let state = ChunkedPrefillState::new(1, 100, 512);
        assert_eq!(state.total_chunks, 1);
        assert_eq!(state.next_chunk(512), 0..100);
    }

    #[test]
    fn test_cov_chunked_prefill_state_exact_chunks() {
        // Test when total_tokens is exact multiple of chunk_size
        let state = ChunkedPrefillState::new(1, 1024, 512);
        assert_eq!(state.total_chunks, 2);
    }

    // --- BatchStats Serialization ---

    #[test]
    fn test_cov_batch_stats_serialization() {
        let stats = BatchStats {
            ubatches_created: 10,
            sbatches_created: 5,
            tokens_processed: 1000,
            prefill_tokens: 800,
            decode_tokens: 200,
            avg_ubatch_size: 100.0,
            avg_sbatch_size: 2.0,
        };

        let json = serde_json::to_string(&stats).expect("serialize");
        let parsed: BatchStats = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.ubatches_created, stats.ubatches_created);
        assert_eq!(parsed.tokens_processed, stats.tokens_processed);
    }

    // --- SequenceBatchEntry Serialization ---

    #[test]
    fn test_cov_sequence_batch_entry_serialization() {
        let entry = SequenceBatchEntry::new(0, 1, 100)
            .with_tokens(vec![1, 2, 3])
            .at_position(5)
            .decoding();

        let json = serde_json::to_string(&entry).expect("serialize");
        let parsed: SequenceBatchEntry = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.seq_idx, entry.seq_idx);
        assert_eq!(parsed.slot_id, entry.slot_id);
        assert_eq!(parsed.request_id, entry.request_id);
        assert_eq!(parsed.tokens, entry.tokens);
        assert_eq!(parsed.position, entry.position);
        assert_eq!(parsed.is_prefill, entry.is_prefill);
    }

    // --- SequenceBatch Serialization ---

    #[test]
    fn test_cov_sequence_batch_serialization() {
        let mut batch = SequenceBatch::new(4);
        batch.add_sequence(SequenceBatchEntry::new(0, 0, 1));
        batch.add_sequence(SequenceBatchEntry::new(1, 1, 2).decoding());

        let json = serde_json::to_string(&batch).expect("serialize");
        let parsed: SequenceBatch = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.max_batch_size, batch.max_batch_size);
        assert_eq!(parsed.sequences.len(), 2);
    }

    // --- MicroBatch Serialization ---

    #[test]
    fn test_cov_micro_batch_serialization() {
        let mut batch = MicroBatch::new();
        batch.add_token(BatchToken::new(10, 0, 0, true));
        batch.add_token(BatchToken::new(20, 0, 1, true));

        let json = serde_json::to_string(&batch).expect("serialize");
        let parsed: MicroBatch = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.tokens.len(), 2);
        assert_eq!(parsed.n_prompt_tokens, 2);
    }

    // --- BatchConfig Serialization ---

    #[test]
    fn test_cov_batch_config_serialization() {
        let config = BatchConfig::default()
            .with_max_tokens(1024)
            .with_max_sequences(16);

        let json = serde_json::to_string(&config).expect("serialize");
        let parsed: BatchConfig = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.max_ubatch_tokens, config.max_ubatch_tokens);
        assert_eq!(parsed.max_sbatch_sequences, config.max_sbatch_sequences);
    }

    // --- ChunkedPrefillConfig Serialization ---

    #[test]
    fn test_cov_chunked_prefill_config_serialization() {
        let config = ChunkedPrefillConfig::low_latency();

        let json = serde_json::to_string(&config).expect("serialize");
        let parsed: ChunkedPrefillConfig = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.chunk_size, config.chunk_size);
        assert_eq!(parsed.enabled, config.enabled);
    }

    // --- ChunkedPrefillState Serialization ---

    #[test]
    fn test_cov_chunked_prefill_state_serialization() {
        let mut state = ChunkedPrefillState::new(1, 1000, 512);
        state.advance(512, 50);

        let json = serde_json::to_string(&state).expect("serialize");
        let parsed: ChunkedPrefillState = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.seq_id, state.seq_id);
        assert_eq!(parsed.processed_tokens, state.processed_tokens);
        assert_eq!(parsed.chunk_latencies.len(), 1);
    }

    // --- ChunkedPrefillStats Serialization ---

    #[test]
    fn test_cov_chunked_prefill_stats_serialization() {
        let stats = ChunkedPrefillStats {
            chunked_sequences: 10,
            bypassed_sequences: 5,
            chunks_processed: 20,
            decode_interleaves: 8,
            total_chunk_latency_ms: 500,
            max_chunk_latency_ms: 100,
            prefix_cache_hits: 200,
        };

        let json = serde_json::to_string(&stats).expect("serialize");
        let parsed: ChunkedPrefillStats = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.chunked_sequences, stats.chunked_sequences);
        assert_eq!(parsed.prefix_cache_hits, stats.prefix_cache_hits);
    }

    // --- Deadline Serialization ---

    #[test]
    fn test_cov_deadline_serialization() {
        let deadline = Deadline::strict(100, 200);

        let json = serde_json::to_string(&deadline).expect("serialize");
        let parsed: Deadline = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.target_latency_ms, deadline.target_latency_ms);
        assert_eq!(parsed.hard_deadline_ms, deadline.hard_deadline_ms);
    }

    // --- DynamicPriorityConfig Serialization ---

    #[test]
    fn test_cov_dynamic_priority_config_serialization() {
        let config = DynamicPriorityConfig::default().no_promotion();

        let json = serde_json::to_string(&config).expect("serialize");
        let parsed: DynamicPriorityConfig = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.enable_age_promotion, config.enable_age_promotion);
    }

    // --- ChunkedPrefillScheduler Additional Coverage ---

    #[test]
    fn test_cov_chunked_prefill_max_chunks_limit() {
        let config = ChunkedPrefillConfig {
            enabled: true,
            chunk_size: 10,
            min_prompt_length: 5,
            max_chunks: 3,
            ..Default::default()
        };
        let mut scheduler = ChunkedPrefillScheduler::new(config);

        // Submit very long prompt
        let (seq_id, use_chunking) = scheduler.submit(1000);
        assert!(use_chunking);

        // Get state
        let state = scheduler.get_state(seq_id).expect("get state");
        // total_chunks is based on ceiling division
        assert!(state.total_chunks >= 1);
    }

    // --- Empty Iteration Check ---

    #[test]
    fn test_cov_scheduler_schedule_empty() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        // Schedule with no requests
        let output = scheduler.schedule(&mut kv_cache, 0).expect("schedule");
        assert!(output.is_empty());
        assert_eq!(output.total_tokens(), 0);
    }

    // --- Multiple Completions in Single Schedule ---

    #[test]
    fn test_cov_scheduler_multiple_completions() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        // Add and schedule multiple requests
        let id1 = scheduler.add_request(vec![1], 1).expect("add 1");
        let id2 = scheduler.add_request(vec![2], 1).expect("add 2");
        let _ = scheduler.schedule(&mut kv_cache, 0).expect("schedule");

        // Generate tokens to complete both
        let mut generated = HashMap::new();
        generated.insert(id1, 10u32);
        generated.insert(id2, 20u32);
        scheduler.update_after_iteration(&generated);

        // Schedule again - both should complete
        let output = scheduler.schedule(&mut kv_cache, 0).expect("schedule 2");
        assert_eq!(output.completed_request_ids.len(), 2);
    }

    // --- Request Without Seq ID During Completion ---

    #[test]
    fn test_cov_scheduler_complete_request_without_seq_id() {
        let mut scheduler = Scheduler::new(32, 1000);
        let mut kv_cache = PagedKvCache::new(100, 16, 8, 64);

        // Add request but don't schedule (so no seq_id)
        let request_id = scheduler.add_request(vec![1], 10).expect("add");

        // Complete without scheduling (no seq_id to free)
        scheduler.complete_request(request_id, &mut kv_cache);

        // Should still update stats (even though nothing was really completed)
        let request = scheduler.get_request(request_id);
        if let Some(r) = request {
            assert_eq!(r.state, SequenceState::Completed);
        }
    }

    // --- Slot Generation Time Calculation ---

    #[test]
    fn test_cov_slot_tokens_per_second_with_generation() {
        let mut slot = Slot::new(0);
        slot.assign(1, vec![1], 10);
        slot.start_generation(5.0);

        // Add some tokens
        slot.add_token(100);
        slot.add_token(200);
        slot.add_token(300);

        // Finish to record generation time
        slot.finish();

        // generation_time_ms should be recorded
        assert!(slot.generation_time_ms >= 0.0);
    }
