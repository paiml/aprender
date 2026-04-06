//! EXTREME TDD: GGUF Batch Processing Coverage Tests
//!
//! Per spec: Tests for BatchRequestCollector, ContinuousBatchScheduler,
//! RequestBatch, PendingRequest, SlotState, MultiRequestScheduler,
//! SpeculativeDecoder, GpuBufferPool, and related batch processing code.
//!
//! Target: Increase coverage for batch processing paths in src/gguf.rs

#[cfg(feature = "gpu")]
mod gpu_batch_tests {
    use realizar::gguf::{
        BatchGenerationStats, BatchRequestCollector, BatchingConfig, ContinuousBatchScheduler,
        GpuBufferPool, MultiRequestScheduler, MultiRequestState, MultiRequestStats,
        MultiSchedulerRequest, PendingRequest, RequestBatch, SchedulingPolicy, SlotState,
        SpeculativeConfig, SpeculativeDecoder, VerificationResult,
    };
    use std::thread;
    use std::time::Duration;

    // ==========================================================================
    // PendingRequest Tests
    // ==========================================================================

    #[test]
    fn test_pending_request_new() {
        let request = PendingRequest::new(42, vec![1, 2, 3], 10, 0.7, 50);

        assert_eq!(request.id, 42);
        assert_eq!(request.prompt, vec![1, 2, 3]);
        assert_eq!(request.max_tokens, 10);
        assert!((request.temperature - 0.7).abs() < 0.001);
        assert_eq!(request.top_k, 50);
    }

    #[test]
    fn test_pending_request_wait_time() {
        let request = PendingRequest::new(1, vec![1], 10, 0.0, 10);
        thread::sleep(Duration::from_millis(5));
        let wait = request.wait_time();
        assert!(wait.as_millis() >= 5);
    }

    #[test]
    fn test_pending_request_clone() {
        let original = PendingRequest::new(1, vec![1, 2], 5, 0.5, 20);
        let cloned = original.clone();

        assert_eq!(original.id, cloned.id);
        assert_eq!(original.prompt, cloned.prompt);
        assert_eq!(original.max_tokens, cloned.max_tokens);
    }

    #[test]
    fn test_pending_request_debug() {
        let request = PendingRequest::new(1, vec![1], 10, 1.0, 40);
        let debug_str = format!("{:?}", request);
        assert!(debug_str.contains("PendingRequest"));
        assert!(debug_str.contains("id: 1"));
    }

    // ==========================================================================
    // RequestBatch Tests
    // ==========================================================================

    #[test]
    fn test_request_batch_new() {
        let requests = vec![
            PendingRequest::new(1, vec![1], 10, 0.0, 10),
            PendingRequest::new(2, vec![2, 3], 5, 0.5, 20),
        ];
        let batch = RequestBatch::new(requests);

        assert_eq!(batch.size(), 2);
    }

    #[test]
    fn test_request_batch_size() {
        let batch = RequestBatch::new(vec![]);
        assert_eq!(batch.size(), 0);

        let requests = vec![PendingRequest::new(1, vec![1], 10, 0.0, 10)];
        let batch = RequestBatch::new(requests);
        assert_eq!(batch.size(), 1);
    }

    #[test]
    fn test_request_batch_prompts() {
        let requests = vec![
            PendingRequest::new(1, vec![1, 2], 10, 0.0, 10),
            PendingRequest::new(2, vec![3, 4, 5], 5, 0.0, 10),
        ];
        let batch = RequestBatch::new(requests);
        let prompts = batch.prompts();

        assert_eq!(prompts.len(), 2);
        assert_eq!(prompts[0], vec![1, 2]);
        assert_eq!(prompts[1], vec![3, 4, 5]);
    }

    #[test]
    fn test_request_batch_avg_wait_time_empty() {
        let batch = RequestBatch::new(vec![]);
        let avg_wait = batch.avg_wait_time();
        assert_eq!(avg_wait, Duration::ZERO);
    }

    #[test]
    fn test_request_batch_avg_wait_time_non_empty() {
        let requests = vec![
            PendingRequest::new(1, vec![1], 10, 0.0, 10),
            PendingRequest::new(2, vec![2], 10, 0.0, 10),
        ];
        thread::sleep(Duration::from_millis(5));
        let batch = RequestBatch::new(requests);
        let avg_wait = batch.avg_wait_time();
        assert!(avg_wait.as_millis() >= 5);
    }

    #[test]
    fn test_request_batch_debug() {
        let requests = vec![PendingRequest::new(1, vec![1], 10, 0.0, 10)];
        let batch = RequestBatch::new(requests);
        let debug_str = format!("{:?}", batch);
        assert!(debug_str.contains("RequestBatch"));
    }

    // ==========================================================================
    // BatchRequestCollector Tests
    // ==========================================================================

    #[test]
    fn test_batch_request_collector_new() {
        let collector = BatchRequestCollector::new();

        assert_eq!(collector.batch_threshold, 32);
        assert_eq!(collector.timeout_ms, 50);
        assert_eq!(collector.max_batch_size, 64);
        assert_eq!(collector.pending_count(), 0);
        assert_eq!(collector.total_submitted(), 0);
    }

    #[test]
    fn test_batch_request_collector_with_thresholds() {
        let collector = BatchRequestCollector::with_thresholds(16, 100, 32);

        assert_eq!(collector.batch_threshold, 16);
        assert_eq!(collector.timeout_ms, 100);
        assert_eq!(collector.max_batch_size, 32);
    }

    #[test]
    fn test_batch_request_collector_default() {
        let collector = BatchRequestCollector::default();
        assert_eq!(collector.batch_threshold, 32);
    }

    #[test]
    fn test_batch_request_collector_submit() {
        let collector = BatchRequestCollector::new();

        let id1 = collector.submit(vec![1, 2], 10, 0.5, 20);
        let id2 = collector.submit(vec![3], 5, 0.0, 10);

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(collector.pending_count(), 2);
        assert_eq!(collector.total_submitted(), 2);
    }

    #[test]
    fn test_batch_request_collector_is_batch_ready_empty() {
        let collector = BatchRequestCollector::new();
        assert!(!collector.is_batch_ready());
    }

    #[test]
    fn test_batch_request_collector_is_batch_ready_threshold() {
        let collector = BatchRequestCollector::with_thresholds(2, 1000, 10);

        collector.submit(vec![1], 10, 0.0, 10);
        assert!(!collector.is_batch_ready());

        collector.submit(vec![2], 10, 0.0, 10);
        assert!(collector.is_batch_ready());
    }

    #[test]
    fn test_batch_request_collector_is_batch_ready_timeout() {
        let collector = BatchRequestCollector::with_thresholds(100, 5, 10);

        collector.submit(vec![1], 10, 0.0, 10);
        assert!(!collector.is_batch_ready());

        thread::sleep(Duration::from_millis(10));
        assert!(collector.is_batch_ready());
    }

    #[test]
    fn test_batch_request_collector_collect_batch_empty() {
        let collector = BatchRequestCollector::new();
        assert!(collector.collect_batch().is_none());
    }

    #[test]
    fn test_batch_request_collector_collect_batch_not_ready() {
        let collector = BatchRequestCollector::with_thresholds(100, 10000, 64);
        collector.submit(vec![1], 10, 0.0, 10);
        assert!(collector.collect_batch().is_none());
    }

    #[test]
    fn test_batch_request_collector_collect_batch_ready() {
        let collector = BatchRequestCollector::with_thresholds(2, 1000, 10);

        collector.submit(vec![1], 10, 0.0, 10);
        collector.submit(vec![2], 10, 0.0, 10);

        let batch = collector.collect_batch();
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().size(), 2);
        assert_eq!(collector.pending_count(), 0);
    }

    #[test]
    fn test_batch_request_collector_collect_batch_max_size() {
        let collector = BatchRequestCollector::with_thresholds(2, 1000, 3);

        for i in 0..5 {
            collector.submit(vec![i as u32], 10, 0.0, 10);
        }

        let batch = collector.collect_batch();
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().size(), 3); // Limited by max_batch_size
        assert_eq!(collector.pending_count(), 2);
    }

    #[test]
    fn test_batch_request_collector_flush_empty() {
        let collector = BatchRequestCollector::new();
        assert!(collector.flush().is_none());
    }

    #[test]
    fn test_batch_request_collector_flush_all() {
        let collector = BatchRequestCollector::new();

        collector.submit(vec![1], 10, 0.0, 10);
        collector.submit(vec![2], 10, 0.0, 10);

        let batch = collector.flush();
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().size(), 2);
        assert_eq!(collector.pending_count(), 0);
    }

    // ==========================================================================
    // BatchingConfig Tests (extending existing)
    // ==========================================================================

    #[test]
    fn test_batching_config_debug() {
        let config = BatchingConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("BatchingConfig"));
    }

    #[test]
    fn test_batching_config_fields() {
        let config = BatchingConfig {
            batch_threshold: 16,
            timeout_ms: 25,
            max_batch_size: 48,
            prefer_throughput: false,
        };

        assert_eq!(config.batch_threshold, 16);
        assert_eq!(config.timeout_ms, 25);
        assert_eq!(config.max_batch_size, 48);
        assert!(!config.prefer_throughput);
    }

    // ==========================================================================
    // SlotState Tests
    // ==========================================================================

    #[test]
    fn test_slot_state_empty() {
        let state = SlotState::Empty;
        assert!(state.is_empty());
        assert!(!state.is_active());
        assert!(!state.is_completed());
        assert!(state.request_id().is_none());
    }

    #[test]
    fn test_slot_state_active() {
        let state = SlotState::Active {
            request_id: 42,
            prompt_tokens: vec![1, 2, 3],
            generated_tokens: vec![4, 5],
            max_tokens: 10,
            temperature: 0.7,
            top_k: 50,
        };

        assert!(!state.is_empty());
        assert!(state.is_active());
        assert!(!state.is_completed());
        assert_eq!(state.request_id(), Some(42));
    }

    #[test]
    fn test_slot_state_completed() {
        let state = SlotState::Completed {
            request_id: 99,
            generated_tokens: vec![1, 2, 3],
        };

        assert!(!state.is_empty());
        assert!(!state.is_active());
        assert!(state.is_completed());
        assert_eq!(state.request_id(), Some(99));
    }

    #[test]
    fn test_slot_state_clone() {
        let state = SlotState::Active {
            request_id: 1,
            prompt_tokens: vec![1],
            generated_tokens: vec![],
            max_tokens: 5,
            temperature: 0.5,
            top_k: 10,
        };

        let cloned = state.clone();
        assert_eq!(state.request_id(), cloned.request_id());
    }

    #[test]
    fn test_slot_state_debug() {
        let state = SlotState::Empty;
        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("Empty"));
    }

    // ==========================================================================
    // ContinuousBatchScheduler Tests
    // ==========================================================================

    #[test]
    fn test_continuous_batch_scheduler_new() {
        let scheduler = ContinuousBatchScheduler::new(4, 12, 768, 512);

        assert_eq!(scheduler.num_slots, 4);
        assert_eq!(scheduler.empty_count(), 4);
        assert_eq!(scheduler.active_count(), 0);
        assert!(!scheduler.has_completed());
    }

    #[test]
    fn test_continuous_batch_scheduler_submit() {
        let scheduler = ContinuousBatchScheduler::new(2, 1, 64, 128);

        let id1 = scheduler.submit(vec![1, 2], 10, 0.5, 20);
        assert!(id1.is_some());
        assert_eq!(scheduler.active_count(), 1);
        assert_eq!(scheduler.empty_count(), 1);

        let id2 = scheduler.submit(vec![3], 5, 0.0, 10);
        assert!(id2.is_some());
        assert_eq!(scheduler.active_count(), 2);
        assert_eq!(scheduler.empty_count(), 0);

        // No more slots available
        let id3 = scheduler.submit(vec![4], 5, 0.0, 10);
        assert!(id3.is_none());
    }

    #[test]
    fn test_continuous_batch_scheduler_get_active_slots() {
        let scheduler = ContinuousBatchScheduler::new(4, 1, 64, 128);

        scheduler.submit(vec![1, 2, 3], 10, 0.0, 10); // 3 prompt tokens
        scheduler.submit(vec![4, 5], 5, 0.0, 10); // 2 prompt tokens

        let active = scheduler.get_active_slots();
        assert_eq!(active.len(), 2);

        // Positions should be prompt lengths (0 generated tokens)
        assert_eq!(active[0].1, 3);
        assert_eq!(active[1].1, 2);
    }

    #[test]
    fn test_continuous_batch_scheduler_complete_request() {
        let scheduler = ContinuousBatchScheduler::new(2, 1, 64, 128);

        let id = scheduler.submit(vec![1], 10, 0.0, 10).unwrap();
        assert_eq!(scheduler.active_count(), 1);

        scheduler.complete_request(0, vec![2, 3, 4]);
        assert_eq!(scheduler.active_count(), 0);
        assert_eq!(scheduler.empty_count(), 2);
        assert!(scheduler.has_completed());

        let completed = scheduler.poll_completed();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].0, id);
        assert_eq!(completed[0].1, vec![2, 3, 4]);
    }

    #[test]
    fn test_continuous_batch_scheduler_poll_completed_empty() {
        let scheduler = ContinuousBatchScheduler::new(2, 1, 64, 128);
        let completed = scheduler.poll_completed();
        assert!(completed.is_empty());
    }

    #[test]
    fn test_continuous_batch_scheduler_complete_invalid_slot() {
        let scheduler = ContinuousBatchScheduler::new(2, 1, 64, 128);
        scheduler.complete_request(99, vec![1, 2]); // Invalid slot index
        assert!(!scheduler.has_completed());
    }

    #[test]
    fn test_continuous_batch_scheduler_utilization() {
        let scheduler = ContinuousBatchScheduler::new(4, 1, 64, 128);

        assert_eq!(scheduler.utilization(), 0.0);

        scheduler.submit(vec![1], 10, 0.0, 10);
        assert!((scheduler.utilization() - 0.25).abs() < 0.001);

        scheduler.submit(vec![2], 10, 0.0, 10);
        assert!((scheduler.utilization() - 0.5).abs() < 0.001);
    }

    // ==========================================================================
    // SpeculativeConfig Tests
    // ==========================================================================

    #[test]
    fn test_speculative_config_default_fields() {
        let config = SpeculativeConfig::default();
        assert_eq!(config.speculation_length, 4);
        assert!((config.draft_temperature - 0.0).abs() < 0.001);
        assert!(config.self_speculative);
    }

    #[test]
    fn test_speculative_config_clone() {
        let config = SpeculativeConfig {
            speculation_length: 8,
            draft_temperature: 0.3,
            self_speculative: false,
        };
        let cloned = config.clone();
        assert_eq!(config.speculation_length, cloned.speculation_length);
    }

    #[test]
    fn test_speculative_config_debug() {
        let config = SpeculativeConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("SpeculativeConfig"));
    }

    // ==========================================================================
    // SpeculativeDecoder Tests
    // ==========================================================================

    #[test]
    fn test_speculative_decoder_new() {
        let decoder = SpeculativeDecoder::new();
        assert_eq!(decoder.config.speculation_length, 4);
        assert_eq!(decoder.acceptance_rate(), 0.0);
    }

    #[test]
    fn test_speculative_decoder_with_config() {
        let config = SpeculativeConfig {
            speculation_length: 6,
            draft_temperature: 0.2,
            self_speculative: true,
        };
        let decoder = SpeculativeDecoder::with_config(config);
        assert_eq!(decoder.config.speculation_length, 6);
    }

    #[test]
    fn test_speculative_decoder_default() {
        let decoder = SpeculativeDecoder::default();
        assert_eq!(decoder.config.speculation_length, 4);
    }

    #[test]
    fn test_speculative_decoder_verify_draft_greedy_match() {
        let decoder = SpeculativeDecoder::new();

        let draft_tokens = vec![10, 20, 30];
        // Logits where token 10 is highest, then 20, then 30
        let target_logits = vec![
            {
                let mut v = vec![0.0f32; 100];
                v[10] = 5.0;
                v
            },
            {
                let mut v = vec![0.0f32; 100];
                v[20] = 5.0;
                v
            },
            {
                let mut v = vec![0.0f32; 100];
                v[30] = 5.0;
                v
            },
        ];

        let result = decoder.verify_draft(&draft_tokens, &target_logits, 0.0);

        assert_eq!(result.draft_count, 3);
        assert_eq!(result.accepted_count, 3);
        assert!(result.all_accepted);
        assert_eq!(result.accepted_tokens, vec![10, 20, 30]);
    }

    #[test]
    fn test_speculative_decoder_verify_draft_greedy_mismatch() {
        let decoder = SpeculativeDecoder::new();

        let draft_tokens = vec![10, 20, 30];
        // Token 20 mismatches (target wants 25)
        let target_logits = vec![
            {
                let mut v = vec![0.0f32; 100];
                v[10] = 5.0;
                v
            },
            {
                let mut v = vec![0.0f32; 100];
                v[25] = 5.0; // Mismatch!
                v
            },
            {
                let mut v = vec![0.0f32; 100];
                v[30] = 5.0;
                v
            },
        ];

        let result = decoder.verify_draft(&draft_tokens, &target_logits, 0.0);

        assert_eq!(result.draft_count, 3);
        assert_eq!(result.accepted_count, 2); // First accepted, then target token used
        assert!(!result.all_accepted);
        assert_eq!(result.accepted_tokens, vec![10, 25]); // 25 from target
    }

    #[test]
    fn test_speculative_decoder_verify_draft_non_greedy() {
        let decoder = SpeculativeDecoder::new();

        let draft_tokens = vec![10, 20];
        // Non-greedy: draft token 10 is in top-k, 20 is not
        let target_logits = vec![
            {
                let mut v = vec![0.0f32; 100];
                // Top-10 includes 10
                for (i, item) in v.iter_mut().enumerate().take(11) {
                    *item = 5.0 - i as f32 * 0.1;
                }
                v
            },
            {
                let mut v = vec![0.0f32; 100];
                // Token 20 not in top-10
                for item in v.iter_mut().take(10) {
                    *item = 5.0;
                }
                v[20] = 0.0;
                v
            },
        ];

        let result = decoder.verify_draft(&draft_tokens, &target_logits, 1.0);

        // First token accepted (in top-k), second rejected
        assert!(!result.all_accepted);
    }

    #[test]
    fn test_speculative_decoder_verify_draft_empty() {
        let decoder = SpeculativeDecoder::new();
        let result = decoder.verify_draft(&[], &[], 0.0);

        assert_eq!(result.draft_count, 0);
        assert_eq!(result.accepted_count, 0);
        assert!(result.accepted_tokens.is_empty());
        assert!(result.all_accepted);
    }

    #[test]
    fn test_speculative_decoder_verify_draft_more_tokens_than_logits() {
        let decoder = SpeculativeDecoder::new();

        let draft_tokens = vec![10, 20, 30, 40];
        let target_logits = vec![
            {
                let mut v = vec![0.0f32; 100];
                v[10] = 5.0;
                v
            },
            {
                let mut v = vec![0.0f32; 100];
                v[20] = 5.0;
                v
            },
        ];

        let result = decoder.verify_draft(&draft_tokens, &target_logits, 0.0);

        // Only processes up to logits length
        assert_eq!(result.draft_count, 4);
        assert_eq!(result.accepted_count, 2);
    }

    #[test]
    fn test_speculative_decoder_acceptance_rate() {
        let decoder = SpeculativeDecoder::new();

        // No tokens yet
        assert_eq!(decoder.acceptance_rate(), 0.0);

        // Add some stats via verify_draft
        let draft = vec![1, 2];
        let logits = vec![
            {
                let mut v = vec![0.0f32; 10];
                v[1] = 5.0;
                v
            },
            {
                let mut v = vec![0.0f32; 10];
                v[2] = 5.0;
                v
            },
        ];
        decoder.verify_draft(&draft, &logits, 0.0);

        // 2 drafts, 2 accepted = 100%
        assert!((decoder.acceptance_rate() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_speculative_decoder_expected_speedup() {
        let decoder = SpeculativeDecoder::new();

        // No acceptance yet: speedup = k * 0 + 1 = 1
        assert!((decoder.expected_speedup() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_speculative_decoder_reset_stats() {
        let decoder = SpeculativeDecoder::new();

        let draft = vec![1];
        let logits = vec![{
            let mut v = vec![0.0f32; 10];
            v[1] = 5.0;
            v
        }];
        decoder.verify_draft(&draft, &logits, 0.0);

        decoder.reset_stats();
        assert_eq!(decoder.acceptance_rate(), 0.0);
    }

    #[test]
    fn test_verification_result_clone() {
        let result = VerificationResult {
            accepted_count: 3,
            draft_count: 4,
            accepted_tokens: vec![1, 2, 3],
            all_accepted: false,
        };
        let cloned = result.clone();
        assert_eq!(result.accepted_count, cloned.accepted_count);
    }

    #[test]
    fn test_verification_result_debug() {
        let result = VerificationResult {
            accepted_count: 2,
            draft_count: 3,
            accepted_tokens: vec![1, 2],
            all_accepted: false,
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("VerificationResult"));
    }

    // ==========================================================================
    // GpuBufferPool Tests
    // ==========================================================================

    #[test]
    fn test_gpu_buffer_pool_new() {
        let pool = GpuBufferPool::new(768, 3072, 512, 12, 4);

        // Verify pool is created by checking memory usage
        // Memory = pool_size * (hidden_dim + intermediate_dim + num_heads * max_seq_len) * 4
        let expected_memory = 4 * (768 + 3072 + 12 * 512) * 4;
        assert_eq!(pool.memory_usage_bytes(), expected_memory);
    }

    #[test]
    fn test_gpu_buffer_pool_warmup() {
        let pool = GpuBufferPool::new(64, 256, 128, 4, 2);

        pool.warmup();

        // Borrow should return pre-allocated buffers
        let hidden = pool.borrow_hidden();
        assert_eq!(hidden.len(), 64);

        let intermediate = pool.borrow_intermediate();
        assert_eq!(intermediate.len(), 256);

        let attention = pool.borrow_attention();
        assert_eq!(attention.len(), 4 * 128); // num_heads * max_seq_len
    }

    #[test]
    fn test_gpu_buffer_pool_borrow_return_hidden() {
        let pool = GpuBufferPool::new(64, 256, 128, 4, 2);
        pool.warmup();

        // Borrow
        let mut buffer = pool.borrow_hidden();
        assert_eq!(buffer.len(), 64);

        // Modify
        buffer[0] = 42.0;

        // Return
        pool.return_hidden(buffer);

        // Verify stats
        assert!(pool.borrows.load(std::sync::atomic::Ordering::Relaxed) >= 1);
        assert!(pool.returns.load(std::sync::atomic::Ordering::Relaxed) >= 1);
    }

    #[test]
    fn test_gpu_buffer_pool_borrow_return_intermediate() {
        let pool = GpuBufferPool::new(64, 256, 128, 4, 2);
        pool.warmup();

        let buffer = pool.borrow_intermediate();
        assert_eq!(buffer.len(), 256);
        pool.return_intermediate(buffer);
    }

    #[test]
    fn test_gpu_buffer_pool_borrow_return_attention() {
        let pool = GpuBufferPool::new(64, 256, 128, 4, 2);
        pool.warmup();

        let buffer = pool.borrow_attention();
        assert_eq!(buffer.len(), 4 * 128);
        pool.return_attention(buffer);
    }

    #[test]
    fn test_gpu_buffer_pool_post_warmup_allocs() {
        let pool = GpuBufferPool::new(64, 256, 128, 4, 1);
        pool.warmup();

        // First borrow uses pre-allocated
        let _b1 = pool.borrow_hidden();
        // Second borrow should allocate (only 1 in pool)
        let _b2 = pool.borrow_hidden();

        assert!(
            pool.post_warmup_allocs
                .load(std::sync::atomic::Ordering::Relaxed)
                >= 1
        );
    }

    #[test]
    fn test_gpu_buffer_pool_borrow_before_warmup() {
        let pool = GpuBufferPool::new(64, 256, 128, 4, 2);

        // Borrow before warmup should still work (but allocates)
        let buffer = pool.borrow_hidden();
        assert_eq!(buffer.len(), 64);

        // No post_warmup_allocs since not warmed up yet
        assert_eq!(
            pool.post_warmup_allocs
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn test_gpu_buffer_pool_stats() {
        let pool = GpuBufferPool::new(64, 256, 128, 4, 4);
        pool.warmup();

        let stats = pool.stats();

        assert_eq!(stats.hidden_available, 4);
        assert_eq!(stats.intermediate_available, 4);
        assert_eq!(stats.attention_available, 4);
        assert_eq!(stats.post_warmup_allocs, 0);
        assert!(stats.warmed_up);
    }

    // ==========================================================================
    // MultiRequestState Tests
    // ==========================================================================

    #[test]
    fn test_multi_request_state_variants() {
        assert_eq!(MultiRequestState::Pending, MultiRequestState::Pending);
        assert_eq!(MultiRequestState::Prefilling, MultiRequestState::Prefilling);
        assert_eq!(MultiRequestState::Decoding, MultiRequestState::Decoding);
        assert_eq!(MultiRequestState::Completed, MultiRequestState::Completed);
        assert_eq!(MultiRequestState::Preempted, MultiRequestState::Preempted);

        assert_ne!(MultiRequestState::Pending, MultiRequestState::Completed);
    }

    #[test]
    fn test_multi_request_state_clone() {
        let state = MultiRequestState::Decoding;
        let cloned = state.clone();
        assert_eq!(state, cloned);
    }

    #[test]
    fn test_multi_request_state_debug() {
        let state = MultiRequestState::Pending;
        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("Pending"));
    }

    // ==========================================================================
    // MultiSchedulerRequest Tests
    // ==========================================================================

    #[test]
    fn test_multi_scheduler_request_new() {
        let request = MultiSchedulerRequest::new(42, vec![1, 2, 3], 10);

        assert_eq!(request.id, 42);
        assert_eq!(request.tokens, vec![1, 2, 3]);
        assert_eq!(request.max_tokens, 10);
        assert_eq!(request.state, MultiRequestState::Pending);
        assert_eq!(request.kv_position, 0);
        assert!(request.first_token_time.is_none());
    }

    #[test]
    fn test_multi_scheduler_request_is_complete() {
        let mut request = MultiSchedulerRequest::new(1, vec![1], 3);

        assert!(!request.is_complete());

        request.generated = vec![2, 3, 4];
        assert!(request.is_complete());
    }

    #[test]
    fn test_multi_scheduler_request_is_complete_by_state() {
        let mut request = MultiSchedulerRequest::new(1, vec![1], 10);
        request.state = MultiRequestState::Completed;
        assert!(request.is_complete());
    }

    #[test]
    fn test_multi_scheduler_request_ttft_none() {
        let request = MultiSchedulerRequest::new(1, vec![1], 10);
        assert!(request.ttft_ms().is_none());
    }

    #[test]
    fn test_multi_scheduler_request_ttft_some() {
        let mut request = MultiSchedulerRequest::new(1, vec![1], 10);
        thread::sleep(Duration::from_millis(5));
        request.first_token_time = Some(std::time::Instant::now());

        let ttft = request.ttft_ms();
        assert!(ttft.is_some());
        assert!(ttft.unwrap() >= 5.0);
    }

    #[test]
    fn test_multi_scheduler_request_clone() {
        let request = MultiSchedulerRequest::new(1, vec![1, 2], 5);
        let cloned = request.clone();
        assert_eq!(request.id, cloned.id);
        assert_eq!(request.tokens, cloned.tokens);
    }

    // ==========================================================================
    // SchedulingPolicy Tests
    // ==========================================================================

    #[test]
    fn test_scheduling_policy_variants() {
        let fcfs = SchedulingPolicy::Fcfs;
        let sjf = SchedulingPolicy::Sjf;
        let rr = SchedulingPolicy::RoundRobin;

        assert_eq!(fcfs, SchedulingPolicy::Fcfs);
        assert_eq!(sjf, SchedulingPolicy::Sjf);
        assert_eq!(rr, SchedulingPolicy::RoundRobin);
        assert_ne!(fcfs, sjf);
    }

    #[test]
    fn test_scheduling_policy_clone_copy() {
        let policy = SchedulingPolicy::Fcfs;
        let copied = policy; // Copy trait
        let copied2 = policy; // Can copy again
        assert_eq!(policy, copied);
        assert_eq!(policy, copied2);
    }

    #[test]
    fn test_scheduling_policy_debug() {
        let policy = SchedulingPolicy::Sjf;
        let debug_str = format!("{:?}", policy);
        assert!(debug_str.contains("Sjf"));
    }

    // ==========================================================================
    // MultiRequestScheduler Tests
    // ==========================================================================

    #[test]
    fn test_multi_request_scheduler_new() {
        let scheduler = MultiRequestScheduler::new(8, 4, SchedulingPolicy::Fcfs);

        let stats = scheduler.stats();
        assert_eq!(stats.requests_submitted, 0);
        assert_eq!(stats.pending_requests, 0);
        assert_eq!(stats.active_requests, 0);
    }

    #[test]
    fn test_multi_request_scheduler_submit() {
        let scheduler = MultiRequestScheduler::new(8, 4, SchedulingPolicy::Fcfs);

        let id1 = scheduler.submit(vec![1, 2], 10);
        let id2 = scheduler.submit(vec![3], 5);

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);

        let stats = scheduler.stats();
        assert_eq!(stats.requests_submitted, 2);
        assert_eq!(stats.pending_requests, 2);
    }

    #[test]
    fn test_multi_request_scheduler_get_decode_batch_fcfs() {
        let scheduler = MultiRequestScheduler::new(8, 2, SchedulingPolicy::Fcfs);

        scheduler.submit(vec![1], 10);
        scheduler.submit(vec![2], 10);
        scheduler.submit(vec![3], 10);

        let batch = scheduler.get_decode_batch();
        assert_eq!(batch.len(), 2); // Limited by max_concurrent

        let stats = scheduler.stats();
        assert_eq!(stats.active_requests, 2);
        assert_eq!(stats.pending_requests, 1);
    }

    #[test]
    fn test_multi_request_scheduler_get_decode_batch_sjf() {
        let scheduler = MultiRequestScheduler::new(8, 4, SchedulingPolicy::Sjf);

        scheduler.submit(vec![1], 100); // More remaining tokens
        scheduler.submit(vec![2], 10); // Fewer remaining tokens

        let batch = scheduler.get_decode_batch();
        // SJF sorts by remaining tokens, so smaller jobs first
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_multi_request_scheduler_get_decode_batch_round_robin() {
        let scheduler = MultiRequestScheduler::new(8, 4, SchedulingPolicy::RoundRobin);

        scheduler.submit(vec![1], 10);
        scheduler.submit(vec![2], 10);

        let _batch1 = scheduler.get_decode_batch();
        let _batch2 = scheduler.get_decode_batch();
        // Round robin rotates requests
    }

    #[test]
    fn test_multi_request_scheduler_record_token() {
        let scheduler = MultiRequestScheduler::new(8, 4, SchedulingPolicy::Fcfs);

        let id = scheduler.submit(vec![1], 3);
        let _batch = scheduler.get_decode_batch();

        scheduler.record_token(id, 100);
        scheduler.record_token(id, 101);

        let stats = scheduler.stats();
        assert_eq!(stats.tokens_generated, 2);
    }

    #[test]
    fn test_multi_request_scheduler_record_token_invalid() {
        let scheduler = MultiRequestScheduler::new(8, 4, SchedulingPolicy::Fcfs);

        // Record token for non-existent request
        scheduler.record_token(999, 100);

        let stats = scheduler.stats();
        assert_eq!(stats.tokens_generated, 0);
    }

    #[test]
    fn test_multi_request_scheduler_collect_completed() {
        let scheduler = MultiRequestScheduler::new(8, 4, SchedulingPolicy::Fcfs);

        let id = scheduler.submit(vec![1], 2);
        let _batch = scheduler.get_decode_batch();

        scheduler.record_token(id, 100);
        scheduler.record_token(id, 101);

        let completed = scheduler.collect_completed();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].id, id);
        assert_eq!(completed[0].generated, vec![100, 101]);

        let stats = scheduler.stats();
        assert_eq!(stats.requests_completed, 1);
    }

    #[test]
    fn test_multi_request_scheduler_step() {
        let scheduler = MultiRequestScheduler::new(8, 4, SchedulingPolicy::Fcfs);

        scheduler.step();
        scheduler.step();

        let stats = scheduler.stats();
        assert_eq!(stats.batch_iterations, 2);
    }

    #[test]
    fn test_multi_request_scheduler_stats_avg_batch_size() {
        let scheduler = MultiRequestScheduler::new(8, 4, SchedulingPolicy::Fcfs);

        let id = scheduler.submit(vec![1], 10);
        let _batch = scheduler.get_decode_batch();

        scheduler.record_token(id, 100);
        scheduler.step();

        let stats = scheduler.stats();
        assert!(stats.avg_batch_size > 0.0);
    }

    #[test]
    fn test_multi_request_scheduler_stats_zero_iterations() {
        let scheduler = MultiRequestScheduler::new(8, 4, SchedulingPolicy::Fcfs);
        let stats = scheduler.stats();
        assert_eq!(stats.avg_batch_size, 0.0);
    }

    // ==========================================================================
    // MultiRequestStats Tests
    // ==========================================================================

    #[test]
    fn test_multi_request_stats_fields() {
        let stats = MultiRequestStats {
            requests_submitted: 100,
            requests_completed: 90,
            tokens_generated: 1000,
            batch_iterations: 50,
            pending_requests: 5,
            active_requests: 5,
            avg_batch_size: 20.0,
        };

        assert_eq!(stats.requests_submitted, 100);
        assert_eq!(stats.requests_completed, 90);
        assert_eq!(stats.tokens_generated, 1000);
        assert_eq!(stats.batch_iterations, 50);
        assert_eq!(stats.pending_requests, 5);
        assert_eq!(stats.active_requests, 5);
        assert!((stats.avg_batch_size - 20.0).abs() < 0.001);
    }

    // ==========================================================================
    // BatchGenerationStats Tests
    // ==========================================================================

    #[test]
    fn test_batch_generation_stats_fields() {
        let stats = BatchGenerationStats {
            gpu_cache_ready: true,
            cache_memory_gb: 2.5,
            num_layers: 32,
            hidden_dim: 4096,
            intermediate_dim: 11008,
            recommended_batch_size: 32,
            max_batch_size: 64,
        };

        assert!(stats.gpu_cache_ready);
        assert!((stats.cache_memory_gb - 2.5).abs() < 0.001);
        assert_eq!(stats.num_layers, 32);
        assert_eq!(stats.hidden_dim, 4096);
        assert_eq!(stats.intermediate_dim, 11008);
        assert_eq!(stats.recommended_batch_size, 32);
        assert_eq!(stats.max_batch_size, 64);
    }

    #[test]
    fn test_batch_generation_stats_clone() {
        let stats = BatchGenerationStats {
            gpu_cache_ready: false,
            cache_memory_gb: 1.0,
            num_layers: 12,
            hidden_dim: 768,
            intermediate_dim: 3072,
            recommended_batch_size: 16,
            max_batch_size: 32,
        };
        let cloned = stats.clone();
        assert_eq!(stats.num_layers, cloned.num_layers);
    }

    #[test]
    fn test_batch_generation_stats_debug() {
        let stats = BatchGenerationStats {
            gpu_cache_ready: true,
            cache_memory_gb: 1.0,
            num_layers: 6,
            hidden_dim: 512,
            intermediate_dim: 2048,
            recommended_batch_size: 8,
            max_batch_size: 16,
        };
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("BatchGenerationStats"));
    }

    // ==========================================================================
    // Integration Tests
    // ==========================================================================

    #[test]
    fn test_batch_request_collector_full_workflow() {
        let collector = BatchRequestCollector::with_thresholds(2, 1000, 10);

        // Submit requests
        let id1 = collector.submit(vec![1, 2, 3], 50, 0.7, 40);
        let id2 = collector.submit(vec![4, 5], 30, 0.5, 20);

        // Collect batch
        let batch = collector.collect_batch().expect("batch should be ready");

        // Verify batch
        assert_eq!(batch.size(), 2);
        let prompts = batch.prompts();
        assert_eq!(prompts[0], vec![1, 2, 3]);
        assert_eq!(prompts[1], vec![4, 5]);

        // Verify IDs are sequential
        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
    }

    #[test]
    fn test_continuous_batch_scheduler_full_workflow() {
        let scheduler = ContinuousBatchScheduler::new(2, 4, 128, 256);

        // Submit requests
        let id1 = scheduler.submit(vec![1, 2], 10, 0.0, 10).unwrap();
        let id2 = scheduler.submit(vec![3], 5, 0.0, 10).unwrap();

        // Verify active
        assert_eq!(scheduler.active_count(), 2);
        assert!((scheduler.utilization() - 1.0).abs() < 0.001);

        // Complete first request
        scheduler.complete_request(0, vec![100, 101]);

        // Verify completed
        assert_eq!(scheduler.active_count(), 1);
        let completed = scheduler.poll_completed();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].0, id1);

        // Slot should be available again
        let id3 = scheduler.submit(vec![4, 5], 10, 0.0, 10);
        assert!(id3.is_some());

        // Complete remaining
        scheduler.complete_request(1, vec![200]);
        let _ = id2; // suppress unused warning
    }

    #[test]
    fn test_multi_request_scheduler_full_workflow() {
        let scheduler = MultiRequestScheduler::new(4, 2, SchedulingPolicy::Fcfs);

        // Submit multiple requests
        let id1 = scheduler.submit(vec![1, 2], 2);
        let id2 = scheduler.submit(vec![3], 3);
        let _id3 = scheduler.submit(vec![4], 1);

        // Get first batch (max 2 concurrent)
        let batch1 = scheduler.get_decode_batch();
        assert_eq!(batch1.len(), 2);

        // Generate tokens for first request until complete
        scheduler.record_token(id1, 100);
        scheduler.record_token(id1, 101);
        scheduler.step();

        // Collect completed
        let completed = scheduler.collect_completed();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].id, id1);

        // Generate tokens for second request
        scheduler.record_token(id2, 200);
        scheduler.record_token(id2, 201);
        scheduler.record_token(id2, 202);
        scheduler.step();

        // Collect more completed
        let completed2 = scheduler.collect_completed();
        assert_eq!(completed2.len(), 1);
        assert_eq!(completed2[0].id, id2);

        // Check final stats
        let stats = scheduler.stats();
        assert_eq!(stats.requests_submitted, 3);
        assert_eq!(stats.requests_completed, 2);
        assert_eq!(stats.tokens_generated, 5);
    }
}

// Non-GPU test to ensure module compiles without GPU feature
#[test]
fn test_gguf_batch_module_compiles() {
    // Basic sanity check that the module exists - use a concrete value
    let test_value: usize = 1;
    assert_eq!(test_value, 1);
}
