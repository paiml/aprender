
// ============================================================================
// Tests (T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ChunkedPrefillConfig Tests
    // =========================================================================

    #[test]
    fn test_config_default() {
        let config = ChunkedPrefillConfig::default();
        assert!(config.enabled);
        assert_eq!(config.chunk_size, 512);
        assert_eq!(config.min_prompt_length, 256);
        assert!(config.allow_decode_interleave);
        assert!(config.boost_partial_prefill);
        assert_eq!(config.max_chunks, 16);
    }

    #[test]
    fn test_config_disabled() {
        let config = ChunkedPrefillConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_config_low_latency() {
        let config = ChunkedPrefillConfig::low_latency();
        assert!(config.enabled);
        assert_eq!(config.chunk_size, 128);
        assert_eq!(config.min_prompt_length, 64);
        assert!(config.allow_decode_interleave);
        assert!(config.boost_partial_prefill);
        assert_eq!(config.max_chunks, 32);
    }

    #[test]
    fn test_config_high_throughput() {
        let config = ChunkedPrefillConfig::high_throughput();
        assert!(config.enabled);
        assert_eq!(config.chunk_size, 1024);
        assert_eq!(config.min_prompt_length, 512);
        assert!(!config.allow_decode_interleave);
        assert!(!config.boost_partial_prefill);
        assert_eq!(config.max_chunks, 8);
    }

    #[test]
    fn test_config_with_chunk_size() {
        let config = ChunkedPrefillConfig::default().with_chunk_size(256);
        assert_eq!(config.chunk_size, 256);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let config = ChunkedPrefillConfig::low_latency();
        let json = serde_json::to_string(&config).expect("serialize");
        let restored: ChunkedPrefillConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(config.enabled, restored.enabled);
        assert_eq!(config.chunk_size, restored.chunk_size);
        assert_eq!(config.min_prompt_length, restored.min_prompt_length);
    }

    // =========================================================================
    // ChunkedPrefillState Tests
    // =========================================================================

    #[test]
    fn test_state_new() {
        let state = ChunkedPrefillState::new(42, 1024, 256);
        assert_eq!(state.seq_id, 42);
        assert_eq!(state.total_tokens, 1024);
        assert_eq!(state.processed_tokens, 0);
        assert_eq!(state.current_chunk, 0);
        assert_eq!(state.total_chunks, 4); // 1024 / 256 = 4
        assert!(state.chunk_latencies.is_empty());
    }

    #[test]
    fn test_state_total_chunks_round_up() {
        // 1000 tokens / 256 = 3.9, should round up to 4
        let state = ChunkedPrefillState::new(1, 1000, 256);
        assert_eq!(state.total_chunks, 4);
    }

    #[test]
    fn test_state_next_chunk() {
        let state = ChunkedPrefillState::new(1, 1000, 256);
        let range = state.next_chunk(256);
        assert_eq!(range, 0..256);
    }

    #[test]
    fn test_state_next_chunk_last() {
        let mut state = ChunkedPrefillState::new(1, 1000, 256);
        state.processed_tokens = 768;
        let range = state.next_chunk(256);
        // Last chunk: 768..1000 (232 tokens, not 256)
        assert_eq!(range, 768..1000);
    }

    #[test]
    fn test_state_advance() {
        let mut state = ChunkedPrefillState::new(1, 1000, 256);
        state.advance(256, 50);
        assert_eq!(state.processed_tokens, 256);
        assert_eq!(state.current_chunk, 1);
        assert_eq!(state.chunk_latencies, vec![50]);
    }

    #[test]
    fn test_state_is_complete() {
        let mut state = ChunkedPrefillState::new(1, 1000, 256);
        assert!(!state.is_complete());
        state.processed_tokens = 1000;
        assert!(state.is_complete());
    }

    #[test]
    fn test_state_progress() {
        let mut state = ChunkedPrefillState::new(1, 1000, 256);
        assert!((state.progress() - 0.0).abs() < f64::EPSILON);
        state.processed_tokens = 500;
        assert!((state.progress() - 50.0).abs() < f64::EPSILON);
        state.processed_tokens = 1000;
        assert!((state.progress() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_state_progress_zero_tokens() {
        let state = ChunkedPrefillState::new(1, 0, 256);
        assert!((state.progress() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_state_remaining_tokens() {
        let mut state = ChunkedPrefillState::new(1, 1000, 256);
        assert_eq!(state.remaining_tokens(), 1000);
        state.processed_tokens = 600;
        assert_eq!(state.remaining_tokens(), 400);
        state.processed_tokens = 1000;
        assert_eq!(state.remaining_tokens(), 0);
    }

    #[test]
    fn test_state_avg_chunk_latency_empty() {
        let state = ChunkedPrefillState::new(1, 1000, 256);
        assert!((state.avg_chunk_latency_ms() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_state_avg_chunk_latency() {
        let mut state = ChunkedPrefillState::new(1, 1000, 256);
        state.advance(256, 10);
        state.advance(256, 20);
        state.advance(256, 30);
        // Average: (10 + 20 + 30) / 3 = 20
        assert!((state.avg_chunk_latency_ms() - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_state_serde_roundtrip() {
        let mut state = ChunkedPrefillState::new(42, 1000, 256);
        state.advance(256, 25);
        let json = serde_json::to_string(&state).expect("serialize");
        let restored: ChunkedPrefillState = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(state.seq_id, restored.seq_id);
        assert_eq!(state.total_tokens, restored.total_tokens);
        assert_eq!(state.processed_tokens, restored.processed_tokens);
        assert_eq!(state.chunk_latencies, restored.chunk_latencies);
    }

    // =========================================================================
    // ChunkedPrefillStats Tests
    // =========================================================================

    #[test]
    fn test_stats_default() {
        let stats = ChunkedPrefillStats::default();
        assert_eq!(stats.chunked_sequences, 0);
        assert_eq!(stats.bypassed_sequences, 0);
        assert_eq!(stats.chunks_processed, 0);
        assert_eq!(stats.decode_interleaves, 0);
        assert_eq!(stats.total_chunk_latency_ms, 0);
        assert_eq!(stats.max_chunk_latency_ms, 0);
        assert_eq!(stats.prefix_cache_hits, 0);
    }

    #[test]
    fn test_stats_avg_chunk_latency_zero() {
        let stats = ChunkedPrefillStats::default();
        assert!((stats.avg_chunk_latency_ms() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_avg_chunk_latency() {
        let stats = ChunkedPrefillStats {
            chunks_processed: 4,
            total_chunk_latency_ms: 100,
            ..Default::default()
        };
        assert!((stats.avg_chunk_latency_ms() - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_chunking_rate_zero() {
        let stats = ChunkedPrefillStats::default();
        assert!((stats.chunking_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_chunking_rate() {
        let stats = ChunkedPrefillStats {
            chunked_sequences: 3,
            bypassed_sequences: 1,
            ..Default::default()
        };
        // 3 / 4 = 0.75
        assert!((stats.chunking_rate() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_serde_roundtrip() {
        let stats = ChunkedPrefillStats {
            chunked_sequences: 10,
            bypassed_sequences: 5,
            chunks_processed: 30,
            decode_interleaves: 20,
            total_chunk_latency_ms: 3000,
            max_chunk_latency_ms: 200,
            prefix_cache_hits: 100,
        };
        let json = serde_json::to_string(&stats).expect("serialize");
        let restored: ChunkedPrefillStats = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(stats.chunked_sequences, restored.chunked_sequences);
        assert_eq!(stats.bypassed_sequences, restored.bypassed_sequences);
        assert_eq!(stats.chunks_processed, restored.chunks_processed);
    }

    // =========================================================================
    // ChunkedPrefillScheduler Tests
    // =========================================================================

    #[test]
    fn test_scheduler_new() {
        let scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::default());
        assert_eq!(scheduler.pending_count(), 0);
        assert_eq!(scheduler.queue_len(), 0);
    }

    #[test]
    fn test_scheduler_default() {
        let scheduler = ChunkedPrefillScheduler::default();
        assert!(scheduler.config().enabled);
    }

    #[test]
    fn test_scheduler_submit_short_prompt() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        // 100 tokens < 256 min_prompt_length, should bypass chunking
        let (seq_id, use_chunking) = scheduler.submit(100);
        assert_eq!(seq_id, 0);
        assert!(!use_chunking);
        assert_eq!(scheduler.stats().bypassed_sequences, 1);
        assert_eq!(scheduler.stats().chunked_sequences, 0);
    }

    #[test]
    fn test_scheduler_submit_long_prompt() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        // 1000 tokens >= 256 min_prompt_length, should use chunking
        let (seq_id, use_chunking) = scheduler.submit(1000);
        assert_eq!(seq_id, 0);
        assert!(use_chunking);
        assert_eq!(scheduler.stats().chunked_sequences, 1);
        assert_eq!(scheduler.stats().bypassed_sequences, 0);
        assert_eq!(scheduler.pending_count(), 1);
    }

    #[test]
    fn test_scheduler_submit_disabled() {
        let mut scheduler = ChunkedPrefillScheduler::new(ChunkedPrefillConfig::disabled());
        let (_, use_chunking) = scheduler.submit(1000);
        assert!(!use_chunking);
    }

    #[test]
    fn test_scheduler_next_chunk() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        scheduler.submit(1000);

        let result = scheduler.next_chunk();
        assert!(result.is_some());
        let (seq_id, range) = result.expect("result");
        assert_eq!(seq_id, 0);
        assert_eq!(range, 0..512); // Default chunk_size is 512
    }

    #[test]
    fn test_scheduler_next_chunk_empty() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        assert!(scheduler.next_chunk().is_none());
    }

    #[test]
    fn test_scheduler_complete_chunk() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        scheduler.submit(1000);

        scheduler.complete_chunk(0, 512, 100);

        assert_eq!(scheduler.stats().chunks_processed, 1);
        assert_eq!(scheduler.stats().total_chunk_latency_ms, 100);
        assert_eq!(scheduler.stats().max_chunk_latency_ms, 100);

        let state = scheduler.get_state(0).expect("state exists");
        assert_eq!(state.processed_tokens, 512);
    }

    #[test]
    fn test_scheduler_complete_all_chunks() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        scheduler.submit(1000);

        // Complete in two chunks
        scheduler.complete_chunk(0, 512, 100);
        scheduler.complete_chunk(0, 488, 90);

        assert!(scheduler.get_state(0).expect("get_state").is_complete());
        assert_eq!(scheduler.pending_count(), 0);
    }

    #[test]
    fn test_scheduler_record_decode_interleave() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        scheduler.record_decode_interleave();
        assert_eq!(scheduler.stats().decode_interleaves, 1);
    }

    #[test]
    fn test_scheduler_should_interleave_decode() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        // No pending prefills, should not interleave
        assert!(!scheduler.should_interleave_decode());

        scheduler.submit(1000);
        // Now has pending prefills, should interleave
        assert!(scheduler.should_interleave_decode());
    }

    #[test]
    fn test_scheduler_should_interleave_disabled() {
        let config = ChunkedPrefillConfig {
            allow_decode_interleave: false,
            ..Default::default()
        };
        let mut scheduler = ChunkedPrefillScheduler::new(config);
        scheduler.submit(1000);
        assert!(!scheduler.should_interleave_decode());
    }

    #[test]
    fn test_scheduler_get_state() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        scheduler.submit(1000);

        let state = scheduler.get_state(0);
        assert!(state.is_some());
        assert_eq!(state.expect("state").total_tokens, 1000);

        let missing = scheduler.get_state(999);
        assert!(missing.is_none());
    }

    #[test]
    fn test_scheduler_has_pending_prefill() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        scheduler.submit(1000);

        assert!(scheduler.has_pending_prefill(0));

        // Complete all chunks
        scheduler.complete_chunk(0, 512, 100);
        scheduler.complete_chunk(0, 488, 100);

        assert!(!scheduler.has_pending_prefill(0));
    }

    #[test]
    fn test_scheduler_remove() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        scheduler.submit(1000);

        let removed = scheduler.remove(0);
        assert!(removed.is_some());
        assert_eq!(removed.expect("removed").seq_id, 0);
        assert!(scheduler.get_state(0).is_none());
        assert_eq!(scheduler.queue_len(), 0);
    }

    #[test]
    fn test_scheduler_clear() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        scheduler.submit(1000);
        scheduler.submit(2000);

        scheduler.clear();

        assert_eq!(scheduler.pending_count(), 0);
        assert_eq!(scheduler.queue_len(), 0);
    }

    #[test]
    fn test_scheduler_record_prefix_cache_hit() {
        let mut scheduler = ChunkedPrefillScheduler::default();
        scheduler.record_prefix_cache_hit(100);
        assert_eq!(scheduler.stats().prefix_cache_hits, 100);

        scheduler.record_prefix_cache_hit(50);
        assert_eq!(scheduler.stats().prefix_cache_hits, 150);
    }

    #[test]
    fn test_scheduler_multiple_sequences() {
        let mut scheduler = ChunkedPrefillScheduler::default();

        let (id1, _) = scheduler.submit(1000);
        let (id2, _) = scheduler.submit(2000);
        let (id3, _) = scheduler.submit(500);

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 2);

        assert_eq!(scheduler.stats().chunked_sequences, 3);
        assert_eq!(scheduler.pending_count(), 3);
    }
}
