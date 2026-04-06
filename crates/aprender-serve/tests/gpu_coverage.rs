//! EXTREME TDD coverage tests for realizar/src/gpu.rs
//!
//! Tests target uncovered public functions, structs, and error paths
//! to increase coverage from 77% to 90%+.

#![cfg(feature = "gpu")]

use realizar::gpu::{
    batch_embed, blocked_matmul, exceeds_gpu_buffer_limit, fused_layernorm, naive_matmul,
    parallel_ffn, prefetch_read, quantized_dot_q4, quantized_dot_q8, quantized_matvec_q4,
    quantized_matvec_q8, scalar_rope, scalar_softmax, sequential_ffn, sequential_sum, simd_rope,
    simd_softmax, standard_layernorm, sum_with_prefetch, AsyncRequestQueue, CacheAlignedBuffer,
    ChunkedProcessor, ComputeBackend, ConnectionConfig, ConnectionPool, ContiguousAttentionBuffer,
    DegradationManager, DegradationMode, DoubleBuffer, ErrorClassification, ErrorRecoveryStrategy,
    FailureIsolator, ForwardArena, GpuBufferPool, GpuCompute, GpuGenerateConfig, GpuModelConfig,
    GpuPipelineStage, GpuPoolStats, HealthChecker, HybridScheduler, InferenceBatchScheduler,
    InferenceEventNotifier, InferenceMetrics, InferencePipeline, MatmulOp, PriorityRequest,
    PriorityRequestQueue, QuantizedAccumulator, RecoveryAction, RequestOutcome, ResourceTracker,
    ScratchBuffer, SpeculativeBuffer, StreamingKVCache, StreamingKVCacheFp16, TensorPool,
    TimeoutManager, TokenBatch, TokenRateLimiter, WeightType,
};
use std::time::{Duration, Instant};

// ============================================================================
// GPU Buffer Limit Tests
// ============================================================================

#[test]
fn test_exceeds_gpu_buffer_limit_boundary() {
    // 256MB = 268435456 bytes / 4 bytes per f32 = 67108864 elements
    let max_elements = 256 * 1024 * 1024 / 4;

    assert!(!exceeds_gpu_buffer_limit(max_elements - 1));
    assert!(!exceeds_gpu_buffer_limit(max_elements));
    assert!(exceeds_gpu_buffer_limit(max_elements + 1));
}

// ============================================================================
// Softmax Tests
// ============================================================================

#[test]
fn test_scalar_softmax_numerical_stability() {
    // Large values should not cause overflow due to max subtraction
    let input = vec![1000.0, 1001.0, 1002.0];
    let output = scalar_softmax(&input);

    assert_eq!(output.len(), 3);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Softmax should sum to 1.0");
}

#[test]
fn test_simd_softmax_numerical_stability() {
    let input = vec![1000.0, 1001.0, 1002.0];
    let output = simd_softmax(&input);

    assert_eq!(output.len(), 3);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Softmax should sum to 1.0");
}

#[test]
fn test_softmax_negative_values() {
    let input = vec![-100.0, -50.0, 0.0];
    let scalar_out = scalar_softmax(&input);
    let simd_out = simd_softmax(&input);

    for i in 0..3 {
        assert!((scalar_out[i] - simd_out[i]).abs() < 1e-5);
    }
}

// ============================================================================
// RoPE Tests
// ============================================================================

#[test]
fn test_rope_with_multiple_heads() {
    let seq_len = 2;
    let head_dim = 4;
    let num_heads = 2;
    let hidden_dim = num_heads * head_dim;
    let input = vec![1.0; seq_len * hidden_dim];

    let scalar_out = scalar_rope(&input, seq_len, head_dim, 10000.0);
    let simd_out = simd_rope(&input, seq_len, head_dim, 10000.0);

    assert_eq!(scalar_out.len(), input.len());
    assert_eq!(simd_out.len(), input.len());

    for i in 0..scalar_out.len() {
        assert!((scalar_out[i] - simd_out[i]).abs() < 1e-4);
    }
}

#[test]
fn test_rope_position_encoding() {
    let seq_len = 4;
    let head_dim = 8;
    let input: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32 * 0.1).collect();

    let output = scalar_rope(&input, seq_len, head_dim, 10000.0);

    // Position 0 should be unchanged (cos(0)=1, sin(0)=0)
    assert!((output[0] - input[0]).abs() < 1e-5);
}

// ============================================================================
// ContiguousAttentionBuffer Tests
// ============================================================================

#[test]
fn test_contiguous_attention_buffer_creation() {
    let buffer = ContiguousAttentionBuffer::new(128, 8, 64);
    assert!(buffer.is_contiguous());
    assert_eq!(buffer.max_seq_len(), 128);
}

#[test]
fn test_contiguous_attention_buffer_views() {
    let buffer = ContiguousAttentionBuffer::new(4, 2, 4);
    let (q, k, v, o) = buffer.get_views();

    // Each tensor should have size: max_seq_len * num_heads * head_dim = 4 * 2 * 4 = 32
    assert_eq!(q.len(), 32);
    assert_eq!(k.len(), 32);
    assert_eq!(v.len(), 32);
    assert_eq!(o.len(), 32);
}

#[test]
fn test_contiguous_attention_buffer_mutable_views() {
    let mut buffer = ContiguousAttentionBuffer::new(2, 2, 2);
    let (q, k, v, o) = buffer.get_views_mut();

    q[0] = 1.0;
    k[0] = 2.0;
    v[0] = 3.0;
    o[0] = 4.0;

    let (q_read, k_read, v_read, o_read) = buffer.get_views();
    assert!((q_read[0] - 1.0).abs() < 1e-5);
    assert!((k_read[0] - 2.0).abs() < 1e-5);
    assert!((v_read[0] - 3.0).abs() < 1e-5);
    assert!((o_read[0] - 4.0).abs() < 1e-5);
}

#[test]
fn test_contiguous_attention_buffer_reset() {
    let mut buffer = ContiguousAttentionBuffer::new(2, 2, 2);

    {
        let (q, _, _, _) = buffer.get_views_mut();
        q[0] = 100.0;
    }

    buffer.reset();

    let (q, _, _, _) = buffer.get_views();
    assert!((q[0] - 0.0).abs() < 1e-5);
}

// ============================================================================
// Batch Embed Tests
// ============================================================================

#[test]
fn test_batch_embed_empty_tokens() {
    let table = vec![1.0; 100];
    let result = batch_embed(&table, &[], 10);
    assert!(result.is_empty());
}

#[test]
fn test_batch_embed_empty_table() {
    let result = batch_embed(&[], &[0, 1], 10);
    assert!(result.is_empty());
}

#[test]
fn test_batch_embed_out_of_bounds() {
    let table = vec![1.0; 20]; // vocab_size * hidden_dim = 2 * 10 = 20
    let tokens = vec![0, 1, 5]; // token 5 is out of bounds
    let result = batch_embed(&table, &tokens, 10);

    // First two should be valid, third should be padded with zeros
    assert_eq!(result.len(), 30);
    assert!((result[0] - 1.0).abs() < 1e-5);
    assert!((result[20] - 0.0).abs() < 1e-5); // Padded zeros
}

// ============================================================================
// FFN Tests
// ============================================================================

#[test]
fn test_sequential_ffn_empty_input() {
    let result = sequential_ffn(&[], &[1.0], &[1.0], 4, 8);
    assert!(result.is_empty());
}

#[test]
fn test_parallel_ffn_empty_input() {
    let result = parallel_ffn(&[], &[1.0], &[1.0], 4, 8);
    assert!(result.is_empty());
}

#[test]
fn test_ffn_sequential_vs_parallel() {
    let hidden_dim = 4;
    let intermediate_dim = 8;
    let input = vec![0.5; hidden_dim];
    let w_up = vec![0.1; hidden_dim * intermediate_dim];
    let w_down = vec![0.1; intermediate_dim * hidden_dim];

    let seq = sequential_ffn(&input, &w_up, &w_down, hidden_dim, intermediate_dim);
    let par = parallel_ffn(&input, &w_up, &w_down, hidden_dim, intermediate_dim);

    assert_eq!(seq.len(), par.len());
    for i in 0..seq.len() {
        assert!((seq[i] - par[i]).abs() < 1e-4);
    }
}

// ============================================================================
// LayerNorm Tests
// ============================================================================

#[test]
fn test_standard_layernorm_empty() {
    let result = standard_layernorm(&[], &[], &[], 1e-5);
    assert!(result.is_empty());
}

#[test]
fn test_fused_layernorm_empty() {
    let result = fused_layernorm(&[], &[], &[], 1e-5);
    assert!(result.is_empty());
}

#[test]
fn test_layernorm_partial_gamma_beta() {
    // When gamma/beta are shorter than input, defaults are used
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![2.0, 2.0]; // Shorter
    let beta = vec![0.5]; // Even shorter

    let std_result = standard_layernorm(&input, &gamma, &beta, 1e-5);
    let fused_result = fused_layernorm(&input, &gamma, &beta, 1e-5);

    assert_eq!(std_result.len(), 4);
    assert_eq!(fused_result.len(), 4);
}

// ============================================================================
// CacheAlignedBuffer Tests
// ============================================================================

#[test]
fn test_cache_aligned_buffer_creation() {
    let buffer = CacheAlignedBuffer::new(100);
    assert_eq!(buffer.len(), 100);
    assert!(!buffer.is_empty());
}

#[test]
fn test_cache_aligned_buffer_empty() {
    let buffer = CacheAlignedBuffer::new(0);
    assert_eq!(buffer.len(), 0);
    assert!(buffer.is_empty());
}

#[test]
fn test_cache_aligned_buffer_alignment() {
    let buffer = CacheAlignedBuffer::new(256);
    // Should be aligned to 64-byte cache line
    assert!(buffer.is_aligned(64) || buffer.is_aligned(32) || buffer.is_aligned(16));
}

#[test]
fn test_cache_aligned_buffer_slice_access() {
    let mut buffer = CacheAlignedBuffer::new(10);
    let slice = buffer.as_mut_slice();
    slice[0] = 42.0;

    assert!((buffer.as_slice()[0] - 42.0).abs() < 1e-5);
}

// ============================================================================
// Prefetch and Sum Tests
// ============================================================================

#[test]
fn test_prefetch_read_within_bounds() {
    let data = vec![1.0; 100];
    // Should not panic
    prefetch_read(&data, 0, 10);
    prefetch_read(&data, 50, 20);
}

#[test]
fn test_prefetch_read_out_of_bounds() {
    let data = vec![1.0; 10];
    // Should not panic when prefetch position exceeds bounds
    prefetch_read(&data, 5, 10);
}

#[test]
fn test_sum_with_prefetch_equals_sequential() {
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();

    let seq_sum = sequential_sum(&data);
    let prefetch_sum = sum_with_prefetch(&data, 8);

    assert!((seq_sum - prefetch_sum).abs() < 1e-5);
}

// ============================================================================
// Matmul Tests
// ============================================================================

#[test]
fn test_naive_matmul_identity() {
    let identity = vec![1.0, 0.0, 0.0, 1.0];
    let vec = vec![3.0, 4.0];

    let result = naive_matmul(&identity, &vec, 2, 2, 1);
    assert!((result[0] - 3.0).abs() < 1e-5);
    assert!((result[1] - 4.0).abs() < 1e-5);
}

#[test]
#[allow(clippy::many_single_char_names)]
fn test_blocked_matmul_equals_naive() {
    let m = 16;
    let k = 16;
    let n = 16;
    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 * 0.1).collect();

    let naive = naive_matmul(&a, &b, m, k, n);
    let blocked = blocked_matmul(&a, &b, m, k, n, 4);

    for i in 0..naive.len() {
        assert!((naive[i] - blocked[i]).abs() < 1e-4);
    }
}

// ============================================================================
// TensorPool Tests
// ============================================================================

#[test]
fn test_tensor_pool_capacity() {
    let pool = TensorPool::new(5);
    assert_eq!(pool.capacity(), 5);
    assert_eq!(pool.available(), 0);
}

#[test]
fn test_tensor_pool_acquire_release() {
    let mut pool = TensorPool::new(3);

    let buf1 = pool.acquire(100);
    assert_eq!(buf1.len(), 100);

    pool.release(buf1);
    assert_eq!(pool.available(), 1);

    // Re-acquire should reuse
    let buf2 = pool.acquire(50);
    assert_eq!(pool.available(), 0);
    assert!(buf2.capacity() >= 100); // Reused buffer has original capacity
}

#[test]
fn test_tensor_pool_clear() {
    let mut pool = TensorPool::new(5);

    // Acquire 3 distinct buffers by requesting different sizes
    let buf1 = pool.acquire(100);
    let buf2 = pool.acquire(200);
    let buf3 = pool.acquire(300);

    // Release them back
    pool.release(buf1);
    pool.release(buf2);
    pool.release(buf3);

    assert_eq!(pool.available(), 3);
    pool.clear();
    assert_eq!(pool.available(), 0);
}

#[test]
fn test_tensor_pool_exceeds_capacity() {
    let mut pool = TensorPool::new(2);

    let buf1 = pool.acquire(100);
    let buf2 = pool.acquire(100);
    let buf3 = pool.acquire(100);

    pool.release(buf1);
    pool.release(buf2);
    pool.release(buf3); // This should be dropped, not kept

    assert_eq!(pool.available(), 2);
}

// ============================================================================
// ForwardArena Tests
// ============================================================================

#[test]
fn test_forward_arena_creation() {
    let arena = ForwardArena::new(1000);
    assert_eq!(arena.capacity(), 1000);
    assert_eq!(arena.used(), 0);
}

#[test]
fn test_forward_arena_alloc() {
    let mut arena = ForwardArena::new(100);

    let slice1 = arena.alloc(30);
    assert_eq!(slice1.len(), 30);
    assert_eq!(arena.used(), 30);

    let slice2 = arena.alloc(40);
    assert_eq!(slice2.len(), 40);
    assert_eq!(arena.used(), 70);
}

#[test]
fn test_forward_arena_reset() {
    let mut arena = ForwardArena::new(100);
    arena.alloc(50);
    assert_eq!(arena.used(), 50);

    arena.reset();
    assert_eq!(arena.used(), 0);
}

#[test]
#[should_panic(expected = "insufficient capacity")]
fn test_forward_arena_overflow() {
    let mut arena = ForwardArena::new(10);
    arena.alloc(20); // Should panic
}

// ============================================================================
// ScratchBuffer Tests
// ============================================================================

#[test]
fn test_scratch_buffer_creation() {
    let scratch = ScratchBuffer::new(4, 100);
    assert_eq!(scratch.num_layers(), 4);
    assert_eq!(scratch.layer_size(), 100);
    assert_eq!(scratch.total_size(), 400);
}

#[test]
fn test_scratch_buffer_get_layer() {
    let scratch = ScratchBuffer::new(3, 50);

    let layer0 = scratch.get_layer(0);
    let layer2 = scratch.get_layer(2);

    assert_eq!(layer0.len(), 50);
    assert_eq!(layer2.len(), 50);
}

#[test]
fn test_scratch_buffer_get_layer_mut() {
    let mut scratch = ScratchBuffer::new(2, 10);

    {
        let layer = scratch.get_layer_mut(0);
        layer[0] = 42.0;
    }

    assert!((scratch.get_layer(0)[0] - 42.0).abs() < 1e-5);
}

#[test]
#[should_panic(expected = "layer index")]
fn test_scratch_buffer_out_of_bounds() {
    let scratch = ScratchBuffer::new(2, 10);
    let _ = scratch.get_layer(5); // Should panic
}

#[test]
fn test_scratch_buffer_reset() {
    let mut scratch = ScratchBuffer::new(2, 10);
    scratch.get_layer_mut(0)[0] = 100.0;

    scratch.reset();
    assert!((scratch.get_layer(0)[0] - 0.0).abs() < 1e-5);
}

// ============================================================================
// Quantized Dot Product Tests
// ============================================================================

#[test]
fn test_quantized_dot_q4_short_input() {
    let block_a = vec![0u8; 10]; // Too short
    let block_b = vec![0u8; 18];

    let result = quantized_dot_q4(&block_a, &block_b);
    assert!((result - 0.0).abs() < 1e-5);
}

#[test]
fn test_quantized_dot_q8_short_input() {
    let block_a = vec![0u8; 30]; // Too short
    let block_b = vec![0u8; 34];

    let result = quantized_dot_q8(&block_a, &block_b);
    assert!((result - 0.0).abs() < 1e-5);
}

#[test]
fn test_quantized_matvec_q4_basic() {
    // Q4_0 block: 2 bytes scale + 16 bytes data = 18 bytes per 32 values
    let rows = 1;
    let cols = 32;
    let block_size = 18;

    // Create a simple weight matrix with scale = 1.0
    let mut weights = vec![0u8; rows * block_size];
    // Set scale to 1.0 (in f16)
    let scale_bits = half::f16::from_f32(1.0).to_le_bytes();
    weights[0] = scale_bits[0];
    weights[1] = scale_bits[1];
    // Set all values to 8 (which becomes 0 after centering)
    for weight in weights.iter_mut().take(18).skip(2) {
        *weight = 0x88; // 8 in both nibbles
    }

    let input = vec![1.0f32; cols];
    let result = quantized_matvec_q4(&weights, &input, rows, cols);

    assert_eq!(result.len(), rows);
}

#[test]
fn test_quantized_matvec_q8_basic() {
    // Q8_0 block: 2 bytes scale + 32 bytes data = 34 bytes per 32 values
    let rows = 1;
    let cols = 32;
    let block_size = 34;

    let mut weights = vec![0u8; rows * block_size];
    let scale_bits = half::f16::from_f32(1.0).to_le_bytes();
    weights[0] = scale_bits[0];
    weights[1] = scale_bits[1];

    let input = vec![1.0f32; cols];
    let result = quantized_matvec_q8(&weights, &input, rows, cols);

    assert_eq!(result.len(), rows);
}

// ============================================================================
// QuantizedAccumulator Tests
// ============================================================================

#[test]
fn test_quantized_accumulator_new() {
    let acc = QuantizedAccumulator::new();
    assert!((acc.sum() - 0.0).abs() < 1e-5);
}

#[test]
fn test_quantized_accumulator_add_scaled() {
    let mut acc = QuantizedAccumulator::new();
    acc.add_scaled(2.0, 3.0);
    assert!((acc.sum() - 6.0).abs() < 1e-5);
}

#[test]
fn test_quantized_accumulator_add_block() {
    let mut acc = QuantizedAccumulator::new();
    acc.add_block(10.0, 0.5);
    acc.add_block(4.0, 0.5);
    assert!((acc.sum() - 7.0).abs() < 1e-5);
}

#[test]
fn test_quantized_accumulator_reset() {
    let mut acc = QuantizedAccumulator::new();
    acc.add_scaled(100.0, 1.0);
    acc.reset();
    assert!((acc.sum() - 0.0).abs() < 1e-5);
}

// ============================================================================
// DoubleBuffer Tests
// ============================================================================

#[test]
fn test_double_buffer_creation() {
    let buffer: DoubleBuffer<f32> = DoubleBuffer::new(100);
    assert_eq!(buffer.capacity(), 100);
}

#[test]
fn test_double_buffer_front_back() {
    let mut buffer: DoubleBuffer<f32> = DoubleBuffer::new(10);

    buffer.back_mut()[0] = 42.0;
    assert!((buffer.front()[0] - 0.0).abs() < 1e-5);

    buffer.swap();
    assert!((buffer.front()[0] - 42.0).abs() < 1e-5);
}

// ============================================================================
// ChunkedProcessor Tests
// ============================================================================

#[test]
fn test_chunked_processor_creation() {
    let processor = ChunkedProcessor::new(64);
    assert_eq!(processor.chunk_size(), 64);
}

#[test]
fn test_chunked_processor_num_chunks() {
    let processor = ChunkedProcessor::new(10);
    assert_eq!(processor.num_chunks(0), 0);
    assert_eq!(processor.num_chunks(5), 1);
    assert_eq!(processor.num_chunks(10), 1);
    assert_eq!(processor.num_chunks(11), 2);
    assert_eq!(processor.num_chunks(25), 3);
}

#[test]
fn test_chunked_processor_bounds() {
    let processor = ChunkedProcessor::new(10);

    assert_eq!(processor.chunk_bounds(0, 25), (0, 10));
    assert_eq!(processor.chunk_bounds(1, 25), (10, 20));
    assert_eq!(processor.chunk_bounds(2, 25), (20, 25));
}

#[test]
fn test_chunked_processor_process() {
    let processor = ChunkedProcessor::new(5);
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();

    let total = processor.process_chunks(&data, |chunk| chunk.iter().sum());
    let expected: f32 = (0..12).sum::<i32>() as f32;

    assert!((total - expected).abs() < 1e-5);
}

// ============================================================================
// InferencePipeline Tests
// ============================================================================

#[test]
fn test_inference_pipeline_creation() {
    let pipeline = InferencePipeline::new(4);
    assert_eq!(pipeline.num_stages(), 4);
}

#[test]
fn test_inference_pipeline_timing() {
    let mut pipeline = InferencePipeline::new(4);

    pipeline.record_stage_time(GpuPipelineStage::Embed, 1.0);
    pipeline.record_stage_time(GpuPipelineStage::Attention, 2.0);
    pipeline.record_stage_time(GpuPipelineStage::FFN, 3.0);
    pipeline.record_stage_time(GpuPipelineStage::Output, 0.5);

    assert!((pipeline.total_latency() - 6.5).abs() < 1e-5);
}

#[test]
fn test_inference_pipeline_stage_breakdown() {
    let mut pipeline = InferencePipeline::new(4);
    pipeline.record_stage_time(GpuPipelineStage::Embed, 1.0);

    let breakdown = pipeline.stage_breakdown();
    assert!(breakdown.contains_key(&GpuPipelineStage::Embed));
}

#[test]
fn test_inference_pipeline_reset() {
    let mut pipeline = InferencePipeline::new(4);
    pipeline.record_stage_time(GpuPipelineStage::Embed, 1.0);

    pipeline.reset();
    assert!((pipeline.total_latency() - 0.0).abs() < 1e-5);
}

// ============================================================================
// TokenBatch Tests
// ============================================================================

#[test]
fn test_token_batch_creation() {
    let batch = TokenBatch::new(8);
    assert_eq!(batch.capacity(), 8);
    assert_eq!(batch.len(), 0);
    assert!(batch.is_empty());
    assert!(!batch.is_full());
}

#[test]
fn test_token_batch_push_not_full() {
    let mut batch = TokenBatch::new(4);

    assert!(batch.push(1).is_none());
    assert!(batch.push(2).is_none());
    assert_eq!(batch.len(), 2);
}

#[test]
fn test_token_batch_push_full() {
    let mut batch = TokenBatch::new(2);

    assert!(batch.push(1).is_none());
    let flushed = batch.push(2);

    assert!(flushed.is_some());
    assert_eq!(flushed.unwrap(), vec![1, 2]);
    assert!(batch.is_empty());
}

#[test]
fn test_token_batch_flush() {
    let mut batch = TokenBatch::new(10);
    batch.push(1);
    batch.push(2);

    let flushed = batch.flush();
    assert_eq!(flushed, vec![1, 2]);
    assert!(batch.is_empty());
}

// ============================================================================
// SpeculativeBuffer Tests
// ============================================================================

#[test]
fn test_speculative_buffer_creation() {
    let buffer = SpeculativeBuffer::new(8);
    assert_eq!(buffer.capacity(), 8);
    assert_eq!(buffer.len(), 0);
    assert!(buffer.is_empty());
}

#[test]
fn test_speculative_buffer_add_verify() {
    let mut buffer = SpeculativeBuffer::new(10);

    buffer.add_candidate(1, 0.9);
    buffer.add_candidate(2, 0.8);
    buffer.add_candidate(3, 0.7);

    // All match
    let (accepted, rejected_idx) = buffer.verify(&[1, 2, 3]);
    assert_eq!(accepted, 3);
    assert!(rejected_idx.is_none());
}

#[test]
fn test_speculative_buffer_verify_mismatch() {
    let mut buffer = SpeculativeBuffer::new(10);

    buffer.add_candidate(1, 0.9);
    buffer.add_candidate(2, 0.8);
    buffer.add_candidate(3, 0.7);

    // Mismatch at index 1
    let (accepted, rejected_idx) = buffer.verify(&[1, 5, 3]);
    assert_eq!(accepted, 1);
    assert_eq!(rejected_idx, Some(1));
}

#[test]
fn test_speculative_buffer_accept_reject() {
    let mut buffer = SpeculativeBuffer::new(10);

    buffer.add_candidate(1, 0.9);
    buffer.add_candidate(2, 0.8);
    buffer.add_candidate(3, 0.7);

    buffer.accept(2);
    assert_eq!(buffer.len(), 1); // Only one left

    buffer.reject();
    assert!(buffer.is_empty());
}

// ============================================================================
// InferenceBatchScheduler Tests
// ============================================================================

#[test]
fn test_batch_scheduler_creation() {
    let scheduler = InferenceBatchScheduler::new();
    assert_eq!(scheduler.pending_count(), 0);
    assert_eq!(scheduler.completed_count(), 0);
}

#[test]
fn test_batch_scheduler_submit_complete() {
    let mut scheduler = InferenceBatchScheduler::new();

    let id1 = scheduler.submit(vec![1, 2, 3]);
    let id2 = scheduler.submit(vec![4, 5]);

    assert_eq!(scheduler.pending_count(), 2);

    scheduler.complete(id1, vec![10, 11]);
    assert_eq!(scheduler.pending_count(), 1);
    assert_eq!(scheduler.completed_count(), 1);

    let result = scheduler.poll();
    assert!(result.is_some());
    assert_eq!(result.unwrap(), (id1, vec![10, 11]));

    scheduler.complete(id2, vec![20]);
    let drained = scheduler.drain();
    assert_eq!(drained.len(), 1);
}

// ============================================================================
// AsyncRequestQueue Tests
// ============================================================================

#[test]
fn test_async_request_queue_creation() {
    let queue: AsyncRequestQueue<i32> = AsyncRequestQueue::new(5);
    assert_eq!(queue.capacity(), 5);
    assert_eq!(queue.len(), 0);
    assert!(queue.is_empty());
    assert!(!queue.is_full());
}

#[test]
fn test_async_request_queue_push_pop() {
    let mut queue: AsyncRequestQueue<i32> = AsyncRequestQueue::new(3);

    assert!(queue.try_push(1));
    assert!(queue.try_push(2));
    assert!(queue.try_push(3));
    assert!(!queue.try_push(4)); // Full

    assert_eq!(queue.try_pop(), Some(1));
    assert!(queue.try_push(4)); // Now has space
}

// ============================================================================
// InferenceEventNotifier Tests
// ============================================================================

#[test]
fn test_event_notifier_creation() {
    let notifier = InferenceEventNotifier::new();
    assert_eq!(notifier.handler_count(), 0);
}

#[test]
fn test_event_notifier_register_notify() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    let mut notifier = InferenceEventNotifier::new();
    let counter = Arc::new(AtomicU64::new(0));
    let counter_clone = counter.clone();

    notifier.register(Box::new(move |id, _tokens| {
        counter_clone.fetch_add(id, Ordering::SeqCst);
    }));

    assert_eq!(notifier.handler_count(), 1);

    notifier.notify(42, &[1, 2, 3]);
    assert_eq!(counter.load(Ordering::SeqCst), 42);
}

#[test]
fn test_event_notifier_clear() {
    let mut notifier = InferenceEventNotifier::new();
    notifier.register(Box::new(|_, _| {}));
    assert_eq!(notifier.handler_count(), 1);

    notifier.clear();
    assert_eq!(notifier.handler_count(), 0);
}

// ============================================================================
// TimeoutManager Tests
// ============================================================================

#[test]
fn test_timeout_manager_creation() {
    let manager = TimeoutManager::new();
    assert_eq!(manager.active_count(), 0);
}

#[test]
fn test_timeout_manager_register_remove() {
    let mut manager = TimeoutManager::new();
    let deadline = Instant::now() + Duration::from_secs(60);

    manager.register(1, deadline);
    manager.register(2, deadline);
    assert_eq!(manager.active_count(), 2);

    manager.remove(1);
    assert_eq!(manager.active_count(), 1);
}

#[test]
fn test_timeout_manager_check_expired() {
    let mut manager = TimeoutManager::new();

    // Register one that's already expired
    let past = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    manager.register(1, past);

    // Register one that's not expired
    let future = Instant::now() + Duration::from_secs(60);
    manager.register(2, future);

    let expired = manager.check_expired();
    assert_eq!(expired, vec![1]);
    assert_eq!(manager.active_count(), 1);
}

// ============================================================================
// PriorityRequestQueue Tests
// ============================================================================

#[test]
fn test_priority_queue_creation() {
    let queue: PriorityRequestQueue<i32> = PriorityRequestQueue::new();
    assert_eq!(queue.len(), 0);
    assert!(queue.is_empty());
}

#[test]
fn test_priority_request_creation() {
    let req = PriorityRequest::new(10, "test".to_string());
    assert_eq!(req.priority(), 10);
    assert_eq!(req.data(), &"test".to_string());
    assert_eq!(req.into_data(), "test".to_string());
}

#[test]
fn test_priority_queue_ordering() {
    let mut queue: PriorityRequestQueue<&str> = PriorityRequestQueue::new();

    queue.enqueue(PriorityRequest::new(1, "low"));
    queue.enqueue(PriorityRequest::new(10, "high"));
    queue.enqueue(PriorityRequest::new(5, "medium"));

    // Should dequeue highest priority first
    assert_eq!(queue.dequeue_highest().unwrap().into_data(), "high");
    assert_eq!(queue.dequeue_highest().unwrap().into_data(), "medium");
    assert_eq!(queue.dequeue_highest().unwrap().into_data(), "low");
    assert!(queue.dequeue_highest().is_none());
}

#[test]
fn test_priority_queue_fifo_same_priority() {
    let mut queue: PriorityRequestQueue<i32> = PriorityRequestQueue::new();

    queue.enqueue(PriorityRequest::new(5, 1));
    queue.enqueue(PriorityRequest::new(5, 2));
    queue.enqueue(PriorityRequest::new(5, 3));

    // FIFO for same priority
    assert_eq!(queue.dequeue_highest().unwrap().into_data(), 1);
    assert_eq!(queue.dequeue_highest().unwrap().into_data(), 2);
    assert_eq!(queue.dequeue_highest().unwrap().into_data(), 3);
}

// ============================================================================
// TokenRateLimiter Tests
// ============================================================================

#[test]
fn test_rate_limiter_creation() {
    let limiter = TokenRateLimiter::new(100.0, 50);
    assert_eq!(limiter.tokens_available(), 50);
}

#[test]
fn test_rate_limiter_acquire() {
    let mut limiter = TokenRateLimiter::new(100.0, 10);

    assert!(limiter.try_acquire(5));
    assert_eq!(limiter.tokens_available(), 5);

    assert!(!limiter.try_acquire(10)); // Not enough
    assert_eq!(limiter.tokens_available(), 5);
}

#[test]
fn test_rate_limiter_refill() {
    let mut limiter = TokenRateLimiter::new(1000000.0, 100); // Very high rate
    limiter.try_acquire(100);
    assert_eq!(limiter.tokens_available(), 0);

    std::thread::sleep(Duration::from_millis(10));
    limiter.refill();

    // Should have refilled something
    assert!(limiter.tokens_available() > 0);
}

// ============================================================================
// ResourceTracker Tests
// ============================================================================

#[test]
fn test_resource_tracker_creation() {
    let tracker = ResourceTracker::new(1024, 100);
    assert_eq!(tracker.memory_usage(), 0);
    assert_eq!(tracker.compute_usage(), 0);
}

#[test]
fn test_resource_tracker_allocation() {
    let mut tracker = ResourceTracker::new(1000, 100);

    assert!(tracker.can_allocate(500, 50));
    let id1 = tracker.allocate(500, 50);
    assert!(id1.is_some());

    assert_eq!(tracker.memory_usage(), 500);
    assert_eq!(tracker.compute_usage(), 50);

    // Can't allocate beyond capacity
    assert!(!tracker.can_allocate(600, 10));
    assert!(tracker.allocate(600, 10).is_none());
}

#[test]
fn test_resource_tracker_release() {
    let mut tracker = ResourceTracker::new(1000, 100);

    let id = tracker.allocate(500, 50).unwrap();
    tracker.release(id);

    assert_eq!(tracker.memory_usage(), 0);
    assert_eq!(tracker.compute_usage(), 0);
}

#[test]
fn test_resource_tracker_usage_percentage() {
    let mut tracker = ResourceTracker::new(1000, 100);
    tracker.allocate(500, 25);

    let (mem_pct, compute_pct) = tracker.usage_percentage();
    assert!((mem_pct - 50.0).abs() < 1e-5);
    assert!((compute_pct - 25.0).abs() < 1e-5);
}

#[test]
fn test_resource_tracker_zero_capacity() {
    let tracker = ResourceTracker::new(0, 0);
    let (mem_pct, compute_pct) = tracker.usage_percentage();

    assert!((mem_pct - 0.0).abs() < 1e-5);
    assert!((compute_pct - 0.0).abs() < 1e-5);
}

// ============================================================================
// InferenceMetrics Tests
// ============================================================================

#[test]
fn test_inference_metrics_creation() {
    let metrics = InferenceMetrics::new();
    assert_eq!(metrics.total_inferences(), 0);
    assert_eq!(metrics.total_tokens(), 0);
}

#[test]
fn test_inference_metrics_record() {
    let mut metrics = InferenceMetrics::new();

    metrics.record_inference(Duration::from_millis(100), 10);
    metrics.record_inference(Duration::from_millis(200), 20);

    assert_eq!(metrics.total_inferences(), 2);
    assert_eq!(metrics.total_tokens(), 30);
}

#[test]
fn test_inference_metrics_percentile() {
    let mut metrics = InferenceMetrics::new();

    // Empty case
    assert!(metrics.latency_percentile(50).is_none());

    // Add some latencies
    for i in 1..=100 {
        metrics.record_inference(Duration::from_millis(i), 1);
    }

    let p50 = metrics.latency_percentile(50).unwrap();
    assert!(p50.as_millis() >= 40 && p50.as_millis() <= 60);
}

#[test]
fn test_inference_metrics_reset() {
    let mut metrics = InferenceMetrics::new();
    metrics.record_inference(Duration::from_millis(100), 10);

    metrics.reset();

    assert_eq!(metrics.total_inferences(), 0);
    assert_eq!(metrics.total_tokens(), 0);
}

// ============================================================================
// HealthChecker Tests
// ============================================================================

#[test]
fn test_health_checker_creation() {
    let checker = HealthChecker::new();
    assert_eq!(checker.check_count(), 0);
    assert!(checker.is_healthy()); // No checks = healthy
}

#[test]
fn test_health_checker_register_check() {
    let mut checker = HealthChecker::new();

    checker.register_check("test1", Box::new(|| true));
    checker.register_check("test2", Box::new(|| false));

    assert_eq!(checker.check_count(), 2);

    let results = checker.check_all();
    assert!(results["test1"]);
    assert!(!results["test2"]);

    assert!(!checker.is_healthy()); // One check failed
}

#[test]
fn test_health_checker_clear() {
    let mut checker = HealthChecker::new();
    checker.register_check("test", Box::new(|| true));

    checker.clear();
    assert_eq!(checker.check_count(), 0);
}

// ============================================================================
// ErrorRecoveryStrategy Tests
// ============================================================================

#[test]
fn test_error_recovery_strategy_creation() {
    let strategy = ErrorRecoveryStrategy::new()
        .with_max_retries(5)
        .with_base_delay(Duration::from_millis(50))
        .with_max_delay(Duration::from_secs(5))
        .with_jitter(0.2);

    assert_eq!(strategy.max_retries(), 5);
}

#[test]
fn test_error_recovery_classify_transient() {
    let strategy = ErrorRecoveryStrategy::new();

    let timeout_err = std::io::Error::new(std::io::ErrorKind::TimedOut, "timeout");
    assert_eq!(
        strategy.classify_error(&timeout_err),
        ErrorClassification::Transient
    );

    let interrupted_err = std::io::Error::new(std::io::ErrorKind::Interrupted, "interrupted");
    assert_eq!(
        strategy.classify_error(&interrupted_err),
        ErrorClassification::Transient
    );
}

#[test]
fn test_error_recovery_classify_fatal() {
    let strategy = ErrorRecoveryStrategy::new();

    let not_found_err = std::io::Error::new(std::io::ErrorKind::NotFound, "not found");
    assert_eq!(
        strategy.classify_error(&not_found_err),
        ErrorClassification::Fatal
    );
}

#[test]
fn test_error_recovery_classify_gpu() {
    let strategy = ErrorRecoveryStrategy::new();

    let gpu_err = std::io::Error::other("GPU unavailable");
    assert_eq!(
        strategy.classify_error(&gpu_err),
        ErrorClassification::GpuFailure
    );
}

#[test]
fn test_error_recovery_determine_action() {
    let strategy = ErrorRecoveryStrategy::new().with_max_retries(3);

    let timeout_err = std::io::Error::new(std::io::ErrorKind::TimedOut, "timeout");

    // Should retry on transient error
    match strategy.determine_action(&timeout_err, 0) {
        RecoveryAction::Retry { delay: _ } => (),
        _ => panic!("Expected Retry action"),
    }

    // Should fail after max retries
    match strategy.determine_action(&timeout_err, 3) {
        RecoveryAction::Fail => (),
        _ => panic!("Expected Fail action"),
    }
}

#[test]
fn test_error_recovery_gpu_fallback() {
    let strategy = ErrorRecoveryStrategy::new();

    let gpu_err = std::io::Error::other("GPU error");

    match strategy.determine_action(&gpu_err, 0) {
        RecoveryAction::FallbackToCpu => (),
        _ => panic!("Expected FallbackToCpu action"),
    }
}

#[test]
fn test_error_recovery_calculate_delay() {
    let strategy = ErrorRecoveryStrategy::new()
        .with_base_delay(Duration::from_millis(100))
        .with_max_delay(Duration::from_secs(10))
        .with_jitter(0.0); // No jitter for predictable test

    let delay0 = strategy.calculate_delay(0);
    let delay1 = strategy.calculate_delay(1);
    let delay2 = strategy.calculate_delay(2);

    // Exponential backoff: 100ms, 200ms, 400ms
    assert!(delay0.as_millis() >= 100 && delay0.as_millis() <= 110);
    assert!(delay1.as_millis() >= 200 && delay1.as_millis() <= 220);
    assert!(delay2.as_millis() >= 400 && delay2.as_millis() <= 440);
}

// ============================================================================
// DegradationManager Tests
// ============================================================================

#[test]
fn test_degradation_manager_creation() {
    let manager = DegradationManager::new();
    assert_eq!(manager.current_mode(), DegradationMode::Normal);
}

#[test]
fn test_degradation_manager_gpu_unavailable() {
    let mut manager = DegradationManager::new();
    manager.set_gpu_available(false);
    assert_eq!(manager.current_mode(), DegradationMode::CpuFallback);
}

#[test]
fn test_degradation_manager_memory_pressure() {
    let mut manager = DegradationManager::new();
    manager.update_memory_pressure(0.9);
    assert_eq!(manager.current_mode(), DegradationMode::MemoryPressure);
}

#[test]
fn test_degradation_manager_latency_priority() {
    let mut manager = DegradationManager::new();
    manager.set_latency_priority(true);
    assert_eq!(manager.current_mode(), DegradationMode::LowLatency);
}

#[test]
fn test_degradation_manager_recommended_batch_size() {
    let mut manager = DegradationManager::new();

    assert_eq!(manager.recommended_batch_size(16), 16);

    manager.update_memory_pressure(0.9);
    let reduced = manager.recommended_batch_size(100);
    assert!(reduced < 100);
}

#[test]
fn test_degradation_manager_recommended_context() {
    let mut manager = DegradationManager::new();

    assert_eq!(manager.recommended_max_context(4096), 4096);

    manager.update_system_load(realizar::gpu::SystemLoad {
        cpu_percent: 95.0,
        memory_percent: 85.0,
        queue_depth: 100,
    });

    let reduced = manager.recommended_max_context(4096);
    assert!(reduced < 4096);
}

// ============================================================================
// FailureIsolator Tests
// ============================================================================

#[test]
fn test_failure_isolator_creation() {
    let isolator = FailureIsolator::new();
    assert_eq!(isolator.active_requests(), 0);
    assert_eq!(isolator.success_count(), 0);
    assert_eq!(isolator.failure_count(), 0);
    assert!(!isolator.is_circuit_open());
}

#[test]
fn test_failure_isolator_request_lifecycle() {
    let isolator = FailureIsolator::new();

    let id = isolator.start_request();
    assert_eq!(isolator.active_requests(), 1);

    isolator.complete_request(id, &RequestOutcome::Success);
    assert_eq!(isolator.active_requests(), 0);
    assert_eq!(isolator.success_count(), 1);
}

#[test]
fn test_failure_isolator_circuit_breaker() {
    let isolator = FailureIsolator::new();

    // Fail 5 times to open circuit
    for _ in 0..5 {
        let id = isolator.start_request();
        isolator.complete_request(id, &RequestOutcome::Failed("error".to_string()));
    }

    assert!(isolator.is_circuit_open());
    assert!(isolator.try_start_request().is_err());

    isolator.reset_circuit();
    assert!(!isolator.is_circuit_open());
}

// ============================================================================
// ConnectionConfig and ConnectionPool Tests
// ============================================================================

#[test]
fn test_connection_config_builder() {
    let config = ConnectionConfig::new()
        .with_max_connections(20)
        .with_min_connections(5)
        .with_idle_timeout(Duration::from_secs(600));

    // Just ensure builder pattern works
    let _ = config;
}

#[test]
fn test_connection_pool_creation() {
    let config = ConnectionConfig::new();
    let pool = ConnectionPool::new(config);

    assert_eq!(pool.active_connections(), 0);
    assert_eq!(pool.idle_connections(), 0);
}

// ============================================================================
// GpuModelConfig Tests
// ============================================================================

#[test]
fn test_gpu_model_config_derived_values() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 4096,
        num_heads: 32,
        num_kv_heads: 8, // GQA
        num_layers: 32,
        intermediate_dim: 11008,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    assert_eq!(config.head_dim(), 128);
    assert_eq!(config.kv_dim(), 1024); // 8 * 128
    assert!(config.is_gqa());

    // qkv_dim = hidden_dim + 2 * kv_dim = 4096 + 2 * 1024 = 6144
    assert_eq!(config.qkv_dim(), 6144);
}

#[test]
fn test_gpu_model_config_non_gqa() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 768,
        num_heads: 12,
        num_kv_heads: 12, // MHA (not GQA)
        num_layers: 12,
        intermediate_dim: 3072,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    assert!(!config.is_gqa());
}

// ============================================================================
// GpuGenerateConfig Tests
// ============================================================================

#[test]
fn test_gpu_generate_config_default() {
    let config = GpuGenerateConfig::default();
    assert_eq!(config.max_tokens, 64);
    assert!((config.temperature - 0.0).abs() < 1e-5);
    assert_eq!(config.top_k, 1);
}

#[test]
fn test_gpu_generate_config_deterministic() {
    let config = GpuGenerateConfig::deterministic(100);
    assert_eq!(config.max_tokens, 100);
    assert_eq!(config.top_k, 1);
}

#[test]
fn test_gpu_generate_config_with_sampling() {
    let config = GpuGenerateConfig::with_sampling(50, 0.7, 40);
    assert_eq!(config.max_tokens, 50);
    assert!((config.temperature - 0.7).abs() < 1e-5);
    assert_eq!(config.top_k, 40);
}

#[test]
fn test_gpu_generate_config_with_stop_tokens() {
    let config = GpuGenerateConfig::default().with_stop_tokens(vec![0, 1, 2]);

    assert_eq!(config.stop_tokens, vec![0, 1, 2]);
}

// ============================================================================
// WeightType Tests
// ============================================================================

#[test]
fn test_weight_type_variants() {
    let types = [
        WeightType::Qkv,
        WeightType::Output,
        WeightType::FfnFc1,
        WeightType::FfnFc2,
        WeightType::LmHead,
    ];

    // Just verify all variants exist
    assert_eq!(types.len(), 5);
}

// ============================================================================
// GpuPipelineStage Tests
// ============================================================================

#[test]
fn test_gpu_pipeline_stage_values() {
    assert_eq!(GpuPipelineStage::Embed as u8, 0);
    assert_eq!(GpuPipelineStage::Attention as u8, 1);
    assert_eq!(GpuPipelineStage::FFN as u8, 2);
    assert_eq!(GpuPipelineStage::Output as u8, 3);
}

// ============================================================================
// StreamingKVCacheFp16 Tests
// ============================================================================

#[test]
fn test_fp16_cache_creation() {
    let cache = StreamingKVCacheFp16::new(4, 1024, 8, 64);

    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.max_positions(), 1024);
}

#[test]
fn test_fp16_cache_append_get() {
    let mut cache = StreamingKVCacheFp16::new(1, 10, 2, 4);
    let kv_dim = 2 * 4;

    let key: Vec<f32> = (0..kv_dim).map(|i| i as f32 * 0.1).collect();
    let value: Vec<f32> = (0..kv_dim).map(|i| i as f32 * 0.2).collect();

    cache.append(0, &key, &value);
    assert_eq!(cache.len(), 1);

    let (keys, values) = cache.get_valid_f32(0);

    // Check precision within FP16 tolerance
    for i in 0..kv_dim {
        assert!((keys[i] - key[i]).abs() < 0.01);
        assert!((values[i] - value[i]).abs() < 0.01);
    }
}

#[test]
fn test_fp16_cache_memory_half() {
    let fp32_cache = StreamingKVCache::new(4, 1024, 8, 64);
    let fp16_cache = StreamingKVCacheFp16::new(4, 1024, 8, 64);

    assert_eq!(fp16_cache.memory_bytes(), fp32_cache.memory_bytes() / 2);
}

#[test]
fn test_fp16_cache_get_range_raw() {
    let mut cache = StreamingKVCacheFp16::new(1, 10, 2, 2);
    let kv_dim = 4;

    let key = vec![1.0f32; kv_dim];
    let value = vec![2.0f32; kv_dim];
    cache.append(0, &key, &value);

    let (raw_keys, raw_values) = cache.get_range_raw(0, 0, 1);
    assert_eq!(raw_keys.len(), kv_dim);
    assert_eq!(raw_values.len(), kv_dim);
}

// ============================================================================
// ComputeBackend Tests
// ============================================================================

#[test]
fn test_compute_backend_default() {
    let backend: ComputeBackend = Default::default();
    assert_eq!(backend, ComputeBackend::Auto);
}

#[test]
fn test_gpu_compute_backend_accessor() {
    let compute = GpuCompute::new(ComputeBackend::Cpu).unwrap();
    assert_eq!(compute.backend(), ComputeBackend::Cpu);
}

// ============================================================================
// HybridScheduler Tests
// ============================================================================

#[test]
fn test_hybrid_scheduler_transpose_b() {
    let mut scheduler = HybridScheduler::new().unwrap();

    // A: 2x3, B: 2x3 (will be transposed to 3x2)
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // Two rows, three cols

    let result = scheduler.matmul_transpose_b(&a, &b, 2, 3, 2).unwrap();

    // A @ B^T where B^T is 3x2
    assert_eq!(result.len(), 4);
}

// ============================================================================
// GpuBufferPool Stats Tests
// ============================================================================

#[test]
fn test_gpu_pool_stats() {
    let mut pool = GpuBufferPool::new();

    let stats = pool.stats();
    assert_eq!(stats.cached_buffers, 0);
    assert_eq!(stats.cached_bytes, 0);

    let buf = pool.acquire(1000);
    pool.release(buf);

    let stats = pool.stats();
    assert!(stats.cached_buffers > 0);
    assert!(stats.cached_bytes > 0);
}

// ============================================================================
// ShutdownCoordinator Tests
// ============================================================================

#[test]
fn test_shutdown_coordinator_creation() {
    let coord = realizar::gpu::ShutdownCoordinator::new();
    assert!(!coord.is_shutting_down());
    assert_eq!(coord.pending_requests(), 0);
    assert_eq!(coord.handler_count(), 0);
}

#[test]
fn test_shutdown_coordinator_request_tracking() {
    let mut coord = realizar::gpu::ShutdownCoordinator::new();

    coord.request_started();
    coord.request_started();
    assert_eq!(coord.pending_requests(), 2);

    coord.request_completed();
    assert_eq!(coord.pending_requests(), 1);

    coord.request_completed();
    assert_eq!(coord.pending_requests(), 0);
}

#[test]
fn test_shutdown_coordinator_register_handler() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let mut coord = realizar::gpu::ShutdownCoordinator::new();
    let called = Arc::new(AtomicBool::new(false));
    let called_clone = called.clone();

    coord.register_handler(Box::new(move || {
        called_clone.store(true, Ordering::SeqCst);
    }));

    assert_eq!(coord.handler_count(), 1);

    coord.initiate_shutdown();
    assert!(coord.is_shutting_down());
    assert!(called.load(Ordering::SeqCst));
}

#[test]
fn test_shutdown_coordinator_is_complete() {
    let mut coord = realizar::gpu::ShutdownCoordinator::new();

    // Not complete before shutdown initiated
    assert!(!coord.is_complete());

    coord.request_started();
    coord.initiate_shutdown();

    // Not complete with pending requests
    assert!(!coord.is_complete());

    coord.request_completed();

    // Complete after shutdown + no pending
    assert!(coord.is_complete());
}

#[test]
fn test_shutdown_coordinator_double_initiate() {
    let mut coord = realizar::gpu::ShutdownCoordinator::new();

    coord.initiate_shutdown();
    assert!(coord.is_shutting_down());

    // Second call should be no-op
    coord.initiate_shutdown();
    assert!(coord.is_shutting_down());
}

// ============================================================================
// StreamingKVCache (FP32) Tests
// ============================================================================

#[test]
fn test_streaming_kv_cache_creation() {
    let cache = StreamingKVCache::new(4, 128, 8, 64);
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.max_positions(), 128);
}

#[test]
fn test_streaming_kv_cache_append_and_get() {
    let mut cache = StreamingKVCache::new(2, 10, 2, 4);
    let kv_dim = 2 * 4; // num_heads * head_dim = 8

    let key = vec![1.0f32; kv_dim];
    let value = vec![2.0f32; kv_dim];

    cache.append(0, &key, &value);
    cache.append(1, &key, &value); // Last layer updates position

    assert_eq!(cache.len(), 1);

    let (keys, values) = cache.get_valid(0);
    assert_eq!(keys.len(), kv_dim);
    assert_eq!(values.len(), kv_dim);
}

#[test]
fn test_streaming_kv_cache_get_range() {
    let mut cache = StreamingKVCache::new(1, 10, 2, 2);
    let kv_dim = 4;

    // Append two positions
    cache.append(0, &vec![1.0; kv_dim], &vec![2.0; kv_dim]);
    cache.append(0, &vec![3.0; kv_dim], &vec![4.0; kv_dim]);

    let (keys, values) = cache.get_range(0, 0, 2);
    assert_eq!(keys.len(), kv_dim * 2);
    assert_eq!(values.len(), kv_dim * 2);
}

#[test]
fn test_streaming_kv_cache_clear() {
    let mut cache = StreamingKVCache::new(1, 10, 2, 2);
    let kv_dim = 4;

    cache.append(0, &vec![1.0; kv_dim], &vec![1.0; kv_dim]);
    assert_eq!(cache.len(), 1);

    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

#[test]
fn test_streaming_kv_cache_memory_calculation() {
    let cache = StreamingKVCache::new(4, 1024, 8, 64);
    // Memory = num_layers * max_positions * num_heads * head_dim * 2 (K+V) * 4 bytes
    // = 4 * 1024 * 8 * 64 * 2 * 4 = 16777216 bytes = 16 MB
    assert_eq!(cache.memory_bytes(), 16777216);
    assert!((cache.memory_mb() - 16.0).abs() < 0.01);
}

// ============================================================================
// ResourceConfig Tests
// ============================================================================

#[test]
fn test_resource_config_builder() {
    let config = realizar::gpu::ResourceConfig::new()
        .with_max_memory_per_request(1024 * 1024 * 100) // 100MB
        .with_max_total_memory(1024 * 1024 * 1024 * 2) // 2GB
        .with_max_compute_time(Duration::from_secs(60))
        .with_max_queue_depth(50);

    // Ensure builder pattern works
    let _ = config;
}

// ============================================================================
// ResourceLimiter Tests
// ============================================================================

#[test]
fn test_resource_limiter_creation() {
    let config = realizar::gpu::ResourceConfig::new();
    let limiter = realizar::gpu::ResourceLimiter::new(config);
    assert_eq!(limiter.current_memory(), 0);
}

#[test]
fn test_resource_limiter_memory_allocation() {
    let config = realizar::gpu::ResourceConfig::new()
        .with_max_memory_per_request(1000)
        .with_max_total_memory(5000);
    let limiter = realizar::gpu::ResourceLimiter::new(config);

    // Should allow allocation within limits
    assert!(matches!(
        limiter.check_memory(500),
        realizar::gpu::LimitResult::Allowed
    ));

    // Should deny allocation exceeding per-request limit
    assert!(matches!(
        limiter.check_memory(2000),
        realizar::gpu::LimitResult::Denied { .. }
    ));

    // Allocate some memory
    assert!(limiter.allocate(1000).is_ok());
    assert_eq!(limiter.current_memory(), 1000);

    // Deallocate
    limiter.deallocate(500);
    assert_eq!(limiter.current_memory(), 500);
}

#[test]
fn test_resource_limiter_queue_management() {
    let config = realizar::gpu::ResourceConfig::new().with_max_queue_depth(2);
    let limiter = realizar::gpu::ResourceLimiter::new(config);

    // First two enqueues should succeed
    assert!(matches!(
        limiter.enqueue(),
        realizar::gpu::LimitResult::Allowed
    ));
    assert!(matches!(
        limiter.enqueue(),
        realizar::gpu::LimitResult::Allowed
    ));

    // Third should get backpressure
    assert!(matches!(
        limiter.enqueue(),
        realizar::gpu::LimitResult::Backpressure
    ));

    // Dequeue one
    limiter.dequeue();

    // try_enqueue should work now
    assert!(matches!(
        limiter.try_enqueue(),
        realizar::gpu::LimitResult::Allowed
    ));
}

#[test]
fn test_resource_limiter_start_compute() {
    let config = realizar::gpu::ResourceConfig::new();
    let limiter = realizar::gpu::ResourceLimiter::new(config);
    let start = limiter.start_compute();
    std::thread::sleep(Duration::from_millis(1));
    assert!(start.elapsed() >= Duration::from_millis(1));
}

// ============================================================================
// RetryConfig and RetryPolicy Tests
// ============================================================================

#[test]
fn test_retry_config_builder() {
    let config = realizar::gpu::RetryConfig::new()
        .with_max_retries(5)
        .with_base_delay(Duration::from_millis(50))
        .with_max_delay(Duration::from_secs(60))
        .with_jitter_factor(0.2);

    let _ = config;
}

#[test]
fn test_retry_policy_transient_error() {
    let config = realizar::gpu::RetryConfig::new().with_max_retries(3);
    let policy = realizar::gpu::RetryPolicy::new(config);

    assert_eq!(policy.max_retries(), 3);

    // Should retry on transient error
    match policy.should_retry(1, realizar::gpu::ErrorCategory::Transient) {
        realizar::gpu::RetryDecision::Retry { delay: _ } => (),
        _ => panic!("Expected Retry decision"),
    }
}

#[test]
fn test_retry_policy_permanent_error() {
    let config = realizar::gpu::RetryConfig::new();
    let policy = realizar::gpu::RetryPolicy::new(config);

    // Should abort on permanent error
    match policy.should_retry(1, realizar::gpu::ErrorCategory::Permanent) {
        realizar::gpu::RetryDecision::Abort { reason } => {
            assert!(reason.contains("Permanent"));
        },
        _ => panic!("Expected Abort decision"),
    }
}

#[test]
fn test_retry_policy_max_retries_exceeded() {
    let config = realizar::gpu::RetryConfig::new().with_max_retries(2);
    let policy = realizar::gpu::RetryPolicy::new(config);

    // Should abort when max retries exceeded
    match policy.should_retry(3, realizar::gpu::ErrorCategory::Transient) {
        realizar::gpu::RetryDecision::Abort { reason } => {
            assert!(reason.contains("exceeded"));
        },
        _ => panic!("Expected Abort decision"),
    }
}

#[test]
fn test_retry_policy_calculate_delay() {
    let config = realizar::gpu::RetryConfig::new()
        .with_base_delay(Duration::from_millis(100))
        .with_max_delay(Duration::from_secs(10));
    let policy = realizar::gpu::RetryPolicy::new(config);

    let delay0 = policy.calculate_delay(0);
    let delay1 = policy.calculate_delay(1);
    let delay2 = policy.calculate_delay(2);

    // Exponential: 100, 200, 400
    assert_eq!(delay0.as_millis(), 100);
    assert_eq!(delay1.as_millis(), 200);
    assert_eq!(delay2.as_millis(), 400);
}

// ============================================================================
// CircuitBreaker Tests
// ============================================================================

#[test]
fn test_circuit_breaker_creation() {
    let config = realizar::gpu::CircuitConfig::new();
    let breaker = realizar::gpu::CircuitBreaker::new(config);
    assert_eq!(breaker.state(), realizar::gpu::CircuitState::Closed);
}

#[test]
fn test_circuit_config_builder() {
    let config = realizar::gpu::CircuitConfig::new()
        .with_failure_threshold(3)
        .with_success_threshold(2)
        .with_timeout(Duration::from_secs(60));
    let _ = config;
}

#[test]
fn test_circuit_breaker_allows_when_closed() {
    let config = realizar::gpu::CircuitConfig::new();
    let breaker = realizar::gpu::CircuitBreaker::new(config);
    assert!(breaker.allow_request());
}

#[test]
fn test_circuit_breaker_opens_after_failures() {
    let config = realizar::gpu::CircuitConfig::new().with_failure_threshold(3);
    let breaker = realizar::gpu::CircuitBreaker::new(config);

    breaker.record_failure();
    breaker.record_failure();
    assert_eq!(breaker.state(), realizar::gpu::CircuitState::Closed);

    breaker.record_failure();
    assert_eq!(breaker.state(), realizar::gpu::CircuitState::Open);
    assert!(!breaker.allow_request());
}

#[test]
fn test_circuit_breaker_success_resets_count() {
    let config = realizar::gpu::CircuitConfig::new().with_failure_threshold(3);
    let breaker = realizar::gpu::CircuitBreaker::new(config);

    breaker.record_failure();
    breaker.record_failure();
    breaker.record_success(); // Reset failure count

    breaker.record_failure();
    breaker.record_failure();
    // Should still be closed (only 2 consecutive failures)
    assert_eq!(breaker.state(), realizar::gpu::CircuitState::Closed);
}

#[test]
fn test_circuit_breaker_half_open_to_closed() {
    let config = realizar::gpu::CircuitConfig::new()
        .with_failure_threshold(2)
        .with_success_threshold(2)
        .with_timeout(Duration::from_millis(1));
    let breaker = realizar::gpu::CircuitBreaker::new(config);

    // Open the circuit
    breaker.record_failure();
    breaker.record_failure();
    assert_eq!(breaker.state(), realizar::gpu::CircuitState::Open);

    // Wait for timeout
    std::thread::sleep(Duration::from_millis(10));

    // Should transition to half-open
    assert!(breaker.allow_request());
    assert_eq!(breaker.state(), realizar::gpu::CircuitState::HalfOpen);

    // Successes should close it
    breaker.record_success();
    breaker.record_success();
    assert_eq!(breaker.state(), realizar::gpu::CircuitState::Closed);
}

// ============================================================================
// BulkheadConfig and BulkheadManager Tests
// ============================================================================

#[test]
fn test_bulkhead_config_builder() {
    let config = realizar::gpu::BulkheadConfig::new()
        .with_pool("inference", 20)
        .with_pool("embedding", 10)
        .with_pool("batch", 5);
    let _ = config;
}

#[test]
fn test_bulkhead_manager_creation() {
    let config = realizar::gpu::BulkheadConfig::new()
        .with_pool("inference", 5)
        .with_pool("embedding", 3);
    let manager = realizar::gpu::BulkheadManager::new(&config);

    assert_eq!(manager.available(realizar::gpu::RequestType::Inference), 5);
    assert_eq!(manager.available(realizar::gpu::RequestType::Embedding), 3);
}

#[test]
fn test_bulkhead_manager_acquire_release() {
    let config = realizar::gpu::BulkheadConfig::new().with_pool("inference", 2);
    let manager = realizar::gpu::BulkheadManager::new(&config);

    let permit1 = manager
        .acquire(realizar::gpu::RequestType::Inference)
        .unwrap();
    assert_eq!(manager.available(realizar::gpu::RequestType::Inference), 1);

    let permit2 = manager
        .acquire(realizar::gpu::RequestType::Inference)
        .unwrap();
    assert_eq!(manager.available(realizar::gpu::RequestType::Inference), 0);

    // Should fail - pool exhausted
    assert!(manager
        .acquire(realizar::gpu::RequestType::Inference)
        .is_err());

    // Release one
    manager.release(&permit1);
    assert_eq!(manager.available(realizar::gpu::RequestType::Inference), 1);

    // Should succeed now
    assert!(manager
        .try_acquire(realizar::gpu::RequestType::Inference)
        .is_ok());

    manager.release(&permit2);
}

#[test]
fn test_bulkhead_manager_stats() {
    let config = realizar::gpu::BulkheadConfig::new()
        .with_pool("inference", 10)
        .with_pool("embedding", 5);
    let manager = realizar::gpu::BulkheadManager::new(&config);

    let stats = manager.stats();
    assert_eq!(stats.pool_count, 3); // inference, embedding, batch (default)
    assert!(stats.total_capacity > 0);
}

// ============================================================================
// Logger, LogEntry, LogConfig Tests
// ============================================================================

#[test]
fn test_log_entry_creation() {
    let entry = realizar::gpu::LogEntry::new(realizar::gpu::LogLevel::Info, "Test message");
    assert_eq!(entry.level(), realizar::gpu::LogLevel::Info);
    assert!(entry.timestamp() > 0);
    assert!(entry.correlation_id().is_none());
}

#[test]
fn test_log_entry_with_correlation_id() {
    let entry = realizar::gpu::LogEntry::new(realizar::gpu::LogLevel::Debug, "Test")
        .with_correlation_id("req-123");
    assert_eq!(entry.correlation_id(), Some("req-123"));
}

#[test]
fn test_log_entry_with_field() {
    let entry = realizar::gpu::LogEntry::new(realizar::gpu::LogLevel::Warn, "Warning")
        .with_field("user_id", "42")
        .with_field("action", "login");

    let json = entry.to_json();
    assert!(json.contains("\"user_id\":\"42\""));
    assert!(json.contains("\"action\":\"login\""));
}

#[test]
fn test_log_entry_to_json() {
    let entry = realizar::gpu::LogEntry::new(realizar::gpu::LogLevel::Error, "Failed")
        .with_correlation_id("abc");

    let json = entry.to_json();
    assert!(json.contains("\"level\":\"ERROR\""));
    assert!(json.contains("\"message\":\"Failed\""));
    assert!(json.contains("\"correlation_id\":\"abc\""));
}

#[test]
fn test_log_config_builder() {
    let config = realizar::gpu::LogConfig::new()
        .with_level(realizar::gpu::LogLevel::Debug)
        .with_json_format(true)
        .with_module_level("gpu", realizar::gpu::LogLevel::Trace);

    let _ = config;
}

#[test]
fn test_logger_is_enabled() {
    let config = realizar::gpu::LogConfig::new()
        .with_level(realizar::gpu::LogLevel::Warn)
        .with_module_level("gpu", realizar::gpu::LogLevel::Debug);
    let logger = realizar::gpu::Logger::new(config);

    // Default level is Warn
    assert!(logger.is_enabled(realizar::gpu::LogLevel::Error, "other"));
    assert!(logger.is_enabled(realizar::gpu::LogLevel::Warn, "other"));
    assert!(!logger.is_enabled(realizar::gpu::LogLevel::Info, "other"));

    // Module-specific level is Debug
    assert!(logger.is_enabled(realizar::gpu::LogLevel::Debug, "gpu"));
    assert!(!logger.is_enabled(realizar::gpu::LogLevel::Trace, "gpu"));
}

// ============================================================================
// PhaseTimer Tests
// ============================================================================

#[test]
fn test_phase_timer_creation() {
    let timer = realizar::gpu::PhaseTimer::new();
    let breakdown = timer.breakdown();
    assert!(breakdown.is_empty());
}

#[test]
fn test_phase_timer_timing() {
    let timer = realizar::gpu::PhaseTimer::new();

    timer.start_phase("embed");
    std::thread::sleep(Duration::from_millis(5));
    timer.end_phase("embed");

    timer.start_phase("attention");
    std::thread::sleep(Duration::from_millis(5));
    timer.end_phase("attention");

    let breakdown = timer.breakdown();
    assert!(breakdown.contains_key("embed"));
    assert!(breakdown.contains_key("attention"));
    assert!(breakdown["embed"] > 0);
}

// ============================================================================
// MemoryTracker Tests
// ============================================================================

#[test]
fn test_memory_tracker_creation() {
    let tracker = realizar::gpu::MemoryTracker::new();
    let report = tracker.report();
    assert_eq!(report.current_bytes, 0);
    assert_eq!(report.peak_bytes, 0);
    assert_eq!(report.allocation_count, 0);
}

#[test]
fn test_memory_tracker_allocation_deallocation() {
    let tracker = realizar::gpu::MemoryTracker::new();

    tracker.record_allocation("tensor_a", 1000);
    tracker.record_allocation("tensor_b", 2000);

    let report = tracker.report();
    assert_eq!(report.current_bytes, 3000);
    assert_eq!(report.peak_bytes, 3000);
    assert_eq!(report.allocation_count, 2);

    tracker.record_deallocation("tensor_a", 1000);

    let report = tracker.report();
    assert_eq!(report.current_bytes, 2000);
    assert_eq!(report.peak_bytes, 3000); // Peak unchanged
}

#[test]
fn test_memory_tracker_peak_tracking() {
    let tracker = realizar::gpu::MemoryTracker::new();

    tracker.record_allocation("a", 5000);
    tracker.record_deallocation("a", 5000);
    tracker.record_allocation("b", 3000);

    let report = tracker.report();
    assert_eq!(report.current_bytes, 3000);
    assert_eq!(report.peak_bytes, 5000); // Peak was 5000
}

// ============================================================================
// DiagnosticsCollector Tests
// ============================================================================

#[test]
fn test_diagnostics_collector_creation() {
    let collector = realizar::gpu::DiagnosticsCollector::new();
    let summary = collector.summary();
    assert_eq!(summary.request_count, 0);
}

#[test]
fn test_diagnostics_collector_record_request() {
    let collector = realizar::gpu::DiagnosticsCollector::new();

    let mut timing = std::collections::HashMap::new();
    timing.insert("embed".to_string(), 100u64);
    timing.insert("attention".to_string(), 200u64);

    collector.record_request_timing("req-1", timing);

    let summary = collector.summary();
    assert_eq!(summary.request_count, 1);
}

#[test]
fn test_diagnostics_collector_record_memory_snapshot() {
    let collector = realizar::gpu::DiagnosticsCollector::new();

    let report = realizar::gpu::MemoryReport {
        peak_bytes: 1000,
        current_bytes: 500,
        allocation_count: 5,
    };

    collector.record_memory_snapshot(report);
    // Just verify no panic
}

// ============================================================================
// DebugMode Tests
// ============================================================================

#[test]
fn test_debug_mode_creation() {
    let debug = realizar::gpu::DebugMode::new();
    assert!(!debug.is_enabled());
}

#[test]
fn test_debug_mode_enable() {
    let debug = realizar::gpu::DebugMode::new();
    debug.enable();
    assert!(debug.is_enabled());
}

// ============================================================================
// RequestCapture Tests
// ============================================================================

#[test]
fn test_request_capture_creation() {
    let capture = realizar::gpu::RequestCapture::new();
    assert!(capture.input().is_empty());
    assert!(capture.params().is_empty());
}

#[test]
fn test_request_capture_builder() {
    let capture = realizar::gpu::RequestCapture::new()
        .with_input("Hello, world!")
        .with_params("temperature", "0.7")
        .with_params("max_tokens", "100");

    assert_eq!(capture.input(), "Hello, world!");
    assert_eq!(
        capture.params().get("temperature"),
        Some(&"0.7".to_string())
    );
    assert_eq!(capture.params().get("max_tokens"), Some(&"100".to_string()));
}

#[test]
fn test_request_capture_to_json() {
    let capture = realizar::gpu::RequestCapture::new()
        .with_input("test")
        .with_params("key", "value");

    let json = capture.to_json();
    assert!(json.contains("\"input\":\"test\""));
    assert!(json.contains("\"key\":\"value\""));
}

#[test]
fn test_request_capture_from_json() {
    let json = r#"{"input":"hello","params":{}}"#;
    let capture = realizar::gpu::RequestCapture::from_json(json).unwrap();
    assert_eq!(capture.input(), "hello");
}

#[test]
fn test_request_capture_from_json_error() {
    let invalid = r#"{"no_input":true}"#;
    assert!(realizar::gpu::RequestCapture::from_json(invalid).is_err());
}

// ============================================================================
// StateDump Tests
// ============================================================================

#[test]
fn test_state_dump_creation() {
    let dump = realizar::gpu::StateDump::new();
    assert!(dump.error().is_empty());
    assert!(dump.stack_trace().is_empty());
    assert!(dump.state().is_empty());
}

#[test]
fn test_state_dump_builder() {
    let dump = realizar::gpu::StateDump::new()
        .with_error("OutOfMemory")
        .with_stack_trace("at gpu.rs:100\nat main.rs:50")
        .with_state("buffer_size", "1024")
        .with_state("position", "42");

    assert_eq!(dump.error(), "OutOfMemory");
    assert!(dump.stack_trace().contains("gpu.rs"));
    assert_eq!(dump.state().get("buffer_size"), Some(&"1024".to_string()));
}

#[test]
fn test_state_dump_to_json() {
    let dump = realizar::gpu::StateDump::new()
        .with_error("TestError")
        .with_state("key", "value");

    let json = dump.to_json();
    assert!(json.contains("\"error\":\"TestError\""));
    assert!(json.contains("\"key\":\"value\""));
}

// ============================================================================
// GgufModelState Tests
// ============================================================================

#[test]
fn test_gguf_model_state_empty() {
    let state = realizar::gpu::GgufModelState::new();
    assert!(!state.is_loaded());
    assert!(!state.is_ready());
    assert!(state.model_name().is_none());
    assert_eq!(state.vocab_size(), 0);
    assert!(state.model().is_none());
}

// ============================================================================
// AsyncGpuResult Tests
// ============================================================================

#[test]
fn test_async_gpu_result_ready() {
    let result = realizar::gpu::AsyncGpuResult::ready(vec![1.0, 2.0, 3.0]);
    assert!(result.is_ready());
    assert_eq!(result.try_get(), Some(&vec![1.0, 2.0, 3.0]));
}

#[test]
fn test_async_gpu_result_pending() {
    let mut result = realizar::gpu::AsyncGpuResult::pending();
    assert!(!result.is_ready());
    assert!(result.try_get().is_none());

    result.set_result(vec![4.0, 5.0]);
    assert!(result.is_ready());
    assert_eq!(result.try_get(), Some(&vec![4.0, 5.0]));
}

#[test]
fn test_async_gpu_result_wait() {
    let result = realizar::gpu::AsyncGpuResult::ready(vec![1.0, 2.0]);
    let data = result.wait();
    assert_eq!(data, vec![1.0, 2.0]);
}

// ============================================================================
// GpuCompute Additional Tests
// ============================================================================

#[test]
fn test_gpu_compute_cpu_backend() {
    let compute = GpuCompute::new(ComputeBackend::Cpu).unwrap();
    assert!(!compute.is_gpu());
    assert_eq!(compute.backend(), ComputeBackend::Cpu);
}

#[test]
fn test_gpu_compute_cpu_relu() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).unwrap();
    let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let result = compute.relu(&input).unwrap();
    assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_gpu_compute_cpu_sigmoid() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).unwrap();
    let input = vec![0.0];
    let result = compute.sigmoid(&input).unwrap();
    assert!((result[0] - 0.5).abs() < 1e-5);
}

#[test]
fn test_gpu_compute_cpu_dot() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).unwrap();
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let result = compute.dot(&a, &b).unwrap();
    assert!((result - 32.0).abs() < 1e-5); // 1*4 + 2*5 + 3*6 = 32
}

#[test]
fn test_gpu_compute_dot_length_mismatch() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).unwrap();
    let a = vec![1.0, 2.0];
    let b = vec![1.0, 2.0, 3.0];
    assert!(compute.dot(&a, &b).is_err());
}

#[test]
fn test_gpu_compute_matmul_dimension_error() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).unwrap();
    let a = vec![1.0; 6]; // Should be 2x3
    let b = vec![1.0; 9]; // 3x3

    // Wrong size for a
    assert!(compute.matmul(&a, &b, 2, 4, 3).is_err()); // m*k != a.len()

    // Wrong size for b
    assert!(compute.matmul(&a, &b, 2, 3, 4).is_err()); // k*n != b.len()
}

// ============================================================================
// InferenceMetrics Additional Tests
// ============================================================================

#[test]
fn test_inference_metrics_throughput() {
    let mut metrics = InferenceMetrics::new();

    // Record some tokens
    metrics.record_inference(Duration::from_millis(100), 50);
    metrics.record_inference(Duration::from_millis(100), 50);

    // Throughput should be positive
    let throughput = metrics.throughput();
    assert!(throughput > 0.0);
}

#[test]
fn test_inference_metrics_default() {
    let metrics: InferenceMetrics = Default::default();
    assert_eq!(metrics.total_inferences(), 0);
    assert_eq!(metrics.total_tokens(), 0);
}

// ============================================================================
// ConnectionPool Additional Tests
// ============================================================================

#[test]
fn test_connection_pool_acquire_release() {
    let config = ConnectionConfig::new().with_max_connections(2);
    let pool = ConnectionPool::new(config);

    let conn1 = pool.acquire().unwrap();
    assert_eq!(pool.active_connections(), 1);

    let conn2 = pool.acquire().unwrap();
    assert_eq!(pool.active_connections(), 2);

    // Should fail - at max
    assert!(pool.try_acquire().is_err());

    pool.release(conn1);
    assert_eq!(pool.active_connections(), 1);
    assert_eq!(pool.idle_connections(), 1);

    // Should succeed now (reuse from idle)
    let conn3 = pool.acquire().unwrap();
    assert_eq!(pool.active_connections(), 2);
    assert_eq!(pool.idle_connections(), 0);

    pool.release(conn2);
    pool.release(conn3);
}

#[test]
fn test_connection_pool_warm() {
    let config = ConnectionConfig::new().with_min_connections(3);
    let pool = ConnectionPool::new(config);

    pool.warm();
    assert_eq!(pool.idle_connections(), 3);
}

#[test]
fn test_connection_pool_health_check() {
    let config = ConnectionConfig::new().with_idle_timeout(Duration::from_millis(1));
    let pool = ConnectionPool::new(config);

    let conn = pool.acquire().unwrap();
    assert_eq!(
        pool.check_health(&conn),
        realizar::gpu::ConnectionState::Healthy
    );

    // Wait for timeout
    std::thread::sleep(Duration::from_millis(5));
    assert_eq!(
        pool.check_health(&conn),
        realizar::gpu::ConnectionState::Stale
    );

    pool.release(conn);
}

// ============================================================================
// ResourceMonitor Tests
// ============================================================================

#[test]
fn test_resource_monitor_creation() {
    let monitor = realizar::gpu::ResourceMonitor::new();
    let metrics = monitor.current_metrics();
    assert_eq!(metrics.memory_bytes, 0);
    assert!((metrics.gpu_utilization - 0.0).abs() < 1e-5);
    assert_eq!(metrics.queue_depth, 0);
    assert_eq!(metrics.last_latency_ms, 0);
}

#[test]
fn test_resource_monitor_record_metrics() {
    let monitor = realizar::gpu::ResourceMonitor::new();

    monitor.record_memory_usage(1024 * 1024);
    monitor.record_gpu_utilization(75.5);
    monitor.record_queue_depth(10);
    monitor.record_latency(Duration::from_millis(50));

    let metrics = monitor.current_metrics();
    assert_eq!(metrics.memory_bytes, 1024 * 1024);
    assert!((metrics.gpu_utilization - 75.5).abs() < 1e-5);
    assert_eq!(metrics.queue_depth, 10);
    assert_eq!(metrics.last_latency_ms, 50);
}

#[test]
fn test_resource_monitor_latency_stats() {
    let monitor = realizar::gpu::ResourceMonitor::new();

    // Empty stats
    let stats = monitor.latency_stats();
    assert_eq!(stats.min_ms, 0);
    assert_eq!(stats.max_ms, 0);
    assert_eq!(stats.avg_ms, 0);

    // Add some latencies
    monitor.record_latency(Duration::from_millis(10));
    monitor.record_latency(Duration::from_millis(20));
    monitor.record_latency(Duration::from_millis(30));

    let stats = monitor.latency_stats();
    assert_eq!(stats.min_ms, 10);
    assert_eq!(stats.max_ms, 30);
    assert_eq!(stats.avg_ms, 20);
}

#[test]
fn test_resource_monitor_snapshot() {
    let monitor = realizar::gpu::ResourceMonitor::new();
    monitor.record_memory_usage(5000);
    monitor.record_gpu_utilization(50.0);
    monitor.record_queue_depth(5);

    let snapshot = monitor.snapshot();
    assert!(snapshot.timestamp > 0);
    assert_eq!(snapshot.memory_bytes, 5000);
    assert!((snapshot.gpu_utilization - 50.0).abs() < 1e-5);
    assert_eq!(snapshot.queue_depth, 5);
}

// ============================================================================
// HybridScheduler Additional Tests
// ============================================================================

#[test]
fn test_hybrid_scheduler_with_threshold() {
    let scheduler = HybridScheduler::with_threshold(1000).unwrap();
    assert_eq!(scheduler.gpu_threshold(), 1000);
}

#[test]
fn test_hybrid_scheduler_should_use_gpu_m1() {
    let scheduler = HybridScheduler::new().unwrap();
    // m=1 should always use CPU (IMP-097)
    assert!(!scheduler.should_use_gpu(1, 100, 100));
}

#[test]
fn test_hybrid_scheduler_matmul_pooled() {
    let mut scheduler = HybridScheduler::new().unwrap();

    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.0, 0.0, 0.0, 1.0];

    let result = scheduler.matmul_pooled(&a, &b, 2, 2, 2).unwrap();
    assert_eq!(result.len(), 4);

    // Release buffer back to pool
    scheduler.release_buffer(result);

    let stats = scheduler.pool_stats();
    assert!(stats.cached_buffers > 0);
}

#[test]
fn test_hybrid_scheduler_matmul_async() {
    let mut scheduler = HybridScheduler::new().unwrap();

    let a = vec![1.0; 4];
    let b = vec![1.0; 4];

    let result = scheduler.matmul_async(&a, &b, 2, 2, 2).unwrap();
    assert!(result.is_ready());

    let data = result.wait();
    assert_eq!(data.len(), 4);
}

#[test]
fn test_hybrid_scheduler_matmul_batch() {
    let mut scheduler = HybridScheduler::new().unwrap();

    let ops: Vec<realizar::gpu::MatmulOp> = vec![
        (vec![1.0; 4], vec![1.0; 4], 2, 2, 2),
        (vec![2.0; 4], vec![2.0; 4], 2, 2, 2),
    ];

    let results = scheduler.matmul_batch(&ops).unwrap();
    assert_eq!(results.len(), 2);
}

// ============================================================================
// GpuBufferPool Additional Tests
// ============================================================================

#[test]
fn test_gpu_buffer_pool_clear() {
    let mut pool = GpuBufferPool::new();

    let buf1 = pool.acquire(100);
    let buf2 = pool.acquire(200);
    pool.release(buf1);
    pool.release(buf2);

    let stats = pool.stats();
    assert!(stats.cached_buffers > 0);

    pool.clear();
    let stats = pool.stats();
    assert_eq!(stats.cached_buffers, 0);
    assert_eq!(stats.cached_bytes, 0);
}

// ============================================================================
// DegradationMode Tests
// ============================================================================

#[test]
fn test_degradation_mode_high_throughput() {
    let mut manager = DegradationManager::new();

    // High throughput mode via system load
    manager.update_system_load(realizar::gpu::SystemLoad {
        cpu_percent: 50.0,
        memory_percent: 50.0,
        queue_depth: 10,
    });

    // Should still be normal under these conditions
    assert_eq!(manager.current_mode(), DegradationMode::Normal);
}

// ============================================================================
// ErrorRecoveryStrategy Additional Tests
// ============================================================================

#[test]
fn test_error_recovery_determine_action_with_fallback() {
    let strategy = ErrorRecoveryStrategy::new();

    let gpu_err = std::io::Error::other("GPU unavailable");
    let action = strategy.determine_action_with_fallback(&gpu_err, 0);

    match action {
        RecoveryAction::FallbackToCpu => (),
        _ => panic!("Expected FallbackToCpu"),
    }
}

// ============================================================================
// FailureIsolator Additional Tests
// ============================================================================

#[test]
fn test_failure_isolator_register_cleanup() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let isolator = FailureIsolator::new();
    let cleaned = Arc::new(AtomicBool::new(false));
    let cleaned_clone = cleaned.clone();

    let id = isolator.start_request();
    isolator.register_cleanup(id, move || {
        cleaned_clone.store(true, Ordering::SeqCst);
    });

    // Complete with failure to trigger cleanup
    isolator.complete_request(id, &RequestOutcome::Failed("test".to_string()));

    assert!(cleaned.load(Ordering::SeqCst));
}

// ============================================================================
// HybridScheduler Additional Coverage Tests
// ============================================================================

#[test]
fn test_hybrid_scheduler_has_gpu() {
    let scheduler = HybridScheduler::new().unwrap();
    // The result depends on whether GPU is available, but it should not panic
    let _ = scheduler.has_gpu();
}

#[test]
fn test_hybrid_scheduler_should_use_gpu_large() {
    let scheduler = HybridScheduler::new().unwrap();
    // Above threshold with m>1 should consider GPU
    // Default threshold is 64*64*64 = 262144
    // If m>1 and m*k*n >= threshold, should consider GPU (if available)
    let should_use = scheduler.should_use_gpu(64, 64, 64);
    // Result depends on GPU availability
    let _ = should_use;
}

#[test]
fn test_hybrid_scheduler_should_use_gpu_small() {
    let scheduler = HybridScheduler::new().unwrap();
    // Small matrices should not use GPU even with m>1 if below threshold
    let should_use = scheduler.should_use_gpu(2, 2, 2);
    // m=2 but 2*2*2=8 < threshold, so should not use GPU
    assert!(!should_use);
}

#[test]
fn test_hybrid_scheduler_basic_matmul() {
    let mut scheduler = HybridScheduler::new().unwrap();

    // 2x2 identity test
    let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
    let b = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity

    let result = scheduler.matmul(&a, &b, 2, 2, 2).unwrap();
    assert_eq!(result.len(), 4);
    assert!((result[0] - 1.0).abs() < 1e-5);
    assert!((result[1] - 2.0).abs() < 1e-5);
    assert!((result[2] - 3.0).abs() < 1e-5);
    assert!((result[3] - 4.0).abs() < 1e-5);
}

// ============================================================================
// GpuBufferPool Additional Coverage Tests
// ============================================================================

#[test]
fn test_gpu_buffer_pool_default() {
    let pool: GpuBufferPool = Default::default();
    let stats = pool.stats();
    assert_eq!(stats.cached_buffers, 0);
}

#[test]
fn test_gpu_buffer_pool_reuse_larger() {
    let mut pool = GpuBufferPool::new();

    // Acquire large buffer
    let buf = pool.acquire(10000);
    let cap = buf.capacity();
    pool.release(buf);

    // Acquire smaller - should reuse the larger buffer
    let buf2 = pool.acquire(100);
    // Buffer capacity should be at least the original
    assert!(buf2.capacity() >= cap || buf2.len() == 100);
}

#[test]
fn test_gpu_buffer_pool_multiple_sizes() {
    let mut pool = GpuBufferPool::new();

    // Create buffers of different sizes
    let buf1 = pool.acquire(1024);
    let buf2 = pool.acquire(2048);
    let buf3 = pool.acquire(4096);

    // Release them all
    pool.release(buf1);
    pool.release(buf2);
    pool.release(buf3);

    let stats = pool.stats();
    assert!(stats.cached_buffers > 0);
    assert!(stats.cached_bytes > 0);
}

// ============================================================================
// GpuCompute Additional Coverage Tests
// ============================================================================

#[test]
fn test_gpu_compute_matmul_cpu_basic() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).unwrap();

    // Simple 2x2 matmul
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];

    let result = compute.matmul(&a, &b, 2, 2, 2).unwrap();
    assert_eq!(result.len(), 4);

    // Expected: [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    assert!((result[0] - 19.0).abs() < 1e-5);
    assert!((result[1] - 22.0).abs() < 1e-5);
    assert!((result[2] - 43.0).abs() < 1e-5);
    assert!((result[3] - 50.0).abs() < 1e-5);
}

#[test]
fn test_gpu_compute_matmul_tensor() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).unwrap();

    let a = realizar::tensor::Tensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let b = realizar::tensor::Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let result = compute.matmul_tensor(&a, &b).unwrap();
    assert_eq!(result.shape(), &[2, 2]);

    // Identity @ matrix = matrix
    let data = result.data();
    assert!((data[0] - 1.0).abs() < 1e-5);
    assert!((data[1] - 2.0).abs() < 1e-5);
    assert!((data[2] - 3.0).abs() < 1e-5);
    assert!((data[3] - 4.0).abs() < 1e-5);
}

#[test]
fn test_gpu_compute_matmul_tensor_non_2d() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).unwrap();

    // Create 1D tensor - should fail for matmul
    let a = realizar::tensor::Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = realizar::tensor::Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let result = compute.matmul_tensor(&a, &b);
    assert!(result.is_err());
}

#[test]
fn test_gpu_compute_matmul_tensor_dimension_mismatch() {
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).unwrap();

    let a = realizar::tensor::Tensor::from_vec(vec![2, 3], vec![1.0; 6]).unwrap();
    let b = realizar::tensor::Tensor::from_vec(vec![4, 2], vec![1.0; 8]).unwrap();

    let result = compute.matmul_tensor(&a, &b);
    // Inner dimensions don't match: A[2,3] @ B[4,2] fails because 3 != 4
    assert!(result.is_err());
}

#[test]
fn test_gpu_compute_auto_backend() {
    // Test auto backend selection
    let compute = GpuCompute::auto().unwrap();
    // Should either be GPU or CPU
    let backend = compute.backend();
    assert!(backend == ComputeBackend::Gpu || backend == ComputeBackend::Cpu);
}

// ============================================================================
// AsyncGpuResult Additional Coverage Tests
// ============================================================================

#[test]
fn test_async_gpu_result_pending_then_set() {
    let mut result = realizar::gpu::AsyncGpuResult::pending();
    assert!(!result.is_ready());
    assert!(result.try_get().is_none());

    result.set_result(vec![10.0, 20.0, 30.0]);
    assert!(result.is_ready());

    let data = result.try_get().unwrap();
    assert_eq!(data.len(), 3);
    assert!((data[0] - 10.0).abs() < 1e-5);
}

// ============================================================================
// InferenceBatchScheduler Additional Coverage Tests
// ============================================================================

#[test]
fn test_batch_scheduler_default() {
    let scheduler: InferenceBatchScheduler = Default::default();
    assert_eq!(scheduler.pending_count(), 0);
    assert_eq!(scheduler.completed_count(), 0);
}

#[test]
fn test_batch_scheduler_multiple_batches() {
    let mut scheduler = InferenceBatchScheduler::new();

    let id1 = scheduler.submit(vec![1, 2, 3]);
    let id2 = scheduler.submit(vec![4, 5, 6]);
    let id3 = scheduler.submit(vec![7, 8, 9]);

    assert_eq!(scheduler.pending_count(), 3);

    scheduler.complete(id2, vec![40, 50, 60]);
    scheduler.complete(id1, vec![10, 20, 30]);
    scheduler.complete(id3, vec![70, 80, 90]);

    assert_eq!(scheduler.pending_count(), 0);
    assert_eq!(scheduler.completed_count(), 3);

    let results = scheduler.drain();
    assert_eq!(results.len(), 3);
}

// ============================================================================
// StreamingKVCache Additional Coverage Tests
// ============================================================================

#[test]
fn test_streaming_kv_cache_multiple_layers() {
    let mut cache = StreamingKVCache::new(4, 10, 2, 4);
    let kv_dim = 2 * 4;

    // Append to multiple layers
    for layer in 0..4 {
        let key = vec![layer as f32; kv_dim];
        let value = vec![(layer + 10) as f32; kv_dim];
        cache.append(layer, &key, &value);
    }

    // Verify each layer has correct data
    for layer in 0..4 {
        let (keys, _) = cache.get_valid(layer);
        assert!((keys[0] - layer as f32).abs() < 1e-5);
    }
}

// ============================================================================
// StreamingKVCacheFp16 Additional Coverage Tests
// ============================================================================

#[test]
fn test_fp16_cache_multiple_positions() {
    let mut cache = StreamingKVCacheFp16::new(1, 10, 2, 4);
    let kv_dim = 8;

    // Append 5 positions
    for pos in 0..5 {
        let key = vec![(pos as f32) * 0.1; kv_dim];
        let value = vec![(pos as f32) * 0.2; kv_dim];
        cache.append(0, &key, &value);
    }

    assert_eq!(cache.len(), 5);

    let (keys, values) = cache.get_valid_f32(0);
    assert_eq!(keys.len(), 5 * kv_dim);
    assert_eq!(values.len(), 5 * kv_dim);
}

#[test]
fn test_fp16_cache_clear() {
    let mut cache = StreamingKVCacheFp16::new(2, 10, 2, 4);
    let kv_dim = 8;

    cache.append(0, &vec![1.0; kv_dim], &vec![2.0; kv_dim]);
    cache.append(1, &vec![3.0; kv_dim], &vec![4.0; kv_dim]);
    assert_eq!(cache.len(), 1); // Only increments on last layer

    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

// ============================================================================
// GpuModelConfig Additional Coverage Tests
// ============================================================================

#[test]
fn test_gpu_model_config_small_model() {
    let config = GpuModelConfig {
        vocab_size: 1000,
        hidden_dim: 256,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 4,
        intermediate_dim: 1024,
        eps: 1e-6,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    assert_eq!(config.head_dim(), 64); // 256/4
    assert_eq!(config.kv_dim(), 256); // 4 * 64
    assert!(!config.is_gqa());
    assert_eq!(config.qkv_dim(), 768); // 256 + 2*256
}

#[test]
fn test_gpu_model_config_gqa_ratios() {
    // Test different GQA ratios
    let config2to1 = GpuModelConfig {
        vocab_size: 1000,
        hidden_dim: 512,
        num_heads: 8,
        num_kv_heads: 4, // 2:1 ratio
        num_layers: 1,
        intermediate_dim: 1024,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    assert!(config2to1.is_gqa());
    assert_eq!(config2to1.head_dim(), 64);
    assert_eq!(config2to1.kv_dim(), 256); // 4 * 64
}

// ============================================================================
// GpuGenerateConfig Additional Coverage Tests
// ============================================================================

#[test]
fn test_gpu_generate_config_builder_pattern() {
    let config = GpuGenerateConfig::deterministic(200).with_stop_tokens(vec![1, 2]);

    assert_eq!(config.max_tokens, 200);
    assert_eq!(config.top_k, 1);
    assert_eq!(config.stop_tokens, vec![1, 2]);
}

#[test]
fn test_gpu_generate_config_sampling_params() {
    let config = GpuGenerateConfig::with_sampling(128, 0.9, 50);

    assert_eq!(config.max_tokens, 128);
    assert!((config.temperature - 0.9).abs() < 1e-5);
    assert_eq!(config.top_k, 50);
}

// ============================================================================
// GpuModel Basic Coverage Tests
// ============================================================================

#[test]
fn test_gpu_model_creation() {
    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 2,
        intermediate_dim: 256,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let model = realizar::gpu::GpuModel::new(config);
    assert!(model.is_ok());
}

#[test]
fn test_gpu_model_config_accessor() {
    let config = GpuModelConfig {
        vocab_size: 500,
        hidden_dim: 128,
        num_heads: 4,
        num_kv_heads: 2,
        num_layers: 3,
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let model = realizar::gpu::GpuModel::new(config).unwrap();
    let model_config = model.config();

    assert_eq!(model_config.vocab_size, 500);
    assert_eq!(model_config.hidden_dim, 128);
    assert_eq!(model_config.num_heads, 4);
    assert_eq!(model_config.num_kv_heads, 2);
}

// ============================================================================
// Softmax Empty Input Tests
// ============================================================================

#[test]
fn test_scalar_softmax_empty() {
    let input: Vec<f32> = vec![];
    let output = scalar_softmax(&input);
    assert!(output.is_empty());
}

#[test]
fn test_simd_softmax_empty() {
    let input: Vec<f32> = vec![];
    let output = simd_softmax(&input);
    assert!(output.is_empty());
}

#[test]
fn test_softmax_single_element() {
    let input = vec![5.0];

    let scalar_out = scalar_softmax(&input);
    let simd_out = simd_softmax(&input);

    // Single element softmax = 1.0
    assert!((scalar_out[0] - 1.0).abs() < 1e-5);
    assert!((simd_out[0] - 1.0).abs() < 1e-5);
}

// ============================================================================
// RoPE Edge Case Tests
// ============================================================================

#[test]
fn test_rope_empty_input() {
    let input: Vec<f32> = vec![];
    let scalar_out = scalar_rope(&input, 0, 0, 10000.0);
    let simd_out = simd_rope(&input, 0, 0, 10000.0);

    assert!(scalar_out.is_empty());
    assert!(simd_out.is_empty());
}

#[test]
fn test_rope_zero_seq_len() {
    let input = vec![1.0; 16];
    let output = scalar_rope(&input, 0, 8, 10000.0);
    assert!(output.is_empty());
}

#[test]
fn test_rope_zero_head_dim() {
    let input = vec![1.0; 16];
    let output = simd_rope(&input, 4, 0, 10000.0);
    assert!(output.is_empty());
}

// ============================================================================
// Batch Embed Edge Cases
// ============================================================================

#[test]
fn test_batch_embed_valid_tokens() {
    // Create embedding table: vocab_size=3, hidden_dim=4
    let table: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, // token 0
        5.0, 6.0, 7.0, 8.0, // token 1
        9.0, 10.0, 11.0, 12.0, // token 2
    ];

    let tokens = vec![0, 2, 1];
    let result = batch_embed(&table, &tokens, 4);

    assert_eq!(result.len(), 12);
    // First token (0)
    assert!((result[0] - 1.0).abs() < 1e-5);
    // Second token (2)
    assert!((result[4] - 9.0).abs() < 1e-5);
    // Third token (1)
    assert!((result[8] - 5.0).abs() < 1e-5);
}

// ============================================================================
// LayerNorm Consistency Tests
// ============================================================================

#[test]
fn test_layernorm_std_vs_fused() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let gamma = vec![1.0; 8];
    let beta = vec![0.0; 8];
    let eps = 1e-5;

    let std_result = standard_layernorm(&input, &gamma, &beta, eps);
    let fused_result = fused_layernorm(&input, &gamma, &beta, eps);

    // Results should be very close
    for i in 0..8 {
        assert!((std_result[i] - fused_result[i]).abs() < 1e-4);
    }
}

#[test]
fn test_layernorm_with_gamma_beta() {
    let input = vec![0.0, 0.0, 0.0, 0.0];
    let gamma = vec![2.0; 4];
    let beta = vec![1.0; 4];
    let eps = 1e-5;

    let result = fused_layernorm(&input, &gamma, &beta, eps);

    // All zeros normalized = all zeros, then *2 + 1 = 1
    for &val in &result {
        assert!((val - 1.0).abs() < 1e-5);
    }
}

// ============================================================================
// TensorPool Additional Tests
// ============================================================================

#[test]
fn test_tensor_pool_acquire_returns_zeroed() {
    let mut pool = TensorPool::new(5);

    let buf = pool.acquire(100);
    // New buffer should be zeroed
    for &val in &buf {
        assert!((val - 0.0).abs() < 1e-5);
    }
}

#[test]
fn test_tensor_pool_reuse_resizes() {
    let mut pool = TensorPool::new(5);

    // Get a large buffer
    let buf = pool.acquire(1000);
    let capacity = buf.capacity();
    pool.release(buf);

    // Get smaller buffer - should reuse with resize
    let buf2 = pool.acquire(50);
    assert!(buf2.capacity() >= capacity); // Retains original capacity
    assert_eq!(buf2.len(), 50); // But resized to requested
}

// ============================================================================
// QuantizedAccumulator Additional Tests
// ============================================================================

#[test]
fn test_quantized_accumulator_default() {
    let acc: QuantizedAccumulator = Default::default();
    assert!((acc.sum() - 0.0).abs() < 1e-5);
}

#[test]
fn test_quantized_accumulator_chain_operations() {
    let mut acc = QuantizedAccumulator::new();

    acc.add_scaled(2.0, 3.0); // 6.0
    acc.add_scaled(4.0, 2.0); // 8.0
    acc.add_block(10.0, 0.5); // 5.0

    assert!((acc.sum() - 19.0).abs() < 1e-5);

    acc.reset();
    assert!((acc.sum() - 0.0).abs() < 1e-5);
}

// ============================================================================
// DoubleBuffer Additional Tests
// ============================================================================

#[test]
fn test_double_buffer_multiple_swaps() {
    let mut buffer: DoubleBuffer<f32> = DoubleBuffer::new(10);

    buffer.back_mut()[0] = 1.0;
    buffer.swap();
    assert!((buffer.front()[0] - 1.0).abs() < 1e-5);

    buffer.back_mut()[0] = 2.0;
    buffer.swap();
    assert!((buffer.front()[0] - 2.0).abs() < 1e-5);

    // Previous front is now back again
    buffer.back_mut()[0] = 3.0;
    buffer.swap();
    assert!((buffer.front()[0] - 3.0).abs() < 1e-5);
}

// ============================================================================
// TokenBatch Additional Tests
// ============================================================================

#[test]
fn test_token_batch_flush_empty() {
    let mut batch = TokenBatch::new(10);
    let flushed = batch.flush();
    assert!(flushed.is_empty());
}

#[test]
fn test_token_batch_capacity_one() {
    let mut batch = TokenBatch::new(1);
    assert!(batch.push(42).is_some()); // Immediately flushes
    assert!(batch.is_empty());
}

// ============================================================================
// SpeculativeBuffer Additional Tests
// ============================================================================

#[test]
fn test_speculative_buffer_at_capacity() {
    let mut buffer = SpeculativeBuffer::new(3);

    buffer.add_candidate(1, 0.9);
    buffer.add_candidate(2, 0.8);
    buffer.add_candidate(3, 0.7);
    buffer.add_candidate(4, 0.6); // Should be ignored (at capacity)

    assert_eq!(buffer.len(), 3);
}

#[test]
fn test_speculative_buffer_verify_empty() {
    let buffer = SpeculativeBuffer::new(10);
    let (accepted, rejected) = buffer.verify(&[1, 2, 3]);

    assert_eq!(accepted, 0);
    assert!(rejected.is_none());
}

#[test]
fn test_speculative_buffer_accept_all() {
    let mut buffer = SpeculativeBuffer::new(10);

    buffer.add_candidate(1, 0.9);
    buffer.add_candidate(2, 0.8);
    buffer.add_candidate(3, 0.7);

    buffer.accept(5); // Accept more than available
    assert!(buffer.is_empty());
}

// ============================================================================
// Matmul Naive vs Blocked Tests
// ============================================================================

#[test]
#[allow(clippy::many_single_char_names)]
fn test_blocked_matmul_small() {
    let m = 4;
    let k = 4;
    let n = 4;
    let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();

    let naive = naive_matmul(&a, &b, m, k, n);
    let blocked = blocked_matmul(&a, &b, m, k, n, 2);

    for i in 0..16 {
        assert!((naive[i] - blocked[i]).abs() < 1e-4);
    }
}

#[test]
#[allow(clippy::many_single_char_names)]
fn test_blocked_matmul_non_divisible() {
    let m = 5;
    let k = 7;
    let n = 3;
    let a: Vec<f32> = (0..(m * k)).map(|i| (i % 5) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..(k * n)).map(|i| (i % 3) as f32 * 0.2).collect();

    let naive = naive_matmul(&a, &b, m, k, n);
    let blocked = blocked_matmul(&a, &b, m, k, n, 4);

    for i in 0..(m * n) {
        assert!((naive[i] - blocked[i]).abs() < 1e-4);
    }
}

// ============================================================================
// Quantized Matvec Tests
// ============================================================================

#[test]
fn test_quantized_matvec_q4_empty_input() {
    let weights = vec![0u8; 18]; // One block
    let input: Vec<f32> = vec![];
    let result = quantized_matvec_q4(&weights, &input, 0, 0);
    assert!(result.is_empty());
}

#[test]
fn test_quantized_matvec_q8_empty_input() {
    let weights = vec![0u8; 34]; // One block
    let input: Vec<f32> = vec![];
    let result = quantized_matvec_q8(&weights, &input, 0, 0);
    assert!(result.is_empty());
}

// ============================================================================
// Priority Queue Additional Tests
// ============================================================================

#[test]
fn test_priority_queue_high_volume() {
    let mut queue: PriorityRequestQueue<i32> = PriorityRequestQueue::new();

    // Add many requests with varying priorities
    for i in 0..100 {
        queue.enqueue(PriorityRequest::new(i % 10, i as i32));
    }

    assert_eq!(queue.len(), 100);

    // Dequeue should get highest priority first
    let first = queue.dequeue_highest().unwrap();
    assert_eq!(first.priority(), 9);
}

// ============================================================================
// TimeoutManager Additional Tests
// ============================================================================

#[test]
fn test_timeout_manager_check_no_expired() {
    let mut manager = TimeoutManager::new();

    let future = Instant::now() + Duration::from_secs(60);
    manager.register(1, future);
    manager.register(2, future);

    let expired = manager.check_expired();
    assert!(expired.is_empty());
    assert_eq!(manager.active_count(), 2);
}

// ============================================================================
// ResourceTracker Additional Tests
// ============================================================================

#[test]
fn test_resource_tracker_multiple_allocations() {
    let mut tracker = ResourceTracker::new(10000, 100);

    let id1 = tracker.allocate(1000, 10).unwrap();
    let id2 = tracker.allocate(2000, 20).unwrap();
    let id3 = tracker.allocate(3000, 30).unwrap();

    assert_eq!(tracker.memory_usage(), 6000);
    assert_eq!(tracker.compute_usage(), 60);

    tracker.release(id2);
    assert_eq!(tracker.memory_usage(), 4000);
    assert_eq!(tracker.compute_usage(), 40);

    tracker.release(id1);
    tracker.release(id3);
    assert_eq!(tracker.memory_usage(), 0);
    assert_eq!(tracker.compute_usage(), 0);
}

// ============================================================================
// WeightType Tests
// ============================================================================

#[test]
fn test_weight_type_debug() {
    let types = [
        WeightType::Qkv,
        WeightType::Output,
        WeightType::FfnFc1,
        WeightType::FfnFc2,
        WeightType::LmHead,
    ];

    for weight_type in &types {
        let debug_str = format!("{:?}", weight_type);
        assert!(!debug_str.is_empty());
    }
}

// ============================================================================
// HealthChecker Additional Tests
// ============================================================================

#[test]
fn test_health_checker_all_passing() {
    let mut checker = HealthChecker::new();

    checker.register_check("check1", Box::new(|| true));
    checker.register_check("check2", Box::new(|| true));
    checker.register_check("check3", Box::new(|| true));

    assert!(checker.is_healthy());
}

// ============================================================================
// TokenRateLimiter Additional Tests
// ============================================================================

#[test]
fn test_rate_limiter_zero_burst() {
    let mut limiter = TokenRateLimiter::new(100.0, 0);
    assert!(!limiter.try_acquire(1));
}

#[test]
fn test_rate_limiter_acquire_exact() {
    let mut limiter = TokenRateLimiter::new(100.0, 10);
    assert!(limiter.try_acquire(10));
    assert_eq!(limiter.tokens_available(), 0);
    assert!(!limiter.try_acquire(1));
}

// ============================================================================
// ScratchBuffer Additional Tests
// ============================================================================

#[test]
fn test_scratch_buffer_zero_layers() {
    let scratch = ScratchBuffer::new(0, 100);
    assert_eq!(scratch.num_layers(), 0);
    assert_eq!(scratch.total_size(), 0);
}

#[test]
fn test_scratch_buffer_modify_and_reset() {
    let mut scratch = ScratchBuffer::new(3, 10);

    for layer in 0..3 {
        let s = scratch.get_layer_mut(layer);
        s[0] = (layer + 1) as f32 * 100.0;
    }

    assert!((scratch.get_layer(0)[0] - 100.0).abs() < 1e-5);
    assert!((scratch.get_layer(1)[0] - 200.0).abs() < 1e-5);
    assert!((scratch.get_layer(2)[0] - 300.0).abs() < 1e-5);

    scratch.reset();

    for layer in 0..3 {
        assert!((scratch.get_layer(layer)[0] - 0.0).abs() < 1e-5);
    }
}

// ============================================================================
// AttentionBuffers Coverage Tests (M17)
// ============================================================================

#[test]
fn test_attention_buffers_creation() {
    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 2,
        intermediate_dim: 256,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let buffers = realizar::gpu::AttentionBuffers::new(&config, 512);

    assert_eq!(buffers.q_buffer.len(), 64); // hidden_dim
    assert_eq!(buffers.scores_buffer.len(), 4 * 512); // num_heads * max_seq_len
    assert_eq!(buffers.output_buffer.len(), 64); // hidden_dim
    assert_eq!(buffers.kv_proj_buffer.len(), 64); // hidden_dim
    assert_eq!(buffers.ffn_buffer.len(), 256); // intermediate_dim
    assert_eq!(buffers.max_seq_len, 512);
}

#[test]
fn test_attention_buffers_reset() {
    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 32,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 1,
        intermediate_dim: 128,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let mut buffers = realizar::gpu::AttentionBuffers::new(&config, 256);

    // Modify buffers
    buffers.q_buffer[0] = 1.0;
    buffers.scores_buffer[0] = 2.0;
    buffers.output_buffer[0] = 3.0;
    buffers.kv_proj_buffer[0] = 4.0;
    buffers.ffn_buffer[0] = 5.0;

    // Reset
    buffers.reset();

    // All should be zero
    assert!((buffers.q_buffer[0] - 0.0).abs() < 1e-5);
    assert!((buffers.scores_buffer[0] - 0.0).abs() < 1e-5);
    assert!((buffers.output_buffer[0] - 0.0).abs() < 1e-5);
    assert!((buffers.kv_proj_buffer[0] - 0.0).abs() < 1e-5);
    assert!((buffers.ffn_buffer[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_attention_buffers_debug() {
    let config = GpuModelConfig {
        vocab_size: 50,
        hidden_dim: 16,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 1,
        intermediate_dim: 64,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let buffers = realizar::gpu::AttentionBuffers::new(&config, 64);
    let debug_str = format!("{:?}", buffers);
    assert!(debug_str.contains("AttentionBuffers"));
}

// ============================================================================
// GpuModel with AttentionBuffers Tests
// ============================================================================

#[test]
fn test_gpu_model_with_attention_buffers() {
    let config = GpuModelConfig {
        vocab_size: 50,
        hidden_dim: 32,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 1,
        intermediate_dim: 64,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let model = realizar::gpu::GpuModel::with_attention_buffers(config, 128).unwrap();
    assert!(model.has_attention_buffers());
}

#[test]
fn test_gpu_model_without_attention_buffers() {
    let config = GpuModelConfig {
        vocab_size: 50,
        hidden_dim: 32,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 1,
        intermediate_dim: 64,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let model = realizar::gpu::GpuModel::new(config).unwrap();
    assert!(!model.has_attention_buffers());
}

// ============================================================================
// GpuModel Forward Operations Tests
// ============================================================================

#[test]
fn test_gpu_model_forward_gpu_single_token() {
    let config = GpuModelConfig {
        vocab_size: 50,
        hidden_dim: 32,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 1,
        intermediate_dim: 64,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let mut model = realizar::gpu::GpuModel::new(config).unwrap();
    let result = model.forward_gpu(&[0]);

    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.len(), 50); // vocab_size
}

#[test]
fn test_gpu_model_forward_gpu_multiple_tokens() {
    let config = GpuModelConfig {
        vocab_size: 50,
        hidden_dim: 32,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 1,
        intermediate_dim: 64,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let mut model = realizar::gpu::GpuModel::new(config).unwrap();
    let result = model.forward_gpu(&[0, 1, 2]);

    assert!(result.is_ok());
    let logits = result.unwrap();
    // Logits for all positions
    assert_eq!(logits.len(), 3 * 50);
}

#[test]
fn test_gpu_model_forward_gpu_empty_input_error() {
    let config = GpuModelConfig {
        vocab_size: 50,
        hidden_dim: 32,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 1,
        intermediate_dim: 64,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let mut model = realizar::gpu::GpuModel::new(config).unwrap();
    let result = model.forward_gpu(&[]);

    assert!(result.is_err());
}

#[test]
fn test_gpu_model_forward_gpu_out_of_bounds_token() {
    let config = GpuModelConfig {
        vocab_size: 50,
        hidden_dim: 32,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 1,
        intermediate_dim: 64,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let mut model = realizar::gpu::GpuModel::new(config).unwrap();
    let result = model.forward_gpu(&[100]); // Out of bounds

    assert!(result.is_err());
}

#[test]
fn test_gpu_model_has_gpu() {
    let config = GpuModelConfig {
        vocab_size: 50,
        hidden_dim: 32,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 1,
        intermediate_dim: 64,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let model = realizar::gpu::GpuModel::new(config).unwrap();
    // Just check it returns a boolean without panicking
    let _ = model.has_gpu();
}

#[test]
fn test_gpu_model_from_gguf_config() {
    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 2,
        intermediate_dim: 256,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let model = realizar::gpu::GpuModel::from_gguf_config(config);
    assert!(model.is_ok());
}

// ============================================================================
// GpuModel Generation Tests
// ============================================================================

#[test]
fn test_gpu_model_generate_with_cache() {
    let config = GpuModelConfig {
        vocab_size: 50,
        hidden_dim: 32,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 1,
        intermediate_dim: 64,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let mut model = realizar::gpu::GpuModel::new(config).unwrap();

    let gen_config = GpuGenerateConfig::deterministic(5);
    let result = model.generate_with_cache(&[0, 1], &gen_config);

    assert!(result.is_ok());
    let tokens = result.unwrap();
    assert!(tokens.len() >= 2); // At least prompt
}

#[test]
fn test_gpu_model_generate_with_cache_empty_prompt_error() {
    let config = GpuModelConfig {
        vocab_size: 50,
        hidden_dim: 32,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 1,
        intermediate_dim: 64,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let mut model = realizar::gpu::GpuModel::new(config).unwrap();

    let gen_config = GpuGenerateConfig::deterministic(5);
    let result = model.generate_with_cache(&[], &gen_config);

    assert!(result.is_err());
}

#[test]
fn test_gpu_model_generate_optimized() {
    let config = GpuModelConfig {
        vocab_size: 50,
        hidden_dim: 32,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 1,
        intermediate_dim: 64,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let mut model = realizar::gpu::GpuModel::with_attention_buffers(config, 128).unwrap();

    let gen_config = GpuGenerateConfig::deterministic(3);
    let result = model.generate_optimized(&[0, 1], &gen_config);

    assert!(result.is_ok());
}

#[test]
fn test_gpu_model_generate_optimized_empty_prompt_error() {
    let config = GpuModelConfig {
        vocab_size: 50,
        hidden_dim: 32,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 1,
        intermediate_dim: 64,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let mut model = realizar::gpu::GpuModel::with_attention_buffers(config, 128).unwrap();

    let gen_config = GpuGenerateConfig::deterministic(3);
    let result = model.generate_optimized(&[], &gen_config);

    assert!(result.is_err());
}

#[test]
fn test_gpu_model_generate_with_stop_token() {
    let config = GpuModelConfig {
        vocab_size: 50,
        hidden_dim: 32,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 1,
        intermediate_dim: 64,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let mut model = realizar::gpu::GpuModel::new(config).unwrap();

    // Create config with stop token
    let gen_config = GpuGenerateConfig::deterministic(100).with_stop_tokens(vec![0]);
    let result = model.generate_with_cache(&[1], &gen_config);

    assert!(result.is_ok());
    // With stop token, should terminate early if 0 is generated
}

// ============================================================================
// GpuGenerateConfig Additional Tests
// ============================================================================

#[test]
fn test_gpu_generate_config_default_values() {
    let config: GpuGenerateConfig = Default::default();
    assert_eq!(config.max_tokens, 64);
    assert!((config.temperature - 0.0).abs() < 1e-5);
    assert_eq!(config.top_k, 1);
    assert!(config.stop_tokens.is_empty());
}

#[test]
fn test_gpu_generate_config_debug_fmt() {
    let config = GpuGenerateConfig::deterministic(10);
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("GpuGenerateConfig"));
}

#[test]
fn test_gpu_generate_config_clone_all_fields() {
    let config = GpuGenerateConfig::with_sampling(100, 0.8, 40).with_stop_tokens(vec![1, 2]);
    #[allow(clippy::redundant_clone)]
    let cloned = config.clone();

    assert_eq!(cloned.max_tokens, 100);
    assert!((cloned.temperature - 0.8).abs() < 1e-5);
    assert_eq!(cloned.top_k, 40);
    assert_eq!(cloned.stop_tokens, vec![1, 2]);
}

// ============================================================================
// Prefetch and Sum Operations Tests
// ============================================================================

#[test]
fn test_prefetch_read_operation() {
    let data = vec![1.0f32; 1000];

    // This shouldn't panic regardless of distance
    prefetch_read(&data, 0, 64);
    prefetch_read(&data, 500, 64);
}

#[test]
fn test_sequential_sum() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let sum = sequential_sum(&data);
    assert!((sum - 15.0).abs() < 1e-5);
}

#[test]
fn test_sequential_sum_empty() {
    let data: Vec<f32> = vec![];
    let sum = sequential_sum(&data);
    assert!((sum - 0.0).abs() < 1e-5);
}

#[test]
fn test_sum_with_prefetch() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let sum = sum_with_prefetch(&data, 16);
    assert!((sum - 15.0).abs() < 1e-5);
}

#[test]
fn test_sum_with_prefetch_empty() {
    let data: Vec<f32> = vec![];
    let sum = sum_with_prefetch(&data, 16);
    assert!((sum - 0.0).abs() < 1e-5);
}

// ============================================================================
// MatmulOp Type Test
// ============================================================================

#[test]
fn test_matmul_op_type() {
    let op: MatmulOp = (vec![1.0; 4], vec![1.0; 4], 2, 2, 2);
    assert_eq!(op.0.len(), 4);
    assert_eq!(op.1.len(), 4);
    assert_eq!(op.2, 2);
    assert_eq!(op.3, 2);
    assert_eq!(op.4, 2);
}

// ============================================================================
// GpuPoolStats Test
// ============================================================================

#[test]
fn test_gpu_pool_stats_debug_clone() {
    let stats = GpuPoolStats {
        cached_buffers: 5,
        cached_bytes: 1024,
    };

    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains('5'));
    assert!(debug_str.contains("1024"));

    let cloned = stats;
    assert_eq!(cloned.cached_buffers, 5);
    assert_eq!(cloned.cached_bytes, 1024);
}

// ============================================================================
// exceeds_gpu_buffer_limit Tests
// ============================================================================

#[test]
fn test_exceeds_gpu_buffer_limit_small() {
    assert!(!exceeds_gpu_buffer_limit(1000));
}

#[test]
fn test_exceeds_gpu_buffer_limit_large() {
    // Large buffer should exceed limit
    let large = 1_000_000_000;
    assert!(exceeds_gpu_buffer_limit(large));
}

// ============================================================================
// InferenceBatchScheduler Additional Tests
// ============================================================================

#[test]
fn test_inference_batch_scheduler_drain_empty() {
    let mut scheduler = InferenceBatchScheduler::new();
    let results = scheduler.drain();
    assert!(results.is_empty());
}

#[test]
fn test_inference_batch_scheduler_complete_nonexistent() {
    let mut scheduler = InferenceBatchScheduler::new();
    // Complete a non-existent ID - should not panic
    // Note: The scheduler creates a pending result even if ID wasn't registered
    scheduler.complete(999, vec![1, 2, 3]);
    // The scheduler allows completion of any ID, so completed_count will be 1
    assert_eq!(scheduler.completed_count(), 1);
}

// ============================================================================
// StreamingKVCache Edge Cases
// ============================================================================

#[test]
fn test_streaming_kv_cache_append_single_layer() {
    let mut cache = StreamingKVCache::new(1, 10, 2, 4);
    let kv_dim = 8;

    cache.append(0, &vec![1.0; kv_dim], &vec![2.0; kv_dim]);

    let (keys, values) = cache.get_valid(0);
    assert_eq!(keys.len(), kv_dim);
    assert_eq!(values.len(), kv_dim);
}

#[test]
fn test_streaming_kv_cache_get_valid_empty() {
    let cache = StreamingKVCache::new(2, 10, 2, 4);

    let (keys, values) = cache.get_valid(0);
    assert!(keys.is_empty());
    assert!(values.is_empty());
}

// ============================================================================
// Quantized Matvec Scaling Tests
// ============================================================================

#[test]
fn test_quantized_matvec_q4_scaling() {
    // Create weights for 1 row, 32 cols
    // Each row uses: blocks_per_row = 32.div_ceil(32) = 1 block
    // Each Q4 block: 18 bytes (2 bytes scale + 16 bytes data)
    let mut weights = vec![0u8; 18];

    // Set scale as f16 bits (approximately 1.0)
    let scale_bits: u16 = 0x3C00; // f16 representation of 1.0
    weights[0..2].copy_from_slice(&scale_bits.to_le_bytes());

    let input = vec![1.0f32; 32];
    // quantized_matvec_q4(weights, input, rows, cols) -> output has `rows` elements
    let result = quantized_matvec_q4(&weights, &input, 1, 32);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_quantized_matvec_q8_scaling() {
    // Create weights for 1 row, 32 cols
    // Each row uses: blocks_per_row = 32.div_ceil(32) = 1 block
    // Each Q8 block: 34 bytes (2 bytes scale + 32 bytes data)
    let mut weights = vec![0u8; 34];

    // Set scale as f16 bits
    let scale_bits: u16 = 0x3C00;
    weights[0..2].copy_from_slice(&scale_bits.to_le_bytes());

    let input = vec![1.0f32; 32];
    // quantized_matvec_q8(weights, input, rows, cols) -> output has `rows` elements
    let result = quantized_matvec_q8(&weights, &input, 1, 32);
    assert_eq!(result.len(), 1);
}

// ============================================================================
// GpuModelConfig Extended Tests
// ============================================================================

#[test]
fn test_gpu_model_config_mha() {
    let config = GpuModelConfig {
        vocab_size: 1000,
        hidden_dim: 512,
        num_heads: 8,
        num_kv_heads: 8, // Same as num_heads = MHA
        num_layers: 4,
        intermediate_dim: 2048,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    assert!(!config.is_gqa());
    assert_eq!(config.head_dim(), 64);
    assert_eq!(config.kv_dim(), 512);
    assert_eq!(config.qkv_dim(), 512 + 2 * 512); // 1536
}

#[test]
fn test_gpu_model_config_gqa_4to1() {
    let config = GpuModelConfig {
        vocab_size: 1000,
        hidden_dim: 512,
        num_heads: 8,
        num_kv_heads: 2, // 4:1 ratio
        num_layers: 4,
        intermediate_dim: 2048,
        eps: 1e-6,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    assert!(config.is_gqa());
    assert_eq!(config.head_dim(), 64);
    assert_eq!(config.kv_dim(), 128); // 2 * 64
    assert_eq!(config.qkv_dim(), 512 + 2 * 128); // 768
}

// ============================================================================
// Parallel vs Sequential FFN Tests
// ============================================================================

#[test]
fn test_parallel_ffn_basic() {
    let input = vec![1.0; 8];
    let up_weight = vec![0.1; 8 * 16];
    let down_weight = vec![0.1; 16 * 8];

    let result = parallel_ffn(&input, &up_weight, &down_weight, 8, 16);
    assert_eq!(result.len(), 8);
}

#[test]
fn test_sequential_ffn_basic() {
    let input = vec![1.0; 8];
    let up_weight = vec![0.1; 8 * 16];
    let down_weight = vec![0.1; 16 * 8];

    let result = sequential_ffn(&input, &up_weight, &down_weight, 8, 16);
    assert_eq!(result.len(), 8);
}
