//! Deep coverage tests for realizar/src/gpu.rs
//!
//! Tests non-GPU-gated functions and structs.

use realizar::gpu::{
    batch_embed, exceeds_gpu_buffer_limit, fused_layernorm, naive_matmul, parallel_ffn,
    prefetch_read, scalar_rope, scalar_softmax, sequential_ffn, sequential_sum, simd_rope,
    simd_softmax, standard_layernorm, sum_with_prefetch, CacheAlignedBuffer, ChunkedProcessor,
    ContiguousAttentionBuffer, DoubleBuffer, ForwardArena, GpuPipelineStage, InferencePipeline,
    QuantizedAccumulator, ScratchBuffer, TensorPool, TokenBatch,
};

// ============================================================================
// Test 1-10: Buffer limit and softmax
// ============================================================================

#[test]
fn test_exceeds_gpu_buffer_limit_small() {
    assert!(!exceeds_gpu_buffer_limit(1000));
}

#[test]
fn test_exceeds_gpu_buffer_limit_large() {
    // Test with a large but non-overflowing value
    assert!(exceeds_gpu_buffer_limit(1_000_000_000));
}

#[test]
fn test_scalar_softmax_simple() {
    let input = vec![1.0, 2.0, 3.0];
    let output = scalar_softmax(&input);
    assert_eq!(output.len(), 3);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 0.001);
}

#[test]
fn test_scalar_softmax_single() {
    let input = vec![1.0];
    let output = scalar_softmax(&input);
    assert_eq!(output.len(), 1);
    assert!((output[0] - 1.0).abs() < 0.001);
}

#[test]
fn test_scalar_softmax_zeros() {
    let input = vec![0.0, 0.0, 0.0];
    let output = scalar_softmax(&input);
    assert!((output[0] - output[1]).abs() < 0.001);
}

#[test]
fn test_simd_softmax_simple() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = simd_softmax(&input);
    assert_eq!(output.len(), 4);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 0.001);
}

#[test]
fn test_simd_softmax_large() {
    let input: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();
    let output = simd_softmax(&input);
    assert_eq!(output.len(), 256);
}

#[test]
fn test_softmax_scalar_vs_simd() {
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let scalar = scalar_softmax(&input);
    let simd = simd_softmax(&input);
    for i in 0..input.len() {
        assert!((scalar[i] - simd[i]).abs() < 0.001);
    }
}

// ============================================================================
// Test 11-20: RoPE functions
// ============================================================================

#[test]
fn test_scalar_rope_basic() {
    let input = vec![1.0; 64];
    let output = scalar_rope(&input, 1, 64, 10000.0);
    assert_eq!(output.len(), 64);
}

#[test]
fn test_scalar_rope_multi_seq() {
    let input = vec![1.0; 128];
    let output = scalar_rope(&input, 2, 64, 10000.0);
    assert_eq!(output.len(), 128);
}

#[test]
fn test_simd_rope_basic() {
    let input = vec![1.0; 64];
    let output = simd_rope(&input, 1, 64, 10000.0);
    assert_eq!(output.len(), 64);
}

#[test]
fn test_simd_rope_large() {
    let input = vec![1.0; 512];
    let output = simd_rope(&input, 4, 128, 10000.0);
    assert_eq!(output.len(), 512);
}

#[test]
fn test_rope_scalar_vs_simd() {
    let input: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
    let scalar = scalar_rope(&input, 2, 64, 10000.0);
    let simd = simd_rope(&input, 2, 64, 10000.0);
    for i in 0..input.len() {
        assert!((scalar[i] - simd[i]).abs() < 0.01, "Mismatch at {}", i);
    }
}

#[test]
fn test_rope_different_theta() {
    // Verify theta parameter is accepted and function works
    // Using seq_len=2 to have positions > 0 where rotation occurs
    let input: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
    let out1 = scalar_rope(&input, 2, 64, 10.0);
    let out2 = scalar_rope(&input, 2, 64, 10000.0);
    // Both should produce expected length output
    assert_eq!(out1.len(), 128);
    assert_eq!(out2.len(), 128);
    // Verify function executed without error
    assert!(!out1.is_empty());
    assert!(!out2.is_empty());
}

// ============================================================================
// Test 21-30: LayerNorm functions
// ============================================================================

#[test]
fn test_standard_layernorm_basic() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let output = standard_layernorm(&input, &gamma, &beta, 1e-5);
    assert_eq!(output.len(), 4);
}

#[test]
fn test_standard_layernorm_with_scale() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![2.0; 4];
    let beta = vec![1.0; 4];
    let output = standard_layernorm(&input, &gamma, &beta, 1e-5);
    assert_eq!(output.len(), 4);
}

#[test]
fn test_fused_layernorm_basic() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let output = fused_layernorm(&input, &gamma, &beta, 1e-5);
    assert_eq!(output.len(), 4);
}

#[test]
fn test_layernorm_standard_vs_fused() {
    let input: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let gamma = vec![1.0; 64];
    let beta = vec![0.0; 64];
    let standard = standard_layernorm(&input, &gamma, &beta, 1e-5);
    let fused = fused_layernorm(&input, &gamma, &beta, 1e-5);
    for i in 0..input.len() {
        assert!((standard[i] - fused[i]).abs() < 0.01);
    }
}

// ============================================================================
// Test 31-40: FFN functions
// ============================================================================

#[test]
fn test_sequential_ffn_basic() {
    let input = vec![1.0; 64];
    let up_weight = vec![0.1; 64 * 256];
    let down_weight = vec![0.1; 256 * 64];
    let output = sequential_ffn(&input, &up_weight, &down_weight, 64, 256);
    assert_eq!(output.len(), 64);
}

#[test]
fn test_parallel_ffn_basic() {
    let input = vec![1.0; 64];
    let up_weight = vec![0.1; 64 * 256];
    let down_weight = vec![0.1; 256 * 64];
    let output = parallel_ffn(&input, &up_weight, &down_weight, 64, 256);
    assert_eq!(output.len(), 64);
}

#[test]
fn test_ffn_sequential_vs_parallel() {
    let input: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let up_weight = vec![0.01; 32 * 128];
    let down_weight = vec![0.01; 128 * 32];
    let seq = sequential_ffn(&input, &up_weight, &down_weight, 32, 128);
    let par = parallel_ffn(&input, &up_weight, &down_weight, 32, 128);
    for i in 0..input.len() {
        assert!((seq[i] - par[i]).abs() < 0.1);
    }
}

// ============================================================================
// Test 41-50: Matrix and embedding operations
// ============================================================================

#[test]
fn test_naive_matmul_basic() {
    let a = vec![1.0; 4];
    let b = vec![1.0; 4];
    let output = naive_matmul(&a, &b, 2, 2, 2);
    assert_eq!(output.len(), 4);
}

#[test]
fn test_naive_matmul_identity() {
    let identity = vec![1.0, 0.0, 0.0, 1.0];
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let output = naive_matmul(&identity, &x, 2, 2, 2);
    for i in 0..4 {
        assert!((output[i] - x[i]).abs() < 0.001);
    }
}

#[test]
fn test_batch_embed_single() {
    let embedding_table = vec![0.1; 100 * 64];
    let tokens = vec![0, 1, 2];
    let output = batch_embed(&embedding_table, &tokens, 64);
    assert_eq!(output.len(), 3 * 64);
}

#[test]
fn test_batch_embed_empty() {
    let embedding_table = vec![0.1; 100 * 64];
    let tokens: Vec<usize> = vec![];
    let output = batch_embed(&embedding_table, &tokens, 64);
    assert!(output.is_empty());
}

#[test]
fn test_sequential_sum() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let sum = sequential_sum(&data);
    assert!((sum - 15.0).abs() < 0.001);
}

#[test]
fn test_sum_with_prefetch() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let sum = sum_with_prefetch(&data, 4);
    assert!((sum - 15.0).abs() < 0.001);
}

#[test]
fn test_sum_methods_equal() {
    let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let seq = sequential_sum(&data);
    let prefetch = sum_with_prefetch(&data, 8);
    assert!((seq - prefetch).abs() < 0.1);
}

#[test]
fn test_prefetch_read_no_panic() {
    let data = vec![1.0; 100];
    prefetch_read(&data, 0, 10);
    prefetch_read(&data, 50, 10);
}

// ============================================================================
// Test 51-60: CacheAlignedBuffer
// ============================================================================

#[test]
fn test_cache_aligned_buffer_new() {
    let buffer = CacheAlignedBuffer::new(1024);
    assert!(buffer.len() >= 1024);
}

#[test]
fn test_cache_aligned_buffer_empty() {
    let buffer = CacheAlignedBuffer::new(0);
    assert!(buffer.is_empty());
}

#[test]
fn test_cache_aligned_buffer_as_slice() {
    let buffer = CacheAlignedBuffer::new(100);
    let slice = buffer.as_slice();
    assert!(slice.len() >= 100);
}

#[test]
fn test_cache_aligned_buffer_as_mut_slice() {
    let mut buffer = CacheAlignedBuffer::new(100);
    let slice = buffer.as_mut_slice();
    slice[0] = 1.0;
    assert_eq!(buffer.as_slice()[0], 1.0);
}

// ============================================================================
// Test 61-70: TensorPool
// ============================================================================

#[test]
fn test_tensor_pool_new() {
    let pool = TensorPool::new(10);
    assert_eq!(pool.capacity(), 10);
}

#[test]
fn test_tensor_pool_available() {
    let pool = TensorPool::new(5);
    assert_eq!(pool.available(), 0);
}

#[test]
fn test_tensor_pool_acquire_release() {
    let mut pool = TensorPool::new(5);
    let tensor = pool.acquire(100);
    assert_eq!(tensor.len(), 100);
    pool.release(tensor);
    assert_eq!(pool.available(), 1);
}

#[test]
fn test_tensor_pool_clear() {
    let mut pool = TensorPool::new(5);
    pool.release(vec![1.0; 100]);
    pool.clear();
    assert_eq!(pool.available(), 0);
}

// ============================================================================
// Test 71-80: ForwardArena
// ============================================================================

#[test]
fn test_forward_arena_new() {
    let arena = ForwardArena::new(4096);
    assert_eq!(arena.capacity(), 4096);
}

#[test]
fn test_forward_arena_alloc() {
    let mut arena = ForwardArena::new(4096);
    let slice = arena.alloc(256);
    assert_eq!(slice.len(), 256);
}

#[test]
fn test_forward_arena_used() {
    let mut arena = ForwardArena::new(4096);
    arena.alloc(1000);
    assert_eq!(arena.used(), 1000);
}

#[test]
fn test_forward_arena_reset() {
    let mut arena = ForwardArena::new(4096);
    arena.alloc(1024);
    arena.reset();
    assert_eq!(arena.used(), 0);
}

// ============================================================================
// Test 81-90: ScratchBuffer
// ============================================================================

#[test]
fn test_scratch_buffer_new() {
    let buffer = ScratchBuffer::new(4, 1024);
    assert_eq!(buffer.num_layers(), 4);
    assert_eq!(buffer.layer_size(), 1024);
}

#[test]
fn test_scratch_buffer_total_size() {
    let buffer = ScratchBuffer::new(4, 1024);
    assert_eq!(buffer.total_size(), 4 * 1024);
}

#[test]
fn test_scratch_buffer_get_layer() {
    let buffer = ScratchBuffer::new(2, 100);
    let layer = buffer.get_layer(0);
    assert_eq!(layer.len(), 100);
}

#[test]
fn test_scratch_buffer_get_layer_mut() {
    let mut buffer = ScratchBuffer::new(2, 100);
    let layer = buffer.get_layer_mut(0);
    layer[0] = 1.0;
    assert_eq!(buffer.get_layer(0)[0], 1.0);
}

#[test]
fn test_scratch_buffer_reset() {
    let mut buffer = ScratchBuffer::new(2, 100);
    buffer.get_layer_mut(0)[0] = 1.0;
    buffer.reset();
    assert_eq!(buffer.get_layer(0)[0], 0.0);
}

// ============================================================================
// Test 91-100: QuantizedAccumulator
// ============================================================================

#[test]
fn test_quantized_accumulator_new() {
    let acc = QuantizedAccumulator::new();
    assert_eq!(acc.sum(), 0.0);
}

#[test]
fn test_quantized_accumulator_add_scaled() {
    let mut acc = QuantizedAccumulator::new();
    acc.add_scaled(2.0, 0.5);
    assert!((acc.sum() - 1.0).abs() < 0.001);
}

#[test]
fn test_quantized_accumulator_add_block() {
    let mut acc = QuantizedAccumulator::new();
    acc.add_block(10.0, 2.0);
    assert!((acc.sum() - 20.0).abs() < 0.001);
}

#[test]
fn test_quantized_accumulator_reset() {
    let mut acc = QuantizedAccumulator::new();
    acc.add_scaled(5.0, 1.0);
    acc.reset();
    assert_eq!(acc.sum(), 0.0);
}

// ============================================================================
// Test 101-110: DoubleBuffer
// ============================================================================

#[test]
fn test_double_buffer_new() {
    let buffer: DoubleBuffer<f32> = DoubleBuffer::new(100);
    assert_eq!(buffer.capacity(), 100);
}

#[test]
fn test_double_buffer_front() {
    let buffer: DoubleBuffer<f32> = DoubleBuffer::new(100);
    assert_eq!(buffer.front().len(), 100);
}

#[test]
fn test_double_buffer_back_mut() {
    let mut buffer: DoubleBuffer<f32> = DoubleBuffer::new(100);
    buffer.back_mut()[0] = 1.0;
}

#[test]
fn test_double_buffer_swap() {
    let mut buffer: DoubleBuffer<f32> = DoubleBuffer::new(100);
    buffer.back_mut()[0] = 42.0;
    buffer.swap();
    assert_eq!(buffer.front()[0], 42.0);
}

// ============================================================================
// Test 111-120: ChunkedProcessor
// ============================================================================

#[test]
fn test_chunked_processor_new() {
    let processor = ChunkedProcessor::new(64);
    assert_eq!(processor.chunk_size(), 64);
}

#[test]
fn test_chunked_processor_num_chunks() {
    let processor = ChunkedProcessor::new(32);
    assert_eq!(processor.num_chunks(128), 4);
}

#[test]
fn test_chunked_processor_chunk_bounds() {
    let processor = ChunkedProcessor::new(32);
    let (start, end) = processor.chunk_bounds(0, 100);
    assert_eq!(start, 0);
    assert_eq!(end, 32);
}

#[test]
fn test_chunked_processor_process() {
    let processor = ChunkedProcessor::new(32);
    let data: Vec<f32> = (0..128).map(|i| i as f32).collect();
    let sum = processor.process_chunks(&data, |chunk| chunk.iter().sum::<f32>());
    // Sum of 0..128 = 127 * 128 / 2 = 8128
    assert!((sum - 8128.0).abs() < 0.1);
}

// ============================================================================
// Test 121-130: InferencePipeline
// ============================================================================

#[test]
fn test_inference_pipeline_new() {
    let pipeline = InferencePipeline::new(4);
    assert_eq!(pipeline.num_stages(), 4);
}

#[test]
fn test_inference_pipeline_record_stage() {
    let mut pipeline = InferencePipeline::new(4);
    pipeline.record_stage_time(GpuPipelineStage::Attention, 10.0);
    assert!(pipeline.total_latency() >= 10.0);
}

#[test]
fn test_inference_pipeline_reset() {
    let mut pipeline = InferencePipeline::new(4);
    pipeline.record_stage_time(GpuPipelineStage::FFN, 5.0);
    pipeline.reset();
    assert_eq!(pipeline.total_latency(), 0.0);
}

#[test]
fn test_gpu_pipeline_stage_variants() {
    let _ = GpuPipelineStage::Embed;
    let _ = GpuPipelineStage::Attention;
    let _ = GpuPipelineStage::FFN;
    let _ = GpuPipelineStage::Output;
}

// ============================================================================
// Test 131-140: TokenBatch
// ============================================================================

#[test]
fn test_token_batch_new() {
    let batch = TokenBatch::new(32);
    assert_eq!(batch.capacity(), 32);
}

#[test]
fn test_token_batch_len() {
    let batch = TokenBatch::new(32);
    assert_eq!(batch.len(), 0);
}

#[test]
fn test_token_batch_is_empty() {
    let batch = TokenBatch::new(32);
    assert!(batch.is_empty());
}

#[test]
fn test_token_batch_push() {
    let mut batch = TokenBatch::new(32);
    let result = batch.push(42);
    assert!(result.is_none());
    assert_eq!(batch.len(), 1);
}

#[test]
fn test_token_batch_is_full() {
    let mut batch = TokenBatch::new(2);
    batch.push(1);
    // After push(1), batch has 1 element, not full
    assert!(!batch.is_full());
    // push(2) makes it full and auto-flushes, returning Some(tokens)
    let result = batch.push(2);
    assert!(result.is_some()); // Batch was full and flushed
    assert_eq!(result.unwrap(), vec![1, 2]);
}

#[test]
fn test_token_batch_flush() {
    let mut batch = TokenBatch::new(32);
    batch.push(1);
    batch.push(2);
    let tokens = batch.flush();
    assert_eq!(tokens, vec![1, 2]);
    assert!(batch.is_empty());
}

// ============================================================================
// Test 141-150: ContiguousAttentionBuffer
// ============================================================================

#[test]
fn test_contiguous_attention_buffer_new() {
    let buffer = ContiguousAttentionBuffer::new(512, 8, 64);
    assert!(buffer.max_seq_len() == 512);
}

#[test]
fn test_contiguous_attention_buffer_is_contiguous() {
    let buffer = ContiguousAttentionBuffer::new(256, 4, 32);
    assert!(buffer.is_contiguous());
}

#[test]
fn test_contiguous_attention_buffer_get_views() {
    let buffer = ContiguousAttentionBuffer::new(128, 4, 32);
    let (q, k, v, out) = buffer.get_views();
    assert!(!q.is_empty());
    assert!(!k.is_empty());
    assert!(!v.is_empty());
    assert!(!out.is_empty());
}

#[test]
fn test_contiguous_attention_buffer_reset() {
    let mut buffer = ContiguousAttentionBuffer::new(128, 4, 32);
    buffer.reset();
    // After reset, buffer should still be usable
    let (q, _, _, _) = buffer.get_views();
    assert!(!q.is_empty());
}
