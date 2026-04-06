//! Part 10: GPU Module Coverage Tests (gpu/mod.rs)
//!
//! Comprehensive tests for gpu/mod.rs covering:
//! - GPU Buffer Limits (exceeds_gpu_buffer_limit, LARGE_VOCAB_THRESHOLD)
//! - ContiguousAttentionBuffer (all methods)
//! - Batch Processing (batch_embed, sequential_ffn, parallel_ffn, layernorm variants)
//! - Quantized Compute Kernels (quantized_dot_q4/q8, quantized_matvec_q4/q8)
//! - QuantizedAccumulator
//! - Streaming/Pipelining (DoubleBuffer, ChunkedProcessor, InferencePipeline)
//! - Error Recovery (ErrorRecoveryStrategy, DegradationManager, FailureIsolator)
//! - Connection Pooling (ConnectionPool, ConnectionConfig, ConnectionState)
//! - Resource Limits (ResourceLimiter, ResourceConfig, ResourceMonitor)
//! - GGUF Model State (GgufModelState, load_gguf_to_gpu)

use std::io::{Error, ErrorKind};
use std::time::Duration;

use crate::gpu::{
    batch_embed, exceeds_gpu_buffer_limit, fused_layernorm, load_gguf_to_gpu, parallel_ffn,
    quantized_dot_q4, quantized_dot_q8, quantized_matvec_q4, quantized_matvec_q8, sequential_ffn,
    standard_layernorm, ChunkedProcessor, ConnectionConfig, ConnectionPool, ConnectionState,
    ContiguousAttentionBuffer, DegradationManager, DegradationMode, DoubleBuffer,
    ErrorClassification, ErrorRecoveryStrategy, FailureIsolator, GgufModelState, GpuPipelineStage,
    InferencePipeline, LimitResult, QuantizedAccumulator, RecoveryAction, RequestOutcome,
    ResourceConfig, ResourceLimiter, ResourceMonitor, SystemLoad, LARGE_VOCAB_THRESHOLD,
};

// ============================================================================
// GPU Buffer Limits Tests
// ============================================================================

#[test]
fn test_exceeds_gpu_buffer_limit_small() {
    // Small buffer should not exceed limit
    let small_elements = 1000;
    assert!(!exceeds_gpu_buffer_limit(small_elements));
}

#[test]
fn test_exceeds_gpu_buffer_limit_large() {
    // Buffer larger than 256MB in f32 (256MB / 4 = 64M elements)
    let large_elements = 70_000_000; // ~280MB
    assert!(exceeds_gpu_buffer_limit(large_elements));
}

#[test]
fn test_exceeds_gpu_buffer_limit_boundary() {
    // Exactly at limit: 256MB / 4 bytes = 67,108,864 elements
    let at_limit = 67_108_864;
    assert!(!exceeds_gpu_buffer_limit(at_limit));

    // Just over the limit
    let over_limit = 67_108_865;
    assert!(exceeds_gpu_buffer_limit(over_limit));
}

#[test]
fn test_large_vocab_threshold_value() {
    // Verify the threshold constant
    assert_eq!(LARGE_VOCAB_THRESHOLD, 65536);
}

#[test]
fn test_exceeds_gpu_buffer_limit_zero() {
    // Zero elements should not exceed
    assert!(!exceeds_gpu_buffer_limit(0));
}

// ============================================================================
// ContiguousAttentionBuffer Tests
// ============================================================================

#[test]
fn test_contiguous_attention_buffer_new() {
    let max_seq_len = 64;
    let num_heads = 4;
    let head_dim = 16;

    let buffer = ContiguousAttentionBuffer::new(max_seq_len, num_heads, head_dim);

    assert!(buffer.is_contiguous());
    assert_eq!(buffer.max_seq_len(), max_seq_len);
}

#[test]
fn test_contiguous_attention_buffer_get_views() {
    let max_seq_len = 4;
    let num_heads = 2;
    let head_dim = 8;
    let tensor_size = max_seq_len * num_heads * head_dim;

    let buffer = ContiguousAttentionBuffer::new(max_seq_len, num_heads, head_dim);
    let (q, k, v, o) = buffer.get_views();

    assert_eq!(q.len(), tensor_size);
    assert_eq!(k.len(), tensor_size);
    assert_eq!(v.len(), tensor_size);
    assert_eq!(o.len(), tensor_size);

    // All should be zero-initialized
    assert!(q.iter().all(|&x| x == 0.0));
}

#[test]
fn test_contiguous_attention_buffer_get_views_mut() {
    let max_seq_len = 2;
    let num_heads = 2;
    let head_dim = 4;

    let mut buffer = ContiguousAttentionBuffer::new(max_seq_len, num_heads, head_dim);

    {
        let (q, k, v, o) = buffer.get_views_mut();

        // Fill with test values
        q.fill(1.0);
        k.fill(2.0);
        v.fill(3.0);
        o.fill(4.0);
    }

    // Verify values persisted
    let (q, k, v, o) = buffer.get_views();
    assert!(q.iter().all(|&x| (x - 1.0).abs() < 1e-6));
    assert!(k.iter().all(|&x| (x - 2.0).abs() < 1e-6));
    assert!(v.iter().all(|&x| (x - 3.0).abs() < 1e-6));
    assert!(o.iter().all(|&x| (x - 4.0).abs() < 1e-6));
}

#[test]
fn test_contiguous_attention_buffer_reset() {
    let mut buffer = ContiguousAttentionBuffer::new(4, 2, 8);

    // Fill with non-zero values
    {
        let (q, k, v, o) = buffer.get_views_mut();
        q.fill(1.0);
        k.fill(2.0);
        v.fill(3.0);
        o.fill(4.0);
    }

    // Reset
    buffer.reset();

    // All should be zero
    let (q, k, v, o) = buffer.get_views();
    assert!(q.iter().all(|&x| x == 0.0));
    assert!(k.iter().all(|&x| x == 0.0));
    assert!(v.iter().all(|&x| x == 0.0));
    assert!(o.iter().all(|&x| x == 0.0));
}

#[test]
fn test_contiguous_attention_buffer_is_contiguous() {
    let buffer = ContiguousAttentionBuffer::new(8, 4, 16);
    // By construction, this is always true
    assert!(buffer.is_contiguous());
}

// ============================================================================
// Batch Processing Tests - batch_embed
// ============================================================================

#[test]
fn test_batch_embed_basic() {
    let hidden_dim = 4;
    let vocab_size = 10;

    // Create embedding table: each token i has embedding [i, i, i, i]
    let embedding_table: Vec<f32> = (0..vocab_size)
        .flat_map(|i| vec![i as f32; hidden_dim])
        .collect();

    let tokens = vec![0, 2, 5];
    let result = batch_embed(&embedding_table, &tokens, hidden_dim);

    assert_eq!(result.len(), tokens.len() * hidden_dim);

    // Token 0 -> [0, 0, 0, 0]
    assert!(result[0..hidden_dim].iter().all(|&x| x == 0.0));

    // Token 2 -> [2, 2, 2, 2]
    assert!(result[hidden_dim..2 * hidden_dim]
        .iter()
        .all(|&x| (x - 2.0).abs() < 1e-6));

    // Token 5 -> [5, 5, 5, 5]
    assert!(result[2 * hidden_dim..3 * hidden_dim]
        .iter()
        .all(|&x| (x - 5.0).abs() < 1e-6));
}

#[test]
fn test_batch_embed_empty_tokens() {
    let embedding_table = vec![1.0, 2.0, 3.0, 4.0];
    let tokens: Vec<usize> = vec![];

    let result = batch_embed(&embedding_table, &tokens, 4);
    assert!(result.is_empty());
}

#[test]
fn test_batch_embed_empty_embedding_table() {
    let embedding_table: Vec<f32> = vec![];
    let tokens = vec![0, 1, 2];

    let result = batch_embed(&embedding_table, &tokens, 4);
    assert!(result.is_empty());
}

#[test]
fn test_batch_embed_out_of_bounds_token() {
    let hidden_dim = 4;
    let vocab_size = 5;
    let embedding_table: Vec<f32> = vec![1.0; vocab_size * hidden_dim];

    // Token 10 is out of bounds
    let tokens = vec![1, 10];
    let result = batch_embed(&embedding_table, &tokens, hidden_dim);

    // Should pad with zeros for out-of-bounds
    assert_eq!(result.len(), 2 * hidden_dim);
    // Second embedding should be zeros
    assert!(result[hidden_dim..].iter().all(|&x| x == 0.0));
}

#[test]
fn test_batch_embed_single_token() {
    let hidden_dim = 3;
    let embedding_table = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 tokens
    let tokens = vec![1];

    let result = batch_embed(&embedding_table, &tokens, hidden_dim);
    assert_eq!(result, vec![4.0, 5.0, 6.0]);
}

// ============================================================================
// Batch Processing Tests - FFN
// ============================================================================

#[test]
fn test_sequential_ffn_basic() {
    let hidden_dim = 4;
    let intermediate_dim = 8;

    let input = vec![1.0f32; hidden_dim];
    let w_up = vec![0.1f32; hidden_dim * intermediate_dim];
    let w_down = vec![0.1f32; intermediate_dim * hidden_dim];

    let result = sequential_ffn(&input, &w_up, &w_down, hidden_dim, intermediate_dim);

    assert_eq!(result.len(), hidden_dim);
    // Result should be non-zero due to GELU activation
}

#[test]
fn test_sequential_ffn_empty_input() {
    let result = sequential_ffn(&[], &[1.0; 8], &[1.0; 8], 2, 4);
    assert!(result.is_empty());
}

#[test]
fn test_parallel_ffn_basic() {
    let hidden_dim = 4;
    let intermediate_dim = 8;

    let input = vec![1.0f32; hidden_dim];
    let w_up = vec![0.1f32; hidden_dim * intermediate_dim];
    let w_down = vec![0.1f32; intermediate_dim * hidden_dim];

    let result = parallel_ffn(&input, &w_up, &w_down, hidden_dim, intermediate_dim);

    assert_eq!(result.len(), hidden_dim);
}

#[test]
fn test_parallel_ffn_empty_input() {
    let result = parallel_ffn(&[], &[1.0; 8], &[1.0; 8], 2, 4);
    assert!(result.is_empty());
}

#[test]
fn test_sequential_vs_parallel_ffn_equivalence() {
    let hidden_dim = 8;
    let intermediate_dim = 16;

    let input: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.1).collect();
    let w_up: Vec<f32> = (0..hidden_dim * intermediate_dim)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let w_down: Vec<f32> = (0..intermediate_dim * hidden_dim)
        .map(|i| (i as f32) * 0.01)
        .collect();

    let seq_result = sequential_ffn(&input, &w_up, &w_down, hidden_dim, intermediate_dim);
    let par_result = parallel_ffn(&input, &w_up, &w_down, hidden_dim, intermediate_dim);

    // Results should be very close
    for (s, p) in seq_result.iter().zip(par_result.iter()) {
        assert!(
            (s - p).abs() < 1e-4,
            "Sequential and parallel FFN should match"
        );
    }
}

// ============================================================================
// Batch Processing Tests - Layer Normalization
// ============================================================================

#[test]
fn test_standard_layernorm_basic() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];

    let result = standard_layernorm(&input, &gamma, &beta, 1e-5);

    assert_eq!(result.len(), 4);

    // Mean of normalized output should be close to 0
    let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
    assert!(mean.abs() < 1e-5, "Mean should be ~0 after layernorm");
}

#[test]
fn test_standard_layernorm_empty_input() {
    let result = standard_layernorm(&[], &[1.0], &[0.0], 1e-5);
    assert!(result.is_empty());
}

#[test]
fn test_fused_layernorm_basic() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];

    let result = fused_layernorm(&input, &gamma, &beta, 1e-5);

    assert_eq!(result.len(), 4);

    // Mean should be close to 0
    let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
    assert!(mean.abs() < 1e-5);
}

#[test]
fn test_fused_layernorm_empty_input() {
    let result = fused_layernorm(&[], &[1.0], &[0.0], 1e-5);
    assert!(result.is_empty());
}

#[test]
fn test_standard_vs_fused_layernorm_equivalence() {
    let input: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5 - 4.0).collect();
    let gamma: Vec<f32> = vec![2.0; 16];
    let beta: Vec<f32> = vec![0.5; 16];

    let std_result = standard_layernorm(&input, &gamma, &beta, 1e-5);
    let fused_result = fused_layernorm(&input, &gamma, &beta, 1e-5);

    // Results should be very close
    for (s, f) in std_result.iter().zip(fused_result.iter()) {
        assert!(
            (s - f).abs() < 1e-4,
            "Standard and fused layernorm should match"
        );
    }
}

#[test]
fn test_layernorm_with_gamma_beta() {
    let input = vec![0.0, 1.0, 2.0, 3.0];
    let gamma = vec![2.0; 4]; // Scale by 2
    let beta = vec![1.0; 4]; // Shift by 1

    let result = fused_layernorm(&input, &gamma, &beta, 1e-5);

    // After normalization, scaling and shifting
    // Mean of result should be close to beta (1.0) since normalized mean is 0
    let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
    assert!((mean - 1.0).abs() < 0.1, "Mean should be shifted by beta");
}

// ============================================================================
// Quantized Compute Kernel Tests - Q4
// ============================================================================

#[test]
fn test_quantized_dot_q4_basic() {
    // Q4_0 block: 2 bytes scale + 16 bytes data = 18 bytes
    let block_a: Vec<u8> = vec![
        0x00, 0x3c, // f16 1.0 scale
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
        0xff,
    ];
    let block_b: Vec<u8> = vec![
        0x00, 0x3c, // f16 1.0 scale
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
        0xff,
    ];

    let result = quantized_dot_q4(&block_a, &block_b);
    // Result should be non-zero (self dot product)
    assert!(result != 0.0);
}

#[test]
fn test_quantized_dot_q4_too_short() {
    let short_block = vec![0u8; 10]; // Less than 18 bytes
    let result = quantized_dot_q4(&short_block, &short_block);
    assert_eq!(result, 0.0);
}

#[test]
fn test_quantized_dot_q4_zeros() {
    // Scale = 0, so output should be 0
    let block = vec![0u8; 18];
    let result = quantized_dot_q4(&block, &block);
    assert_eq!(result, 0.0);
}

// ============================================================================
// Quantized Compute Kernel Tests - Q8
// ============================================================================

#[test]
fn test_quantized_dot_q8_basic() {
    // Q8_0 block: 2 bytes scale + 32 bytes data = 34 bytes
    let mut block_a = vec![0u8; 34];
    block_a[0] = 0x00;
    block_a[1] = 0x3c; // f16 1.0 scale
                       // Fill data with some values
    for i in 2..34 {
        block_a[i] = i as u8;
    }

    let result = quantized_dot_q8(&block_a, &block_a);
    assert!(result != 0.0);
}

#[test]
fn test_quantized_dot_q8_too_short() {
    let short_block = vec![0u8; 20]; // Less than 34 bytes
    let result = quantized_dot_q8(&short_block, &short_block);
    assert_eq!(result, 0.0);
}

include!("quantized_dot_matvec.rs");
include!("error_recovery.rs");
include!("resource_limiter.rs");
