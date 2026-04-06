//! CUDA Weights Tests (PMAT-018: weights.rs coverage)
//!
//! Tests for GPU config structs, encoder/decoder block weights,
//! conv frontend weights, forward encoder block, and KV cache.

use crate::driver::CudaContext;
use crate::memory::resident::GpuResidentTensor;

/// Helper to create CUDA context, skipping test if unavailable
macro_rules! cuda_ctx {
    () => {
        match CudaContext::new(0) {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Skipping CUDA test: {:?}", e);
                return;
            }
        }
    };
}

// ============================================================================
// Config Creation Tests
// ============================================================================

#[test]
fn test_gpu_encoder_config_creation() {
    use crate::memory::resident::GpuEncoderConfig;

    let config = GpuEncoderConfig {
        d_model: 256,
        n_heads: 4,
        ffn_dim: 1024,
    };
    assert_eq!(config.d_model, 256);
    assert_eq!(config.n_heads, 4);
    assert_eq!(config.ffn_dim, 1024);
}

#[test]
fn test_gpu_decoder_config_creation() {
    use crate::memory::resident::GpuDecoderConfig;

    let config = GpuDecoderConfig {
        d_model: 512,
        n_heads: 8,
        ffn_dim: 2048,
        max_seq_len: 1024,
        n_layers: 6,
    };
    assert_eq!(config.d_model, 512);
    assert_eq!(config.n_heads, 8);
    assert_eq!(config.ffn_dim, 2048);
    assert_eq!(config.max_seq_len, 1024);
    assert_eq!(config.n_layers, 6);
}

#[test]
fn test_gpu_kv_cache_creation() {
    use crate::memory::resident::GpuKvCache;
    let ctx = cuda_ctx!();

    let d_model = 256usize;
    let max_seq_len = 512usize;
    let cache_size = max_seq_len * d_model;

    let key = GpuResidentTensor::from_host(&ctx, &vec![0.0f32; cache_size]).unwrap();
    let value = GpuResidentTensor::from_host(&ctx, &vec![0.0f32; cache_size]).unwrap();

    let kv_cache = GpuKvCache {
        key,
        value,
        seq_len: 0,
        max_seq_len,
        d_model,
    };
    assert_eq!(kv_cache.seq_len, 0);
    assert_eq!(kv_cache.max_seq_len, max_seq_len);
    assert_eq!(kv_cache.d_model, d_model);
}

// ============================================================================
// Encoder Block Weights Tests
// ============================================================================

#[test]
fn test_gpu_encoder_block_weights_structure() {
    use crate::memory::resident::GpuEncoderBlockWeights;
    let ctx = cuda_ctx!();

    let d_model = 64usize;
    let ffn_dim = 256usize;

    // Create minimal encoder block weights
    let weights = GpuEncoderBlockWeights {
        ln1_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln1_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_q: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        b_q: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_k: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        b_k: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_v: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        b_v: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_o: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        b_o: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        ln2_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln2_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        ffn_up_w: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * ffn_dim]).unwrap(),
        ffn_up_b: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; ffn_dim]).unwrap(),
        ffn_down_w: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; ffn_dim * d_model]).unwrap(),
        ffn_down_b: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
    };

    // Verify all tensors are device resident
    assert!(weights.ln1_gamma.is_device_resident());
    assert!(weights.w_q.is_device_resident());
    assert!(weights.ffn_up_w.is_device_resident());
}

// ============================================================================
// Forward Encoder Block Tests
// ============================================================================

#[test]
fn test_forward_encoder_block_gpu() {
    use crate::memory::resident::{
        clear_kernel_cache, forward_encoder_block_gpu, GpuEncoderBlockWeights, GpuEncoderConfig,
    };
    // Clear cache because kernel modules are tied to CUDA contexts
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let d_model = 32usize; // Small for testing
    let n_heads = 2u32;
    let ffn_dim = 128usize;
    let seq_len = 4usize;

    let config = GpuEncoderConfig {
        d_model: d_model as u32,
        n_heads,
        ffn_dim: ffn_dim as u32,
    };

    // Create weights
    let weights = GpuEncoderBlockWeights {
        ln1_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln1_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_q: GpuResidentTensor::from_host(&ctx, &vec![0.02f32; d_model * d_model]).unwrap(),
        b_q: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_k: GpuResidentTensor::from_host(&ctx, &vec![0.02f32; d_model * d_model]).unwrap(),
        b_k: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_v: GpuResidentTensor::from_host(&ctx, &vec![0.02f32; d_model * d_model]).unwrap(),
        b_v: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_o: GpuResidentTensor::from_host(&ctx, &vec![0.02f32; d_model * d_model]).unwrap(),
        b_o: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        ln2_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln2_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        ffn_up_w: GpuResidentTensor::from_host(&ctx, &vec![0.02f32; d_model * ffn_dim]).unwrap(),
        ffn_up_b: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; ffn_dim]).unwrap(),
        ffn_down_w: GpuResidentTensor::from_host(&ctx, &vec![0.02f32; ffn_dim * d_model]).unwrap(),
        ffn_down_b: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
    };

    // Input: [seq_len * d_model]
    let input_data: Vec<f32> = (0..(seq_len * d_model))
        .map(|i| (i as f32) * 0.01)
        .collect();
    let input = GpuResidentTensor::from_host(&ctx, &input_data).unwrap();

    let output = forward_encoder_block_gpu(&ctx, &input, &weights, &config).unwrap();
    assert_eq!(output.len(), seq_len * d_model);
    assert!(output.is_device_resident());
}

// ============================================================================
// KV Cache Tests
// ============================================================================

#[test]
fn test_gpu_kv_cache_new_and_methods() {
    use crate::memory::resident::GpuKvCache;
    let ctx = cuda_ctx!();

    let max_seq_len = 64usize;
    let d_model = 32usize;

    // Test GpuKvCache::new
    let mut cache = GpuKvCache::new(&ctx, max_seq_len, d_model).unwrap();

    // Test is_empty and len on new cache
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);

    // Manually set seq_len to simulate token addition
    cache.seq_len = 10;
    assert!(!cache.is_empty());
    assert_eq!(cache.len(), 10);

    // Test reset
    cache.reset();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

// ============================================================================
// Conv Frontend Weights Tests
// ============================================================================

#[test]
fn test_gpu_conv_frontend_weights_structure() {
    use crate::memory::resident::GpuConvFrontendWeights;
    let ctx = cuda_ctx!();

    // Whisper-like configuration: conv1 [384, 80, 3], conv2 [384, 384, 3]
    let in_channels = 80usize;
    let hidden = 384usize;
    let kernel_size = 3usize;

    let weights = GpuConvFrontendWeights {
        conv1_weight: GpuResidentTensor::from_host(
            &ctx,
            &vec![0.01f32; hidden * in_channels * kernel_size],
        )
        .unwrap(),
        conv1_bias: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; hidden]).unwrap(),
        conv2_weight: GpuResidentTensor::from_host(
            &ctx,
            &vec![0.01f32; hidden * hidden * kernel_size],
        )
        .unwrap(),
        conv2_bias: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; hidden]).unwrap(),
    };

    assert!(weights.conv1_weight.is_device_resident());
    assert!(weights.conv1_bias.is_device_resident());
    assert!(weights.conv2_weight.is_device_resident());
    assert!(weights.conv2_bias.is_device_resident());
    assert_eq!(
        weights.conv1_weight.len(),
        hidden * in_channels * kernel_size
    );
    assert_eq!(weights.conv2_weight.len(), hidden * hidden * kernel_size);
}

// ============================================================================
// Decoder Block Weights Tests
// ============================================================================

#[test]
fn test_gpu_decoder_block_weights_structure() {
    use crate::memory::resident::GpuDecoderBlockWeights;
    let ctx = cuda_ctx!();

    let d_model = 32usize;
    let ffn_dim = 128usize;

    // Create decoder block weights (self-attention + cross-attention + FFN)
    let weights = GpuDecoderBlockWeights {
        // Self-attention
        ln1_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln1_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        self_w_q: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        self_b_q: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        self_w_k: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        self_b_k: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        self_w_v: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        self_b_v: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        self_w_o: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        self_b_o: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        // Cross-attention
        ln2_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln2_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        cross_w_q: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        cross_b_q: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        cross_w_k: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        cross_b_k: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        cross_w_v: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        cross_b_v: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        cross_w_o: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        cross_b_o: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        // FFN
        ln3_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln3_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        ffn_up_w: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * ffn_dim]).unwrap(),
        ffn_up_b: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; ffn_dim]).unwrap(),
        ffn_down_w: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; ffn_dim * d_model]).unwrap(),
        ffn_down_b: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
    };

    // Verify all tensors are device resident
    assert!(weights.ln1_gamma.is_device_resident());
    assert!(weights.self_w_q.is_device_resident());
    assert!(weights.cross_w_q.is_device_resident());
    assert!(weights.ffn_up_w.is_device_resident());
}

// ============================================================================
// Forward Encoder Block with Debug Mode
// ============================================================================

#[test]
fn test_forward_encoder_block_with_debug() {
    use crate::memory::resident::{
        clear_kernel_cache, forward_encoder_block_gpu, GpuEncoderBlockWeights, GpuEncoderConfig,
    };
    // Clear cache because kernel modules are tied to CUDA contexts
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // Enable debug mode
    std::env::set_var("WHISPER_DEBUG_GPU_INTERNALS", "1");

    let d_model = 16usize;
    let n_heads = 2u32;
    let ffn_dim = 64usize;
    let seq_len = 2usize;

    let config = GpuEncoderConfig {
        d_model: d_model as u32,
        n_heads,
        ffn_dim: ffn_dim as u32,
    };

    // Create weights
    let weights = GpuEncoderBlockWeights {
        ln1_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln1_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_q: GpuResidentTensor::from_host(&ctx, &vec![0.05f32; d_model * d_model]).unwrap(),
        b_q: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_k: GpuResidentTensor::from_host(&ctx, &vec![0.05f32; d_model * d_model]).unwrap(),
        b_k: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_v: GpuResidentTensor::from_host(&ctx, &vec![0.05f32; d_model * d_model]).unwrap(),
        b_v: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_o: GpuResidentTensor::from_host(&ctx, &vec![0.05f32; d_model * d_model]).unwrap(),
        b_o: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        ln2_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln2_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        ffn_up_w: GpuResidentTensor::from_host(&ctx, &vec![0.05f32; d_model * ffn_dim]).unwrap(),
        ffn_up_b: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; ffn_dim]).unwrap(),
        ffn_down_w: GpuResidentTensor::from_host(&ctx, &vec![0.05f32; ffn_dim * d_model]).unwrap(),
        ffn_down_b: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
    };

    // Input
    let input_data: Vec<f32> = (0..(seq_len * d_model)).map(|i| (i as f32) * 0.1).collect();
    let input = GpuResidentTensor::from_host(&ctx, &input_data).unwrap();

    let output = forward_encoder_block_gpu(&ctx, &input, &weights, &config).unwrap();
    assert_eq!(output.len(), seq_len * d_model);

    // Clean up env var
    std::env::remove_var("WHISPER_DEBUG_GPU_INTERNALS");
}

// ============================================================================
// Config Clone/Debug Tests
// ============================================================================

#[test]
fn test_gpu_encoder_config_clone_and_debug() {
    use crate::memory::resident::GpuEncoderConfig;

    let config = GpuEncoderConfig {
        d_model: 512,
        n_heads: 8,
        ffn_dim: 2048,
    };

    // Test Clone
    let cloned = config;
    assert_eq!(cloned.d_model, 512);
    assert_eq!(cloned.n_heads, 8);
    assert_eq!(cloned.ffn_dim, 2048);

    // Test Debug
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("GpuEncoderConfig"));
    assert!(debug_str.contains("512"));
}

#[test]
fn test_gpu_decoder_config_clone_and_debug() {
    use crate::memory::resident::GpuDecoderConfig;

    let config = GpuDecoderConfig {
        d_model: 768,
        n_heads: 12,
        ffn_dim: 3072,
        max_seq_len: 1024,
        n_layers: 12,
    };

    // Test Clone (Copy trait)
    let cloned = config;
    assert_eq!(cloned.d_model, 768);
    assert_eq!(cloned.n_heads, 12);
    assert_eq!(cloned.ffn_dim, 3072);
    assert_eq!(cloned.max_seq_len, 1024);
    assert_eq!(cloned.n_layers, 12);

    // Test Debug
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("GpuDecoderConfig"));
    assert!(debug_str.contains("768"));
}
