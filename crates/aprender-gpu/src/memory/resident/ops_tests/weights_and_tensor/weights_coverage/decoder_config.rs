//! Decoder Block Weights, Conv Frontend, and Config Copy Trait Tests

use super::*;

// ============================================================================
// Decoder Block Weights All-Fields Validation
// ============================================================================

#[test]
fn test_gpu_decoder_block_weights_all_fields() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::GpuDecoderBlockWeights;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let d_model = 16usize;
    let ffn_dim = 64usize;

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

    // Verify all self-attention weights
    assert!(weights.self_w_q.is_device_resident());
    assert!(weights.self_b_q.is_device_resident());
    assert!(weights.self_w_k.is_device_resident());
    assert!(weights.self_b_k.is_device_resident());
    assert!(weights.self_w_v.is_device_resident());
    assert!(weights.self_b_v.is_device_resident());
    assert!(weights.self_w_o.is_device_resident());
    assert!(weights.self_b_o.is_device_resident());

    // Verify all cross-attention weights
    assert!(weights.cross_w_q.is_device_resident());
    assert!(weights.cross_b_q.is_device_resident());
    assert!(weights.cross_w_k.is_device_resident());
    assert!(weights.cross_b_k.is_device_resident());
    assert!(weights.cross_w_v.is_device_resident());
    assert!(weights.cross_b_v.is_device_resident());
    assert!(weights.cross_w_o.is_device_resident());
    assert!(weights.cross_b_o.is_device_resident());

    // Verify FFN weights
    assert!(weights.ln3_gamma.is_device_resident());
    assert!(weights.ln3_beta.is_device_resident());
}

// ============================================================================
// Conv Frontend Tensor Size Tests
// ============================================================================

#[test]
fn test_gpu_conv_frontend_weights_tensor_sizes() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::GpuConvFrontendWeights;
    clear_kernel_cache();
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

    // Verify tensor sizes match expected dimensions
    assert_eq!(
        weights.conv1_weight.len(),
        hidden * in_channels * kernel_size
    );
    assert_eq!(weights.conv1_bias.len(), hidden);
    assert_eq!(weights.conv2_weight.len(), hidden * hidden * kernel_size);
    assert_eq!(weights.conv2_bias.len(), hidden);
}

// ============================================================================
// Config Copy Trait Tests
// ============================================================================

#[test]
fn test_gpu_encoder_config_copy_trait() {
    use crate::memory::resident::GpuEncoderConfig;

    let config1 = GpuEncoderConfig {
        d_model: 256,
        n_heads: 4,
        ffn_dim: 1024,
    };

    // Test Copy trait - should work without move
    let config2 = config1;
    let config3 = config1;

    assert_eq!(config2.d_model, config3.d_model);
    assert_eq!(config2.n_heads, config3.n_heads);
    assert_eq!(config2.ffn_dim, config3.ffn_dim);

    // Original is still usable due to Copy
    assert_eq!(config1.d_model, 256);
}

#[test]
fn test_gpu_decoder_config_copy_trait() {
    use crate::memory::resident::GpuDecoderConfig;

    let config1 = GpuDecoderConfig {
        d_model: 512,
        n_heads: 8,
        ffn_dim: 2048,
        max_seq_len: 1024,
        n_layers: 6,
    };

    // Test Copy trait - should work without move
    let config2 = config1;
    let config3 = config1;

    assert_eq!(config2.d_model, config3.d_model);
    assert_eq!(config2.n_heads, config3.n_heads);
    assert_eq!(config2.ffn_dim, config3.ffn_dim);
    assert_eq!(config2.max_seq_len, config3.max_seq_len);
    assert_eq!(config2.n_layers, config3.n_layers);

    // Original is still usable due to Copy
    assert_eq!(config1.n_layers, 6);
}

// ============================================================================
// Decoder Block Weights Field Size Verification
// ============================================================================

#[test]
fn test_gpu_decoder_block_weights_field_sizes() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::GpuDecoderBlockWeights;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let d_model = 16usize;
    let ffn_dim = 32usize;

    let weights = GpuDecoderBlockWeights {
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
        ln3_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln3_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        ffn_up_w: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * ffn_dim]).unwrap(),
        ffn_up_b: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; ffn_dim]).unwrap(),
        ffn_down_w: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; ffn_dim * d_model]).unwrap(),
        ffn_down_b: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
    };

    // Verify self-attention field sizes
    assert_eq!(weights.self_w_q.len(), d_model * d_model);
    assert_eq!(weights.self_b_q.len(), d_model);
    assert_eq!(weights.self_w_k.len(), d_model * d_model);
    assert_eq!(weights.self_b_k.len(), d_model);
    assert_eq!(weights.self_w_v.len(), d_model * d_model);
    assert_eq!(weights.self_b_v.len(), d_model);
    assert_eq!(weights.self_w_o.len(), d_model * d_model);
    assert_eq!(weights.self_b_o.len(), d_model);

    // Verify cross-attention field sizes
    assert_eq!(weights.cross_w_q.len(), d_model * d_model);
    assert_eq!(weights.cross_b_q.len(), d_model);
    assert_eq!(weights.cross_w_k.len(), d_model * d_model);
    assert_eq!(weights.cross_b_k.len(), d_model);
    assert_eq!(weights.cross_w_v.len(), d_model * d_model);
    assert_eq!(weights.cross_b_v.len(), d_model);
    assert_eq!(weights.cross_w_o.len(), d_model * d_model);
    assert_eq!(weights.cross_b_o.len(), d_model);

    // Verify FFN field sizes
    assert_eq!(weights.ln3_gamma.len(), d_model);
    assert_eq!(weights.ln3_beta.len(), d_model);
    assert_eq!(weights.ffn_up_w.len(), d_model * ffn_dim);
    assert_eq!(weights.ffn_up_b.len(), ffn_dim);
    assert_eq!(weights.ffn_down_w.len(), ffn_dim * d_model);
    assert_eq!(weights.ffn_down_b.len(), d_model);
}
