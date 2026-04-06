//! Q4_0 GPU Integration Tests (PMAT-803)
//!
//! Tests for `AprQ4ToGpuAdapter` and `GpuModelQ4` to drive coverage
//! of the Q4_0 GPU inference path.
//!
//! # Coverage Targets
//!
//! - `src/gpu/adapters/apr_q4.rs` - Q4 adapter (~500 lines)
//! - `src/cuda/executor/q_basic.rs` - Q4_0 GEMV kernel
//! - `src/apr_transformer/q4_simd.rs` - Quantized tensor structures

#![allow(
    unused_imports,
    unused_variables,
    clippy::needless_range_loop,
    clippy::manual_repeat_n
)]

use realizar::apr_transformer::{
    AprTransformerConfig, QuantizedAprLayerQ4, QuantizedAprTensorQ4, QuantizedAprTransformerQ4,
};
#[cfg(feature = "cuda")]
use realizar::gpu::adapters::{AprQ4ToGpuAdapter, GpuModelQ4, LayerNorms};

// ============================================================================
// Model Creation Tests
// ============================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_q4_adapter_create_model_basic() {
    let apr = create_minimal_q4_transformer(1);
    let model = AprQ4ToGpuAdapter::create_model(&apr);

    assert_eq!(model.num_layers, 1);
    assert_eq!(model.config.hidden_dim, 64);
    assert_eq!(model.config.vocab_size, 100);
    assert_eq!(model.token_embedding.len(), 100 * 64);
    assert_eq!(model.output_norm_weight.len(), 64);
    assert_eq!(model.layer_norms.len(), 1);
}

#[test]
#[cfg(feature = "cuda")]
fn test_q4_adapter_create_model_with_gate() {
    let apr = create_q4_transformer_with_gate(2);
    let model = AprQ4ToGpuAdapter::create_model(&apr);

    assert!(model.has_gate, "Model should detect SwiGLU gate");
    assert_eq!(model.num_layers, 2);
}

#[test]
#[cfg(feature = "cuda")]
fn test_q4_adapter_create_model_without_gate() {
    let apr = create_minimal_q4_transformer(2);
    let model = AprQ4ToGpuAdapter::create_model(&apr);

    assert!(
        !model.has_gate,
        "Model should not have gate when not provided"
    );
    assert_eq!(model.num_layers, 2);
}

#[test]
#[cfg(feature = "cuda")]
fn test_q4_adapter_layer_norms_extraction() {
    let hidden_dim = 128;
    let mut apr = create_minimal_q4_transformer(3);

    // Set specific norm weights for verification
    for (i, layer) in apr.layers.iter_mut().enumerate() {
        layer.attn_norm_weight = vec![i as f32 + 0.1; hidden_dim];
        layer.ffn_norm_weight = Some(vec![i as f32 + 0.2; hidden_dim]);
    }

    // Update config to match
    apr.config.hidden_dim = hidden_dim;
    apr.token_embedding = vec![0.0; 100 * hidden_dim];
    apr.output_norm_weight = vec![1.0; hidden_dim];

    let model = AprQ4ToGpuAdapter::create_model(&apr);

    assert_eq!(model.layer_norms.len(), 3);

    // Verify layer norm values
    for (i, norms) in model.layer_norms.iter().enumerate() {
        assert!((norms.attn_norm[0] - (i as f32 + 0.1)).abs() < 1e-6);
        assert!((norms.ffn_norm[0] - (i as f32 + 0.2)).abs() < 1e-6);
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_q4_adapter_missing_ffn_norm() {
    let hidden_dim = 64;
    let mut apr = create_minimal_q4_transformer(1);

    // Remove FFN norm (should default to identity)
    apr.layers[0].ffn_norm_weight = None;
    apr.config.hidden_dim = hidden_dim;
    apr.token_embedding = vec![0.0; 100 * hidden_dim];
    apr.output_norm_weight = vec![1.0; hidden_dim];

    let model = AprQ4ToGpuAdapter::create_model(&apr);

    // FFN norm should default to all 1.0s (identity)
    assert_eq!(model.layer_norms[0].ffn_norm.len(), hidden_dim);
    assert!(model.layer_norms[0]
        .ffn_norm
        .iter()
        .all(|&x| (x - 1.0).abs() < 1e-6));
}

#[test]
#[cfg(feature = "cuda")]
fn test_q4_adapter_large_model() {
    // Test with larger dimensions closer to real models
    let hidden_dim = 512;
    let vocab_size = 1000;
    let intermediate_dim = 2048;
    let num_layers = 8;

    let layers: Vec<QuantizedAprLayerQ4> = (0..num_layers)
        .map(|_| QuantizedAprLayerQ4 {
            attn_norm_weight: vec![1.0; hidden_dim],
            qkv_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim * 3),
            attn_output_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim),
            ffn_up_weight: QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim),
            ffn_down_weight: QuantizedAprTensorQ4::zeros(intermediate_dim, hidden_dim),
            ffn_gate_weight: Some(QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim)),
            ffn_norm_weight: Some(vec![1.0; hidden_dim]),
        })
        .collect();

    let apr = QuantizedAprTransformerQ4 {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers,
            num_heads: 16,
            num_kv_heads: 4, // GQA
            vocab_size,
            intermediate_dim,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
            ..Default::default()
        },
        token_embedding: vec![0.0; vocab_size * hidden_dim],
        layers,
        output_norm_weight: vec![1.0; hidden_dim],
        lm_head_weight: QuantizedAprTensorQ4::zeros(hidden_dim, vocab_size),
    };

    let model = AprQ4ToGpuAdapter::create_model(&apr);

    assert_eq!(model.num_layers, num_layers);
    assert!(model.has_gate);
    assert_eq!(model.layer_norms.len(), num_layers);
    assert_eq!(model.config.num_heads, 16);
    assert_eq!(model.config.num_kv_heads, 4);
}

// ============================================================================
// LayerNorms Struct Tests
// ============================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_layer_norms_struct() {
    let hidden_dim = 64;
    let norms = LayerNorms {
        attn_norm: vec![0.5; hidden_dim],
        ffn_norm: vec![1.5; hidden_dim],
    };

    assert_eq!(norms.attn_norm.len(), hidden_dim);
    assert_eq!(norms.ffn_norm.len(), hidden_dim);
    assert!((norms.attn_norm[0] - 0.5).abs() < 1e-6);
    assert!((norms.ffn_norm[0] - 1.5).abs() < 1e-6);
}

// ============================================================================
// GpuModelQ4 Field Access Tests
// ============================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_gpu_model_q4_config_access() {
    let apr = create_minimal_q4_transformer(2);
    let model = AprQ4ToGpuAdapter::create_model(&apr);

    // Test config field access
    assert_eq!(model.config.architecture, "test");
    assert_eq!(model.config.hidden_dim, 64);
    assert_eq!(model.config.num_layers, 2);
    assert_eq!(model.config.num_heads, 4);
    assert_eq!(model.config.num_kv_heads, 4);
    assert_eq!(model.config.vocab_size, 100);
    assert_eq!(model.config.intermediate_dim, 128);
    assert_eq!(model.config.context_length, 128);
    assert!((model.config.rope_theta - 10000.0).abs() < 1e-6);
    assert!((model.config.eps - 1e-5).abs() < 1e-10);
}

#[test]
#[cfg(feature = "cuda")]
fn test_gpu_model_q4_embedding_dimensions() {
    let apr = create_minimal_q4_transformer(1);
    let model = AprQ4ToGpuAdapter::create_model(&apr);

    let vocab_size = model.config.vocab_size;
    let hidden_dim = model.config.hidden_dim;

    // Token embedding should be [vocab_size, hidden_dim] flattened
    assert_eq!(model.token_embedding.len(), vocab_size * hidden_dim);
}

#[test]
#[cfg(feature = "cuda")]
fn test_gpu_model_q4_output_norm_dimensions() {
    let apr = create_minimal_q4_transformer(1);
    let model = AprQ4ToGpuAdapter::create_model(&apr);

    // Output norm should match hidden_dim
    assert_eq!(model.output_norm_weight.len(), model.config.hidden_dim);
}

// ============================================================================
// CUDA Integration Tests (require GPU)
// ============================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_q4_adapter_upload_weights_minimal() {
    use realizar::cuda::CudaExecutor;

    let mut executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {e:?}");
            return;
        },
    };

    let apr = create_minimal_q4_transformer(1);

    let bytes = AprQ4ToGpuAdapter::upload_weights(&apr, &mut executor)
        .expect("Weight upload should succeed");

    assert!(bytes > 0, "Should upload some bytes");

    // Verify weights are cached
    assert!(executor.has_quantized_weights("layer_0.attn.qkv"));
    assert!(executor.has_quantized_weights("layer_0.attn.out"));
    assert!(executor.has_quantized_weights("layer_0.ffn.up"));
    assert!(executor.has_quantized_weights("layer_0.ffn.down"));
    assert!(executor.has_quantized_weights("lm_head"));
}

#[test]
#[cfg(feature = "cuda")]
fn test_q4_adapter_upload_weights_with_gate() {
    use realizar::cuda::CudaExecutor;

    let mut executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {e:?}");
            return;
        },
    };

    let apr = create_q4_transformer_with_gate(2);

    let bytes = AprQ4ToGpuAdapter::upload_weights(&apr, &mut executor)
        .expect("Weight upload should succeed");

    assert!(bytes > 0);

    // Verify gate weights are cached for SwiGLU models
    assert!(executor.has_quantized_weights("layer_0.ffn.gate"));
    assert!(executor.has_quantized_weights("layer_1.ffn.gate"));
}

#[test]
#[cfg(feature = "cuda")]
fn test_q4_adapter_upload_multi_layer() {
    use realizar::cuda::CudaExecutor;

    let mut executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {e:?}");
            return;
        },
    };

    let num_layers = 4;
    let apr = create_minimal_q4_transformer(num_layers);

    let _bytes = AprQ4ToGpuAdapter::upload_weights(&apr, &mut executor)
        .expect("Weight upload should succeed");

    // Verify all layer weights are cached
    for i in 0..num_layers {
        assert!(executor.has_quantized_weights(&format!("layer_{i}.attn.qkv")));
        assert!(executor.has_quantized_weights(&format!("layer_{i}.attn.out")));
        assert!(executor.has_quantized_weights(&format!("layer_{i}.ffn.up")));
        assert!(executor.has_quantized_weights(&format!("layer_{i}.ffn.down")));
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_q4_adapter_weight_byte_count() {
    use realizar::cuda::CudaExecutor;

    let mut executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {e:?}");
            return;
        },
    };

    let apr = create_minimal_q4_transformer(1);
    let bytes = AprQ4ToGpuAdapter::upload_weights(&apr, &mut executor)
        .expect("Weight upload should succeed");

    // Q4_0: 18 bytes per 32 elements
    // For hidden_dim=64, intermediate_dim=128, vocab_size=100:
    // - QKV: 64 * 192 = 12288 elements = 384 blocks * 18 = 6912 bytes
    // - Out: 64 * 64 = 4096 elements = 128 blocks * 18 = 2304 bytes
    // - Up: 64 * 128 = 8192 elements = 256 blocks * 18 = 4608 bytes
    // - Down: 128 * 64 = 8192 elements = 256 blocks * 18 = 4608 bytes
    // - LM Head: 64 * 100 = 6400 elements = 200 blocks * 18 = 3600 bytes
    // Total per layer: ~18432 bytes + lm_head ~3600 = ~22032

    // Just verify we uploaded a reasonable amount
    assert!(bytes > 10000, "Should upload at least 10KB, got {bytes}");
    assert!(
        bytes < 100000,
        "Should not upload more than 100KB, got {bytes}"
    );
}

// ============================================================================
// Helper Functions
// ============================================================================

#[allow(dead_code)]
fn create_minimal_q4_transformer(num_layers: usize) -> QuantizedAprTransformerQ4 {
    let hidden_dim = 64;
    let vocab_size = 100;
    let intermediate_dim = 128;

    let layers: Vec<QuantizedAprLayerQ4> = (0..num_layers)
        .map(|_| QuantizedAprLayerQ4 {
            attn_norm_weight: vec![1.0; hidden_dim],
            qkv_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim * 3),
            attn_output_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim),
            ffn_up_weight: QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim),
            ffn_down_weight: QuantizedAprTensorQ4::zeros(intermediate_dim, hidden_dim),
            ffn_gate_weight: None, // No gate = standard FFN
            ffn_norm_weight: Some(vec![1.0; hidden_dim]),
        })
        .collect();

    QuantizedAprTransformerQ4 {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size,
            intermediate_dim,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-5,
            ..Default::default()
        },
        token_embedding: vec![0.0; vocab_size * hidden_dim],
        layers,
        output_norm_weight: vec![1.0; hidden_dim],
        lm_head_weight: QuantizedAprTensorQ4::zeros(hidden_dim, vocab_size),
    }
}

#[allow(dead_code)]
fn create_q4_transformer_with_gate(num_layers: usize) -> QuantizedAprTransformerQ4 {
    let hidden_dim = 64;
    let vocab_size = 100;
    let intermediate_dim = 128;

    let layers: Vec<QuantizedAprLayerQ4> = (0..num_layers)
        .map(|_| QuantizedAprLayerQ4 {
            attn_norm_weight: vec![1.0; hidden_dim],
            qkv_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim * 3),
            attn_output_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim),
            ffn_up_weight: QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim),
            ffn_down_weight: QuantizedAprTensorQ4::zeros(intermediate_dim, hidden_dim),
            ffn_gate_weight: Some(QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim)), // With gate = SwiGLU
            ffn_norm_weight: Some(vec![1.0; hidden_dim]),
        })
        .collect();

    QuantizedAprTransformerQ4 {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size,
            intermediate_dim,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-5,
            ..Default::default()
        },
        token_embedding: vec![0.0; vocab_size * hidden_dim],
        layers,
        output_norm_weight: vec![1.0; hidden_dim],
        lm_head_weight: QuantizedAprTensorQ4::zeros(hidden_dim, vocab_size),
    }
}

// ============================================================================
// Q4_0 Correctness Verification Tests (Phase 12 - Fix Corruption)
// ============================================================================

/// Create a Q4_0 block with known scale and nibble pattern.
///
/// Q4_0 format: 18 bytes per 32 elements
/// - bytes 0-1: f16 scale (d)
/// - bytes 2-17: 16 packed bytes (32 nibbles)
///
/// Dequantization: value[i] = d * (nibble[i] - 8)
#[cfg(feature = "cuda")]
fn create_q4_block(scale: f32, nibbles: &[u8; 32]) -> [u8; 18] {
    let mut block = [0u8; 18];

    // Pack scale as f16 (little-endian)
    let scale_f16 = half::f16::from_f32(scale);
    let scale_bytes = scale_f16.to_le_bytes();
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];

    // Pack nibbles: nibble[2i] = low nibble, nibble[2i+1] = high nibble
    for i in 0..16 {
        let low = nibbles[i * 2] & 0x0F;
        let high = nibbles[i * 2 + 1] & 0x0F;
        block[2 + i] = low | (high << 4);
    }

    block
}

/// Create Q4_0 weight matrix with identity-like pattern (row-major layout).
///
/// Each row has ceil(k/32) blocks, stored contiguously.
/// For simplicity, uses scale=1.0 and nibble=9 (+1 after centering) on diagonal,
/// nibble=8 (0 after centering) elsewhere.
#[cfg(feature = "cuda")]
fn create_q4_identity_weights(n: usize, k: usize) -> Vec<u8> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18;

    let num_blocks_per_row = k.div_ceil(BLOCK_SIZE);
    let total_bytes = n * num_blocks_per_row * BLOCK_BYTES;
    let mut data = vec![0u8; total_bytes];

    for row in 0..n {
        for blk in 0..num_blocks_per_row {
            let block_offset = row * num_blocks_per_row * BLOCK_BYTES + blk * BLOCK_BYTES;

            // Create nibble pattern: 9 on diagonal (becomes +1), 8 elsewhere (becomes 0)
            let mut nibbles = [8u8; 32]; // All zeros after centering

            // Check if diagonal falls in this block
            let col_start = blk * BLOCK_SIZE;
            let col_end = (col_start + BLOCK_SIZE).min(k);

            if row >= col_start && row < col_end {
                let nibble_idx = row - col_start;
                nibbles[nibble_idx] = 9; // +1 after centering
            }

            let block = create_q4_block(1.0, &nibbles);
            data[block_offset..block_offset + BLOCK_BYTES].copy_from_slice(&block);
        }
    }

    data
}

/// CPU reference implementation of Q4_0 GEMV.
///
/// y[i] = sum_j(W[i,j] * x[j])
/// where W is stored in row-major Q4_0 format.
#[cfg(feature = "cuda")]
fn cpu_q4_gemv(weights: &[u8], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18;

    let num_blocks_per_row = k.div_ceil(BLOCK_SIZE);
    let mut y = vec![0.0f32; n];

    for row in 0..n {
        let row_base = row * num_blocks_per_row * BLOCK_BYTES;
        let mut acc = 0.0f32;

        for blk in 0..num_blocks_per_row {
            let block_offset = row_base + blk * BLOCK_BYTES;

            // Read scale (f16)
            let scale_bytes = [weights[block_offset], weights[block_offset + 1]];
            let scale = half::f16::from_le_bytes(scale_bytes).to_f32();

            // Process 32 nibbles
            for i in 0..32 {
                let col = blk * BLOCK_SIZE + i;
                if col >= k {
                    break;
                }

                // Extract nibble
                let byte_idx = block_offset + 2 + i / 2;
                let byte = weights[byte_idx];
                let nibble = if i % 2 == 0 {
                    byte & 0x0F
                } else {
                    (byte >> 4) & 0x0F
                };

                // Dequantize: value = scale * (nibble - 8)
                let dequant = scale * (nibble as f32 - 8.0);
                acc += dequant * x[col];
            }
        }

        y[row] = acc;
    }

    y
}

#[test]
#[cfg(feature = "cuda")]
fn test_q4_gemv_correctness_identity() {
    use realizar::cuda::CudaExecutor;
    use trueno_gpu::driver::GpuBuffer;

    let mut executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {e:?}");
            return;
        },
    };

    // Use dimensions that are multiples of 32 to avoid block boundary edge cases
    let n = 64; // output dimension
    let k = 64; // input dimension

    // Create identity-like weights (diagonal = +1, off-diagonal = 0)
    let weights = create_q4_identity_weights(n, k);

    // Create input vector [1, 2, 3, ..., k]
    let x: Vec<f32> = (1..=k).map(|i| i as f32).collect();

    // Expected output: y[i] = x[i] for identity matrix
    let expected: Vec<f32> = (1..=n).map(|i| i as f32).collect();

    // CPU reference
    let cpu_result = cpu_q4_gemv(&weights, &x, n, k);

    // Verify CPU matches expected
    for i in 0..n {
        assert!(
            (cpu_result[i] - expected[i]).abs() < 1e-3,
            "CPU mismatch at {}: expected {}, got {}",
            i,
            expected[i],
            cpu_result[i]
        );
    }

    // Upload weights to GPU
    let weight_ptr = executor
        .load_quantized_weights_with_type("test_identity", &weights, 2) // Q4_0 type = 2
        .expect("Weight upload should succeed");

    // Create GPU buffers
    let x_gpu = GpuBuffer::from_host(executor.context(), &x).expect("Failed to upload x");
    let y_gpu = GpuBuffer::new(executor.context(), n).expect("Failed to allocate y");

    // Get weight pointer
    let w_ptr = executor
        .get_quantized_weight_ptr("test_identity")
        .expect("Weight should be cached");

    // Execute GPU GEMV
    executor
        .q4_0_gemv_into(w_ptr, &x_gpu, &y_gpu, n as u32, k as u32)
        .expect("GEMV should succeed");

    // Sync and download
    executor.synchronize().expect("Sync should succeed");
    let mut gpu_result = vec![0.0f32; n];
    y_gpu
        .copy_to_host(&mut gpu_result)
        .expect("Download should succeed");

    // Compare GPU vs CPU
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  TUI SIMULATION: Q4_0 GEMV Correctness Test                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!("\nDimensions: n={n}, k={k}");
    println!("First 10 elements comparison:");
    println!(
        "{:>5} {:>12} {:>12} {:>12} {:>12}",
        "idx", "expected", "cpu", "gpu", "error"
    );
    println!("{}", "-".repeat(60));

    let mut max_error = 0.0f32;
    for i in 0..10.min(n) {
        let error = (gpu_result[i] - expected[i]).abs();
        max_error = max_error.max(error);
        let status = if error < 1e-2 { "✓" } else { "✗" };
        println!(
            "{:>5} {:>12.4} {:>12.4} {:>12.4} {:>12.4} {}",
            i, expected[i], cpu_result[i], gpu_result[i], error, status
        );
    }

    println!("\nMax error: {max_error:.6}");

    // Assert correctness with tolerance for quantization
    for i in 0..n {
        assert!(
            (gpu_result[i] - expected[i]).abs() < 0.1, // Allow some tolerance for quantization
            "GPU mismatch at {}: expected {:.4}, got {:.4} (CPU ref: {:.4})",
            i,
            expected[i],
            gpu_result[i],
            cpu_result[i]
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_q4_gemv_correctness_known_pattern() {
    use realizar::cuda::CudaExecutor;
    use trueno_gpu::driver::GpuBuffer;

    let mut executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {e:?}");
            return;
        },
    };

    // Small test: 2 rows, 32 columns (1 block per row)
    let n = 2;
    let k = 32;

    // Create weights with known pattern:
    // Row 0: scale=1.0, all nibbles=9 -> all values = +1
    // Row 1: scale=2.0, all nibbles=12 -> all values = +8
    let mut weights = vec![0u8; 2 * 18]; // 2 rows * 18 bytes per block

    // Row 0: scale=1.0 (f16: 0x3C00), nibbles all 9
    let scale1 = half::f16::from_f32(1.0).to_le_bytes();
    weights[0] = scale1[0];
    weights[1] = scale1[1];
    for i in 0..16 {
        weights[2 + i] = 0x99; // Both nibbles = 9
    }

    // Row 1: scale=2.0 (f16: 0x4000), nibbles all 12
    let scale2 = half::f16::from_f32(2.0).to_le_bytes();
    weights[18] = scale2[0];
    weights[19] = scale2[1];
    for i in 0..16 {
        weights[20 + i] = 0xCC; // Both nibbles = 12
    }

    // Input: all 1s
    let x = vec![1.0f32; k];

    // Expected output:
    // y[0] = sum of 32 * (1.0 * (9 - 8)) = 32 * 1.0 = 32.0
    // y[1] = sum of 32 * (2.0 * (12 - 8)) = 32 * 8.0 = 256.0
    let expected = vec![32.0f32, 256.0f32];

    // CPU reference
    let cpu_result = cpu_q4_gemv(&weights, &x, n, k);

    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  TUI SIMULATION: Q4_0 GEMV Known Pattern Test                        ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    println!("\nCPU Reference:");
    println!("  Row 0: expected=32.0, got={:.4}", cpu_result[0]);
    println!("  Row 1: expected=256.0, got={:.4}", cpu_result[1]);

    // Verify CPU
    assert!((cpu_result[0] - 32.0).abs() < 1e-3, "CPU row 0 wrong");
    assert!((cpu_result[1] - 256.0).abs() < 1e-3, "CPU row 1 wrong");

    // Upload weights to GPU
    let _weight_size = executor
        .load_quantized_weights_with_type("test_pattern", &weights, 2)
        .expect("Weight upload should succeed");

    let x_gpu = GpuBuffer::from_host(executor.context(), &x).expect("Failed to upload x");
    let y_gpu = GpuBuffer::new(executor.context(), n).expect("Failed to allocate y");

    let w_ptr = executor
        .get_quantized_weight_ptr("test_pattern")
        .expect("Weight should be cached");

    // Execute GPU GEMV
    executor
        .q4_0_gemv_into(w_ptr, &x_gpu, &y_gpu, n as u32, k as u32)
        .expect("GEMV should succeed");

    executor.synchronize().expect("Sync should succeed");
    let mut gpu_result = vec![0.0f32; n];
    y_gpu
        .copy_to_host(&mut gpu_result)
        .expect("Download should succeed");

    println!("\nGPU Results:");
    println!("  Row 0: expected=32.0, got={:.4}", gpu_result[0]);
    println!("  Row 1: expected=256.0, got={:.4}", gpu_result[1]);

    // Check for common failure patterns
    if (gpu_result[0] - 0.0).abs() < 1e-3 && (gpu_result[1] - 0.0).abs() < 1e-3 {
        println!("\n❌ FAILURE PATTERN: All zeros - kernel may not be writing output");
    } else if (gpu_result[0] - gpu_result[1]).abs() < 1e-3 {
        println!("\n❌ FAILURE PATTERN: Same output for different rows - indexing bug");
    } else if gpu_result[0] > 100.0 || gpu_result[1] > 1000.0 {
        println!("\n❌ FAILURE PATTERN: Values too large - accumulator not cleared");
    }

    // Assert correctness
    assert!(
        (gpu_result[0] - expected[0]).abs() < 1.0,
        "GPU row 0: expected {}, got {}",
        expected[0],
        gpu_result[0]
    );
    assert!(
        (gpu_result[1] - expected[1]).abs() < 1.0,
        "GPU row 1: expected {}, got {}",
        expected[1],
        gpu_result[1]
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_q4_forward_pass_minimal() {
    use realizar::cuda::CudaExecutor;

    let mut executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed: {e:?}");
            return;
        },
    };

    // Create minimal transformer with known embeddings
    let hidden_dim = 32; // Multiple of block size
    let vocab_size = 32; // Multiple of block size
    let intermediate_dim = 64;

    // Create embeddings: token i has embedding [i, i, i, ...]
    let token_embedding: Vec<f32> = (0..vocab_size)
        .flat_map(|tok| std::iter::repeat(tok as f32 * 0.1).take(hidden_dim))
        .collect();

    // Create layer with identity-like weights (will output same as input)
    let qkv_weights = create_q4_identity_weights(hidden_dim * 3, hidden_dim);
    let out_weights = create_q4_identity_weights(hidden_dim, hidden_dim);
    let up_weights = create_q4_identity_weights(intermediate_dim, hidden_dim);
    let down_weights = create_q4_identity_weights(hidden_dim, intermediate_dim);

    let apr = QuantizedAprTransformerQ4 {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size,
            intermediate_dim,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-5,
            ..Default::default()
        },
        token_embedding,
        layers: vec![QuantizedAprLayerQ4 {
            attn_norm_weight: vec![1.0; hidden_dim],
            qkv_weight: QuantizedAprTensorQ4::new(qkv_weights, hidden_dim, hidden_dim * 3),
            attn_output_weight: QuantizedAprTensorQ4::new(out_weights, hidden_dim, hidden_dim),
            ffn_up_weight: QuantizedAprTensorQ4::new(up_weights, hidden_dim, intermediate_dim),
            ffn_down_weight: QuantizedAprTensorQ4::new(down_weights, intermediate_dim, hidden_dim),
            ffn_gate_weight: None,
            ffn_norm_weight: Some(vec![1.0; hidden_dim]),
        }],
        output_norm_weight: vec![1.0; hidden_dim],
        lm_head_weight: QuantizedAprTensorQ4::new(
            create_q4_identity_weights(vocab_size, hidden_dim),
            hidden_dim,
            vocab_size,
        ),
    };

    // Upload weights
    let bytes_uploaded = AprQ4ToGpuAdapter::upload_weights(&apr, &mut executor)
        .expect("Weight upload should succeed");

    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  TUI SIMULATION: Q4_0 Forward Pass Test                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!("\nModel config:");
    println!("  hidden_dim: {}", hidden_dim);
    println!("  vocab_size: {}", vocab_size);
    println!("  num_layers: 1");
    println!("  bytes_uploaded: {}", bytes_uploaded);

    // Create GPU model
    let model = AprQ4ToGpuAdapter::create_model(&apr);

    // Run forward pass with token 10
    let token_ids = vec![10usize];

    let result = model.forward(&mut executor, &token_ids);

    match result {
        Ok(logits) => {
            println!("\nForward pass succeeded!");
            println!("  Output logits length: {}", logits.len());

            let argmax = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            println!("  Argmax token: {}", argmax);
            println!("  First 10 logits:");
            for (i, &l) in logits.iter().take(10).enumerate() {
                println!("    logit[{}] = {:.4}", i, l);
            }

            // Check for corruption patterns
            let all_same = logits.iter().all(|&x| (x - logits[0]).abs() < 1e-6);
            let all_nan = logits.iter().any(|x| x.is_nan());
            let all_inf = logits.iter().any(|x| x.is_infinite());

            if all_nan {
                println!("\n❌ CORRUPTION: NaN values detected");
            } else if all_inf {
                println!("\n❌ CORRUPTION: Infinite values detected");
            } else if all_same {
                println!("\n❌ CORRUPTION: All logits identical (no differentiation)");
            } else {
                println!("\n✓ Basic sanity checks passed");
            }

            assert_eq!(
                logits.len(),
                vocab_size,
                "Output should have vocab_size logits"
            );
            assert!(!all_nan, "Logits should not be NaN");
            assert!(!all_inf, "Logits should not be infinite");
        },
        Err(e) => {
            println!("\n❌ Forward pass failed: {:?}", e);
            panic!("Forward pass should succeed");
        },
    }
}
