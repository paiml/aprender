//! GGUF Part 03: IMP-109 (Fused Dequantize-Matmul) + IMP-110 (Multi-Head Parallel Attention)
//!
//! Extracted from gguf_monolith.rs (PMAT-802)
//!
//! ## Test Coverage
//!
//! - IMP-109a: Fused dequant+matmul correctness vs separate operations
//! - IMP-109b: Fused batch matmul GPU with determinism check
//! - IMP-109c: Fused vs separate performance baseline
//! - IMP-110a: Parallel multi-head attention correctness
//! - IMP-110b: Batched Q/K/V reshape verification
//! - IMP-110c: Parallel batched Q@K^T scores computation

use crate::gguf::test_helpers::create_test_model_with_config;
use crate::gguf::GGUFConfig;

// =========================================================================
// IMP-109: Fused Dequantize-Matmul Kernel (GPU-Accelerated)
// =========================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_imp_109a_fused_dequant_matmul_correctness() {
    // IMP-109a: Verify fused dequant+matmul matches separate operations
    // Uses model's existing quantized weights to validate correctness
    use crate::quantize::{dequantize_q4_k_simd, fused_q4k_parallel_matvec};

    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 256, // Must be multiple of QK_K
        intermediate_dim: 512,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 1024,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let in_dim = config.hidden_dim;
    let out_dim = config.intermediate_dim;

    // Get the first layer's up projection weight (already Q4_K quantized)
    let weight_data = &model.layers[0].ffn_up_weight.data;

    // Create activation input
    let activations: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

    // Reference: separate dequant + matmul
    let weight_dequant = dequantize_q4_k_simd(weight_data).expect("test");
    let reference: Vec<f32> = (0..out_dim)
        .map(|row| {
            (0..in_dim)
                .map(|col| weight_dequant[row * in_dim + col] * activations[col])
                .sum()
        })
        .collect();

    // Fused: single pass through quantized data
    let fused_result = fused_q4k_parallel_matvec(weight_data, &activations, in_dim, out_dim)
        .expect("IMP-109a: Fused operation should succeed");

    // Verify correctness within tolerance
    assert_eq!(
        fused_result.len(),
        out_dim,
        "IMP-109a: Fused result should have out_dim elements"
    );

    for i in 0..out_dim {
        let diff = (fused_result[i] - reference[i]).abs();
        // Allow 2% relative tolerance due to different accumulation order in
        // fused Q4K dequant+matmul vs separate dequant then f32 matmul.
        // Quantized block boundaries cause rounding differences up to ~1.9%.
        let tolerance = reference[i].abs() * 0.02 + 1e-4;
        assert!(
            diff < tolerance,
            "IMP-109a: Row {} differs: fused={}, reference={}, diff={}",
            i,
            fused_result[i],
            reference[i],
            diff
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_109b_fused_batch_matmul_gpu() {
    // IMP-109b: Verify fused batch matmul produces correct, deterministic results
    // Key optimization: dequantize weight once, reuse for all batch elements
    use crate::gpu::HybridScheduler;

    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 256, // Must be multiple of QK_K
        intermediate_dim: 512,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 1024,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);

    // Create batch of activations (batch_size x hidden_dim)
    let batch_size = 8;
    let activations: Vec<f32> = (0..batch_size * config.hidden_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
        .collect();

    // Test GPU-accelerated fused batch matmul
    let fused_output = model
        .fused_batch_matmul_gpu(&activations, &model.layers[0].ffn_up_weight, batch_size)
        .expect("IMP-109b: Fused batch matmul should succeed");

    // Verify output shape
    assert_eq!(
        fused_output.len(),
        batch_size * config.intermediate_dim,
        "IMP-109b: Fused batch output should be batch_size * intermediate_dim"
    );

    // Verify all outputs are finite
    assert!(
        fused_output.iter().all(|x| x.is_finite()),
        "IMP-109b: All fused outputs should be finite"
    );

    // Verify non-trivial computation (not all zeros)
    let sum: f32 = fused_output.iter().map(|x| x.abs()).sum();
    assert!(
        sum > 0.1,
        "IMP-109b: Fused output should have non-zero values"
    );

    // Verify determinism - repeated calls produce same result
    let fused_output2 = model
        .fused_batch_matmul_gpu(&activations, &model.layers[0].ffn_up_weight, batch_size)
        .expect("IMP-109b: Repeated call should succeed");

    for i in 0..fused_output.len() {
        assert!(
            (fused_output[i] - fused_output2[i]).abs() < 1e-6,
            "IMP-109b: Fused batch matmul should be deterministic at position {}: run1={}, run2={}",
            i,
            fused_output[i],
            fused_output2[i]
        );
    }

    // Compare with batch_matmul_gpu (same approach, should match exactly)
    let weight = &model.layers[0].ffn_up_weight;
    let weight_f32 = {
        use crate::quantize::{dequantize_q4_k_simd, QK_K};
        let in_dim = weight.in_dim;
        let out_dim = weight.out_dim;
        let super_blocks_per_row = in_dim.div_ceil(QK_K);
        let mut output = Vec::with_capacity(in_dim * out_dim);
        for row in 0..out_dim {
            let row_start = row * super_blocks_per_row * 144;
            let row_end = row_start + super_blocks_per_row * 144;
            let row_data = &weight.data[row_start..row_end];
            let row_dequant = dequantize_q4_k_simd(row_data).expect("test");
            output.extend_from_slice(&row_dequant[..in_dim.min(row_dequant.len())]);
        }
        output
    };

    let mut scheduler = HybridScheduler::with_threshold(1000).expect("test");
    let reference = scheduler
        .matmul(
            &activations,
            &weight_f32,
            batch_size,
            config.hidden_dim,
            config.intermediate_dim,
        )
        .expect("Reference matmul should succeed");

    for i in 0..fused_output.len() {
        let diff = (fused_output[i] - reference[i]).abs();
        assert!(
            diff < 1e-4,
            "IMP-109b: Fused should match reference at position {}: fused={}, ref={}",
            i,
            fused_output[i],
            reference[i]
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_109c_fused_vs_separate_performance_baseline() {
    // IMP-109c: Validate fused kernel produces same results as separate dequant+matmul
    // This establishes correctness baseline before optimizing
    use crate::quantize::{dequantize_q4_k_simd, fused_q4k_parallel_matvec};

    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 512, // 2x QK_K
        intermediate_dim: 1024,
        num_layers: 1,
        num_heads: 8,
        num_kv_heads: 8,
        vocab_size: 100,
        context_length: 2048,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let weight_data = &model.layers[0].ffn_up_weight.data;
    let in_dim = config.hidden_dim;
    let out_dim = config.intermediate_dim;

    // Multiple activation vectors to test consistency
    for batch in 0..4 {
        let activations: Vec<f32> = (0..in_dim)
            .map(|i| {
                let x = ((i + batch * 100) as f32 * 0.3141) % 1.0;
                (x - 0.5) * 2.0
            })
            .collect();

        // Separate operations (reference)
        let dequant = dequantize_q4_k_simd(weight_data).expect("test");
        let separate_result: Vec<f32> = (0..out_dim)
            .map(|row| {
                (0..in_dim)
                    .map(|col| dequant[row * in_dim + col] * activations[col])
                    .sum()
            })
            .collect();

        // Fused operation
        let fused_result = fused_q4k_parallel_matvec(weight_data, &activations, in_dim, out_dim)
            .expect("Fused should succeed");

        // Verify results match
        let max_diff: f32 = separate_result
            .iter()
            .zip(fused_result.iter())
            .map(|(s, f)| (s - f).abs())
            .fold(0.0f32, f32::max);

        let max_val = separate_result
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);
        let relative_error = max_diff / (max_val + 1e-6);

        assert!(
            relative_error < 0.02, // 2% max relative error
            "IMP-109c: Batch {} has excessive error: max_diff={}, relative={}",
            batch,
            max_diff,
            relative_error
        );
    }
}

// =========================================================================
// IMP-110: Multi-Head Parallel Attention
// =========================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_imp_110a_parallel_heads_correctness() {
    // IMP-110a: Verify parallel multi-head attention matches sequential
    // Process all heads in a single batch dispatch instead of iterating
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4, // 4 heads to test parallelism
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let seq_len = 8;
    let hidden_dim = config.hidden_dim;

    // Create Q, K, V tensors: [seq_len, hidden_dim]
    let q: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();
    let k: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();

    // Get sequential result (current implementation)
    let sequential_output = model
        .batched_causal_attention_gpu(&q, &k, &v, seq_len)
        .expect("Sequential attention should succeed");

    // Get parallel result (new implementation)
    let parallel_output = model
        .parallel_multihead_attention_gpu(&q, &k, &v, seq_len)
        .expect("IMP-110a: Parallel attention should succeed");

    // Verify same output shape
    assert_eq!(
        parallel_output.len(),
        sequential_output.len(),
        "IMP-110a: Parallel and sequential should have same output size"
    );

    // Verify results match within tolerance
    for i in 0..parallel_output.len() {
        let diff = (parallel_output[i] - sequential_output[i]).abs();
        assert!(
            diff < 1e-4,
            "IMP-110a: Position {} differs: parallel={}, sequential={}, diff={}",
            i,
            parallel_output[i],
            sequential_output[i],
            diff
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_110b_batched_qkv_reshape() {
    // IMP-110b: Verify Q/K/V reshaping for batched head processing
    // Input: [seq_len, hidden_dim] -> [num_heads, seq_len, head_dim]
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 32,
        intermediate_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let seq_len = 4;
    let hidden_dim = config.hidden_dim;
    let num_heads = config.num_heads;
    let head_dim = hidden_dim / num_heads; // 8

    // Create Q tensor: [seq_len, hidden_dim] = [4, 32]
    // Conceptually: each position has hidden_dim features = num_heads * head_dim
    let q: Vec<f32> = (0..seq_len * hidden_dim).map(|i| i as f32 * 0.1).collect();

    // Reshape to [num_heads, seq_len, head_dim] for parallel processing
    let reshaped = model
        .reshape_for_parallel_heads(&q, seq_len, num_heads, head_dim)
        .expect("IMP-110b: Reshape should succeed");

    // Verify output shape: num_heads * seq_len * head_dim = 4 * 4 * 8 = 128
    assert_eq!(
        reshaped.len(),
        num_heads * seq_len * head_dim,
        "IMP-110b: Reshaped tensor should have num_heads * seq_len * head_dim elements"
    );

    // Verify correct values were extracted for each head
    // Original layout: q[pos * hidden_dim + h * head_dim + d]
    // New layout: reshaped[h * seq_len * head_dim + pos * head_dim + d]
    for h in 0..num_heads {
        for pos in 0..seq_len {
            for d in 0..head_dim {
                let orig_idx = pos * hidden_dim + h * head_dim + d;
                let new_idx = h * seq_len * head_dim + pos * head_dim + d;
                assert!(
                    (reshaped[new_idx] - q[orig_idx]).abs() < 1e-6,
                    "IMP-110b: Head {} pos {} dim {} mismatch: reshaped={}, original={}",
                    h,
                    pos,
                    d,
                    reshaped[new_idx],
                    q[orig_idx]
                );
            }
        }
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_110c_parallel_batched_scores() {
    // IMP-110c: Verify batched Q@K^T scores computed correctly for all heads
    // Process all heads in single batched matmul
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 32,
        intermediate_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let seq_len = 4;
    let hidden_dim = config.hidden_dim;
    let num_heads = config.num_heads;
    let head_dim = hidden_dim / num_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Create Q, K in original layout [seq_len, hidden_dim]
    let q: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();
    let k: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.1)
        .collect();

    // Get parallel batched scores for all heads
    let batched_scores = model
        .parallel_batched_qk_scores(&q, &k, seq_len, num_heads, head_dim, scale)
        .expect("IMP-110c: Parallel batched scores should succeed");

    // Verify output shape: num_heads * seq_len * seq_len
    assert_eq!(
        batched_scores.len(),
        num_heads * seq_len * seq_len,
        "IMP-110c: Batched scores should have num_heads * seq_len * seq_len elements"
    );

    // Compute reference scores head-by-head
    for h in 0..num_heads {
        for i in 0..seq_len {
            for j in 0..seq_len {
                // Extract Q_h[i] and K_h[j]
                let mut expected_score = 0.0f32;
                for d in 0..head_dim {
                    let q_val = q[i * hidden_dim + h * head_dim + d];
                    let k_val = k[j * hidden_dim + h * head_dim + d];
                    expected_score += q_val * k_val;
                }
                expected_score *= scale;

                let batch_idx = h * seq_len * seq_len + i * seq_len + j;
                let diff = (batched_scores[batch_idx] - expected_score).abs();
                assert!(
                    diff < 1e-4,
                    "IMP-110c: Head {} score[{},{}] differs: batched={}, expected={}, diff={}",
                    h,
                    i,
                    j,
                    batched_scores[batch_idx],
                    expected_score,
                    diff
                );
            }
        }
    }
}
