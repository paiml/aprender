//! GGUF Part 02: IMP-106 (Batch Prefill) + IMP-107 (GPU Batch Matmul) + IMP-108 (Batched Attention)
//!
//! Extracted from gguf_monolith.rs (PMAT-802)
//!
//! ## Test Coverage
#![allow(clippy::many_single_char_names)]
//!
//! - IMP-106a: Batch matmul correctness (sequential vs batch)
//! - IMP-106b: forward_batch output shape and determinism
//! - IMP-106c: prefill_batch KV cache population
//! - IMP-107a: GPU batch matmul correctness (requires `gpu` feature)
//! - IMP-107b: forward_batch_gpu integration (requires `gpu` feature)
//! - IMP-107c: HybridScheduler GPU/CPU decision (requires `gpu` feature)
//! - IMP-108a: Batched causal attention correctness (requires `gpu` feature)
//! - IMP-108b: Causal mask verification in GPU attention (requires `gpu` feature)
//! - IMP-108c: Attention softmax normalization (requires `gpu` feature)

use crate::gguf::test_helpers::create_test_model_with_config;
use crate::gguf::{GGUFConfig, OwnedQuantizedKVCache};

// =========================================================================
// IMP-106: Batch Prefill Optimization
// =========================================================================

#[test]
fn test_imp_106a_batch_matmul_correctness() {
    // IMP-106a: Verify batch matmul produces same results as sequential
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 32,
        intermediate_dim: 64,
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
    let hidden_dim = config.hidden_dim;

    // Create batch of 4 input vectors
    let batch_size = 4;
    let mut batch_input = Vec::with_capacity(batch_size * hidden_dim);
    for i in 0..batch_size {
        for j in 0..hidden_dim {
            batch_input.push((i * hidden_dim + j) as f32 * 0.01);
        }
    }

    // Sequential processing (current approach)
    let mut sequential_results = Vec::new();
    for i in 0..batch_size {
        let single_input = &batch_input[i * hidden_dim..(i + 1) * hidden_dim];
        let result = model.fused_matmul(single_input, &model.layers[0].ffn_up_weight);
        sequential_results.push(result.expect("test"));
    }

    // Batch processing (new approach)
    let batch_result = model
        .fused_matmul(&batch_input, &model.layers[0].ffn_up_weight)
        .expect("test");

    // Verify batch output has correct total length
    let expected_out_dim = model.layers[0].ffn_up_weight.out_dim;
    assert_eq!(
        batch_result.len(),
        batch_size * expected_out_dim,
        "IMP-106a: Batch output should have batch_size * out_dim elements"
    );

    // Verify each position matches sequential result
    for i in 0..batch_size {
        let batch_pos = &batch_result[i * expected_out_dim..(i + 1) * expected_out_dim];
        let seq_pos = &sequential_results[i];

        for (j, (&b, &s)) in batch_pos.iter().zip(seq_pos.iter()).enumerate() {
            assert!(
                (b - s).abs() < 1e-4,
                "IMP-106a: Batch[{i}][{j}]={b} should match sequential={s}"
            );
        }
    }
}

#[test]
fn test_imp_106b_forward_batch_correctness() {
    // IMP-106b: Verify forward_batch produces correct output shape
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 32,
        intermediate_dim: 64,
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

    // Batch of 4 tokens
    let tokens = vec![1u32, 5, 10, 20];

    // Forward batch should return [batch_size, vocab_size] logits
    let logits = model.forward_batch(&tokens).expect("test");

    assert_eq!(
        logits.len(),
        tokens.len() * config.vocab_size,
        "IMP-106b: forward_batch should return batch_size * vocab_size logits"
    );

    // Verify logits are finite (no NaN or infinity)
    assert!(
        logits.iter().all(|&x| x.is_finite()),
        "IMP-106b: All logits should be finite"
    );

    // Verify output is deterministic (run twice, get same result)
    let logits2 = model.forward_batch(&tokens).expect("test");
    assert_eq!(
        logits, logits2,
        "IMP-106b: forward_batch should be deterministic"
    );
}

#[test]
fn test_imp_106c_prefill_with_batch() {
    // IMP-106c: Verify prefill_batch populates KV cache correctly
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 32,
        intermediate_dim: 64,
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
    let mut cache = OwnedQuantizedKVCache::from_config(&config, 128);

    // Prefill with batch of 4 tokens
    let prompt = vec![1u32, 5, 10, 20];
    let last_logits = model.prefill_batch(&prompt, &mut cache).expect("test");

    // Should return only the last position's logits
    assert_eq!(
        last_logits.len(),
        config.vocab_size,
        "IMP-106c: prefill_batch should return vocab_size logits for last position"
    );

    // KV cache should be populated with all prompt positions
    assert_eq!(
        cache.len(),
        prompt.len(),
        "IMP-106c: KV cache should have {} positions after prefill",
        prompt.len()
    );
}

// =========================================================================
// IMP-107: GPU Batch Matmul Integration
// =========================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_imp_107a_gpu_batch_matmul_correctness() {
    // IMP-107a: Verify GPU batch matmul produces correct results
    // Uses HybridScheduler which routes to GPU for batch_size > 1
    use crate::gpu::HybridScheduler;

    let mut scheduler = HybridScheduler::with_threshold(100).expect("test");

    // Batch of 4 vectors (m=4), weight matrix 8x16 (k=8, n=16)
    // This exceeds threshold: 4 * 8 * 16 = 512 > 100
    let m = 4;
    let k = 8;
    let n = 16;

    // Create test data: A[m, k] @ B[k, n] = C[m, n]
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 8) as f32) * 0.1).collect();

    let result = scheduler.matmul(&a, &b, m, k, n).expect("test");

    assert_eq!(
        result.len(),
        m * n,
        "IMP-107a: GPU batch matmul should produce m*n outputs"
    );

    // Verify correctness with CPU reference
    let expected = cpu_matmul_reference(&a, &b, m, k, n);
    for i in 0..result.len() {
        assert!(
            (result[i] - expected[i]).abs() < 1e-4,
            "IMP-107a: GPU matmul result[{}] = {} differs from expected {}",
            i,
            result[i],
            expected[i]
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_107b_forward_batch_gpu() {
    // IMP-107b: Verify forward_batch_gpu uses GPU matmul for batch ops
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
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

    // Batch of 8 tokens - should trigger GPU path
    let tokens = vec![1u32, 5, 10, 20, 30, 40, 50, 60];
    let logits = model.forward_batch_gpu(&tokens).expect("test");

    assert_eq!(
        logits.len(),
        tokens.len() * config.vocab_size,
        "IMP-107b: forward_batch_gpu should produce batch_size * vocab_size logits"
    );

    // Verify logits are finite (not NaN or Inf)
    for (i, &logit) in logits.iter().enumerate() {
        assert!(
            logit.is_finite(),
            "IMP-107b: logit[{}] should be finite, got {}",
            i,
            logit
        );
    }

    // Verify determinism - same input produces same output
    let logits2 = model.forward_batch_gpu(&tokens).expect("test");
    for i in 0..logits.len() {
        assert!(
            (logits[i] - logits2[i]).abs() < 1e-6,
            "IMP-107b: GPU forward should be deterministic"
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_107c_gpu_crossover_decision() {
    // IMP-107c: Verify HybridScheduler makes correct GPU vs CPU decisions
    use crate::gpu::HybridScheduler;

    let scheduler = HybridScheduler::with_threshold(1000).expect("test");

    // Single token (m=1) should always use CPU
    assert!(
        !scheduler.should_use_gpu(1, 256, 128),
        "IMP-107c: m=1 (single token) should use CPU regardless of matrix size"
    );

    // Small batch below threshold: 2 * 10 * 10 = 200 < 1000
    assert!(
        !scheduler.should_use_gpu(2, 10, 10),
        "IMP-107c: Small batch below threshold should use CPU"
    );

    // Large batch above threshold: 4 * 256 * 128 = 131072 > 1000
    if scheduler.has_gpu() {
        assert!(
            scheduler.should_use_gpu(4, 256, 128),
            "IMP-107c: Large batch above threshold should use GPU"
        );

        // Medium batch at threshold boundary: 2 * 32 * 16 = 1024 > 1000
        assert!(
            scheduler.should_use_gpu(2, 32, 16),
            "IMP-107c: Batch just above threshold should use GPU"
        );
    }
}

/// CPU reference matmul for correctness verification
#[cfg(feature = "gpu")]
fn cpu_matmul_reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

// =========================================================================
// IMP-108: Batched Causal Attention with GPU
// =========================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_imp_108a_batched_causal_attention_correctness() {
    // IMP-108a: Verify batched causal attention matches sequential computation
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 32,
        intermediate_dim: 64,
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

    // Create test Q, K, V for 4 positions
    let seq_len = 4;
    let hidden_dim = config.hidden_dim;
    let q: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();
    let k: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
        .collect();

    // Get batched result (GPU-accelerated when beneficial)
    let batched_output = model
        .batched_causal_attention_gpu(&q, &k, &v, seq_len)
        .expect("test");

    // Get sequential reference result
    let sequential_output = model.causal_attention(&q, &k, &v, seq_len);

    // Should have same shape
    assert_eq!(
        batched_output.len(),
        sequential_output.len(),
        "IMP-108a: Batched and sequential attention should have same output size"
    );

    // Verify results match (within floating point tolerance)
    for i in 0..batched_output.len() {
        assert!(
            (batched_output[i] - sequential_output[i]).abs() < 1e-4,
            "IMP-108a: Position {} differs: batched={}, sequential={}",
            i,
            batched_output[i],
            sequential_output[i]
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_108b_causal_mask_gpu() {
    // IMP-108b: Verify causal mask is correctly applied in GPU attention
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 16, // Small for easy verification
        intermediate_dim: 32,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
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

    // Create Q, K, V where we can detect if future tokens are attended to
    // K at position 3 has very large values - if attended to by position 0,
    // output would be very different
    let q = vec![0.1f32; seq_len * hidden_dim];
    let mut k = vec![0.1f32; seq_len * hidden_dim];
    let mut v = vec![0.1f32; seq_len * hidden_dim];

    // Make K[3] very large - should NOT affect position 0's output
    for d in 0..hidden_dim {
        k[3 * hidden_dim + d] = 100.0;
        v[3 * hidden_dim + d] = 100.0;
    }

    let output = model
        .batched_causal_attention_gpu(&q, &k, &v, seq_len)
        .expect("test");

    // Position 0 can only attend to position 0, so should NOT see the large K[3]/V[3]
    let pos0_norm: f32 = output[0..hidden_dim].iter().map(|x| x.abs()).sum();

    // Position 0's output should be based only on V[0] (which is small)
    // If causal mask is wrong, pos0_norm would be ~100 (from V[3])
    assert!(
        pos0_norm < 5.0, // Should be small since V[0] = 0.1
        "IMP-108b: Position 0 should not attend to future positions, got norm={}",
        pos0_norm
    );

    // Position 3 CAN attend to position 3, so its output includes the large values
    let pos3_norm: f32 = output[3 * hidden_dim..4 * hidden_dim]
        .iter()
        .map(|x| x.abs())
        .sum();
    assert!(
        pos3_norm > 10.0, // Should include contribution from V[3]
        "IMP-108b: Position 3 should attend to itself (large V), got norm={}",
        pos3_norm
    );
}

include!("imp_108c.rs");
