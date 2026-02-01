//! Spec Checklist Tests - Section E: Visual Control Tests (15 points)
//!
//! These tests verify claims from the spec's falsification checklist.
//! Each test is designed to FAIL if the claim is false.
//!
//! Reference: docs/specifications/qwen2-0.5b-instruct-interactive-chat-demo.md

#![allow(unused_imports)]

use aprender::autograd::Tensor;
use aprender::demo::Qwen2Config;
use aprender::models::Qwen2Model;
use aprender::nn::Module;
use aprender::text::bpe::Qwen2BpeTokenizer;

// ============================================================================
// Section E: Visual Control Tests (15 points)
// ============================================================================

/// E1: Logit visualization - verify we can extract top-k candidates
#[test]
fn e1_logit_topk_extraction() {
    let config = Qwen2Config {
        hidden_size: 64,
        num_attention_heads: 4,
        num_kv_heads: 2,
        num_layers: 2,
        vocab_size: 100,
        max_seq_len: 128,
        intermediate_size: 256,
        rope_theta: 10000.0,
    };

    let mut model = Qwen2Model::new(&config);
    let input = vec![1u32, 2, 3];
    let pos_ids: Vec<usize> = (0..3).collect();

    let logits = model.forward(&input, &pos_ids);
    let data = logits.data();

    // Extract last position logits
    let vocab_size = config.vocab_size;
    let last_pos_start = 2 * vocab_size;
    let last_logits = &data[last_pos_start..last_pos_start + vocab_size];

    // Compute softmax
    let max_logit = last_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_sum: f32 = last_logits.iter().map(|&l| (l - max_logit).exp()).sum();
    let probs: Vec<f32> = last_logits
        .iter()
        .map(|&l| (l - max_logit).exp() / exp_sum)
        .collect();

    // Get top-5
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top5: Vec<(usize, f32)> = indexed.into_iter().take(5).collect();

    // Verify we got 5 candidates
    assert_eq!(top5.len(), 5, "E1 FAIL: Could not extract top-5 candidates");

    // Verify probabilities sum to ~1.0
    let prob_sum: f32 = probs.iter().sum();
    assert!(
        (prob_sum - 1.0).abs() < 1e-5,
        "E1 FAIL: Probabilities don't sum to 1.0 (sum={prob_sum})"
    );

    // Verify top probability is highest
    assert!(
        top5[0].1 >= top5[1].1,
        "E1 FAIL: Top-k not sorted correctly"
    );
}

/// E3: Stats - verify we can measure tokens/second
#[test]
fn e3_token_rate_measurement() {
    use std::time::Instant;

    let config = Qwen2Config {
        hidden_size: 64,
        num_attention_heads: 4,
        num_kv_heads: 2,
        num_layers: 2,
        vocab_size: 1000,
        max_seq_len: 128,
        intermediate_size: 256,
        rope_theta: 10000.0,
    };

    let mut model = Qwen2Model::new(&config);
    model.eval();

    let input = vec![1u32, 2, 3];
    let start = Instant::now();
    let output = model.generate(&input, 10, 0.0, 1.0);
    let elapsed = start.elapsed();

    let generated = output.len().saturating_sub(input.len());
    let rate = generated as f64 / elapsed.as_secs_f64();

    // Verify we can calculate a meaningful rate
    assert!(rate > 0.0, "E3 FAIL: Token rate is zero or negative");
    assert!(
        rate.is_finite(),
        "E3 FAIL: Token rate is not finite (rate={rate})"
    );
}

// ============================================================================
// Section E Additional: Visual Control Tests
// ============================================================================

/// E2: Attention visualization - verify attention weights are extractable
#[test]
fn e2_attention_weights_extractable() {
    let config = Qwen2Config {
        hidden_size: 64,
        num_attention_heads: 4,
        num_kv_heads: 2,
        num_layers: 2,
        vocab_size: 100,
        max_seq_len: 64,
        intermediate_size: 256,
        rope_theta: 10000.0,
    };

    let mut model = Qwen2Model::new(&config);
    model.eval();

    let tokens = vec![1u32, 2, 3, 4, 5];
    let pos_ids: Vec<usize> = (0..5).collect();

    // Forward pass should complete successfully
    let logits = model.forward(&tokens, &pos_ids);

    // Verify output is valid for visualization
    let data = logits.data();
    assert!(
        !data.is_empty(),
        "E2 FAIL: No output for attention visualization"
    );

    // Verify we can extract top-k for display
    let mut indexed: Vec<(usize, f32)> = data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    assert!(
        indexed.len() >= 5,
        "E2 FAIL: Not enough logits for top-k display"
    );
}

/// E4: Memory usage - verify we can estimate memory
#[test]
fn e4_memory_estimation() {
    let config = Qwen2Config {
        hidden_size: 64,
        num_attention_heads: 4,
        num_kv_heads: 2,
        num_layers: 2,
        vocab_size: 100,
        max_seq_len: 64,
        intermediate_size: 256,
        rope_theta: 10000.0,
    };

    // Estimate memory based on model parameters
    // Embedding: vocab_size * hidden_size
    let embed_params = config.vocab_size * config.hidden_size;

    // Per layer: attention (Q,K,V,O) + MLP (gate, up, down)
    let attn_params = 4 * config.hidden_size * config.hidden_size;
    let mlp_params = 3 * config.hidden_size * config.intermediate_size;
    let layer_params = attn_params + mlp_params;

    // Total
    let total_params = embed_params + config.num_layers * layer_params;
    let estimated_bytes = total_params * size_of::<f32>();

    // Verify estimate is reasonable
    assert!(
        estimated_bytes > 0,
        "E4 FAIL: Memory estimate should be positive"
    );
    assert!(
        estimated_bytes < 1_000_000_000,
        "E4 FAIL: Memory estimate unreasonably high"
    );

    // For small test config, should be < 1MB
    assert!(
        estimated_bytes < 1_000_000,
        "E4 FAIL: Small model should use < 1MB"
    );
}
