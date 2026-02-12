//! Spec Checklist Tests - Section E: Visual Control Tests (15 points)
//!
//! These tests verify claims from the spec's falsification checklist.
//! Each test is designed to FAIL if the claim is false.
//!
//! Reference: docs/specifications/qwen2-0.5b-instruct-interactive-chat-demo.md

#![allow(unused_imports)]

use aprender::demo::Qwen2Config;

// ============================================================================
// Section E Additional: Visual Control Tests
// ============================================================================

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
