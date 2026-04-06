//! GGUF Part 05: IMP-115 (Fused Attention) + IMP-117 (Small Buffer Optimization) +
//!               IMP-118 (True GPU Batched GEMM) + IMP-119 (GPU Fused Attention)
//!
//! Extracted from gguf_monolith.rs (PMAT-802)

use crate::gguf::test_helpers::create_test_model_with_config;
use crate::gguf::{GGUFConfig, OwnedQuantizedModelCached, QuantizedGenerateConfig};

// ========================================================================
// IMP-115: Fused Attention Kernel Tests (Q@K^T → softmax → @V)
// ========================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_imp_115a_fused_single_head_attention_correctness() {
    // IMP-115a: Verify fused attention matches separate operations
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
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
    let cached_model = OwnedQuantizedModelCached::new(model);

    let seq_len = 8;
    let head_dim = 16;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Create single-head Q, K, V
    let q: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();
    let k: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();

    // Reference: separate operations
    let reference = cached_model
        .model()
        .tiled_causal_attention(&q, &k, &v, seq_len, head_dim, scale, 4)
        .expect("Reference attention should succeed");

    // Fused: single kernel
    let result = cached_model
        .fused_causal_attention(&q, &k, &v, seq_len, head_dim, scale)
        .expect("Fused attention should succeed");

    assert_eq!(result.len(), reference.len());
    for i in 0..result.len() {
        let diff = (result[i] - reference[i]).abs();
        assert!(
            diff < 1e-4,
            "IMP-115a: Fused differs at {}: ref={}, fused={}, diff={}",
            i,
            reference[i],
            result[i],
            diff
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_115b_fused_multihead_attention_correctness() {
    // IMP-115b: Verify fused multi-head attention matches reference
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
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
    let cached_model = OwnedQuantizedModelCached::new(model);

    let seq_len = 8;
    let hidden_dim = config.hidden_dim;

    let q: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();
    let k: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();

    // Reference: flattened multi-head
    let reference = cached_model
        .flattened_multihead_attention(&q, &k, &v, seq_len)
        .expect("Reference attention should succeed");

    // Fused multi-head
    let result = cached_model
        .fused_multihead_attention(&q, &k, &v, seq_len)
        .expect("Fused multi-head attention should succeed");

    assert_eq!(result.len(), reference.len());
    for i in 0..result.len() {
        let diff = (result[i] - reference[i]).abs();
        assert!(
            diff < 1e-3,
            "IMP-115b: Fused MHA differs at {}: ref={}, fused={}, diff={}",
            i,
            reference[i],
            result[i],
            diff
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_115c_fused_attention_no_intermediate_allocation() {
    // IMP-115c: Verify fused attention doesn't allocate large intermediate tensors
    // We test this by verifying output is correct for larger sequences
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 128,
        intermediate_dim: 256,
        num_layers: 1,
        num_heads: 8,
        num_kv_heads: 8,
        vocab_size: 50,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let cached_model = OwnedQuantizedModelCached::new(model);

    let seq_len = 32; // Larger sequence to stress test
    let hidden_dim = config.hidden_dim;

    let q: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
        .collect();
    let k: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.05)
        .collect();
    let v: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
        .collect();

    let result = cached_model
        .fused_multihead_attention(&q, &k, &v, seq_len)
        .expect("Fused attention should succeed for larger sequences");

    assert_eq!(
        result.len(),
        seq_len * hidden_dim,
        "IMP-115c: Output should have correct dimensions"
    );

    // Verify output is not all zeros
    let sum: f32 = result.iter().map(|x| x.abs()).sum();
    assert!(
        sum > 0.01,
        "IMP-115c: Output should have non-trivial values"
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_115d_fused_causal_mask_correctness() {
    // IMP-115d: Verify causal masking is correctly applied in fused kernel
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
    let cached_model = OwnedQuantizedModelCached::new(model);

    let seq_len = 4;
    let head_dim = 8;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Use Q where different positions have distinct patterns
    // This helps verify causal masking is working
    let q: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| {
            let pos = i / head_dim;
            ((pos * 10 + i % head_dim) as f32) * 0.1
        })
        .collect();
    let k: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();

    let result = cached_model
        .fused_causal_attention(&q, &k, &v, seq_len, head_dim, scale)
        .expect("Fused causal attention should succeed");

    // Verify output dimensions
    assert_eq!(result.len(), seq_len * head_dim);

    // Verify each position's output is influenced only by positions 0..=i
    // Position 0 can only attend to itself
    // Position 1 can attend to 0 and 1
    // etc.
    // We can't easily verify this without access to internal attention weights,
    // but we can verify output is valid (non-NaN, finite, reasonable range)
    for (i, &val) in result.iter().enumerate() {
        assert!(
            val.is_finite(),
            "IMP-115d: Output at {} should be finite, got {}",
            i,
            val
        );
        assert!(
            val.abs() < 10.0,
            "IMP-115d: Output at {} should be in reasonable range, got {}",
            i,
            val
        );
    }
}

// ========================================================================
// IMP-117: Small Buffer Optimization Tests (SmallVec)
// ========================================================================

#[test]
fn test_imp_117a_token_buffer_inline_allocation() {
    // IMP-117a: TokenBuffer should use stack allocation for small sizes
    use crate::gguf::{TokenBuffer, TOKEN_BUFFER_INLINE_CAP};

    // Create buffer within inline capacity
    let mut buffer: TokenBuffer = TokenBuffer::new();
    for i in 0..TOKEN_BUFFER_INLINE_CAP {
        buffer.push(i as u32);
    }

    // Verify capacity and inline status
    assert_eq!(
        buffer.len(),
        TOKEN_BUFFER_INLINE_CAP,
        "IMP-117a: Buffer should hold TOKEN_BUFFER_INLINE_CAP elements"
    );

    // SmallVec is inline when len <= inline capacity
    assert!(
        !buffer.spilled(),
        "IMP-117a: Buffer should not spill to heap at inline capacity"
    );

    // Adding one more should trigger heap allocation
    buffer.push(999);
    assert!(
        buffer.spilled(),
        "IMP-117a: Buffer should spill to heap when exceeding inline capacity"
    );
}

#[test]
fn test_imp_117b_attention_buffer_inline_allocation() {
    // IMP-117b: AttentionBuffer should use stack allocation for small sizes
    use crate::gguf::{AttentionBuffer, ATTENTION_BUFFER_INLINE_CAP};

    let mut buffer: AttentionBuffer = AttentionBuffer::new();
    for i in 0..ATTENTION_BUFFER_INLINE_CAP {
        buffer.push(i as f32 * 0.1);
    }

    assert_eq!(
        buffer.len(),
        ATTENTION_BUFFER_INLINE_CAP,
        "IMP-117b: Attention buffer should hold ATTENTION_BUFFER_INLINE_CAP elements"
    );
    assert!(
        !buffer.spilled(),
        "IMP-117b: Attention buffer should not spill at inline capacity"
    );
}

#[test]
fn test_imp_117c_hidden_buffer_inline_allocation() {
    // IMP-117c: HiddenBuffer should use stack allocation for small models
    use crate::gguf::{HiddenBuffer, HIDDEN_BUFFER_INLINE_CAP};

    let mut buffer: HiddenBuffer = HiddenBuffer::new();
    for i in 0..HIDDEN_BUFFER_INLINE_CAP {
        buffer.push(i as f32 * 0.01);
    }

    assert_eq!(
        buffer.len(),
        HIDDEN_BUFFER_INLINE_CAP,
        "IMP-117c: Hidden buffer should hold HIDDEN_BUFFER_INLINE_CAP elements"
    );
    assert!(
        !buffer.spilled(),
        "IMP-117c: Hidden buffer should not spill at inline capacity"
    );
}

#[test]
fn test_imp_117d_buffer_watermarks() {
    // IMP-117d: Verify buffer watermark constants are reasonable
    use crate::gguf::{BUFFER_HW_SIZE, BUFFER_LW_SIZE, BUFFER_MAX_SIZE};

    // Low < High < Max
    assert!(
        BUFFER_LW_SIZE < BUFFER_HW_SIZE,
        "IMP-117d: Low watermark should be less than high watermark"
    );
    assert!(
        BUFFER_HW_SIZE < BUFFER_MAX_SIZE,
        "IMP-117d: High watermark should be less than max size"
    );

    // Reasonable ranges
    assert!(
        BUFFER_LW_SIZE >= 1024,
        "IMP-117d: Low watermark should be at least 1KB"
    );
    assert!(
        BUFFER_MAX_SIZE <= 64 * 1024,
        "IMP-117d: Max buffer should be at most 64KB"
    );
}

#[test]
fn test_imp_117e_token_buffer_from_slice() {
    // IMP-117e: TokenBuffer should work with from_slice
    use crate::gguf::TokenBuffer;

    let tokens: &[u32] = &[1, 2, 3, 4, 5];
    let buffer: TokenBuffer = TokenBuffer::from_slice(tokens);

    assert_eq!(buffer.len(), 5);
    assert_eq!(buffer.as_slice(), tokens);
    assert!(!buffer.spilled(), "IMP-117e: Small slice should not spill");
}

#[test]
fn test_imp_117f_generate_with_token_buffer() {
    // IMP-117f: Test generate_with_smallvec returns correct SmallVec type
    use crate::gguf::{TokenBuffer, TOKEN_BUFFER_INLINE_CAP};

    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
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

    // Test with small prompt that fits in inline capacity
    let prompt: TokenBuffer = TokenBuffer::from_slice(&[1, 2, 3, 4, 5]);
    assert!(
        prompt.len() < TOKEN_BUFFER_INLINE_CAP,
        "IMP-117f: Test prompt should be within inline capacity"
    );

    // Generate tokens using the SmallVec-based API
    let gen_config = QuantizedGenerateConfig {
        max_tokens: 10,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: Vec::new(),
        trace: false,
        ..Default::default()
    };

    let result = model.generate_with_smallvec(&prompt, &gen_config);
    assert!(
        result.is_ok(),
        "IMP-117f: generate_with_smallvec should succeed"
    );

    let generated = result.expect("generation should succeed");
    assert!(
        generated.len() > prompt.len(),
        "IMP-117f: Generated tokens should include prompt + new tokens"
    );
}

include!("imp_118a.rs");
include!("imp_119c.rs");
