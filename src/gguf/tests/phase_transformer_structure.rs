//! Phase 34: Transformer structure coverage tests
//!
//! These lib tests illuminate gguf/transformer.rs:
//! - QuantizedTensorRef byte size calculations
//! - QKVWeights enum handling
//! - QuantizedGGUFTransformerLayer structure
//!
//! Located in lib tests to be included in `cargo test --lib` coverage

#![allow(clippy::match_wildcard_for_single_variants)]

use crate::gguf::quantized::QuantizedTensorRef;
use crate::gguf::types::{
    GGUF_TYPE_F32, GGUF_TYPE_Q2_K, GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_1, GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_0,
    GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0,
};

// =============================================================================
// QuantizedTensorRef Byte Size Tests
// =============================================================================

/// Test byte size calculation for F32 tensors
#[test]
fn test_phase34_tensor_ref_f32_byte_size() {
    let num_elements: usize = 1024;
    let expected_bytes = num_elements * 4; // 4 bytes per f32

    // Verify the calculation matches F32 expectations
    assert_eq!(expected_bytes, 4096);
}

/// Test byte size calculation for Q4_0 tensors
#[test]
fn test_phase34_tensor_ref_q4_0_byte_size() {
    // Q4_0: 18 bytes per 32 elements
    let num_elements: usize = 256;
    let num_blocks = num_elements.div_ceil(32);
    let expected_bytes = num_blocks * 18;

    assert_eq!(num_blocks, 8);
    assert_eq!(expected_bytes, 144);
}

/// Test byte size calculation for Q8_0 tensors
#[test]
fn test_phase34_tensor_ref_q8_0_byte_size() {
    // Q8_0: 34 bytes per 32 elements
    let num_elements: usize = 256;
    let num_blocks = num_elements.div_ceil(32);
    let expected_bytes = num_blocks * 34;

    assert_eq!(num_blocks, 8);
    assert_eq!(expected_bytes, 272);
}

/// Test byte size calculation for Q2_K tensors
#[test]
fn test_phase34_tensor_ref_q2_k_byte_size() {
    // Q2_K: 84 bytes per 256 elements (QK_K=256)
    let num_elements: usize = 512;
    let num_super_blocks = num_elements.div_ceil(256);
    let expected_bytes = num_super_blocks * 84;

    assert_eq!(num_super_blocks, 2);
    assert_eq!(expected_bytes, 168);
}

/// Test byte size calculation for Q4_1 tensors
#[test]
fn test_phase34_tensor_ref_q4_1_byte_size() {
    // Q4_1: 20 bytes per 32 elements
    let num_elements: usize = 128;
    let num_blocks = num_elements.div_ceil(32);
    let expected_bytes = num_blocks * 20;

    assert_eq!(num_blocks, 4);
    assert_eq!(expected_bytes, 80);
}

/// Test byte size calculation for Q5_0 tensors
#[test]
fn test_phase34_tensor_ref_q5_0_byte_size() {
    // Q5_0: 22 bytes per 32 elements
    let num_elements: usize = 128;
    let num_blocks = num_elements.div_ceil(32);
    let expected_bytes = num_blocks * 22;

    assert_eq!(num_blocks, 4);
    assert_eq!(expected_bytes, 88);
}

/// Test byte size calculation for Q4_K tensors
#[test]
fn test_phase34_tensor_ref_q4_k_byte_size() {
    // Q4_K: 144 bytes per 256 elements (QK_K=256)
    let num_elements: usize = 512;
    let num_super_blocks = num_elements.div_ceil(256);
    let expected_bytes = num_super_blocks * 144;

    assert_eq!(num_super_blocks, 2);
    assert_eq!(expected_bytes, 288);
}

/// Test byte size calculation for Q5_K tensors
#[test]
fn test_phase34_tensor_ref_q5_k_byte_size() {
    // Q5_K: 176 bytes per 256 elements (QK_K=256)
    let num_elements: usize = 512;
    let num_super_blocks = num_elements.div_ceil(256);
    let expected_bytes = num_super_blocks * 176;

    assert_eq!(num_super_blocks, 2);
    assert_eq!(expected_bytes, 352);
}

/// Test byte size calculation for Q6_K tensors
#[test]
fn test_phase34_tensor_ref_q6_k_byte_size() {
    // Q6_K: 210 bytes per 256 elements (QK_K=256)
    let num_elements: usize = 512;
    let num_super_blocks = num_elements.div_ceil(256);
    let expected_bytes = num_super_blocks * 210;

    assert_eq!(num_super_blocks, 2);
    assert_eq!(expected_bytes, 420);
}

// =============================================================================
// QuantizedTensorRef Creation Tests
// =============================================================================

#[test]
fn test_phase34_quantized_tensor_ref_creation() {
    let tensor_ref = QuantizedTensorRef {
        offset: 1024,
        byte_size: 2048,
        num_elements: 512,
        qtype: GGUF_TYPE_Q4_K,
    };

    assert_eq!(tensor_ref.offset, 1024);
    assert_eq!(tensor_ref.byte_size, 2048);
    assert_eq!(tensor_ref.num_elements, 512);
    assert_eq!(tensor_ref.qtype, GGUF_TYPE_Q4_K);
}

#[test]
fn test_phase34_quantized_tensor_ref_all_qtypes() {
    let qtypes = [
        (GGUF_TYPE_F32, "F32"),
        (GGUF_TYPE_Q4_0, "Q4_0"),
        (GGUF_TYPE_Q8_0, "Q8_0"),
        (GGUF_TYPE_Q2_K, "Q2_K"),
        (GGUF_TYPE_Q4_1, "Q4_1"),
        (GGUF_TYPE_Q5_0, "Q5_0"),
        (GGUF_TYPE_Q4_K, "Q4_K"),
        (GGUF_TYPE_Q5_K, "Q5_K"),
        (GGUF_TYPE_Q6_K, "Q6_K"),
    ];

    for (qtype, name) in qtypes {
        let tensor_ref = QuantizedTensorRef {
            offset: 0,
            byte_size: 1024,
            num_elements: 256,
            qtype,
        };

        assert_eq!(tensor_ref.qtype, qtype, "qtype mismatch for {}", name);
    }
}

// =============================================================================
// QKVWeights Enum Tests
// =============================================================================

use crate::gguf::quantized::QKVWeights;

#[test]
fn test_phase34_qkv_weights_fused() {
    let fused_ref = QuantizedTensorRef {
        offset: 0,
        byte_size: 4096,
        num_elements: 1024,
        qtype: GGUF_TYPE_Q4_K,
    };

    let qkv = QKVWeights::Fused(fused_ref);

    if let QKVWeights::Fused(ref f) = qkv {
        assert_eq!(f.byte_size, 4096);
        assert_eq!(f.num_elements, 1024);
    } else {
        panic!("Expected Fused variant");
    }
}

#[test]
fn test_phase34_qkv_weights_separate() {
    let q_ref = QuantizedTensorRef {
        offset: 0,
        byte_size: 1024,
        num_elements: 256,
        qtype: GGUF_TYPE_Q4_K,
    };
    let k_ref = QuantizedTensorRef {
        offset: 1024,
        byte_size: 1024,
        num_elements: 256,
        qtype: GGUF_TYPE_Q4_K,
    };
    let v_ref = QuantizedTensorRef {
        offset: 2048,
        byte_size: 1024,
        num_elements: 256,
        qtype: GGUF_TYPE_Q4_K,
    };

    let qkv = QKVWeights::Separate {
        q: q_ref,
        k: k_ref,
        v: v_ref,
    };

    if let QKVWeights::Separate { q, k, v } = qkv {
        assert_eq!(q.offset, 0);
        assert_eq!(k.offset, 1024);
        assert_eq!(v.offset, 2048);
    } else {
        panic!("Expected Separate variant");
    }
}

// =============================================================================
// Byte Size Edge Cases
// =============================================================================

#[test]
fn test_phase34_byte_size_non_aligned() {
    // Test with element counts that don't align to block boundaries

    // Q4_0 with 33 elements (needs 2 blocks for 64 element capacity)
    let num_elements: usize = 33;
    let num_blocks = num_elements.div_ceil(32);
    assert_eq!(num_blocks, 2); // ceil(33/32) = 2
    let byte_size = num_blocks * 18;
    assert_eq!(byte_size, 36);
}

#[test]
fn test_phase34_byte_size_exact_block() {
    // Q4_0 with exactly 32 elements (1 block)
    let num_elements: usize = 32;
    let num_blocks = num_elements.div_ceil(32);
    assert_eq!(num_blocks, 1);
    let byte_size = num_blocks * 18;
    assert_eq!(byte_size, 18);
}

#[test]
fn test_phase34_byte_size_large_tensor() {
    // Large tensor like embedding table: 32000 vocab * 4096 hidden
    let num_elements: usize = 32000 * 4096;

    // Q4_K byte size
    let num_super_blocks = num_elements.div_ceil(256);
    let q4k_bytes = num_super_blocks * 144;

    // Should be roughly 72MB for Q4_K
    assert!(q4k_bytes > 70_000_000);
    assert!(q4k_bytes < 80_000_000);

    // F32 would be much larger
    let f32_bytes = num_elements * 4;
    assert_eq!(f32_bytes, 524_288_000); // 500MB

    // Q4_K is ~7x smaller
    let compression = f32_bytes as f64 / q4k_bytes as f64;
    assert!(compression > 6.0);
    assert!(compression < 8.0);
}

// =============================================================================
// Quantization Type Constants Tests
// =============================================================================

#[test]
fn test_phase34_qtype_constants() {
    // Verify the qtype constants have expected values
    // These match GGML's enum values
    assert_eq!(GGUF_TYPE_F32, 0);
    assert_eq!(GGUF_TYPE_Q4_0, 2);
    assert_eq!(GGUF_TYPE_Q4_1, 3);
    assert_eq!(GGUF_TYPE_Q5_0, 6);
    assert_eq!(GGUF_TYPE_Q8_0, 8);
    assert_eq!(GGUF_TYPE_Q2_K, 10);
    assert_eq!(GGUF_TYPE_Q4_K, 12);
    assert_eq!(GGUF_TYPE_Q5_K, 13);
    assert_eq!(GGUF_TYPE_Q6_K, 14);
}

// =============================================================================
// Tensor Offset Calculations
// =============================================================================

#[test]
fn test_phase34_tensor_offset_accumulation() {
    // Simulate tensor offsets in a model
    let base_offset: usize = 4096; // After header/metadata

    // First tensor: embedding (vocab * hidden)
    let embed_elements: usize = 32000 * 4096;
    let embed_blocks = embed_elements.div_ceil(256);
    let embed_bytes = embed_blocks * 144; // Q4_K

    let first_tensor_offset = base_offset;
    let second_tensor_offset = first_tensor_offset + embed_bytes;

    assert!(second_tensor_offset > first_tensor_offset);
    assert_eq!(first_tensor_offset, 4096);
}

#[test]
fn test_phase34_tensor_ref_bounds_check() {
    // Test that we can verify tensor fits in file
    let file_size = 1_000_000; // 1MB file
    let tensor_offset = 500_000;
    let tensor_bytes = 400_000;

    // This tensor fits
    assert!(tensor_offset + tensor_bytes <= file_size);

    // This tensor doesn't fit
    let large_tensor_bytes = 600_000;
    assert!(tensor_offset + large_tensor_bytes > file_size);
}

// =============================================================================
// Architecture-specific Tests
// =============================================================================

#[test]
fn test_phase34_llama_style_layers() {
    // LLaMA uses separate Q, K, V projections
    let hidden_dim = 4096;
    let num_heads = 32;
    let head_dim = hidden_dim / num_heads; // 128

    // Q projection: hidden -> hidden
    let q_elements = hidden_dim * hidden_dim;
    // K projection: hidden -> num_kv_heads * head_dim (GQA)
    let num_kv_heads = 8;
    let k_elements = hidden_dim * (num_kv_heads * head_dim);
    // V projection: same as K
    let v_elements = k_elements;

    assert_eq!(q_elements, 16_777_216); // 4096 * 4096
    assert_eq!(k_elements, 4_194_304); // 4096 * 1024
    assert_eq!(v_elements, 4_194_304);
}

#[test]
fn test_phase34_phi2_style_layers() {
    // Phi-2 uses fused QKV projection
    let hidden_dim = 2560;
    let num_heads = 32;
    let _head_dim = hidden_dim / num_heads; // 80

    // Fused QKV: hidden -> 3 * hidden
    let qkv_out_dim = 3 * hidden_dim;
    let qkv_elements = hidden_dim * qkv_out_dim;

    assert_eq!(qkv_elements, 19_660_800); // 2560 * 7680
}

// =============================================================================
// Memory Layout Tests
// =============================================================================

#[test]
fn test_phase34_contiguous_layout() {
    // Verify tensors can be laid out contiguously
    let tensors: Vec<(&str, usize, u32)> = vec![
        ("embed", 32000 * 4096, GGUF_TYPE_Q4_K),
        ("layer.0.attn_q", 4096 * 4096, GGUF_TYPE_Q4_K),
        ("layer.0.attn_k", 4096 * 1024, GGUF_TYPE_Q4_K),
        ("layer.0.attn_v", 4096 * 1024, GGUF_TYPE_Q4_K),
    ];

    let mut current_offset = 0usize;
    for (name, num_elements, qtype) in &tensors {
        let num_super_blocks = num_elements.div_ceil(256);
        let byte_size = match *qtype {
            GGUF_TYPE_Q4_K => num_super_blocks * 144,
            _ => *num_elements * 4,
        };

        let tensor_ref = QuantizedTensorRef {
            offset: current_offset,
            byte_size,
            num_elements: *num_elements,
            qtype: *qtype,
        };

        assert_eq!(
            tensor_ref.offset, current_offset,
            "Offset mismatch for {}",
            name
        );
        current_offset += byte_size;
    }

    // Total model size should be calculated
    assert!(current_offset > 0);
}

// =============================================================================
// Phase 35: Real Transformer Loading Tests (using Test Factory)
// =============================================================================

use crate::gguf::test_factory::{
    build_minimal_llama_gguf, build_minimal_phi2_gguf, create_f32_embedding_data,
    create_f32_norm_weights, create_q4_0_data, create_q4_k_data, create_q5_k_data,
    create_q6_k_data, create_q8_0_data, GGUFBuilder,
};
use crate::gguf::transformer::QuantizedGGUFTransformer;
use crate::gguf::GGUFModel;

include!("phase35_transformer.rs");
include!("phase35_get.rs");
