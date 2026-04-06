//! T-COV-95 Phase 53: AprTransformer::from_apr_bytes dtype coverage
//!
//! Covers the uncovered dtype dispatch branches in the `get_f32_tensor` closure
//! within `from_apr_bytes`: Q4_K(12), Q5_K(13), Q6_K(14), Q8_0(8), F16(1).
//! Also covers weight tying, GGUF naming, metadata aliases, and Q4K raw bytes paths.

use crate::apr_transformer::AprTransformer;

// APR v2 constants
const MAGIC: [u8; 4] = [0x41, 0x50, 0x52, 0x00]; // "APR\0"
const HEADER_SIZE: usize = 64;

// GGML dtype constants
const DTYPE_F32: u8 = 0;
const DTYPE_F16: u8 = 1;
const DTYPE_Q8_0: u8 = 8;
const DTYPE_Q4_K: u8 = 12;
const DTYPE_Q5_K: u8 = 13;
const DTYPE_Q6_K: u8 = 14;

/// Tensor definition for building synthetic APR v2 binary data
struct TensorDef {
    name: String,
    dtype: u8,
    dims: Vec<u64>,
    data: Vec<u8>,
}

/// Build APR v2 binary data from metadata JSON and tensor definitions.
fn build_apr_v2(metadata_json: &str, tensors: &[TensorDef]) -> Vec<u8> {
    let metadata_bytes = metadata_json.as_bytes();
    let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

    // Build tensor index entries
    let mut index_bytes = Vec::new();
    let mut current_offset = 0u64;

    for t in tensors {
        // name_len (2) + name + dtype (1) + ndim (1) + dims (8 each) + offset (8) + size (8)
        index_bytes.extend_from_slice(&(t.name.len() as u16).to_le_bytes());
        index_bytes.extend_from_slice(t.name.as_bytes());
        index_bytes.push(t.dtype);
        index_bytes.push(t.dims.len() as u8);
        for &dim in &t.dims {
            index_bytes.extend_from_slice(&dim.to_le_bytes());
        }
        index_bytes.extend_from_slice(&current_offset.to_le_bytes());
        index_bytes.extend_from_slice(&(t.data.len() as u64).to_le_bytes());
        current_offset += t.data.len() as u64;
    }

    let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
    let data_offset = tensor_index_offset + index_bytes.len() as u64;
    let total_data_size: usize = tensors.iter().map(|t| t.data.len()).sum();
    let total_size = data_offset as usize + total_data_size;

    let mut buf = vec![0u8; total_size];

    // Write header
    buf[0..4].copy_from_slice(&MAGIC);
    buf[4] = 2; // version major
    buf[5] = 0;
    buf[8..12].copy_from_slice(&(tensors.len() as u32).to_le_bytes());
    buf[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    buf[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    buf[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    buf[32..40].copy_from_slice(&data_offset.to_le_bytes());

    // Write metadata
    buf[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

    // Write tensor index
    let idx_start = tensor_index_offset as usize;
    buf[idx_start..idx_start + index_bytes.len()].copy_from_slice(&index_bytes);

    // Write tensor data
    let mut pos = data_offset as usize;
    for t in tensors {
        buf[pos..pos + t.data.len()].copy_from_slice(&t.data);
        pos += t.data.len();
    }

    buf
}

/// Create F32 tensor data filled with a constant value
fn make_f32_data(num_elements: usize, value: f32) -> Vec<u8> {
    let mut data = Vec::with_capacity(num_elements * 4);
    for _ in 0..num_elements {
        data.extend_from_slice(&value.to_le_bytes());
    }
    data
}

/// Create F16 tensor data: each element is f16 encoding of `value`
fn make_f16_data(num_elements: usize, value: f32) -> Vec<u8> {
    let bits = half::f16::from_f32(value).to_bits();
    let mut data = Vec::with_capacity(num_elements * 2);
    for _ in 0..num_elements {
        data.extend_from_slice(&bits.to_le_bytes());
    }
    data
}

/// Create Q8_0 tensor data: 34 bytes per block (2 f16 scale + 32 i8 quants), 32 elements/block
fn make_q8_0_data(num_elements: usize) -> Vec<u8> {
    let num_blocks = num_elements.div_ceil(32);
    let mut data = Vec::with_capacity(num_blocks * 34);
    for _ in 0..num_blocks {
        // scale = f16(1.0)
        data.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        // 32 quants, all 1
        data.extend_from_slice(&[1i8 as u8; 32]);
    }
    data
}

/// Create Q4_K tensor data: 144 bytes per block, 256 elements/block
fn make_q4k_data(num_elements: usize) -> Vec<u8> {
    let num_blocks = num_elements.div_ceil(256);
    // Each block: 144 bytes (12 scale bytes + 4 min bytes + 128 quant bytes)
    // Just fill with zeros — produces zeros when dequantized, but exercises the code path
    vec![0u8; num_blocks * 144]
}

/// Create Q6_K tensor data: 210 bytes per block, 256 elements/block
fn make_q6k_data(num_elements: usize) -> Vec<u8> {
    let num_blocks = num_elements.div_ceil(256);
    // Each block: 210 bytes
    vec![0u8; num_blocks * 210]
}

/// Standard metadata JSON for a tiny model
fn minimal_metadata(
    hidden: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    vocab: usize,
    intermediate: usize,
) -> String {
    format!(
        r#"{{
        "architecture": "llama",
        "hidden_size": {hidden},
        "num_hidden_layers": {layers},
        "num_attention_heads": {heads},
        "num_key_value_heads": {kv_heads},
        "vocab_size": {vocab},
        "intermediate_size": {intermediate},
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "context_length": 512
    }}"#
    )
}

/// Build a minimal tensor set using HF naming with a given dtype for the weight tensors.
/// Returns a TensorDef vec suitable for build_apr_v2.
///
/// embedding and lm_head use `embed_dtype` and `lm_head_dtype` respectively.
/// layer weight tensors use `weight_dtype`.
/// norm tensors always use F32.
fn make_hf_tensors(
    hidden: usize,
    intermediate: usize,
    heads: usize,
    kv_heads: usize,
    vocab: usize,
    embed_dtype: u8,
    weight_dtype: u8,
    lm_head_dtype: u8,
) -> Vec<TensorDef> {
    let head_dim = hidden / heads;
    let kv_dim = kv_heads * head_dim;
    let qkv_out = hidden + 2 * kv_dim;

    let make_data = |dtype: u8, num_elements: usize| -> Vec<u8> {
        match dtype {
            DTYPE_F16 => make_f16_data(num_elements, 0.01),
            DTYPE_Q8_0 => make_q8_0_data(num_elements),
            DTYPE_Q4_K | DTYPE_Q5_K => make_q4k_data(num_elements),
            DTYPE_Q6_K => make_q6k_data(num_elements),
            _ => make_f32_data(num_elements, 0.01), // F32 default
        }
    };

    vec![
        TensorDef {
            name: "model.embed_tokens.weight".into(),
            dtype: embed_dtype,
            dims: vec![vocab as u64, hidden as u64],
            data: make_data(embed_dtype, vocab * hidden),
        },
        TensorDef {
            name: "model.layers.0.input_layernorm.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "model.layers.0.self_attn.q_proj.weight".into(),
            dtype: weight_dtype,
            dims: vec![hidden as u64, hidden as u64],
            data: make_data(weight_dtype, hidden * hidden),
        },
        TensorDef {
            name: "model.layers.0.self_attn.k_proj.weight".into(),
            dtype: weight_dtype,
            dims: vec![kv_dim as u64, hidden as u64],
            data: make_data(weight_dtype, kv_dim * hidden),
        },
        TensorDef {
            name: "model.layers.0.self_attn.v_proj.weight".into(),
            dtype: weight_dtype,
            dims: vec![kv_dim as u64, hidden as u64],
            data: make_data(weight_dtype, kv_dim * hidden),
        },
        TensorDef {
            name: "model.layers.0.self_attn.o_proj.weight".into(),
            dtype: weight_dtype,
            dims: vec![hidden as u64, hidden as u64],
            data: make_data(weight_dtype, hidden * hidden),
        },
        TensorDef {
            name: "model.layers.0.post_attention_layernorm.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "model.layers.0.mlp.gate_proj.weight".into(),
            dtype: weight_dtype,
            dims: vec![intermediate as u64, hidden as u64],
            data: make_data(weight_dtype, intermediate * hidden),
        },
        TensorDef {
            name: "model.layers.0.mlp.up_proj.weight".into(),
            dtype: weight_dtype,
            dims: vec![intermediate as u64, hidden as u64],
            data: make_data(weight_dtype, intermediate * hidden),
        },
        TensorDef {
            name: "model.layers.0.mlp.down_proj.weight".into(),
            dtype: weight_dtype,
            dims: vec![hidden as u64, intermediate as u64],
            data: make_data(weight_dtype, hidden * intermediate),
        },
        TensorDef {
            name: "model.norm.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "lm_head.weight".into(),
            dtype: lm_head_dtype,
            dims: vec![vocab as u64, hidden as u64],
            data: make_data(lm_head_dtype, vocab * hidden),
        },
    ]
}

// ============================================================================
// F16 dtype branch coverage
// ============================================================================

#[test]
fn test_from_apr_bytes_f16_embedding_and_lm_head() {
    let hidden = 8;
    let intermediate = 32;
    let vocab = 16;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F16,
        DTYPE_F32,
        DTYPE_F16,
    );
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(
        result.is_ok(),
        "F16 embedding+lm_head: {}",
        result.unwrap_err()
    );

    let apr = result.expect("test value should be present");
    assert_eq!(apr.token_embedding.len(), vocab * hidden);
    assert_eq!(apr.lm_head_weight.len(), vocab * hidden);
    // F16(0.01) -> F32 should be approximately 0.01
    assert!((apr.token_embedding[0] - 0.01).abs() < 0.002);
}

// ============================================================================
// Q8_0 dtype branch coverage
// ============================================================================

#[test]
fn test_from_apr_bytes_q8_0_weights() {
    let hidden = 32; // must be multiple of 32 for Q8_0 blocks
    let intermediate = 64;
    let vocab = 16;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_Q8_0,
        DTYPE_F32,
    );
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Q8_0 weights: {}", result.unwrap_err());

    let apr = result.expect("test value should be present");
    assert_eq!(apr.layers.len(), 1);
    // Q8_0 dequantized weights should be 1.0 (scale=1.0, quant=1)
    assert!((apr.layers[0].qkv_weight[0] - 1.0).abs() < 0.01);
}

// ============================================================================
// Q4_K dtype branch coverage (flat path — dims divisible by 256)
// ============================================================================

#[test]
fn test_from_apr_bytes_q4k_flat_weights() {
    let hidden = 256; // multiple of 256 for Q4_K flat path
    let intermediate = 256;
    let vocab = 4;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_Q4_K,
        DTYPE_Q4_K,
    );
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Q4_K flat: {}", result.unwrap_err());

    let apr = result.expect("test value should be present");
    // Q4K weights loaded — also check q4k_layers populated
    assert!(
        apr.q4k_layers.is_some(),
        "Q4K raw bytes should be extracted"
    );
    let q4k = apr.q4k_layers.as_ref().expect("test value should be present");
    assert_eq!(q4k.len(), 1);
    assert!(q4k[0].attn_q_weight.is_some());
    assert!(q4k[0].ffn_gate_weight.is_some());
}

// ============================================================================
// Q5_K dtype branch coverage (flat path — uses Q4_K dequant)
// ============================================================================

#[test]
fn test_from_apr_bytes_q5k_flat_weights() {
    let hidden = 256;
    let intermediate = 256;
    let vocab = 4;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_Q5_K,
        DTYPE_F32,
    );
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Q5_K flat: {}", result.unwrap_err());

    let apr = result.expect("test value should be present");
    assert_eq!(apr.layers.len(), 1);
    // Q5_K should also produce q4k raw bytes (dtype 13 accepted by get_q4k_raw_bytes)
    assert!(apr.q4k_layers.is_some());
}

// ============================================================================
// Q6_K dtype branch coverage (flat path)
// ============================================================================

#[test]
fn test_from_apr_bytes_q6k_flat_weights() {
    let hidden = 256;
    let intermediate = 256;
    let vocab = 4;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    // Use Q6_K for down_proj specifically (tests q6k_ffn_down path)
    let mut tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_F32,
    );
    // Override specific tensors to Q6_K
    for t in &mut tensors {
        if t.name.contains("down_proj") || t.name.contains("up_proj") || t.name.contains("v_proj") {
            let num_elements: usize = t.dims.iter().map(|d| *d as usize).product();
            t.dtype = DTYPE_Q6_K;
            t.data = make_q6k_data(num_elements);
        }
    }
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Q6_K flat: {}", result.unwrap_err());

    let apr = result.expect("test value should be present");
    // Q6K tensors should produce q4k_layers with q6k fields populated
    assert!(apr.q4k_layers.is_some());
    let q4k = apr.q4k_layers.as_ref().expect("test value should be present");
    assert!(q4k[0].ffn_down_weight_q6k.is_some());
    assert!(q4k[0].ffn_up_weight_q6k.is_some());
    assert!(q4k[0].attn_v_weight_q6k.is_some());
}

include!("apr_03.rs");
include!("apr_02_02.rs");
include!("forward_02.rs");
