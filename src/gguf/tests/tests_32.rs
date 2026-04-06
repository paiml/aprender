//! T-COV-95 Menagerie of Pygmies: Complex Structural GGUF Generators (PMAT-802)
//!
//! Dr. Popper's directive: "Stop adding tests; start generating *data*."
//!
//! This module creates structurally complex GGUF files that force the converter
//! and loader to execute their loops, not just their headers:
//! - 100+ tensors with specific names ("attn_q", "ffn_gate")
//! - Mixed quantization types (Q4_K + Q8_0 + F32)
//! - Multiple layers (4-8)
//! - Sharding metadata
//!
//! Target: convert/mod.rs (234 missed), gguf/loader.rs (618 missed)

use crate::gguf::{
    GGUFModel, GGUF_ALIGNMENT, GGUF_MAGIC, GGUF_TYPE_F32, GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_K,
    GGUF_TYPE_Q8_0, GGUF_VERSION_V3,
};

// ============================================================================
// Complex Pygmy Builder - Multi-Layer, Mixed Quantization
// ============================================================================

/// Build a Complex Pygmy with multiple layers and mixed quantization
/// This is the core generator for the Menagerie
fn build_complex_pygmy(
    num_layers: usize,
    hidden_dim: usize,
    vocab_size: usize,
    intermediate_dim: usize,
) -> Vec<u8> {
    let mut data = Vec::new();

    // Calculate tensor count:
    // - 1 token_embedding (F32)
    // - Per layer: 10 tensors (attn_norm, q, k, v, output, ffn_norm, gate, up, down, + 1 bias)
    // - 2 output tensors (output_norm, lm_head)
    let tensors_per_layer = 10;
    let global_tensors = 3; // token_embd, output_norm, lm_head
    let tensor_count = num_layers * tensors_per_layer + global_tensors;

    // Metadata: architecture + config
    let metadata = vec![
        // general.architecture
        (
            "general.architecture".to_string(),
            build_gguf_string("llama"),
        ),
        // llama.embedding_length
        (
            "llama.embedding_length".to_string(),
            build_gguf_u32(hidden_dim as u32),
        ),
        // llama.block_count
        (
            "llama.block_count".to_string(),
            build_gguf_u32(num_layers as u32),
        ),
        // llama.attention.head_count
        ("llama.attention.head_count".to_string(), build_gguf_u32(4)),
        // llama.attention.head_count_kv
        (
            "llama.attention.head_count_kv".to_string(),
            build_gguf_u32(4),
        ),
        // llama.context_length
        ("llama.context_length".to_string(), build_gguf_u32(512)),
        // llama.rope.freq_base
        ("llama.rope.freq_base".to_string(), build_gguf_f32(10000.0)),
        // llama.attention.layer_norm_rms_epsilon
        (
            "llama.attention.layer_norm_rms_epsilon".to_string(),
            build_gguf_f32(1e-5),
        ),
        // llama.feed_forward_length
        (
            "llama.feed_forward_length".to_string(),
            build_gguf_u32(intermediate_dim as u32),
        ),
    ];

    // Header
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&(tensor_count as u64).to_le_bytes());
    data.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

    // Write metadata
    for (key, value) in &metadata {
        // Key: length + bytes
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        // Value type + value
        data.extend_from_slice(value);
    }

    // Collect tensor info and data
    let mut tensor_infos = Vec::new();
    let mut tensor_data_list = Vec::new();
    let mut offset = 0usize;

    // Token embedding - F32
    let embed_size = vocab_size * hidden_dim;
    let embed_data = vec![0u8; embed_size * 4]; // F32
    tensor_infos.push(build_tensor_info(
        "token_embd.weight",
        &[vocab_size as u64, hidden_dim as u64],
        GGUF_TYPE_F32,
        offset as u64,
    ));
    offset += embed_data.len();
    tensor_data_list.push(embed_data);

    // Per-layer tensors
    for layer in 0..num_layers {
        let prefix = format!("blk.{layer}");

        // Attention norm - F32
        let norm_data = vec![0u8; hidden_dim * 4];
        tensor_infos.push(build_tensor_info(
            &format!("{prefix}.attn_norm.weight"),
            &[hidden_dim as u64],
            GGUF_TYPE_F32,
            offset as u64,
        ));
        offset += norm_data.len();
        tensor_data_list.push(norm_data);

        // Q projection - Q4_0 (layer % 3 == 0) or Q8_0 (layer % 3 == 1) or Q4_K (else)
        let q_size = hidden_dim * hidden_dim;
        let (q_type, q_data) = if layer % 3 == 0 {
            (GGUF_TYPE_Q4_0, create_q4_0_block_data(q_size))
        } else if layer % 3 == 1 {
            (GGUF_TYPE_Q8_0, create_q8_0_block_data(q_size))
        } else {
            (GGUF_TYPE_Q4_K, create_q4_k_block_data(q_size))
        };
        tensor_infos.push(build_tensor_info(
            &format!("{prefix}.attn_q.weight"),
            &[hidden_dim as u64, hidden_dim as u64],
            q_type,
            offset as u64,
        ));
        offset += q_data.len();
        tensor_data_list.push(q_data);

        // K projection - mixed types
        let kv_dim = hidden_dim; // Simplified
        let k_size = hidden_dim * kv_dim;
        let (k_type, k_data) = if layer % 2 == 0 {
            (GGUF_TYPE_Q4_0, create_q4_0_block_data(k_size))
        } else {
            (GGUF_TYPE_Q8_0, create_q8_0_block_data(k_size))
        };
        tensor_infos.push(build_tensor_info(
            &format!("{prefix}.attn_k.weight"),
            &[hidden_dim as u64, kv_dim as u64],
            k_type,
            offset as u64,
        ));
        offset += k_data.len();
        tensor_data_list.push(k_data);

        // V projection
        let (v_type, v_data) = (GGUF_TYPE_Q4_0, create_q4_0_block_data(k_size));
        tensor_infos.push(build_tensor_info(
            &format!("{prefix}.attn_v.weight"),
            &[hidden_dim as u64, kv_dim as u64],
            v_type,
            offset as u64,
        ));
        offset += v_data.len();
        tensor_data_list.push(v_data);

        // Output projection
        let out_size = hidden_dim * hidden_dim;
        let (out_type, out_data) = (GGUF_TYPE_Q4_0, create_q4_0_block_data(out_size));
        tensor_infos.push(build_tensor_info(
            &format!("{prefix}.attn_output.weight"),
            &[hidden_dim as u64, hidden_dim as u64],
            out_type,
            offset as u64,
        ));
        offset += out_data.len();
        tensor_data_list.push(out_data);

        // FFN norm - F32
        let ffn_norm_data = vec![0u8; hidden_dim * 4];
        tensor_infos.push(build_tensor_info(
            &format!("{prefix}.ffn_norm.weight"),
            &[hidden_dim as u64],
            GGUF_TYPE_F32,
            offset as u64,
        ));
        offset += ffn_norm_data.len();
        tensor_data_list.push(ffn_norm_data);

        // FFN gate - Q4_K for variety
        let ffn_size = hidden_dim * intermediate_dim;
        let (gate_type, gate_data) = (GGUF_TYPE_Q4_K, create_q4_k_block_data(ffn_size));
        tensor_infos.push(build_tensor_info(
            &format!("{prefix}.ffn_gate.weight"),
            &[hidden_dim as u64, intermediate_dim as u64],
            gate_type,
            offset as u64,
        ));
        offset += gate_data.len();
        tensor_data_list.push(gate_data);

        // FFN up
        let (up_type, up_data) = (GGUF_TYPE_Q4_0, create_q4_0_block_data(ffn_size));
        tensor_infos.push(build_tensor_info(
            &format!("{prefix}.ffn_up.weight"),
            &[hidden_dim as u64, intermediate_dim as u64],
            up_type,
            offset as u64,
        ));
        offset += up_data.len();
        tensor_data_list.push(up_data);

        // FFN down
        let (down_type, down_data) = (GGUF_TYPE_Q8_0, create_q8_0_block_data(ffn_size));
        tensor_infos.push(build_tensor_info(
            &format!("{prefix}.ffn_down.weight"),
            &[intermediate_dim as u64, hidden_dim as u64],
            down_type,
            offset as u64,
        ));
        offset += down_data.len();
        tensor_data_list.push(down_data);

        // Attention bias (optional, triggers bias paths)
        let bias_data = vec![0u8; hidden_dim * 4];
        tensor_infos.push(build_tensor_info(
            &format!("{prefix}.attn_q.bias"),
            &[hidden_dim as u64],
            GGUF_TYPE_F32,
            offset as u64,
        ));
        offset += bias_data.len();
        tensor_data_list.push(bias_data);
    }

    // Output norm - F32
    let out_norm_data = vec![0u8; hidden_dim * 4];
    tensor_infos.push(build_tensor_info(
        "output_norm.weight",
        &[hidden_dim as u64],
        GGUF_TYPE_F32,
        offset as u64,
    ));
    offset += out_norm_data.len();
    tensor_data_list.push(out_norm_data);

    // LM head - Q4_K
    let lm_head_size = hidden_dim * vocab_size;
    let (lm_type, lm_data) = (GGUF_TYPE_Q4_K, create_q4_k_block_data(lm_head_size));
    tensor_infos.push(build_tensor_info(
        "output.weight",
        &[hidden_dim as u64, vocab_size as u64],
        lm_type,
        offset as u64,
    ));
    tensor_data_list.push(lm_data);

    // Write tensor infos
    for info in tensor_infos {
        data.extend_from_slice(&info);
    }

    // Align to 32-byte boundary
    let current_len = data.len();
    let aligned = current_len.div_ceil(GGUF_ALIGNMENT) * GGUF_ALIGNMENT;
    data.resize(aligned, 0);

    // Write tensor data
    for tensor_data in tensor_data_list {
        data.extend_from_slice(&tensor_data);
    }

    data
}

// ============================================================================
// Helper Functions
// ============================================================================

fn build_gguf_string(s: &str) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&8u32.to_le_bytes()); // type = string
    data.extend_from_slice(&(s.len() as u64).to_le_bytes());
    data.extend_from_slice(s.as_bytes());
    data
}

fn build_gguf_u32(v: u32) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&4u32.to_le_bytes()); // type = u32
    data.extend_from_slice(&v.to_le_bytes());
    data
}

fn build_gguf_f32(v: f32) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&6u32.to_le_bytes()); // type = f32
    data.extend_from_slice(&v.to_le_bytes());
    data
}

fn build_tensor_info(name: &str, dims: &[u64], qtype: u32, offset: u64) -> Vec<u8> {
    let mut data = Vec::new();
    // Name
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    // n_dims
    data.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    // Dimensions (reversed for GGML)
    for &dim in dims.iter().rev() {
        data.extend_from_slice(&dim.to_le_bytes());
    }
    // qtype
    data.extend_from_slice(&qtype.to_le_bytes());
    // offset
    data.extend_from_slice(&offset.to_le_bytes());
    data
}

/// Q4_0: 18 bytes per 32 elements
fn create_q4_0_block_data(n_elements: usize) -> Vec<u8> {
    let n_blocks = n_elements.div_ceil(32);
    vec![0u8; n_blocks * 18]
}

/// Q8_0: 34 bytes per 32 elements
fn create_q8_0_block_data(n_elements: usize) -> Vec<u8> {
    let n_blocks = n_elements.div_ceil(32);
    vec![0u8; n_blocks * 34]
}

/// Q4_K: 144 bytes per 256 elements
fn create_q4_k_block_data(n_elements: usize) -> Vec<u8> {
    let n_blocks = n_elements.div_ceil(256);
    vec![0u8; n_blocks * 144]
}

// ============================================================================
// Tests: The Menagerie
// ============================================================================

#[test]
fn test_menagerie_4_layer_pygmy_parses() {
    let data = build_complex_pygmy(4, 64, 256, 128);

    let model = GGUFModel::from_bytes(&data);
    assert!(
        model.is_ok(),
        "4-layer Pygmy should parse: {:?}",
        model.err()
    );

    let model = model.expect("test value should be present");
    // 4 layers * 10 tensors + 3 global = 43 tensors
    assert!(
        model.tensors.len() >= 40,
        "Expected 40+ tensors, got {}",
        model.tensors.len()
    );
}

#[test]
fn test_menagerie_8_layer_pygmy_parses() {
    let data = build_complex_pygmy(8, 64, 256, 128);

    let model = GGUFModel::from_bytes(&data);
    assert!(
        model.is_ok(),
        "8-layer Pygmy should parse: {:?}",
        model.err()
    );

    let model = model.expect("test value should be present");
    // 8 layers * 10 tensors + 3 global = 83 tensors
    assert!(
        model.tensors.len() >= 80,
        "Expected 80+ tensors, got {}",
        model.tensors.len()
    );
}

#[test]
fn test_menagerie_mixed_quantization_types() {
    let data = build_complex_pygmy(6, 64, 256, 128);

    let model = GGUFModel::from_bytes(&data).expect("should parse");

    // Count quantization types
    let mut q4_0_count = 0;
    let mut q8_0_count = 0;
    let mut q4_k_count = 0;
    let mut f32_count = 0;

    for tensor in &model.tensors {
        match tensor.qtype {
            t if t == GGUF_TYPE_Q4_0 => q4_0_count += 1,
            t if t == GGUF_TYPE_Q8_0 => q8_0_count += 1,
            t if t == GGUF_TYPE_Q4_K => q4_k_count += 1,
            t if t == GGUF_TYPE_F32 => f32_count += 1,
            _ => {},
        }
    }

    // Verify mixed types
    assert!(q4_0_count > 0, "Should have Q4_0 tensors");
    assert!(q8_0_count > 0, "Should have Q8_0 tensors");
    assert!(q4_k_count > 0, "Should have Q4_K tensors");
    assert!(f32_count > 0, "Should have F32 tensors");
}

include!("menagerie_tensor.rs");
