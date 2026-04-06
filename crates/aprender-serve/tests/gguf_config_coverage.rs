//! EXTREME TDD: GGUFConfig and InferenceScratchBuffer Coverage Tests

use realizar::gguf::{
    ArchConstraints, GGUFConfig, GGUFModel, InferenceScratchBuffer, OwnedInferenceScratchBuffer,
    OwnedQKVWeights, OwnedQuantizedTensor,
};

/// Helper to build GGUF model with architecture metadata
fn build_gguf_with_arch(arch: &str, hidden: usize, layers: usize, heads: usize) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&4u64.to_le_bytes()); // metadata_count = 4

    // general.architecture
    let key1 = "general.architecture";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1.as_bytes());
    data.extend_from_slice(&8u32.to_le_bytes()); // String
    data.extend_from_slice(&(arch.len() as u64).to_le_bytes());
    data.extend_from_slice(arch.as_bytes());

    // {arch}.embedding_length
    let key2 = format!("{arch}.embedding_length");
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes()); // UInt32
    data.extend_from_slice(&(hidden as u32).to_le_bytes());

    // {arch}.block_count
    let key3 = format!("{arch}.block_count");
    data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
    data.extend_from_slice(key3.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes()); // UInt32
    data.extend_from_slice(&(layers as u32).to_le_bytes());

    // {arch}.attention.head_count
    let key4 = format!("{arch}.attention.head_count");
    data.extend_from_slice(&(key4.len() as u64).to_le_bytes());
    data.extend_from_slice(key4.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes()); // UInt32
    data.extend_from_slice(&(heads as u32).to_le_bytes());

    data
}

// ===== GGUFConfig Tests =====

#[test]
fn test_cov_gguf_config_basic() {
    let data = build_gguf_with_arch("llama", 2048, 22, 32);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let config = GGUFConfig::from_gguf(&model).expect("config");

    assert_eq!(config.architecture, "llama");
    assert_eq!(config.hidden_dim, 2048);
    assert_eq!(config.num_layers, 22);
    assert_eq!(config.num_heads, 32);
}

#[test]
fn test_cov_gguf_config_phi2() {
    let data = build_gguf_with_arch("phi2", 2560, 32, 32);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let config = GGUFConfig::from_gguf(&model).expect("config");

    assert_eq!(config.architecture, "phi2");
    assert_eq!(config.hidden_dim, 2560);
}

#[test]
fn test_cov_gguf_config_qwen2() {
    let data = build_gguf_with_arch("qwen2", 896, 24, 14);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let config = GGUFConfig::from_gguf(&model).expect("config");

    assert_eq!(config.architecture, "qwen2");
    assert_eq!(config.num_heads, 14);
}

#[test]
fn test_cov_gguf_config_defaults() {
    // Test default values when optional fields missing
    let data = build_gguf_with_arch("test", 1024, 12, 16);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let config = GGUFConfig::from_gguf(&model).expect("config");

    // Defaults
    assert_eq!(config.num_kv_heads, 16); // defaults to num_heads
    assert_eq!(config.context_length, 2048); // default
    assert!((config.rope_theta - 10000.0).abs() < 1.0); // default
    assert!((config.eps - 1e-5).abs() < 1e-7); // default
    assert_eq!(config.rope_type, 0); // default NORM
}

#[test]
fn test_cov_gguf_config_full_options() {
    // Build GGUF with all optional fields manually (simpler than using complex helper)
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    data.extend_from_slice(&8u64.to_le_bytes()); // metadata_count = 8

    fn add_string_meta(data: &mut Vec<u8>, key: &str, value: &str) {
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&8u32.to_le_bytes()); // String type
        data.extend_from_slice(&(value.len() as u64).to_le_bytes());
        data.extend_from_slice(value.as_bytes());
    }
    fn add_u32_meta(data: &mut Vec<u8>, key: &str, value: u32) {
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&4u32.to_le_bytes()); // UInt32 type
        data.extend_from_slice(&value.to_le_bytes());
    }
    fn add_f32_meta(data: &mut Vec<u8>, key: &str, value: f32) {
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&6u32.to_le_bytes()); // Float32 type
        data.extend_from_slice(&value.to_le_bytes());
    }

    add_string_meta(&mut data, "general.architecture", "llama");
    add_u32_meta(&mut data, "llama.embedding_length", 2048);
    add_u32_meta(&mut data, "llama.block_count", 22);
    add_u32_meta(&mut data, "llama.attention.head_count", 32);
    add_u32_meta(&mut data, "llama.attention.head_count_kv", 8);
    add_u32_meta(&mut data, "llama.context_length", 4096);
    add_f32_meta(&mut data, "llama.rope.freq_base", 100000.0);
    add_f32_meta(&mut data, "llama.attention.layer_norm_rms_epsilon", 1e-6);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let config = GGUFConfig::from_gguf(&model).expect("config");

    assert_eq!(config.num_kv_heads, 8);
    assert_eq!(config.context_length, 4096);
    assert!((config.rope_theta - 100000.0).abs() < 1.0);
    assert!((config.eps - 1e-6).abs() < 1e-8);
    // rope_type is inferred from architecture name ("llama" -> 0, "qwen" -> 2)
    assert_eq!(config.rope_type, 0); // llama uses NORM style
}

#[test]
fn test_cov_gguf_config_vocab_from_tensor() {
    // Build a GGUF with a token_embd.weight tensor to infer vocab_size
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
    data.extend_from_slice(&3u64.to_le_bytes()); // metadata_count = 3

    // Metadata
    fn add_string_meta(data: &mut Vec<u8>, key: &str, value: &str) {
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&8u32.to_le_bytes());
        data.extend_from_slice(&(value.len() as u64).to_le_bytes());
        data.extend_from_slice(value.as_bytes());
    }
    fn add_u32_meta(data: &mut Vec<u8>, key: &str, value: u32) {
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&4u32.to_le_bytes());
        data.extend_from_slice(&value.to_le_bytes());
    }

    add_string_meta(&mut data, "general.architecture", "llama");
    add_u32_meta(&mut data, "llama.embedding_length", 512);
    add_u32_meta(&mut data, "llama.block_count", 4);

    // Tensor: token_embd.weight with dims [512, 1000] in GGML order
    // After reversal: [1000, 512], so vocab_size = 1000
    let name = "token_embd.weight";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&2u32.to_le_bytes()); // n_dims
    data.extend_from_slice(&512u64.to_le_bytes()); // dim[0]
    data.extend_from_slice(&1000u64.to_le_bytes()); // dim[1]
    data.extend_from_slice(&0u32.to_le_bytes()); // qtype = F32
    data.extend_from_slice(&0u64.to_le_bytes()); // offset

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let config = GGUFConfig::from_gguf(&model).expect("config");

    // vocab_size should be inferred from token_embd.weight tensor
    assert_eq!(config.vocab_size, 1000);
}

#[test]
fn test_cov_gguf_config_intermediate_from_tensor() {
    // Build a GGUF with a blk.0.ffn_up.weight tensor to infer intermediate_dim
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
    data.extend_from_slice(&3u64.to_le_bytes()); // metadata_count = 3

    fn add_string_meta(data: &mut Vec<u8>, key: &str, value: &str) {
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&8u32.to_le_bytes());
        data.extend_from_slice(&(value.len() as u64).to_le_bytes());
        data.extend_from_slice(value.as_bytes());
    }
    fn add_u32_meta(data: &mut Vec<u8>, key: &str, value: u32) {
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&4u32.to_le_bytes());
        data.extend_from_slice(&value.to_le_bytes());
    }

    add_string_meta(&mut data, "general.architecture", "llama");
    add_u32_meta(&mut data, "llama.embedding_length", 512);
    add_u32_meta(&mut data, "llama.block_count", 4);

    // Tensor: blk.0.ffn_up.weight with dims [512, 2048] in GGML order
    // After reversal: [2048, 512], so intermediate_dim = 2048
    let name = "blk.0.ffn_up.weight";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&2u32.to_le_bytes()); // n_dims
    data.extend_from_slice(&512u64.to_le_bytes()); // dim[0]
    data.extend_from_slice(&2048u64.to_le_bytes()); // dim[1]
    data.extend_from_slice(&0u32.to_le_bytes()); // qtype = F32
    data.extend_from_slice(&0u64.to_le_bytes()); // offset

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let config = GGUFConfig::from_gguf(&model).expect("config");

    // intermediate_dim should be inferred from ffn_up tensor
    assert_eq!(config.intermediate_dim, 2048);
}

#[test]
fn test_cov_gguf_config_missing_architecture() {
    // Build GGUF without general.architecture
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count = 0

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let result = GGUFConfig::from_gguf(&model);
    assert!(result.is_err());
}

#[test]
fn test_cov_gguf_config_missing_embedding_length() {
    // Build GGUF with architecture but no embedding_length
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

    // Only add architecture
    let key = "general.architecture";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    let arch = "llama";
    data.extend_from_slice(&(arch.len() as u64).to_le_bytes());
    data.extend_from_slice(arch.as_bytes());

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let result = GGUFConfig::from_gguf(&model);
    assert!(result.is_err());
}

// ===== InferenceScratchBuffer Tests =====

#[test]
fn test_cov_scratch_buffer_from_config() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 2048,
        num_layers: 22,
        num_heads: 32,
        num_kv_heads: 8,
        vocab_size: 32000,
        intermediate_dim: 8192,
        context_length: 2048,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let scratch = InferenceScratchBuffer::from_config(&config);

    assert_eq!(scratch.hidden.len(), 2048);
    assert_eq!(scratch.normed.len(), 2048);
    assert_eq!(scratch.qkv.len(), 2048 * 3); // Q + K + V
    assert_eq!(scratch.q.len(), 2048);
    assert_eq!(scratch.k.len(), 2048);
    assert_eq!(scratch.v.len(), 2048);
    assert_eq!(scratch.attn_out.len(), 2048);
    assert_eq!(scratch.attn_proj.len(), 2048);
    assert_eq!(scratch.ffn_up.len(), 8192);
    assert_eq!(scratch.ffn_gate.len(), 8192);
    assert_eq!(scratch.ffn_down.len(), 2048);
    assert_eq!(scratch.logits.len(), 32000);
}

#[test]
fn test_cov_scratch_buffer_q8k_buffers() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 2048,
        num_layers: 22,
        num_heads: 32,
        num_kv_heads: 8,
        vocab_size: 32000,
        intermediate_dim: 8192,
        context_length: 2048,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let scratch = InferenceScratchBuffer::from_config(&config);

    // Q8K uses 256-element super-blocks
    // hidden_dim = 2048 -> 2048/256 = 8 scale values
    assert_eq!(scratch.q8k_hidden_scales.len(), 8);
    assert_eq!(scratch.q8k_hidden_quants.len(), 2048);

    // intermediate_dim = 8192 -> 8192/256 = 32 scale values
    assert_eq!(scratch.q8k_inter_scales.len(), 32);
    assert_eq!(scratch.q8k_inter_quants.len(), 8192);
}

#[test]
fn test_cov_scratch_buffer_reset() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 1000,
        intermediate_dim: 512,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let mut scratch = InferenceScratchBuffer::from_config(&config);

    // Fill with non-zero values
    for v in &mut scratch.hidden {
        *v = 1.0;
    }
    for v in &mut scratch.normed {
        *v = 2.0;
    }

    scratch.reset();

    // hidden and normed should be zeroed
    assert!(scratch.hidden.iter().all(|&x| x == 0.0));
    assert!(scratch.normed.iter().all(|&x| x == 0.0));
}

#[test]
fn test_cov_scratch_buffer_small_model() {
    let config = GGUFConfig {
        architecture: "tiny".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 100,
        intermediate_dim: 256,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let scratch = InferenceScratchBuffer::from_config(&config);

    assert_eq!(scratch.hidden.len(), 64);
    assert_eq!(scratch.logits.len(), 100);
    assert_eq!(scratch.ffn_up.len(), 256);
}

// ===== OwnedInferenceScratchBuffer Tests =====

#[test]
fn test_cov_owned_scratch_buffer_from_config() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 2048,
        num_layers: 22,
        num_heads: 32,
        num_kv_heads: 8,
        vocab_size: 32000,
        intermediate_dim: 8192,
        context_length: 2048,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };
    let scratch = OwnedInferenceScratchBuffer::from_config(&config);

    // OwnedInferenceScratchBuffer has different fields than InferenceScratchBuffer
    assert_eq!(scratch.attn_out.len(), 2048);
    assert_eq!(scratch.ffn_down.len(), 2048);
    assert_eq!(scratch.logits.len(), 32000);
    // ffn_up and ffn_gate use conservative estimate (hidden * 6)
    assert!(scratch.ffn_up.len() >= 8192);
    assert!(scratch.ffn_gate.len() >= 8192);
}

#[test]
fn test_cov_owned_scratch_buffer_qkv_size() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 2048,
        num_layers: 22,
        num_heads: 32,
        num_kv_heads: 8, // GQA: 8 kv heads vs 32 q heads
        vocab_size: 32000,
        intermediate_dim: 8192,
        context_length: 2048,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };
    let scratch = OwnedInferenceScratchBuffer::from_config(&config);

    // qkv_dim = hidden_dim + 2 * kv_dim
    // head_dim = 2048 / 32 = 64
    // kv_dim = 8 * 64 = 512
    // qkv_dim = 2048 + 2 * 512 = 3072
    assert_eq!(scratch.qkv.len(), 3072);
}

#[test]
fn test_cov_owned_scratch_buffer_reset() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 1000,
        intermediate_dim: 512,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };
    let mut scratch = OwnedInferenceScratchBuffer::from_config(&config);

    // Fill with values
    scratch.qkv.iter_mut().for_each(|x| *x = 5.0);
    scratch.attn_out.iter_mut().for_each(|x| *x = 3.0);

    scratch.reset();

    // reset() calls clear() on all buffers which sets len to 0 but preserves capacity
    assert_eq!(scratch.attn_out.len(), 0);
    assert!(scratch.attn_out.capacity() >= 128); // Capacity preserved
    assert_eq!(scratch.qkv.len(), 0);
    assert!(scratch.qkv.capacity() > 0);
}

#[test]
fn test_cov_owned_scratch_buffer_q8k_buffers() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 2048,
        num_layers: 22,
        num_heads: 32,
        num_kv_heads: 8,
        vocab_size: 32000,
        intermediate_dim: 8192,
        context_length: 2048,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };
    let scratch = OwnedInferenceScratchBuffer::from_config(&config);

    // Q8K uses 256-element super-blocks
    // hidden_dim = 2048 -> 2048/256 = 8 scale values
    assert_eq!(scratch.q8k_hidden_scales.len(), 8);
    assert_eq!(scratch.q8k_hidden_quants.len(), 2048);

    // intermediate uses hidden * 6 = 12288, padded to 256
    // 12288 / 256 = 48
    assert_eq!(scratch.q8k_inter_scales.len(), 48);
}

// ===== OwnedQuantizedTensor Tests =====

#[test]
fn test_cov_owned_quantized_tensor_basic() {
    let tensor = OwnedQuantizedTensor {
        data: vec![1, 2, 3, 4],
        in_dim: 128,
        out_dim: 256,
        qtype: 2, // Q4_0
    };

    assert_eq!(tensor.data.len(), 4);
    assert_eq!(tensor.in_dim, 128);
    assert_eq!(tensor.out_dim, 256);
    assert_eq!(tensor.qtype, 2);
}

#[test]
fn test_cov_owned_quantized_tensor_clone() {
    let tensor = OwnedQuantizedTensor {
        data: vec![10, 20, 30],
        in_dim: 64,
        out_dim: 128,
        qtype: 8, // Q8_0
    };

    let cloned = tensor.clone();

    assert_eq!(cloned.data, tensor.data);
    assert_eq!(cloned.in_dim, tensor.in_dim);
    assert_eq!(cloned.out_dim, tensor.out_dim);
    assert_eq!(cloned.qtype, tensor.qtype);
}

// ===== OwnedQKVWeights Tests =====

#[test]
fn test_cov_owned_qkv_weights_fused() {
    let fused = OwnedQuantizedTensor {
        data: vec![0; 100],
        in_dim: 2048,
        out_dim: 6144, // 3 * 2048
        qtype: 12,     // Q4_K
    };

    let qkv = OwnedQKVWeights::Fused(fused);

    assert_eq!(qkv.out_dim(), 6144);
}

#[test]
fn test_cov_owned_qkv_weights_separate() {
    let q = OwnedQuantizedTensor {
        data: vec![0; 50],
        in_dim: 2048,
        out_dim: 2048,
        qtype: 12,
    };
    let k = OwnedQuantizedTensor {
        data: vec![0; 25],
        in_dim: 2048,
        out_dim: 512, // GQA: fewer KV heads
        qtype: 12,
    };
    let v = OwnedQuantizedTensor {
        data: vec![0; 25],
        in_dim: 2048,
        out_dim: 512,
        qtype: 12,
    };

    let qkv = OwnedQKVWeights::Separate { q, k, v };

    assert_eq!(qkv.out_dim(), 2048 + 512 + 512);
}

#[test]
fn test_cov_owned_qkv_weights_clone() {
    let fused = OwnedQuantizedTensor {
        data: vec![1, 2, 3],
        in_dim: 128,
        out_dim: 384,
        qtype: 2,
    };
    let qkv = OwnedQKVWeights::Fused(fused);
    let cloned = qkv.clone();

    assert_eq!(qkv.out_dim(), cloned.out_dim());
}
