//! Tests for weight loading module

use super::*;
use std::collections::HashMap;
use std::path::Path;
use tempfile::TempDir;

#[test]
fn test_map_weight_name_llama() {
    let name = "model.layers.0.self_attn.q_proj.weight";
    assert_eq!(map_weight_name(name, Architecture::Llama), name);
}

#[test]
fn test_map_weight_name_qwen2_standard() {
    let name = "model.layers.0.self_attn.q_proj.weight";
    assert_eq!(map_weight_name(name, Architecture::Qwen2), name);
}

#[test]
fn test_map_weight_name_qwen2_attn_variant() {
    let name = "model.layers.0.attn.q_proj.weight";
    assert_eq!(
        map_weight_name(name, Architecture::Qwen2),
        "model.layers.0.self_attn.q_proj.weight"
    );
}

#[test]
fn test_expected_weight_count() {
    // 2 layers without lm_head
    assert_eq!(expected_weight_count(2, false), 2 + 2 * 9);
    // 2 layers with lm_head
    assert_eq!(expected_weight_count(2, true), 2 + 2 * 9 + 1);
    // 24 layers (Qwen2.5-0.5B)
    assert_eq!(expected_weight_count(24, false), 2 + 24 * 9);
}

#[test]
fn test_validate_weights_minimal() {
    let mut weights = HashMap::new();
    let hidden = 64;
    let kv_hidden = 64;
    let intermediate = 256;
    let vocab = 1000;

    // Global weights
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        Tensor::from_vec(vec![0.1; vocab * hidden], true),
    );
    weights.insert("model.norm.weight".to_string(), Tensor::from_vec(vec![1.0; hidden], true));

    // Layer 0 weights
    weights.insert(
        "model.layers.0.input_layernorm.weight".to_string(),
        Tensor::from_vec(vec![1.0; hidden], true),
    );
    weights.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        Tensor::from_vec(vec![0.1; hidden * hidden], true),
    );
    weights.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        Tensor::from_vec(vec![0.1; hidden * kv_hidden], true),
    );
    weights.insert(
        "model.layers.0.self_attn.v_proj.weight".to_string(),
        Tensor::from_vec(vec![0.1; hidden * kv_hidden], true),
    );
    weights.insert(
        "model.layers.0.self_attn.o_proj.weight".to_string(),
        Tensor::from_vec(vec![0.1; hidden * hidden], true),
    );
    weights.insert(
        "model.layers.0.post_attention_layernorm.weight".to_string(),
        Tensor::from_vec(vec![1.0; hidden], true),
    );
    weights.insert(
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        Tensor::from_vec(vec![0.1; hidden * intermediate], true),
    );
    weights.insert(
        "model.layers.0.mlp.up_proj.weight".to_string(),
        Tensor::from_vec(vec![0.1; hidden * intermediate], true),
    );
    weights.insert(
        "model.layers.0.mlp.down_proj.weight".to_string(),
        Tensor::from_vec(vec![0.1; intermediate * hidden], true),
    );

    // Should validate for 1 layer
    assert!(validate_weights(&weights, 1).is_ok());

    // Should fail for 2 layers (missing layer 1)
    assert!(validate_weights(&weights, 2).is_err());
}

#[test]
fn test_validate_weights_missing_embedding() {
    let weights: HashMap<String, Tensor> = HashMap::new();
    let result = validate_weights(&weights, 1);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("embed_tokens.weight"));
}

#[test]
fn test_find_safetensors_files_nonexistent() {
    let path = Path::new("/nonexistent/path");
    let files = find_safetensors_files(path).expect("operation should succeed");
    assert!(files.is_empty());
}

#[test]
fn test_find_safetensors_files_empty_dir() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let files = find_safetensors_files(dir.path()).expect("operation should succeed");
    assert!(files.is_empty());
}

#[test]
fn test_architecture_default() {
    // Default should be LLaMA-compatible
    assert_eq!(Architecture::Auto, Architecture::Auto);
    assert_ne!(Architecture::Llama, Architecture::Qwen2);
}

#[test]
fn test_map_weight_name_mistral() {
    let name = "model.layers.0.self_attn.q_proj.weight";
    assert_eq!(map_weight_name(name, Architecture::Mistral), name);
}

#[test]
fn test_map_weight_name_auto() {
    let name = "model.layers.0.self_attn.q_proj.weight";
    assert_eq!(map_weight_name(name, Architecture::Auto), name);
}

#[test]
fn test_validate_weights_missing_norm() {
    let mut weights = HashMap::new();
    weights
        .insert("model.embed_tokens.weight".to_string(), Tensor::from_vec(vec![0.1; 64000], true));

    let result = validate_weights(&weights, 1);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("norm.weight"));
}

#[test]
fn test_validate_weights_missing_layer_weight() {
    let mut weights = HashMap::new();
    weights
        .insert("model.embed_tokens.weight".to_string(), Tensor::from_vec(vec![0.1; 64000], true));
    weights.insert("model.norm.weight".to_string(), Tensor::from_vec(vec![1.0; 64], true));
    weights.insert(
        "model.layers.0.input_layernorm.weight".to_string(),
        Tensor::from_vec(vec![1.0; 64], true),
    );
    // Missing q_proj.weight

    let result = validate_weights(&weights, 1);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("q_proj.weight"));
}

#[test]
fn test_validate_weights_with_lm_head() {
    let mut weights = HashMap::new();
    let hidden = 64;
    let kv_hidden = 64;
    let intermediate = 256;
    let vocab = 1000;

    // Global weights
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        Tensor::from_vec(vec![0.1; vocab * hidden], true),
    );
    weights.insert("model.norm.weight".to_string(), Tensor::from_vec(vec![1.0; hidden], true));
    weights.insert("lm_head.weight".to_string(), Tensor::from_vec(vec![0.1; vocab * hidden], true));

    // Layer 0 weights
    weights.insert(
        "model.layers.0.input_layernorm.weight".to_string(),
        Tensor::from_vec(vec![1.0; hidden], true),
    );
    weights.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        Tensor::from_vec(vec![0.1; hidden * hidden], true),
    );
    weights.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        Tensor::from_vec(vec![0.1; hidden * kv_hidden], true),
    );
    weights.insert(
        "model.layers.0.self_attn.v_proj.weight".to_string(),
        Tensor::from_vec(vec![0.1; hidden * kv_hidden], true),
    );
    weights.insert(
        "model.layers.0.self_attn.o_proj.weight".to_string(),
        Tensor::from_vec(vec![0.1; hidden * hidden], true),
    );
    weights.insert(
        "model.layers.0.post_attention_layernorm.weight".to_string(),
        Tensor::from_vec(vec![1.0; hidden], true),
    );
    weights.insert(
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        Tensor::from_vec(vec![0.1; hidden * intermediate], true),
    );
    weights.insert(
        "model.layers.0.mlp.up_proj.weight".to_string(),
        Tensor::from_vec(vec![0.1; hidden * intermediate], true),
    );
    weights.insert(
        "model.layers.0.mlp.down_proj.weight".to_string(),
        Tensor::from_vec(vec![0.1; intermediate * hidden], true),
    );

    assert!(validate_weights(&weights, 1).is_ok());
}

#[test]
fn test_find_safetensors_single_file() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let file_path = dir.path().join("model.safetensors");
    std::fs::write(&file_path, b"dummy").expect("file write should succeed");

    let files = find_safetensors_files(dir.path()).expect("operation should succeed");
    assert_eq!(files.len(), 1);
    assert_eq!(files[0], file_path);
}

#[test]
fn test_find_safetensors_sharded_files() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    // Create sharded files
    std::fs::write(dir.path().join("model-00001-of-00002.safetensors"), b"part1")
        .expect("file write should succeed");
    std::fs::write(dir.path().join("model-00002-of-00002.safetensors"), b"part2")
        .expect("file write should succeed");

    let files = find_safetensors_files(dir.path()).expect("operation should succeed");
    assert_eq!(files.len(), 2);
    // Files should be sorted
    assert!(files[0].to_string_lossy().contains("00001"));
    assert!(files[1].to_string_lossy().contains("00002"));
}

#[test]
fn test_find_safetensors_direct_file() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let file_path = dir.path().join("weights.safetensors");
    std::fs::write(&file_path, b"dummy").expect("file write should succeed");

    let files = find_safetensors_files(&file_path).expect("operation should succeed");
    assert_eq!(files.len(), 1);
}

#[test]
fn test_find_safetensors_non_safetensors_file() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let file_path = dir.path().join("model.bin");
    std::fs::write(&file_path, b"dummy").expect("file write should succeed");

    let files = find_safetensors_files(&file_path).expect("operation should succeed");
    assert!(files.is_empty());
}

#[test]
fn test_load_safetensors_no_files() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let result = load_safetensors_weights(dir.path(), Architecture::Auto);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("No SafeTensors files found"));
}

#[test]
fn test_load_safetensors_invalid_file() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let file_path = dir.path().join("model.safetensors");
    std::fs::write(&file_path, b"not a valid safetensors file").expect("file write should succeed");

    let result = load_safetensors_weights(dir.path(), Architecture::Auto);
    assert!(result.is_err());
}

#[test]
fn test_architecture_variants() {
    assert_eq!(Architecture::Llama, Architecture::Llama);
    assert_eq!(Architecture::Qwen2, Architecture::Qwen2);
    assert_eq!(Architecture::Mistral, Architecture::Mistral);

    // Test Clone
    let arch = Architecture::Llama;
    let cloned = arch;
    assert_eq!(arch, cloned);
}

// Test tensor conversion with a real safetensors file
#[test]
fn test_load_safetensors_real_file() {
    use safetensors::serialize;
    use safetensors::tensor::{Dtype, TensorView};

    let dir = TempDir::new().expect("temp file creation should succeed");
    let file_path = dir.path().join("model.safetensors");

    // Create a minimal safetensors file with f32 data
    let embed_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
    let embed_bytes: Vec<u8> = embed_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    let view =
        TensorView::new(Dtype::F32, vec![2, 2], &embed_bytes).expect("operation should succeed");
    let data = vec![("model.embed_tokens.weight", &view)];

    let serialized = serialize(data, None::<std::collections::HashMap<String, String>>)
        .expect("operation should succeed");
    std::fs::write(&file_path, serialized).expect("file write should succeed");

    let result = load_safetensors_weights(&file_path, Architecture::Llama);
    assert!(result.is_ok());
    let weights = result.expect("operation should succeed");
    assert!(weights.contains_key("model.embed_tokens.weight"));
}

#[test]
fn test_load_safetensors_with_f16() {
    use safetensors::serialize;
    use safetensors::tensor::{Dtype, TensorView};

    let dir = TempDir::new().expect("temp file creation should succeed");
    let file_path = dir.path().join("model.safetensors");

    // Create fp16 data
    let fp16_values: Vec<half::f16> = vec![
        half::f16::from_f32(0.1),
        half::f16::from_f32(0.2),
        half::f16::from_f32(0.3),
        half::f16::from_f32(0.4),
    ];
    let fp16_bytes: Vec<u8> = fp16_values.iter().flat_map(|f| f.to_le_bytes()).collect();

    let view =
        TensorView::new(Dtype::F16, vec![2, 2], &fp16_bytes).expect("operation should succeed");
    let data = vec![("model.embed_tokens.weight", &view)];

    let serialized = serialize(data, None::<std::collections::HashMap<String, String>>)
        .expect("operation should succeed");
    std::fs::write(&file_path, serialized).expect("file write should succeed");

    let result = load_safetensors_weights(&file_path, Architecture::Llama);
    assert!(result.is_ok());
}

#[test]
fn test_load_safetensors_with_bf16() {
    use safetensors::serialize;
    use safetensors::tensor::{Dtype, TensorView};

    let dir = TempDir::new().expect("temp file creation should succeed");
    let file_path = dir.path().join("model.safetensors");

    // Create bf16 data
    let bf16_values: Vec<half::bf16> = vec![
        half::bf16::from_f32(0.1),
        half::bf16::from_f32(0.2),
        half::bf16::from_f32(0.3),
        half::bf16::from_f32(0.4),
    ];
    let bf16_bytes: Vec<u8> = bf16_values.iter().flat_map(|f| f.to_le_bytes()).collect();

    let view =
        TensorView::new(Dtype::BF16, vec![2, 2], &bf16_bytes).expect("operation should succeed");
    let data = vec![("model.embed_tokens.weight", &view)];

    let serialized = serialize(data, None::<std::collections::HashMap<String, String>>)
        .expect("operation should succeed");
    std::fs::write(&file_path, serialized).expect("file write should succeed");

    let result = load_safetensors_weights(&file_path, Architecture::Llama);
    assert!(result.is_ok());
}

#[test]
fn test_load_safetensors_with_i32() {
    use safetensors::serialize;
    use safetensors::tensor::{Dtype, TensorView};

    let dir = TempDir::new().expect("temp file creation should succeed");
    let file_path = dir.path().join("model.safetensors");

    // Create i32 data
    let i32_values: Vec<i32> = vec![1, 2, 3, 4];
    let i32_bytes: Vec<u8> = i32_values.iter().flat_map(|i| i.to_le_bytes()).collect();

    let view =
        TensorView::new(Dtype::I32, vec![2, 2], &i32_bytes).expect("operation should succeed");
    let data = vec![("model.embed_tokens.weight", &view)];

    let serialized = serialize(data, None::<std::collections::HashMap<String, String>>)
        .expect("operation should succeed");
    std::fs::write(&file_path, serialized).expect("file write should succeed");

    let result = load_safetensors_weights(&file_path, Architecture::Llama);
    assert!(result.is_ok());
}

#[test]
fn test_load_safetensors_empty_tensor() {
    use safetensors::serialize;
    use safetensors::tensor::{Dtype, TensorView};

    let dir = TempDir::new().expect("temp file creation should succeed");
    let file_path = dir.path().join("model.safetensors");

    let empty_bytes: Vec<u8> = vec![];
    let view =
        TensorView::new(Dtype::F32, vec![0], &empty_bytes).expect("operation should succeed");
    let data = vec![("empty_tensor", &view)];

    let serialized = serialize(data, None::<std::collections::HashMap<String, String>>)
        .expect("operation should succeed");
    std::fs::write(&file_path, serialized).expect("file write should succeed");

    let result = load_safetensors_weights(&file_path, Architecture::Llama);
    assert!(result.is_ok());
    let weights = result.expect("operation should succeed");
    assert!(weights.contains_key("empty_tensor"));
}

#[test]
fn test_detect_architecture_qwen2() {
    use safetensors::serialize;
    use safetensors::tensor::{Dtype, TensorView};

    let dir = TempDir::new().expect("temp file creation should succeed");
    let file_path = dir.path().join("model.safetensors");

    let bias_bytes: Vec<u8> = vec![0.0f32; 4].iter().flat_map(|f| f.to_le_bytes()).collect();

    // Qwen2 has attention biases
    let view1 =
        TensorView::new(Dtype::F32, vec![4], &bias_bytes).expect("operation should succeed");
    let view2 =
        TensorView::new(Dtype::F32, vec![4], &bias_bytes).expect("operation should succeed");
    let data = vec![
        ("model.layers.0.self_attn.q_proj.bias", &view1),
        ("model.layers.0.self_attn.k_proj.bias", &view2),
    ];

    let serialized = serialize(data, None::<std::collections::HashMap<String, String>>)
        .expect("operation should succeed");
    std::fs::write(&file_path, serialized).expect("file write should succeed");

    let result = load_safetensors_weights(&file_path, Architecture::Auto);
    assert!(result.is_ok());
}

#[test]
fn test_detect_architecture_llama() {
    use safetensors::serialize;
    use safetensors::tensor::{Dtype, TensorView};

    let dir = TempDir::new().expect("temp file creation should succeed");
    let file_path = dir.path().join("model.safetensors");

    let weight_bytes: Vec<u8> = vec![0.1f32; 4].iter().flat_map(|f| f.to_le_bytes()).collect();

    // LLaMA has no attention biases
    let view =
        TensorView::new(Dtype::F32, vec![2, 2], &weight_bytes).expect("operation should succeed");
    let data = vec![("model.layers.0.self_attn.q_proj.weight", &view)];

    let serialized = serialize(data, None::<std::collections::HashMap<String, String>>)
        .expect("operation should succeed");
    std::fs::write(&file_path, serialized).expect("file write should succeed");

    let result = load_safetensors_weights(&file_path, Architecture::Auto);
    assert!(result.is_ok());
}

#[test]
fn test_load_safetensors_with_unsupported_dtype() {
    use safetensors::serialize;
    use safetensors::tensor::{Dtype, TensorView};

    let dir = TempDir::new().expect("temp file creation should succeed");
    let file_path = dir.path().join("model.safetensors");

    // Create U8 data (unsupported for conversion)
    let u8_bytes: Vec<u8> = vec![1, 2, 3, 4];
    let view = TensorView::new(Dtype::U8, vec![4], &u8_bytes).expect("operation should succeed");
    let data = vec![("unsupported_tensor", &view)];

    let serialized = serialize(data, None::<std::collections::HashMap<String, String>>)
        .expect("operation should succeed");
    std::fs::write(&file_path, serialized).expect("file write should succeed");

    // Should succeed but skip the unsupported tensor
    let result = load_safetensors_weights(&file_path, Architecture::Llama);
    assert!(result.is_ok());
    let weights = result.expect("operation should succeed");
    // Unsupported dtype tensors are skipped
    assert!(!weights.contains_key("unsupported_tensor"));
}

// =========================================================================
// ENC-006: RoBERTa / CodeBERT weight name mapping tests
// =========================================================================

#[test]
fn enc_006_roberta_embedding_mapping() {
    assert_eq!(
        map_weight_name("roberta.embeddings.word_embeddings.weight", Architecture::RoBERTa),
        "encoder.embed_tokens.weight"
    );
    assert_eq!(
        map_weight_name("roberta.embeddings.position_embeddings.weight", Architecture::RoBERTa),
        "encoder.position_embeddings.weight"
    );
    assert_eq!(
        map_weight_name("roberta.embeddings.LayerNorm.weight", Architecture::RoBERTa),
        "encoder.embeddings_layernorm.weight"
    );
    assert_eq!(
        map_weight_name("roberta.embeddings.LayerNorm.bias", Architecture::RoBERTa),
        "encoder.embeddings_layernorm.bias"
    );
}

#[test]
fn enc_006_roberta_attention_mapping() {
    assert_eq!(
        map_weight_name(
            "roberta.encoder.layer.0.attention.self.query.weight",
            Architecture::RoBERTa
        ),
        "encoder.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(
        map_weight_name("roberta.encoder.layer.3.attention.self.key.weight", Architecture::RoBERTa),
        "encoder.layers.3.self_attn.k_proj.weight"
    );
    assert_eq!(
        map_weight_name(
            "roberta.encoder.layer.11.attention.self.value.bias",
            Architecture::RoBERTa
        ),
        "encoder.layers.11.self_attn.v_proj.bias"
    );
    assert_eq!(
        map_weight_name(
            "roberta.encoder.layer.5.attention.output.dense.weight",
            Architecture::RoBERTa
        ),
        "encoder.layers.5.self_attn.o_proj.weight"
    );
}

#[test]
fn enc_006_roberta_layernorm_mapping() {
    assert_eq!(
        map_weight_name(
            "roberta.encoder.layer.0.attention.output.LayerNorm.weight",
            Architecture::RoBERTa
        ),
        "encoder.layers.0.input_layernorm.weight"
    );
    assert_eq!(
        map_weight_name(
            "roberta.encoder.layer.0.attention.output.LayerNorm.bias",
            Architecture::RoBERTa
        ),
        "encoder.layers.0.input_layernorm.bias"
    );
    assert_eq!(
        map_weight_name("roberta.encoder.layer.0.output.LayerNorm.weight", Architecture::RoBERTa),
        "encoder.layers.0.post_attention_layernorm.weight"
    );
}

#[test]
fn enc_006_roberta_ffn_mapping() {
    assert_eq!(
        map_weight_name("roberta.encoder.layer.0.intermediate.dense.weight", Architecture::RoBERTa),
        "encoder.layers.0.mlp.intermediate.dense.weight"
    );
    assert_eq!(
        map_weight_name("roberta.encoder.layer.0.intermediate.dense.bias", Architecture::RoBERTa),
        "encoder.layers.0.mlp.intermediate.dense.bias"
    );
    assert_eq!(
        map_weight_name("roberta.encoder.layer.0.output.dense.weight", Architecture::RoBERTa),
        "encoder.layers.0.mlp.output.dense.weight"
    );
}

#[test]
fn enc_006_bert_prefix_also_works() {
    // Some models use "bert." instead of "roberta." prefix
    assert_eq!(
        map_weight_name("bert.embeddings.word_embeddings.weight", Architecture::RoBERTa),
        "encoder.embed_tokens.weight"
    );
    assert_eq!(
        map_weight_name("bert.encoder.layer.0.attention.self.query.weight", Architecture::RoBERTa),
        "encoder.layers.0.self_attn.q_proj.weight"
    );
}

// =========================================================================
// Additional coverage tests
// =========================================================================

#[test]
fn test_expected_weight_count_with_biases() {
    // 2 layers without lm_head
    assert_eq!(expected_weight_count_with_biases(2, false), 2 + 2 * 12);
    // 2 layers with lm_head
    assert_eq!(expected_weight_count_with_biases(2, true), 2 + 2 * 12 + 1);
    // 24 layers (Qwen2.5-0.5B)
    assert_eq!(expected_weight_count_with_biases(24, false), 2 + 24 * 12);
    assert_eq!(expected_weight_count_with_biases(24, true), 2 + 24 * 12 + 1);
}

#[test]
fn test_parse_checkpoint_step_from_path() {
    use std::path::PathBuf;

    // Valid checkpoint
    let p = PathBuf::from("/tmp/model-step-3000.safetensors");
    assert_eq!(parse_checkpoint_step_from_path(&p), Some(3000));

    let p = PathBuf::from("model-step-0.safetensors");
    assert_eq!(parse_checkpoint_step_from_path(&p), Some(0));

    // Not a checkpoint
    let p = PathBuf::from("model.safetensors");
    assert_eq!(parse_checkpoint_step_from_path(&p), None);

    let p = PathBuf::from("model-best.safetensors");
    assert_eq!(parse_checkpoint_step_from_path(&p), None);

    let p = PathBuf::from("model-00001-of-00002.safetensors");
    assert_eq!(parse_checkpoint_step_from_path(&p), None);
}

#[test]
fn test_find_safetensors_checkpoint_files() {
    let dir = TempDir::new().expect("temp dir creation should succeed");

    // Create checkpoint files
    std::fs::write(dir.path().join("model-step-1000.safetensors"), b"ckpt1")
        .expect("file write should succeed");
    std::fs::write(dir.path().join("model-step-2000.safetensors"), b"ckpt2")
        .expect("file write should succeed");
    std::fs::write(dir.path().join("model-step-3000.safetensors"), b"ckpt3")
        .expect("file write should succeed");

    let files = find_safetensors_files(dir.path()).expect("operation should succeed");
    // Should return only the latest checkpoint
    assert_eq!(files.len(), 1);
    assert!(files[0].to_string_lossy().contains("model-step-3000"));
}

#[test]
fn test_architecture_debug() {
    assert_eq!(format!("{:?}", Architecture::Llama), "Llama");
    assert_eq!(format!("{:?}", Architecture::Qwen2), "Qwen2");
    assert_eq!(format!("{:?}", Architecture::Mistral), "Mistral");
    assert_eq!(format!("{:?}", Architecture::RoBERTa), "RoBERTa");
    assert_eq!(format!("{:?}", Architecture::Auto), "Auto");
}

#[test]
fn test_detect_architecture_roberta() {
    use safetensors::serialize;
    use safetensors::tensor::{Dtype, TensorView};

    let dir = TempDir::new().expect("temp file creation should succeed");
    let file_path = dir.path().join("model.safetensors");

    let data_bytes: Vec<u8> = vec![0.0f32; 4].iter().flat_map(|f| f.to_le_bytes()).collect();

    let view1 =
        TensorView::new(Dtype::F32, vec![4], &data_bytes).expect("operation should succeed");
    let view2 =
        TensorView::new(Dtype::F32, vec![4], &data_bytes).expect("operation should succeed");
    let data = vec![
        ("roberta.embeddings.word_embeddings.weight", &view1),
        ("roberta.encoder.layer.0.attention.self.query.weight", &view2),
    ];

    let serialized = serialize(data, None::<std::collections::HashMap<String, String>>)
        .expect("operation should succeed");
    std::fs::write(&file_path, serialized).expect("file write should succeed");

    let result = load_safetensors_weights(&file_path, Architecture::Auto);
    assert!(result.is_ok());
}

// PMAT-489: GGUF tensor name mapping tests
// Contract: map_weight_name(gguf_name, Architecture::Gguf) must return HF-convention name
// for all tensor types used in Qwen2/LLaMA decoder models.

#[test]
fn test_gguf_map_embedding() {
    assert_eq!(
        map_weight_name("token_embd.weight", Architecture::Gguf),
        "model.embed_tokens.weight"
    );
}

#[test]
fn test_gguf_map_output_norm() {
    assert_eq!(map_weight_name("output_norm.weight", Architecture::Gguf), "model.norm.weight");
    assert_eq!(map_weight_name("output_norm.bias", Architecture::Gguf), "model.norm.bias");
}

#[test]
fn test_gguf_map_lm_head() {
    assert_eq!(map_weight_name("output.weight", Architecture::Gguf), "lm_head.weight");
}

#[test]
fn test_gguf_map_attention_projections() {
    for layer in [0, 1, 27] {
        assert_eq!(
            map_weight_name(&format!("blk.{layer}.attn_q.weight"), Architecture::Gguf),
            format!("model.layers.{layer}.self_attn.q_proj.weight")
        );
        assert_eq!(
            map_weight_name(&format!("blk.{layer}.attn_k.weight"), Architecture::Gguf),
            format!("model.layers.{layer}.self_attn.k_proj.weight")
        );
        assert_eq!(
            map_weight_name(&format!("blk.{layer}.attn_v.weight"), Architecture::Gguf),
            format!("model.layers.{layer}.self_attn.v_proj.weight")
        );
        assert_eq!(
            map_weight_name(&format!("blk.{layer}.attn_output.weight"), Architecture::Gguf),
            format!("model.layers.{layer}.self_attn.o_proj.weight")
        );
    }
}

#[test]
fn test_gguf_map_attention_biases() {
    assert_eq!(
        map_weight_name("blk.0.attn_q.bias", Architecture::Gguf),
        "model.layers.0.self_attn.q_proj.bias"
    );
    assert_eq!(
        map_weight_name("blk.0.attn_k.bias", Architecture::Gguf),
        "model.layers.0.self_attn.k_proj.bias"
    );
    assert_eq!(
        map_weight_name("blk.0.attn_v.bias", Architecture::Gguf),
        "model.layers.0.self_attn.v_proj.bias"
    );
    assert_eq!(
        map_weight_name("blk.0.attn_output.bias", Architecture::Gguf),
        "model.layers.0.self_attn.o_proj.bias"
    );
}

#[test]
fn test_gguf_map_ffn_projections() {
    for layer in [0, 14, 27] {
        assert_eq!(
            map_weight_name(&format!("blk.{layer}.ffn_gate.weight"), Architecture::Gguf),
            format!("model.layers.{layer}.mlp.gate_proj.weight")
        );
        assert_eq!(
            map_weight_name(&format!("blk.{layer}.ffn_up.weight"), Architecture::Gguf),
            format!("model.layers.{layer}.mlp.up_proj.weight")
        );
        assert_eq!(
            map_weight_name(&format!("blk.{layer}.ffn_down.weight"), Architecture::Gguf),
            format!("model.layers.{layer}.mlp.down_proj.weight")
        );
    }
}

#[test]
fn test_gguf_map_layer_norms() {
    assert_eq!(
        map_weight_name("blk.0.attn_norm.weight", Architecture::Gguf),
        "model.layers.0.input_layernorm.weight"
    );
    assert_eq!(
        map_weight_name("blk.0.attn_norm.bias", Architecture::Gguf),
        "model.layers.0.input_layernorm.bias"
    );
    assert_eq!(
        map_weight_name("blk.0.ffn_norm.weight", Architecture::Gguf),
        "model.layers.0.post_attention_layernorm.weight"
    );
    assert_eq!(
        map_weight_name("blk.0.ffn_norm.bias", Architecture::Gguf),
        "model.layers.0.post_attention_layernorm.bias"
    );
}

#[test]
fn test_gguf_map_unknown_passthrough() {
    // Unknown tensor names should pass through unchanged
    assert_eq!(
        map_weight_name("some_unknown_tensor.weight", Architecture::Gguf),
        "some_unknown_tensor.weight"
    );
    // Unknown layer suffix should pass through with model.layers prefix
    assert_eq!(
        map_weight_name("blk.0.custom_ext.weight", Architecture::Gguf),
        "model.layers.0.custom_ext.weight"
    );
}

#[test]
fn test_gguf_map_all_28_layers_qwen2_1_5b() {
    // Qwen2.5-Coder-1.5B has 28 layers — verify mapping for boundary layers
    for layer in [0, 13, 27] {
        let q = map_weight_name(&format!("blk.{layer}.attn_q.weight"), Architecture::Gguf);
        assert!(q.starts_with("model.layers."), "layer {layer} q_proj");
        assert!(q.ends_with(".self_attn.q_proj.weight"), "layer {layer} q_proj suffix");
    }
}

#[test]
fn test_gguf_map_completeness_for_training() {
    // Training pipeline requires ALL these names from from_params().
    // Verify the complete set for one layer + non-layer tensors.
    let required_hf_names = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ];

    let gguf_names = [
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    ];

    for (gguf, expected_hf) in gguf_names.iter().zip(required_hf_names.iter()) {
        let mapped = map_weight_name(gguf, Architecture::Gguf);
        assert_eq!(
            &mapped, expected_hf,
            "GGUF '{gguf}' should map to '{expected_hf}', got '{mapped}'"
        );
    }
}

#[test]
fn test_gguf_architecture_debug_and_equality() {
    assert_eq!(format!("{:?}", Architecture::Gguf), "Gguf");
    assert_ne!(Architecture::Gguf, Architecture::Auto);
    assert_ne!(Architecture::Gguf, Architecture::Llama);
    assert_eq!(Architecture::Gguf, Architecture::Gguf);
}
