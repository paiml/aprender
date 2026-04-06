//! Integration tests for GGUF transformer forward pass
//!
//! Tests loading real GGUF models and running inference.
//!
//! Requires phi-2 model file at specified path (ignored by default).

use std::fs;

use realizar::gguf::{GGUFConfig, GGUFModel, GGUFTransformer};

/// Path to phi-2 Q4_K_M quantized model
const PHI2_MODEL_PATH: &str = "/home/noah/src/single-shot-eval/models/raw/phi-2-q4_k_m.gguf";

/// Test loading phi-2 embedding layer weights
#[test]
#[ignore = "requires phi-2 model file"]
fn test_phi2_load_embedding_weights() {
    let data = fs::read(PHI2_MODEL_PATH).expect("Failed to read phi-2 model");
    let model = GGUFModel::from_bytes(&data).expect("Failed to parse phi-2 model");

    // Load token embedding weights
    let embedding = model
        .get_tensor_f32("token_embd.weight", &data)
        .expect("Failed to load embedding weights");

    // phi-2 has vocab_size=51200, hidden_dim=2560
    // Total elements = 51200 * 2560 = 131,072,000
    println!("Embedding shape: {} elements", embedding.len());
    assert_eq!(embedding.len(), 131_072_000, "Unexpected embedding size");

    // Verify values are reasonable (not inf/nan)
    let valid_values = embedding.iter().filter(|v| v.is_finite()).count();
    assert_eq!(
        valid_values,
        embedding.len(),
        "Found non-finite values in embedding"
    );

    // Check value range
    let min = embedding.iter().copied().fold(f32::INFINITY, f32::min);
    let max = embedding.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!("Embedding value range: [{}, {}]", min, max);
    assert!(
        min > -10.0 && max < 10.0,
        "Embedding values out of expected range"
    );
}

/// Test loading phi-2 attention layer weights
#[test]
#[ignore = "requires phi-2 model file"]
fn test_phi2_load_attention_weights() {
    let data = fs::read(PHI2_MODEL_PATH).expect("Failed to read phi-2 model");
    let model = GGUFModel::from_bytes(&data).expect("Failed to parse phi-2 model");

    // phi-2 uses combined QKV projection
    // blk.0.attn_qkv.weight: [2560, 7680] where 7680 = 2560 * 3
    let qkv_weights = model
        .get_tensor_f32("blk.0.attn_qkv.weight", &data)
        .expect("Failed to load QKV weights");

    // 2560 * 7680 = 19,660,800
    println!("QKV weight shape: {} elements", qkv_weights.len());
    assert_eq!(qkv_weights.len(), 19_660_800, "Unexpected QKV weight size");

    // Load attention output projection
    let attn_output = model
        .get_tensor_f32("blk.0.attn_output.weight", &data)
        .expect("Failed to load attention output weights");

    // 2560 * 2560 = 6,553,600
    println!("Attention output shape: {} elements", attn_output.len());
    assert_eq!(attn_output.len(), 6_553_600, "Unexpected attn_output size");
}

/// Test loading phi-2 FFN layer weights
#[test]
#[ignore = "requires phi-2 model file"]
fn test_phi2_load_ffn_weights() {
    let data = fs::read(PHI2_MODEL_PATH).expect("Failed to read phi-2 model");
    let model = GGUFModel::from_bytes(&data).expect("Failed to parse phi-2 model");

    // FFN up projection: [2560, 10240]
    let ffn_up = model
        .get_tensor_f32("blk.0.ffn_up.weight", &data)
        .expect("Failed to load FFN up weights");

    // 2560 * 10240 = 26,214,400
    println!("FFN up shape: {} elements", ffn_up.len());
    assert_eq!(ffn_up.len(), 26_214_400, "Unexpected FFN up size");

    // FFN down projection: [10240, 2560]
    let ffn_down = model
        .get_tensor_f32("blk.0.ffn_down.weight", &data)
        .expect("Failed to load FFN down weights");

    // 10240 * 2560 = 26,214,400
    println!("FFN down shape: {} elements", ffn_down.len());
    assert_eq!(ffn_down.len(), 26_214_400, "Unexpected FFN down size");
}

/// Test loading all 32 layers of phi-2
#[test]
#[ignore = "requires phi-2 model file"]
fn test_phi2_load_all_layers() {
    let data = fs::read(PHI2_MODEL_PATH).expect("Failed to read phi-2 model");
    let model = GGUFModel::from_bytes(&data).expect("Failed to parse phi-2 model");

    // Load weights for all 32 layers
    for layer_idx in 0..32 {
        let qkv_name = format!("blk.{}.attn_qkv.weight", layer_idx);
        let output_name = format!("blk.{}.attn_output.weight", layer_idx);
        let ffn_up_name = format!("blk.{}.ffn_up.weight", layer_idx);
        let ffn_down_name = format!("blk.{}.ffn_down.weight", layer_idx);

        // Test each weight loads successfully
        model
            .get_tensor_f32(&qkv_name, &data)
            .unwrap_or_else(|_| panic!("Failed to load {}", qkv_name));
        model
            .get_tensor_f32(&output_name, &data)
            .unwrap_or_else(|_| panic!("Failed to load {}", output_name));
        model
            .get_tensor_f32(&ffn_up_name, &data)
            .unwrap_or_else(|_| panic!("Failed to load {}", ffn_up_name));
        model
            .get_tensor_f32(&ffn_down_name, &data)
            .unwrap_or_else(|_| panic!("Failed to load {}", ffn_down_name));

        if layer_idx % 8 == 0 {
            println!("Loaded layer {} successfully", layer_idx);
        }
    }

    println!("All 32 layers loaded successfully!");
}

/// Test output layer weights
#[test]
#[ignore = "requires phi-2 model file"]
fn test_phi2_load_output_weights() {
    let data = fs::read(PHI2_MODEL_PATH).expect("Failed to read phi-2 model");
    let model = GGUFModel::from_bytes(&data).expect("Failed to parse phi-2 model");

    // Output norm
    let output_norm_weight = model
        .get_tensor_f32("output_norm.weight", &data)
        .expect("Failed to load output norm weight");
    assert_eq!(
        output_norm_weight.len(),
        2560,
        "Unexpected output norm size"
    );

    let output_norm_bias = model
        .get_tensor_f32("output_norm.bias", &data)
        .expect("Failed to load output norm bias");
    assert_eq!(
        output_norm_bias.len(),
        2560,
        "Unexpected output norm bias size"
    );

    // LM head (output projection)
    let output_weight = model
        .get_tensor_f32("output.weight", &data)
        .expect("Failed to load output weight");

    // 2560 * 51200 = 131,072,000 (same as embedding in tied models)
    println!("Output weight shape: {} elements", output_weight.len());
}

/// Test simple embedding lookup
#[test]
#[ignore = "requires phi-2 model file"]
fn test_phi2_embedding_lookup() {
    let data = fs::read(PHI2_MODEL_PATH).expect("Failed to read phi-2 model");
    let model = GGUFModel::from_bytes(&data).expect("Failed to parse phi-2 model");

    // Load embedding weights
    let embedding = model
        .get_tensor_f32("token_embd.weight", &data)
        .expect("Failed to load embedding weights");

    let _vocab_size = 51200;
    let hidden_dim = 2560;

    // Look up token 0
    let token_0_embedding: Vec<f32> = embedding[0..hidden_dim].to_vec();
    println!(
        "Token 0 embedding (first 10): {:?}",
        &token_0_embedding[..10]
    );

    // Look up token 1
    let token_1_start = hidden_dim;
    let token_1_embedding: Vec<f32> = embedding[token_1_start..token_1_start + hidden_dim].to_vec();
    println!(
        "Token 1 embedding (first 10): {:?}",
        &token_1_embedding[..10]
    );

    // Embeddings should be different
    assert_ne!(
        token_0_embedding[..10],
        token_1_embedding[..10],
        "Token 0 and 1 should have different embeddings"
    );
}

/// Test extracting GGUFConfig from model
#[test]
#[ignore = "requires phi-2 model file"]
fn test_phi2_config_extraction() {
    let data = fs::read(PHI2_MODEL_PATH).expect("Failed to read phi-2 model");
    let model = GGUFModel::from_bytes(&data).expect("Failed to parse phi-2 model");

    let config = GGUFConfig::from_gguf(&model).expect("Failed to extract config");

    println!("phi-2 config:");
    println!("  architecture: {}", config.architecture);
    println!("  hidden_dim: {}", config.hidden_dim);
    println!("  num_layers: {}", config.num_layers);
    println!("  num_heads: {}", config.num_heads);
    println!("  vocab_size: {}", config.vocab_size);
    println!("  intermediate_dim: {}", config.intermediate_dim);
    println!("  context_length: {}", config.context_length);

    assert_eq!(config.architecture, "phi2");
    assert_eq!(config.hidden_dim, 2560);
    assert_eq!(config.num_layers, 32);
    assert_eq!(config.vocab_size, 51200);
    assert_eq!(config.intermediate_dim, 10240);
    assert_eq!(config.context_length, 2048);
}

/// Test loading GGUFTransformer from model
#[test]
#[ignore = "requires phi-2 model file"]
fn test_phi2_transformer_loading() {
    let data = fs::read(PHI2_MODEL_PATH).expect("Failed to read phi-2 model");
    let model = GGUFModel::from_bytes(&data).expect("Failed to parse phi-2 model");

    println!("Loading phi-2 transformer weights...");
    let start = std::time::Instant::now();
    let transformer =
        GGUFTransformer::from_gguf(&model, &data).expect("Failed to load transformer");
    let elapsed = start.elapsed();
    println!("Loaded in {:?}", elapsed);

    println!("Transformer structure:");
    println!(
        "  Token embedding: {} elements",
        transformer.token_embedding.len()
    );
    println!("  Layers: {}", transformer.layers.len());
    println!(
        "  Output norm weight: {} elements",
        transformer.output_norm_weight.len()
    );
    println!(
        "  LM head weight: {} elements",
        transformer.lm_head_weight.len()
    );

    assert_eq!(transformer.layers.len(), 32);
    assert_eq!(transformer.token_embedding.len(), 51200 * 2560);
    assert_eq!(transformer.output_norm_weight.len(), 2560);
}
