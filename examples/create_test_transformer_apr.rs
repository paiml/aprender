//! Generate test APR v2 file with full transformer metadata for inference testing
//! Creates a minimal decoder-only LLM with proper tensor structure
use aprender::format::v2::{AprV2Metadata, AprV2Writer, TensorDType};
use std::collections::HashMap;

fn main() {
    // Transformer config for a tiny model
    let vocab_size: usize = 128; // Very small vocab for testing
    let hidden_size: usize = 64;
    let num_layers: usize = 2;
    let num_heads: usize = 4;
    let num_kv_heads: usize = 4;
    let intermediate_size: usize = 256;
    let head_dim: usize = hidden_size / num_heads;

    // Create metadata with CRITICAL transformer config
    let mut metadata = AprV2Metadata::new("llm");
    metadata.name = Some("test-transformer".to_string());
    metadata.description = Some("Minimal test transformer for APR inference testing".to_string());

    // Set transformer config (CRITICAL for realizar inference)
    metadata.architecture = Some("qwen2".to_string());
    metadata.hidden_size = Some(hidden_size);
    metadata.num_layers = Some(num_layers);
    metadata.num_heads = Some(num_heads);
    metadata.num_kv_heads = Some(num_kv_heads);
    metadata.vocab_size = Some(vocab_size);
    metadata.intermediate_size = Some(intermediate_size);
    metadata.max_position_embeddings = Some(512);
    metadata.rope_theta = Some(10000.0);
    metadata.rms_norm_eps = Some(1e-6);

    // Add tokenizer info
    let mut custom = HashMap::new();

    // Create a simple vocabulary (ASCII characters + special tokens)
    let mut vocab: Vec<serde_json::Value> = vec![
        serde_json::Value::String("<pad>".to_string()),
        serde_json::Value::String("<bos>".to_string()),
        serde_json::Value::String("<eos>".to_string()),
        serde_json::Value::String("<unk>".to_string()),
    ];
    // Add printable ASCII
    for c in 32u8..127 {
        vocab.push(serde_json::Value::String((c as char).to_string()));
    }
    // Pad to vocab_size
    while vocab.len() < vocab_size {
        vocab.push(serde_json::Value::String(format!("<unused{}>", vocab.len())));
    }

    custom.insert(
        "tokenizer.vocabulary".to_string(),
        serde_json::Value::Array(vocab),
    );
    custom.insert(
        "tokenizer.vocab_size".to_string(),
        serde_json::Value::Number(serde_json::Number::from(vocab_size)),
    );
    custom.insert(
        "tokenizer.bos_token_id".to_string(),
        serde_json::Value::Number(serde_json::Number::from(1)),
    );
    custom.insert(
        "tokenizer.eos_token_id".to_string(),
        serde_json::Value::Number(serde_json::Number::from(2)),
    );
    custom.insert(
        "tokenizer.model_type".to_string(),
        serde_json::Value::String("qwen2".to_string()),
    );
    custom.insert(
        "tokenizer.architecture".to_string(),
        serde_json::Value::String("qwen2".to_string()),
    );

    metadata.custom = custom;

    let mut writer = AprV2Writer::new(metadata);

    // Helper to create f32 tensor bytes with small random-ish values
    fn f32_bytes(size: usize, seed: u32) -> Vec<u8> {
        (0..size)
            .map(|i| {
                // Simple deterministic "random" values scaled small
                let x = ((i as u32).wrapping_add(seed).wrapping_mul(2654435761)) as f32;
                (x / u32::MAX as f32 - 0.5) * 0.02 // Values in [-0.01, 0.01]
            })
            .flat_map(|f| f.to_le_bytes())
            .collect()
    }

    // Token embedding: [vocab_size, hidden_size]
    writer.add_tensor(
        "model.embed_tokens.weight",
        TensorDType::F32,
        vec![vocab_size, hidden_size],
        f32_bytes(vocab_size * hidden_size, 1),
    );

    // Decoder layers
    for i in 0..num_layers {
        let prefix = format!("model.layers.{}", i);
        let seed_base = (i as u32 + 1) * 1000;

        // Input layer norm
        writer.add_tensor(
            &format!("{}.input_layernorm.weight", prefix),
            TensorDType::F32,
            vec![hidden_size],
            // RMSNorm weights should be ~1.0
            vec![1.0f32; hidden_size]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect(),
        );

        // Self attention
        // Q, K, V projections: [hidden_size, hidden_size] (or [num_heads * head_dim, hidden_size])
        writer.add_tensor(
            &format!("{}.self_attn.q_proj.weight", prefix),
            TensorDType::F32,
            vec![num_heads * head_dim, hidden_size],
            f32_bytes(num_heads * head_dim * hidden_size, seed_base + 1),
        );
        writer.add_tensor(
            &format!("{}.self_attn.k_proj.weight", prefix),
            TensorDType::F32,
            vec![num_kv_heads * head_dim, hidden_size],
            f32_bytes(num_kv_heads * head_dim * hidden_size, seed_base + 2),
        );
        writer.add_tensor(
            &format!("{}.self_attn.v_proj.weight", prefix),
            TensorDType::F32,
            vec![num_kv_heads * head_dim, hidden_size],
            f32_bytes(num_kv_heads * head_dim * hidden_size, seed_base + 3),
        );
        writer.add_tensor(
            &format!("{}.self_attn.o_proj.weight", prefix),
            TensorDType::F32,
            vec![hidden_size, num_heads * head_dim],
            f32_bytes(hidden_size * num_heads * head_dim, seed_base + 4),
        );

        // Post attention layer norm
        writer.add_tensor(
            &format!("{}.post_attention_layernorm.weight", prefix),
            TensorDType::F32,
            vec![hidden_size],
            vec![1.0f32; hidden_size]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect(),
        );

        // MLP (SwiGLU style: gate, up, down)
        writer.add_tensor(
            &format!("{}.mlp.gate_proj.weight", prefix),
            TensorDType::F32,
            vec![intermediate_size, hidden_size],
            f32_bytes(intermediate_size * hidden_size, seed_base + 5),
        );
        writer.add_tensor(
            &format!("{}.mlp.up_proj.weight", prefix),
            TensorDType::F32,
            vec![intermediate_size, hidden_size],
            f32_bytes(intermediate_size * hidden_size, seed_base + 6),
        );
        writer.add_tensor(
            &format!("{}.mlp.down_proj.weight", prefix),
            TensorDType::F32,
            vec![hidden_size, intermediate_size],
            f32_bytes(hidden_size * intermediate_size, seed_base + 7),
        );
    }

    // Final layer norm
    writer.add_tensor(
        "model.norm.weight",
        TensorDType::F32,
        vec![hidden_size],
        vec![1.0f32; hidden_size]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect(),
    );

    // LM head (output projection): [vocab_size, hidden_size]
    writer.add_tensor(
        "lm_head.weight",
        TensorDType::F32,
        vec![vocab_size, hidden_size],
        f32_bytes(vocab_size * hidden_size, 9999),
    );

    let apr_bytes = writer.write().expect("Failed to write APR");
    std::fs::write("/tmp/test-transformer.apr", &apr_bytes).expect("Failed to save");
    println!(
        "Created /tmp/test-transformer.apr ({} bytes)",
        apr_bytes.len()
    );
    println!();
    println!("Transformer config:");
    println!("  vocab_size: {}", vocab_size);
    println!("  hidden_size: {}", hidden_size);
    println!("  num_layers: {}", num_layers);
    println!("  num_heads: {}", num_heads);
    println!("  intermediate_size: {}", intermediate_size);
    println!();
    println!("Test with:");
    println!("  cargo run --bin apr -- inspect /tmp/test-transformer.apr");
    println!("  cargo run --bin apr --features inference -- serve /tmp/test-transformer.apr --port 8093");
}
