use super::*;

#[test]
fn test_load_from_apr_with_file() {
    use crate::format::v2::{AprV2Metadata, AprV2Writer};
    use std::io::Write;
    use tempfile::NamedTempFile;

    let config = create_tiny_config();

    // Create a minimal APR file with just embed_tokens weight
    let mut writer = AprV2Writer::new(AprV2Metadata::default());

    // Add a small embed_tokens weight tensor
    let embed_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    writer.add_f32_tensor(
        "embed_tokens.weight",
        vec![config.vocab_size, config.hidden_size],
        &embed_data,
    );

    // Add norm weight
    let norm_data: Vec<f32> = vec![1.0; config.hidden_size];
    writer.add_f32_tensor("norm.weight", vec![config.hidden_size], &norm_data);

    // Add lm_head weight
    let lm_head_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    writer.add_f32_tensor(
        "lm_head.weight",
        vec![config.vocab_size, config.hidden_size],
        &lm_head_data,
    );

    // Add layer 0 weights
    for layer_idx in 0..config.num_layers {
        let prefix = format!("layers.{layer_idx}");
        let h = config.hidden_size;
        let i = config.intermediate_size;
        let head_dim = h / config.num_attention_heads;
        let kv_dim = config.num_kv_heads * head_dim;

        // Attention weights
        writer.add_f32_tensor(
            &format!("{prefix}.self_attn.q_proj.weight"),
            vec![h, h],
            &vec![0.01; h * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.self_attn.k_proj.weight"),
            vec![kv_dim, h],
            &vec![0.01; kv_dim * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.self_attn.v_proj.weight"),
            vec![kv_dim, h],
            &vec![0.01; kv_dim * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.self_attn.o_proj.weight"),
            vec![h, h],
            &vec![0.01; h * h],
        );

        // MLP weights
        writer.add_f32_tensor(
            &format!("{prefix}.mlp.gate_proj.weight"),
            vec![i, h],
            &vec![0.01; i * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.mlp.up_proj.weight"),
            vec![i, h],
            &vec![0.01; i * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.mlp.down_proj.weight"),
            vec![h, i],
            &vec![0.01; h * i],
        );

        // Layer norms
        writer.add_f32_tensor(
            &format!("{prefix}.input_layernorm.weight"),
            vec![h],
            &vec![1.0; h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.post_attention_layernorm.weight"),
            vec![h],
            &vec![1.0; h],
        );
    }

    // Write to temp file
    let mut temp_file = NamedTempFile::new().expect("create temp file");
    let apr_bytes = writer.write().expect("serialize APR");
    temp_file.write_all(&apr_bytes).expect("write APR data");
    temp_file.flush().expect("flush");

    // Now test loading
    let mut model = Qwen2Model::new_uninitialized(&config);
    let result = model.load_from_apr(temp_file.path());

    assert!(result.is_ok(), "Should load APR file: {:?}", result.err());
    let loaded = result.unwrap();
    // Should load: embed + norm + lm_head + layers*(9 weights)
    let expected = 3 + config.num_layers * 9;
    assert_eq!(loaded, expected, "Should load {} tensors", expected);
}

#[test]
fn test_load_from_apr_weight_tying() {
    use crate::format::v2::{AprV2Metadata, AprV2Writer};
    use std::io::Write;
    use tempfile::NamedTempFile;

    let config = create_tiny_config();

    // Create APR file WITHOUT lm_head.weight (triggers weight tying)
    let mut writer = AprV2Writer::new(AprV2Metadata::default());

    // Add embed_tokens weight
    let embed_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    writer.add_f32_tensor(
        "embed_tokens.weight",
        vec![config.vocab_size, config.hidden_size],
        &embed_data,
    );

    // Add norm weight
    writer.add_f32_tensor(
        "norm.weight",
        vec![config.hidden_size],
        &vec![1.0; config.hidden_size],
    );

    // NO lm_head.weight - should fall back to embed_tokens.weight

    // Add layer weights
    for layer_idx in 0..config.num_layers {
        let prefix = format!("layers.{layer_idx}");
        let h = config.hidden_size;
        let i = config.intermediate_size;
        let head_dim = h / config.num_attention_heads;
        let kv_dim = config.num_kv_heads * head_dim;

        writer.add_f32_tensor(
            &format!("{prefix}.self_attn.q_proj.weight"),
            vec![h, h],
            &vec![0.01; h * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.self_attn.k_proj.weight"),
            vec![kv_dim, h],
            &vec![0.01; kv_dim * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.self_attn.v_proj.weight"),
            vec![kv_dim, h],
            &vec![0.01; kv_dim * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.self_attn.o_proj.weight"),
            vec![h, h],
            &vec![0.01; h * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.mlp.gate_proj.weight"),
            vec![i, h],
            &vec![0.01; i * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.mlp.up_proj.weight"),
            vec![i, h],
            &vec![0.01; i * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.mlp.down_proj.weight"),
            vec![h, i],
            &vec![0.01; h * i],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.input_layernorm.weight"),
            vec![h],
            &vec![1.0; h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.post_attention_layernorm.weight"),
            vec![h],
            &vec![1.0; h],
        );
    }

    // Write to temp file
    let mut temp_file = NamedTempFile::new().expect("create temp file");
    let apr_bytes = writer.write().expect("serialize APR");
    temp_file.write_all(&apr_bytes).expect("write APR data");
    temp_file.flush().expect("flush");

    // Test loading with weight tying
    let mut model = Qwen2Model::new_uninitialized(&config);
    let result = model.load_from_apr(temp_file.path());

    assert!(
        result.is_ok(),
        "Should load APR file with weight tying: {:?}",
        result.err()
    );
    // Should load embed + norm + lm_head(from embed) + layers*(9 weights)
    let loaded = result.unwrap();
    assert!(
        loaded >= 2 + config.num_layers * 9,
        "Should load tensors with weight tying"
    );
}

#[test]
fn test_from_apr_static_method() {
    use crate::format::v2::{AprV2Metadata, AprV2Writer};
    use std::io::Write;
    use tempfile::NamedTempFile;

    let config = create_tiny_config();

    // Create minimal valid APR file
    let mut writer = AprV2Writer::new(AprV2Metadata::default());
    writer.add_f32_tensor(
        "embed_tokens.weight",
        vec![config.vocab_size, config.hidden_size],
        &vec![0.01; config.vocab_size * config.hidden_size],
    );
    writer.add_f32_tensor(
        "norm.weight",
        vec![config.hidden_size],
        &vec![1.0; config.hidden_size],
    );

    // Add minimal layer weights
    for layer_idx in 0..config.num_layers {
        let prefix = format!("layers.{layer_idx}");
        let h = config.hidden_size;
        let i = config.intermediate_size;
        let head_dim = h / config.num_attention_heads;
        let kv_dim = config.num_kv_heads * head_dim;

        writer.add_f32_tensor(
            &format!("{prefix}.self_attn.q_proj.weight"),
            vec![h, h],
            &vec![0.01; h * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.self_attn.k_proj.weight"),
            vec![kv_dim, h],
            &vec![0.01; kv_dim * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.self_attn.v_proj.weight"),
            vec![kv_dim, h],
            &vec![0.01; kv_dim * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.self_attn.o_proj.weight"),
            vec![h, h],
            &vec![0.01; h * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.mlp.gate_proj.weight"),
            vec![i, h],
            &vec![0.01; i * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.mlp.up_proj.weight"),
            vec![i, h],
            &vec![0.01; i * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.mlp.down_proj.weight"),
            vec![h, i],
            &vec![0.01; h * i],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.input_layernorm.weight"),
            vec![h],
            &vec![1.0; h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.post_attention_layernorm.weight"),
            vec![h],
            &vec![1.0; h],
        );
    }

    // Write to temp file
    let mut temp_file = NamedTempFile::new().expect("create temp file");
    let apr_bytes = writer.write().expect("serialize APR");
    temp_file.write_all(&apr_bytes).expect("write APR data");
    temp_file.flush().expect("flush");

    // Test from_apr static method
    let result = Qwen2Model::from_apr(&config, temp_file.path());
    assert!(result.is_ok(), "from_apr should work: {:?}", result.err());

    let model = result.unwrap();
    assert_eq!(model.num_layers(), config.num_layers);
}
