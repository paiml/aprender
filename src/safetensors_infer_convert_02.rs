
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_convert_with_num_kv_heads_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        // Config with GQA (num_key_value_heads < num_attention_heads)
        let config = r#"{
            "hidden_size": 64,
            "num_hidden_layers": 0,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "vocab_size": 100
        }"#;
        std::fs::write(&config_path, config).expect("write config");

        let embed_data = valid_f32_bytes(100 * 64);
        let norm_data: Vec<u8> = (0..64).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        let data = create_safetensors_bytes(&[
            ("model.embed_tokens.weight", "F32", &[100, 64], &embed_data),
            ("model.norm.weight", "F32", &[64], &norm_data),
        ]);
        std::fs::write(&model_path, data).expect("write safetensors");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_ok(), "convert failed: {:?}", result.err());
        let transformer = result.expect("operation failed");
        assert_eq!(transformer.config.num_heads, 8);
        assert_eq!(transformer.config.num_kv_heads, 4);
    }
include!("safetensors_infer_concat_qkv_valid.rs");
include!("safetensors_infer_convert_separate.rs");
include!("safetensors_infer_transpose_weight.rs");
}
