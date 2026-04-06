
// ============================================================================
// ROSETTA-003: GQA round-trip tests (Option C - Store Separate, Fuse at Runtime)
// ============================================================================
#[cfg(test)]
mod tests_rosetta003_gqa_roundtrip {
    use super::*;
    use crate::format::test_factory::harness::ConversionTestHarness;
    use crate::format::test_factory::PygmyConfig;
    use crate::format::v2::AprV2Reader;

    /// ROSETTA-003: GQA model round-trip preserves separate Q/K/V tensors.
    #[test]
    fn test_rosetta003_gqa_roundtrip_safetensors() {
        ConversionTestHarness::assert_roundtrip_ok(PygmyConfig::qwen2_gqa());
    }

    /// ROSETTA-003: GQA import produces correct tensor names (no qkv_proj fusion).
    #[test]
    fn test_rosetta003_gqa_no_fusion_in_apr() {
        let h = ConversionTestHarness::new()
            .with_safetensors(PygmyConfig::qwen2_gqa())
            .import_to_apr(ImportOptions {
                allow_no_config: true,
                ..ImportOptions::default()
            });

        let output = h.output_path().expect("output exists");
        let data = fs::read(output).expect("read output");
        let reader = AprV2Reader::from_bytes(&data).expect("parse APR");

        let names = reader.tensor_names();

        // ROSETTA-003: Must have separate Q/K/V, never fused qkv_proj
        assert!(
            names.iter().any(|n| n.contains("q_proj.weight")),
            "APR must contain separate q_proj.weight"
        );
        assert!(
            names.iter().any(|n| n.contains("k_proj.weight")),
            "APR must contain separate k_proj.weight"
        );
        assert!(
            names.iter().any(|n| n.contains("v_proj.weight")),
            "APR must contain separate v_proj.weight"
        );
        assert!(
            !names.iter().any(|n| n.contains("qkv_proj")),
            "APR must NOT contain fused qkv_proj (ROSETTA-003): found {:?}",
            names
                .iter()
                .filter(|n| n.contains("qkv_proj"))
                .collect::<Vec<_>>()
        );
    }

    /// ROSETTA-003: GQA K/V dimensions preserved through round-trip.
    /// Q: [hidden_size, hidden_size], K/V: [kv_dim, hidden_size]
    #[test]
    fn test_rosetta003_gqa_kv_dimensions_preserved() {
        let config = PygmyConfig::qwen2_gqa();
        let kv_dim = config.kv_dim(); // 4 for this config
        let hidden = config.hidden_size; // 8

        let h = ConversionTestHarness::new()
            .with_safetensors(config)
            .import_to_apr(ImportOptions {
                allow_no_config: true,
                ..ImportOptions::default()
            });

        let output = h.output_path().expect("output exists");
        let data = fs::read(output).expect("read output");
        let reader = AprV2Reader::from_bytes(&data).expect("parse APR");

        // Check Q has full hidden_size dimension
        let q = reader
            .get_tensor("model.layers.0.self_attn.q_proj.weight")
            .expect("q_proj exists");
        assert_eq!(
            q.shape,
            vec![hidden, hidden],
            "Q shape must be [hidden_size, hidden_size]"
        );

        // Check K/V have reduced kv_dim
        let k = reader
            .get_tensor("model.layers.0.self_attn.k_proj.weight")
            .expect("k_proj exists");
        assert_eq!(
            k.shape,
            vec![kv_dim, hidden],
            "K shape must be [kv_dim, hidden_size] for GQA"
        );

        let v = reader
            .get_tensor("model.layers.0.self_attn.v_proj.weight")
            .expect("v_proj exists");
        assert_eq!(
            v.shape,
            vec![kv_dim, hidden],
            "V shape must be [kv_dim, hidden_size] for GQA"
        );
    }

    /// ROSETTA-003: Attention biases preserved in round-trip.
    #[test]
    fn test_rosetta003_gqa_biases_preserved() {
        let config = PygmyConfig::qwen2_gqa();
        let kv_dim = config.kv_dim();
        let hidden = config.hidden_size;

        let h = ConversionTestHarness::new()
            .with_safetensors(config)
            .import_to_apr(ImportOptions {
                allow_no_config: true,
                ..ImportOptions::default()
            });

        let output = h.output_path().expect("output exists");
        let data = fs::read(output).expect("read output");
        let reader = AprV2Reader::from_bytes(&data).expect("parse APR");

        // Q bias: [hidden_size]
        let q_bias = reader
            .get_tensor("model.layers.0.self_attn.q_proj.bias")
            .expect("q_proj.bias exists");
        assert_eq!(
            q_bias.shape,
            vec![hidden],
            "Q bias shape must be [hidden_size]"
        );

        // K/V bias: [kv_dim]
        let k_bias = reader
            .get_tensor("model.layers.0.self_attn.k_proj.bias")
            .expect("k_proj.bias exists");
        assert_eq!(k_bias.shape, vec![kv_dim], "K bias shape must be [kv_dim]");

        let v_bias = reader
            .get_tensor("model.layers.0.self_attn.v_proj.bias")
            .expect("v_proj.bias exists");
        assert_eq!(v_bias.shape, vec![kv_dim], "V bias shape must be [kv_dim]");
    }

    /// ROSETTA-003: Multi-layer GQA verified (layer 0 and layer 1 both correct).
    #[test]
    fn test_rosetta003_gqa_multi_layer() {
        let config = PygmyConfig::qwen2_gqa();
        let kv_dim = config.kv_dim();
        let hidden = config.hidden_size;

        let h = ConversionTestHarness::new()
            .with_safetensors(config)
            .import_to_apr(ImportOptions {
                allow_no_config: true,
                ..ImportOptions::default()
            });

        let output = h.output_path().expect("output exists");
        let data = fs::read(output).expect("read output");
        let reader = AprV2Reader::from_bytes(&data).expect("parse APR");

        // Both layers must have separate Q/K/V with correct shapes
        for layer in 0..2 {
            let q = reader
                .get_tensor(&format!("model.layers.{layer}.self_attn.q_proj.weight"))
                .unwrap_or_else(|| panic!("layer {layer} q_proj missing"));
            assert_eq!(q.shape, vec![hidden, hidden]);

            let k = reader
                .get_tensor(&format!("model.layers.{layer}.self_attn.k_proj.weight"))
                .unwrap_or_else(|| panic!("layer {layer} k_proj missing"));
            assert_eq!(k.shape, vec![kv_dim, hidden]);
        }
    }

    /// ROSETTA-003: Tied embeddings flag set in APR metadata.
    #[test]
    fn test_rosetta003_tied_embeddings_flag() {
        let config = PygmyConfig::qwen2_gqa_tied();

        let h = ConversionTestHarness::new()
            .with_safetensors(config)
            .import_to_apr(ImportOptions {
                allow_no_config: true,
                ..ImportOptions::default()
            });

        let output = h.output_path().expect("output exists");
        let data = fs::read(output).expect("read output");
        let reader = AprV2Reader::from_bytes(&data).expect("parse APR");
        let metadata = reader.metadata();

        // Verify lm_head.weight was synthesized (exists in APR)
        assert!(
            reader.get_tensor("lm_head.weight").is_some(),
            "lm_head.weight should be synthesized from embed_tokens"
        );

        // Verify tied_embeddings flag is set
        let is_tied = metadata
            .custom
            .get("tied_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        assert!(
            is_tied,
            "ROSETTA-003: tied_embeddings flag must be set when lm_head was synthesized"
        );
    }

    /// GH-200: map_tensor_names converts GGUF names to HF canonical names.
    #[test]
    fn test_gh200_export_gguf_maps_names() {
        use std::collections::BTreeMap;

        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert("blk.0.attn_q.weight".into(), (vec![0.0; 4], vec![2, 2]));
        tensors.insert("blk.0.attn_k.weight".into(), (vec![0.0; 4], vec![2, 2]));
        tensors.insert("blk.0.ffn_gate.weight".into(), (vec![0.0; 4], vec![2, 2]));
        tensors.insert("token_embd.weight".into(), (vec![0.0; 4], vec![2, 2]));
        tensors.insert("output.weight".into(), (vec![0.0; 4], vec![2, 2]));
        tensors.insert("output_norm.weight".into(), (vec![0.0; 4], vec![2, 2]));

        let mapped = map_tensor_names(&tensors, Architecture::Qwen2);

        assert!(mapped.contains_key("model.layers.0.self_attn.q_proj.weight"));
        assert!(mapped.contains_key("model.layers.0.self_attn.k_proj.weight"));
        assert!(mapped.contains_key("model.layers.0.mlp.gate_proj.weight"));
        assert!(mapped.contains_key("model.embed_tokens.weight"));
        assert!(mapped.contains_key("lm_head.weight"));
        assert!(mapped.contains_key("model.norm.weight"));
        assert!(
            !mapped.contains_key("blk.0.attn_q.weight"),
            "GGUF names must not survive mapping"
        );
    }

    /// GH-200: All 13 GGUF→HF tensor name mappings are correct.
    #[test]
    fn test_gh200_name_mapping_all_tensor_types() {
        use std::collections::BTreeMap;

        let gguf_names = [
            "blk.0.attn_q.weight",
            "blk.0.attn_q.bias",
            "blk.0.attn_k.weight",
            "blk.0.attn_k.bias",
            "blk.0.attn_v.weight",
            "blk.0.attn_v.bias",
            "blk.0.attn_output.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.0.ffn_norm.weight",
            "token_embd.weight",
            "output.weight",
            "output_norm.weight",
        ];

        let expected_hf_names = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.k_proj.bias",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.v_proj.bias",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
        ];

        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        for name in &gguf_names {
            tensors.insert((*name).to_string(), (vec![0.0; 4], vec![2, 2]));
        }

        let mapped = map_tensor_names(&tensors, Architecture::Qwen2);

        for (gguf, hf) in gguf_names.iter().zip(expected_hf_names.iter()) {
            assert!(
                mapped.contains_key(*hf),
                "GGUF '{}' should map to HF '{}', but '{}' not found in mapped keys: {:?}",
                gguf,
                hf,
                hf,
                mapped.keys().collect::<Vec<_>>()
            );
        }

        assert_eq!(
            mapped.len(),
            gguf_names.len(),
            "No tensors should be lost in mapping"
        );
    }

    /// ROSETTA-003: Tensor count matches expected (no fusion reduces count).
    #[test]
    fn test_rosetta003_gqa_tensor_count() {
        let config = PygmyConfig::qwen2_gqa();

        let h = ConversionTestHarness::new()
            .with_safetensors(config)
            .import_to_apr(ImportOptions {
                allow_no_config: true,
                ..ImportOptions::default()
            });

        let output = h.output_path().expect("output exists");
        let data = fs::read(output).expect("read output");
        let reader = AprV2Reader::from_bytes(&data).expect("parse APR");

        let names = reader.tensor_names();

        // Expected per layer: q_proj, k_proj, v_proj, o_proj (weights)
        //                    + q_proj, k_proj, v_proj (biases)
        //                    + input_layernorm, post_attention_layernorm
        //                    + gate_proj, up_proj, down_proj
        // = 4 + 3 + 2 + 3 = 12 per layer
        // Plus: embed_tokens, lm_head, model.norm = 3 global
        // 2 layers * 12 + 3 = 27 total
        assert_eq!(
            names.len(),
            27,
            "GQA model should have 27 tensors (no fusion), got: {}\nTensors: {:?}",
            names.len(),
            names
        );
    }
}
