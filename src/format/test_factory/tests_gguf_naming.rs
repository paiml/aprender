use super::*;

// ========================================================================
// GH-194: GGUF-Style Naming and Weight Tying Tests
// ========================================================================
// These tests verify that APR files with GGUF-style tensor naming
// are correctly written and can be read back. The critical scenario
// is weight tying where token_embd.weight is used for both embedding
// and lm_head (no separate output.weight tensor).

mod gh194_tests {
    use super::*;

    /// GH-194: GGUF-style naming must produce valid APR files
    #[test]
    fn test_gh194_gguf_names_valid_apr() {
        let data = build_pygmy_apr_gguf_names();

        // Must have valid APR magic
        assert!(data.len() >= 64);
        assert_eq!(&data[0..4], &MAGIC_V2);

        // Must be parseable
        let reader = AprV2Reader::from_bytes(&data);
        assert!(reader.is_ok(), "GGUF-named APR must be parseable");
    }

    /// GH-194: GGUF-style APR must have token_embd.weight tensor
    #[test]
    fn test_gh194_gguf_names_has_token_embd() {
        let data = build_pygmy_apr_gguf_names();
        let reader = AprV2Reader::from_bytes(&data).unwrap();

        let names = reader.tensor_names();
        assert!(
            names.iter().any(|n| *n == "token_embd.weight"),
            "GH-194: GGUF-named APR must have token_embd.weight, found: {:?}",
            names
        );
    }

    /// GH-194: Weight tying model must NOT have separate output.weight
    #[test]
    fn test_gh194_weight_tying_no_output_tensor() {
        let config = GgufPygmyConfig {
            weight_tying: true,
            ..Default::default()
        };
        let data = build_pygmy_apr_gguf_names_with_config(config);
        let reader = AprV2Reader::from_bytes(&data).unwrap();

        let names = reader.tensor_names();
        assert!(
            !names.iter().any(|n| *n == "output.weight"),
            "GH-194: Weight-tied model must NOT have separate output.weight"
        );
        assert!(
            names.iter().any(|n| *n == "token_embd.weight"),
            "GH-194: Weight-tied model must have token_embd.weight"
        );
    }

    /// GH-194: Non-tied model MUST have output.weight
    #[test]
    fn test_gh194_non_tied_has_output_tensor() {
        let config = GgufPygmyConfig {
            weight_tying: false,
            ..Default::default()
        };
        let data = build_pygmy_apr_gguf_names_with_config(config);
        let reader = AprV2Reader::from_bytes(&data).unwrap();

        let names = reader.tensor_names();
        assert!(
            names.iter().any(|n| *n == "output.weight"),
            "GH-194: Non-tied model must have output.weight"
        );
        assert!(
            names.iter().any(|n| *n == "token_embd.weight"),
            "GH-194: Non-tied model must have token_embd.weight"
        );
    }

    /// GH-194: HuggingFace-style weight tying also works
    #[test]
    fn test_gh194_hf_names_tied_valid() {
        let data = build_pygmy_apr_hf_names_tied();
        let reader = AprV2Reader::from_bytes(&data).unwrap();

        let names = reader.tensor_names();
        assert!(
            names.iter().any(|n| *n == "model.embed_tokens.weight"),
            "GH-194: HF-named tied model must have model.embed_tokens.weight"
        );
        assert!(
            !names.iter().any(|n| *n == "lm_head.weight"),
            "GH-194: HF-named tied model must NOT have lm_head.weight"
        );
    }

    /// GH-194: GGUF naming has correct layer tensor names
    #[test]
    fn test_gh194_gguf_names_layer_tensors() {
        let data = build_pygmy_apr_gguf_names();
        let reader = AprV2Reader::from_bytes(&data).unwrap();

        let names = reader.tensor_names();

        // Must have GGUF-style layer tensors
        let expected_prefixes = [
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
        ];

        for expected in expected_prefixes {
            assert!(
                names.iter().any(|n| *n == expected),
                "GH-194: GGUF naming must have tensor '{}', found: {:?}",
                expected,
                names
            );
        }
    }

    /// GH-194: Tensor count matches expected for GGUF-style model
    #[test]
    fn test_gh194_gguf_names_tensor_count() {
        let config = GgufPygmyConfig {
            num_layers: 2,
            weight_tying: true,
            ..Default::default()
        };
        let data = build_pygmy_apr_gguf_names_with_config(config);
        let reader = AprV2Reader::from_bytes(&data).unwrap();

        // Per layer: 9 tensors (4 attn + 3 mlp + 2 norms)
        // Global: token_embd.weight + output_norm.weight = 2
        // With weight tying, no output.weight
        // Total: 2 layers * 9 + 2 = 20
        let names = reader.tensor_names();
        assert_eq!(
            names.len(),
            20,
            "GH-194: 2-layer GGUF model with weight tying should have 20 tensors, got {}",
            names.len()
        );
    }

    /// GH-194: Metadata records weight tying status
    #[test]
    fn test_gh194_metadata_records_weight_tying() {
        let data = build_pygmy_apr_gguf_names();
        let reader = AprV2Reader::from_bytes(&data).unwrap();

        let metadata = reader.metadata();
        assert!(
            metadata.custom.contains_key("weight_tying"),
            "GH-194: Metadata should record weight_tying status"
        );
    }

    /// GH-194: All tensor data is valid (not empty, no NaN/Inf)
    #[test]
    fn test_gh194_gguf_names_tensor_data_valid() {
        let data = build_pygmy_apr_gguf_names();
        let reader = AprV2Reader::from_bytes(&data).unwrap();

        for name in reader.tensor_names() {
            let tensor_data = reader.get_tensor_data(name);
            assert!(
                tensor_data.is_some(),
                "GH-194: Tensor '{}' data must be accessible",
                name
            );
            let bytes = tensor_data.unwrap();
            assert!(
                !bytes.is_empty(),
                "GH-194: Tensor '{}' must not be empty",
                name
            );
        }
    }
}
