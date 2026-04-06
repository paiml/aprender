
// =============================================================================
// POPPERIAN FALSIFICATION TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // LEGACY FUNCTION TESTS (validate_embedding, etc.)
    // =========================================================================

    #[test]
    fn test_validates_good_embedding() {
        let vocab_size = 100;
        let hidden_dim = 64;
        let data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();

        let result = validate_embedding("test", &data, vocab_size, hidden_dim);
        assert!(
            result.passed,
            "Good embedding should pass: {:?}",
            result.failures
        );
    }

    #[test]
    fn test_rejects_mostly_zero_embedding() {
        let vocab_size = 100;
        let hidden_dim = 64;
        let mut data = vec![0.0f32; vocab_size * hidden_dim];
        for i in (vocab_size * 95 / 100 * hidden_dim)..(vocab_size * hidden_dim) {
            data[i] = 0.1;
        }

        let result = validate_embedding("test", &data, vocab_size, hidden_dim);
        assert!(!result.passed, "95% zero embedding should fail");
        assert!(result.failures.iter().any(|f| f.contains("DENSITY")));
    }

    #[test]
    fn test_rejects_nan_embedding() {
        let vocab_size = 10;
        let hidden_dim = 8;
        let mut data: Vec<f32> = vec![0.1; vocab_size * hidden_dim];
        data[5] = f32::NAN;

        let result = validate_embedding("test", &data, vocab_size, hidden_dim);
        assert!(!result.passed, "NaN embedding should fail");
    }

    #[test]
    fn test_rejects_wrong_shape() {
        let data = vec![0.1f32; 1000];
        let result = validate_embedding("test", &data, 100, 64);
        assert!(!result.passed, "Wrong shape should fail");
    }

    // =========================================================================
    // POPPERIAN FALSIFICATION TESTS FOR NEWTYPES (PMAT-235)
    // Per Popper (1959), these attempt to DISPROVE the contract works.
    // =========================================================================

    #[test]
    fn falsify_001_validated_embedding_rejects_all_zeros() {
        let bad_data = vec![0.0f32; 100 * 64];
        let result = ValidatedEmbedding::new(bad_data, 100, 64);
        assert!(result.is_err(), "Should reject 100% zeros");
        let err = result.unwrap_err();
        assert!(err.message.contains("DENSITY"), "Error: {}", err.message);
    }

    #[test]
    fn falsify_001_validated_embedding_rejects_94pct_zeros() {
        // Simulate PMAT-234 bug
        let vocab_size = 1000;
        let hidden_dim = 64;
        let mut data = vec![0.0f32; vocab_size * hidden_dim];
        for i in (945 * hidden_dim)..(vocab_size * hidden_dim) {
            data[i] = 0.1;
        }
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should reject 94.5% zeros");
    }

    #[test]
    fn falsify_001_validated_embedding_accepts_good_data() {
        let vocab_size = 100;
        let hidden_dim = 64;
        let data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(
            result.is_ok(),
            "Should accept good data: {:?}",
            result.err()
        );
    }

    #[test]
    fn falsify_003_validated_embedding_rejects_nan() {
        let vocab_size = 10;
        let hidden_dim = 8;
        let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| i as f32 * 0.01)
            .collect();
        data[5] = f32::NAN;
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should reject NaN");
    }

    #[test]
    fn falsify_004_spot_check_catches_offset_bug() {
        let vocab_size = 100;
        let hidden_dim = 64;
        let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();

        // Zero out token at 10% (token 10)
        let token_10_start = 10 * hidden_dim;
        for i in token_10_start..(token_10_start + hidden_dim) {
            data[i] = 0.0;
        }

        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should catch zero token at 10%");
        assert_eq!(result.unwrap_err().rule_id, "F-DATA-QUALITY-004");
    }

    #[test]
    fn falsify_005_rejects_wrong_shape() {
        let data = vec![0.1f32; 1000];
        let result = ValidatedEmbedding::new(data, 100, 64);
        assert!(result.is_err(), "Should reject wrong shape");
    }

    #[test]
    fn validated_weight_rejects_all_zeros() {
        let data = vec![0.0f32; 100];
        let result = ValidatedWeight::new(data, 10, 10, "test");
        assert!(result.is_err());
    }

    #[test]
    fn validated_weight_accepts_good_data() {
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let result = ValidatedWeight::new(data, 10, 10, "test");
        assert!(result.is_ok());
    }

    #[test]
    fn validated_vector_rejects_wrong_length() {
        let data = vec![0.1f32; 50];
        let result = ValidatedVector::new(data, 100, "test");
        assert!(result.is_err());
    }

    #[test]
    fn validated_vector_accepts_good_data() {
        let data = vec![1.0f32; 100];
        let result = ValidatedVector::new(data, 100, "test");
        assert!(result.is_ok());
    }

    // =========================================================================
    // VALIDATED APR TRANSFORMER FALSIFICATION TESTS (PMAT-235)
    // =========================================================================

    /// Helper: create a valid AprTransformer for testing
    fn make_valid_transformer(num_layers: usize) -> AprTransformer {
        use crate::apr_transformer::{AprTransformerConfig, AprTransformerLayer};

        let hidden_dim = 16;
        let num_heads = 4;
        let num_kv_heads = 4;
        let vocab_size = 32;
        let intermediate_dim = 64;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_out_dim = hidden_dim + 2 * kv_dim;

        let config = AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-6,
            eos_token_id: None,
        ..Default::default()
        };

        // Non-zero sin pattern data
        let make_data = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (i as f32 * 0.01).sin() * 0.1 + 0.05)
                .collect()
        };

        let layers = (0..num_layers)
            .map(|_| AprTransformerLayer {
                attn_norm_weight: vec![1.0; hidden_dim],
                attn_norm_bias: None,
                qkv_weight: make_data(qkv_out_dim * hidden_dim),
                qkv_bias: None,
                attn_output_weight: make_data(hidden_dim * hidden_dim),
                attn_output_bias: None,
                ffn_gate_weight: Some(make_data(intermediate_dim * hidden_dim)),
                ffn_gate_bias: None,
                ffn_up_weight: make_data(intermediate_dim * hidden_dim),
                ffn_up_bias: None,
                ffn_down_weight: make_data(hidden_dim * intermediate_dim),
                ffn_down_bias: None,
                ffn_norm_weight: Some(vec![1.0; hidden_dim]),
                ffn_norm_bias: None,
                attn_q_norm_weight: None,
                attn_k_norm_weight: None,
                linear_attn_z_weight: None,
                linear_attn_b_weight: None,
                linear_attn_a_weight: None,
                linear_attn_conv1d_weight: None,
                linear_attn_a_log: None,
                linear_attn_dt_bias: None,
                linear_attn_norm_weight: None,
                moe_gate_weight: None,
                moe_expert_gate_up: None,
                moe_expert_down: None,
                moe_shared_gate: None,
                moe_shared_up: None,
                moe_shared_down: None,
                moe_shared_expert_gate_weight: None,
            })
            .collect();

        AprTransformer {
            config,
            token_embedding: make_data(vocab_size * hidden_dim),
            layers,
            output_norm_weight: vec![1.0; hidden_dim],
            output_norm_bias: None,
            lm_head_weight: make_data(vocab_size * hidden_dim),
            lm_head_bias: None,
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
        }
    }

    #[test]
    fn falsify_validated_transformer_rejects_nan_embedding() {
        let mut t = make_valid_transformer(1);
        t.token_embedding[5] = f32::NAN;
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(), "Should reject NaN in embedding");
        let err = result.unwrap_err();
        assert!(err.tensor_name.contains("embedding"), "Error: {err}");
    }

    #[test]
    fn falsify_validated_transformer_rejects_zero_layer_weight() {
        let mut t = make_valid_transformer(1);
        // Zero out entire qkv_weight → density gate
        let len = t.layers[0].qkv_weight.len();
        t.layers[0].qkv_weight = vec![0.0; len];
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(), "Should reject all-zero qkv_weight");
        let err = result.unwrap_err();
        assert!(err.tensor_name.contains("qkv_weight"), "Error: {err}");
    }

    #[test]
    fn falsify_validated_transformer_rejects_nan_in_deep_layer() {
        let mut t = make_valid_transformer(4);
        t.layers[3].ffn_up_weight[0] = f32::NAN;
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(), "Should reject NaN in layer 3 ffn_up");
        let err = result.unwrap_err();
        assert!(
            err.tensor_name.contains("layers.3.ffn_up_weight"),
            "Error: {err}"
        );
    }

    #[test]
    fn falsify_validated_transformer_identifies_tensor_name() {
        let mut t = make_valid_transformer(2);
        // Corrupt lm_head_weight
        let len = t.lm_head_weight.len();
        t.lm_head_weight = vec![0.0; len];
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(
            err.tensor_name, "lm_head_weight",
            "Error should name the tensor: {err}"
        );
    }

    #[test]
    fn validated_transformer_accepts_good_model() {
        let t = make_valid_transformer(2);
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_ok(), "Good model should pass: {:?}", result.err());
    }

    #[test]
    fn validated_transformer_deref_transparent_access() {
        let t = make_valid_transformer(1);
        let validated = ValidatedAprTransformer::validate(t).expect("validation should pass");

        // Access config through Deref
        assert_eq!(validated.config.hidden_dim, 16);
        assert_eq!(validated.config.num_layers, 1);
        assert_eq!(validated.config.vocab_size, 32);

        // Access fields through Deref
        assert!(!validated.token_embedding.is_empty());
        assert_eq!(validated.layers.len(), 1);

        // Access through explicit methods
        assert_eq!(validated.config().hidden_dim, 16);
        assert_eq!(validated.transformer().config.num_layers, 1);

        // into_inner works
        let inner = validated.into_inner();
        assert_eq!(inner.config.hidden_dim, 16);
    }

    // =========================================================================
    // GH-46 FALSIFICATION: Rosetta strict validation boundaries
    // =========================================================================

    /// GH-46: Embedding density gate must reject >50% zeros.
    /// Before the fix, validation was too lenient and passed all-zero embeddings.
    #[test]
    fn test_falsify_gh46_embedding_density_threshold_50pct() {
        let vocab = 32_usize;
        let hidden = 16_usize;
        let total = vocab * hidden;
        // 51% zeros — must FAIL
        let zero_count = (total as f64 * 0.51).ceil() as usize;
        let mut data = vec![1.0_f32; total];
        for v in data.iter_mut().take(zero_count) {
            *v = 0.0;
        }
        let result = validate_embedding("test_embed", &data, vocab, hidden);
        assert!(
            !result.passed,
            "GH-46: >50% zero embedding must be rejected, failures: {:?}",
            result.failures
        );
    }

    /// GH-46: Weight density gate must reject >80% zeros.
    /// Before the fix, weights with mostly zeros passed validation silently.
    #[test]
    fn test_falsify_gh46_weight_density_threshold_80pct() {
        let rows = 32_usize;
        let cols = 16_usize;
        let total = rows * cols;
        // 81% zeros — must FAIL
        let zero_count = (total as f64 * 0.81).ceil() as usize;
        let mut data = vec![1.0_f32; total];
        for v in data.iter_mut().take(zero_count) {
            *v = 0.0;
        }
        let result = validate_weight("test_weight", &data, rows, cols);
        assert!(
            !result.passed,
            "GH-46: >80% zero weight must be rejected, failures: {:?}",
            result.failures
        );
    }

    /// GH-46: L2 norm gate must reject flat (constant) tensors.
    /// A constant tensor has zero variance — signals import corruption.
    #[test]
    fn test_falsify_gh46_rejects_flat_tensor() {
        let vocab = 32_usize;
        let hidden = 16_usize;
        // All identical non-zero values — L2 norm > 0 but max-min == 0
        let data = vec![0.5_f32; vocab * hidden];
        let result = validate_embedding("test_flat", &data, vocab, hidden);
        assert!(
            !result.passed,
            "GH-46: Flat (constant) embedding must be rejected, failures: {:?}",
            result.failures
        );
    }

    /// GH-46: NaN gate must catch even a single NaN in embeddings.
    /// Before strict validation, NaN could propagate through inference.
    #[test]
    fn test_falsify_gh46_single_nan_detected() {
        let vocab = 32_usize;
        let hidden = 16_usize;
        let mut data: Vec<f32> = (0..vocab * hidden)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        // Inject single NaN at arbitrary position
        data[vocab * hidden / 2] = f32::NAN;
        let result = validate_embedding("test_nan", &data, vocab, hidden);
        assert!(
            !result.passed,
            "GH-46: Single NaN must be caught by validation"
        );
        assert!(
            result
                .failures
                .iter()
                .any(|f| f.to_lowercase().contains("nan")),
            "GH-46: Failure message must mention NaN"
        );
    }

    // =========================================================================
    // PMAT-299: Architecture completeness falsification tests
    // =========================================================================

    /// FALSIFY-GAP4-001: Qwen3 without QK norm MUST be rejected.
    /// This is the exact GH-279 root cause — Qwen3 missing attn_q_norm produced garbage.
    #[test]
    fn falsify_gap4_001_qwen3_without_qk_norm_rejected() {
        let mut t = make_valid_transformer(2);
        t.config.architecture = "qwen3".to_string();
        // QK norm is None — Qwen3 REQUIRES it
        assert!(t.layers[0].attn_q_norm_weight.is_none());
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(), "Qwen3 without QK norm must be rejected");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("QK norm") || err.message.contains("attn_q_norm"),
            "Error must mention QK norm: {}",
            err.message
        );
    }

    /// FALSIFY-GAP4-002: Qwen3 WITH QK norm passes.
    #[test]
    fn falsify_gap4_002_qwen3_with_qk_norm_passes() {
        let mut t = make_valid_transformer(2);
        t.config.architecture = "qwen3".to_string();
        let head_dim = t.config.hidden_dim / t.config.num_heads;
        for layer in &mut t.layers {
            layer.attn_q_norm_weight = Some(vec![1.0; head_dim]);
            layer.attn_k_norm_weight = Some(vec![1.0; head_dim]);
        }
        let result = ValidatedAprTransformer::validate(t);
        assert!(
            result.is_ok(),
            "Qwen3 with QK norm should pass: {:?}",
            result.err()
        );
    }

    /// FALSIFY-GAP4-003: Qwen2 without bias MUST be rejected.
    #[test]
    fn falsify_gap4_003_qwen2_without_bias_rejected() {
        let mut t = make_valid_transformer(1);
        t.config.architecture = "qwen2".to_string();
        assert!(t.layers[0].qkv_bias.is_none());
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(), "Qwen2 without bias must be rejected");
    }

    /// FALSIFY-GAP4-004: LLaMA without QK norm or bias is fine.
    #[test]
    fn falsify_gap4_004_llama_without_optional_passes() {
        let mut t = make_valid_transformer(2);
        t.config.architecture = "llama".to_string();
        let result = ValidatedAprTransformer::validate(t);
        assert!(
            result.is_ok(),
            "LLaMA should pass without QK norm or bias: {:?}",
            result.err()
        );
    }

    /// FALSIFY-GAP4-005: Missing QK norm detected on ANY layer, not just layer 0.
    #[test]
    fn falsify_gap4_005_qwen3_missing_norm_on_later_layer() {
        let mut t = make_valid_transformer(4);
        t.config.architecture = "qwen3".to_string();
        let head_dim = t.config.hidden_dim / t.config.num_heads;
        for layer in &mut t.layers {
            layer.attn_q_norm_weight = Some(vec![1.0; head_dim]);
            layer.attn_k_norm_weight = Some(vec![1.0; head_dim]);
        }
        // Remove QK norm from layer 3 only
        t.layers[3].attn_k_norm_weight = None;
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(), "Must catch missing QK norm on layer 3");
        let err = result.unwrap_err();
        assert!(
            err.tensor_name.contains("3"),
            "Error must identify layer 3: {}",
            err.tensor_name
        );
    }

    // =========================================================================
    // FALSIFY-E6: Embedding contract gap analysis (Refs PMAT-325, PMAT-327)
    //
    // Five-Whys: §2.1.1 "What Are Embeddings" falsification sweep
    //   Why 1: GPU path could silently load garbage embeddings
    //   Why 2: GPU validates shape but not data quality
    //   Why 3: ValidatedEmbedding not wired into GGUF load path
    //   Why 4: GGUF loader predates ValidatedEmbedding
    //   Why 5: No test existed to verify ALL load paths enforce same gates
    //
    // Popper (1959): "These tests try to break the claim that
    // ValidatedEmbedding prevents ALL embedding garbage across ALL paths."
    // =========================================================================

    /// FALSIFY-E6a: ValidatedEmbedding rejects Inf values
    #[test]
    fn falsify_e6a_embedding_rejects_inf() {
        let vocab_size = 10;
        let hidden_dim = 8;
        let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        data[7] = f32::INFINITY;
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "FALSIFY-E6a: Should reject Inf in embedding");
        assert_eq!(result.unwrap_err().rule_id, "F-DATA-QUALITY-002");
    }

    /// FALSIFY-E6b: ValidatedEmbedding rejects NEG_INFINITY
    #[test]
    fn falsify_e6b_embedding_rejects_neg_inf() {
        let vocab_size = 10;
        let hidden_dim = 8;
        let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        data[3] = f32::NEG_INFINITY;
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "FALSIFY-E6b: Should reject -Inf in embedding");
        assert_eq!(result.unwrap_err().rule_id, "F-DATA-QUALITY-002");
    }

    /// FALSIFY-E6c: ValidatedEmbedding rejects near-zero L2 norm
    #[test]
    fn falsify_e6c_embedding_rejects_near_zero_l2() {
        let vocab_size = 10;
        let hidden_dim = 8;
        // Values above zero threshold but L2 < 1e-6
        let data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| 1e-8 + (i as f32) * 1e-12)
            .collect();
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "FALSIFY-E6c: Near-zero L2 embedding must be rejected");
        assert_eq!(result.unwrap_err().rule_id, "F-DATA-QUALITY-003");
    }

    /// FALSIFY-E6d: ValidatedEmbedding catches trailing corruption at 90% of vocab
    #[test]
    fn falsify_e6d_spot_check_catches_trailing_corruption() {
        let vocab_size = 100;
        let hidden_dim = 64;
        let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        // Zero out token at 90% of vocab
        let token_90_start = 90 * hidden_dim;
        for v in &mut data[token_90_start..token_90_start + hidden_dim] {
            *v = 0.0;
        }
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "FALSIFY-E6d: Must catch zero token at 90%");
        assert_eq!(result.unwrap_err().rule_id, "F-DATA-QUALITY-004");
    }

    /// FALSIFY-E6e: Zero vocab_size produces empty embedding — must be rejected
    #[test]
    fn falsify_e6e_zero_vocab_size_rejected() {
        let result = ValidatedEmbedding::new(vec![], 0, 64);
        assert!(result.is_err(), "FALSIFY-E6e: vocab_size=0 must be rejected");
    }

    /// FALSIFY-E6f: Zero hidden_dim produces empty embedding — must be rejected
    #[test]
    fn falsify_e6f_zero_hidden_dim_rejected() {
        let result = ValidatedEmbedding::new(vec![], 100, 0);
        assert!(result.is_err(), "FALSIFY-E6f: hidden_dim=0 must be rejected");
    }

    /// FALSIFY-E6g: ValidatedAprTransformer rejects truncated embedding
    ///
    /// Simulates PMAT-327: embedding_weights.len() < vocab_size * hidden_dim.
    /// The Validated path must catch this at construction, preventing runtime OOB.
    #[test]
    fn falsify_e6g_truncated_embedding_rejected() {
        let mut t = make_valid_transformer(1);
        // Truncate embedding: remove last 10 elements
        let len = t.token_embedding.len();
        t.token_embedding.truncate(len - 10);
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(), "FALSIFY-E6g: Truncated embedding must be rejected");
        let err = result.unwrap_err();
        assert!(err.tensor_name.contains("embedding"),
            "Error must identify embedding: {}", err);
    }

    /// FALSIFY-E6h: ValidatedAprTransformer rejects oversized embedding
    ///
    /// If embedding has MORE data than vocab*hidden, it could indicate
    /// wrong vocab_size in config or concatenated garbage.
    #[test]
    fn falsify_e6h_oversized_embedding_rejected() {
        let mut t = make_valid_transformer(1);
        // Add 10 extra elements
        for _ in 0..10 {
            t.token_embedding.push(0.1);
        }
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(), "FALSIFY-E6h: Oversized embedding must be rejected");
    }

    // =========================================================================
    // FALSIFY-L: §2.1.2 LM Head Contract — Five-Whys Gap Analysis (Refs PMAT-328)
    //
    // Contract: tensor-layout-v1.yaml §tensors.lm_head
    //   apr_shape: "[vocab, hidden]"
    //   kernel: "matmul_q*k_rowmajor(W, x, vocab_size, hidden_dim)"
    //   critical: "true"
    //
    // Five-Whys:
    //   Why 1: GPU/GGUF lm_head could produce wrong logits
    //   Why 2: GGUF path skips ValidatedWeight for lm_head
    //   Why 3: GGUF load predates ValidatedAprTransformer
    //   Why 4: No GGUF→ValidatedWeight bridge exists
    //   Why 5: No test verified GGUF lm_head goes through validation
    //
    // Popper (1959): "These tests attempt to falsify the claim that
    // realizar's lm_head handling prevents garbage logit output."
    // =========================================================================

    /// FALSIFY-L1a: ValidatedWeight rejects wrong-shape lm_head
    ///
    /// If lm_head data length != vocab_size * hidden_dim, it's structural corruption.
    #[test]
    fn falsify_l1a_validated_weight_rejects_wrong_shape_lm_head() {
        // 100*64=6400 elements but declared as 200*64=12800
        let data: Vec<f32> = (0..6400)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        let result = ValidatedWeight::new(data, 200, 64, "lm_head.weight");
        assert!(result.is_err(),
            "FALSIFY-L1a: Wrong-shape lm_head must be rejected");
        assert_eq!(result.unwrap_err().rule_id, "F-LAYOUT-CONTRACT-001");
    }

    /// FALSIFY-L1b: ValidatedWeight rejects all-NaN lm_head
    ///
    /// All-NaN lm_head → all-NaN logits → argmax(NaN) → token 0 = [PAD].
    /// This is exactly the GH-202 failure mode.
    #[test]
    fn falsify_l1b_validated_weight_rejects_nan_lm_head() {
        let data = vec![f32::NAN; 100 * 64];
        let result = ValidatedWeight::new(data, 100, 64, "lm_head.weight");
        assert!(result.is_err(),
            "FALSIFY-L1b: All-NaN lm_head must be rejected — produces [PAD] garbage");
    }

    /// FALSIFY-L1c: ValidatedWeight rejects Inf lm_head
    #[test]
    fn falsify_l1c_validated_weight_rejects_inf_lm_head() {
        let mut data: Vec<f32> = (0..100 * 64)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        data[42] = f32::INFINITY;
        let result = ValidatedWeight::new(data, 100, 64, "lm_head.weight");
        assert!(result.is_err(),
            "FALSIFY-L1c: Inf in lm_head must be rejected");
    }

    /// FALSIFY-L2: ValidatedAprTransformer catches corrupted lm_head
    ///
    /// The full validation pipeline must reject a transformer with all-zero lm_head.
    /// This is the SafeTensors path validation.
    #[test]
    fn falsify_l2_validated_transformer_catches_zero_lm_head() {
        let mut t = make_valid_transformer(1);
        let len = t.lm_head_weight.len();
        t.lm_head_weight = vec![0.0; len];
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(),
            "FALSIFY-L2: All-zero lm_head must be rejected by ValidatedAprTransformer");
        let err = result.unwrap_err();
        assert_eq!(err.tensor_name, "lm_head_weight",
            "Error must identify lm_head_weight: {}", err);
    }

    /// FALSIFY-L3: ValidatedAprTransformer catches truncated lm_head
    ///
    /// If lm_head has fewer elements than vocab*hidden (e.g., wrong vocab_size
    /// in config.json), shape check must fire.
    #[test]
    fn falsify_l3_validated_transformer_catches_truncated_lm_head() {
        let mut t = make_valid_transformer(1);
        let len = t.lm_head_weight.len();
        t.lm_head_weight.truncate(len - 10);
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(),
            "FALSIFY-L3: Truncated lm_head must be rejected");
        let err = result.unwrap_err();
        assert!(err.tensor_name.contains("lm_head"),
            "Error must identify lm_head: {}", err);
    }

    /// FALSIFY-L4: ValidatedAprTransformer catches oversized lm_head
    ///
    /// If lm_head has MORE elements than vocab*hidden (e.g., 152064 data with
    /// config claiming 151936), the extra elements are unreachable garbage.
    #[test]
    fn falsify_l4_validated_transformer_catches_oversized_lm_head() {
        let mut t = make_valid_transformer(1);
        for _ in 0..10 {
            t.lm_head_weight.push(0.1);
        }
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(),
            "FALSIFY-L4: Oversized lm_head must be rejected");
    }

    /// FALSIFY-L5: ValidatedAprTransformer catches NaN in lm_head
    #[test]
    fn falsify_l5_validated_transformer_catches_nan_lm_head() {
        let mut t = make_valid_transformer(1);
        t.lm_head_weight[10] = f32::NAN;
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(),
            "FALSIFY-L5: NaN in lm_head must be rejected by full validation pipeline");
    }

    // =========================================================================
    // FALSIFY-A: §2.1.3 Attention Projections — Five-Whys Gap Analysis (Refs PMAT-330)
    //
    // Contract: tensor-layout-v1.yaml §tensors.q_proj/k_proj/v_proj/o_proj
    //   Fused QKV: [hidden + 2*kv_dim, hidden]
    //   attn_output: [hidden, hidden]
    //
    // Five-Whys:
    //   Why 1: Attention could produce wrong Q/K/V splits
    //   Why 2: Fused QKV concatenation order wrong (KQV instead of QKV)
    //   Why 3: No contractual test for QKV fusion layout
    //   Why 4: GQA kv_dim != hidden_dim makes wrong fusion silently corrupt
    //   Why 5: ValidatedWeight validates total size but not internal layout
    //
    // Popper (1959): "These tests attempt to falsify the claim that
    // realizar's attention projection validation prevents garbage attention."
    // =========================================================================

    /// FALSIFY-A1r: ValidatedAprTransformer catches truncated QKV weight
    ///
    /// If fused QKV has fewer elements than (hidden + 2*kv_dim) * hidden,
    /// the Q/K/V split will read past buffer bounds.
    #[test]
    fn falsify_a1r_validated_transformer_catches_truncated_qkv() {
        let mut t = make_valid_transformer(1);
        let len = t.layers[0].qkv_weight.len();
        t.layers[0].qkv_weight.truncate(len - 10);
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(),
            "FALSIFY-A1r: Truncated QKV weight must be rejected");
        let err = result.unwrap_err();
        assert!(err.tensor_name.contains("qkv"),
            "Error must identify qkv_weight: {}", err);
    }

    /// FALSIFY-A2r: ValidatedAprTransformer catches all-zero QKV
    ///
    /// All-zero Q → identical attention scores → uniform attention → garbage.
    #[test]
    fn falsify_a2r_validated_transformer_catches_zero_qkv() {
        let mut t = make_valid_transformer(1);
        let len = t.layers[0].qkv_weight.len();
        t.layers[0].qkv_weight = vec![0.0; len];
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(),
            "FALSIFY-A2r: All-zero QKV weight must be rejected");
    }

    /// FALSIFY-A3r: ValidatedAprTransformer catches NaN in QKV
    #[test]
    fn falsify_a3r_validated_transformer_catches_nan_qkv() {
        let mut t = make_valid_transformer(1);
        t.layers[0].qkv_weight[5] = f32::NAN;
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(),
            "FALSIFY-A3r: NaN in QKV weight must be rejected");
    }

    /// FALSIFY-A4r: ValidatedAprTransformer catches truncated attn_output
    #[test]
    fn falsify_a4r_validated_transformer_catches_truncated_attn_output() {
        let mut t = make_valid_transformer(1);
        let len = t.layers[0].attn_output_weight.len();
        t.layers[0].attn_output_weight.truncate(len - 5);
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(),
            "FALSIFY-A4r: Truncated attn_output must be rejected");
    }

    /// FALSIFY-A5r: ValidatedAprTransformer catches all-zero attn_output
    ///
    /// All-zero o_proj → attention output is always zero → layer produces
    /// only the residual connection → attention is effectively disabled.
    #[test]
    fn falsify_a5r_validated_transformer_catches_zero_attn_output() {
        let mut t = make_valid_transformer(1);
        let len = t.layers[0].attn_output_weight.len();
        t.layers[0].attn_output_weight = vec![0.0; len];
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(),
            "FALSIFY-A5r: All-zero attn_output must be rejected — disables attention");
    }

    /// FALSIFY-A6r: ValidatedWeight correctly validates GQA fused QKV dimensions
    ///
    /// For GQA with num_kv_heads < num_heads, the fused QKV shape is
    /// [hidden + 2*kv_dim, hidden] which is NOT square.
    #[test]
    fn falsify_a6r_validated_weight_gqa_fused_qkv_shape() {
        let hidden = 16;
        let num_heads = 4;
        let num_kv_heads = 2; // GQA: 2 KV heads, 4 Q heads
        let head_dim = hidden / num_heads; // 4
        let kv_dim = num_kv_heads * head_dim; // 8
        let qkv_out = hidden + 2 * kv_dim; // 16 + 16 = 32

        // Correct shape: [qkv_out=32, hidden=16] = 512 elements
        let data: Vec<f32> = (0..qkv_out * hidden)
            .map(|i| (i as f32 * 0.01).sin() * 0.1 + 0.05)
            .collect();
        let result = ValidatedWeight::new(data, qkv_out, hidden, "qkv_weight");
        assert!(result.is_ok(),
            "FALSIFY-A6r: GQA fused QKV [32, 16] must be valid: {:?}", result.err());

        // Wrong shape (MHA assumption: qkv_out = 3*hidden = 48 instead of 32)
        let wrong_qkv_out = 3 * hidden; // MHA assumption: 48
        let wrong_data: Vec<f32> = (0..qkv_out * hidden) // 512 elements
            .map(|i| (i as f32 * 0.01).sin() * 0.1 + 0.05)
            .collect();
        let result = ValidatedWeight::new(wrong_data, wrong_qkv_out, hidden, "qkv_weight");
        assert!(result.is_err(),
            "FALSIFY-A6r: MHA-assumed shape with GQA data must fail");
    }

    // =========================================================================
    // FALSIFY-F: §2.1.4 FFN Projections — Five-Whys Gap Analysis (Refs PMAT-333)
    //
    // Contract: tensor-layout-v1.yaml §tensors.gate_proj/up_proj/down_proj
    //   gate_proj: apr_shape: "[intermediate, hidden]", transpose: "true"
    //   up_proj:   apr_shape: "[intermediate, hidden]", transpose: "true"
    //   down_proj:  apr_shape: "[hidden, intermediate]", transpose: "true"
    //   SwiGLU: FFN(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    //
    // Five-Whys:
    //   Why 1: FFN could produce wrong-dimension hidden states
    //   Why 2: gate_proj/up_proj must share shape, down_proj must reverse
    //   Why 3: ValidatedWeight validates total size but not gate==up constraint
    //   Why 4: SwiGLU element-wise multiply requires exact shape match
    //   Why 5: No test enforced gate_proj.shape == up_proj.shape across stack
    //
    // Popper (1959): "These tests attempt to falsify the claim that
    // realizar's FFN validation prevents garbage hidden states."
    // =========================================================================

    /// FALSIFY-F1r: ValidatedAprTransformer catches truncated gate_proj
    ///
    /// If gate_proj has fewer elements than intermediate*hidden, SwiGLU
    /// element-wise multiply will read garbage or OOB.
    #[test]
    fn falsify_f1r_validated_transformer_catches_truncated_gate() {
        let mut t = make_valid_transformer(1);
        if let Some(ref mut gate) = t.layers[0].ffn_gate_weight {
            let len = gate.len();
            gate.truncate(len - 5);
        }
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(),
            "FALSIFY-F1r: Truncated gate_proj must be rejected");
        let err = result.unwrap_err();
        assert!(err.tensor_name.contains("ffn_gate"),
            "Error must identify ffn_gate_weight: {}", err);
    }

    /// FALSIFY-F2r: ValidatedAprTransformer catches all-zero gate_proj
    ///
    /// All-zero gate_proj → SiLU(0)=0 → SwiGLU output always zero →
    /// FFN contributes nothing → only residual passes through.
    #[test]
    fn falsify_f2r_validated_transformer_catches_zero_gate() {
        let mut t = make_valid_transformer(1);
        if let Some(ref mut gate) = t.layers[0].ffn_gate_weight {
            let len = gate.len();
            *gate = vec![0.0; len];
        }
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(),
            "FALSIFY-F2r: All-zero gate_proj must be rejected — disables FFN");
    }

    /// FALSIFY-F3r: ValidatedAprTransformer catches NaN in up_proj
    ///
    /// NaN in up_proj → NaN * SiLU(gate) = NaN → down_proj(NaN) = NaN →
    /// residual + NaN = NaN → cascade through all subsequent layers.
    #[test]
    fn falsify_f3r_validated_transformer_catches_nan_up() {
        let mut t = make_valid_transformer(1);
        t.layers[0].ffn_up_weight[0] = f32::NAN;
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(),
            "FALSIFY-F3r: NaN in up_proj must be rejected");
    }

    /// FALSIFY-F4r: ValidatedAprTransformer catches truncated down_proj
    ///
    /// down_proj maps [intermediate] → [hidden]. Wrong size means the
    /// residual add will have mismatched dimensions.
    #[test]
    fn falsify_f4r_validated_transformer_catches_truncated_down() {
        let mut t = make_valid_transformer(1);
        let len = t.layers[0].ffn_down_weight.len();
        t.layers[0].ffn_down_weight.truncate(len - 5);
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(),
            "FALSIFY-F4r: Truncated down_proj must be rejected");
        let err = result.unwrap_err();
        assert!(err.tensor_name.contains("ffn_down"),
            "Error must identify ffn_down_weight: {}", err);
    }

    /// FALSIFY-F5r: ValidatedAprTransformer catches all-zero down_proj
    ///
    /// All-zero down_proj → output always zero → only residual → FFN disabled.
    #[test]
    fn falsify_f5r_validated_transformer_catches_zero_down() {
        let mut t = make_valid_transformer(1);
        let len = t.layers[0].ffn_down_weight.len();
        t.layers[0].ffn_down_weight = vec![0.0; len];
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(),
            "FALSIFY-F5r: All-zero down_proj must be rejected — disables FFN");
    }

    // =========================================================================
    // FALSIFY-N: §2.1.5-6 Layer Norms — Five-Whys Gap Analysis (Refs PMAT-332)
    //
    // Contract: tensor-layout-v1.yaml §tensors.input_layernorm/post_attention_layernorm/final_norm
    //   apr_shape: "[hidden]"
    //   transpose: "false"
    //   kernel: "element-wise multiply"
    //
    // Five-Whys:
    //   Why 1: Wrong norm weights produce wrong scale → garbage activations
    //   Why 2: Norms are 1D — only length and data quality matter
    //   Why 3: ValidatedVector validates length + NaN/Inf but not zero-length
    //   Why 4: Zero-length norm → division by zero in RMS computation
    //   Why 5: No minimum-length gate in ValidatedVector (PMAT-332)
    //
    // Popper (1959): "These tests attempt to falsify the claim that
    // realizar's norm validation prevents garbage normalization."
    // =========================================================================

    /// FALSIFY-N1r: ValidatedAprTransformer catches wrong-length attn_norm
    ///
    /// attn_norm.len() must == hidden_dim. If shorter, element-wise multiply
    /// reads past buffer; if longer, extra elements are unreachable garbage.
    #[test]
    fn falsify_n1r_validated_transformer_catches_wrong_attn_norm() {
        let mut t = make_valid_transformer(1);
        // Truncate attn_norm to wrong length
        t.layers[0].attn_norm_weight = vec![1.0; 8]; // hidden=16, so 8 is wrong
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(),
            "FALSIFY-N1r: Wrong-length attn_norm must be rejected");
        let err = result.unwrap_err();
        assert!(err.tensor_name.contains("attn_norm"),
            "Error must identify attn_norm: {}", err);
    }

    /// FALSIFY-N2r: ValidatedAprTransformer catches NaN in ffn_norm
    #[test]
    fn falsify_n2r_validated_transformer_catches_nan_ffn_norm() {
        let mut t = make_valid_transformer(1);
        if let Some(ref mut norm) = t.layers[0].ffn_norm_weight {
            norm[0] = f32::NAN;
        }
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(),
            "FALSIFY-N2r: NaN in ffn_norm must be rejected");
    }

    /// FALSIFY-N3r: ValidatedAprTransformer catches wrong-length output_norm
    ///
    /// output_norm is the final norm before lm_head. Wrong length →
    /// wrong-scale hidden state → garbage logits.
    #[test]
    fn falsify_n3r_validated_transformer_catches_wrong_output_norm() {
        let mut t = make_valid_transformer(1);
        t.output_norm_weight = vec![1.0; 8]; // hidden=16, so 8 is wrong
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(),
            "FALSIFY-N3r: Wrong-length output_norm must be rejected");
        let err = result.unwrap_err();
        assert!(err.tensor_name.contains("output_norm"),
            "Error must identify output_norm: {}", err);
    }

    /// FALSIFY-N4r: ValidatedAprTransformer catches NaN in output_norm
    ///
    /// NaN in final norm → all hidden dims become NaN → lm_head logits all NaN →
    /// argmax(NaN) → token 0 = [PAD]. Exact GH-202 failure mode.
    #[test]
    fn falsify_n4r_validated_transformer_catches_nan_output_norm() {
        let mut t = make_valid_transformer(1);
        t.output_norm_weight[5] = f32::NAN;
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(),
            "FALSIFY-N4r: NaN in output_norm must be rejected — produces GH-202 garbage");
    }

    /// FALSIFY-N5r: ValidatedAprTransformer catches Inf in attn_norm
    ///
    /// Inf norm weight → Inf after multiply → softmax(Inf) = NaN →
    /// attention weights become NaN → cascade failure.
    #[test]
    fn falsify_n5r_validated_transformer_catches_inf_attn_norm() {
        let mut t = make_valid_transformer(1);
        t.layers[0].attn_norm_weight[3] = f32::INFINITY;
        let result = ValidatedAprTransformer::validate(t);
        assert!(result.is_err(),
            "FALSIFY-N5r: Inf in attn_norm must be rejected");
    }
}

// T-COV-95 Coverage Bridge (Part 02 - Accessors, error paths, optional biases)
#[cfg(test)]
#[path = "validation_tests_02.rs"]
mod validation_tests_02;
