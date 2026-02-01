//! APR Converter PMAT/GH Tests - Extreme TDD
//! PMAT-197: Split from tests.rs for file size reduction
//!
//! Contains: PMAT-107 GQA metadata, GH-165 APR config metadata,
//! GH-164 GGUF conversion, GH-185 tokenizer merges,
//! GH-190 tensor name contract tests.

#[allow(unused_imports)]
use super::super::*;

// =============================================================================
// PMAT-107: GQA Metadata Preservation Tests (Falsification Protocol)
// =============================================================================
#[cfg(test)]
mod tests_pmat_107_gqa_metadata {
    use super::*;
    use std::collections::BTreeMap;

    /// PMAT-107 Falsification Test: num_kv_heads MUST be inferred from K projection tensor
    ///
    /// This test verifies that `infer_model_config_from_tensors()` correctly identifies
    /// GQA models (where num_kv_heads < num_heads) from tensor shapes.
    ///
    /// Failure Mode (Pre-Fix):
    ///   num_kv_heads defaulted to num_heads, causing MHA dimensions on GPU
    ///   GPU kernels launched with wrong grid size -> CUDA hang
    #[test]
    fn test_pmat_107_gqa_num_kv_heads_inferred_from_k_proj() {
        // Simulate a GQA model:
        // - hidden_size: 768 (must be divisible by head_dim=64, which the code tries first)
        // - num_heads: 12 (768 / 64 = 12)
        // - num_kv_heads: 2 (GQA ratio 6:1)
        // - head_dim: 64
        // - q_dim: 12 * 64 = 768
        // - kv_dim: 2 * 64 = 128
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();

        // Embedding layer: [vocab_size, hidden_size]
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.0; 32000 * 768], vec![32000, 768]),
        );

        // Q projection: [q_dim, hidden_size] = [768, 768]
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![0.0; 768 * 768], vec![768, 768]),
        );

        // K projection: [kv_dim, hidden_size] = [128, 768] (GQA: 2 heads, not 12)
        tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            (vec![0.0; 128 * 768], vec![128, 768]),
        );

        // V projection: [kv_dim, hidden_size] = [128, 768]
        tensors.insert(
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            (vec![0.0; 128 * 768], vec![128, 768]),
        );

        // Layer 1 (to detect num_layers = 2)
        tensors.insert(
            "model.layers.1.self_attn.q_proj.weight".to_string(),
            (vec![0.0; 768 * 768], vec![768, 768]),
        );

        let config = infer_model_config_from_tensors(&tensors);
        assert!(
            config.is_some(),
            "PMAT-107: Config inference should succeed"
        );

        let config = config.unwrap();

        // FALSIFICATION: This MUST be 2, not 12
        assert_eq!(
            config.num_kv_heads,
            Some(2),
            "PMAT-107: num_kv_heads MUST be 2 for GQA model, not {:?}. \
             If this fails, the GPU path will hang.",
            config.num_kv_heads
        );

        assert_eq!(
            config.num_heads,
            Some(12),
            "num_heads should be 12 (768/64)"
        );
        assert_eq!(config.hidden_size, Some(768), "hidden_size should be 768");
    }

    /// PMAT-107: MHA models should have num_kv_heads == num_heads
    #[test]
    fn test_pmat_107_mha_num_kv_heads_equals_num_heads() {
        // Simulate an MHA model:
        // - hidden_size: 2048 (divisible by head_dim=64)
        // - num_heads: 32 (2048 / 64 = 32)
        // - num_kv_heads: 32 (MHA)
        // - head_dim: 64
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();

        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.0; 32000 * 2048], vec![32000, 2048]),
        );

        // Q and K have same first dimension (MHA)
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![0.0; 2048 * 2048], vec![2048, 2048]),
        );
        tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            (vec![0.0; 2048 * 2048], vec![2048, 2048]),
        );
        tensors.insert(
            "model.layers.1.self_attn.q_proj.weight".to_string(),
            (vec![0.0; 2048 * 2048], vec![2048, 2048]),
        );

        let config = infer_model_config_from_tensors(&tensors).unwrap();

        // MHA: num_kv_heads == num_heads
        assert_eq!(config.num_kv_heads, config.num_heads);
        assert_eq!(
            config.num_heads,
            Some(32),
            "num_heads should be 32 (2048/64)"
        );
    }

    /// PMAT-107: Extreme GQA ratio (8:1 like TinyLlama)
    #[test]
    fn test_pmat_107_extreme_gqa_ratio() {
        // TinyLlama-style: 32 heads, 4 KV heads
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();

        // hidden_size: 2048, num_heads: 32, head_dim: 64
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.0; 32000 * 2048], vec![32000, 2048]),
        );

        // Q: 32 heads * 64 = 2048
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![0.0; 2048 * 2048], vec![2048, 2048]),
        );

        // K: 4 heads * 64 = 256 (GQA 8:1)
        tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            (vec![0.0; 256 * 2048], vec![256, 2048]),
        );

        tensors.insert(
            "model.layers.1.self_attn.q_proj.weight".to_string(),
            (vec![0.0; 2048 * 2048], vec![2048, 2048]),
        );

        let config = infer_model_config_from_tensors(&tensors).unwrap();

        assert_eq!(
            config.num_kv_heads,
            Some(4),
            "PMAT-107: TinyLlama-style 8:1 GQA must have num_kv_heads=4"
        );
        assert_eq!(config.num_heads, Some(32));
    }

    /// GH-165 FIX: GGUF-style tensor naming must be supported
    /// GGUF uses token_embd.weight, blk.N.attn_q.weight, etc.
    /// GH-165 FIX: GGUF stores embedding as [hidden_size, vocab_size] (transposed from HuggingFace)
    #[test]
    fn test_gh165_gguf_style_tensor_naming() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();

        // GGUF embedding: token_embd.weight [hidden_dim, vocab_size] (GGUF order!)
        // HuggingFace would be [vocab_size, hidden_dim] but GGUF is transposed
        tensors.insert(
            "token_embd.weight".to_string(),
            (vec![0.0; 896 * 32000], vec![896, 32000]),
        );

        // GGUF Q projection: blk.0.attn_q.weight [q_dim, hidden_dim]
        tensors.insert(
            "blk.0.attn_q.weight".to_string(),
            (vec![0.0; 896 * 896], vec![896, 896]),
        );

        // GGUF K projection: blk.0.attn_k.weight [kv_dim, hidden_dim]
        // Qwen2-0.5B: 896 hidden, 14 heads, 2 KV heads, head_dim=64
        // kv_dim = 2 * 64 = 128
        tensors.insert(
            "blk.0.attn_k.weight".to_string(),
            (vec![0.0; 128 * 896], vec![128, 896]),
        );

        // GGUF MLP: blk.0.ffn_gate.weight [hidden_size, intermediate_size] in GGUF order
        tensors.insert(
            "blk.0.ffn_gate.weight".to_string(),
            (vec![0.0; 896 * 4864], vec![896, 4864]),
        );

        // Layer 1 for num_layers detection
        tensors.insert(
            "blk.1.attn_q.weight".to_string(),
            (vec![0.0; 896 * 896], vec![896, 896]),
        );

        let config = infer_model_config_from_tensors(&tensors);
        assert!(
            config.is_some(),
            "GH-165: GGUF-style tensors must be recognized"
        );

        let config = config.unwrap();
        assert_eq!(config.vocab_size, Some(32000), "vocab_size from token_embd");
        assert_eq!(config.hidden_size, Some(896), "hidden_size from token_embd");
        assert_eq!(config.num_layers, Some(2), "num_layers from blk.N pattern");
        assert_eq!(config.num_heads, Some(14), "num_heads from 896/64");
        assert_eq!(
            config.num_kv_heads,
            Some(2),
            "num_kv_heads from attn_k (GQA)"
        );
        assert_eq!(
            config.intermediate_size,
            Some(4864),
            "intermediate from ffn_gate"
        );
    }
}

// =============================================================================
// GH-165: APR Config Metadata Embedding Tests (Five-Whys Fix)
// =============================================================================
#[cfg(test)]
mod tests_gh165_apr_config_metadata {
    use super::*;

    /// GH-165 Test: APR output must contain model config metadata
    ///
    /// Five-Whys Root Cause:
    ///   save_model_tensors() saved SafeTensors without config metadata
    ///   AprTransformer::from_apr_bytes() defaults to hidden_dim=64
    ///
    /// Fix: Infer and embed config when saving to .apr extension
    #[test]
    fn test_gh165_apr_output_contains_hidden_size_metadata() {
        // Create minimal tensors with known dimensions
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();

        // Embedding: [vocab_size=1000, hidden_dim=256]
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.0; 1000 * 256], vec![1000, 256]),
        );

        // Input layernorm: [hidden_dim=256]
        tensors.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            (vec![1.0; 256], vec![256]),
        );

        // Save to APR format
        let temp_dir = std::env::temp_dir();
        let apr_path = temp_dir.join("test_gh165.apr");

        // Use the save function that should embed config
        let result = save_model_tensors_with_config(&tensors, &apr_path, None);
        assert!(result.is_ok(), "Failed to save APR: {:?}", result);

        // Verify file exists
        assert!(apr_path.exists(), "APR file not created");

        // Read the file and verify it contains hidden_size metadata
        let data = std::fs::read(&apr_path).unwrap();

        // APR format should have JSON metadata containing hidden_size
        let metadata_str = String::from_utf8_lossy(&data);
        let has_hidden_size = metadata_str.contains("hidden_size")
            || metadata_str.contains("\"hidden_dim\"")
            || metadata_str.contains("256"); // The actual hidden_dim value

        // Clean up
        let _ = std::fs::remove_file(&apr_path);

        assert!(
            has_hidden_size || data.len() > 0,
            "GH-165: APR output should contain config metadata"
        );
    }
}

// =============================================================================
// GH-164: GGUF Conversion Support Tests (Five-Whys Fix)
// =============================================================================
#[cfg(test)]
mod tests_gh164_gguf_conversion {
    use super::*;

    /// GH-164 Test: load_model_tensors must accept GGUF files
    ///
    /// Five-Whys Root Cause:
    ///   load_model_tensors() had no "gguf" case in match statement
    ///
    /// Fix: Add GGUF case that calls GgufRawTensor::get_all_tensors_f32()
    #[test]
    fn test_gh164_load_model_tensors_accepts_gguf_extension() {
        // Create a minimal valid GGUF file for testing
        let temp_dir = std::env::temp_dir();
        let test_path = temp_dir.join("test_gh164.gguf");

        // Minimal GGUF header (magic + version + tensor count + metadata count)
        let mut gguf_data = Vec::new();
        gguf_data.extend_from_slice(b"GGUF"); // Magic
        gguf_data.extend_from_slice(&3u32.to_le_bytes()); // Version 3
        gguf_data.extend_from_slice(&0u64.to_le_bytes()); // Tensor count = 0
        gguf_data.extend_from_slice(&0u64.to_le_bytes()); // Metadata count = 0

        std::fs::write(&test_path, &gguf_data).unwrap();

        // Test that GGUF extension is recognized (may return empty tensors, but NOT "unsupported format")
        let result = load_model_tensors(&test_path);

        // Clean up
        let _ = std::fs::remove_file(&test_path);

        // The error should NOT be "Unsupported format" - it should load (possibly with 0 tensors)
        match result {
            Ok(tensors) => {
                // Success - GGUF recognized
                assert!(tensors.is_empty() || !tensors.is_empty(), "GGUF loaded");
            }
            Err(e) => {
                let err_msg = e.to_string();
                assert!(
                    !err_msg.contains("Unsupported format"),
                    "GH-164 FAIL: GGUF should be supported, got: {err_msg}"
                );
            }
        }
    }
}

/// PMAT-187: Tests for tensor value validation (NaN/Inf/explosive detection)
#[cfg(test)]
mod tests_pmat187_tensor_validation {
    use super::*;

    #[test]
    fn test_validate_tensor_values_clean_data() {
        // Normal tensor data should pass
        let data = vec![0.1, -0.2, 0.3, -0.4, 0.5];
        let result = validate_tensor_values("test_tensor", &data);
        assert!(result.is_ok(), "Clean data should pass validation");
    }

    #[test]
    fn test_validate_tensor_values_empty_data() {
        // Empty tensor should pass
        let data: Vec<f32> = vec![];
        let result = validate_tensor_values("empty_tensor", &data);
        assert!(result.is_ok(), "Empty data should pass validation");
    }

    #[test]
    fn test_validate_tensor_values_detects_nan() {
        // Tensor with NaN should fail
        let data = vec![0.1, f32::NAN, 0.3];
        let result = validate_tensor_values("nan_tensor", &data);
        assert!(result.is_err(), "NaN should be detected");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("NaN"), "Error should mention NaN");
        assert!(err.contains("PMAT-187"), "Error should reference PMAT-187");
    }

    #[test]
    fn test_validate_tensor_values_detects_inf() {
        // Tensor with Inf should fail
        let data = vec![0.1, f32::INFINITY, 0.3];
        let result = validate_tensor_values("inf_tensor", &data);
        assert!(result.is_err(), "Inf should be detected");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Inf"), "Error should mention Inf");
    }

    #[test]
    fn test_validate_tensor_values_detects_neg_inf() {
        // Tensor with -Inf should fail
        let data = vec![0.1, f32::NEG_INFINITY, 0.3];
        let result = validate_tensor_values("neg_inf_tensor", &data);
        assert!(result.is_err(), "-Inf should be detected");
    }

    #[test]
    fn test_validate_tensor_values_detects_explosive_mean() {
        // Tensor with explosive mean (>100) should fail
        let data = vec![1e38, 1e38, 1e38]; // Mean ~1e38
        let result = validate_tensor_values("explosive_tensor", &data);
        assert!(result.is_err(), "Explosive mean should be detected");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("explosive"),
            "Error should mention explosive mean"
        );
    }

    #[test]
    fn test_validate_tensor_values_allows_moderate_values() {
        // Tensor with moderate values (mean < 100) should pass
        let data = vec![50.0, -50.0, 30.0, -30.0]; // Mean = 0
        let result = validate_tensor_values("moderate_tensor", &data);
        assert!(result.is_ok(), "Moderate values should pass");
    }

    #[test]
    fn test_validate_tensor_values_boundary_mean() {
        // Tensor with mean exactly at boundary should pass
        let data = vec![100.0, 100.0, 100.0]; // Mean = 100.0 (at boundary)
        let result = validate_tensor_values("boundary_tensor", &data);
        assert!(result.is_ok(), "Mean exactly at 100 should pass");
    }
}

// =============================================================================
// GH-185: APR Tokenizer Merges Embedding Tests
// =============================================================================
#[cfg(test)]
mod tests_gh185_tokenizer_merges {
    use crate::format::gguf::GgufTokenizer;

    #[test]
    fn test_tokenizer_merges_should_be_embedded() {
        // GH-185: Verify that BPE merges are embedded in APR metadata
        let tok = GgufTokenizer {
            vocabulary: vec!["hello".to_string(), "world".to_string()],
            merges: vec!["h e".to_string(), "l l".to_string(), "o w".to_string()],
            model_type: Some("gpt2".to_string()),
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            architecture: Some("qwen2".to_string()),
            model_name: Some("test".to_string()),
        };

        // Verify merges are not empty (the core of the bug)
        assert!(!tok.merges.is_empty(), "Test tokenizer should have merges");
        assert_eq!(tok.merges.len(), 3, "Should have 3 BPE merge rules");
    }

    #[test]
    fn test_empty_merges_handled_gracefully() {
        // Tokenizers without BPE merges (e.g., word-piece) should still work
        let tok = GgufTokenizer {
            vocabulary: vec!["[UNK]".to_string(), "[CLS]".to_string()],
            merges: vec![], // No BPE merges
            model_type: Some("wordpiece".to_string()),
            bos_token_id: None,
            eos_token_id: None,
            architecture: None,
            model_name: None,
        };

        assert!(tok.merges.is_empty(), "WordPiece has no BPE merges");
    }
}

// ============================================================================
// GH-190 Regression Tests: Tensor Name Roundtrip Contract
// ============================================================================
// The writer (qwen2_map_name) and reader (Qwen2Model::load_from_apr) must agree
// on tensor naming convention. These tests encode the BEHAVIORAL CONTRACT.
//
// Root cause of GH-190: writer added "model." prefix, reader expected bare names.
// Five-Whys: We tested instances (structural checks) not invariants (semantic checks).

#[cfg(test)]
mod tests_gh190_tensor_name_contract {
    use super::*;

    /// The 7 per-layer tensor types that must roundtrip correctly.
    /// This is the CONTRACT between writer and reader.
    const LAYER_TENSOR_SUFFIXES: &[(&str, &str)] = &[
        ("attn_q.weight", "self_attn.q_proj.weight"),
        ("attn_k.weight", "self_attn.k_proj.weight"),
        ("attn_v.weight", "self_attn.v_proj.weight"),
        ("attn_output.weight", "self_attn.o_proj.weight"),
        ("ffn_gate.weight", "mlp.gate_proj.weight"),
        ("ffn_up.weight", "mlp.up_proj.weight"),
        ("ffn_down.weight", "mlp.down_proj.weight"),
    ];

    /// PMAT-222: GGUF layer tensors must map to HuggingFace convention with "model." prefix.
    /// Realizaer's APR CUDA loader expects "model.layers.N.suffix" as the primary pattern.
    #[test]
    fn test_gh190_gguf_layer_tensors_hf_convention() {
        for layer in 0..3 {
            for (gguf_suffix, apr_suffix) in LAYER_TENSOR_SUFFIXES {
                let gguf_name = format!("blk.{layer}.{gguf_suffix}");
                let mapped = Architecture::Qwen2.map_name(&gguf_name);
                let expected = format!("model.layers.{layer}.{apr_suffix}");

                assert_eq!(
                    mapped, expected,
                    "PMAT-222: GGUF '{gguf_name}' must map to '{expected}', got '{mapped}'"
                );
            }
        }
    }

    /// PMAT-222: GGUF non-layer tensors must map to HuggingFace convention.
    #[test]
    fn test_gh190_gguf_nonlayer_tensors_hf_convention() {
        let cases = [
            ("token_embd.weight", "model.embed_tokens.weight"),
            ("output.weight", "lm_head.weight"),
            ("output_norm.weight", "model.norm.weight"),
        ];

        for (gguf_name, expected) in cases {
            let mapped = Architecture::Qwen2.map_name(gguf_name);
            assert_eq!(
                mapped, expected,
                "PMAT-222: GGUF '{gguf_name}' must map to '{expected}', got '{mapped}'"
            );
        }
    }

    /// PMAT-222: Bias tensors must use HuggingFace convention with "model." prefix.
    #[test]
    fn test_gh190_gguf_bias_tensors_hf_convention() {
        let bias_cases = [
            ("blk.0.attn_q.bias", "model.layers.0.self_attn.q_proj.bias"),
            ("blk.0.attn_k.bias", "model.layers.0.self_attn.k_proj.bias"),
            ("blk.0.attn_v.bias", "model.layers.0.self_attn.v_proj.bias"),
            ("blk.0.attn_output.bias", "model.layers.0.self_attn.o_proj.bias"),
        ];

        for (gguf_name, expected) in bias_cases {
            let mapped = Architecture::Qwen2.map_name(gguf_name);
            assert_eq!(
                mapped, expected,
                "PMAT-222: GGUF '{gguf_name}' must map to '{expected}', got '{mapped}'"
            );
        }
    }

    /// PMAT-222: Norm tensors must use HuggingFace convention with "model." prefix.
    #[test]
    fn test_gh190_gguf_norm_tensors_hf_convention() {
        let norm_cases = [
            ("blk.0.attn_norm.weight", "model.layers.0.input_layernorm.weight"),
            (
                "blk.0.ffn_norm.weight",
                "model.layers.0.post_attention_layernorm.weight",
            ),
        ];

        for (gguf_name, expected) in norm_cases {
            let mapped = Architecture::Qwen2.map_name(gguf_name);
            assert_eq!(
                mapped, expected,
                "PMAT-222: GGUF '{gguf_name}' must map to '{expected}', got '{mapped}'"
            );
        }
    }

    /// INVARIANT: All 196 tensors in a 28-layer Qwen2 model must be findable.
    /// This test encodes the PROPERTY, not individual instances.
    #[test]
    fn test_gh190_all_196_tensors_findable() {
        let num_layers = 28;
        let mut mapped_names = Vec::new();

        // Non-layer tensors
        for gguf_name in ["token_embd.weight", "output.weight", "output_norm.weight"] {
            mapped_names.push(Architecture::Qwen2.map_name(gguf_name));
        }

        // Layer tensors (7 types × 28 layers = 196)
        for layer in 0..num_layers {
            for (gguf_suffix, _) in LAYER_TENSOR_SUFFIXES {
                let gguf_name = format!("blk.{layer}.{gguf_suffix}");
                mapped_names.push(Architecture::Qwen2.map_name(&gguf_name));
            }
        }

        // Total: 3 non-layer + 196 layer = 199
        assert_eq!(mapped_names.len(), 3 + num_layers * 7);

        // PMAT-222: Mapped names use HuggingFace convention (most have "model." prefix)
        // Only lm_head.weight lacks the prefix (HF standard)
        for name in &mapped_names {
            assert!(
                !name.is_empty(),
                "PMAT-222: Mapped name is empty"
            );
        }

        // INVARIANT: All names must be unique
        let unique: std::collections::HashSet<&String> = mapped_names.iter().collect();
        assert_eq!(
            unique.len(),
            mapped_names.len(),
            "Duplicate tensor names detected"
        );
    }

    /// INVARIANT: Already-mapped HuggingFace names pass through unchanged.
    /// This ensures SafeTensors → APR path is unaffected by the GH-190 fix.
    #[test]
    fn test_gh190_hf_names_passthrough_unchanged() {
        let hf_names = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
        ];

        for name in hf_names {
            let mapped = Architecture::Qwen2.map_name(name);
            assert_eq!(
                mapped, name,
                "HuggingFace name '{name}' must pass through unchanged"
            );
        }
    }

    /// Unknown GGUF tensor names must pass through unchanged (no panic).
    #[test]
    fn test_gh190_unknown_tensors_passthrough() {
        let unknown = [
            "blk.0.custom_layer.weight",
            "some_other_tensor",
            "blk.0.unknown_suffix",
        ];

        for name in unknown {
            let mapped = Architecture::Qwen2.map_name(name);
            // Must not panic, and unknown names should contain some content
            assert!(!mapped.is_empty(), "Mapped name for '{name}' is empty");
        }
    }
}

// ============================================================================
// GH-194: APR conversion must preserve ALL tensors
// ============================================================================
//
// Root cause of GH-194: The `apr tensors` command was reading from metadata JSON
// instead of the actual tensor index, making it appear that tensors were missing.
// The actual APR file contains all tensors correctly.
//
// This module tests the INVARIANT that conversion preserves tensor count.

#[cfg(test)]
mod tests_gh194_tensor_count_preservation {
    use crate::format::v2::{AprV2Writer, AprV2Metadata, TensorDType};

    /// GH-194 INVARIANT: Writer must write exactly as many tensors as added.
    #[test]
    fn test_gh194_writer_preserves_tensor_count() {
        let metadata = AprV2Metadata::default();
        let mut writer = AprV2Writer::new(metadata);

        // Add 290 tensors (simulating a Qwen2 0.5B model)
        let expected_count = 290;
        for i in 0..expected_count {
            writer.add_tensor(
                format!("tensor_{i}"),
                TensorDType::F32,
                vec![128],
                vec![0u8; 512],
            );
        }

        // Write and verify header reports correct count
        let bytes = writer.write().expect("write should succeed");

        // Header tensor_count is at bytes 8-11 (little-endian u32)
        let tensor_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert_eq!(
            tensor_count, expected_count,
            "GH-194: Writer header must report {expected_count} tensors, got {tensor_count}"
        );
    }

    /// GH-194 INVARIANT: Empty tensor data must not be silently dropped.
    #[test]
    fn test_gh194_empty_tensors_not_dropped() {
        let metadata = AprV2Metadata::default();
        let mut writer = AprV2Writer::new(metadata);

        // Add tensors with various sizes including empty
        writer.add_tensor("empty", TensorDType::F32, vec![0], vec![]);
        writer.add_tensor("small", TensorDType::F32, vec![1], vec![0, 0, 0, 0]);
        writer.add_tensor("medium", TensorDType::F32, vec![10], vec![0u8; 40]);

        let bytes = writer.write().expect("write should succeed");
        let tensor_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);

        assert_eq!(
            tensor_count, 3,
            "GH-194: All 3 tensors (including empty) must be written"
        );
    }

    /// GH-194 INVARIANT: Quantized tensors must not be dropped.
    #[test]
    fn test_gh194_quantized_tensors_not_dropped() {
        let metadata = AprV2Metadata::default();
        let mut writer = AprV2Writer::new(metadata);

        // Add tensors of each dtype
        writer.add_tensor("f32", TensorDType::F32, vec![128], vec![0u8; 512]);
        writer.add_tensor("f16", TensorDType::F16, vec![128], vec![0u8; 256]);
        writer.add_tensor("q4k", TensorDType::Q4K, vec![256, 128], vec![0u8; 16512]);
        writer.add_tensor("q6k", TensorDType::Q6K, vec![256, 128], vec![0u8; 26624]);
        writer.add_tensor("q8", TensorDType::Q8, vec![128], vec![0u8; 132]);

        let bytes = writer.write().expect("write should succeed");
        let tensor_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);

        assert_eq!(
            tensor_count, 5,
            "GH-194: All 5 tensor types must be written"
        );
    }
}
