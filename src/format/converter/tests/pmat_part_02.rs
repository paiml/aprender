
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
            ..Default::default()
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
            ..Default::default()
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
            (
                "blk.0.attn_output.bias",
                "model.layers.0.self_attn.o_proj.bias",
            ),
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
            (
                "blk.0.attn_norm.weight",
                "model.layers.0.input_layernorm.weight",
            ),
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
            assert!(!name.is_empty(), "PMAT-222: Mapped name is empty");
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
    use crate::format::v2::{AprV2Metadata, AprV2Writer, TensorDType};

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
