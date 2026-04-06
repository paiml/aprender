#[cfg(test)]
mod tests {
    use crate::convert::*;

    #[test]
    fn test_deep_ccov_stats_with_single_param() {
        // Edge case: single parameter model
        let stats = ConversionStats {
            total_parameters: 1,
            memory_bytes_f32: 4,
            num_layers: 1,
            hidden_dim: 1,
            vocab_size: 1,
            architecture: "minimal".to_string(),
        };

        assert!((stats.parameters_m() - 0.000001).abs() < 0.0000001);
        assert!((stats.parameters_b() - 0.000000001).abs() < 0.0000000001);
        assert!(stats.memory_mb() > 0.0);
        assert!(stats.memory_gb() > 0.0);
    }

    #[test]
    fn test_deep_ccov_from_apr_bytes_empty_tensor_index() {
        // Create APR with empty tensor index (no tensors)
        let mut bytes = vec![0u8; 128];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 2; // v2
        bytes[8..12].copy_from_slice(&0u32.to_le_bytes()); // 0 tensors
        bytes[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata offset
        bytes[20..24].copy_from_slice(&2u32.to_le_bytes()); // metadata size
        bytes[24..32].copy_from_slice(&66u64.to_le_bytes()); // tensor index offset
        bytes[32..40].copy_from_slice(&68u64.to_le_bytes()); // data offset

        bytes[64..66].copy_from_slice(b"{}");
        bytes[66..68].copy_from_slice(b"[]"); // Empty array

        let result = GgufToAprConverter::from_apr_bytes(&bytes);
        assert!(
            result.is_err(),
            "Should fail with no weights tensor: {:?}",
            result
        );
    }

    // ==========================================================================
    // PMAT-107: Falsification Test for GQA num_kv_heads Preservation
    // ==========================================================================
    // This test was added after discovering that APR models hang on GPU
    // because num_kv_heads was being stripped during conversion.
    //
    // Five-Whys Root Cause:
    // 1. Why did APR hang? GPU treated GQA (2 kv_heads) as MHA (12 kv_heads)
    // 2. Why wrong kv_heads? metadata.num_kv_heads was None
    // 3. Why None? APR loading returned default() on parse failure
    // 4. Why parse failure? (We need to verify this is NOT the case)
    // 5. Root cause: Silent failure via unwrap_or_default()

    /// FALSIFICATION TEST: Verify num_kv_heads survives APR round-trip
    /// This test MUST catch the bug where GQA models are converted to MHA.
    #[test]
    fn test_falsification_gqa_num_kv_heads_preserved() {
        use crate::gguf::{GGUFConfig, GGUFTransformerLayer};

        // Create a GQA model: 12 Q heads, 2 KV heads (like Qwen 1.5B)
        let num_heads = 12;
        let num_kv_heads = 2; // GQA: fewer KV heads than Q heads
        let hidden_dim = 64;
        let num_layers = 2;
        let vocab_size = 100;
        let intermediate_dim = 128;

        let config = GGUFConfig {
            architecture: "qwen2".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("qwen2"),
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads, // CRITICAL: This must be preserved!
            vocab_size,
            intermediate_dim,
            context_length: 512,
            rope_theta: 1_000_000.0,
            eps: 1e-6,
            rope_type: 2, // NEOX style
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let layers: Vec<GGUFTransformerLayer> = (0..num_layers)
            .map(|_| GGUFTransformerLayer {
                attn_norm_weight: vec![1.0; hidden_dim],
                attn_norm_bias: None,
                qkv_weight: vec![0.01; hidden_dim * 3 * hidden_dim],
                qkv_bias: None,
                attn_output_weight: vec![0.01; hidden_dim * hidden_dim],
                attn_output_bias: None,
                ffn_gate_weight: None,
                ffn_gate_bias: None,
                ffn_up_weight: vec![0.01; hidden_dim * intermediate_dim],
                ffn_up_bias: None,
                ffn_down_weight: vec![0.01; intermediate_dim * hidden_dim],
                ffn_down_bias: None,
                ffn_norm_weight: None,
                ffn_norm_bias: None,
                attn_q_norm_weight: None,
                attn_k_norm_weight: None,
            })
            .collect();

        let gguf = crate::gguf::GGUFTransformer {
            config,
            token_embedding: vec![0.1; vocab_size * hidden_dim],
            position_embedding: None,
            layers,
            output_norm_weight: vec![1.0; hidden_dim],
            output_norm_bias: None,
            lm_head_weight: vec![0.01; hidden_dim * vocab_size],
            lm_head_bias: None,
        };

        // Step 1: Convert GGUF -> APR Transformer
        let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

        // Verify config is preserved in memory
        assert_eq!(
            apr.config.num_heads, num_heads,
            "num_heads not preserved in AprTransformer"
        );
        assert_eq!(
            apr.config.num_kv_heads, num_kv_heads,
            "FALSIFICATION FAILED: num_kv_heads not preserved in AprTransformer config"
        );

        // Step 2: Serialize to APR bytes
        let apr_bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("Failed to serialize APR");

        // Step 3: Verify num_kv_heads is in the serialized JSON metadata
        // Find the metadata section and check it contains num_kv_heads
        let metadata_json = String::from_utf8_lossy(&apr_bytes[64..512]);
        assert!(
            metadata_json.contains("\"num_kv_heads\":2"),
            "FALSIFICATION FAILED: num_kv_heads not in serialized APR metadata.\n\
             Metadata: {}",
            &metadata_json[..200.min(metadata_json.len())]
        );

        // Step 4: Deserialize back and verify
        let apr_loaded =
            GgufToAprConverter::from_apr_bytes(&apr_bytes).expect("Failed to load APR from bytes");

        assert_eq!(
            apr_loaded.config.num_heads, num_heads,
            "num_heads not preserved after round-trip"
        );
        assert_eq!(
            apr_loaded.config.num_kv_heads, num_kv_heads,
            "FALSIFICATION FAILED: num_kv_heads corrupted after APR round-trip!\n\
             Expected: {}, Got: {}\n\
             This bug causes GPU inference to hang for GQA models.",
            num_kv_heads, apr_loaded.config.num_kv_heads
        );

        println!(
            "✅ FALSIFICATION TEST PASSED: num_kv_heads={} preserved through APR round-trip",
            num_kv_heads
        );
    }
include!("tests_gguf_field_preservation.rs");
include!("tests_get_string.rs");
include!("tests_apr_bytes.rs");
include!("tests_gguf.rs");
include!("tests_deep_ccov.rs");
}
