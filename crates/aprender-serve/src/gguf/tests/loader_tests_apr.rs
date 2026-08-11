
#[test]
fn test_to_apr_bytes_metadata_contains_architecture() {
    let model = build_minimal_owned_quantized_model();
    let bytes = model.to_apr_bytes().expect("should produce bytes");

    // Metadata starts at offset 64 (HEADER_SIZE)
    let metadata_size = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]) as usize;
    let metadata_slice = &bytes[64..64 + metadata_size];
    let metadata_str = String::from_utf8_lossy(metadata_slice);

    assert!(
        metadata_str.contains("\"architecture\":\"llama\""),
        "Metadata should contain architecture: {metadata_str}"
    );
    assert!(
        metadata_str.contains("\"hidden_size\":8"),
        "Metadata should contain hidden_size"
    );
}

// ============================================================================
// to_apr_bytes: Multi-layer model
// ============================================================================

#[test]
fn test_to_apr_bytes_multi_layer() {
    let mut model = build_minimal_owned_quantized_model();
    // Add a second layer
    let second_layer = model.layers[0].clone();
    model.layers.push(second_layer);
    model.config.num_layers = 2;

    let bytes = model.to_apr_bytes().expect("should produce bytes");

    let bytes_str = String::from_utf8_lossy(&bytes);
    assert!(
        bytes_str.contains("blk.0.attn_q.weight"),
        "Should have layer 0"
    );
    assert!(
        bytes_str.contains("blk.1.attn_q.weight"),
        "Should have layer 1"
    );
}

// ============================================================================
// to_apr_bytes: Empty model (zero layers)
// ============================================================================

#[test]
fn test_to_apr_bytes_zero_layers() {
    let mut model = build_minimal_owned_quantized_model();
    model.layers.clear();
    model.config.num_layers = 0;

    let result = model.to_apr_bytes();
    assert!(
        result.is_ok(),
        "Zero-layer model should serialize: {:?}",
        result.err()
    );
}

// ============================================================================
// to_apr_bytes -> from_apr roundtrip
// ============================================================================

#[test]
fn test_to_apr_bytes_roundtrip_via_mapped_model() {
    let model = build_minimal_owned_quantized_model();
    let apr_bytes = model.to_apr_bytes().expect("should produce bytes");

    // Write to a temp file and load via MappedAprModel
    let dir = std::env::temp_dir();
    let path = dir.join("test_roundtrip_loader.apr");
    std::fs::write(&path, &apr_bytes).expect("should write file");

    let mapped = crate::apr::MappedAprModel::from_path(&path);
    assert!(
        mapped.is_ok(),
        "MappedAprModel should load: {:?}",
        mapped.err()
    );

    let mapped = mapped.expect("should load");

    // Verify metadata
    assert_eq!(mapped.metadata.architecture.as_deref(), Some("llama"));
    assert_eq!(mapped.metadata.hidden_size, Some(8));
    assert_eq!(mapped.metadata.num_layers, Some(1));

    // Verify tensor count (should match what we wrote)
    assert!(
        mapped.tensor_count() > 0,
        "Should have tensors loaded from index"
    );

    // Now load from_apr
    let restored = OwnedQuantizedModel::from_apr(&mapped);
    assert!(
        restored.is_ok(),
        "from_apr should succeed: {:?}",
        restored.err()
    );

    let restored = restored.expect("should restore model");
    assert_eq!(restored.config.architecture, "llama");
    assert_eq!(restored.config.hidden_dim, 8);
    assert_eq!(restored.config.num_layers, 1);
    assert_eq!(restored.config.num_heads, 2);
    assert_eq!(restored.config.num_kv_heads, 2);
    assert_eq!(restored.layers.len(), 1);
    assert!(!restored.token_embedding.is_empty());
    assert!(!restored.output_norm_weight.is_empty());

    // Clean up
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_to_apr_bytes_roundtrip_q4k() {
    let model = build_q4k_model();
    let apr_bytes = model.to_apr_bytes().expect("should produce bytes");

    let dir = std::env::temp_dir();
    let path = dir.join("test_roundtrip_q4k_loader.apr");
    std::fs::write(&path, &apr_bytes).expect("should write file");

    let mapped = crate::apr::MappedAprModel::from_path(&path).expect("should load");

    // Verify tensors have correct dtype
    let q_tensor = mapped.find_tensor("blk.0.attn_q.weight");
    assert!(q_tensor.is_some(), "Should find Q tensor");
    assert_eq!(
        q_tensor.expect("Q tensor").dtype,
        "Q4_K",
        "Q tensor should be Q4_K"
    );

    let lm_tensor = mapped.find_tensor("output.weight");
    assert!(lm_tensor.is_some(), "Should find lm_head tensor");
    assert_eq!(
        lm_tensor.expect("lm_head tensor").dtype,
        "Q6_K",
        "lm_head should be Q6_K"
    );

    // Roundtrip from_apr
    let restored = OwnedQuantizedModel::from_apr(&mapped);
    assert!(
        restored.is_ok(),
        "from_apr should succeed for Q4K model: {:?}",
        restored.err()
    );

    let restored = restored.expect("should restore");
    assert_eq!(restored.config.architecture, "qwen2");
    assert_eq!(restored.config.hidden_dim, 8);

    // Clean up
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_to_apr_bytes_roundtrip_fused_qkv() {
    let model = build_fused_qkv_model();
    let apr_bytes = model.to_apr_bytes().expect("should produce bytes");

    let dir = std::env::temp_dir();
    let path = dir.join("test_roundtrip_fused_qkv_loader.apr");
    std::fs::write(&path, &apr_bytes).expect("should write file");

    let mapped = crate::apr::MappedAprModel::from_path(&path).expect("should load");

    // Fused QKV should produce a single "blk.0.attn_qkv.weight" tensor
    let fused = mapped.find_tensor("blk.0.attn_qkv.weight");
    assert!(fused.is_some(), "Should find fused QKV tensor");

    let _ = std::fs::remove_file(&path);
}

// ============================================================================
// to_apr_bytes: Output includes all expected tensor names
// ============================================================================

#[test]
fn test_to_apr_bytes_all_tensor_names_separate_qkv() {
    let model = build_minimal_owned_quantized_model();
    let bytes = model.to_apr_bytes().expect("should produce bytes");
    let bytes_str = String::from_utf8_lossy(&bytes);

    let expected_tensors = [
        "token_embd.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "output_norm.weight",
        "output.weight",
    ];

    for name in &expected_tensors {
        assert!(
            bytes_str.contains(name),
            "Missing tensor name in APR output: {name}"
        );
    }
}

// ============================================================================
// to_apr_bytes: Various qtype mappings
// ============================================================================

#[test]
fn test_to_apr_bytes_various_qtypes() {
    // Build a model with various quantization types to exercise qtype_to_dtype paths
    let hidden_dim = 8;

    fn make_tensor_with_qtype(in_dim: usize, out_dim: usize, qtype: u32) -> OwnedQuantizedTensor {
        // Size doesn't matter for header serialization test; use small data
        OwnedQuantizedTensor {
            data: vec![0u8; 64],
            in_dim,
            out_dim,
            qtype,
        }
    }

    // Test various qtypes: F16(1), Q4_0(2), Q4_1(3), Q5_0(6), Q5_1(7), Q8_0(8), Q8_1(9),
    // Q2_K(10), Q3_K(11), Q4_K(12), Q5_K(13), Q6_K(14), IQ2_XXS(16), IQ2_XS(17), BF16(30)
    let qtypes_to_test = [1u32, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 30, 99];

    for qtype in qtypes_to_test {
        let config = GGUFConfig {
            architecture: "test".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("test"),
            hidden_dim,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 10,
            intermediate_dim: 16,
            context_length: 32,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            query_pre_attn_scalar: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let layer = OwnedQuantizedLayer {
            attn_norm_weight: vec![1.0; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: OwnedQKVWeights::Separate {
                q: make_tensor_with_qtype(hidden_dim, hidden_dim, qtype),
                k: make_tensor_with_qtype(hidden_dim, hidden_dim, qtype),
                v: make_tensor_with_qtype(hidden_dim, hidden_dim, qtype),
            },
            qkv_bias: None,
            attn_output_weight: make_tensor_with_qtype(hidden_dim, hidden_dim, qtype),
            attn_output_bias: None,
            ffn_up_weight: make_tensor_with_qtype(hidden_dim, 16, qtype),
            ffn_up_bias: None,
            ffn_down_weight: make_tensor_with_qtype(16, hidden_dim, qtype),
            ffn_down_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
            post_attn_norm_weight: None,
            post_ffw_norm_weight: None,
        };

        let model = OwnedQuantizedModel {
            config,
            token_embedding: vec![0.0f32; 80],
            position_embedding: None,
            layers: vec![layer],
            encoder_layers: vec![],
            encoder_output_norm_weight: None,
            encoder_output_norm_bias: None,
            output_norm_weight: vec![1.0; hidden_dim],
            output_norm_bias: None,
            lm_head_weight: make_tensor_with_qtype(hidden_dim, 10, qtype),
            lm_head_bias: None,
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        };

        let result = model.to_apr_bytes();
        assert!(
            result.is_ok(),
            "to_apr_bytes should succeed for qtype {}: {:?}",
            qtype,
            result.err()
        );
    }
}

// ============================================================================
// to_apr_bytes: Data size validation
// ============================================================================

#[test]
fn test_to_apr_bytes_total_size_reasonable() {
    let model = build_minimal_owned_quantized_model();
    let bytes = model.to_apr_bytes().expect("should produce bytes");

    // Must be larger than just header
    assert!(bytes.len() > 64, "Must be larger than header");

    // Must not be excessively large for tiny model
    assert!(
        bytes.len() < 100_000,
        "Tiny model should be < 100KB, got {} bytes",
        bytes.len()
    );
}

// ============================================================================
// from_apr: Error paths
// ============================================================================

#[test]
fn test_from_apr_missing_embedding_tensor() {
    // Build an APR file with no embedding tensor
    use crate::apr::{HEADER_SIZE, MAGIC};

    let metadata = r#"{"architecture":"llama","hidden_size":8,"num_layers":1,"num_heads":2,"num_kv_heads":2,"vocab_size":10,"intermediate_size":16,"rms_norm_eps":1e-6}"#;
    let metadata_bytes = metadata.as_bytes();
    let metadata_padded_len = metadata_bytes.len().div_ceil(64) * 64;

    // Empty tensor index (no tensors)
    let tensor_index_bytes: Vec<u8> = Vec::new();

    let metadata_offset = HEADER_SIZE as u64;
    let tensor_index_offset = metadata_offset + metadata_padded_len as u64;
    let data_offset = tensor_index_offset + tensor_index_bytes.len() as u64;

    let mut header = vec![0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(&MAGIC);
    header[4] = 2;
    header[5] = 0;
    header[8..12].copy_from_slice(&0u32.to_le_bytes());
    header[12..20].copy_from_slice(&metadata_offset.to_le_bytes());
    header[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    header[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    header[32..40].copy_from_slice(&data_offset.to_le_bytes());

    let total_size = HEADER_SIZE + metadata_padded_len;
    let mut data = Vec::with_capacity(total_size);
    data.extend_from_slice(&header);
    data.extend_from_slice(metadata_bytes);
    data.resize(total_size, 0);

    let dir = std::env::temp_dir();
    let path = dir.join("test_missing_embed.apr");
    std::fs::write(&path, &data).expect("should write");

    let mapped = crate::apr::MappedAprModel::from_path(&path);
    if let Ok(mapped) = mapped {
        let result = OwnedQuantizedModel::from_apr(&mapped);
        // Should fail because embedding tensor is missing
        assert!(
            result.is_err(),
            "from_apr should fail when embedding is missing"
        );
    }

    let _ = std::fs::remove_file(&path);
}

// ============================================================================
// from_apr: Vocab size inference from embedding tensor shape
// ============================================================================

#[test]
fn test_from_apr_infers_vocab_from_embedding_shape() {
    let model = build_minimal_owned_quantized_model();
    let apr_bytes = model.to_apr_bytes().expect("should produce bytes");

    let dir = std::env::temp_dir();
    let path = dir.join("test_vocab_inference.apr");
    std::fs::write(&path, &apr_bytes).expect("should write");

    let mapped = crate::apr::MappedAprModel::from_path(&path).expect("should load");

    // The metadata has vocab_size, but if it were 0, from_apr should infer from embedding shape
    let restored = OwnedQuantizedModel::from_apr(&mapped).expect("should load model");
    assert_eq!(
        restored.config.vocab_size, 10,
        "Should have correct vocab_size"
    );

    let _ = std::fs::remove_file(&path);
}

// ============================================================================
// from_apr: Config defaults
// ============================================================================

#[test]
fn test_from_apr_uses_metadata_defaults() {
    let model = build_minimal_owned_quantized_model();
    let apr_bytes = model.to_apr_bytes().expect("should produce bytes");

    let dir = std::env::temp_dir();
    let path = dir.join("test_metadata_defaults.apr");
    std::fs::write(&path, &apr_bytes).expect("should write");

    let mapped = crate::apr::MappedAprModel::from_path(&path).expect("should load");
    let restored = OwnedQuantizedModel::from_apr(&mapped).expect("should load model");

    // Verify config was populated from metadata
    assert_eq!(restored.config.hidden_dim, 8);
    assert_eq!(restored.config.num_heads, 2);
    assert_eq!(restored.config.num_kv_heads, 2);
    assert!(restored.config.eps > 0.0);
    assert!(restored.config.rope_theta > 0.0);

    let _ = std::fs::remove_file(&path);
}

// ============================================================================
// PMAT-888: .apr non-Gemma2 post-norm collision (APR-PARITY falsifier)
// ============================================================================

/// PMAT-888 RED→GREEN falsifier: `OwnedQuantizedModel::from_apr` must NOT
/// populate the Gemma2-only `post_attn_norm_weight` / `post_ffw_norm_weight`
/// slots for a non-Gemma2 (`llama`/`qwen2`/...) `.apr`.
///
/// ROOT CAUSE: the HF tensor name `post_attention_layernorm.weight` is the
/// *FFN (pre-feedforward) norm* for llama/qwen2/mistral/phi/deepseek/qwen3 (see
/// `tensor_names_fallback::FfnNormWeight`). PMAT-810b added a Gemma2 post-norm
/// load to the APR loader keyed on that exact HF name, so for every non-Gemma2
/// `.apr` the FFN norm was ALSO loaded into the post-attention-norm slot, and
/// `ffn_block::forward_single_with_cache` — which gates only on `is_some()`, not
/// on arch — applied a spurious extra RMSNorm to the attention output before the
/// residual add. Result: garbage output (`çļĦåıªæĺ¯…`) for every non-Gemma2
/// `.apr` (CPU and GPU), while the byte-identical GGUF ran coherently (the GGUF
/// loader reads the disambiguated `post_attention_norm.weight`, no "layer").
///
/// `build_executable_pygmy_apr` is `architecture == "llama"` and DOES contain
/// `model.layers.0.post_attention_layernorm.weight` (its FFN norm), so it is the
/// exact collision fixture.
///
/// RED before the fix: `post_attn_norm_weight == Some(..)` (the FFN norm).
/// GREEN after the fix: `None` (gated on `config.is_gemma2()`).
///
/// Contract: apr-load-fail-closed-gemma-v1.yaml §NON-GEMMA-APR-POSTNORM-NONE.
#[test]
fn test_pmat888_non_gemma2_apr_has_no_post_attn_norm() {
    let apr_bytes = crate::apr::test_factory::build_executable_pygmy_apr();
    let dir = std::env::temp_dir();
    let path = dir.join("test_pmat888_non_gemma2_postnorm.apr");
    std::fs::write(&path, &apr_bytes).expect("should write file");

    let mapped = crate::apr::MappedAprModel::from_path(&path).expect("should load apr");
    // Sanity: the FFN-norm tensor whose HF name collides with the Gemma2
    // post-attn-norm name IS present in this non-Gemma2 model.
    assert!(
        mapped
            .find_tensor("model.layers.0.post_attention_layernorm.weight")
            .is_some(),
        "fixture must contain the colliding FFN-norm tensor for the test to be load-bearing"
    );

    let model = OwnedQuantizedModel::from_apr(&mapped).expect("from_apr should succeed");
    assert_eq!(model.config.architecture, "llama");
    assert!(!model.config.is_gemma2(), "fixture is non-Gemma2");

    for (i, layer) in model.layers.iter().enumerate() {
        assert!(
            layer.post_attn_norm_weight.is_none(),
            "PMAT-888: non-Gemma2 .apr layer {i} must NOT load post_attn_norm_weight \
             (the HF name post_attention_layernorm.weight is the FFN norm, not a \
             post-attention norm — loading + applying it corrupts all output)"
        );
        assert!(
            layer.post_ffw_norm_weight.is_none(),
            "PMAT-888: non-Gemma2 .apr layer {i} must NOT load post_ffw_norm_weight"
        );
        // The FFN norm itself MUST still be loaded into its correct slot.
        assert!(
            layer.ffn_norm_weight.is_some(),
            "PMAT-888: the post_attention_layernorm.weight tensor must still populate \
             the FFN-norm slot (ffn_norm_weight) for layer {i}"
        );
    }

    let _ = std::fs::remove_file(&path);
}

// ============================================================================
// #2309 / #2441: tied word embeddings — 0-byte lm_head placeholder
// ============================================================================

/// Raw bytes of a named tensor in an APR file, straight out of the mapped model.
fn apr_tensor_bytes(mapped: &crate::apr::MappedAprModel, name: &str) -> Vec<u8> {
    let t = mapped
        .find_tensor(name)
        .unwrap_or_else(|| panic!("fixture must contain {name}"));
    let start = mapped.data_offset() as usize + t.offset as usize;
    mapped.data()[start..start + t.size as usize].to_vec()
}

/// #2309 RED→GREEN falsifier: `apr run` on a tied-embedding `.apr` must decode.
///
/// A `tie_word_embeddings=true` checkpoint converts to an `.apr` whose
/// `lm_head.weight` descriptor carries the full `[vocab, hidden]` shape but ZERO
/// bytes of data — the matrix lives once, under `model.embed_tokens.weight`.
/// `OwnedQuantizedModel::from_apr` registered that descriptor verbatim, so the
/// output projection was an `OwnedQuantizedTensor` with an empty `data` buffer and
/// EVERY decode died in `fused_matmul`:
///
///   Inference failed: Invalid shape: matmul weight has EMPTY data buffer
///   (in_dim=896, out_dim=151936, qtype=0)
///
/// (The error's MoE/#1789 hypothesis is a red herring — the model is not MoE.)
///
/// RED before the fix: `forward` returns that error. GREEN after: the loader ties
/// the head to the embedding matrix and the forward pass produces real logits.
#[test]
fn test_2309_tied_lm_head_placeholder_decodes() {
    let apr_bytes = crate::apr::test_factory::build_executable_pygmy_apr_tied_lm_head_placeholder();
    let dir = std::env::temp_dir();
    let path = dir.join("test_2309_tied_lm_head_placeholder.apr");
    std::fs::write(&path, &apr_bytes).expect("should write file");

    let mapped = crate::apr::MappedAprModel::from_path(&path).expect("should load apr");

    // The fixture must actually BE the defect shape, or this test proves nothing.
    let lm = mapped
        .find_tensor("lm_head.weight")
        .expect("fixture must declare lm_head.weight");
    assert_eq!(lm.size, 0, "fixture's lm_head must be a 0-byte placeholder");
    assert_eq!(
        lm.shape,
        vec![10, 8],
        "the placeholder still carries the full [vocab, hidden] shape"
    );

    let model =
        OwnedQuantizedModel::from_apr(&mapped).expect("#2309: a tied-embedding .apr must load");

    // Behaviour first: the forward pass that used to die on the empty buffer.
    let logits = model
        .forward(&[1u32])
        .expect("#2309: decode must not fail with 'matmul weight has EMPTY data buffer'");
    assert_eq!(logits.len(), 10, "one logit per vocab entry");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "tied lm_head must produce finite logits, got {logits:?}"
    );
    assert!(
        logits.iter().any(|v| *v != 0.0),
        "tied lm_head must produce non-degenerate logits (an all-zero head would \
         also 'succeed'), got {logits:?}"
    );

    // And it must be the RIGHT matrix: the output projection IS the embedding,
    // byte for byte, shaped in_dim=hidden / out_dim=vocab for the logits matmul.
    let embed_bytes = apr_tensor_bytes(&mapped, "model.embed_tokens.weight");
    assert_eq!(
        model.lm_head_weight.data, embed_bytes,
        "#2309: the tied lm_head must be the embedding matrix, byte for byte"
    );
    assert_eq!(model.lm_head_weight.in_dim, 8);
    assert_eq!(model.lm_head_weight.out_dim, 10);

    let _ = std::fs::remove_file(&path);
}

/// #2309, second tie spelling: an `.apr` that OMITS `lm_head.weight` entirely.
///
/// GGUF-derived and explicitly-tied conversions drop the descriptor rather than
/// writing a 0-byte one. That spelling used to fail earlier and louder —
/// "APR: tensor not found (tried: lm_head.weight, output.weight)" — so the quantized
/// loader could not load a tied model under either spelling.
#[test]
fn test_2309_omitted_lm_head_ties_to_embeddings() {
    let apr_bytes = crate::apr::test_factory::build_executable_pygmy_apr_embed_tied();
    let dir = std::env::temp_dir();
    let path = dir.join("test_2309_omitted_lm_head.apr");
    std::fs::write(&path, &apr_bytes).expect("should write file");

    let mapped = crate::apr::MappedAprModel::from_path(&path).expect("should load apr");
    assert!(
        mapped.find_tensor("lm_head.weight").is_none(),
        "fixture must omit lm_head.weight for this test to bite"
    );

    let model = OwnedQuantizedModel::from_apr(&mapped)
        .expect("#2309: an .apr with no lm_head descriptor is tied, not broken");
    let logits = model.forward(&[1u32]).expect("#2309: tied decode must work");
    assert_eq!(logits.len(), 10);
    assert!(logits.iter().all(|v| v.is_finite()));

    let _ = std::fs::remove_file(&path);
}

/// #2309 negative control: a model with a REAL lm_head must keep its own weights.
///
/// Guards the fix from being over-broad — "always tie" would silently replace a
/// genuinely separate output projection with the embedding matrix and change what
/// the model emits.
#[test]
fn test_2309_untied_lm_head_is_not_replaced_by_embeddings() {
    let apr_bytes = crate::apr::test_factory::build_executable_pygmy_apr();
    let dir = std::env::temp_dir();
    let path = dir.join("test_2309_untied_lm_head_control.apr");
    std::fs::write(&path, &apr_bytes).expect("should write file");

    let mapped = crate::apr::MappedAprModel::from_path(&path).expect("should load apr");
    let lm_bytes = apr_tensor_bytes(&mapped, "lm_head.weight");
    let embed_bytes = apr_tensor_bytes(&mapped, "model.embed_tokens.weight");
    assert_ne!(
        lm_bytes, embed_bytes,
        "fixture must have a genuinely different lm_head for this control to bite"
    );

    let model = OwnedQuantizedModel::from_apr(&mapped).expect("untied .apr must load");
    assert_eq!(
        model.lm_head_weight.data, lm_bytes,
        "#2309: an untied lm_head must keep its OWN weights, not the embeddings"
    );

    let _ = std::fs::remove_file(&path);
}
