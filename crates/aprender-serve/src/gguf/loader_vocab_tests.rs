
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::test_factory::{build_minimal_llama_gguf, build_minimal_phi2_gguf};

    #[test]
    fn test_gguf_model_vocabulary_with_empty_strings() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .add_string_array("tokenizer.ggml.tokens", &["", "a", "", "b"])
            .build();
        let model = GGUFModel::from_bytes(&data).unwrap();
        let vocab = model.vocabulary();
        assert!(vocab.is_some());
        let vocab = vocab.unwrap();
        assert_eq!(vocab.len(), 4);
        assert_eq!(vocab[0], "");
        assert_eq!(vocab[1], "a");
    }

    #[test]
    fn test_gguf_model_encode_gpt2_style() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("qwen2")
            .add_string("tokenizer.ggml.model", "gpt2")
            .add_string_array("tokenizer.ggml.tokens", &["a", "b", "c", "ab", "bc"])
            .build();
        let model = GGUFModel::from_bytes(&data).unwrap();
        // With GPT-2 style tokenizer, encoding should find matches
        let tokens = model.encode("a");
        assert!(tokens.is_some());
    }

    #[test]
    fn test_gguf_header_zero_counts() {
        // Valid GGUF with 0 tensors and 0 metadata
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count = 0

        let model = GGUFModel::from_bytes(&data).unwrap();
        assert_eq!(model.header.tensor_count, 0);
        assert_eq!(model.header.metadata_count, 0);
        assert!(model.tensors.is_empty());
        assert!(model.metadata.is_empty());
    }

    #[test]
    fn test_gguf_model_empty_data() {
        let data: Vec<u8> = Vec::new();
        let result = GGUFModel::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_model_only_magic() {
        // Just the magic number, nothing else
        let data = GGUF_MAGIC.to_le_bytes().to_vec();
        let result = GGUFModel::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_model_tensor_data_alignment() {
        use crate::gguf::test_factory::GGUFBuilder;
        // Create model with varying metadata sizes to test alignment
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 64)
            .add_string("some.long.metadata.key", "some value that adds length")
            .add_f32_tensor("test", &[4], &[1.0, 2.0, 3.0, 4.0])
            .build();
        let model = GGUFModel::from_bytes(&data).unwrap();
        // tensor_data_start must always be 32-byte aligned
        assert_eq!(model.tensor_data_start % GGUF_ALIGNMENT, 0);
    }

    // ========================================================================
    // T-COV-95: from_apr + to_apr_bytes roundtrip coverage
    // Targets: from_apr (173 uncov, 0%), to_apr_bytes (212 uncov, 3.6%)
    // ========================================================================

    /// Build a minimal GGUF with output.weight included (required by from_apr)
    fn build_llama_gguf_with_lm_head(
        vocab_size: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> Vec<u8> {
        use crate::gguf::test_factory::*;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
        let norm_data = create_f32_norm_weights(hidden_dim);
        let q_data = create_q4_k_data_2d(hidden_dim, hidden_dim);
        let k_data = create_q4_k_data_2d(hidden_dim, kv_dim);
        let v_data = create_q4_k_data_2d(hidden_dim, kv_dim);
        let attn_out_data = create_q4_k_data_2d(hidden_dim, hidden_dim);
        let ffn_up_data = create_q4_k_data_2d(hidden_dim, intermediate_dim);
        let ffn_down_data = create_q4_k_data_2d(intermediate_dim, hidden_dim);
        let ffn_gate_data = create_q4_k_data_2d(hidden_dim, intermediate_dim);
        let lm_head_data = create_q4_k_data_2d(hidden_dim, vocab_size);

        GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", hidden_dim as u32)
            .num_layers("llama", 1)
            .num_heads("llama", num_heads as u32)
            .num_kv_heads("llama", num_kv_heads as u32)
            .context_length("llama", 256)
            .rope_freq_base("llama", 10000.0)
            .rms_epsilon("llama", 1e-5)
            .ffn_hidden_dim("llama", intermediate_dim as u32)
            .add_f32_tensor(
                "token_embd.weight",
                &[vocab_size as u64, hidden_dim as u64],
                &embed_data,
            )
            .add_f32_tensor("blk.0.attn_norm.weight", &[hidden_dim as u64], &norm_data)
            .add_q4_k_tensor(
                "blk.0.attn_q.weight",
                &[hidden_dim as u64, hidden_dim as u64],
                &q_data,
            )
            .add_q4_k_tensor(
                "blk.0.attn_k.weight",
                &[hidden_dim as u64, kv_dim as u64],
                &k_data,
            )
            .add_q4_k_tensor(
                "blk.0.attn_v.weight",
                &[hidden_dim as u64, kv_dim as u64],
                &v_data,
            )
            .add_q4_k_tensor(
                "blk.0.attn_output.weight",
                &[hidden_dim as u64, hidden_dim as u64],
                &attn_out_data,
            )
            .add_f32_tensor("blk.0.ffn_norm.weight", &[hidden_dim as u64], &norm_data)
            .add_q4_k_tensor(
                "blk.0.ffn_up.weight",
                &[hidden_dim as u64, intermediate_dim as u64],
                &ffn_up_data,
            )
            .add_q4_k_tensor(
                "blk.0.ffn_down.weight",
                &[intermediate_dim as u64, hidden_dim as u64],
                &ffn_down_data,
            )
            .add_q4_k_tensor(
                "blk.0.ffn_gate.weight",
                &[hidden_dim as u64, intermediate_dim as u64],
                &ffn_gate_data,
            )
            .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
            .add_q4_k_tensor(
                "output.weight",
                &[hidden_dim as u64, vocab_size as u64],
                &lm_head_data,
            )
            .build()
    }

    /// Helper: create a valid OwnedQuantizedModel via GGUF→convert→APR→from_apr
    fn build_model_via_gguf_roundtrip() -> (OwnedQuantizedModel, Vec<u8>) {
        use crate::convert::GgufToAprQ4KConverter;
        use std::io::Write as _;

        // 1. Build synthetic GGUF with LM head included
        let gguf_data = build_llama_gguf_with_lm_head(32, 64, 128, 4, 4);

        // 2. Write to temp file
        let mut gguf_tmp = tempfile::NamedTempFile::new().expect("create gguf tmp");
        gguf_tmp.write_all(&gguf_data).expect("write gguf");
        gguf_tmp.flush().expect("flush");

        // 3. Convert GGUF → APR via Q4KConverter
        let apr_tmp = tempfile::NamedTempFile::new().expect("create apr tmp");
        let apr_path = apr_tmp.path().to_path_buf();
        GgufToAprQ4KConverter::convert(gguf_tmp.path(), &apr_path).expect("q4k convert");

        // 4. Load APR via MappedAprModel
        let mapped = crate::apr::MappedAprModel::from_path(&apr_path).expect("map apr");
        let apr_bytes = std::fs::read(&apr_path).expect("read apr");

        // 5. Build OwnedQuantizedModel from APR
        let model = OwnedQuantizedModel::from_apr(&mapped).expect("from_apr");
        (model, apr_bytes)
    }

    #[test]
    fn test_from_apr_loads_config_correctly() {
        let (model, _) = build_model_via_gguf_roundtrip();
        assert_eq!(model.config.architecture, "llama");
        assert_eq!(model.config.hidden_dim, 64);
        assert_eq!(model.config.num_layers, 1);
        assert_eq!(model.config.num_heads, 4);
        // C-03 (Meyer DbC): num_kv_heads defaults to num_heads when absent.
        // Q4K converter writes "num_key_value_heads" but AprMetadata reads "num_kv_heads",
        // so the field is None. Correct default = num_heads (non-GQA) = 4.
        assert_eq!(model.config.num_kv_heads, 4);
        assert_eq!(model.config.intermediate_dim, 128);
    }

    #[test]
    fn test_from_apr_loads_embeddings() {
        let (model, _) = build_model_via_gguf_roundtrip();
        // 32 vocab × 64 hidden = 2048 f32 values
        assert_eq!(model.token_embedding.len(), 32 * 64);
    }

    #[test]
    fn test_from_apr_loads_layers() {
        let (model, _) = build_model_via_gguf_roundtrip();
        assert_eq!(model.layers.len(), 1, "should have 1 layer");
        let layer = &model.layers[0];
        assert_eq!(layer.attn_norm_weight.len(), 64, "attn norm = hidden_dim");
        assert!(layer.ffn_norm_weight.is_some(), "should have ffn norm");
        assert!(layer.ffn_gate_weight.is_some(), "should have ffn gate");
    }

    #[test]
    fn test_from_apr_loads_output_norm() {
        let (model, _) = build_model_via_gguf_roundtrip();
        assert_eq!(model.output_norm_weight.len(), 64);
    }

    #[test]
    fn test_from_apr_loads_lm_head() {
        let (model, _) = build_model_via_gguf_roundtrip();
        assert!(
            !model.lm_head_weight.data.is_empty(),
            "lm head should have data"
        );
    }

    #[test]
    fn test_to_apr_bytes_produces_valid_header() {
        let (model, _) = build_model_via_gguf_roundtrip();
        let apr_bytes = model.to_apr_bytes().expect("to_apr_bytes");
        assert!(apr_bytes.len() >= crate::apr::HEADER_SIZE);
        assert_eq!(&apr_bytes[0..4], &crate::apr::MAGIC);
        assert_eq!(apr_bytes[4], 2, "version major");
    }

    #[test]
    fn test_to_apr_bytes_tensor_count_nonzero() {
        let (model, _) = build_model_via_gguf_roundtrip();
        let apr_bytes = model.to_apr_bytes().expect("to_apr_bytes");
        let tensor_count = u32::from_le_bytes(apr_bytes[8..12].try_into().unwrap());
        // 1 embed + (2 norms + 7 weights) per layer + output norm + lm head
        assert!(
            tensor_count >= 10,
            "should have at least 10 tensors, got {tensor_count}"
        );
    }

    #[test]
    fn test_to_apr_bytes_metadata_valid_json() {
        let (model, _) = build_model_via_gguf_roundtrip();
        let apr_bytes = model.to_apr_bytes().expect("to_apr_bytes");

        let metadata_offset = u64::from_le_bytes(apr_bytes[12..20].try_into().unwrap()) as usize;
        let metadata_len = u32::from_le_bytes(apr_bytes[20..24].try_into().unwrap()) as usize;
        let metadata: serde_json::Value =
            serde_json::from_slice(&apr_bytes[metadata_offset..metadata_offset + metadata_len])
                .expect("valid JSON metadata");

        assert_eq!(metadata["architecture"], "llama");
        assert_eq!(metadata["hidden_size"], 64);
        assert_eq!(metadata["num_layers"], 1);
    }

    #[test]
    fn test_to_apr_bytes_roundtrip_loadable() {
        let (model, _) = build_model_via_gguf_roundtrip();
        let apr_bytes = model.to_apr_bytes().expect("to_apr_bytes");

        // Write to file and load back via MappedAprModel
        let mut tmp = tempfile::NamedTempFile::new().expect("create tmp");
        {
            use std::io::Write;
            tmp.write_all(&apr_bytes).expect("write");
            tmp.flush().expect("flush");
        }

        let mapped = crate::apr::MappedAprModel::from_path(tmp.path()).expect("map");
        let model2 = OwnedQuantizedModel::from_apr(&mapped).expect("from_apr round 2");

        assert_eq!(model2.config.architecture, model.config.architecture);
        assert_eq!(model2.config.hidden_dim, model.config.hidden_dim);
        assert_eq!(model2.config.num_layers, model.config.num_layers);
        assert_eq!(model2.layers.len(), model.layers.len());
        assert_eq!(model2.token_embedding.len(), model.token_embedding.len());
        assert_eq!(
            model2.output_norm_weight.len(),
            model.output_norm_weight.len()
        );
    }

    #[test]
    fn test_to_apr_bytes_data_offset_after_tensor_index() {
        let (model, _) = build_model_via_gguf_roundtrip();
        let apr_bytes = model.to_apr_bytes().expect("to_apr_bytes");

        let tensor_index_offset = u64::from_le_bytes(apr_bytes[24..32].try_into().unwrap());
        let data_offset = u64::from_le_bytes(apr_bytes[32..40].try_into().unwrap());
        assert!(
            data_offset > tensor_index_offset,
            "data offset must follow tensor index"
        );
        assert!(
            (apr_bytes.len() as u64) >= data_offset,
            "file must be at least as large as data offset"
        );
    }

    #[test]
    fn test_to_apr_bytes_metadata_has_model_fields() {
        let (model, _) = build_model_via_gguf_roundtrip();
        let apr_bytes = model.to_apr_bytes().expect("to_apr_bytes");

        let metadata_offset = u64::from_le_bytes(apr_bytes[12..20].try_into().unwrap()) as usize;
        let metadata_len = u32::from_le_bytes(apr_bytes[20..24].try_into().unwrap()) as usize;
        let metadata: serde_json::Value =
            serde_json::from_slice(&apr_bytes[metadata_offset..metadata_offset + metadata_len])
                .expect("valid JSON");

        // to_apr_bytes writes model config fields (not quantization)
        assert_eq!(metadata["architecture"], "llama");
        assert_eq!(metadata["hidden_size"], 64);
        assert_eq!(metadata["num_layers"], 1);
        assert_eq!(metadata["num_heads"], 4);
        assert!(metadata["vocab_size"].as_u64().unwrap() > 0);
    }

    #[test]
    fn test_from_apr_gqa_model() {
        use crate::convert::GgufToAprQ4KConverter;

        // GQA: 8 heads, 2 KV heads
        let gguf_data = build_llama_gguf_with_lm_head(32, 128, 256, 8, 2);
        let mut gguf_tmp = tempfile::NamedTempFile::new().unwrap();
        {
            use std::io::Write;
            gguf_tmp.write_all(&gguf_data).unwrap();
            gguf_tmp.flush().unwrap();
        }

        let apr_tmp = tempfile::NamedTempFile::new().unwrap();
        GgufToAprQ4KConverter::convert(gguf_tmp.path(), apr_tmp.path()).unwrap();

        let mapped = crate::apr::MappedAprModel::from_path(apr_tmp.path()).unwrap();
        let model = OwnedQuantizedModel::from_apr(&mapped).unwrap();

        assert_eq!(model.config.num_heads, 8);
        // PMAT-111: AprMetadata now has alias "num_key_value_heads" for num_kv_heads,
        // so the Q4K converter's "num_key_value_heads" field is correctly parsed.
        // GQA model has 2 KV heads (build_llama_gguf_with_lm_head args: n_head=8, n_head_kv=2).
        assert_eq!(model.config.num_kv_heads, 2);
        assert_eq!(model.config.hidden_dim, 128);
    }
include!("loader_gguf_model.rs");
include!("loader_gguf_model_02.rs");
include!("loader_gguf_read.rs");
}
