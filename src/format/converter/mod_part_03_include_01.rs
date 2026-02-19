
/// Save model tensors to APR format with GGUF config AND tokenizer (PMAT-113 fix)
///
/// This extends `save_model_tensors_with_gguf_config` to also embed the tokenizer
/// vocabulary for standalone APR inference without sibling tokenizer.json files.
fn save_model_tensors_with_gguf_config_and_tokenizer(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    output: &Path,
    _compression: Option<Compression>,
    gguf_config: &GgufModelConfig,
    tokenizer: Option<&GgufTokenizer>,
    quantize: Option<QuantizationType>,
) -> Result<()> {
    // Build AprV2Metadata with GGUF config (not inferred from tensor shapes)
    let mut metadata = AprV2Metadata::new(gguf_config.architecture.as_deref().unwrap_or("qwen2"));
    metadata.original_format = Some("gguf".to_string());
    metadata.model_type = gguf_config
        .architecture
        .clone()
        .unwrap_or_else(|| "qwen2".to_string());
    // PMAT-113 FIX: Set architecture for chat template detection
    metadata.architecture = gguf_config.architecture.clone();

    // Copy all GGUF config fields to APR metadata
    metadata.hidden_size = gguf_config.hidden_size;
    metadata.num_layers = gguf_config.num_layers;
    metadata.num_heads = gguf_config.num_heads;
    metadata.num_kv_heads = gguf_config.num_kv_heads;
    metadata.vocab_size = gguf_config.vocab_size;
    metadata.intermediate_size = gguf_config.intermediate_size;
    metadata.max_position_embeddings = gguf_config.max_position_embeddings;

    // F-REGR-231 FIX: These fields are CRITICAL for correct inference
    metadata.rope_theta = gguf_config.rope_theta;
    metadata.rope_type = gguf_config.rope_type;
    metadata.rms_norm_eps = gguf_config.rms_norm_eps;

    // PMAT-113 FIX: Embed tokenizer vocabulary for standalone APR inference
    if let Some(tok) = tokenizer {
        if !tok.vocabulary.is_empty() {
            eprintln!(
                "[PMAT-113] Embedding {} vocabulary tokens into APR metadata",
                tok.vocabulary.len()
            );
            // Store vocabulary as JSON array in custom metadata
            let vocab_array: Vec<serde_json::Value> = tok
                .vocabulary
                .iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect();
            metadata.custom.insert(
                "tokenizer.vocabulary".to_string(),
                serde_json::Value::Array(vocab_array),
            );
            metadata.custom.insert(
                "tokenizer.vocab_size".to_string(),
                serde_json::Value::Number(serde_json::Number::from(tok.vocabulary.len())),
            );
        }
        if let Some(ref model_type) = tok.model_type {
            metadata.custom.insert(
                "tokenizer.model".to_string(),
                serde_json::Value::String(model_type.clone()),
            );
        }
        if let Some(bos_id) = tok.bos_token_id {
            metadata.custom.insert(
                "tokenizer.bos_token_id".to_string(),
                serde_json::Value::Number(serde_json::Number::from(bos_id)),
            );
        }
        if let Some(eos_id) = tok.eos_token_id {
            metadata.custom.insert(
                "tokenizer.eos_token_id".to_string(),
                serde_json::Value::Number(serde_json::Number::from(eos_id)),
            );
        }
        // PMAT-171: Embed BPE merge rules for standalone APR encoding
        if !tok.merges.is_empty() {
            eprintln!(
                "[PMAT-171] Embedding {} BPE merge rules into APR metadata",
                tok.merges.len()
            );
            let merges_array: Vec<serde_json::Value> = tok
                .merges
                .iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect();
            metadata.custom.insert(
                "tokenizer.merges".to_string(),
                serde_json::Value::Array(merges_array),
            );
        }
    }

    // GH-237: Create writer and add tensors with correct dtype dispatch
    let mut writer = AprV2Writer::new(metadata);
    for (name, (data, shape)) in tensors {
        add_tensor_with_quantization(&mut writer, name, shape, data, quantize);
    }

    // Write to file
    let apr_bytes = writer.write().map_err(|e| AprenderError::FormatError {
        message: format!("Failed to write APR format: {e}"),
    })?;

    fs::write(output, apr_bytes).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to write output file: {e}"),
    })
}

/// Inferred model configuration from tensor shapes for Q4K quantization.
///
/// Used by `save_model_tensors_q4k` to populate APR metadata fields.
struct InferredQ4kConfig {
    hidden_size: Option<usize>,
    num_layers: Option<usize>,
    num_kv_heads: Option<usize>,
    vocab_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_heads: Option<usize>,
}

/// Infer model configuration from tensor names and shapes.
///
/// Scans the tensor map for well-known naming patterns (norm weights, embeddings,
/// layer indices, projection matrices, gate projections) and extracts architecture
/// dimensions. Assumes `head_dim=64` for head-count inference.
fn infer_q4k_config(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> InferredQ4kConfig {
    let mut cfg = InferredQ4kConfig {
        hidden_size: None,
        num_layers: None,
        num_kv_heads: None,
        vocab_size: None,
        intermediate_size: None,
        num_heads: None,
    };

    for (name, (_, shape)) in tensors {
        infer_q4k_single_tensor(&mut cfg, name, shape);
    }

    cfg
}
