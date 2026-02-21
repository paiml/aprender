
/// Insert model config metadata into the custom metadata map.
fn insert_model_config_metadata(
    cfg: &GgufModelConfig,
    custom: &mut std::collections::HashMap<String, serde_json::Value>,
) {
    if let Some(arch) = &cfg.architecture {
        custom.insert(
            "model.architecture".to_string(),
            serde_json::Value::String(arch.clone()),
        );
    }
    if let Some(hidden_size) = cfg.hidden_size {
        custom.insert(
            "model.hidden_size".to_string(),
            serde_json::Value::Number(serde_json::Number::from(hidden_size)),
        );
    }
    if let Some(num_layers) = cfg.num_layers {
        custom.insert(
            "model.num_layers".to_string(),
            serde_json::Value::Number(serde_json::Number::from(num_layers)),
        );
    }
    if let Some(num_heads) = cfg.num_heads {
        custom.insert(
            "model.num_heads".to_string(),
            serde_json::Value::Number(serde_json::Number::from(num_heads)),
        );
    }
    if let Some(num_kv_heads) = cfg.num_kv_heads {
        custom.insert(
            "model.num_kv_heads".to_string(),
            serde_json::Value::Number(serde_json::Number::from(num_kv_heads)),
        );
    }
    if let Some(vocab_size) = cfg.vocab_size {
        custom.insert(
            "model.vocab_size".to_string(),
            serde_json::Value::Number(serde_json::Number::from(vocab_size)),
        );
    }
    if let Some(intermediate_size) = cfg.intermediate_size {
        custom.insert(
            "model.intermediate_size".to_string(),
            serde_json::Value::Number(serde_json::Number::from(intermediate_size)),
        );
    }
    if let Some(max_pos) = cfg.max_position_embeddings {
        custom.insert(
            "model.max_position_embeddings".to_string(),
            serde_json::Value::Number(serde_json::Number::from(max_pos)),
        );
    }
    if let Some(rope_theta) = cfg.rope_theta {
        custom.insert(
            "model.rope_theta".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(f64::from(rope_theta))
                    .unwrap_or_else(|| serde_json::Number::from(10000u64)),
            ),
        );
    }
    if let Some(rms_eps) = cfg.rms_norm_eps {
        custom.insert(
            "model.rms_norm_eps".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(f64::from(rms_eps))
                    .unwrap_or_else(|| serde_json::Number::from(0u64)),
            ),
        );
    }
    // PMAT-114: Write rope_type for correct RoPE style
    if let Some(rope_type) = cfg.rope_type {
        custom.insert(
            "model.rope_type".to_string(),
            serde_json::Value::Number(serde_json::Number::from(rope_type)),
        );
    }
}

/// Map GGUF dtype code to APR `TensorDType`, or return an error for unsupported formats.
///
/// Supported passthrough: F32 (0), F16 (1), Q4_K (12), Q6_K (14).
/// All other dtypes fail with a clear error directing users to `apr convert`.
fn map_gguf_dtype(dtype: u32, tensor_name: &str) -> Result<TensorDType> {
    match dtype {
        0 => Ok(TensorDType::F32),
        1 => Ok(TensorDType::F16),
        12 => Ok(TensorDType::Q4K),
        14 => Ok(TensorDType::Q6K),
        2 | 3 | 6 | 8 | 13 => {
            let (dtype_name, suggestion) = match dtype {
                2 => ("Q4_0", "q4_k"),
                3 => ("Q4_1", "q4_k"),
                6 => ("Q5_0", "q6_k"),
                8 => ("Q8_0", "q6_k"),
                _ => ("Q5_K", "q6_k"),
            };
            Err(AprenderError::FormatError {
                message: format!(
                    "GGUF tensor '{tensor_name}' uses {dtype_name} quantization which APR cannot \
                     represent exactly. Import requires exact format preservation. \
                     Use `apr convert --quantize {suggestion}` to convert to a supported format."
                ),
            })
        }
        7 | 9 => Err(AprenderError::FormatError {
            message: format!(
                "GGUF dtype {dtype} (Q5_1/Q8_1) for tensor '{tensor_name}' not yet supported. \
                 Cannot store raw bytes - would violate LAYOUT-002 mandate."
            ),
        }),
        _ => Err(AprenderError::FormatError {
            message: format!(
                "Unsupported GGUF dtype {dtype} for tensor '{tensor_name}'. \
                 Cannot store raw bytes - would violate LAYOUT-002 mandate."
            ),
        }),
    }
}

/// Write APR file from raw quantized GGUF tensors (preserves Q4_K/Q6_K exactly)
///
/// PMAT-103: This function preserves the original GGUF quantization format,
/// ensuring format parity with Ollama/llama.cpp. No dequantization occurs.
pub(crate) fn write_apr_file_raw(
    tensors: &BTreeMap<String, GgufRawTensor>,
    output: &Path,
    _options: &ImportOptions,
    tokenizer: Option<&GgufTokenizer>,
    model_config: Option<&GgufModelConfig>,
) -> Result<()> {
    // GH-202: Handle tied embeddings (common in Qwen2.5, LLaMA, etc.)
    let (tensors, has_tied_embeddings) = resolve_tied_embeddings(tensors);

    // Calculate total parameter count (approximate - based on shapes)
    let param_count: u64 = tensors
        .values()
        .map(|t| t.shape.iter().product::<usize>() as u64)
        .sum();

    // Build tensor_shapes map for metadata
    // GH-202 FIX: All 2D tensors get shape reversal [ne0, ne1] → [ne1, ne0]
    // This is the standard convention where shape[0]=rows, shape[1]=cols.
    // 1D tensors keep original shape.
    let tensor_shapes: serde_json::Map<String, serde_json::Value> = tensors
        .iter()
        .map(|(name, tensor)| {
            let output_shape = if tensor.shape.len() == 2 {
                vec![tensor.shape[1], tensor.shape[0]]
            } else {
                tensor.shape.clone()
            };
            let shape_array: Vec<serde_json::Value> = output_shape
                .iter()
                .map(|&dim| serde_json::Value::Number(serde_json::Number::from(dim as u64)))
                .collect();
            (name.clone(), serde_json::Value::Array(shape_array))
        })
        .collect();

    // Create metadata
    let mut custom = std::collections::HashMap::new();
    custom.insert(
        "tensor_shapes".to_string(),
        serde_json::Value::Object(tensor_shapes),
    );

    // Add tokenizer data if available (PMAT-171)
    if let Some(tok) = tokenizer {
        insert_tokenizer_metadata(tok, &mut custom);
    }

    // Add model config if available
    if let Some(cfg) = model_config {
        insert_model_config_metadata(cfg, &mut custom);
    }

    // ROSETTA-003: Record tied embeddings for round-trip export fidelity
    if has_tied_embeddings {
        custom.insert("tied_embeddings".to_string(), serde_json::Value::Bool(true));
    }

    // Build metadata using correct AprV2Metadata structure
    let metadata = AprV2Metadata {
        model_type: model_config
            .and_then(|c| c.architecture.clone())
            .unwrap_or_else(|| "qwen2".to_string()),
        name: model_config.and_then(|c| c.architecture.clone()),
        description: Some("GGUF Q4_K model imported with native quantization".to_string()),
        author: None,
        license: None,
        version: Some("1.0.0".to_string()),
        source: None,
        original_format: Some("gguf".to_string()),
        created_at: None,
        total_size: 0, // Will be calculated from tensor data
        param_count,
        quantization: None, // Q4_K stored as raw dtype, not quantization metadata
        sharding: None,
        // GH-253: Propagate chat_template from tokenizer for GGUF round-trip
        chat_template: tokenizer.and_then(|t| t.chat_template.clone()),
        chat_format: None,
        special_tokens: None,
        architecture: model_config.and_then(|c| c.architecture.clone()),
        hidden_size: model_config.and_then(|c| c.hidden_size),
        num_layers: model_config.and_then(|c| c.num_layers),
        num_heads: model_config.and_then(|c| c.num_heads),
        num_kv_heads: model_config.and_then(|c| c.num_kv_heads),
        vocab_size: model_config.and_then(|c| c.vocab_size),
        intermediate_size: model_config.and_then(|c| c.intermediate_size),
        max_position_embeddings: model_config.and_then(|c| c.max_position_embeddings),
        rope_theta: model_config.and_then(|c| c.rope_theta),
        rope_type: model_config.and_then(|c| c.rope_type),
        rms_norm_eps: model_config.and_then(|c| c.rms_norm_eps),
        custom,
    };

    // Create APR writer
    let mut writer = AprV2Writer::new(metadata);

    // GH-202/GH-208: Add tensors with native quantization format.
    // Shape is ALREADY in APR format after enforce_import_contract().
    // GGML data layout is row-major when shape is reversed — no transpose needed.
    for (name, tensor) in tensors {
        let apr_dtype = map_gguf_dtype(tensor.dtype, &name)?;
        writer.add_tensor(name, apr_dtype, tensor.shape.clone(), tensor.data.clone());
    }

    // Write to file
    let bytes = writer.write().map_err(|e| AprenderError::FormatError {
        message: format!("Failed to serialize APR format: {e}"),
    })?;

    let mut file = fs::File::create(output).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to create output file: {e}"),
    })?;

    file.write_all(&bytes)
        .map_err(|e| AprenderError::FormatError {
            message: format!("Failed to write APR file: {e}"),
        })?;

    Ok(())
}
