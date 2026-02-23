
/// Build GGUF architecture metadata from APR model metadata
fn build_gguf_arch_metadata(
    apr_metadata: &crate::format::v2::AprV2Metadata,
) -> Vec<(String, crate::format::gguf::GgufValue)> {
    use crate::format::gguf::GgufValue;

    let arch = resolve_architecture(apr_metadata);
    // C-07 (Meyer DbC): Require dimensions from model metadata — no silent LLaMA-7B defaults.
    // These fields are always populated during import/conversion. If missing, the APR file
    // is malformed and exporting with wrong dimensions would produce a corrupt GGUF.
    let hidden_size = apr_metadata
        .hidden_size
        .expect("C-07: hidden_size required for GGUF export (missing in APR metadata)");
    let num_layers = apr_metadata
        .num_layers
        .expect("C-07: num_layers required for GGUF export (missing in APR metadata)");
    let num_heads = apr_metadata
        .num_heads
        .expect("C-07: num_heads required for GGUF export (missing in APR metadata)");
    let num_kv_heads = apr_metadata.num_kv_heads.unwrap_or(num_heads);
    let vocab_size = apr_metadata
        .vocab_size
        .expect("C-07: vocab_size required for GGUF export (missing in APR metadata)");
    let intermediate_size = apr_metadata
        .intermediate_size
        .expect("C-07: intermediate_size required for GGUF export (missing in APR metadata)");
    let max_pos = apr_metadata.max_position_embeddings.unwrap_or(0);
    // N-01 (Meyer DbC): rope_theta from metadata, or architecture-specific default.
    let rope_theta = apr_metadata.rope_theta.unwrap_or_else(||
        super::export::default_rope_theta_for_architecture(arch));
    let rms_norm_eps = apr_metadata.rms_norm_eps.unwrap_or(1e-6);
    let head_dim = if num_heads > 0 {
        hidden_size / num_heads
    } else {
        0
    };
    let model_name = apr_metadata
        .name
        .clone()
        .unwrap_or_else(|| "model".to_string());

    let mut metadata = vec![
        (
            "general.architecture".to_string(),
            GgufValue::String(arch.to_string()),
        ),
        ("general.name".to_string(), GgufValue::String(model_name)),
        (
            "general.quantization_version".to_string(),
            GgufValue::Uint32(2),
        ),
        ("general.file_type".to_string(), GgufValue::Uint32(0)),
        (
            format!("{arch}.context_length"),
            GgufValue::Uint32(max_pos as u32),
        ),
        (
            format!("{arch}.embedding_length"),
            GgufValue::Uint32(hidden_size as u32),
        ),
        (
            format!("{arch}.block_count"),
            GgufValue::Uint32(num_layers as u32),
        ),
        (
            format!("{arch}.feed_forward_length"),
            GgufValue::Uint32(intermediate_size as u32),
        ),
        (
            format!("{arch}.attention.head_count"),
            GgufValue::Uint32(num_heads as u32),
        ),
        (
            format!("{arch}.attention.head_count_kv"),
            GgufValue::Uint32(num_kv_heads as u32),
        ),
    ];

    // GH-277: GPT-2 uses standard LayerNorm, not RMSNorm
    if arch == "gpt2" {
        metadata.push((
            format!("{arch}.attention.layer_norm_epsilon"),
            GgufValue::Float32(rms_norm_eps),
        ));
    } else {
        metadata.push((
            format!("{arch}.attention.layer_norm_rms_epsilon"),
            GgufValue::Float32(rms_norm_eps),
        ));
    }

    // GH-277: Only emit RoPE keys for architectures that use RoPE
    if uses_rope(arch) {
        metadata.push((
            format!("{arch}.rope.dimension_count"),
            GgufValue::Uint32(head_dim as u32),
        ));
        metadata.push((
            format!("{arch}.rope.freq_base"),
            GgufValue::Float32(rope_theta),
        ));
    }

    metadata.push((
        format!("{arch}.vocab_size"),
        GgufValue::Uint32(vocab_size as u32),
    ));

    metadata
}

/// Push a string array from APR custom fields to GGUF entries.
fn push_string_array(
    entries: &mut Vec<(String, crate::format::gguf::GgufValue)>,
    custom: &std::collections::HashMap<String, serde_json::Value>,
    src_key: &str,
    gguf_key: &str,
) {
    let arr = custom.get(src_key).and_then(|v| v.as_array());
    let Some(arr) = arr else { return };
    let strings: Vec<String> = arr
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    if !strings.is_empty() {
        entries.push((
            gguf_key.to_string(),
            crate::format::gguf::GgufValue::ArrayString(strings),
        ));
    }
}

/// Push a u32 value from APR custom fields to GGUF entries.
fn push_u32_field(
    entries: &mut Vec<(String, crate::format::gguf::GgufValue)>,
    custom: &std::collections::HashMap<String, serde_json::Value>,
    src_key: &str,
    gguf_key: &str,
) {
    if let Some(val) = custom.get(src_key).and_then(|v| v.as_u64()) {
        entries.push((
            gguf_key.to_string(),
            crate::format::gguf::GgufValue::Uint32(val as u32),
        ));
    }
}

/// Push an i32 array from APR custom fields to GGUF entries.
fn push_i32_array(
    entries: &mut Vec<(String, crate::format::gguf::GgufValue)>,
    custom: &std::collections::HashMap<String, serde_json::Value>,
    src_key: &str,
    gguf_key: &str,
) {
    let arr = custom.get(src_key).and_then(|v| v.as_array());
    let Some(arr) = arr else { return };
    let types: Vec<i32> = arr
        .iter()
        .filter_map(|v| v.as_i64().map(|n| n as i32))
        .collect();
    if !types.is_empty() {
        entries.push((
            gguf_key.to_string(),
            crate::format::gguf::GgufValue::ArrayInt32(types),
        ));
    }
}

/// Extract tokenizer metadata from APR custom fields for GGUF export (GH-253)
fn extract_apr_tokenizer_for_gguf(
    apr_metadata: &crate::format::v2::AprV2Metadata,
) -> Vec<(String, crate::format::gguf::GgufValue)> {
    use crate::format::gguf::GgufValue;

    let mut entries = Vec::new();
    let custom = &apr_metadata.custom;
    let arch = resolve_architecture(apr_metadata);

    // Tokenizer model type: "gpt2" for byte-level BPE (Qwen, GPT-2), "llama" for SentencePiece
    // GH-253-3: APR stores raw model_type from GGUF which may be "bpe" — map to "gpt2"
    let raw_model_type = custom
        .get("tokenizer.model")
        .and_then(|v| v.as_str())
        .unwrap_or("gpt2");
    let model_type = match raw_model_type {
        "bpe" => "gpt2",
        other => other,
    };
    entries.push((
        "tokenizer.ggml.model".to_string(),
        GgufValue::String(model_type.to_string()),
    ));
    // GH-277: Use pre-tokenizer type mapping, preferring round-trip preserved value
    let model_name = apr_metadata.name.as_deref().unwrap_or("");
    let pre_type = custom
        .get("tokenizer.pre_type")
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| resolve_pre_tokenizer_type(arch, model_name));
    entries.push((
        "tokenizer.ggml.pre".to_string(),
        GgufValue::String(pre_type.to_string()),
    ));

    push_string_array(
        &mut entries,
        custom,
        "tokenizer.vocabulary",
        "tokenizer.ggml.tokens",
    );
    push_string_array(
        &mut entries,
        custom,
        "tokenizer.merges",
        "tokenizer.ggml.merges",
    );
    push_u32_field(
        &mut entries,
        custom,
        "tokenizer.bos_token_id",
        "tokenizer.ggml.bos_token_id",
    );
    push_u32_field(
        &mut entries,
        custom,
        "tokenizer.eos_token_id",
        "tokenizer.ggml.eos_token_id",
    );
    push_i32_array(
        &mut entries,
        custom,
        "tokenizer.token_type",
        "tokenizer.ggml.token_type",
    );
    push_u32_field(
        &mut entries,
        custom,
        "tokenizer.padding_token_id",
        "tokenizer.ggml.padding_token_id",
    );

    // GH-253-1: add_bos_token flag
    if let Some(add_bos) = custom
        .get("tokenizer.add_bos_token")
        .and_then(|v| v.as_bool())
    {
        entries.push((
            "tokenizer.ggml.add_bos_token".to_string(),
            GgufValue::Bool(add_bos),
        ));
    }

    // GH-253-1: Chat template (Jinja2)
    let chat_tmpl = apr_metadata.chat_template.as_deref().or_else(|| {
        custom
            .get("tokenizer.chat_template")
            .and_then(|v| v.as_str())
    });
    if let Some(tmpl) = chat_tmpl {
        entries.push((
            "tokenizer.chat_template".to_string(),
            GgufValue::String(tmpl.to_string()),
        ));
    }

    entries
}

/// GH-246: Export to MLX format (Apple Silicon).
///
/// MLX models are stored as a directory containing:
/// - `model.safetensors` — weights in SafeTensors format
/// - `config.json` — model configuration (HuggingFace-compatible)
/// - `tokenizer.json` — tokenizer (optional, from APR metadata)
///
/// This reuses the SafeTensors export path since MLX uses SafeTensors as its
/// underlying weight format. The key difference is the directory structure.
fn export_mlx(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    input_path: &Path,
    output_path: &Path,
    options: &ExportOptions,
) -> Result<()> {
    // Output path is the directory
    fs::create_dir_all(output_path).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to create MLX output directory: {e}"),
    })?;

    // Write model.safetensors
    let weights_path = output_path.join("model.safetensors");
    let user_metadata = extract_user_metadata(input_path);
    if user_metadata.is_empty() {
        save_safetensors(&weights_path, tensors).map_err(|e| AprenderError::FormatError {
            message: format!("Failed to write MLX weights: {e}"),
        })?;
    } else {
        save_safetensors_with_metadata(&weights_path, tensors, &user_metadata).map_err(|e| {
            AprenderError::FormatError {
                message: format!("Failed to write MLX weights: {e}"),
            }
        })?;
    }

    // Write config.json
    let config = infer_model_config(tensors);
    let config_path = output_path.join("config.json");
    fs::write(&config_path, config).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to write MLX config.json: {e}"),
    })?;

    // Write tokenizer.json if available
    if options.include_tokenizer {
        let tokenizer_json = infer_tokenizer_json(input_path);
        if !tokenizer_json.is_empty() {
            let tokenizer_path = output_path.join("tokenizer.json");
            if let Err(e) = fs::write(&tokenizer_path, &tokenizer_json) {
                eprintln!("[GH-246] Warning: Failed to write tokenizer.json: {e}");
            }
        }
    }

    Ok(())
}

/// PMAT-252: Raw block passthrough for APR→GGUF export.
///
/// Reads raw tensor bytes directly from APR file (Q4K super-blocks, F32 vectors,
/// etc.) and writes them to GGUF without any dequantization/requantization.
/// This is LOSSLESS for quantized data — zero quality degradation.
///
/// The key insight: APR and GGUF both store Q4K blocks in the same binary format
/// (256-element super-blocks, 144 bytes each). The only differences are:
/// 1. Tensor names (HF convention in APR → GGML convention in GGUF)
/// 2. Shape representation (APR [rows, cols] → GGUF [ne0=cols, ne1=rows])
/// 3. File-level metadata (APR header → GGUF KV pairs)
fn export_apr_to_gguf_raw(input: &Path, output: &Path) -> Result<ExportReport> {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor};
    use crate::format::v2::{AprV2Reader, TensorDType};
    use std::fs::File;
    use std::io::BufWriter;

    let data = fs::read(input).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to read APR file: {e}"),
    })?;
    let original_size = data.len();

    let reader = AprV2Reader::from_bytes(&data).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to parse APR file: {e:?}"),
    })?;

    let apr_metadata = reader.metadata().clone();

    let arch = resolve_architecture(&apr_metadata);
    // C-07 (Meyer DbC): Required dimensions — no silent LLaMA-7B defaults.
    let num_layers = apr_metadata
        .num_layers
        .expect("C-07: num_layers required for GGUF export");
    let num_heads = apr_metadata
        .num_heads
        .expect("C-07: num_heads required for GGUF export");
    let num_kv_heads = apr_metadata.num_kv_heads.unwrap_or(num_heads);
    let hidden_size = apr_metadata
        .hidden_size
        .expect("C-07: hidden_size required for GGUF export");

    // Build metadata from architecture config + tokenizer custom fields
    let mut metadata = build_gguf_arch_metadata(&apr_metadata);
    metadata.extend(extract_apr_tokenizer_for_gguf(&apr_metadata));

    // GH-253-4: Validate metadata completeness before writing
    let validated = ValidatedGgufMetadata::validate(metadata)?;

    eprintln!(
        "[PMAT-252] Writing {} metadata keys (arch={}, layers={}, heads={}/{}kv, hidden={})",
        validated.as_slice().len(),
        arch,
        num_layers,
        num_heads,
        num_kv_heads,
        hidden_size
    );

    // GH-277: Build contract-driven tensor name mapper
    let mapper = build_gguf_mapper(arch);

    // Build GGUF tensors with raw byte passthrough
    let tensor_names = reader.tensor_names();
    let mut gguf_tensors = Vec::with_capacity(tensor_names.len());

    for name in &tensor_names {
        // GH-277: Use contract-driven mapping; skip tensors that return None
        let Some(gguf_name) = mapper.map_name(name) else {
            eprintln!("[GH-277] Skipping tensor '{}' (not in GGUF contract)", name);
            continue;
        };

        let entry = reader
            .get_tensor(name)
            .ok_or_else(|| AprenderError::FormatError {
                message: format!("Tensor '{}' missing from index", name),
            })?;
        let raw_bytes = reader
            .get_tensor_data(name)
            .ok_or_else(|| AprenderError::FormatError {
                message: format!("Tensor '{}' data not found", name),
            })?;

        // Map APR dtype → GGUF dtype (same discriminant values)
        let gguf_dtype = match entry.dtype {
            TensorDType::F32 => GgmlType::F32,
            TensorDType::F16 => GgmlType::F16,
            TensorDType::Q4K => GgmlType::Q4K,
            TensorDType::Q6K => GgmlType::Q6K,
            TensorDType::Q8 => GgmlType::Q8_0,
            _ => GgmlType::F32, // Fallback for BF16, I32, etc.
        };

        // Reverse shape for GGUF: [rows, cols] → [ne0=cols, ne1=rows]
        let gguf_shape = if entry.shape.len() == 2 {
            vec![entry.shape[1] as u64, entry.shape[0] as u64]
        } else {
            entry.shape.iter().map(|&d| d as u64).collect()
        };

        eprintln!(
            "[PMAT-252] '{}': {} bytes (dtype={:?})",
            gguf_name,
            raw_bytes.len(),
            entry.dtype
        );

        gguf_tensors.push(GgufTensor {
            name: gguf_name,
            shape: gguf_shape,
            dtype: gguf_dtype,
            data: raw_bytes.to_vec(),
        });
    }

    // GH-277: Add fused tensors (e.g., QKV fusion for GPT-2)
    let fused = build_fused_tensors_raw(&mapper, &reader);
    gguf_tensors.extend(fused);

    // Write to file
    let file = File::create(output).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to create output file: {e}"),
    })?;
    let mut writer = BufWriter::new(file);

    export_tensors_to_gguf(&mut writer, &gguf_tensors, validated.as_slice())?;

    let exported_size = fs::metadata(output).map(|m| m.len() as usize).unwrap_or(0);

    Ok(ExportReport {
        original_size,
        exported_size,
        tensor_count: gguf_tensors.len(),
        format: ExportFormat::Gguf,
        quantization: Some(QuantizationType::Q4K),
    })
}

/// Legacy mapper for test compatibility.
/// Uses the fallback legacy mapper (same behavior as old hardcoded function).
#[cfg(test)]
fn hf_to_gguf_name(name: &str) -> String {
    let mapper = build_legacy_mapper();
    mapper.map_name(name).unwrap_or_else(|| name.to_string())
}
