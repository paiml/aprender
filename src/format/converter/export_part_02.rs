
/// Export to SafeTensors with optional companion files (config.json, tokenizer.json)
fn export_safetensors_with_companions(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    input_path: &Path,
    output_path: &Path,
    options: &ExportOptions,
    original_dtypes: &BTreeMap<String, String>,
) -> Result<()> {
    // PMAT-223: Extract user metadata from APR custom field for round-trip
    let user_metadata = extract_user_metadata(input_path);

    // PMAT-260: Log dtype preservation when BF16/F16 tensors are present
    let non_f32_count = original_dtypes
        .values()
        .filter(|d| d.as_str() != "F32")
        .count();
    if non_f32_count > 0 {
        eprintln!(
            "[PMAT-260] Preserving original dtypes for {non_f32_count} non-F32 tensors (BF16/F16)"
        );
    }

    if user_metadata.is_empty() {
        save_safetensors_typed(output_path, tensors, original_dtypes).map_err(|e| {
            AprenderError::FormatError {
                message: format!("Failed to export to SafeTensors: {e}"),
            }
        })?;
    } else {
        eprintln!(
            "[PMAT-223] Restoring {} user metadata key(s) to SafeTensors __metadata__",
            user_metadata.len()
        );
        save_safetensors_with_metadata_typed(
            output_path,
            tensors,
            &user_metadata,
            original_dtypes,
        )
        .map_err(|e| AprenderError::FormatError {
            message: format!("Failed to export to SafeTensors: {e}"),
        })?;
    }

    // GH-182: Write companion files alongside SafeTensors
    let output_dir = output_path.parent().unwrap_or(Path::new("."));

    if options.include_config {
        let config = infer_model_config(tensors);
        let config_path = output_dir.join("config.json");
        if let Err(e) = fs::write(&config_path, config) {
            eprintln!("[GH-182] Warning: Failed to write config.json: {e}");
        }
    }

    if options.include_tokenizer {
        let tokenizer_json = infer_tokenizer_json(input_path);
        if !tokenizer_json.is_empty() {
            let tokenizer_path = output_dir.join("tokenizer.json");
            if let Err(e) = fs::write(&tokenizer_path, &tokenizer_json) {
                eprintln!("[GH-182] Warning: Failed to write tokenizer.json: {e}");
            }
        }
    }

    Ok(())
}

/// Export tensors to GGUF format (GGUF-EXPORT-001 fix)
///
/// Reads APR metadata to populate GGUF KV pairs and maps tensor names
/// from HuggingFace convention to GGUF convention.
///
/// BUG-1 FIX: Now supports Q4_K quantization for GGUF inference compatibility.
/// F32 GGUF files don't work with realizar's fused matmul kernels.
///
/// BUG-EXPORT-004 FIX: Now includes tokenizer metadata for realizar inference.
/// Without BOS/EOS token IDs, the model produces empty output.
/// Resolved GGUF export configuration (APR metadata with inferred fallbacks).
struct GgufExportConfig {
    arch: String,
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    vocab_size: usize,
    intermediate_size: usize,
    max_pos: usize,
    rope_theta: f32,
    rms_norm_eps: f32,
    head_dim: usize,
    model_name: String,
}

/// Resolve GGUF export config from APR metadata + inferred fallbacks.
fn resolve_gguf_config(
    apr_metadata: Option<&crate::format::v2::AprV2Metadata>,
    inferred: Option<&crate::format::gguf::GgufModelConfig>,
) -> GgufExportConfig {
    /// Resolve a field: APR metadata → inferred → default.
    fn resolve<T: Copy>(
        apr: Option<&crate::format::v2::AprV2Metadata>,
        inf: Option<&crate::format::gguf::GgufModelConfig>,
        apr_f: impl Fn(&crate::format::v2::AprV2Metadata) -> Option<T>,
        inf_f: impl Fn(&crate::format::gguf::GgufModelConfig) -> Option<T>,
        default: T,
    ) -> T {
        apr.and_then(&apr_f)
            .or_else(|| inf.and_then(&inf_f))
            .unwrap_or(default)
    }

    let num_heads = resolve(apr_metadata, inferred, |m| m.num_heads, |c| c.num_heads, 32);
    let hidden_size = resolve(
        apr_metadata,
        inferred,
        |m| m.hidden_size,
        |c| c.hidden_size,
        4096,
    );

    GgufExportConfig {
        arch: apr_metadata
            .and_then(|m| m.architecture.clone())
            .or_else(|| inferred.and_then(|c| c.architecture.clone()))
            .unwrap_or_else(|| "qwen2".to_string()),
        hidden_size,
        num_layers: resolve(
            apr_metadata,
            inferred,
            |m| m.num_layers,
            |c| c.num_layers,
            32,
        ),
        num_heads,
        num_kv_heads: resolve(
            apr_metadata,
            inferred,
            |m| m.num_kv_heads,
            |c| c.num_kv_heads,
            num_heads,
        ),
        vocab_size: resolve(
            apr_metadata,
            inferred,
            |m| m.vocab_size,
            |c| c.vocab_size,
            32000,
        ),
        intermediate_size: resolve(
            apr_metadata,
            inferred,
            |m| m.intermediate_size,
            |c| c.intermediate_size,
            11008,
        ),
        max_pos: apr_metadata
            .and_then(|m| m.max_position_embeddings)
            .unwrap_or(32768),
        rope_theta: apr_metadata
            .and_then(|m| m.rope_theta)
            .unwrap_or(1_000_000.0),
        rms_norm_eps: apr_metadata.and_then(|m| m.rms_norm_eps).unwrap_or(1e-6),
        head_dim: if num_heads > 0 {
            hidden_size / num_heads
        } else {
            128
        },
        model_name: apr_metadata
            .and_then(|m| m.name.clone())
            .unwrap_or_else(|| "model".to_string()),
    }
}

/// Build GGUF architecture metadata KV pairs from resolved config.
fn build_gguf_config_metadata(
    cfg: &GgufExportConfig,
) -> Vec<(String, crate::format::gguf::GgufValue)> {
    use crate::format::gguf::GgufValue;
    let arch = &cfg.arch;
    let mut metadata = vec![
        (
            "general.architecture".to_string(),
            GgufValue::String(arch.clone()),
        ),
        (
            "general.name".to_string(),
            GgufValue::String(cfg.model_name.clone()),
        ),
        (
            "general.quantization_version".to_string(),
            GgufValue::Uint32(2),
        ),
        ("general.file_type".to_string(), GgufValue::Uint32(0)),
        (
            format!("{arch}.context_length"),
            GgufValue::Uint32(cfg.max_pos as u32),
        ),
        (
            format!("{arch}.embedding_length"),
            GgufValue::Uint32(cfg.hidden_size as u32),
        ),
        (
            format!("{arch}.block_count"),
            GgufValue::Uint32(cfg.num_layers as u32),
        ),
        (
            format!("{arch}.feed_forward_length"),
            GgufValue::Uint32(cfg.intermediate_size as u32),
        ),
        (
            format!("{arch}.attention.head_count"),
            GgufValue::Uint32(cfg.num_heads as u32),
        ),
        (
            format!("{arch}.attention.head_count_kv"),
            GgufValue::Uint32(cfg.num_kv_heads as u32),
        ),
    ];

    // GH-277: GPT-2 uses standard LayerNorm, not RMSNorm
    if arch == "gpt2" {
        metadata.push((
            format!("{arch}.attention.layer_norm_epsilon"),
            GgufValue::Float32(cfg.rms_norm_eps),
        ));
    } else {
        metadata.push((
            format!("{arch}.attention.layer_norm_rms_epsilon"),
            GgufValue::Float32(cfg.rms_norm_eps),
        ));
    }

    // GH-277: Only emit RoPE keys for architectures that use RoPE
    if uses_rope(arch) {
        metadata.push((
            format!("{arch}.rope.dimension_count"),
            GgufValue::Uint32(cfg.head_dim as u32),
        ));
        metadata.push((
            format!("{arch}.rope.freq_base"),
            GgufValue::Float32(cfg.rope_theta),
        ));
    }

    metadata.push((
        format!("{arch}.vocab_size"),
        GgufValue::Uint32(cfg.vocab_size as u32),
    ));

    metadata
}

/// Build tokenizer metadata KV pairs for GGUF export.
fn build_tokenizer_gguf_metadata(
    tokenizer: &crate::format::gguf::GgufTokenizer,
    arch: &str,
    model_name: &str,
) -> Vec<(String, crate::format::gguf::GgufValue)> {
    use crate::format::gguf::GgufValue;
    let mut metadata = Vec::new();
    let model_type = tokenizer.model_type.as_deref().unwrap_or("gpt2");

    metadata.push((
        "tokenizer.ggml.model".to_string(),
        GgufValue::String(model_type.to_lowercase()),
    ));
    // GH-277: Use pre-tokenizer type mapping, preferring round-trip preserved value
    let pre_type = tokenizer
        .pre_type
        .as_deref()
        .unwrap_or_else(|| resolve_pre_tokenizer_type(arch, model_name));
    metadata.push((
        "tokenizer.ggml.pre".to_string(),
        GgufValue::String(pre_type.to_string()),
    ));

    if let Some(bos) = tokenizer.bos_token_id {
        metadata.push((
            "tokenizer.ggml.bos_token_id".to_string(),
            GgufValue::Uint32(bos),
        ));
    }
    if let Some(eos) = tokenizer.eos_token_id {
        metadata.push((
            "tokenizer.ggml.eos_token_id".to_string(),
            GgufValue::Uint32(eos),
        ));
    }
    if !tokenizer.vocabulary.is_empty() {
        // GH-279: Dedup token table for llama.cpp compatibility.
        // HuggingFace tokenizers (Qwen3, etc.) may have multiple reserved tokens
        // mapped to "<unk>" — llama.cpp requires unique token strings.
        // Fix: append "_N" suffix to duplicates (same approach as convert.py).
        let mut seen = std::collections::HashMap::with_capacity(tokenizer.vocabulary.len());
        let deduped: Vec<String> = tokenizer
            .vocabulary
            .iter()
            .enumerate()
            .map(|(idx, tok)| {
                let count = seen.entry(tok.clone()).or_insert(0u32);
                *count += 1;
                if *count > 1 {
                    eprintln!(
                        "[GH-279] Dedup token id={idx}: {tok:?} → {tok}_{c}",
                        c = *count - 1
                    );
                    format!("{tok}_{}", *count - 1)
                } else {
                    tok.clone()
                }
            })
            .collect();
        metadata.push((
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::ArrayString(deduped),
        ));
        eprintln!(
            "[BUG-EXPORT-004] Added tokenizer metadata: model={}, vocab_size={}, bos={:?}, eos={:?}",
            model_type, tokenizer.vocabulary.len(), tokenizer.bos_token_id, tokenizer.eos_token_id
        );
    }
    if !tokenizer.merges.is_empty() {
        metadata.push((
            "tokenizer.ggml.merges".to_string(),
            GgufValue::ArrayString(tokenizer.merges.clone()),
        ));
    }
    metadata
}

fn export_to_gguf(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    output: &Path,
    input: &Path,
    quantize: Option<&QuantizationType>,
) -> Result<()> {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor};
    use crate::format::v2::AprV2Reader;
    use std::fs::File;
    use std::io::BufWriter;

    let tokenizer = super::import::load_tokenizer_from_json(input);

    let apr_metadata = if input.extension().and_then(|e| e.to_str()) == Some("apr") {
        fs::read(input)
            .ok()
            .and_then(|d| AprV2Reader::from_bytes(&d).ok())
            .map(|r| r.metadata().clone())
    } else {
        None
    };
    let inferred = super::import::infer_model_config_from_tensors(tensors);
    let cfg = resolve_gguf_config(apr_metadata.as_ref(), inferred.as_ref());

    let mut metadata = build_gguf_config_metadata(&cfg);
    append_tokenizer_to_metadata(
        &mut metadata,
        tokenizer.as_ref(),
        apr_metadata.as_ref(),
        &cfg.arch,
        &cfg.model_name,
        input,
    );

    eprintln!(
        "[GGUF-EXPORT-001] Writing {} metadata keys (arch={}, layers={}, heads={}/{}kv, hidden={})",
        metadata.len(),
        cfg.arch,
        cfg.num_layers,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.hidden_size
    );

    // GGUF-EXPORT-001: Map tensor names from HF convention to GGUF convention
    // PMAT-222 FIX: Reverse 2D shapes from standard [rows, cols] to GGML [ne0, ne1]
    // GGML convention: ne[0] is the contiguous dimension (cols), ne[1] is rows.
    // This is the inverse of write.rs:520 which reverses GGML→standard on import.
    //
    // BUG-1 FIX: Support Q4_K quantization for GGUF inference compatibility.
    // F32 GGUF files don't work with realizar's fused matmul kernels which
    // only support Q4_0/Q8_0/Q4_K/Q5_K/Q6_K types.
    let use_q4k = matches!(
        quantize,
        Some(QuantizationType::Q4K | QuantizationType::Int4)
    );

    // GH-202 FIX: Build GGUF tensors WITHOUT data transpose.
    //
    // CRITICAL INSIGHT: GGML data layout data[i0 + i1*ne0] is IDENTICAL to
    // C row-major data[row*cols + col] when shape is reversed. The data does
    // NOT need transposing for GGUF export — only shape needs reversal from
    // standard [rows, cols] back to GGML [ne0=cols, ne1=rows].
    let gguf_tensors: Vec<GgufTensor> = tensors
        .iter()
        .map(|(name, (data, shape))| {
            let gguf_name = hf_to_gguf_name(name);

            // lm_head and embeddings skip quantization — keep F32 to preserve full precision
            let is_embedding = gguf_name == "token_embd.weight" || name.contains("embed_tokens");
            let is_lm_head = gguf_name == "output.weight" || name.contains("lm_head");

            // Reverse shape for GGUF: [rows, cols] → [ne0=cols, ne1=rows]
            let gguf_shape = if shape.len() == 2 {
                vec![shape[1] as u64, shape[0] as u64]
            } else {
                shape.iter().map(|&d| d as u64).collect()
            };

            // GH-202 FIX: No data transpose needed. Data is row-major in APR,
            // and GGML's layout with reversed shape is identical.
            let (dtype, bytes) =
                if use_q4k && shape.len() == 2 && data.len() >= 256 && !is_embedding && !is_lm_head
                {
                    // Quantize row-major F32 to Q4K using GGUF shape [ne0, ne1]
                    // quantize_q4_k_matrix processes per-row with ne0 elements per row
                    let gguf_shape_usize = vec![shape[1], shape[0]]; // [ne0=cols, ne1=rows]
                    let q4k_bytes = super::quantize_q4_k_matrix(data, &gguf_shape_usize);
                    (GgmlType::Q4K, q4k_bytes)
                } else {
                    // F32 (weights, embeddings, 1D) - just convert to bytes, no transpose
                    let f32_bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
                    (GgmlType::F32, f32_bytes)
                };

            GgufTensor {
                name: gguf_name,
                shape: gguf_shape,
                dtype,
                data: bytes,
            }
        })
        .collect();

    // BUG-4 FIX: For tied embedding models, create Q4K output.weight from embedding
    let has_lm_head = gguf_tensors.iter().any(|t| t.name == "output.weight");
    let mut gguf_tensors = gguf_tensors;

    if use_q4k && !has_lm_head {
        if let Some(tied) = build_tied_output_weight(tensors) {
            gguf_tensors.push(tied);
        }
    }

    // Write to file
    let file = File::create(output).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to create output file: {e}"),
    })?;
    let mut writer = BufWriter::new(file);

    export_tensors_to_gguf(&mut writer, &gguf_tensors, &metadata)
}

/// Append tokenizer metadata to GGUF metadata, preferring tokenizer.json over APR fallback.
fn append_tokenizer_to_metadata(
    metadata: &mut Vec<(String, crate::format::gguf::GgufValue)>,
    tokenizer: Option<&crate::format::gguf::GgufTokenizer>,
    apr_metadata: Option<&crate::format::v2::AprV2Metadata>,
    arch: &str,
    model_name: &str,
    input: &Path,
) {
    if let Some(tok) = tokenizer {
        metadata.extend(build_tokenizer_gguf_metadata(tok, arch, model_name));
        return;
    }

    eprintln!(
        "[BUG-EXPORT-004] Warning: No tokenizer.json found near {}, GGUF may lack tokenizer metadata",
        input.display()
    );

    // GH-211: Fallback — extract tokenizer from APR metadata when no tokenizer.json
    let Some(apr_meta) = apr_metadata else {
        return;
    };
    let apr_tok_entries = extract_apr_tokenizer_for_gguf(apr_meta);
    if !apr_tok_entries.is_empty() {
        eprintln!(
            "[GH-211] Extracted {} tokenizer entries from APR metadata",
            apr_tok_entries.len()
        );
        metadata.extend(apr_tok_entries);
    }
}

/// Build a Q4K output.weight tensor from embedding data for tied-embedding models (BUG-4).
fn build_tied_output_weight(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
) -> Option<crate::format::gguf::GgufTensor> {
    use crate::format::gguf::{GgmlType, GgufTensor};

    let (_, (data, shape)) = tensors
        .iter()
        .find(|(name, _)| name.contains("embed_tokens") || name.contains("token_embedding"))?;

    if shape.len() != 2 || data.len() < 256 {
        return None;
    }

    eprintln!("[BUG-4-FIX] Creating Q4K output.weight from embedding for tied embeddings");

    let gguf_shape_usize = vec![shape[1], shape[0]]; // [ne0=cols, ne1=rows]
    let q4k_bytes = super::quantize_q4_k_matrix(data, &gguf_shape_usize);
    let gguf_shape = vec![shape[1] as u64, shape[0] as u64];

    Some(GgufTensor {
        name: "output.weight".to_string(),
        shape: gguf_shape,
        dtype: GgmlType::Q4K,
        data: q4k_bytes,
    })
}
