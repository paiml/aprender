
/// Architecture-specific defaults for rope_type, rope_theta, max_position_embeddings.
fn arch_specific_defaults(arch: Option<&str>) -> (Option<u32>, Option<f32>, Option<usize>) {
    let rope_type = match arch {
        Some("qwen2" | "qwen2.5" | "qwen" | "qwen3") => Some(2),
        Some("phi" | "phi3" | "phi4") => Some(2),
        _ => Some(0),
    };
    let rope_theta = match arch {
        Some("qwen2" | "qwen2.5" | "qwen" | "qwen3") => Some(1_000_000.0f32),
        Some("llama") => Some(500_000.0),
        Some("phi" | "phi3" | "phi4") => Some(10_000.0),
        _ => Some(10_000.0),
    };
    let max_pos = match arch {
        Some("qwen2" | "qwen2.5" | "qwen" | "qwen3") => Some(32768),
        Some("llama") => Some(8192),
        Some("phi" | "phi3" | "phi4") => Some(4096),
        _ => Some(4096),
    };
    (rope_type, rope_theta, max_pos)
}

/// Infer model config from tensor shapes (for SafeTensors which has no metadata)
///
/// GH-165 FIX: Now handles both HuggingFace and GGUF tensor naming conventions:
/// - HuggingFace: model.layers.N.self_attn.q_proj.weight, embed_tokens.weight
/// - GGUF: blk.N.attn_q.weight, token_embd.weight
pub(crate) fn infer_model_config_from_tensors(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
) -> Option<GgufModelConfig> {
    let (vocab_size, hidden_size) = infer_embedding_dims(tensors)?;
    let num_layers = count_transformer_layers(tensors);

    let kv_dim = find_projection_dim(tensors, &["k_proj.weight", "key.weight", "attn_k.weight"]);
    let q_dim = find_projection_dim(tensors, &["q_proj.weight", "query.weight", "attn_q.weight"]);
    let (num_heads, inferred_num_kv_heads) = infer_head_counts(q_dim, kv_dim, hidden_size);

    let intermediate_size = infer_intermediate_size_from_tensors(tensors);
    let architecture = infer_architecture_from_names(tensors);

    let num_kv_heads = inferred_num_kv_heads.or(num_heads);
    let (rope_type, rope_theta, max_position_embeddings) =
        arch_specific_defaults(architecture.as_deref());

    Some(GgufModelConfig {
        architecture,
        hidden_size: Some(hidden_size),
        num_layers: Some(num_layers),
        num_heads,
        num_kv_heads,
        vocab_size: Some(vocab_size),
        intermediate_size,
        max_position_embeddings,
        rope_theta,
        rms_norm_eps: Some(1e-6),
        rope_type,
    })
}

/// Validate and handle missing config.json for SafeTensors imports.
/// Returns `Ok(())` if config was found or `allow_no_config` is set,
/// otherwise returns an error.
fn validate_config_json_presence(
    config_json_found: bool,
    config_path: &Path,
    allow_no_config: bool,
) -> Result<()> {
    if config_json_found {
        return Ok(());
    }
    if allow_no_config {
        eprintln!(
            "[WARNING] config.json not found at {}",
            config_path.display()
        );
        eprintln!(
            "[WARNING] Model config inferred from tensor shapes. \
             rope_theta and other params may be wrong."
        );
        eprintln!(
            "[WARNING] Proceeding anyway (--allow-no-config). \
             For best results, download config.json alongside your model file."
        );
        Ok(())
    } else {
        Err(AprenderError::FormatError {
            message: format!(
                "config.json not found at {}. This file is required for correct \
                 model hyperparameters (rope_theta, max_position_embeddings, etc.). \
                 Download config.json alongside your model file, or pass \
                 --allow-no-config to proceed with inferred values (may produce \
                 garbage output).",
                config_path.display()
            ),
        })
    }
}

/// Load tensors from a single SafeTensors file with config.json validation.
fn load_single_safetensors(
    path: &Path,
    options: &ImportOptions,
) -> Result<SourceLoadResult> {
    let st_result = load_safetensors_with_f16_passthrough(path)?;
    let config_from_json = load_model_config_from_json(path);
    let config_json_found = config_from_json.is_some();
    let model_config =
        config_from_json.or_else(|| infer_model_config_from_tensors(&st_result.tensors));

    validate_config_json_presence(
        config_json_found,
        &path.with_file_name("config.json"),
        options.allow_no_config,
    )?;

    let tokenizer = load_tokenizer_from_json(path);
    Ok(SourceLoadResult {
        tensors: st_result.tensors,
        f16_raw_tensors: st_result.f16_raw_tensors,
        tokenizer,
        model_config,
        user_metadata: st_result.user_metadata,
    })
}

/// Load tensors from source file (`SafeTensors` format)
pub(crate) fn load_source_tensors(
    path: &Path,
    options: &ImportOptions,
) -> Result<SourceLoadResult> {
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match extension {
        // GH-218: Sharded SafeTensors via index.json
        "json"
            if path
                .file_name()
                .is_some_and(|n| n.to_string_lossy().ends_with(".index.json")) =>
        {
            load_sharded_safetensors(path, options)
        }
        "safetensors" => load_single_safetensors(path, options),
        "apr" => {
            // Already APR format - extract tensors
            Err(AprenderError::FormatError {
                message: "Cannot import from APR format - use direct loading instead".to_string(),
            })
        }
        "gguf" => {
            // Load GGUF with tokenizer AND model config (CRITICAL for inference)
            let result = load_gguf_with_tokenizer(path)?;
            Ok(SourceLoadResult {
                tensors: result.tensors,
                f16_raw_tensors: BTreeMap::new(), // GGUF uses different quant formats
                tokenizer: Some(result.tokenizer),
                model_config: Some(result.model_config),
                user_metadata: UserMetadata::new(),
            })
        }
        // GH-238: ONNX format import
        "onnx" => {
            let onnx_reader =
                crate::format::onnx::OnnxReader::from_file(path).map_err(|e| {
                    AprenderError::FormatError {
                        message: format!("Failed to parse ONNX file: {e}"),
                    }
                })?;

            let tensors = onnx_reader.to_f32_tensors();
            if tensors.is_empty() {
                return Err(AprenderError::FormatError {
                    message: "ONNX file contains no tensor initializers (weights)".to_string(),
                });
            }

            eprintln!(
                "[GH-238] Loaded {} tensors from ONNX (ir_version={}, producer={})",
                tensors.len(),
                onnx_reader.metadata().ir_version,
                onnx_reader.metadata().producer_name
            );

            let model_config = infer_model_config_from_tensors(&tensors);
            let tokenizer = load_tokenizer_from_json(path);

            Ok(SourceLoadResult {
                tensors,
                f16_raw_tensors: BTreeMap::new(),
                tokenizer,
                model_config,
                user_metadata: UserMetadata::new(),
            })
        }
        // GH-238: NeMo format (tar.gz archive)
        "nemo" => Err(AprenderError::FormatError {
            message: "NeMo format (.nemo) import is not yet implemented. \
                      Convert to SafeTensors first using: \
                      python -c \"from nemo.collections.asr.models import ... ; model.export('model.onnx')\""
                .to_string(),
        }),
        "bin" | "pt" | "pth" => Err(AprenderError::FormatError {
            message: format!(
                "PyTorch format ({extension}) not supported. Convert to SafeTensors first."
            ),
        }),
        other => Err(AprenderError::FormatError {
            message: format!(
                "Unknown file format: .{other}. Supported: .safetensors, .gguf, .onnx"
            ),
        }),
    }
}

/// GH-218: Load tensors from sharded SafeTensors model (via index.json).
///
/// Iterates shard files, calling `load_safetensors_with_f16_passthrough()` per shard,
/// and merges results into a single `SourceLoadResult`.
pub(crate) fn load_sharded_safetensors(
    index_path: &Path,
    options: &ImportOptions,
) -> Result<SourceLoadResult> {
    let content = fs::read_to_string(index_path).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to read shard index {}: {e}", index_path.display()),
    })?;
    let index = ShardIndex::from_json(&content)?;

    if index.shard_count() == 0 {
        return Err(AprenderError::FormatError {
            message: format!(
                "Shard index {} contains no shard files",
                index_path.display()
            ),
        });
    }

    let base_dir = index_path
        .parent()
        .ok_or_else(|| AprenderError::FormatError {
            message: format!(
                "Cannot determine parent directory of {}",
                index_path.display()
            ),
        })?;

    eprintln!(
        "[GH-218] Loading sharded SafeTensors: {} shards, {} tensors",
        index.shard_count(),
        index.tensor_count(),
    );

    let mut merged_tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    let mut merged_f16: BTreeMap<String, (Vec<u8>, Vec<usize>)> = BTreeMap::new();
    let mut merged_metadata = UserMetadata::new();

    for shard_file in index.shard_files() {
        let shard_path = base_dir.join(shard_file);
        if !shard_path.exists() {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Shard file {} referenced in index but not found at {}",
                    shard_file,
                    shard_path.display()
                ),
            });
        }

        eprintln!("[GH-218] Loading shard: {shard_file}");
        let st_result = load_safetensors_with_f16_passthrough(&shard_path)?;

        merged_tensors.extend(st_result.tensors);
        merged_f16.extend(st_result.f16_raw_tensors);
        // First shard wins for metadata conflicts
        for (k, v) in st_result.user_metadata {
            merged_metadata.entry(k).or_insert(v);
        }
    }

    eprintln!(
        "[GH-218] Merged {} tensors ({} F16 passthrough) from {} shards",
        merged_tensors.len(),
        merged_f16.len(),
        index.shard_count(),
    );

    // Load config.json and tokenizer.json from the same directory as index
    // Use a dummy file path in the base directory so sibling lookup works
    let sibling_path = base_dir.join("model.safetensors.index.json");
    let config_from_json = load_model_config_from_json(&sibling_path);
    let config_json_found = config_from_json.is_some();
    let model_config =
        config_from_json.or_else(|| infer_model_config_from_tensors(&merged_tensors));

    // GH-223: Error when config.json is missing for sharded models too.
    validate_config_json_presence(
        config_json_found,
        &base_dir.join("config.json"),
        options.allow_no_config,
    )?;

    let tokenizer = load_tokenizer_from_json(&sibling_path);

    Ok(SourceLoadResult {
        tensors: merged_tensors,
        f16_raw_tensors: merged_f16,
        tokenizer,
        model_config,
        user_metadata: merged_metadata,
    })
}

/// Load tensors from `SafeTensors` file using memory-mapped I/O for efficiency
///
/// PMAT-187: Validates all tensors after loading to catch corruption early.
pub(crate) fn load_safetensors_tensors(
    path: &Path,
) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    // Use MappedSafeTensors for zero-copy mmap access (much faster for large models)
    let mapped = MappedSafeTensors::open(path).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to mmap SafeTensors: {e}"),
    })?;

    let mut tensors = BTreeMap::new();
    let names: Vec<String> = mapped
        .tensor_names()
        .iter()
        .map(|&s| (*s).to_string())
        .collect();

    for name in &names {
        // Skip __metadata__ key if present
        if name.starts_with("__") {
            continue;
        }

        let meta = mapped
            .get_metadata(name)
            .ok_or_else(|| AprenderError::FormatError {
                message: format!("Tensor metadata not found for '{name}'"),
            })?;

        let data = mapped
            .get_tensor(name)
            .map_err(|e| AprenderError::FormatError {
                message: format!("Failed to extract tensor '{name}': {e}"),
            })?;

        // PMAT-187: Validate tensor values after loading (Jidoka - stop the line)
        validate_tensor_values(name, &data)?;

        tensors.insert(name.clone(), (data, meta.shape.clone()));
    }

    Ok(tensors)
}

/// GH-205: Result of loading SafeTensors with F16 passthrough support.
pub(crate) struct SafeTensorsLoadResult {
    /// F32 tensors (native F32 or converted from other dtypes for non-passthrough)
    pub tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    /// Raw F16 tensor bytes for passthrough (avoids F16→F32→F16 precision loss)
    pub f16_raw_tensors: BTreeMap<String, (Vec<u8>, Vec<usize>)>,
    /// User metadata from `__metadata__` section
    pub user_metadata: UserMetadata,
}

/// GH-205: Load SafeTensors with F16 passthrough support.
///
/// This function preserves raw F16 bytes for direct passthrough to APR format,
/// avoiding the precision loss from F16→F32→F16 round-trip conversion.
///
/// Returns:
/// - `tensors`: All tensors as F32 (for backward compatibility and validation)
/// - `f16_raw_tensors`: Raw F16 bytes for passthrough (only F16 tensors)
/// - `user_metadata`: User metadata from SafeTensors header
pub(crate) fn load_safetensors_with_f16_passthrough(path: &Path) -> Result<SafeTensorsLoadResult> {
    let mapped = MappedSafeTensors::open(path).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to mmap SafeTensors: {e}"),
    })?;

    let user_metadata = mapped.user_metadata().clone();
    if !user_metadata.is_empty() {
        eprintln!(
            "[PMAT-223] Extracted {} user metadata key(s) from SafeTensors __metadata__",
            user_metadata.len()
        );
    }

    let mut tensors = BTreeMap::new();
    let mut f16_raw_tensors = BTreeMap::new();
    let mut f16_count = 0usize;

    let names: Vec<String> = mapped
        .tensor_names()
        .iter()
        .map(|&s| (*s).to_string())
        .collect();

    for name in &names {
        if name.starts_with("__") {
            continue;
        }

        let meta = mapped
            .get_metadata(name)
            .ok_or_else(|| AprenderError::FormatError {
                message: format!("Tensor metadata not found for '{name}'"),
            })?;

        // GH-205: Check if this is an F16 tensor for passthrough
        if meta.dtype == "F16" {
            // Get raw bytes for passthrough (no conversion)
            if let Some(raw_bytes) = mapped.get_tensor_bytes(name) {
                f16_raw_tensors.insert(name.clone(), (raw_bytes.to_vec(), meta.shape.clone()));
                f16_count += 1;
            }
        }

        // Always also get F32 representation (for validation and backward compat)
        let data = mapped
            .get_tensor(name)
            .map_err(|e| AprenderError::FormatError {
                message: format!("Failed to extract tensor '{name}': {e}"),
            })?;

        validate_tensor_values(name, &data)?;
        tensors.insert(name.clone(), (data, meta.shape.clone()));
    }

    if f16_count > 0 {
        eprintln!(
            "[GH-205] F16 passthrough: {} of {} tensors will be written as raw F16",
            f16_count,
            tensors.len()
        );
    }

    Ok(SafeTensorsLoadResult {
        tensors,
        f16_raw_tensors,
        user_metadata,
    })
}
