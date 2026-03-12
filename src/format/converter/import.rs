//! APR Import Pipeline
//! PMAT-197: Extracted from mod.rs

use crate::error::{AprenderError, Result};
use crate::format::converter_types::{
    Architecture, ImportError, ImportOptions, QuantizationType, Source, TensorExpectation,
    ValidationConfig,
};
use crate::format::gguf::{
    load_gguf_raw, load_gguf_with_tokenizer, GgufModelConfig, GgufRawTensor, GgufTokenizer,
};
use crate::format::layout_contract::contract;
use crate::format::sharded::ShardIndex;
use crate::format::validation::{AprValidator, TensorStats, ValidationReport};
use crate::serialization::safetensors::{MappedSafeTensors, UserMetadata};
use std::collections::BTreeMap;

// Import write functions and helpers from parent module
use super::{validate_tensor_values, write_apr_file, write_apr_file_raw};
use std::fs;
use std::path::{Path, PathBuf};

#[cfg(feature = "hf-hub-integration")]
use crate::format::converter_types::parse_import_error;

pub fn apr_import<P: AsRef<Path>>(
    source: &str,
    output: P,
    options: ImportOptions,
) -> Result<ValidationReport> {
    let parsed_source = Source::parse(source)?;
    let output_path = output.as_ref();

    // Step 1: Resolve source to local path
    let local_path = resolve_source(&parsed_source, options.cache)?;

    // Step 2: Check if GGUF - use raw import path to preserve quantization
    // PMAT-271: Use magic bytes first, extension fallback for extensionless HF cache blobs
    let is_gguf = crate::format::rosetta::FormatType::from_magic(&local_path)
        .map(|f| matches!(f, crate::format::rosetta::FormatType::Gguf))
        .unwrap_or_else(|_| {
            local_path.extension().and_then(|e| e.to_str()) == Some("gguf")
        });
    if is_gguf {
        // PMAT-103: Use raw GGUF loading to preserve Q4_K/Q6_K quantization
        // GH-375: Falls back to dequant→requant for unsupported dtypes (Q4_0, Q5_0, Q8_0)
        if let Some(report) = try_gguf_raw_import(&local_path, output_path, &options)? {
            return Ok(report);
        }
    }

    // realizar#136: Streaming import for sharded SafeTensors (>10B params).
    // Writes tensors to disk incrementally — peak RAM = 1 shard (~5 GB) instead of full model.
    let is_sharded_safetensors = local_path
        .file_name()
        .is_some_and(|n| n.to_string_lossy().ends_with(".index.json"));
    if is_sharded_safetensors {
        return streaming_sharded_import(&local_path, output_path, &options);
    }

    // Non-GGUF path: Load tensors as f32, apply quantization during write
    let mut load_result = load_source_tensors(&local_path, &options)?;

    // PMAT-SAFETENSORS-TOK-001: For HuggingFace SafeTensors imports, try to find
    // tokenizer.json from the same repo if not found as sibling file
    resolve_hf_tokenizer_fallback(&mut load_result, &parsed_source);

    // model-metadata-bounds-v1.yaml: warn on out-of-bounds config values at import time
    if let Some(config) = load_result.model_config.as_ref() {
        config.warn_out_of_bounds();
    }

    // PMAT-224: Warn about unverified architectures before proceeding
    let metadata_arch = infer_architecture(
        &options.architecture,
        load_result
            .model_config
            .as_ref()
            .and_then(|c| c.architecture.as_deref()),
    );
    // CONTRACT: Tensor evidence overrides metadata claims
    let effective_arch = verify_architecture_from_tensor_evidence(
        metadata_arch,
        load_result.tensors.keys().map(String::as_str),
    );
    warn_unverified_architecture(&effective_arch, options.strict)?;

    // Step 3: Map tensor names to canonical APR names
    let mut mapped_tensors = map_tensor_names(&load_result.tensors, effective_arch);

    // GH-233: Split fused QKV tensors for GPT-2 after name mapping
    if effective_arch == Architecture::Gpt2 {
        Architecture::split_gpt2_fused_qkv(&mut mapped_tensors);
    }
    // GH-311: Split fused QKV tensors for GPT-NeoX after name mapping
    if effective_arch == Architecture::GptNeoX {
        Architecture::split_neox_fused_qkv(&mut mapped_tensors);
    }

    // GH-205 + GH-353: Also map F16/BF16 raw tensor names for passthrough
    let mapped_f16_raw: BTreeMap<String, (Vec<u8>, Vec<usize>, bool)> = load_result
        .f16_raw_tensors
        .iter()
        .map(|(name, (bytes, shape, is_bf16))| {
            let mapped_name = effective_arch.map_name(name);
            (mapped_name, (bytes.clone(), shape.clone(), *is_bf16))
        })
        .collect();

    // Step 4: ENFORCE CONTRACT (P0 - contracts/tensor-layout-v1.yaml)
    // The contract is the SOURCE OF TRUTH for tensor shapes.
    let layout_contract = contract();
    let vocab_size = load_result
        .model_config
        .as_ref()
        .and_then(|c| c.vocab_size)
        .unwrap_or(0);
    let hidden_dim = load_result
        .model_config
        .as_ref()
        .and_then(|c| c.hidden_size)
        .unwrap_or(0);

    validate_contract_f32(
        &layout_contract,
        &mapped_tensors,
        vocab_size,
        hidden_dim,
        options.strict,
    )?;

    // GH-279: Architecture completeness gate for SafeTensors path
    // PMAT-296: Gate MUST run even without model_config — infer num_layers from tensor names
    if let Some(config) = load_result.model_config.as_ref() {
        enforce_arch_completeness_gate_f32(&effective_arch, &mapped_tensors, config)?;
    } else {
        enforce_arch_completeness_gate_inferred(&effective_arch, &mapped_tensors)?;
    }

    // Step 5: Validate tensors (inline validation)
    let validation_result = validate_tensors(&mapped_tensors, &options)?;

    // Step 5: Write APR format (with tokenizer AND model config - CRITICAL for inference)
    // Note: Quantization (fp16/int8/int4) is applied during write for true packed storage
    // PMAT-223: Pass user metadata for preservation in APR custom field
    // GH-205: Pass F16 raw tensors for passthrough
    write_apr_file(
        &mapped_tensors,
        &mapped_f16_raw,
        output_path,
        &options,
        load_result.tokenizer.as_ref(),
        load_result.model_config.as_ref(),
        &load_result.user_metadata,
    )?;

    Ok(validation_result)
}

/// GH-375: Try raw GGUF import, falling back to dequant path for unsupported dtypes.
///
/// Returns `Ok(Some(report))` on raw success, `Ok(None)` to fall through to
/// the dequant→requant path, or `Err` for non-recoverable failures.
fn try_gguf_raw_import(
    path: &Path,
    output_path: &Path,
    options: &ImportOptions,
) -> Result<Option<ValidationReport>> {
    match apr_import_gguf_raw(path, output_path, options) {
        Ok(report) => Ok(Some(report)),
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("cannot represent exactly") || msg.contains("not yet supported") {
                eprintln!(
                    "[GH-375] Raw import failed ({}), falling back to dequant→requant path",
                    msg.lines().next().unwrap_or("unsupported dtype")
                );
                Ok(None)
            } else {
                Err(e)
            }
        }
    }
}

/// Import GGUF file preserving original quantization (Q4_K, Q6_K, etc.)
///
/// This is the preferred path for GGUF import as it preserves the exact
/// quantization from the source file, ensuring format parity with Ollama/llama.cpp.
pub(crate) fn apr_import_gguf_raw(
    gguf_path: &Path,
    output_path: &Path,
    options: &ImportOptions,
) -> Result<ValidationReport> {
    let raw_result = load_gguf_raw(gguf_path)?;

    // model-metadata-bounds-v1.yaml: warn on out-of-bounds config values at import time
    raw_result.model_config.warn_out_of_bounds();

    let effective_tokenizer = resolve_gguf_tokenizer(
        &raw_result.tokenizer,
        gguf_path,
        options.tokenizer_path.as_deref(),
    )?;

    let metadata_arch = resolve_and_log_architecture(
        &options.architecture,
        raw_result.model_config.architecture.as_deref(),
        options.strict,
    )?;
    // CONTRACT: Tensor evidence overrides metadata claims (e.g., bartowski Qwen3 GGUFs claim "qwen2")
    let effective_arch = verify_architecture_from_tensor_evidence(
        metadata_arch,
        raw_result.tensors.keys().map(String::as_str),
    );

    let mapped_tensors =
        map_and_enforce_raw_tensors(raw_result.tensors, &effective_arch, &raw_result.model_config)?;

    // GH-279: Architecture completeness gate — refuse to write incomplete models
    enforce_arch_completeness_gate(&effective_arch, &mapped_tensors, &raw_result.model_config)?;

    let mut validation_result = ValidationReport::new();
    validation_result.total_score = 85;

    write_apr_file_raw(
        &mapped_tensors,
        output_path,
        options,
        Some(&effective_tokenizer),
        Some(&raw_result.model_config),
    )?;

    Ok(validation_result)
}

/// Resolve architecture from options/GGUF config, log detection, and warn if unverified.
fn resolve_and_log_architecture(
    user_arch: &Architecture,
    gguf_arch: Option<&str>,
    strict: bool,
) -> Result<Architecture> {
    let effective_arch = infer_architecture(user_arch, gguf_arch);
    if effective_arch != Architecture::Auto {
        eprintln!(
            "[PMAT-222] Auto-detected architecture: {:?} (tensor names will be mapped)",
            effective_arch
        );
    }
    warn_unverified_architecture(&effective_arch, strict)?;
    Ok(effective_arch)
}

/// Map tensor names, split GPT-2 QKV if needed, and enforce layout contract.
fn map_and_enforce_raw_tensors(
    tensors: BTreeMap<String, GgufRawTensor>,
    effective_arch: &Architecture,
    model_config: &crate::format::gguf::GgufModelConfig,
) -> Result<BTreeMap<String, GgufRawTensor>> {
    use crate::format::layout_contract::enforce_import_contract;

    // Stage 1: Name mapping
    let mut mapped: BTreeMap<String, GgufRawTensor> = tensors
        .into_iter()
        .map(|(name, tensor)| (effective_arch.map_name(&name), tensor))
        .collect();

    // Stage 2: GPT-2 / GPT-NeoX QKV splitting
    if *effective_arch == Architecture::Gpt2 {
        Architecture::split_gpt2_fused_qkv_raw(&mut mapped);
    }
    if *effective_arch == Architecture::GptNeoX {
        Architecture::split_neox_fused_qkv_raw(&mut mapped);
    }

    // Stage 3: Contract enforcement (GH-208)
    let vocab_size = model_config.vocab_size.unwrap_or(0);
    let hidden_dim = model_config.hidden_size.unwrap_or(0);

    if vocab_size == 0 || hidden_dim == 0 {
        return Err(AprenderError::FormatError {
            message: format!(
                "CONTRACT ENFORCEMENT FAILED: Missing vocab_size ({}) or hidden_dim ({}). \
                 Cannot validate tensor layouts without model config. \
                 This GGUF file may be malformed.",
                vocab_size, hidden_dim
            ),
        });
    }

    let mapped: BTreeMap<String, GgufRawTensor> = mapped
        .into_iter()
        .map(|(name, mut tensor)| {
            let (apr_shape, needs_data_transpose) =
                enforce_import_contract(&name, &tensor.shape, vocab_size, hidden_dim);
            assert!(
                !needs_data_transpose,
                "CONTRACT BUG: enforce_import_contract returned needs_data_transpose=true for '{}'. \
                 GGUF→APR NEVER needs data transpose. See GH-208.",
                name
            );
            tensor.shape = apr_shape;
            (name, tensor)
        })
        .collect();

    eprintln!(
        "[CONTRACT-ENFORCED] {} tensors transformed via tensor-layout-v1.yaml (vocab={}, hidden={})",
        mapped.len(),
        vocab_size,
        hidden_dim
    );

    Ok(mapped)
}

/// GH-279: Architecture completeness gate for raw GGUF tensor import.
///
/// Verifies that ALL tensors required by the declared architecture are present
/// BEFORE writing the APR file. Missing tensor = hard error, not silent garbage later.
fn enforce_arch_completeness_gate(
    arch: &Architecture,
    tensors: &BTreeMap<String, GgufRawTensor>,
    config: &GgufModelConfig,
) -> Result<()> {
    let Some(arch_key) = arch.completeness_key() else {
        return Ok(()); // Non-transformer architectures skip this gate
    };
    let Some(num_layers) = config.num_layers else {
        return Ok(()); // Can't check without layer count
    };
    // Skip if model has no layer tensors — a config.json may describe layers
    // that aren't present in this particular file (e.g., standalone embeddings)
    let has_layers = tensors
        .keys()
        .any(|n| n.contains("model.layers.") || n.contains("blk."));
    if !has_layers {
        return Ok(());
    }
    let names: Vec<&str> = tensors.keys().map(String::as_str).collect();
    crate::format::layout_contract::enforce_architecture_completeness(&names, arch_key, num_layers)
        .map_err(|e| AprenderError::FormatError {
            message: format!("GH-279 architecture completeness gate: {e}"),
        })
}

/// GH-279: Architecture completeness gate for F32 SafeTensors import.
fn enforce_arch_completeness_gate_f32(
    arch: &Architecture,
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    config: &GgufModelConfig,
) -> Result<()> {
    let Some(arch_key) = arch.completeness_key() else {
        return Ok(());
    };
    let Some(num_layers) = config.num_layers else {
        return Ok(());
    };
    // Skip if model has no layer tensors — a config.json may describe layers
    // that aren't present in this particular file (e.g., standalone embeddings)
    let has_layers = tensors
        .keys()
        .any(|n| n.contains("model.layers.") || n.contains("blk."));
    if !has_layers {
        return Ok(());
    }
    let names: Vec<&str> = tensors.keys().map(String::as_str).collect();
    crate::format::layout_contract::enforce_architecture_completeness(&names, arch_key, num_layers)
        .map_err(|e| AprenderError::FormatError {
            message: format!("GH-279 architecture completeness gate: {e}"),
        })
}

/// PMAT-296: Architecture completeness gate when model_config is unavailable.
///
/// Infers `num_layers` from tensor names (counting unique layer indices).
/// This closes the GAP-1 bypass: SafeTensors without config.json still get checked.
fn enforce_arch_completeness_gate_inferred(
    arch: &Architecture,
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<()> {
    let Some(arch_key) = arch.completeness_key() else {
        return Ok(());
    };
    let num_layers = infer_num_layers_from_tensor_names(tensors.keys().map(String::as_str));
    if num_layers == 0 {
        return Ok(()); // No layer tensors → skip (e.g., standalone embeddings)
    }
    let names: Vec<&str> = tensors.keys().map(String::as_str).collect();
    crate::format::layout_contract::enforce_architecture_completeness(&names, arch_key, num_layers)
        .map_err(|e| AprenderError::FormatError {
            message: format!("GH-279 architecture completeness gate (inferred): {e}"),
        })
}

/// Infer the number of transformer layers from tensor name patterns.
///
/// Scans for `blk.{N}.` or `model.layers.{N}.` patterns and returns max(N) + 1.
fn infer_num_layers_from_tensor_names<'a>(names: impl Iterator<Item = &'a str>) -> usize {
    let mut max_layer: Option<usize> = None;
    for name in names {
        let idx = if let Some(rest) = name.strip_prefix("blk.") {
            rest.split('.').next().and_then(|s| s.parse::<usize>().ok())
        } else if let Some(rest) = name.strip_prefix("model.layers.") {
            rest.split('.').next().and_then(|s| s.parse::<usize>().ok())
        } else {
            None
        };
        if let Some(i) = idx {
            max_layer = Some(max_layer.map_or(i, |m: usize| m.max(i)));
        }
    }
    max_layer.map_or(0, |m| m + 1)
}

/// PMAT-SAFETENSORS-TOK-001: Try to find tokenizer.json from HF cache for this repo.
fn resolve_hf_tokenizer_fallback(load_result: &mut SourceLoadResult, source: &Source) {
    if load_result.tokenizer.is_some() {
        return;
    }
    if let Source::HuggingFace { org, repo, .. } = source {
        if let Some(tokenizer_path) = find_in_cache(org, repo, "tokenizer.json") {
            load_result.tokenizer = load_tokenizer_from_json(&tokenizer_path);
        }
    }
}

/// Resolve a source to a local file path
pub(crate) fn resolve_source(source: &Source, cache: bool) -> Result<PathBuf> {
    match source {
        Source::Local(path) => resolve_local_source(path),
        Source::HuggingFace { org, repo, file } => {
            resolve_hf_source(org, repo, file.as_ref(), cache)
        }
        Source::Url(url) => resolve_url_source(url),
    }
}

/// Resolve a local file or directory to a model path.
fn resolve_local_source(path: &Path) -> Result<PathBuf> {
    if !path.exists() {
        // GH-129: Use ImportError for actionable message
        let err = ImportError::NotFound {
            resource: path.display().to_string(),
            status: 0, // Local file, not HTTP
        };
        return Err(AprenderError::from(err));
    }
    // GH-218: Handle sharded SafeTensors directories
    if path.is_dir() {
        return resolve_local_directory(path);
    }
    Ok(path.to_path_buf())
}

/// Resolve a local directory to the best model file within it.
fn resolve_local_directory(path: &Path) -> Result<PathBuf> {
    let index = path.join("model.safetensors.index.json");
    if index.exists() {
        return Ok(index);
    }
    let single = path.join("model.safetensors");
    if single.exists() {
        return Ok(single);
    }
    Err(AprenderError::FormatError {
        message: format!(
            "Directory {} contains no model.safetensors.index.json or model.safetensors",
            path.display()
        ),
    })
}

/// Resolve a HuggingFace source by checking cache and optionally downloading.
fn resolve_hf_source(org: &str, repo: &str, file: Option<&String>, cache: bool) -> Result<PathBuf> {
    // PMAT-168: Smart default filename based on repo type
    let filename = file.map(String::as_str).unwrap_or_else(|| {
        // Detect GGUF repos by name convention
        if repo.to_lowercase().contains("gguf") {
            // Try common GGUF naming patterns
            // e.g., Qwen2.5-Coder-1.5B-Instruct-GGUF -> qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
            "model.gguf" // We'll try multiple patterns in find_in_cache
        } else {
            "model.safetensors"
        }
    });

    // Check standard cache locations first
    if cache {
        if let Some(path) = find_hf_in_cache(org, repo, file, filename) {
            return Ok(path);
        }
        // GH-279-2: For SafeTensors repos, also check for sharded index
        // Sharded models (e.g. Qwen3-8B) have model.safetensors.index.json
        // instead of a single model.safetensors file.
        if file.is_none() && filename == "model.safetensors" {
            if let Some(path) = find_in_cache(org, repo, "model.safetensors.index.json") {
                return Ok(path);
            }
        }
    }

    // Try to download using hf-hub if feature is enabled (GH-129: proper error handling)
    #[cfg(feature = "hf-hub-integration")]
    {
        let repo_id = format!("{org}/{repo}");
        // Return the result directly without explicit return statements
        download_from_hf(&repo_id, filename)
    }

    // Only reach here if hf-hub-integration feature is disabled
    #[cfg(not(feature = "hf-hub-integration"))]
    Err(AprenderError::FormatError {
        message: format!(
            "HuggingFace model not found in cache. Download manually:\n\
             huggingface-cli download {org}/{repo} {filename}\n\
             Or provide a local path to the SafeTensors/GGUF file.",
        ),
    })
}

/// Search HuggingFace cache for a model file, trying GGUF patterns if applicable.
fn find_hf_in_cache(
    org: &str,
    repo: &str,
    file: Option<&String>,
    filename: &str,
) -> Option<PathBuf> {
    // PMAT-168: Try multiple common filenames for GGUF repos
    if repo.to_lowercase().contains("gguf") && file.is_none() {
        let base_name = repo
            .to_lowercase()
            .replace("-gguf", "")
            .replace("_gguf", "");
        let gguf_patterns = [
            format!("{base_name}-q4_k_m.gguf"),
            format!("{base_name}-q4_k.gguf"),
            format!("{base_name}-q8_0.gguf"),
            "model.gguf".to_string(),
        ];
        for pattern in &gguf_patterns {
            if let Some(path) = find_in_cache(org, repo, pattern) {
                return Some(path);
            }
        }
    }
    find_in_cache(org, repo, filename)
}

/// Resolve a URL source (not yet implemented).
fn resolve_url_source(url: &str) -> Result<PathBuf> {
    Err(AprenderError::FormatError {
        message: format!("URL download not yet implemented: {url}"),
    })
}


/// GH-478: Quantization dispatch for streaming imports.
///
/// Mirrors `dispatch_quantize()` in write.rs but operates on `AprV2StreamingWriter`.
/// Respects `should_skip_quantization` for sensitive tensors (norms, biases, small tensors).
fn streaming_dispatch_quantize(
    writer: &mut crate::format::v2::AprV2StreamingWriter,
    name: &str,
    data: &[f32],
    shape: Vec<usize>,
    quantize: Option<QuantizationType>,
) -> std::result::Result<(), crate::format::v2::V2FormatError> {
    let should_skip = super::should_skip_quantization(name, data.len());
    match quantize {
        Some(QuantizationType::Fp16) => writer.add_f16_tensor(name, shape, data),
        Some(QuantizationType::Int8) if !should_skip => writer.add_q8_tensor(name, shape, data),
        Some(QuantizationType::Int4) if !should_skip => writer.add_q4_tensor(name, shape, data),
        Some(QuantizationType::Q4K) if !should_skip => {
            let q4k_bytes = super::quantize_q4_k_matrix(data, &shape);
            writer.add_q4k_raw_tensor(name, shape, &q4k_bytes)
        }
        _ => writer.add_f32_tensor(name, shape, data),
    }
}

/// realizar#136: Streaming import for sharded SafeTensors models.
///
/// Processes one shard at a time, writing tensors to disk immediately via
/// `AprV2StreamingWriter`. Peak RAM = 1 shard's mmap (~5 GB) + streaming
/// writer's index entries (~180 KB for 1811 tensors).
///
/// For Qwen3.5-35B-A3B (67 GB, 14 shards, 1811 tensors), this uses ~5 GB
/// peak RAM instead of ~134 GB with the non-streaming path.
fn streaming_sharded_import(
    index_path: &Path,
    output_path: &Path,
    options: &ImportOptions,
) -> Result<ValidationReport> {
    use crate::format::v2::{AprV2Metadata, AprV2StreamingWriter};

    let content = fs::read_to_string(index_path).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to read shard index {}: {e}", index_path.display()),
    })?;
    let index = ShardIndex::from_json(&content)?;

    if index.shard_count() == 0 {
        return Err(AprenderError::FormatError {
            message: "Shard index contains no shard files".to_string(),
        });
    }

    let canonical_index =
        std::fs::canonicalize(index_path).unwrap_or_else(|_| index_path.to_path_buf());
    let base_dir = canonical_index
        .parent()
        .ok_or_else(|| AprenderError::FormatError {
            message: format!(
                "Cannot determine parent directory of {}",
                index_path.display()
            ),
        })?;

    // Load config.json and tokenizer
    let sibling_path = base_dir.join("model.safetensors.index.json");
    let model_config = load_model_config_from_json(&sibling_path);
    let _tokenizer = load_tokenizer_from_json(&sibling_path);

    if model_config.is_none() && !options.allow_no_config {
        return Err(AprenderError::FormatError {
            message: format!(
                "config.json not found at {}. Use --allow-no-config to proceed without it.",
                base_dir.join("config.json").display()
            ),
        });
    }

    // Infer architecture
    let metadata_arch = infer_architecture(
        &options.architecture,
        model_config
            .as_ref()
            .and_then(|c| c.architecture.as_deref()),
    );

    // Build metadata for APR file
    let param_count = 0u64; // Computed during finalize from tensor shapes
    let metadata = AprV2Metadata {
        model_type: format!("{metadata_arch:?}"),
        name: Some(
            output_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model")
                .to_string(),
        ),
        param_count,
        architecture: model_config.as_ref().and_then(|c| c.architecture.clone()),
        hidden_size: model_config.as_ref().and_then(|c| c.hidden_size),
        num_layers: model_config.as_ref().and_then(|c| c.num_layers),
        num_heads: model_config.as_ref().and_then(|c| c.num_heads),
        num_kv_heads: model_config.as_ref().and_then(|c| c.num_kv_heads),
        vocab_size: model_config.as_ref().and_then(|c| c.vocab_size),
        intermediate_size: model_config.as_ref().and_then(|c| c.intermediate_size),
        max_position_embeddings: model_config
            .as_ref()
            .and_then(|c| c.max_position_embeddings),
        rope_theta: model_config.as_ref().and_then(|c| c.rope_theta),
        rope_type: model_config.as_ref().and_then(|c| c.rope_type),
        rms_norm_eps: model_config.as_ref().and_then(|c| c.rms_norm_eps),
        ..Default::default()
    };

    eprintln!(
        "[realizar#136] Streaming import: {} shards, {} tensors → {}",
        index.shard_count(),
        index.tensor_count(),
        output_path.display(),
    );

    let mut writer =
        AprV2StreamingWriter::new(metadata).map_err(|e| AprenderError::FormatError {
            message: format!("Failed to create streaming writer: {e}"),
        })?;

    // Process each shard: mmap → extract tensors → write to streaming writer → drop mmap
    let mut total_tensors = 0usize;
    let mut f16_passthrough = 0usize;

    for shard_file in index.shard_files() {
        let shard_path = base_dir.join(shard_file);
        if !shard_path.exists() {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Shard file {} not found at {}",
                    shard_file,
                    shard_path.display()
                ),
            });
        }

        let mapped =
            MappedSafeTensors::open(&shard_path).map_err(|e| AprenderError::FormatError {
                message: format!("Failed to mmap shard {shard_file}: {e}"),
            })?;

        let names: Vec<String> = mapped
            .tensor_names()
            .iter()
            .map(|&s| (*s).to_string())
            .collect();

        let mut shard_f16 = 0usize;

        for name in &names {
            if name.starts_with("__") {
                continue;
            }

            let meta = mapped
                .get_metadata(name)
                .ok_or_else(|| AprenderError::FormatError {
                    message: format!("Tensor metadata not found for '{name}'"),
                })?;

            // Map tensor name to canonical APR name
            let mapped_name = metadata_arch.map_name(name);

            let is_bf16 = meta.dtype == "BF16";
            let is_f16 = meta.dtype == "F16" || is_bf16;

            // GH-478: Only passthrough F16/BF16 when no quantization (or Fp16) is requested.
            // When user requests Int4/Int8/Q4K, dequantize to F32 and dispatch quantization.
            if is_f16 && matches!(options.quantize, None | Some(QuantizationType::Fp16)) {
                if let Some(raw_bytes) = mapped.get_tensor_bytes(name) {
                    writer
                        .add_raw_f16_tensor(&mapped_name, meta.shape.clone(), raw_bytes, is_bf16)
                        .map_err(|e| AprenderError::FormatError {
                            message: format!("Failed to write tensor '{mapped_name}': {e}"),
                        })?;
                    shard_f16 += 1;
                    total_tensors += 1;
                    continue;
                }
            }

            // Dequantize to F32 (handles F16/BF16→F32 and native F32)
            let data = mapped
                .get_tensor(name)
                .map_err(|e| AprenderError::FormatError {
                    message: format!("Failed to extract tensor '{name}': {e}"),
                })?;

            // GH-478: Dispatch quantization for streaming imports
            streaming_dispatch_quantize(
                &mut writer,
                &mapped_name,
                &data,
                meta.shape.clone(),
                options.quantize,
            )
            .map_err(|e| AprenderError::FormatError {
                message: format!("Failed to write tensor '{mapped_name}': {e}"),
            })?;
            total_tensors += 1;
        }

        let shard_quantized = names.len() - shard_f16;
        f16_passthrough += shard_f16;
        eprintln!(
            "[realizar#136] Shard {shard_file}: {} tensors ({shard_f16} F16 passthrough, {shard_quantized} quantized)",
            names.len(),
        );

        // mapped (mmap) dropped here — OS reclaims virtual address space
    }

    let total_quantized = total_tensors - f16_passthrough;
    eprintln!(
        "[realizar#136] Streaming write complete: {} tensors ({} F16 passthrough, {} quantized), {:.1} GB data",
        total_tensors,
        f16_passthrough,
        total_quantized,
        writer.data_bytes_written() as f64 / 1_073_741_824.0,
    );

    // Finalize: write header + metadata + index + data from temp file
    writer
        .finalize(output_path)
        .map_err(|e| AprenderError::FormatError {
            message: format!("Failed to finalize APR file: {e}"),
        })?;

    let file_size = fs::metadata(output_path)
        .map(|m| m.len())
        .unwrap_or(0);
    eprintln!(
        "[realizar#136] Written {} ({:.1} GB)",
        output_path.display(),
        file_size as f64 / 1_073_741_824.0,
    );

    // Return a basic validation report
    Ok(ValidationReport::new())
}

include!("import_include_01.rs");
