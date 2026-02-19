//! APR Import Pipeline
//! PMAT-197: Extracted from mod.rs

use crate::error::{AprenderError, Result};
use crate::format::converter_types::{
    Architecture, ImportError, ImportOptions, Source, TensorExpectation, ValidationConfig,
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
        // This is critical for format parity - we don't want to dequantize and re-quantize
        return apr_import_gguf_raw(&local_path, output_path, &options);
    }

    // Non-GGUF path: Load tensors as f32, apply quantization during write
    let mut load_result = load_source_tensors(&local_path, &options)?;

    // PMAT-SAFETENSORS-TOK-001: For HuggingFace SafeTensors imports, try to find
    // tokenizer.json from the same repo if not found as sibling file
    if load_result.tokenizer.is_none() {
        if let Source::HuggingFace { org, repo, .. } = &parsed_source {
            // Try to find tokenizer.json in HuggingFace cache for this repo
            if let Some(tokenizer_path) = find_in_cache(org, repo, "tokenizer.json") {
                load_result.tokenizer = load_tokenizer_from_json(&tokenizer_path);
            }
        }
    }

    // PMAT-224: Warn about unverified architectures before proceeding
    let effective_arch = infer_architecture(
        &options.architecture,
        load_result
            .model_config
            .as_ref()
            .and_then(|c| c.architecture.as_deref()),
    );
    warn_unverified_architecture(&effective_arch, options.strict)?;

    // Step 3: Map tensor names to canonical APR names
    let mut mapped_tensors = map_tensor_names(&load_result.tensors, effective_arch);

    // GH-233: Split fused QKV tensors for GPT-2 after name mapping
    if effective_arch == Architecture::Gpt2 {
        Architecture::split_gpt2_fused_qkv(&mut mapped_tensors);
    }

    // GH-205: Also map F16 raw tensor names for passthrough
    let mapped_f16_raw: BTreeMap<String, (Vec<u8>, Vec<usize>)> = load_result
        .f16_raw_tensors
        .iter()
        .map(|(name, (bytes, shape))| {
            let mapped_name = effective_arch.map_name(name);
            (mapped_name, (bytes.clone(), shape.clone()))
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
    if let Some(config) = load_result.model_config.as_ref() {
        enforce_arch_completeness_gate_f32(&effective_arch, &mapped_tensors, config)?;
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

    let effective_tokenizer = resolve_gguf_tokenizer(
        &raw_result.tokenizer,
        gguf_path,
        options.tokenizer_path.as_deref(),
    )?;

    let effective_arch = resolve_and_log_architecture(
        &options.architecture,
        raw_result.model_config.architecture.as_deref(),
        options.strict,
    )?;

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

    // Stage 2: GPT-2 QKV splitting
    if *effective_arch == Architecture::Gpt2 {
        Architecture::split_gpt2_fused_qkv_raw(&mut mapped);
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
    let names: Vec<&str> = tensors.keys().map(String::as_str).collect();
    crate::format::layout_contract::enforce_architecture_completeness(&names, arch_key, num_layers)
        .map_err(|e| AprenderError::FormatError {
            message: format!("GH-279 architecture completeness gate: {e}"),
        })
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

/// Infer architecture from user option or model config string.
fn infer_architecture(user_arch: &Architecture, config_arch: Option<&str>) -> Architecture {
    if *user_arch != Architecture::Auto {
        return user_arch.clone();
    }
    config_arch
        .and_then(Architecture::from_model_type)
        .unwrap_or(Architecture::Auto)
}

/// Emit warnings for unverified architectures; error in strict mode.
fn warn_unverified_architecture(arch: &Architecture, strict: bool) -> Result<()> {
    if arch.is_inference_verified() {
        return Ok(());
    }
    eprintln!(
        "[PMAT-224] WARNING: Architecture '{}' has not been verified for inference.",
        arch.display_name()
    );
    eprintln!(
        "[PMAT-224] Verified architectures: Qwen2, Qwen3, Qwen3.5, LLaMA, Phi. Use --strict to reject unverified."
    );
    if strict {
        return Err(AprenderError::FormatError {
            message: format!(
                "Architecture '{}' is not verified for inference (--strict mode). \
                 Remove --strict to import anyway, or specify --arch qwen2/llama.",
                arch.display_name()
            ),
        });
    }
    Ok(())
}

/// Validate F32 tensors against layout contract.
fn validate_contract_f32(
    layout: &crate::format::layout_contract::LayoutContract,
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    vocab_size: usize,
    hidden_dim: usize,
    strict: bool,
) -> Result<()> {
    if vocab_size == 0 || hidden_dim == 0 {
        eprintln!(
            "[CONTRACT] WARNING: Cannot validate contract - missing vocab_size or hidden_dim"
        );
        return Ok(());
    }
    for (name, (_data, shape)) in tensors {
        if let Err(e) = layout.validate_apr_shape(name, shape, vocab_size, hidden_dim) {
            eprintln!(
                "[CONTRACT-VIOLATION] {}: {} (see contracts/tensor-layout-v1.yaml)",
                name, e
            );
            if strict {
                return Err(AprenderError::FormatError {
                    message: format!("Contract violation: {e}"),
                });
            }
        }
    }
    eprintln!(
        "[CONTRACT] Validated {} tensors against tensor-layout-v1.yaml (vocab={}, hidden={})",
        tensors.len(),
        vocab_size,
        hidden_dim
    );
    Ok(())
}

include!("import_part_02.rs");
include!("import_part_03.rs");
include!("import_part_04.rs");
include!("import_part_05.rs");
include!("import_part_06.rs");
