//! APR Converter - Export to SafeTensors/GGUF (APR-SPEC §4.6)
//! PMAT-197: Extracted from mod.rs for file size reduction

use crate::error::{AprenderError, Result};
use crate::format::converter_types::{Architecture, QuantizationType};
use crate::format::gguf::GgufReader;
use crate::format::layout_contract::contract;
use crate::serialization::safetensors::{
    save_safetensors, save_safetensors_with_metadata, UserMetadata,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

// Import shared functions from parent module
use super::{
    calculate_tensor_size, load_model_tensors_provenance, map_tensor_names, quantize_tensors,
    NativeF32Tensors,
};
// NOTE: quantize_q4_k_matrix imported via super:: through mod.rs

// GH-202 FIX: Removed transpose_f32_rowmajor_to_colmajor.
// GGML data layout is C row-major with reversed shape [ne0, ne1].
// No data transposition is needed for GGUF export — only shape reversal.

/// Export format options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// `SafeTensors` format (.safetensors) - `HuggingFace` ecosystem
    SafeTensors,
    /// GGUF format (.gguf) - llama.cpp / local inference
    Gguf,
    /// ONNX format (.onnx) - Cross-framework inference (not yet implemented)
    Onnx,
    /// `TorchScript` format (.pt) - `PyTorch` deployment (not yet implemented)
    TorchScript,
}

impl std::str::FromStr for ExportFormat {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "safetensors" | "st" => Ok(Self::SafeTensors),
            "gguf" => Ok(Self::Gguf),
            "onnx" => Ok(Self::Onnx),
            "torchscript" | "pt" | "torch" => Ok(Self::TorchScript),
            _ => Err(format!("Unknown export format: {s}")),
        }
    }
}

impl ExportFormat {
    /// Get default file extension
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::SafeTensors => "safetensors",
            Self::Gguf => "gguf",
            Self::Onnx => "onnx",
            Self::TorchScript => "pt",
        }
    }

    /// Check if format is supported
    #[must_use]
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::SafeTensors | Self::Gguf)
    }
}

/// Options for model export
#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// Target format
    pub format: ExportFormat,
    /// Optional quantization
    pub quantize: Option<QuantizationType>,
    /// GH-182: Include tokenizer.json companion file (SafeTensors only)
    pub include_tokenizer: bool,
    /// GH-182: Include config.json companion file (SafeTensors only)
    pub include_config: bool,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::SafeTensors,
            quantize: None,
            include_tokenizer: true, // Default to true for HuggingFace compatibility
            include_config: true,
        }
    }
}

/// Report from export operation
#[derive(Debug, Clone)]
pub struct ExportReport {
    /// Original size in bytes
    pub original_size: usize,
    /// Exported size in bytes
    pub exported_size: usize,
    /// Number of tensors exported
    pub tensor_count: usize,
    /// Export format used
    pub format: ExportFormat,
    /// Quantization applied
    pub quantization: Option<QuantizationType>,
}

/// Export APR/SafeTensors model to another format
///
/// # Arguments
///
/// * `input` - Input model path (.apr or .safetensors)
/// * `output` - Output file path
/// * `options` - Export options
///
/// # Returns
///
/// Export report with size and format information
///
/// # Errors
///
/// Returns error if:
/// - Input file doesn't exist
/// - Format not supported
/// - Export fails
///
/// # Example
///
/// ```rust,ignore
/// use aprender::format::{apr_export, ExportOptions, ExportFormat};
///
/// let options = ExportOptions {
///     format: ExportFormat::Gguf,
///     quantize: None,
/// };
/// let report = apr_export("model.apr", "model.gguf", options)?;
/// ```
pub fn apr_export<P: AsRef<Path>>(
    input: P,
    output: P,
    options: ExportOptions,
) -> Result<ExportReport> {
    let input_path = input.as_ref();
    let output_path = output.as_ref();

    // Validate input exists
    if !input_path.exists() {
        return Err(AprenderError::FormatError {
            message: format!("Input file not found: {}", input_path.display()),
        });
    }

    // Check if format is supported
    if !options.format.is_supported() {
        return Err(AprenderError::FormatError {
            message: format!(
                "Export format {:?} is not yet supported. Use 'safetensors' or 'gguf'.",
                options.format
            ),
        });
    }

    // PMAT-252: For APR→GGUF with quantized source, use raw block passthrough
    // to avoid lossy double quantization (Q4K→F32→Q4K).
    if options.format == ExportFormat::Gguf
        && options.quantize.is_none()
        && input_path.extension().and_then(|e| e.to_str()) == Some("apr")
    {
        if let Some(detected) = detect_apr_quantization(input_path) {
            eprintln!(
                "[PMAT-252] Raw passthrough: detected {:?} in APR source. Copying blocks directly (zero loss).",
                detected
            );
            let report = export_apr_to_gguf_raw(input_path, output_path)?;
            return Ok(report);
        }
    }

    // DOUBLE-QUANT-001: Load with provenance tracking to prevent double quantization.
    let provenance = load_model_tensors_provenance(input_path)?;
    let original_size = calculate_tensor_size(provenance.as_map());

    // Consume provenance into raw map for downstream processing.
    let tensors = provenance.into_map();

    // GH-200: Map GGUF tensor names to HF canonical format before export.
    // GGUF uses names like "blk.0.attn_q.weight" but SafeTensors/HF expects
    // "model.layers.0.self_attn.q_proj.weight". Without this, exported
    // SafeTensors files have wrong names and fail inference.
    let tensors = if input_path.extension().and_then(|e| e.to_str()) == Some("gguf") {
        let arch = detect_gguf_architecture(input_path);
        map_tensor_names(&tensors, arch)
    } else {
        tensors
    };

    // ROSETTA-003: Unfuse legacy QKV tensors for lossless round-trip export.
    // Old APR files (pre-ROSETTA-003) fused Q/K/V into qkv_proj for realizar.
    // New APR files store separate Q/K/V. This handles both.
    let tensors = unfuse_qkv_tensors(tensors, input_path);

    // ROSETTA-003: Remove synthesized lm_head for SafeTensors export (tied embeddings).
    // HuggingFace convention omits lm_head.weight when tied to embed_tokens.
    let tensors = if options.format == ExportFormat::SafeTensors {
        remove_tied_lm_head(tensors, input_path)
    } else {
        tensors
    };

    // ENFORCE CONTRACT (P0 - contracts/tensor-layout-v1.yaml)
    // Validate tensors before export to catch any corrupted APR files.
    let layout_contract = contract();
    // Infer vocab_size and hidden_dim from tensors
    let (vocab_size, hidden_dim) = infer_vocab_hidden(&tensors);
    if vocab_size > 0 && hidden_dim > 0 {
        for (name, (_data, shape)) in &tensors {
            if let Err(e) = layout_contract.validate_apr_shape(name, shape, vocab_size, hidden_dim)
            {
                eprintln!(
                    "[CONTRACT-VIOLATION] Export validation failed for {}: {}",
                    name, e
                );
                // Don't hard fail on export - log the warning
            }
        }
    }

    // DOUBLE-QUANT-001: Apply quantization only to natively F32 tensors.
    // Attempting to quantize dequantized tensors is rejected at compile time
    // (quantize_tensors only accepts NativeF32Tensors), but we also need a
    // runtime check here since we've already destructured the provenance above.
    let tensors = if let Some(ref quant_type) = options.quantize {
        // Re-check provenance by re-loading (the provenance was consumed by into_map above).
        // This is cheap since we only read the header, not the tensor data.
        if let Some(detected) = detect_apr_quantization(input_path) {
            return Err(AprenderError::FormatError {
                message: format!(
                    "DOUBLE-QUANT-001: Cannot re-quantize to {quant_type:?}: source is already \
                     {detected:?}. Remove --quantize flag to use raw passthrough."
                ),
            });
        }
        let native = NativeF32Tensors::new(tensors);
        quantize_tensors(&native, quant_type)?.into_inner()
    } else {
        tensors
    };

    // Export to target format
    match options.format {
        ExportFormat::SafeTensors => {
            // PMAT-223: Extract user metadata from APR custom field for round-trip
            let user_metadata = extract_user_metadata(input_path);
            if user_metadata.is_empty() {
                save_safetensors(output_path, &tensors).map_err(|e| {
                    AprenderError::FormatError {
                        message: format!("Failed to export to SafeTensors: {e}"),
                    }
                })?;
            } else {
                eprintln!(
                    "[PMAT-223] Restoring {} user metadata key(s) to SafeTensors __metadata__",
                    user_metadata.len()
                );
                save_safetensors_with_metadata(output_path, &tensors, &user_metadata).map_err(
                    |e| AprenderError::FormatError {
                        message: format!("Failed to export to SafeTensors: {e}"),
                    },
                )?;
            }

            // GH-182: Write companion files alongside SafeTensors
            let output_dir = output_path.parent().unwrap_or(Path::new("."));

            if options.include_config {
                let config = infer_model_config(&tensors);
                let config_path = output_dir.join("config.json");
                if let Err(e) = fs::write(&config_path, config) {
                    eprintln!("[GH-182] Warning: Failed to write config.json: {e}");
                }
            }

            if options.include_tokenizer {
                // Try to extract tokenizer from APR input if available
                let tokenizer_json = infer_tokenizer_json(input_path);
                if !tokenizer_json.is_empty() {
                    let tokenizer_path = output_dir.join("tokenizer.json");
                    if let Err(e) = fs::write(&tokenizer_path, &tokenizer_json) {
                        eprintln!("[GH-182] Warning: Failed to write tokenizer.json: {e}");
                    }
                }
            }
        }
        ExportFormat::Gguf => {
            export_to_gguf(&tensors, output_path, input_path, options.quantize.as_ref())?;
        }
        ExportFormat::Onnx | ExportFormat::TorchScript => {
            return Err(AprenderError::FormatError {
                message: format!("Export format {:?} is not yet implemented", options.format),
            });
        }
    }

    // Get exported file size
    let exported_size = fs::metadata(output_path)
        .map(|m| m.len() as usize)
        .unwrap_or(0);

    // BUG-EXPORT-003 FIX: Report actual exported tensor count, not original.
    // After unfuse_qkv_tensors (increases count) and remove_tied_lm_head
    // (decreases count for SafeTensors), the count may differ from original.
    Ok(ExportReport {
        original_size,
        exported_size,
        tensor_count: tensors.len(),
        format: options.format,
        quantization: options.quantize,
    })
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
fn export_to_gguf(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    output: &Path,
    input: &Path,
    quantize: Option<&QuantizationType>,
) -> Result<()> {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use crate::format::v2::AprV2Reader;
    use std::fs::File;
    use std::io::BufWriter;

    // BUG-EXPORT-004: Load tokenizer from sibling tokenizer.json for GGUF metadata
    eprintln!(
        "[DEBUG-TOK] Looking for tokenizer near: {}",
        input.display()
    );
    let tokenizer = super::import::load_tokenizer_from_json(input);
    eprintln!("[DEBUG-TOK] Tokenizer loaded: {}", tokenizer.is_some());

    // GGUF-EXPORT-001: Read APR metadata for GGUF KV pairs
    let apr_metadata = if input.extension().and_then(|e| e.to_str()) == Some("apr") {
        let data = fs::read(input).ok();
        data.and_then(|d| AprV2Reader::from_bytes(&d).ok())
            .map(|r| r.metadata().clone())
    } else {
        None
    };

    // GGUF-EXPORT-001: Infer config from tensors as fallback
    let inferred = super::import::infer_model_config_from_tensors(tensors);

    let arch = apr_metadata
        .as_ref()
        .and_then(|m| m.architecture.as_deref())
        .or(inferred.as_ref().and_then(|c| c.architecture.as_deref()))
        .unwrap_or("qwen2");

    let hidden_size = apr_metadata
        .as_ref()
        .and_then(|m| m.hidden_size)
        .or(inferred.as_ref().and_then(|c| c.hidden_size))
        .unwrap_or(4096);

    let num_layers = apr_metadata
        .as_ref()
        .and_then(|m| m.num_layers)
        .or(inferred.as_ref().and_then(|c| c.num_layers))
        .unwrap_or(32);

    let num_heads = apr_metadata
        .as_ref()
        .and_then(|m| m.num_heads)
        .or(inferred.as_ref().and_then(|c| c.num_heads))
        .unwrap_or(32);

    let num_kv_heads = apr_metadata
        .as_ref()
        .and_then(|m| m.num_kv_heads)
        .or(inferred.as_ref().and_then(|c| c.num_kv_heads))
        .unwrap_or(num_heads);

    let vocab_size = apr_metadata
        .as_ref()
        .and_then(|m| m.vocab_size)
        .or(inferred.as_ref().and_then(|c| c.vocab_size))
        .unwrap_or(32000);

    let intermediate_size = apr_metadata
        .as_ref()
        .and_then(|m| m.intermediate_size)
        .or(inferred.as_ref().and_then(|c| c.intermediate_size))
        .unwrap_or(11008);

    let max_pos = apr_metadata
        .as_ref()
        .and_then(|m| m.max_position_embeddings)
        .unwrap_or(32768);

    let rope_theta = apr_metadata
        .as_ref()
        .and_then(|m| m.rope_theta)
        .unwrap_or(1_000_000.0);

    let rms_norm_eps = apr_metadata
        .as_ref()
        .and_then(|m| m.rms_norm_eps)
        .unwrap_or(1e-6);

    let head_dim = if num_heads > 0 {
        hidden_size / num_heads
    } else {
        128
    };

    let model_name = apr_metadata
        .as_ref()
        .and_then(|m| m.name.clone())
        .unwrap_or_else(|| "model".to_string());

    // Build GGUF metadata KV pairs
    let metadata = vec![
        (
            "general.architecture".to_string(),
            GgufValue::String(arch.to_string()),
        ),
        ("general.name".to_string(), GgufValue::String(model_name)),
        (
            "general.quantization_version".to_string(),
            GgufValue::Uint32(2),
        ),
        (
            "general.file_type".to_string(),
            GgufValue::Uint32(0), // F32
        ),
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
        (
            format!("{arch}.attention.layer_norm_rms_epsilon"),
            GgufValue::Float32(rms_norm_eps),
        ),
        (
            format!("{arch}.rope.dimension_count"),
            GgufValue::Uint32(head_dim as u32),
        ),
        (
            format!("{arch}.rope.freq_base"),
            GgufValue::Float32(rope_theta),
        ),
        (
            format!("{arch}.vocab_size"),
            GgufValue::Uint32(vocab_size as u32),
        ),
    ];

    // BUG-EXPORT-004: Add tokenizer metadata for realizar inference
    // Without BOS/EOS tokens, realizar can't properly tokenize/detokenize
    let mut metadata = metadata; // Make mutable for tokenizer additions
    if let Some(ref tok) = tokenizer {
        // Add tokenizer model type (gpt2 for BPE models like Qwen)
        let model_type = tok.model_type.as_deref().unwrap_or("gpt2");
        metadata.push((
            "tokenizer.ggml.model".to_string(),
            GgufValue::String(model_type.to_lowercase()),
        ));

        // Add pre-tokenizer type (qwen2 for Qwen models)
        metadata.push((
            "tokenizer.ggml.pre".to_string(),
            GgufValue::String(arch.to_string()),
        ));

        // Add BOS token ID (critical for inference)
        if let Some(bos) = tok.bos_token_id {
            metadata.push((
                "tokenizer.ggml.bos_token_id".to_string(),
                GgufValue::Uint32(bos),
            ));
        }

        // Add EOS token ID (critical for knowing when to stop)
        if let Some(eos) = tok.eos_token_id {
            metadata.push((
                "tokenizer.ggml.eos_token_id".to_string(),
                GgufValue::Uint32(eos),
            ));
        }

        // Add vocabulary (required for tokenization)
        if !tok.vocabulary.is_empty() {
            metadata.push((
                "tokenizer.ggml.tokens".to_string(),
                GgufValue::ArrayString(tok.vocabulary.clone()),
            ));
            eprintln!(
                "[BUG-EXPORT-004] Added tokenizer metadata: model={}, vocab_size={}, bos={:?}, eos={:?}",
                model_type,
                tok.vocabulary.len(),
                tok.bos_token_id,
                tok.eos_token_id
            );
        }

        // Add merges if available (for BPE tokenization)
        if !tok.merges.is_empty() {
            metadata.push((
                "tokenizer.ggml.merges".to_string(),
                GgufValue::ArrayString(tok.merges.clone()),
            ));
        }
    } else {
        eprintln!(
            "[BUG-EXPORT-004] Warning: No tokenizer.json found near {}, GGUF may lack tokenizer metadata",
            input.display()
        );
    }

    eprintln!(
        "[GGUF-EXPORT-001] Writing {} metadata keys (arch={}, layers={}, heads={}/{}kv, hidden={})",
        metadata.len(),
        arch,
        num_layers,
        num_heads,
        num_kv_heads,
        hidden_size
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

            let is_embedding = gguf_name == "token_embd.weight" || name.contains("embed_tokens");

            // Reverse shape for GGUF: [rows, cols] → [ne0=cols, ne1=rows]
            let gguf_shape = if shape.len() == 2 {
                vec![shape[1] as u64, shape[0] as u64]
            } else {
                shape.iter().map(|&d| d as u64).collect()
            };

            // GH-202 FIX: No data transpose needed. Data is row-major in APR,
            // and GGML's layout with reversed shape is identical.
            let (dtype, bytes) =
                if use_q4k && shape.len() == 2 && data.len() >= 256 && !is_embedding {
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
        if let Some(embed_data) = tensors
            .iter()
            .find(|(name, _)| name.contains("embed_tokens") || name.contains("token_embedding"))
        {
            let (_, (data, shape)) = embed_data;
            if shape.len() == 2 && data.len() >= 256 {
                eprintln!(
                    "[BUG-4-FIX] Creating Q4K output.weight from embedding for tied embeddings"
                );

                // GH-202 FIX: No transpose needed. Quantize with GGUF shape.
                let gguf_shape_usize = vec![shape[1], shape[0]]; // [ne0=cols, ne1=rows]
                let q4k_bytes = super::quantize_q4_k_matrix(data, &gguf_shape_usize);
                let gguf_shape = vec![shape[1] as u64, shape[0] as u64];

                gguf_tensors.push(GgufTensor {
                    name: "output.weight".to_string(),
                    shape: gguf_shape,
                    dtype: GgmlType::Q4K,
                    data: q4k_bytes,
                });
            }
        }
    }

    // Write to file
    let file = File::create(output).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to create output file: {e}"),
    })?;
    let mut writer = BufWriter::new(file);

    export_tensors_to_gguf(&mut writer, &gguf_tensors, &metadata)
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
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
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

    // Build GGUF metadata (same logic as export_to_gguf but from APR metadata directly)
    let arch = apr_metadata.architecture.as_deref().unwrap_or("qwen2");
    let hidden_size = apr_metadata.hidden_size.unwrap_or(4096);
    let num_layers = apr_metadata.num_layers.unwrap_or(32);
    let num_heads = apr_metadata.num_heads.unwrap_or(32);
    let num_kv_heads = apr_metadata.num_kv_heads.unwrap_or(num_heads);
    let vocab_size = apr_metadata.vocab_size.unwrap_or(32000);
    let intermediate_size = apr_metadata.intermediate_size.unwrap_or(11008);
    let max_pos = apr_metadata.max_position_embeddings.unwrap_or(32768);
    let rope_theta = apr_metadata.rope_theta.unwrap_or(1_000_000.0);
    let rms_norm_eps = apr_metadata.rms_norm_eps.unwrap_or(1e-6);
    let head_dim = if num_heads > 0 {
        hidden_size / num_heads
    } else {
        128
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
        (
            format!("{arch}.attention.layer_norm_rms_epsilon"),
            GgufValue::Float32(rms_norm_eps),
        ),
        (
            format!("{arch}.rope.dimension_count"),
            GgufValue::Uint32(head_dim as u32),
        ),
        (
            format!("{arch}.rope.freq_base"),
            GgufValue::Float32(rope_theta),
        ),
        (
            format!("{arch}.vocab_size"),
            GgufValue::Uint32(vocab_size as u32),
        ),
    ];

    // GH-253: Read tokenizer data from APR embedded custom fields (NOT sibling files).
    // GGUF→APR import stores vocab, merges, token_type, BOS/EOS/pad IDs, and chat_template
    // in APR custom fields. Export reads them back for lossless GGUF round-trip.
    let custom = &apr_metadata.custom;

    // Tokenizer model type: "gpt2" for byte-level BPE (Qwen, GPT-2), "llama" for SentencePiece
    // GH-253-3: APR stores raw model_type from GGUF which may be "bpe" — map to "gpt2"
    let raw_model_type = custom
        .get("tokenizer.model")
        .and_then(|v| v.as_str())
        .unwrap_or("gpt2");
    let model_type = match raw_model_type {
        "bpe" => "gpt2", // byte-level BPE uses "gpt2" in GGUF, not "bpe"
        other => other,
    };
    metadata.push((
        "tokenizer.ggml.model".to_string(),
        GgufValue::String(model_type.to_string()),
    ));
    metadata.push((
        "tokenizer.ggml.pre".to_string(),
        GgufValue::String(arch.to_string()),
    ));

    // Vocabulary: read from APR custom field "tokenizer.vocabulary"
    if let Some(vocab_val) = custom.get("tokenizer.vocabulary") {
        if let Some(vocab_arr) = vocab_val.as_array() {
            let tokens: Vec<String> = vocab_arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            if !tokens.is_empty() {
                metadata.push((
                    "tokenizer.ggml.tokens".to_string(),
                    GgufValue::ArrayString(tokens),
                ));
            }
        }
    }

    // BPE merges
    if let Some(merges_val) = custom.get("tokenizer.merges") {
        if let Some(merges_arr) = merges_val.as_array() {
            let merges: Vec<String> = merges_arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            if !merges.is_empty() {
                metadata.push((
                    "tokenizer.ggml.merges".to_string(),
                    GgufValue::ArrayString(merges),
                ));
            }
        }
    }

    // BOS token ID
    if let Some(bos_val) = custom.get("tokenizer.bos_token_id") {
        if let Some(bos) = bos_val.as_u64() {
            metadata.push((
                "tokenizer.ggml.bos_token_id".to_string(),
                GgufValue::Uint32(bos as u32),
            ));
        }
    }

    // EOS token ID
    if let Some(eos_val) = custom.get("tokenizer.eos_token_id") {
        if let Some(eos) = eos_val.as_u64() {
            metadata.push((
                "tokenizer.ggml.eos_token_id".to_string(),
                GgufValue::Uint32(eos as u32),
            ));
        }
    }

    // GH-253-1: Token type array (per-token: 1=normal, 3=special, etc.)
    if let Some(tt_val) = custom.get("tokenizer.token_type") {
        if let Some(tt_arr) = tt_val.as_array() {
            let types: Vec<i32> = tt_arr
                .iter()
                .filter_map(|v| v.as_i64().map(|n| n as i32))
                .collect();
            if !types.is_empty() {
                metadata.push((
                    "tokenizer.ggml.token_type".to_string(),
                    GgufValue::ArrayInt32(types),
                ));
            }
        }
    }

    // GH-253-1: Padding token ID
    if let Some(pad_val) = custom.get("tokenizer.padding_token_id") {
        if let Some(pad) = pad_val.as_u64() {
            metadata.push((
                "tokenizer.ggml.padding_token_id".to_string(),
                GgufValue::Uint32(pad as u32),
            ));
        }
    }

    // GH-253-1: add_bos_token flag
    if let Some(add_bos_val) = custom.get("tokenizer.add_bos_token") {
        if let Some(add_bos) = add_bos_val.as_bool() {
            metadata.push((
                "tokenizer.ggml.add_bos_token".to_string(),
                GgufValue::Bool(add_bos),
            ));
        }
    }

    // GH-253-1: Chat template (Jinja2)
    // Check APR metadata chat_template field first, then custom field
    let chat_tmpl = apr_metadata
        .chat_template
        .as_deref()
        .or_else(|| custom.get("tokenizer.chat_template").and_then(|v| v.as_str()));
    if let Some(tmpl) = chat_tmpl {
        metadata.push((
            "tokenizer.chat_template".to_string(),
            GgufValue::String(tmpl.to_string()),
        ));
    }

    eprintln!(
        "[PMAT-252] Writing {} metadata keys (arch={}, layers={}, heads={}/{}kv, hidden={})",
        metadata.len(),
        arch,
        num_layers,
        num_heads,
        num_kv_heads,
        hidden_size
    );

    // Build GGUF tensors with raw byte passthrough
    let tensor_names = reader.tensor_names();
    let mut gguf_tensors = Vec::with_capacity(tensor_names.len());

    for name in &tensor_names {
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

        let gguf_name = hf_to_gguf_name(name);

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

    // Write to file
    let file = File::create(output).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to create output file: {e}"),
    })?;
    let mut writer = BufWriter::new(file);

    export_tensors_to_gguf(&mut writer, &gguf_tensors, &metadata)?;

    let exported_size = fs::metadata(output).map(|m| m.len() as usize).unwrap_or(0);

    Ok(ExportReport {
        original_size,
        exported_size,
        tensor_count: gguf_tensors.len(),
        format: ExportFormat::Gguf,
        quantization: Some(QuantizationType::Q4K),
    })
}

/// Map HuggingFace-style tensor names to GGUF convention (GGUF-EXPORT-001)
///
/// Reverse of `Architecture::qwen2_map_name()` which maps GGUF→HF.
/// This maps HF→GGUF for export.
fn hf_to_gguf_name(name: &str) -> String {
    // Handle layer tensors: model.layers.N.suffix → blk.N.suffix
    if let Some(rest) = name.strip_prefix("model.layers.") {
        if let Some(dot_pos) = rest.find('.') {
            let layer_num = &rest[..dot_pos];
            let suffix = &rest[dot_pos + 1..];

            let gguf_suffix = match suffix {
                "self_attn.q_proj.weight" => "attn_q.weight",
                "self_attn.q_proj.bias" => "attn_q.bias",
                "self_attn.k_proj.weight" => "attn_k.weight",
                "self_attn.k_proj.bias" => "attn_k.bias",
                "self_attn.v_proj.weight" => "attn_v.weight",
                "self_attn.v_proj.bias" => "attn_v.bias",
                "self_attn.o_proj.weight" => "attn_output.weight",
                "self_attn.o_proj.bias" => "attn_output.bias",
                "self_attn.qkv_proj.weight" => "attn_qkv.weight",
                "self_attn.qkv_proj.bias" => "attn_qkv.bias",
                "input_layernorm.weight" => "attn_norm.weight",
                "mlp.gate_proj.weight" => "ffn_gate.weight",
                "mlp.up_proj.weight" => "ffn_up.weight",
                "mlp.down_proj.weight" => "ffn_down.weight",
                "post_attention_layernorm.weight" => "ffn_norm.weight",
                other => other, // Preserve unknown suffixes
            };

            return format!("blk.{layer_num}.{gguf_suffix}");
        }
    }

    // Handle non-layer tensors
    match name {
        "model.embed_tokens.weight" => "token_embd.weight".to_string(),
        "lm_head.weight" => "output.weight".to_string(),
        "model.norm.weight" => "output_norm.weight".to_string(),
        _ => name.to_string(), // Preserve unknown names
    }
}

// ============================================================================
// GH-182: COMPANION FILE HELPERS
// ============================================================================

/// Infer model config.json from tensor shapes (GH-182, GH-193)
///
/// Creates a HuggingFace-compatible config.json based on tensor analysis.
/// GH-193: Now includes all required fields for SafeTensors inference:
/// - num_attention_heads, intermediate_size, max_position_embeddings, etc.
fn infer_model_config(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> String {
    // GH-197 FIX: Track inference failures for diagnostics
    let mut inference_warnings: Vec<String> = Vec::new();

    // Infer hidden_size from embedding or first layer weight
    // BUG-EXPORT-001 FIX: GGUF and HuggingFace have different embedding layouts:
    //   - HuggingFace: [vocab_size, hidden_size]
    //   - GGUF: Can be [hidden_size, vocab_size] (transposed)
    // For LLMs: vocab_size >> hidden_size (32k-150k vs 512-8192), so pick the smaller dim
    let (hidden_size, hidden_inferred) = tensors
        .iter()
        .find(|(name, _)| name.contains("embed_tokens") || name.contains("token_embd"))
        .map(|(name, (_, shape))| {
            let dim = if shape.len() >= 2 {
                // Pick the smaller dimension (hidden_size is always << vocab_size)
                let dim0 = shape[0];
                let dim1 = shape[1];
                let inferred = dim0.min(dim1);
                eprintln!(
                    "[GH-197] Inferred hidden_size={inferred} from tensor '{name}' \
                     (shape={shape:?}, picked smaller dim)"
                );
                inferred
            } else {
                // 1D tensor - use as-is
                shape.last().copied().unwrap_or(4096)
            };
            (dim, true)
        })
        .unwrap_or_else(|| {
            inference_warnings.push(
                "hidden_size: No embed_tokens/token_embd tensor found, defaulting to 4096"
                    .to_string(),
            );
            (4096, false)
        });

    // Count layers by looking for layer patterns
    let layer_numbers: Vec<usize> = tensors
        .keys()
        .filter_map(|name| {
            if name.contains("layers.") || name.contains("blk.") {
                // Extract layer number from "layers.N." or "blk.N."
                let parts: Vec<&str> = name.split(&['.', '_'][..]).collect();
                for (i, part) in parts.iter().enumerate() {
                    if (*part == "layers" || *part == "blk") && i + 1 < parts.len() {
                        return parts[i + 1].parse::<usize>().ok();
                    }
                }
            }
            None
        })
        .collect();

    let num_layers = if let Some(&max_layer) = layer_numbers.iter().max() {
        let count = max_layer + 1;
        eprintln!("[GH-197] Inferred num_layers={count} from layer indices 0..{max_layer}");
        count
    } else {
        inference_warnings
            .push("num_layers: No layers.N/blk.N tensors found, defaulting to 12".to_string());
        12
    };

    // Infer vocab_size from lm_head, output weight, or embedding tensor
    // BUG-EXPORT-001 FIX: Use larger dimension (vocab_size >> hidden_size)
    // Also check embedding tensor since GGUF often uses weight tying
    let (vocab_size, vocab_inferred) = tensors
        .iter()
        .find(|(name, _)| name.contains("lm_head") || name.contains("output.weight"))
        .or_else(|| {
            // Fallback: use embedding tensor (GGUF weight tying)
            tensors
                .iter()
                .find(|(name, _)| name.contains("embed_tokens") || name.contains("token_embd"))
        })
        .map(|(name, (_, shape))| {
            let dim = if shape.len() >= 2 {
                // Pick the larger dimension (vocab_size is always >> hidden_size)
                let inferred = shape[0].max(shape[1]);
                eprintln!(
                    "[GH-197] Inferred vocab_size={inferred} from tensor '{name}' \
                     (shape={shape:?}, picked larger dim)"
                );
                inferred
            } else {
                shape.first().copied().unwrap_or(32000)
            };
            (dim, true)
        })
        .unwrap_or_else(|| {
            inference_warnings.push(
                "vocab_size: No lm_head/output/embed tensor found, defaulting to 32000".to_string(),
            );
            (32000, false)
        });

    // GH-197 FIX: Sanity validation - vocab_size should be >> hidden_size for LLMs
    if vocab_inferred && hidden_inferred && vocab_size < hidden_size {
        eprintln!(
            "[GH-197] WARNING: vocab_size ({vocab_size}) < hidden_size ({hidden_size}). \
             This is unusual for LLMs - dimensions may be swapped!"
        );
    }

    // Print all inference warnings
    for warning in &inference_warnings {
        eprintln!("[GH-197] WARNING: {warning}");
    }

    // GH-193: Infer num_attention_heads from attention Q/K/V weights
    // Shape is typically [hidden_size, num_heads * head_dim]
    let num_attention_heads = tensors
        .iter()
        .find(|(name, _)| {
            name.contains("self_attn.q_proj")
                || name.contains("attn.q_proj")
                || name.contains("attention.wq")
        })
        .map(|(_, (_, _shape))| {
            // Common head dimensions: 64, 128 - infer num_heads from hidden_size
            // Most models use head_dim = hidden_size / num_heads
            // Common configs: 4096/32=128 head_dim, 2048/16=128, etc.
            let head_dim = if hidden_size >= 4096 { 128 } else { 64 };
            hidden_size / head_dim
        })
        .unwrap_or_else(|| {
            // Fallback: standard ratios
            match hidden_size {
                896 => 14,                       // Qwen2.5-0.5B
                1536 => 12,                      // Qwen2.5-1.5B
                2048 => 16,                      // Llama-7B style
                4096 => 32,                      // Llama-7B
                5120 => 40,                      // Llama-13B
                8192 => 64,                      // Llama-70B
                _ => (hidden_size / 128).max(1), // Default: head_dim=128
            }
        });

    // GH-193: Infer intermediate_size from MLP weights
    let intermediate_size = tensors
        .iter()
        .find(|(name, _)| {
            name.contains("mlp.gate_proj")
                || name.contains("mlp.up_proj")
                || name.contains("feed_forward.w1")
        })
        .map(|(_, (_, shape))| shape.first().copied().unwrap_or(hidden_size * 4))
        .unwrap_or(hidden_size * 4); // Default to 4x hidden_size (common in transformers)

    // GH-193: Infer head_dim (guard against division by zero)
    let head_dim = if num_attention_heads > 0 {
        hidden_size / num_attention_heads
    } else {
        64 // Default head dimension
    };

    // GH-193: Infer num_key_value_heads (GQA support)
    // Look for k_proj shape to detect if using GQA (grouped query attention)
    let num_key_value_heads = tensors
        .iter()
        .find(|(name, _)| {
            name.contains("self_attn.k_proj")
                || name.contains("attn.k_proj")
                || name.contains("attention.wk")
        })
        .map(|(_, (_, shape))| {
            // For GQA: k_proj shape is [hidden_size, num_kv_heads * head_dim]
            // If shape[0] < hidden_size, it's GQA
            let kv_dim = shape.first().copied().unwrap_or(hidden_size);
            // Guard against division by zero
            if head_dim > 0 {
                (kv_dim / head_dim).max(1)
            } else {
                1
            }
        })
        .unwrap_or(num_attention_heads); // Default: same as num_attention_heads (MHA)

    // Create HuggingFace-compatible config.json with all required fields (GH-193)
    format!(
        r#"{{
  "architectures": ["Qwen2ForCausalLM"],
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": {hidden_size},
  "initializer_range": 0.02,
  "intermediate_size": {intermediate_size},
  "max_position_embeddings": 32768,
  "model_type": "qwen2",
  "num_attention_heads": {num_attention_heads},
  "num_hidden_layers": {num_layers},
  "num_key_value_heads": {num_key_value_heads},
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": {vocab_size}
}}"#
    )
}

/// Extract tokenizer.json from APR input file (GH-182)
///
/// If the input is APR format with embedded tokenizer, extract it.
/// Otherwise return empty string.
fn infer_tokenizer_json(input_path: &Path) -> String {
    let extension = input_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    if extension == "apr" {
        // Try to read APR metadata which may contain tokenizer
        if let Ok(data) = fs::read(input_path) {
            if data.len() > 44 {
                // Check for embedded tokenizer in APR metadata
                // APR format: header (44 bytes) + metadata (JSON) + tensors
                // Look for "tokenizer" in metadata section
                let metadata_start = 44;
                if let Some(metadata_end) = data[metadata_start..]
                    .windows(4)
                    .position(|w| w == b"}\n\n\n" || w == b"}\r\n\r")
                    .map(|p| metadata_start + p + 1)
                {
                    if let Ok(metadata_str) =
                        std::str::from_utf8(&data[metadata_start..metadata_end])
                    {
                        // Check if tokenizer data is embedded
                        if metadata_str.contains("\"tokenizer\"")
                            || metadata_str.contains("\"vocabulary\"")
                        {
                            // For now, return a minimal tokenizer.json
                            // In production, we'd extract the actual vocabulary
                            return r#"{"version": "1.0", "model": {"type": "BPE"}}"#.to_string();
                        }
                    }
                }
            }
        }
    }

    // Return empty if no tokenizer found
    String::new()
}

/// ROSETTA-003: Read APR v2 metadata from file.
///
/// Returns `None` for non-APR files or on any read/parse failure.
fn read_apr_metadata(apr_path: &Path) -> Option<crate::format::v2::AprV2Metadata> {
    if apr_path.extension().and_then(|e| e.to_str()) != Some("apr") {
        return None;
    }
    let data = fs::read(apr_path).ok()?;
    let reader = crate::format::v2::AprV2Reader::from_bytes(&data).ok()?;
    Some(reader.metadata().clone())
}

/// ROSETTA-003: Unfuse legacy QKV tensors for lossless round-trip export.
///
/// Old APR files (pre-ROSETTA-003) stored fused `qkv_proj.weight` tensors.
/// This function splits them back into separate `q_proj`, `k_proj`, `v_proj`
/// for correct GGUF/SafeTensors export. New APR files with separate Q/K/V
/// pass through unchanged.
fn unfuse_qkv_tensors(
    tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    apr_path: &Path,
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let has_fused = tensors.keys().any(|k| k.contains("qkv_proj."));
    if !has_fused {
        return tensors;
    }

    let metadata = read_apr_metadata(apr_path);
    let (hidden_size, num_heads, num_kv_heads) = match &metadata {
        Some(m) => {
            let hs = m.hidden_size.unwrap_or(0);
            let nh = m.num_heads.unwrap_or(0);
            let nkv = m.num_kv_heads.unwrap_or(nh);
            (hs, nh, nkv)
        }
        None => return tensors,
    };

    if hidden_size == 0 || num_heads == 0 {
        return tensors;
    }

    let head_dim = hidden_size / num_heads;
    let kv_dim = num_kv_heads * head_dim;

    let mut result = BTreeMap::new();

    for (name, (data, shape)) in tensors {
        if name.contains("qkv_proj.weight") {
            // Split [Q;K;V] weight back into separate tensors.
            // Shape was [qkv_dim, hidden_dim] where qkv_dim = hidden_size + 2*kv_dim.
            let hidden_dim = if shape.len() >= 2 {
                shape[1]
            } else {
                hidden_size
            };
            let q_elements = hidden_size * hidden_dim;
            let kv_elements = kv_dim * hidden_dim;

            if data.len() >= q_elements + 2 * kv_elements {
                let prefix = name.strip_suffix("qkv_proj.weight").unwrap_or(&name);

                result.insert(
                    format!("{prefix}q_proj.weight"),
                    (data[..q_elements].to_vec(), vec![hidden_size, hidden_dim]),
                );
                result.insert(
                    format!("{prefix}k_proj.weight"),
                    (
                        data[q_elements..q_elements + kv_elements].to_vec(),
                        vec![kv_dim, hidden_dim],
                    ),
                );
                result.insert(
                    format!("{prefix}v_proj.weight"),
                    (
                        data[q_elements + kv_elements..q_elements + 2 * kv_elements].to_vec(),
                        vec![kv_dim, hidden_dim],
                    ),
                );
            } else {
                result.insert(name, (data, shape));
            }
        } else if name.contains("qkv_proj.bias") {
            // Split [Q_bias; K_bias; V_bias] back into separate biases.
            let qkv_dim = hidden_size + 2 * kv_dim;
            if data.len() == qkv_dim {
                let prefix = name.strip_suffix("qkv_proj.bias").unwrap_or(&name);

                result.insert(
                    format!("{prefix}q_proj.bias"),
                    (data[..hidden_size].to_vec(), vec![hidden_size]),
                );
                result.insert(
                    format!("{prefix}k_proj.bias"),
                    (
                        data[hidden_size..hidden_size + kv_dim].to_vec(),
                        vec![kv_dim],
                    ),
                );
                result.insert(
                    format!("{prefix}v_proj.bias"),
                    (data[hidden_size + kv_dim..].to_vec(), vec![kv_dim]),
                );
            } else {
                result.insert(name, (data, shape));
            }
        } else {
            result.insert(name, (data, shape));
        }
    }

    result
}

/// ROSETTA-003: Remove synthesized `lm_head.weight` for SafeTensors export.
///
/// When the APR metadata has `tied_embeddings: true`, the lm_head was copied
/// from embed_tokens during import. For SafeTensors round-trip fidelity,
/// remove it so the exported file matches the original HuggingFace convention.
fn remove_tied_lm_head(
    mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    apr_path: &Path,
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let metadata = read_apr_metadata(apr_path);
    let is_tied = metadata
        .as_ref()
        .and_then(|m| m.custom.get("tied_embeddings"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if is_tied {
        tensors.remove("lm_head.weight");
    }

    tensors
}

/// PMAT-223: Extract user metadata from APR file's custom field.
///
/// Reads the APR metadata JSON and looks for the `"source_metadata"` key
/// that was preserved during import from SafeTensors.
fn extract_user_metadata(apr_path: &Path) -> UserMetadata {
    let data = match fs::read(apr_path) {
        Ok(d) => d,
        Err(_) => return UserMetadata::new(),
    };

    // APR v2 format: magic(4) + version(4) + metadata_len(8) + metadata_json
    if data.len() < 16 {
        return UserMetadata::new();
    }

    let metadata_len = u64::from_le_bytes(data[8..16].try_into().unwrap_or([0u8; 8])) as usize;

    if data.len() < 16 + metadata_len {
        return UserMetadata::new();
    }

    let metadata_json = match std::str::from_utf8(&data[16..16 + metadata_len]) {
        Ok(s) => s,
        Err(_) => return UserMetadata::new(),
    };

    let parsed: serde_json::Value = match serde_json::from_str(metadata_json) {
        Ok(v) => v,
        Err(_) => return UserMetadata::new(),
    };

    // Look for custom.source_metadata
    if let Some(serde_json::Value::Object(map)) =
        parsed.get("custom").and_then(|c| c.get("source_metadata"))
    {
        let mut result = UserMetadata::new();
        for (k, v) in map {
            if let serde_json::Value::String(s) = v {
                result.insert(k.clone(), s.clone());
            }
        }
        return result;
    }

    UserMetadata::new()
}

/// Detect predominant quantization type from an APR file (PMAT-252).
///
/// Reads the tensor index and checks the dtype of 2D weight tensors.
/// Returns the `QuantizationType` if the majority of weights use a
/// quantized format (Q4K, Q6K), or `None` for F32/F16 files.
pub(crate) fn detect_apr_quantization(apr_path: &Path) -> Option<QuantizationType> {
    use crate::format::v2::{AprV2Reader, TensorDType};

    let data = fs::read(apr_path).ok()?;
    let reader = AprV2Reader::from_bytes(&data).ok()?;

    // Count dtypes across 2D weight tensors (skip 1D biases/norms)
    let mut q4k_count = 0usize;
    let mut q6k_count = 0usize;
    let mut other_count = 0usize;

    for name in reader.tensor_names() {
        if let Some(entry) = reader.get_tensor(name) {
            if entry.shape.len() >= 2 {
                match entry.dtype {
                    TensorDType::Q4K => q4k_count += 1,
                    TensorDType::Q6K => q6k_count += 1,
                    _ => other_count += 1,
                }
            }
        }
    }

    let total = q4k_count + q6k_count + other_count;
    if total == 0 {
        return None;
    }

    // If majority of 2D tensors are Q4K, default to Q4K export
    if q4k_count > q6k_count && q4k_count > other_count {
        return Some(QuantizationType::Q4K);
    }

    // Q6K not yet in QuantizationType — treat as no auto-detect
    None
}

/// Detect GGUF model architecture for tensor name mapping (GH-200).
fn detect_gguf_architecture(path: &Path) -> Architecture {
    GgufReader::from_file(path)
        .ok()
        .and_then(|r| r.architecture())
        .map(|a| match a.to_lowercase().as_str() {
            "qwen2" | "qwen" => Architecture::Qwen2,
            "llama" => Architecture::Llama,
            _ => Architecture::Qwen2, // Safe default: most GGUF models use same mapping
        })
        .unwrap_or(Architecture::Qwen2)
}

/// Infer vocab_size and hidden_dim from tensor shapes.
///
/// Used for contract validation during export.
fn infer_vocab_hidden(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> (usize, usize) {
    let mut vocab_size = 0;
    let mut hidden_dim = 0;

    // Try embedding first (most reliable)
    for (name, (_, shape)) in tensors {
        if (name.contains("embed_tokens") || name.contains("token_embd")) && shape.len() == 2 {
            // Embedding shape: [vocab_size, hidden_dim]
            vocab_size = shape[0];
            hidden_dim = shape[1];
            break;
        }
    }

    // Fallback to lm_head
    if vocab_size == 0 {
        for (name, (_, shape)) in tensors {
            if (name.contains("lm_head") || name.contains("output.weight")) && shape.len() == 2 {
                vocab_size = shape[0];
                hidden_dim = shape[1];
                break;
            }
        }
    }

    // Fallback to layer weights for hidden_dim
    if hidden_dim == 0 {
        for (name, (_, shape)) in tensors {
            if name.contains("q_proj") && shape.len() == 2 {
                hidden_dim = shape[1];
                break;
            }
        }
    }

    (vocab_size, hidden_dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    // ========================================================================
    // hf_to_gguf_name: Attention projection patterns
    // ========================================================================

    #[test]
    fn test_hf_to_gguf_name_q_proj_weight() {
        assert_eq!(
            hf_to_gguf_name("model.layers.0.self_attn.q_proj.weight"),
            "blk.0.attn_q.weight"
        );
    }

    #[test]
    fn test_hf_to_gguf_name_q_proj_bias() {
        assert_eq!(
            hf_to_gguf_name("model.layers.5.self_attn.q_proj.bias"),
            "blk.5.attn_q.bias"
        );
    }

    #[test]
    fn test_hf_to_gguf_name_k_proj_weight() {
        assert_eq!(
            hf_to_gguf_name("model.layers.3.self_attn.k_proj.weight"),
            "blk.3.attn_k.weight"
        );
    }

    #[test]
    fn test_hf_to_gguf_name_k_proj_bias() {
        assert_eq!(
            hf_to_gguf_name("model.layers.11.self_attn.k_proj.bias"),
            "blk.11.attn_k.bias"
        );
    }

    #[test]
    fn test_hf_to_gguf_name_v_proj_weight() {
        assert_eq!(
            hf_to_gguf_name("model.layers.7.self_attn.v_proj.weight"),
            "blk.7.attn_v.weight"
        );
    }

    #[test]
    fn test_hf_to_gguf_name_v_proj_bias() {
        assert_eq!(
            hf_to_gguf_name("model.layers.0.self_attn.v_proj.bias"),
            "blk.0.attn_v.bias"
        );
    }

    #[test]
    fn test_hf_to_gguf_name_o_proj_weight() {
        assert_eq!(
            hf_to_gguf_name("model.layers.2.self_attn.o_proj.weight"),
            "blk.2.attn_output.weight"
        );
    }

    #[test]
    fn test_hf_to_gguf_name_o_proj_bias() {
        assert_eq!(
            hf_to_gguf_name("model.layers.31.self_attn.o_proj.bias"),
            "blk.31.attn_output.bias"
        );
    }

    // ========================================================================
    // hf_to_gguf_name: Fused QKV patterns
    // ========================================================================

    #[test]
    fn test_hf_to_gguf_name_qkv_proj_weight() {
        assert_eq!(
            hf_to_gguf_name("model.layers.0.self_attn.qkv_proj.weight"),
            "blk.0.attn_qkv.weight"
        );
    }

    #[test]
    fn test_hf_to_gguf_name_qkv_proj_bias() {
        assert_eq!(
            hf_to_gguf_name("model.layers.4.self_attn.qkv_proj.bias"),
            "blk.4.attn_qkv.bias"
        );
    }

    // ========================================================================
    // hf_to_gguf_name: MLP / FFN patterns
    // ========================================================================

    #[test]
    fn test_hf_to_gguf_name_gate_proj() {
        assert_eq!(
            hf_to_gguf_name("model.layers.1.mlp.gate_proj.weight"),
            "blk.1.ffn_gate.weight"
        );
    }

    #[test]
    fn test_hf_to_gguf_name_up_proj() {
        assert_eq!(
            hf_to_gguf_name("model.layers.10.mlp.up_proj.weight"),
            "blk.10.ffn_up.weight"
        );
    }

    #[test]
    fn test_hf_to_gguf_name_down_proj() {
        assert_eq!(
            hf_to_gguf_name("model.layers.23.mlp.down_proj.weight"),
            "blk.23.ffn_down.weight"
        );
    }

    // ========================================================================
    // hf_to_gguf_name: Layer norm patterns
    // ========================================================================

    #[test]
    fn test_hf_to_gguf_name_input_layernorm() {
        assert_eq!(
            hf_to_gguf_name("model.layers.0.input_layernorm.weight"),
            "blk.0.attn_norm.weight"
        );
    }

    #[test]
    fn test_hf_to_gguf_name_post_attention_layernorm() {
        assert_eq!(
            hf_to_gguf_name("model.layers.15.post_attention_layernorm.weight"),
            "blk.15.ffn_norm.weight"
        );
    }

    // ========================================================================
    // hf_to_gguf_name: Non-layer tensors (embedding, lm_head, output norm)
    // ========================================================================

    #[test]
    fn test_hf_to_gguf_name_embed_tokens() {
        assert_eq!(
            hf_to_gguf_name("model.embed_tokens.weight"),
            "token_embd.weight"
        );
    }

    #[test]
    fn test_hf_to_gguf_name_lm_head() {
        assert_eq!(hf_to_gguf_name("lm_head.weight"), "output.weight");
    }

    #[test]
    fn test_hf_to_gguf_name_output_norm() {
        assert_eq!(hf_to_gguf_name("model.norm.weight"), "output_norm.weight");
    }

    // ========================================================================
    // hf_to_gguf_name: Unknown / passthrough
    // ========================================================================

    #[test]
    fn test_hf_to_gguf_name_unknown_passthrough() {
        // Completely unknown names should pass through unchanged
        assert_eq!(hf_to_gguf_name("some.custom.tensor"), "some.custom.tensor");
    }

    #[test]
    fn test_hf_to_gguf_name_unknown_layer_suffix_passthrough() {
        // A layer tensor with an unrecognized suffix passes through as-is
        assert_eq!(
            hf_to_gguf_name("model.layers.0.some_unknown_suffix.weight"),
            "blk.0.some_unknown_suffix.weight"
        );
    }

    #[test]
    fn test_hf_to_gguf_name_empty_string() {
        assert_eq!(hf_to_gguf_name(""), "");
    }

    // ========================================================================
    // hf_to_gguf_name: Multi-digit layer indices (regression guard)
    //
    // Bug class: off-by-one in layer number parsing. Models with 100+ layers
    // (e.g. Llama-70B has 80 layers) must produce correct multi-digit indices.
    // ========================================================================

    #[test]
    fn test_hf_to_gguf_name_high_layer_index() {
        assert_eq!(
            hf_to_gguf_name("model.layers.79.self_attn.q_proj.weight"),
            "blk.79.attn_q.weight"
        );
    }

    #[test]
    fn test_hf_to_gguf_name_three_digit_layer_index() {
        // Some very deep models exceed 100 layers
        assert_eq!(
            hf_to_gguf_name("model.layers.127.mlp.gate_proj.weight"),
            "blk.127.ffn_gate.weight"
        );
    }

    // ========================================================================
    // hf_to_gguf_name: Consistency — roundtrip with Architecture::qwen2_map_name
    //
    // Bug class: asymmetric mapping. If HF->GGUF and GGUF->HF aren't inverses,
    // round-trip export/import corrupts tensor names silently.
    // ========================================================================

    #[test]
    fn test_hf_to_gguf_name_all_layer_suffixes_mapped() {
        // Verify every known suffix produces a different GGUF name (no collisions)
        let suffixes = [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ];

        let mut gguf_names: Vec<String> = suffixes
            .iter()
            .map(|s| hf_to_gguf_name(&format!("model.layers.0.{s}")))
            .collect();

        let count_before = gguf_names.len();
        gguf_names.sort();
        gguf_names.dedup();
        assert_eq!(
            gguf_names.len(),
            count_before,
            "Name collision detected in hf_to_gguf_name mapping"
        );
    }

    // ========================================================================
    // infer_vocab_hidden: Embedding tensor present
    // ========================================================================

    #[test]
    fn test_infer_vocab_hidden_from_embed_tokens() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // Qwen2-0.5B: vocab=151936, hidden=896
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.0; 151936 * 896], vec![151936, 896]),
        );
        let (vocab, hidden) = infer_vocab_hidden(&tensors);
        assert_eq!(vocab, 151936);
        assert_eq!(hidden, 896);
    }

    #[test]
    fn test_infer_vocab_hidden_from_token_embd() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // GGUF-style naming
        tensors.insert(
            "token_embd.weight".to_string(),
            (vec![0.0; 32000 * 4096], vec![32000, 4096]),
        );
        let (vocab, hidden) = infer_vocab_hidden(&tensors);
        assert_eq!(vocab, 32000);
        assert_eq!(hidden, 4096);
    }

    // ========================================================================
    // infer_vocab_hidden: Fallback to lm_head
    // ========================================================================

    #[test]
    fn test_infer_vocab_hidden_fallback_to_lm_head() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // No embedding tensor, but lm_head present
        tensors.insert(
            "lm_head.weight".to_string(),
            (vec![0.0; 32000 * 4096], vec![32000, 4096]),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![0.0; 4096 * 4096], vec![4096, 4096]),
        );
        let (vocab, hidden) = infer_vocab_hidden(&tensors);
        assert_eq!(vocab, 32000);
        assert_eq!(hidden, 4096);
    }

    #[test]
    fn test_infer_vocab_hidden_fallback_to_output_weight() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // GGUF-style output.weight as lm_head equivalent
        tensors.insert(
            "output.weight".to_string(),
            (vec![0.0; 128256 * 4096], vec![128256, 4096]),
        );
        let (vocab, hidden) = infer_vocab_hidden(&tensors);
        assert_eq!(vocab, 128256);
        assert_eq!(hidden, 4096);
    }

    // ========================================================================
    // infer_vocab_hidden: Fallback to q_proj for hidden_dim only
    // ========================================================================

    #[test]
    fn test_infer_vocab_hidden_hidden_from_q_proj() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // No embedding or lm_head — only layer weights
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![0.0; 4096 * 4096], vec![4096, 4096]),
        );
        tensors.insert(
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            (vec![0.0; 11008 * 4096], vec![11008, 4096]),
        );
        let (vocab, hidden) = infer_vocab_hidden(&tensors);
        // vocab_size cannot be inferred without embedding/lm_head
        assert_eq!(vocab, 0);
        assert_eq!(hidden, 4096);
    }

    // ========================================================================
    // infer_vocab_hidden: Empty tensor map
    // ========================================================================

    #[test]
    fn test_infer_vocab_hidden_empty_tensors() {
        let tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        let (vocab, hidden) = infer_vocab_hidden(&tensors);
        assert_eq!(vocab, 0);
        assert_eq!(hidden, 0);
    }

    // ========================================================================
    // infer_vocab_hidden: 1D tensors (should NOT match — requires 2D)
    //
    // Bug class: 1D norm weights like input_layernorm.weight with shape [4096]
    // could be misinterpreted as vocab_size=4096, hidden=0 if the dimension
    // check is missing.
    // ========================================================================

    #[test]
    fn test_infer_vocab_hidden_ignores_1d_embedding() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // 1D tensor should NOT be treated as an embedding
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.0; 4096], vec![4096]),
        );
        let (vocab, hidden) = infer_vocab_hidden(&tensors);
        // 1D tensor doesn't satisfy the shape.len() == 2 check
        assert_eq!(vocab, 0);
        assert_eq!(hidden, 0);
    }

    #[test]
    fn test_infer_vocab_hidden_ignores_1d_lm_head() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // 1D lm_head should NOT match
        tensors.insert(
            "lm_head.weight".to_string(),
            (vec![0.0; 32000], vec![32000]),
        );
        // But 2D q_proj should give hidden_dim
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![0.0; 2048 * 2048], vec![2048, 2048]),
        );
        let (vocab, hidden) = infer_vocab_hidden(&tensors);
        assert_eq!(vocab, 0); // 1D lm_head skipped
        assert_eq!(hidden, 2048); // from q_proj fallback
    }

    // ========================================================================
    // infer_vocab_hidden: Embedding takes priority over lm_head
    //
    // Bug class: If lm_head is checked first and embed_tokens second, the
    // wrong tensor could provide dimensions for tied-embedding models where
    // lm_head and embed_tokens have different names but same data.
    // ========================================================================

    #[test]
    fn test_infer_vocab_hidden_embedding_priority_over_lm_head() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // Both present — embedding should win
        tensors.insert(
            "lm_head.weight".to_string(),
            (vec![0.0; 32000 * 4096], vec![32000, 4096]),
        );
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.0; 151936 * 896], vec![151936, 896]),
        );
        let (vocab, hidden) = infer_vocab_hidden(&tensors);
        // Embedding shapes should be used, not lm_head
        assert_eq!(vocab, 151936);
        assert_eq!(hidden, 896);
    }

    // ========================================================================
    // unfuse_qkv_tensors: Passthrough when no fused tensors present
    // ========================================================================

    #[test]
    fn test_unfuse_qkv_tensors_no_fused_passthrough() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![1.0; 16], vec![4, 4]),
        );
        tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            (vec![2.0; 16], vec![4, 4]),
        );
        tensors.insert(
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            (vec![3.0; 16], vec![4, 4]),
        );

        // Non-APR path means read_apr_metadata returns None, but since no
        // fused tensors exist, the early return fires first.
        let result = unfuse_qkv_tensors(tensors.clone(), Path::new("/tmp/fake.safetensors"));

        assert_eq!(result.len(), 3);
        assert!(result.contains_key("model.layers.0.self_attn.q_proj.weight"));
        assert!(result.contains_key("model.layers.0.self_attn.k_proj.weight"));
        assert!(result.contains_key("model.layers.0.self_attn.v_proj.weight"));
    }

    // ========================================================================
    // unfuse_qkv_tensors: Fused tensors present but no APR metadata
    //
    // Bug class: If metadata is unavailable (non-APR input, corrupt file),
    // the function should return tensors unchanged rather than panicking.
    // ========================================================================

    #[test]
    fn test_unfuse_qkv_tensors_fused_but_no_metadata_returns_unchanged() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.layers.0.self_attn.qkv_proj.weight".to_string(),
            (vec![1.0; 48], vec![12, 4]),
        );
        tensors.insert(
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            (vec![2.0; 16], vec![4, 4]),
        );

        // Non-APR path -> read_apr_metadata returns None -> early return with original tensors
        let result = unfuse_qkv_tensors(tensors.clone(), Path::new("/tmp/nonexistent.safetensors"));

        // Should be unchanged since metadata couldn't be read
        assert_eq!(result.len(), 2);
        assert!(result.contains_key("model.layers.0.self_attn.qkv_proj.weight"));
    }

    // ========================================================================
    // ExportFormat: from_str edge cases beyond what coverage_types tests
    // ========================================================================

    #[test]
    fn test_export_format_from_str_mixed_case() {
        use std::str::FromStr;
        // Mixed case should work because of to_lowercase()
        assert_eq!(
            ExportFormat::from_str("SafeTensors"),
            Ok(ExportFormat::SafeTensors)
        );
        assert_eq!(ExportFormat::from_str("GgUf"), Ok(ExportFormat::Gguf));
        assert_eq!(
            ExportFormat::from_str("TorchScript"),
            Ok(ExportFormat::TorchScript)
        );
        assert_eq!(ExportFormat::from_str("PT"), Ok(ExportFormat::TorchScript));
    }

    #[test]
    fn test_export_format_from_str_error_message_contains_input() {
        use std::str::FromStr;
        // The error message should contain the bad input for debugging
        let err = ExportFormat::from_str("parquet").unwrap_err();
        assert!(
            err.contains("parquet"),
            "Error should contain the unrecognized input, got: {err}"
        );
    }

    // ========================================================================
    // ExportFormat: extension + is_supported consistency
    //
    // Bug class: Adding a new format variant without updating extension() or
    // is_supported() leads to runtime panics or silent mis-routing.
    // ========================================================================

    #[test]
    fn test_export_format_all_variants_have_nonempty_extension() {
        let all_formats = [
            ExportFormat::SafeTensors,
            ExportFormat::Gguf,
            ExportFormat::Onnx,
            ExportFormat::TorchScript,
        ];
        for fmt in &all_formats {
            let ext = fmt.extension();
            assert!(!ext.is_empty(), "Format {:?} has empty extension", fmt);
            // Extension should not contain dots (it's the bare extension)
            assert!(
                !ext.contains('.'),
                "Format {:?} extension contains a dot: {}",
                fmt,
                ext
            );
        }
    }

    #[test]
    fn test_export_format_supported_formats_have_valid_extensions() {
        // Every supported format should produce a known extension
        let supported: Vec<ExportFormat> = [
            ExportFormat::SafeTensors,
            ExportFormat::Gguf,
            ExportFormat::Onnx,
            ExportFormat::TorchScript,
        ]
        .into_iter()
        .filter(|f| f.is_supported())
        .collect();

        assert!(
            supported.len() >= 2,
            "At least SafeTensors and Gguf should be supported"
        );
        for fmt in &supported {
            assert!(
                ["safetensors", "gguf"].contains(&fmt.extension()),
                "Supported format {:?} has unexpected extension: {}",
                fmt,
                fmt.extension()
            );
        }
    }

    // ========================================================================
    // ExportOptions: Default values
    // ========================================================================

    #[test]
    fn test_export_options_default() {
        let opts = ExportOptions::default();
        assert_eq!(opts.format, ExportFormat::SafeTensors);
        assert!(opts.quantize.is_none());
        assert!(opts.include_tokenizer);
        assert!(opts.include_config);
    }

    // ========================================================================
    // ExportReport: Clone and Debug
    // ========================================================================

    #[test]
    fn test_export_report_clone_and_debug() {
        let report = ExportReport {
            original_size: 1024,
            exported_size: 512,
            tensor_count: 42,
            format: ExportFormat::Gguf,
            quantization: Some(QuantizationType::Q4K),
        };
        let cloned = report.clone();
        assert_eq!(cloned.original_size, 1024);
        assert_eq!(cloned.exported_size, 512);
        assert_eq!(cloned.tensor_count, 42);
        assert_eq!(cloned.format, ExportFormat::Gguf);

        // Debug should not panic
        let debug_str = format!("{:?}", report);
        assert!(debug_str.contains("ExportReport"));
    }

    // ========================================================================
    // infer_vocab_hidden: Realistic multi-tensor model configurations
    //
    // Bug class: The function iterates a BTreeMap (sorted order). If the
    // first matching tensor has unexpected shape, inference fails silently.
    // These tests use realistic tensor maps mimicking real models.
    // ========================================================================

    #[test]
    fn test_infer_vocab_hidden_llama_7b_layout() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // Llama-7B: vocab=32000, hidden=4096, 32 layers
        // Use small data vectors (shapes are what matter)
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![32000, 4096]),
        );
        tensors.insert("lm_head.weight".to_string(), (vec![], vec![32000, 4096]));
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![], vec![4096, 4096]),
        );
        tensors.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            (vec![], vec![4096]),
        );
        let (vocab, hidden) = infer_vocab_hidden(&tensors);
        assert_eq!(vocab, 32000);
        assert_eq!(hidden, 4096);
    }

    #[test]
    fn test_infer_vocab_hidden_qwen2_05b_layout() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // Qwen2-0.5B: vocab=151936, hidden=896, 24 layers
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![151936, 896]),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![], vec![896, 896]),
        );
        let (vocab, hidden) = infer_vocab_hidden(&tensors);
        assert_eq!(vocab, 151936);
        assert_eq!(hidden, 896);
    }

    #[test]
    fn test_infer_vocab_hidden_only_unrelated_tensors() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // Tensors that don't match any known patterns
        tensors.insert(
            "encoder.block.0.weight".to_string(),
            (vec![], vec![512, 512]),
        );
        tensors.insert(
            "decoder.block.0.weight".to_string(),
            (vec![], vec![512, 512]),
        );
        let (vocab, hidden) = infer_vocab_hidden(&tensors);
        assert_eq!(vocab, 0);
        assert_eq!(hidden, 0);
    }

    // ========================================================================
    // infer_model_config: Basic Qwen2-0.5B layout
    // ========================================================================

    #[test]
    fn test_infer_model_config_qwen2_05b_layout() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // Qwen2-0.5B: vocab=151936, hidden=896, 24 layers, 14 heads
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![151936, 896]),
        );
        // 24 layers (0..23)
        for i in 0..24 {
            tensors.insert(
                format!("model.layers.{i}.self_attn.q_proj.weight"),
                (vec![], vec![896, 896]),
            );
            tensors.insert(
                format!("model.layers.{i}.self_attn.k_proj.weight"),
                (vec![], vec![128, 896]),
            );
            tensors.insert(
                format!("model.layers.{i}.mlp.gate_proj.weight"),
                (vec![], vec![4864, 896]),
            );
        }

        let config = infer_model_config(&tensors);
        // Parse as JSON to verify fields
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

        assert_eq!(v["hidden_size"], 896);
        assert_eq!(v["num_hidden_layers"], 24);
        assert_eq!(v["vocab_size"], 151936);
        assert_eq!(v["num_attention_heads"], 14);
        assert_eq!(v["intermediate_size"], 4864);
    }

    // ========================================================================
    // infer_model_config: Llama-7B layout
    // ========================================================================

    #[test]
    fn test_infer_model_config_llama_7b_layout() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // Llama-7B: vocab=32000, hidden=4096, 32 layers
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![32000, 4096]),
        );
        tensors.insert("lm_head.weight".to_string(), (vec![], vec![32000, 4096]));
        for i in 0..32 {
            tensors.insert(
                format!("model.layers.{i}.self_attn.q_proj.weight"),
                (vec![], vec![4096, 4096]),
            );
            tensors.insert(
                format!("model.layers.{i}.self_attn.k_proj.weight"),
                (vec![], vec![4096, 4096]),
            );
            tensors.insert(
                format!("model.layers.{i}.mlp.gate_proj.weight"),
                (vec![], vec![11008, 4096]),
            );
        }

        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

        assert_eq!(v["hidden_size"], 4096);
        assert_eq!(v["num_hidden_layers"], 32);
        assert_eq!(v["vocab_size"], 32000);
        assert_eq!(v["num_attention_heads"], 32);
        assert_eq!(v["intermediate_size"], 11008);
        // MHA: num_key_value_heads == num_attention_heads
        assert_eq!(v["num_key_value_heads"], 32);
    }

    // ========================================================================
    // infer_model_config: Empty tensors - all defaults
    // ========================================================================

    #[test]
    fn test_infer_model_config_empty_tensors_returns_defaults() {
        let tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

        // Defaults when nothing is inferable
        assert_eq!(v["hidden_size"], 4096);
        assert_eq!(v["num_hidden_layers"], 12);
        assert_eq!(v["vocab_size"], 32000);
    }

    // ========================================================================
    // infer_model_config: No embedding, fallback to lm_head
    // ========================================================================

    #[test]
    fn test_infer_model_config_no_embedding_fallback_to_lm_head() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // Only lm_head, no embed_tokens
        tensors.insert("lm_head.weight".to_string(), (vec![], vec![32000, 4096]));
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![], vec![4096, 4096]),
        );

        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

        // hidden_size defaults to 4096 (no embed_tokens/token_embd found)
        assert_eq!(v["hidden_size"], 4096);
        // vocab_size inferred from lm_head (larger dim)
        assert_eq!(v["vocab_size"], 32000);
    }

    // ========================================================================
    // infer_model_config: No layer tensors - num_layers defaults to 12
    // ========================================================================

    #[test]
    fn test_infer_model_config_no_layers_defaults_to_12() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // Only embedding, no layer tensors
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![32000, 4096]),
        );

        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

        assert_eq!(v["num_hidden_layers"], 12);
    }

    // ========================================================================
    // infer_model_config: GGUF-style tensor names (blk.N / token_embd)
    // ========================================================================

    #[test]
    fn test_infer_model_config_gguf_style_names() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // GGUF naming convention
        tensors.insert("token_embd.weight".to_string(), (vec![], vec![32000, 4096]));
        for i in 0..16 {
            tensors.insert(format!("blk.{i}.attn_q.weight"), (vec![], vec![4096, 4096]));
        }

        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

        // hidden_size from token_embd (smaller dim)
        assert_eq!(v["hidden_size"], 4096);
        // num_layers from blk.0..blk.15 => 16 layers
        assert_eq!(v["num_hidden_layers"], 16);
        // vocab from token_embd (larger dim)
        assert_eq!(v["vocab_size"], 32000);
    }

    // ========================================================================
    // infer_model_config: Vocab < hidden triggers warning but still works
    // ========================================================================

    #[test]
    fn test_infer_model_config_vocab_less_than_hidden_unusual() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // Unusual scenario: embedding where both dims are close
        // embed_tokens shape [100, 4096] -> hidden=100 (min), vocab from lm_head
        // lm_head shape [200, 100] -> vocab=200 (max)
        // This results in vocab=200 < hidden? No, hidden=100 and vocab=200 so vocab > hidden.
        // To trigger vocab < hidden, we need a specific arrangement.
        // embed_tokens [512, 4096] -> hidden=512 (min dim)
        // No lm_head -> fallback to embed_tokens for vocab, vocab = 4096 (max dim)
        // So vocab=4096 > hidden=512 -- still fine.
        // The warning triggers when vocab_inferred && hidden_inferred && vocab < hidden.
        // This happens if the embedding shape is [small, large] -> hidden=small, vocab=large
        // and lm_head shape [small2, large2] with max=large2 but large2 < hidden.
        // Use: embed_tokens [8192, 256] -> hidden=256, vocab=8192 from embed_tokens fallback.
        // Actually lm_head is checked first for vocab. If lm_head has shape [100, 256]:
        //   vocab = max(100, 256) = 256, but hidden = 256 -> vocab == hidden, not less.
        // Use: lm_head [100, 50] -> vocab=100 (max), but hidden from embed_tokens min dim.
        // embed_tokens [200, 50] -> hidden=50 (min). vocab from lm_head = max(100,50)=100.
        // Now vocab=100 > hidden=50. Still no warning.
        // We need vocab < hidden: embed_tokens [200,50] -> hidden=50.
        // lm_head must give vocab < 50. lm_head [30, 20] -> vocab=30.
        // vocab_inferred=true, hidden_inferred=true, vocab=30 < hidden=50 -> WARNING
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![200, 50]),
        );
        tensors.insert("lm_head.weight".to_string(), (vec![], vec![30, 20]));

        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

        // hidden = min(200, 50) = 50
        assert_eq!(v["hidden_size"], 50);
        // vocab = max(30, 20) = 30 (from lm_head, checked before embed_tokens fallback)
        assert_eq!(v["vocab_size"], 30);
        // The function should still produce valid JSON despite the warning
    }

    // ========================================================================
    // infer_model_config: GQA detection (k_proj < q_proj)
    // ========================================================================

    #[test]
    fn test_infer_model_config_gqa_detection() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // Model with GQA: hidden=4096, 32 attention heads, 8 KV heads
        // head_dim = 4096/32 = 128
        // k_proj shape: [num_kv_heads * head_dim, hidden] = [1024, 4096]
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![32000, 4096]),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![], vec![4096, 4096]),
        );
        tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            (vec![], vec![1024, 4096]),
        );

        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

        assert_eq!(v["num_attention_heads"], 32);
        // GQA: k_proj first dim = 1024, head_dim = 128
        // num_kv_heads = 1024 / 128 = 8
        assert_eq!(v["num_key_value_heads"], 8);
    }

    // ========================================================================
    // infer_model_config: 1D embedding tensor fallback
    // ========================================================================

    #[test]
    fn test_infer_model_config_1d_embedding_fallback() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // 1D embedding tensor - uses last dim as-is
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![4096]),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![], vec![4096, 4096]),
        );

        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

        // 1D tensor: shape.last() = 4096
        assert_eq!(v["hidden_size"], 4096);
    }

    // ========================================================================
    // infer_model_config: num_attention_heads fallback for various hidden sizes
    // ========================================================================

    #[test]
    fn test_infer_model_config_heads_fallback_hidden_896() {
        // Qwen2.5-0.5B: hidden=896 -> 14 heads
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![32000, 896]),
        );
        // No q_proj -> fallback path for num_attention_heads
        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
        assert_eq!(v["num_attention_heads"], 14);
    }

    #[test]
    fn test_infer_model_config_heads_fallback_hidden_1536() {
        // Qwen2.5-1.5B: hidden=1536 -> 12 heads
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![32000, 1536]),
        );
        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
        assert_eq!(v["num_attention_heads"], 12);
    }

    #[test]
    fn test_infer_model_config_heads_fallback_hidden_2048() {
        // Llama-style: hidden=2048 -> 16 heads
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![32000, 2048]),
        );
        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
        assert_eq!(v["num_attention_heads"], 16);
    }

    #[test]
    fn test_infer_model_config_heads_fallback_hidden_4096() {
        // Llama-7B: hidden=4096 -> 32 heads
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![32000, 4096]),
        );
        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
        assert_eq!(v["num_attention_heads"], 32);
    }

    #[test]
    fn test_infer_model_config_heads_fallback_hidden_5120() {
        // Llama-13B: hidden=5120 -> 40 heads
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![32000, 5120]),
        );
        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
        assert_eq!(v["num_attention_heads"], 40);
    }

    #[test]
    fn test_infer_model_config_heads_fallback_hidden_8192() {
        // Llama-70B: hidden=8192 -> 64 heads
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![32000, 8192]),
        );
        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
        assert_eq!(v["num_attention_heads"], 64);
    }

    #[test]
    fn test_infer_model_config_heads_fallback_unknown_hidden_size() {
        // Non-standard hidden size: 3072 -> default formula: (3072 / 128).max(1) = 24
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![32000, 3072]),
        );
        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
        // 3072 / 128 = 24
        assert_eq!(v["num_attention_heads"], 24);
    }

    // ========================================================================
    // infer_model_config: num_attention_heads with q_proj present (non-fallback)
    // ========================================================================

    #[test]
    fn test_infer_model_config_heads_from_q_proj_small_hidden() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // hidden < 4096 -> head_dim = 64
        // hidden = 2048, head_dim = 64, num_heads = 2048/64 = 32
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![32000, 2048]),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![], vec![2048, 2048]),
        );
        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
        // hidden=2048 < 4096 -> head_dim=64, num_heads = 2048/64 = 32
        assert_eq!(v["num_attention_heads"], 32);
    }

    #[test]
    fn test_infer_model_config_heads_from_q_proj_large_hidden() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // hidden >= 4096 -> head_dim = 128
        // hidden = 4096, head_dim = 128, num_heads = 4096/128 = 32
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![32000, 4096]),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![], vec![4096, 4096]),
        );
        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
        assert_eq!(v["num_attention_heads"], 32);
    }

    // ========================================================================
    // infer_model_config: intermediate_size default when no MLP tensors
    // ========================================================================

    #[test]
    fn test_infer_model_config_intermediate_size_default() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![32000, 2048]),
        );
        // No gate_proj/up_proj/feed_forward.w1 -> default to hidden_size * 4
        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
        // hidden=2048, default intermediate = 2048*4 = 8192
        assert_eq!(v["intermediate_size"], 8192);
    }

    // ========================================================================
    // infer_model_config: intermediate_size from up_proj
    // ========================================================================

    #[test]
    fn test_infer_model_config_intermediate_size_from_up_proj() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![32000, 4096]),
        );
        // up_proj with specific intermediate size
        tensors.insert(
            "model.layers.0.mlp.up_proj.weight".to_string(),
            (vec![], vec![14336, 4096]),
        );
        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
        assert_eq!(v["intermediate_size"], 14336);
    }

    // ========================================================================
    // infer_model_config: intermediate_size from feed_forward.w1
    // ========================================================================

    #[test]
    fn test_infer_model_config_intermediate_size_from_feed_forward_w1() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![32000, 4096]),
        );
        tensors.insert(
            "model.layers.0.feed_forward.w1.weight".to_string(),
            (vec![], vec![11008, 4096]),
        );
        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
        assert_eq!(v["intermediate_size"], 11008);
    }

    // ========================================================================
    // infer_model_config: num_key_value_heads defaults to num_attention_heads (MHA)
    // ========================================================================

    #[test]
    fn test_infer_model_config_kv_heads_default_mha() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![32000, 4096]),
        );
        // q_proj present but NO k_proj -> num_kv_heads defaults to num_attention_heads
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![], vec![4096, 4096]),
        );
        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
        assert_eq!(v["num_key_value_heads"], v["num_attention_heads"]);
    }

    // ========================================================================
    // infer_model_config: Output is always valid JSON
    // ========================================================================

    #[test]
    fn test_infer_model_config_output_is_valid_json() {
        // Test with various tensor configurations to ensure JSON is always valid
        let configs: Vec<BTreeMap<String, (Vec<f32>, Vec<usize>)>> = vec![
            BTreeMap::new(), // empty
            {
                let mut m = BTreeMap::new();
                m.insert(
                    "model.embed_tokens.weight".to_string(),
                    (vec![], vec![100, 64]),
                );
                m
            },
            {
                let mut m = BTreeMap::new();
                m.insert("token_embd.weight".to_string(), (vec![], vec![50000, 768]));
                m.insert("blk.0.attn_q.weight".to_string(), (vec![], vec![768, 768]));
                m.insert("blk.5.attn_q.weight".to_string(), (vec![], vec![768, 768]));
                m
            },
        ];

        for tensors in &configs {
            let config = infer_model_config(tensors);
            let result: std::result::Result<serde_json::Value, _> = serde_json::from_str(&config);
            assert!(
                result.is_ok(),
                "infer_model_config produced invalid JSON for {} tensors: {config}",
                tensors.len()
            );
        }
    }

    // ========================================================================
    // infer_model_config: Config contains all required HuggingFace fields
    // ========================================================================

    #[test]
    fn test_infer_model_config_contains_all_required_fields() {
        let tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

        // All fields required by HuggingFace SafeTensors inference (GH-193)
        let required_fields = [
            "architectures",
            "bos_token_id",
            "eos_token_id",
            "hidden_act",
            "hidden_size",
            "initializer_range",
            "intermediate_size",
            "max_position_embeddings",
            "model_type",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "rms_norm_eps",
            "rope_theta",
            "sliding_window",
            "tie_word_embeddings",
            "torch_dtype",
            "use_cache",
            "use_sliding_window",
            "vocab_size",
        ];

        for field in &required_fields {
            assert!(
                !v[field].is_null(),
                "Required field '{field}' is missing from config JSON"
            );
        }
    }

    // ========================================================================
    // infer_model_config: head_dim calculation
    // ========================================================================

    #[test]
    fn test_infer_model_config_head_dim_affects_kv_heads() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // hidden=8192, 64 heads -> head_dim = 128
        // k_proj shape [2048, 8192] -> kv_heads = 2048/128 = 16
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![32000, 8192]),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![], vec![8192, 8192]),
        );
        tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            (vec![], vec![2048, 8192]),
        );

        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

        // hidden=8192 >= 4096 -> head_dim=128
        // num_heads = 8192/128 = 64
        assert_eq!(v["num_attention_heads"], 64);
        // kv_heads = 2048/128 = 16
        assert_eq!(v["num_key_value_heads"], 16);
    }

    // ========================================================================
    // infer_model_config: Embedding shape with GGUF transposed layout
    // ========================================================================

    #[test]
    fn test_infer_model_config_transposed_embedding_gguf() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // GGUF sometimes stores embeddings as [hidden_size, vocab_size]
        // The function picks the smaller dim as hidden_size
        tensors.insert(
            "token_embd.weight".to_string(),
            (vec![], vec![4096, 151936]),
        );

        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

        // min(4096, 151936) = 4096 -> hidden
        assert_eq!(v["hidden_size"], 4096);
        // max(4096, 151936) = 151936 -> vocab
        assert_eq!(v["vocab_size"], 151936);
    }

    // ========================================================================
    // infer_model_config: Alternative q_proj name patterns
    // ========================================================================

    #[test]
    fn test_infer_model_config_attn_q_proj_pattern() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![32000, 2048]),
        );
        // "attn.q_proj" pattern (without "self_attn" prefix)
        tensors.insert(
            "model.layers.0.attn.q_proj.weight".to_string(),
            (vec![], vec![2048, 2048]),
        );
        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
        // Should detect q_proj via "attn.q_proj" pattern
        // hidden=2048 < 4096 -> head_dim=64, num_heads=32
        assert_eq!(v["num_attention_heads"], 32);
    }

    #[test]
    fn test_infer_model_config_attention_wq_pattern() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![32000, 4096]),
        );
        // "attention.wq" pattern (Llama-style)
        tensors.insert(
            "model.layers.0.attention.wq.weight".to_string(),
            (vec![], vec![4096, 4096]),
        );
        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
        assert_eq!(v["num_attention_heads"], 32);
    }

    // ========================================================================
    // infer_model_config: Alternative k_proj name patterns for GQA
    // ========================================================================

    #[test]
    fn test_infer_model_config_gqa_via_attn_k_proj() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![32000, 4096]),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![], vec![4096, 4096]),
        );
        // "attn.k_proj" pattern
        tensors.insert(
            "model.layers.0.attn.k_proj.weight".to_string(),
            (vec![], vec![512, 4096]),
        );
        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
        // kv_dim=512, head_dim=128, kv_heads = 512/128 = 4
        assert_eq!(v["num_key_value_heads"], 4);
    }

    #[test]
    fn test_infer_model_config_gqa_via_attention_wk() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![], vec![32000, 4096]),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![], vec![4096, 4096]),
        );
        // "attention.wk" pattern (Llama-style)
        tensors.insert(
            "model.layers.0.attention.wk.weight".to_string(),
            (vec![], vec![1024, 4096]),
        );
        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
        // kv_dim=1024, head_dim=128, kv_heads = 1024/128 = 8
        assert_eq!(v["num_key_value_heads"], 8);
    }

    // ========================================================================
    // infer_model_config: output.weight used for vocab instead of lm_head
    // ========================================================================

    #[test]
    fn test_infer_model_config_output_weight_for_vocab() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // GGUF-style output.weight instead of lm_head
        tensors.insert("output.weight".to_string(), (vec![], vec![128256, 4096]));
        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
        assert_eq!(v["vocab_size"], 128256);
    }

    // ========================================================================
    // infer_model_config: 1D vocab tensor fallback
    // ========================================================================

    #[test]
    fn test_infer_model_config_1d_lm_head_vocab() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // 1D lm_head - uses first dim
        tensors.insert("lm_head.weight".to_string(), (vec![], vec![50000]));
        let config = infer_model_config(&tensors);
        let v: serde_json::Value = serde_json::from_str(&config)
            .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
        assert_eq!(v["vocab_size"], 50000);
    }

    // ========================================================================
    // infer_model_config: 1D embedding + 1D lm_head (degenerate case)
    // ========================================================================

    #[test]
    fn test_infer_model_config_empty_1d_embedding() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // 1D embedding with no elements - shape.last() returns Some(0)
        tensors.insert("token_embd.weight".to_string(), (vec![], vec![0]));
        let config = infer_model_config(&tensors);
        // Should still produce valid JSON
        let result: std::result::Result<serde_json::Value, _> = serde_json::from_str(&config);
        assert!(
            result.is_ok(),
            "Should produce valid JSON even with 0-dim tensor"
        );
    }

    // ========================================================================
    // extract_user_metadata: Empty/short data
    // ========================================================================

    #[test]
    fn test_extract_user_metadata_nonexistent_file() {
        let result = extract_user_metadata(Path::new("/tmp/nonexistent_apr_file_12345.apr"));
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_user_metadata_short_data() {
        let dir = std::env::temp_dir().join("apr_test_short_data");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("short.apr");
        // Write less than 16 bytes
        fs::write(&path, &[0u8; 10]).expect("write failed");

        let result = extract_user_metadata(&path);
        assert!(result.is_empty());

        let _ = fs::remove_file(&path);
    }

    // ========================================================================
    // extract_user_metadata: Invalid metadata JSON
    // ========================================================================

    #[test]
    fn test_extract_user_metadata_invalid_json() {
        let dir = std::env::temp_dir().join("apr_test_invalid_json");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("invalid.apr");

        // Build APR-like bytes: magic(4) + version(4) + metadata_len(8) + garbage
        let mut data = Vec::new();
        data.extend_from_slice(b"APR\x00"); // magic
        data.extend_from_slice(&2u32.to_le_bytes()); // version
        let garbage = b"not valid json{{{";
        let len = garbage.len() as u64;
        data.extend_from_slice(&len.to_le_bytes()); // metadata_len
        data.extend_from_slice(garbage);

        fs::write(&path, &data).expect("write failed");

        let result = extract_user_metadata(&path);
        assert!(result.is_empty());

        let _ = fs::remove_file(&path);
    }

    // ========================================================================
    // extract_user_metadata: Valid with source_metadata
    // ========================================================================

    #[test]
    fn test_extract_user_metadata_with_source_metadata() {
        let dir = std::env::temp_dir().join("apr_test_source_metadata");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("with_meta.apr");

        let json = r#"{"custom":{"source_metadata":{"key1":"val1","key2":"val2"}}}"#;
        let mut data = Vec::new();
        data.extend_from_slice(b"APR\x00"); // magic
        data.extend_from_slice(&2u32.to_le_bytes()); // version
        let json_bytes = json.as_bytes();
        let len = json_bytes.len() as u64;
        data.extend_from_slice(&len.to_le_bytes()); // metadata_len
        data.extend_from_slice(json_bytes);

        fs::write(&path, &data).expect("write failed");

        let result = extract_user_metadata(&path);
        assert_eq!(result.len(), 2);
        assert_eq!(result.get("key1").map(String::as_str), Some("val1"));
        assert_eq!(result.get("key2").map(String::as_str), Some("val2"));

        let _ = fs::remove_file(&path);
    }

    // ========================================================================
    // extract_user_metadata: Valid without source_metadata
    // ========================================================================

    #[test]
    fn test_extract_user_metadata_without_source_metadata() {
        let dir = std::env::temp_dir().join("apr_test_no_source_metadata");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("no_source_meta.apr");

        let json = r#"{"custom":{"other_key":"other_val"}}"#;
        let mut data = Vec::new();
        data.extend_from_slice(b"APR\x00");
        data.extend_from_slice(&2u32.to_le_bytes());
        let json_bytes = json.as_bytes();
        let len = json_bytes.len() as u64;
        data.extend_from_slice(&len.to_le_bytes());
        data.extend_from_slice(json_bytes);

        fs::write(&path, &data).expect("write failed");

        let result = extract_user_metadata(&path);
        assert!(result.is_empty());

        let _ = fs::remove_file(&path);
    }

    // ========================================================================
    // extract_user_metadata: Non-string values in source_metadata skipped
    // ========================================================================

    #[test]
    fn test_extract_user_metadata_non_string_values_skipped() {
        let dir = std::env::temp_dir().join("apr_test_non_string_values");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("mixed_types.apr");

        let json = r#"{"custom":{"source_metadata":{"str_key":"str_val","num_key":42,"bool_key":true,"null_key":null}}}"#;
        let mut data = Vec::new();
        data.extend_from_slice(b"APR\x00");
        data.extend_from_slice(&2u32.to_le_bytes());
        let json_bytes = json.as_bytes();
        let len = json_bytes.len() as u64;
        data.extend_from_slice(&len.to_le_bytes());
        data.extend_from_slice(json_bytes);

        fs::write(&path, &data).expect("write failed");

        let result = extract_user_metadata(&path);
        // Only the string value should be extracted
        assert_eq!(result.len(), 1);
        assert_eq!(result.get("str_key").map(String::as_str), Some("str_val"));

        let _ = fs::remove_file(&path);
    }

    // ========================================================================
    // extract_user_metadata: Metadata length exceeds data length
    // ========================================================================

    #[test]
    fn test_extract_user_metadata_metadata_len_exceeds_data() {
        let dir = std::env::temp_dir().join("apr_test_len_overflow");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("overflow.apr");

        let mut data = Vec::new();
        data.extend_from_slice(b"APR\x00");
        data.extend_from_slice(&2u32.to_le_bytes());
        // Claim metadata is 9999 bytes but only provide 5
        let len: u64 = 9999;
        data.extend_from_slice(&len.to_le_bytes());
        data.extend_from_slice(b"hello");

        fs::write(&path, &data).expect("write failed");

        let result = extract_user_metadata(&path);
        assert!(result.is_empty());

        let _ = fs::remove_file(&path);
    }

    // ========================================================================
    // extract_user_metadata: Invalid UTF-8 in metadata
    // ========================================================================

    #[test]
    fn test_extract_user_metadata_invalid_utf8() {
        let dir = std::env::temp_dir().join("apr_test_invalid_utf8");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("bad_utf8.apr");

        let mut data = Vec::new();
        data.extend_from_slice(b"APR\x00");
        data.extend_from_slice(&2u32.to_le_bytes());
        // Invalid UTF-8 sequence
        let bad_bytes: &[u8] = &[0xFF, 0xFE, 0x80, 0x81, 0x82];
        let len = bad_bytes.len() as u64;
        data.extend_from_slice(&len.to_le_bytes());
        data.extend_from_slice(bad_bytes);

        fs::write(&path, &data).expect("write failed");

        let result = extract_user_metadata(&path);
        assert!(result.is_empty());

        let _ = fs::remove_file(&path);
    }

    // ========================================================================
    // infer_tokenizer_json: Non-APR extension returns empty
    // ========================================================================

    #[test]
    fn test_infer_tokenizer_json_non_apr_extension() {
        let result = infer_tokenizer_json(Path::new("/tmp/model.safetensors"));
        assert!(result.is_empty());
    }

    #[test]
    fn test_infer_tokenizer_json_gguf_extension() {
        let result = infer_tokenizer_json(Path::new("/tmp/model.gguf"));
        assert!(result.is_empty());
    }

    #[test]
    fn test_infer_tokenizer_json_no_extension() {
        let result = infer_tokenizer_json(Path::new("/tmp/model"));
        assert!(result.is_empty());
    }

    // ========================================================================
    // infer_tokenizer_json: APR file that doesn't exist
    // ========================================================================

    #[test]
    fn test_infer_tokenizer_json_nonexistent_apr() {
        let result = infer_tokenizer_json(Path::new("/tmp/nonexistent_12345.apr"));
        assert!(result.is_empty());
    }

    // ========================================================================
    // infer_tokenizer_json: APR file too short
    // ========================================================================

    #[test]
    fn test_infer_tokenizer_json_short_apr_file() {
        let dir = std::env::temp_dir().join("apr_test_short_apr_tokenizer");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("short.apr");
        // Write less than 44 bytes (header size)
        fs::write(&path, &[0u8; 20]).expect("write failed");

        let result = infer_tokenizer_json(&path);
        assert!(result.is_empty());

        let _ = fs::remove_file(&path);
    }

    // ========================================================================
    // infer_tokenizer_json: APR file with no tokenizer in metadata
    // ========================================================================

    #[test]
    fn test_infer_tokenizer_json_apr_without_tokenizer() {
        let dir = std::env::temp_dir().join("apr_test_no_tokenizer");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("no_tok.apr");

        // Build fake APR: 44 bytes header + metadata JSON without tokenizer + terminator
        let mut data = vec![0u8; 44]; // header
        let metadata = r#"{"model_type": "qwen2", "hidden_size": 896}"#;
        data.extend_from_slice(metadata.as_bytes());
        data.extend_from_slice(b"}\n\n\n"); // terminator pattern

        fs::write(&path, &data).expect("write failed");

        let result = infer_tokenizer_json(&path);
        // Metadata doesn't contain "tokenizer" or "vocabulary"
        assert!(result.is_empty());

        let _ = fs::remove_file(&path);
    }
}
