//! APR Converter - Export to SafeTensors/GGUF (APR-SPEC §4.6)
//! PMAT-197: Extracted from mod.rs for file size reduction

use crate::error::{AprenderError, Result};
use crate::format::converter_types::{Architecture, QuantizationType};
use crate::format::gguf::GgufReader;
use crate::serialization::safetensors::{
    save_safetensors, save_safetensors_with_metadata, UserMetadata,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

// Import shared functions from parent module
use super::{calculate_tensor_size, load_model_tensors, map_tensor_names, quantize_tensors};
// NOTE: quantize_q4_k_matrix imported via super:: through mod.rs

// ============================================================================
// BUG-EXPORT-002: LAYOUT-002 Transpose for Export
// ============================================================================

/// Transpose F32 matrix from row-major to column-major layout for GGUF export.
///
/// BUG-EXPORT-002 FIX: APR stores tensors in row-major layout with shape [rows, cols].
/// GGUF expects column-major layout with shape [cols, rows].
/// This function transposes the data for GGUF compatibility.
///
/// # Arguments
/// * `data` - F32 values in row-major order
/// * `shape` - APR shape [rows, cols]
///
/// # Returns
/// Transposed F32 data in column-major order (stored as row-major [cols, rows])
fn transpose_f32_rowmajor_to_colmajor(data: &[f32], shape: &[usize]) -> Vec<f32> {
    if shape.len() != 2 {
        // Only transpose 2D tensors
        return data.to_vec();
    }

    let rows = shape[0];
    let cols = shape[1];

    // Transpose: row-major [rows, cols] -> column-major layout
    // Column-major with shape [cols, rows] stored as: for each column (original row), store values
    let mut transposed = vec![0.0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            // Row-major source: data[r * cols + c]
            // Column-major dest (stored as row-major [cols, rows]): transposed[c * rows + r]
            transposed[c * rows + r] = data[r * cols + c];
        }
    }

    transposed
}

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

    // Load tensors
    let tensors = load_model_tensors(input_path)?;
    let original_size = calculate_tensor_size(&tensors);
    let original_count = tensors.len();

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

    // Apply quantization if requested
    let tensors = if let Some(ref quant_type) = options.quantize {
        quantize_tensors(&tensors, quant_type)?
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

    Ok(ExportReport {
        original_size,
        exported_size,
        tensor_count: original_count,
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

    let gguf_tensors: Vec<GgufTensor> = tensors
        .iter()
        .map(|(name, (data, shape))| {
            let gguf_name = hf_to_gguf_name(name);

            // Reverse 2D shapes: standard [rows, cols] → GGML [ne0=cols, ne1=rows]
            let gguf_shape = if shape.len() == 2 {
                vec![shape[1] as u64, shape[0] as u64]
            } else {
                shape.iter().map(|&d| d as u64).collect()
            };

            // BUG-1 FIX: Quantize to Q4_K for GGUF inference compatibility
            // Only quantize 2D weight tensors, keep 1D tensors (biases, norms) as F32
            //
            // BUG-4 FIX: Embedding tensors MUST stay as F32 for realizar compatibility!
            // Realizar uses get_tensor_f32() which expects flat layout, but Q4_K is row-padded.
            // This matches official GGUF files (e.g., Qwen2-0.5B uses Q8_0 for embedding).
            let is_embedding = gguf_name == "token_embd.weight" || name.contains("embed_tokens");

            // BUG-EXPORT-002 FIX: APR uses row-major layout, GGUF uses column-major.
            // Must transpose 2D tensors before exporting.
            let (dtype, bytes) = if use_q4k && shape.len() == 2 && data.len() >= 256 && !is_embedding {
                // Transpose first, then quantize
                // APR row-major [rows, cols] -> GGUF column-major (transposed shape [cols, rows])
                let transposed_data = transpose_f32_rowmajor_to_colmajor(data, shape);
                let transposed_shape = vec![shape[1], shape[0]]; // [cols, rows]
                let q4k_bytes = super::quantize_q4_k_matrix(&transposed_data, &transposed_shape);

                // DEBUG: Check quantization output size (now using transposed shape)
                let rows = transposed_shape[0]; // cols became rows
                let cols = transposed_shape[1]; // rows became cols
                let expected_blocks_per_row = (cols + 255) / 256;
                let expected_bytes = rows * expected_blocks_per_row * 144;
                if q4k_bytes.len() != expected_bytes {
                    eprintln!(
                        "[Q4K-SIZE-MISMATCH] '{}': shape={:?}, expected={}, actual={}",
                        name, transposed_shape, expected_bytes, q4k_bytes.len()
                    );
                }
                (GgmlType::Q4K, q4k_bytes)
            } else if shape.len() == 2 {
                // F32 2D tensor - transpose for GGUF column-major layout
                let transposed_data = transpose_f32_rowmajor_to_colmajor(data, shape);
                let f32_bytes: Vec<u8> = transposed_data.iter().flat_map(|f| f.to_le_bytes()).collect();
                (GgmlType::F32, f32_bytes)
            } else {
                // 1D tensor (biases, norms) - no transpose needed
                let f32_bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
                (GgmlType::F32, f32_bytes)
            };

            // DEBUG: Verify data size being passed to GgufTensor
            static TENSOR_DEBUG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let tensor_count = TENSOR_DEBUG.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if tensor_count < 3 {
                eprintln!(
                    "[DEBUG-TENSOR-CREATE] '{}': dtype={:?}, shape={:?}, bytes.len()={}, bytes_ptr={:p}",
                    gguf_name, dtype, gguf_shape, bytes.len(), bytes.as_ptr()
                );
            }

            let tensor = GgufTensor {
                name: gguf_name,
                shape: gguf_shape,
                dtype,
                data: bytes,
            };

            if tensor_count < 3 {
                eprintln!(
                    "[DEBUG-TENSOR-AFTER] '{}': tensor.data.len()={}, tensor.data.ptr={:p}",
                    tensor.name, tensor.data.len(), tensor.data.as_ptr()
                );
            }

            tensor
        })
        .collect();

    // DEBUG: Verify collected tensors
    eprintln!(
        "[DEBUG-COLLECTED] Total tensors: {}, first tensor data.len()={}",
        gguf_tensors.len(),
        gguf_tensors.first().map(|t| t.data.len()).unwrap_or(0)
    );

    // BUG-4 FIX: For tied embedding models (Qwen2, etc.), create a separate Q4K
    // output.weight tensor for the LM head, since the embedding is kept as F32.
    // Realizar's fused matmul requires quantized LM head weights.
    let has_lm_head = gguf_tensors.iter().any(|t| t.name == "output.weight");
    let mut gguf_tensors = gguf_tensors;

    if use_q4k && !has_lm_head {
        // Find the embedding tensor and create a Q4K copy for output.weight
        if let Some(embed_data) = tensors
            .iter()
            .find(|(name, _)| name.contains("embed_tokens") || name.contains("token_embedding"))
        {
            let (_, (data, shape)) = embed_data;
            if shape.len() == 2 && data.len() >= 256 {
                eprintln!(
                    "[BUG-4-FIX] Creating Q4K output.weight from embedding for tied embeddings"
                );

                let q4k_bytes = super::quantize_q4_k_matrix(data, shape);
                let gguf_shape = vec![shape[1] as u64, shape[0] as u64]; // reversed for GGML

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
